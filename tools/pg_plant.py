"""The planted round's offline readout (docs/training_design_20261019/
current_design.md §5 "Planted round", its C-offline half).

A plant writes a known signal into the labels and reads it back through
every piece a round runs. In side A's frame each game's result becomes

    z' = z + delta * Y,   Y = sum_t sigma_t (1[a_t in X] - p_t(X)),

over every logged decision t of the game, refused ones included, sigma_t
+1 at side A's decisions and -1 at side B's, p_t(X) the logged law's mass on
X (X = attack; `--class` names another class of pg_loss.CLASSES). The luck
walk's table (tools/match_luck.py rows: one a game, `z` side A's result) is
copied with z' in place of z, and the production join
(tools/selfplay_rows.py), b and beta (tools/pg_labels.py) and the gradient
pass (tools/pg_pass.py) read the copy. Each term of Y has mean 0 given the
history before its decision, so the plant's expected gradient is exactly

    delta * u,   u = (1/2N) sum_games sum_t grad p_t(X | h_t)

over the planted pass's N games: a side's own terms pair with its own
scores, every other pairing has mean 0.

The readout, per seat (A, B), on the held-out games H (a fraction of them,
`--readout-fraction`, drawn by game):
  measured  the teacher-forced change of the seat's mean over its game-sides
            of sum_t p_t(X) at theta_k + s d', d' the planted pass's
            reduced-AdamW direction and s its step at the mean per-decision
            KL eps = K_SIDE / decisions per side-game: its odd part
            (M(s) - M(-s)) / 2, free of the second-order term;
  forecast  s delta D_{P'u} M, P' the planted pass's metric (P'u is
            reduced AdamW applied to u), u from code separate from the pass
            (tools/pg_plant_grad.py: no label, no R, the pass's own windows),
            the derivative a central difference along P'u with the memory
            carried along each game-side;
  SE        s times the SE over the planted pass's eight splits of
            <P' grad M, g' - delta u>, g' its gradient, grad M taken through
            windows of the pass's length;
and passes when on both seats the forecast is resolved (MIN_RESOLUTION SE)
and the measured change lies within AGREE_SE SE of it. A wrong direction or
step reads the wrong size. One network plays both seats, so a seat whose
label is reversed pulls the shared weights against the other seat's plant
(the seats alternate sides, and their plants' expected gradients are equal):
both seats then read no change, as for a broken join or hand-off.

delta (the `delta` command): TARGET_SE times, over the seats, the largest
SE0 / |<P grad M, u>|, SE0 the SE of <P grad M, g> over the main pass's
eight splits scaled to the planted pass's games (SE goes as 1/sqrt(games)),
P and g the main pass's. The plant's own noise comes on top of the main
pass's, so the resolution reached is read, not assumed.

    python tools/pg_plant.py delta --checkpoint own2.pt --games DIR [DIR ...] --labels labels.jsonl \\
        --pass pass.pt --walk walk.jsonl --seed-base 800000 --train-games 5000 --out plant_grads.pt
    python tools/pg_plant.py plant --walk walk.jsonl --games DIR [DIR ...] --grads plant_grads.pt \\
        --out walk_planted.jsonl
    # selfplay_rows.py --walk walk_planted.jsonl, pg_labels.py, pg_pass.py on its labels
    python tools/pg_plant.py readout --checkpoint own2.pt --games DIR [DIR ...] --labels labels_planted.jsonl \\
        --pass pass_planted.pt --grads plant_grads.pt --out plant_readout.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from tools.pg_positions import RecordIndex, read_record, result_path  # noqa: E402
from wesnoth_ai.pg_loss import CLASSES  # noqa: E402
from wesnoth_ai.pg_records import decisions_of, read_labels  # noqa: E402
from wesnoth_ai.pg_step import dot, norm, reduced_adamw, second_moment  # noqa: E402
from wesnoth_ai.sequence_streams import unit_draw  # noqa: E402
from wesnoth_ai.server_priors import KIND_ATTACK, KIND_END_TURN, KIND_RECRUIT  # noqa: E402

log = logging.getLogger("pg_plant")

DEFAULT_CLASS = "attack"
KIND_OF = {"end_turn": KIND_END_TURN, "attack": KIND_ATTACK, "recruit": KIND_RECRUIT}
LOGGED = {"end_turn": "p_end", "attack": "p_att", "recruit": "p_rec"}
TARGET_SE = 10.0            # delta puts the plant this many SE of the main pass's noise above 0 (§5)
AGREE_SE = 3.0              # the measured change lies within this many SE of its forecast (§5)
# The forecast must stand this far above its noise for the check to mean
# anything: AGREE_SE plus a 2-SE margin, so a seat that does not move (a
# broken join) misses by more than AGREE_SE.
MIN_RESOLUTION = 5.0
K_SIDE = 0.25               # the step's KL per side-game: stage C's arms' (tools/pg_stage_c.py)
FD_FRACTION = 0.1           # the central difference's step: a tenth of the planted step's length
SEATS = ("A", "B")


# ---------------------------------------------------------------- the plant

def planted_sum(decisions, side_a: int, cls: str) -> float:
    """Y = sum_t sigma_t (1[a_t in X] - p_t(X)) over a game's logged
    decisions, in side A's frame."""
    kind, key = KIND_OF[cls], LOGGED[cls]
    return float(sum((1.0 if d.side == side_a else -1.0) * (float(d.kind == kind) - getattr(d, key))
                     for d in decisions))


def plant_rows(walk_rows: Sequence[Dict], index: RecordIndex, delta: float, cls: str,
               keep: Optional[set] = None) -> List[Dict]:
    """The walk's rows with z' = z + delta Y (side A's frame); `keep` names
    the games the copy holds (all when None). A row whose walk failed is
    copied as it is: the labels leave its game out."""
    out = []
    for row in walk_rows:
        if keep is not None and row["game"] not in keep:
            continue
        row = dict(row)
        if "error" not in row:
            y = planted_sum(decisions_of(read_record(index.path(row["game"]))), int(row["side_a"]), cls)
            row["planted"] = {"z": float(row["z"]), "Y": y, "delta": float(delta), "class": cls}
            row["z"] = float(row["z"]) + float(delta) * y
        out.append(row)
    return out


# ---------------------------------------------------------------- statistics

def split_total_se(parts: Sequence[float], split_games: Sequence[int]) -> float:
    """The SE of sum_i parts[i], each part (1/2N) times the sum of one
    split's per-game contributions (N the games of all splits): the
    contributions' variance from the splits' means, n_i (y_i - y)^2 summed
    over the k non-empty splits over k - 1, y_i = 2N parts[i] / n_i."""
    n = np.asarray(split_games, dtype=np.float64)
    x = np.asarray(parts, dtype=np.float64)
    live = n > 0
    N = float(n.sum())
    if live.sum() < 2:
        return math.nan
    y = 2.0 * N * x[live] / n[live]
    mean = float((n[live] * y).sum() / N)
    variance = float((n[live] * (y - mean) ** 2).sum() / (live.sum() - 1))
    return math.sqrt(N * variance) / (2.0 * N)


def _project(w: Mapping[str, torch.Tensor], parts: Sequence[Mapping[str, torch.Tensor]]) -> List[float]:
    return [dot(w, {n: p[n] for n in w}) for p in parts]


def _metric(pass_: Mapping) -> Dict[str, torch.Tensor]:
    return {n: t.float() for n, t in second_moment(pass_).items()}


def _scaled(parts: Sequence[Mapping[str, torch.Tensor]], c: float) -> List[Dict[str, torch.Tensor]]:
    return [{n: c * t for n, t in p.items()} for p in parts]


# ---------------------------------------------------------------- the games

def seat_of(index: RecordIndex, games: Sequence[str]) -> Dict[str, int]:
    """Each game's side A (its match result)."""
    return {g: int(json.loads(result_path(index.path(g)).read_text(encoding="utf-8"))["side_a"]) for g in games}


def readout_games(heldout: Sequence[str], fraction: float, seed: int) -> List[str]:
    """The held-out games the readout reads, drawn by game."""
    chosen = sorted(g for g in heldout if unit_draw("pg_plant", seed, g) < fraction)
    return chosen or sorted(heldout)[:2]


def seat_sides(games: Sequence[str], side_a: Mapping[str, int]) -> Dict[str, List[tuple]]:
    return {"A": [(g, side_a[g]) for g in games], "B": [(g, 3 - side_a[g]) for g in games]}


# ---------------------------------------------------------------- delta

def delta_rule(main_pass: Mapping, grad_M: Mapping[str, Mapping[str, torch.Tensor]],
               u_splits: Sequence[Mapping[str, torch.Tensor]], planted_games: int) -> Dict:
    """delta for TARGET_SE (module docstring), with its inputs per seat."""
    v = _metric(main_pass)
    scale = math.sqrt(main_pass["games"] / planted_games)
    per_seat, ratios = {}, []
    for seat in SEATS:
        w = reduced_adamw({n: grad_M[seat][n].float() for n in main_pass["names"]}, v)
        noise = split_total_se(_project(w, main_pass["splits"]), main_pass["split_games"]) * scale
        signal = sum(_project(w, u_splits))
        if signal == 0 or not math.isfinite(noise):
            raise ValueError(f"seat {seat}: the plant's projection is {signal}, the noise {noise}: no delta")
        per_seat[seat] = {"noise_se": noise, "signal_per_delta": signal}
        ratios.append(noise / abs(signal))
    return {"delta": TARGET_SE * max(ratios), "target_se": TARGET_SE, "seats": per_seat,
            "main_games": int(main_pass["games"]), "planted_games": int(planted_games)}


# ---------------------------------------------------------------- the readout

def _seat_means(evaluator, s: float, cls: str, side_a: Mapping[str, int]) -> Dict[str, float]:
    """Per seat, the mean over its game-sides of sum_t p_t(X) at step s."""
    by_seat = evaluator.class_mass_by_seat(s, lambda game, side: "A" if side == side_a[game] else "B")
    return {seat: float(np.mean(by_seat[seat][cls])) for seat in SEATS}


def _check_inputs(rows, pass_: Mapping, grads: Mapping) -> None:
    train = sorted({r.game for r in rows if not r.heldout})
    if train != list(grads["train_games"]):
        raise ValueError(f"the planted labels train on {len(train)} games, u was read on {len(grads['train_games'])}"
                         f" others: the plant and its pass do not belong together")
    if pass_["games"] != len(train) or pass_["checkpoint_sha256"] != grads["checkpoint_sha256"]:
        raise ValueError("the planted pass is not the pass of these labels and this checkpoint")
    for key, value in grads["config"].items():
        if pass_["config"].get(key) != value:
            raise ValueError(f"the planted pass ran at {key}={pass_['config'].get(key)}, u at {value}: "
                             f"its windows are not the pass's")


def plant_readout(checkpoint: Path, records: Mapping[str, Path], rows, pass_: Mapping, grads: Mapping,
                  device: torch.device, eval_config, k_side: float = K_SIDE) -> Dict:
    """The readout of module docstring, per seat, and its verdict."""
    from tools.pg_eval import EvalContext
    from wesnoth_ai.pg_step import decisions_per_side, kl_search
    _check_inputs(rows, pass_, grads)
    cls, delta, side_a = grads["class"], float(grads["delta"]), dict(grads["side_a"])
    v = _metric(pass_)
    d = reduced_adamw({n: pass_["total"][n].float() for n in pass_["names"]}, v)
    u_total = {n: sum(part[n] for part in grads["u"]) for n in pass_["names"]}
    pu = reduced_adamw(u_total, v)
    readout = set(grads["readout_games"])
    context = EvalContext(checkpoint, records, [r for r in rows if r.heldout and r.game in readout], device,
                          eval_config)
    try:
        ev = context.bind(d)
        eps = k_side / decisions_per_side(ev.game_sides())
        point = kl_search(ev.mean_kl, eps, s0=1.0 / norm(d))
        s = point.s
        ev.evaluate([s, -s, 0.0])
        at = {x: _seat_means(ev, x, cls, side_a) for x in (s, -s, 0.0)}
        h = FD_FRACTION * s * norm(d) / norm(pu)
        ev_u = context.bind(pu)
        ev_u.evaluate([h, -h])
        along = {x: _seat_means(ev_u, x, cls, side_a) for x in (h, -h)}
    finally:
        context.close()
    seats, ok = {}, True
    for seat in SEATS:
        w = reduced_adamw({n: grads["grad_M"][seat][n].float() for n in pass_["names"]}, v)
        u_parts = _project(w, grads["u"])
        deviation = [g - delta * x for g, x in zip(_project(w, pass_["splits"]), u_parts)]
        se = s * split_total_se(deviation, pass_["split_games"])
        forecast = s * delta * (along[h][seat] - along[-h][seat]) / (2.0 * h)
        measured = (at[s][seat] - at[-s][seat]) / 2.0
        resolution = forecast / se if se > 0 else math.nan
        miss = (measured - forecast) / se if se > 0 else math.nan
        passed = bool(resolution >= MIN_RESOLUTION and abs(miss) <= AGREE_SE)
        ok = ok and passed
        seats[seat] = {"measured": measured, "measured_plain": at[s][seat] - at[0.0][seat], "forecast": forecast,
                       "forecast_truncated": s * delta * sum(u_parts), "se": se, "resolution": resolution,
                       "miss_se": miss, "base": at[0.0][seat], "pass": passed}
    return {"class": cls, "delta": delta, "step": {"s": s, "kl": point.kl, "eps": eps, "k_side": k_side},
            "fd_step": h, "seats": seats, "games": {"train": len(grads["train_games"]), "readout": len(readout)},
            "rules": {"agree_se": AGREE_SE, "min_resolution": MIN_RESOLUTION},
            "verdict": "pass" if ok else "fail"}


# ---------------------------------------------------------------- the commands

def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def cmd_delta(args) -> int:
    from tools.eval_provenance import file_sha256
    from tools.pg_plant_grad import MassGradients
    rows = read_labels(args.labels)
    seeds = {r["game"]: int(r["seed"]) - args.seed_base for r in _read_jsonl(args.walk)}
    split = {r.game: r.split for r in rows}
    train = sorted({r.game for r in rows if not r.heldout
                    and (args.train_games <= 0 or seeds[r.game] < args.train_games)})
    heldout = sorted({r.game for r in rows if r.heldout})
    chosen = readout_games(heldout, args.readout_fraction, args.seed)
    index = RecordIndex(args.games)
    side_a = seat_of(index, chosen)
    sha = file_sha256(args.checkpoint)
    reader = MassGradients(args.checkpoint, {g: index.path(g) for g in train + chosen}, torch.device(args.device),
                           streams=args.streams, window=args.window, seed=args.seed, slots=args.slots,
                           workers=args.workers)
    try:
        u = reader.split_gradients({g: split[g] for g in train}, args.cls)
        grad_M = reader.seat_gradients(seat_sides(chosen, side_a), args.cls)
    finally:
        reader.close()
    if args.delta is not None:
        rule = {"delta": float(args.delta), "given": True}
    else:
        main_pass = torch.load(args.pass_path, map_location="cpu", weights_only=True)
        if main_pass["checkpoint_sha256"] != sha:
            raise SystemExit("the main pass is not this checkpoint's")
        rule = delta_rule(main_pass, grad_M, u, len(train))
    grads = {"class": args.cls, "delta": rule["delta"], "delta_rule": rule, "train_games": train,
             "heldout_games": heldout, "readout_games": chosen, "side_a": side_a, "checkpoint_sha256": sha,
             "config": {"streams": args.streams, "window": args.window, "seed": args.seed, "slots": args.slots},
             "u": u, "grad_M": grad_M}
    tmp = args.out.with_name(args.out.name + ".tmp")
    torch.save(grads, tmp)
    os.replace(tmp, args.out)
    log.info("PG_PLANT_DELTA %.6g (%s) on %d training games, %d read out", rule["delta"],
             "given" if args.delta is not None else f"{TARGET_SE:g} SE", len(train), len(chosen))
    return 0


def cmd_plant(args) -> int:
    grads = torch.load(args.grads, map_location="cpu", weights_only=True)
    keep = set(grads["train_games"]) | set(grads["heldout_games"])
    out = plant_rows(_read_jsonl(args.walk), RecordIndex(args.games), grads["delta"], grads["class"], keep)
    tmp = args.out.with_name(args.out.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in out:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    os.replace(tmp, args.out)
    log.info("PG_PLANT_WRITTEN %d games at delta %.6g into %s", len(out), grads["delta"], args.out)
    return 0


def cmd_readout(args) -> int:
    from tools.pg_eval import EvalConfig
    grads = torch.load(args.grads, map_location="cpu", weights_only=True)
    pass_ = torch.load(args.pass_path, map_location="cpu", weights_only=True)
    rows = read_labels(args.labels)
    index = RecordIndex(args.games)
    records = {g: index.path(g) for g in grads["readout_games"]}
    config = EvalConfig(streams=args.streams, workers=args.workers, kl_fraction=args.kl_fraction,
                        slots=grads["config"]["slots"])
    result = plant_readout(args.checkpoint, records, rows, pass_, grads, torch.device(args.device), config,
                           args.k_side)
    _write_json(args.out, result)
    log.info("PG_PLANT_READOUT %s %s", result["verdict"],
             json.dumps({s: {k: r[k] for k in ("resolution", "miss_se")} for s, r in result["seats"].items()}))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    from tools.pg_pass import PassConfig
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    defaults = PassConfig()
    for name in ("delta", "plant", "readout"):
        p = sub.add_parser(name)
        p.add_argument("--games", type=Path, nargs="+", required=True, help="the match directories")
        p.add_argument("--out", type=Path, required=True)
        if name != "plant":
            p.add_argument("--checkpoint", type=Path, required=True)
            p.add_argument("--labels", type=Path, required=True)
            p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
            p.add_argument("--workers", type=int, default=0)
            p.add_argument("--pass", dest="pass_path", type=Path, required=name == "readout")
    p = sub.choices["delta"]
    p.add_argument("--walk", type=Path, required=True, help="the luck walk's table (the games' seeds)")
    p.add_argument("--seed-base", type=int, required=True)
    p.add_argument("--train-games", type=int, default=0, help="plant the training games below this index (0: all)")
    p.add_argument("--class", dest="cls", choices=CLASSES, default=DEFAULT_CLASS)
    p.add_argument("--readout-fraction", type=float, default=0.2)
    p.add_argument("--delta", type=float, default=None, help="delta itself, in place of the rule")
    for name in ("streams", "window", "seed", "slots"):
        p.add_argument("--" + name, type=int, default=getattr(defaults, name), help="the planted pass's")
    p = sub.choices["plant"]
    p.add_argument("--walk", type=Path, required=True)
    p.add_argument("--grads", type=Path, required=True)
    p = sub.choices["readout"]
    p.add_argument("--grads", type=Path, required=True)
    p.add_argument("--streams", type=int, default=64)
    p.add_argument("--kl-fraction", type=float, default=0.05)
    p.add_argument("--k-side", type=float, default=K_SIDE)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if args.command == "delta" and args.delta is None and args.pass_path is None:
        ap.error("delta needs --pass (the main pass) or --delta")
    return {"delta": cmd_delta, "plant": cmd_plant, "readout": cmd_readout}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
