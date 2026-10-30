"""Stage P's readouts (docs/training_design_20261019/current_design.md §5
"Planted round", its P half, and §3.4 / §6 "the link at temperature 1"),
from the matches' records:

`planted`: the planted step theta' (tools/pg_plant.py readout --candidate)
against pi_0, both at temperature 1:
  - X per side-game rises on each seat. A seat is the game's side, 1 (who
    moves first) or 2, which the match alternates (`side_a`) while the labels
    keep their places: run_elo_batch puts the candidate on label A in every
    game. Per seat, theta''s mean count of its played actions in X per
    side-game, over the games it held that side, is set against pi_0's over
    the games pi_0 held it. The difference has an SE over independent games,
    and each seat passes at 2 SE.
  - The played law is the trainer's theta'. Every game-side theta' played
    must name the candidate's checkpoint (its SHA-256, the match results'
    checkpoint_sha256_<seat>) at its memory slots under raw:t1. The served
    numerics are the rounds' own, which each round's replay identity holds
    to the trainer (tools/pg_pass.py), so the weights are what this check
    adds.

`link`: the antisymmetric temperature match, pi_0 at T = 1/(1 + e) on side A
against pi_0 at T = 1/(1 - e) on side B. Stage 1's sharpening slope (its
readout.json) is the first-order value of sampling colder, d E[z]/d beta at
beta = 1 against a temperature-1 opponent. By the game's antisymmetry the
forecast E[z_A] = 2 e slope is exact to second order, with SE 2 e se(slope).
The match's mean z_A is read over the games that ended in play: a win, a
loss, or a cap (the turn or action horizon, or no progress) at 0. A game
killed on the wall clock (`timeout_kill`, run_elo_batch) ends on the
machine's speed, not the play, and is left out. The link holds when
the two agree within 2 combined SE. A miss means first-order labels do not
price play at temperature 1: stop and diagnose before rounds (§6).

    python tools/pg_planted_play.py planted --games DIR --planted-label planted --candidate planted.pt --out p.json
    python tools/pg_planted_play.py link --games DIR --readout stage1/readout.json --e 0.02 --out link.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from wesnoth_ai.server_priors import KIND_ATTACK, KIND_END_TURN, KIND_RECRUIT  # noqa: E402

KIND_OF = {"attack": KIND_ATTACK, "recruit": KIND_RECRUIT, "end_turn": KIND_END_TURN}
RISE_SE = 2.0
SEATS = ("side_1", "side_2")
DECIDED = {"win": 1.0, "loss": -1.0}
CAPPED = ("draw", "timeout")          # the game's own horizons: z 0
LINK_SE = 2.0
RESULT_SUFFIX = ".json"
RECORD_SUFFIX = ".game.jsonl.gz"


def _results(games: Path) -> List[Dict]:
    out = []
    for p in sorted(Path(games).glob("game_*" + RESULT_SUFFIX)):
        if p.name.endswith(RECORD_SUFFIX):
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        r["_record"] = p.with_name(p.name[:-len(RESULT_SUFFIX)] + RECORD_SUFFIX)
        out.append(r)
    return out


def _mean_se(x: Sequence[float]) -> tuple:
    a = np.asarray(x, dtype=float)
    return (float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a)))) if len(a) > 1 else (float("nan"),) * 2


def planted(games: Path, planted_label: str, candidate_sha: str, cls: str = "attack",
            slots: int = 64) -> Dict:
    """The `planted` readout (module docstring)."""
    from tools.pg_positions import read_record
    from wesnoth_ai.pg_records import decisions_of
    kind = KIND_OF[cls]
    counts = {(who, seat): [] for who in ("planted", "base") for seat in SEATS}
    wrong: List[str] = []
    for r in _results(games):
        side_a = int(r["side_a"])
        key = "a" if r["label_a"] == planted_label else "b" if r["label_b"] == planted_label else None
        if key is None:
            wrong.append(f"{r['_record'].name}: no side is {planted_label}")
            continue
        if (r.get(f"checkpoint_sha256_{key}") != candidate_sha or r.get(f"memory_{key}") != slots
                or r.get(f"procedure_{key}") != "raw:t1"):
            wrong.append(f"{r['_record'].name}: label {key.upper()} played "
                         f"{str(r.get(f'checkpoint_sha256_{key}'))[:12]} at {r.get(f'memory_{key}')} slots under "
                         f"{r.get(f'procedure_{key}')}")
            continue
        planted_side = side_a if key == "a" else 3 - side_a
        played = [d for d in decisions_of(read_record(r["_record"])) if not d.refused]
        for side in (1, 2):
            who = "planted" if side == planted_side else "base"
            counts[(who, SEATS[side - 1])].append(sum(d.kind == kind for d in played if d.side == side))
    seats = {}
    for seat in SEATS:
        (mp, sp), (mb, sb) = _mean_se(counts[("planted", seat)]), _mean_se(counts[("base", seat)])
        diff, se = mp - mb, math.sqrt(sp * sp + sb * sb)
        seats[seat] = {"planted": mp, "planted_se": sp, "base": mb, "base_se": sb, "rise": diff, "se": se,
                       "z": diff / se if se > 0 else float("nan"), "games": [len(counts[("planted", seat)]),
                                                                            len(counts[("base", seat)])],
                       "pass": bool(se > 0 and diff > RISE_SE * se)}
    ok = all(s["pass"] for s in seats.values()) and not wrong
    return {"class": cls, "seats": seats, "weights_check": {"games_refused": len(wrong), "examples": wrong[:5]},
            "rule": {"rise_se": RISE_SE}, "verdict": "pass" if ok else "fail"}


def link(games: Path, readout: Path, e: float) -> Dict:
    """The `link` readout (module docstring)."""
    sharp = json.loads(Path(readout).read_text(encoding="utf-8"))["sharpening"]
    forecast, forecast_se = 2.0 * e * float(sharp["slope"]), 2.0 * e * float(sharp["se"])
    zs, left_out = [], 0
    for r in _results(games):
        outcome = r.get("outcome_a")
        if outcome in DECIDED or outcome in CAPPED:
            zs.append(DECIDED.get(outcome, 0.0))
        else:
            left_out += 1
    measured, measured_se = _mean_se(zs)
    combined = math.sqrt(measured_se ** 2 + forecast_se ** 2)
    miss = (measured - forecast) / combined if combined > 0 else float("nan")
    holds = bool(abs(measured - forecast) <= LINK_SE * combined)
    return {"e": e, "temperatures": {"a": 1.0 / (1.0 + e), "b": 1.0 / (1.0 - e)},
            "slope": {"estimate": float(sharp["slope"]), "se": float(sharp["se"])},
            "forecast": forecast, "forecast_se": forecast_se, "measured": measured, "measured_se": measured_se,
            "games": len(zs), "games_left_out": left_out, "miss_se": miss, "rule": {"link_se": LINK_SE},
            "verdict": "holds" if holds else "misses: first-order labels do not price play at temperature 1"}


def _write(path: Path, obj: Dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from tools.eval_provenance import file_sha256
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("planted")
    p.add_argument("--games", type=Path, required=True)
    p.add_argument("--planted-label", required=True)
    p.add_argument("--candidate", type=Path, required=True, help="the planted step's checkpoint")
    p.add_argument("--class", dest="cls", choices=sorted(KIND_OF), default="attack")
    p.add_argument("--slots", type=int, default=64)
    p.add_argument("--out", type=Path, required=True)
    q = sub.add_parser("link")
    q.add_argument("--games", type=Path, required=True)
    q.add_argument("--readout", type=Path, required=True, help="stage 1's readout.json (its sharpening slope)")
    q.add_argument("--e", type=float, required=True)
    q.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.command == "planted":
        result = planted(args.games, args.planted_label, file_sha256(args.candidate), args.cls, args.slots)
        print(json.dumps({"verdict": result["verdict"], "rise_z": {s: r["z"] for s, r in result["seats"].items()}}))
    else:
        result = link(args.games, args.readout, args.e)
        print(json.dumps({k: result[k] for k in ("verdict", "forecast", "measured", "miss_se")}))
    _write(args.out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
