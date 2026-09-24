"""Value head and material against human game outcomes, by game phase.

Holdout games of the imitation manifest (never trained on). At every
side's turn start (the state after its init_side, what the mover
faces), the label is
whether the side to move went on to win. Two predictors of the label:
the value head's expected outcome for the side to move, and material
(sum over the mover's units of cost x HP fraction minus the same over
the opponent's units the mover can see, the encoder's own fog filter).

Same-turn AUC: for each game and turn t where both sides' turn-start
states exist, does the predictor rate the eventual winner's state
above the eventual loser's? (Both readings are from the mover's side,
so the winner's should be the higher one.) Averaged per game inside
each turn bucket, then across games; the error bar is between games.
This is the like-for-like comparison: the pooled AUC of the training
evals mixes turns and rewards knowing that late positions are
decided. Also reported: the pooled AUC per bucket over single states
(winner-to-move against loser-to-move, across games) and the head's
Brier score with P(win) = (value + 1) / 2.

Usage (a box: --jobs for the reconstruction, --device cuda for the head):
    python tools/analysis/value_head_by_phase.py --checkpoint CKPT.pt --jobs 12 \\
        --device cuda --out training/metrics/value_head/seed_by_phase.json [--limit 40]
"""
import argparse
import copy
import gzip
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

BUCKETS = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 10 ** 6)]
DATASET = ROOT / "replays_dataset_imitation"


def bucket_of(turn: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= turn <= hi:
            return f"{lo}-{hi}" if hi < 10 ** 6 else f"{lo}+"
    return "?"


def material(gs, mover: int) -> float:
    """The study's material metric (wesnoth_ai/material.py: cost x HP
    fraction, mover minus the enemies the mover can see)."""
    from wesnoth_ai.material import material_score
    return material_score(gs, mover)


def turn_start_states(data: dict):
    """(turn, side, GameState) at every player side's turn start."""
    from tools.replay_dataset import (_apply_command, _build_initial_gamestate,
                                      _setup_scenario_events)
    gs = _build_initial_gamestate(data)
    _setup_scenario_events(gs, data.get("scenario_id", ""))
    out = []
    for cmd in data.get("commands", []):
        _apply_command(gs, cmd)
        if cmd and cmd[0] == "init_side" and len(cmd) > 1 and cmd[1] in (1, 2):
            if not {1, 2} <= {u.side for u in gs.map.units if u.is_leader}:
                break
            out.append((int(gs.global_info.turn_number), int(cmd[1]), copy.deepcopy(gs)))
    return out


def village_lead(gs, mover: int, visible_only: bool) -> float:
    """Own villages minus the enemy's: the true count (what global
    feature 5 carried before the fog gate) or the count on hexes the
    mover sees (what a player can know under fog)."""
    from wesnoth_ai.visibility import enemy_villages_visible_to
    own = gs.sides[mover - 1].nb_villages_controlled
    enemy = 3 - mover
    if visible_only:
        return float(own - enemy_villages_visible_to(gs, mover))
    return float(own - gs.sides[enemy - 1].nb_villages_controlled)


# Standalone predictors scored beside the head: (column, row key).
PREDICTORS = (("material", "material"), ("villages true", "villages_true"),
              ("villages seen", "villages_seen"))


def _game_rows(args):
    """One game's turn-start states as picklable rows (worker side):
    turn, side, the predictors and the RawEncoded for the head."""
    path, winner_side, type_to_id, faction_to_id, relevant_set, gate_villages, terrain_multi_hot = args
    from wesnoth_ai.encoder import encode_raw
    data = json.load(gzip.open(path, "rt", encoding="utf-8"))
    rows = []
    for turn, side, gs in turn_start_states(data):
        raw = encode_raw(gs, type_to_id=type_to_id, faction_to_id=faction_to_id,
                         relevant_set=relevant_set, fog_hides_enemy_villages=gate_villages,
                         terrain_multi_hot=terrain_multi_hot)
        rows.append({"turn": turn, "side": side, "winner_side": int(winner_side),
                     "material": material(gs, side),
                     "villages_true": village_lead(gs, side, visible_only=False),
                     "villages_seen": village_lead(gs, side, visible_only=True),
                     "raw": raw})
    return rows


def head_values(policy, states: Sequence, batch: int = 16, device=None) -> List[float]:
    """The head's expected outcome for the side to move, batched."""
    import torch
    from tools.inference_seam import InferenceServer
    enc = policy._inference_encoder
    device = device or torch.device("cpu")
    server = InferenceServer(policy._inference_model, enc, device=device,
                             output_device=torch.device("cpu"), autocast_bf16=False)
    values: List[float] = []
    for i in range(0, len(states), batch):
        raws = []
        for gs in states[i:i + batch]:
            enc.register_names(gs)
            raws.append(enc.raw_of(gs))
        for out in server.infer_batch(raws):
            values.append(float(out.value.reshape(-1)[0].item()))
    return values


def _mean_se(xs: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    if not xs:
        return None, None
    m = float(np.mean(xs))
    se = float(np.std(xs, ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else None
    return m, se


def pooled_auc(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    if not pos or not neg:
        return None
    p = np.asarray(pos)
    n = np.asarray(neg)
    wins = (p[:, None] > n[None, :]).sum() + 0.5 * (p[:, None] == n[None, :]).sum()
    return float(wins / (len(p) * len(n)))


def _same_turn_score(winner: dict, loser: dict, key: str) -> float:
    if winner[key] > loser[key]:
        return 1.0
    return 0.5 if winner[key] == loser[key] else 0.0


def summarize(rows: List[dict]) -> Dict:
    """rows: game, turn, side, winner_side, value and the predictors.
    Per bucket: the head's and each predictor's same-turn AUC (mean
    over games of the share of turns where the winner scores higher)
    and pooled AUC; a predictor absent from the rows (older records)
    is skipped."""
    by_game_turn: Dict[Tuple[str, int], Dict[int, dict]] = defaultdict(dict)
    for r in rows:
        by_game_turn[(r["game"], r["turn"])][r["side"]] = r
    scored = [("head", "value")] + [(k, k) for _, k in PREDICTORS if rows and all(k in r for r in rows)]
    per_bucket: Dict[str, dict] = {}
    for lo, hi in BUCKETS:
        name = bucket_of(lo)
        same_turn = {k: defaultdict(list) for _, k in scored}
        pos = {k: [] for _, k in scored}
        neg = {k: [] for _, k in scored}
        brier = []
        for (game, turn), sides in by_game_turn.items():
            if not (lo <= turn <= hi):
                continue
            for r in sides.values():
                win = r["side"] == r["winner_side"]
                for _, k in scored:
                    (pos[k] if win else neg[k]).append(r[k])
                brier.append((((r["value"] + 1.0) / 2.0) - (1.0 if win else 0.0)) ** 2)
            if len(sides) == 2:
                w = sides[sides[1]["winner_side"]]
                ls = sides[3 - w["side"]]
                for _, k in scored:
                    same_turn[k][game].append(_same_turn_score(w, ls, k))
        b = {"n_states": len(pos["value"]) + len(neg["value"]),
             "n_games_same_turn": len(same_turn["value"]),
             "n_turn_pairs": int(sum(len(v) for v in same_turn["value"].values())),
             "brier_head": float(np.mean(brier)) if brier else None,
             "winner_share": (len(pos["value"]) / (len(pos["value"]) + len(neg["value"]))
                             if pos["value"] or neg["value"] else None)}
        for label, k in scored:
            per_game = [float(np.mean(v)) for v in same_turn[k].values()]
            b[f"same_turn_auc_{label}"], b[f"same_turn_auc_{label}_se"] = _mean_se(per_game)
            b[f"pooled_auc_{label}"] = pooled_auc(pos[k], neg[k])
        per_bucket[name] = b
    return per_bucket


def markdown(per_bucket: Dict) -> str:
    def f(v, spec=".3f"):
        return "-" if v is None else format(v, spec)
    first = next(iter(per_bucket.values()), {})
    scored = ["head"] + [k for _, k in PREDICTORS if f"same_turn_auc_{k}" in first]
    label = {k: c for c, k in PREDICTORS}
    label["head"] = "head"
    lines = ["| turns | states | games | "
             + " | ".join(f"same-turn AUC {label[k]}" for k in scored) + " | "
             + " | ".join(f"pooled AUC {label[k]}" for k in scored) + " | Brier head |",
             "|---" * (4 + 2 * len(scored)) + "|"]
    for name, b in per_bucket.items():
        lines.append(
            f"| {name} | {b['n_states']} | {b['n_games_same_turn']} | "
            + " | ".join(f"{f(b[f'same_turn_auc_{k}'])} +- {f(b[f'same_turn_auc_{k}_se'])}"
                         for k in scored) + " | "
            + " | ".join(f(b[f"pooled_auc_{k}"]) for k in scored)
            + f" | {f(b['brier_head'])} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--limit", type=int, default=None, help="first N holdout games")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Worker processes reconstructing and encoding the games; "
                         "the head runs in this process.")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--gate-enemy-villages", action="store_true",
                    help="Feed the head global feature 5 gated by fog (the visible enemy "
                         "villages) whatever the checkpoint says: what the head loses "
                         "when the hidden count is taken away.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    import torch
    torch.set_num_threads(4)
    from tools.eval_sim import _load_policy
    device = torch.device(args.device)
    policy = _load_policy(args.checkpoint, device, label="value_by_phase")
    enc = policy._inference_encoder
    manifest = [json.loads(line) for line in
                (args.dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
    games = [m for m in manifest if m.get("holdout")][:args.limit]
    t0 = time.time()
    rows: List[dict] = []
    gate = bool(args.gate_enemy_villages or getattr(enc, "fog_hides_enemy_villages", False))
    tasks = [(str(args.dataset / m["file"]), m["winner_side"], dict(enc.unit_type_to_id),
              dict(enc.faction_to_id), bool(getattr(enc, "relevant_set_hexes", False)), gate,
              bool(getattr(enc, "terrain_multi_hot", False)))
             for m in games]
    if args.jobs > 1:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(args.jobs)
        game_iter = zip(games, pool.imap(_game_rows, tasks, chunksize=2))
    else:
        pool = None
        game_iter = ((m, _game_rows(t)) for m, t in zip(games, tasks))
    from tools.inference_seam import InferenceServer
    server = InferenceServer(policy._inference_model, enc, device=device,
                             output_device=torch.device("cpu"), autocast_bf16=False)
    for i, (m, game_rows) in enumerate(game_iter):
        raws = [r.pop("raw") for r in game_rows]
        vals = []
        for j in range(0, len(raws), 16):
            vals.extend(float(o.value.reshape(-1)[0].item())
                        for o in server.infer_batch(raws[j:j + 16]))
        for r, v in zip(game_rows, vals):
            r["game"] = m["file"]
            r["value"] = v
            rows.append(r)
        if (i + 1) % 20 == 0 or i + 1 == len(games):
            print(f"{i + 1}/{len(games)} games, {len(rows)} states, {time.time() - t0:.0f} s",
                  flush=True)
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.with_suffix(".partial.json").write_text(
                    json.dumps({"games": i + 1, "rows": rows}), encoding="utf-8")
    if pool is not None:
        pool.close()
    per_bucket = summarize(rows)
    report = markdown(per_bucket)
    print(report)
    if args.out:
        args.out.write_text(json.dumps({"checkpoint": str(args.checkpoint), "games": len(games),
                                        "gate_enemy_villages": gate,
                                        "n_states": len(rows), "buckets": per_bucket,
                                        "rows": rows}, indent=1), encoding="utf-8")
        args.out.with_suffix(".md").write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
