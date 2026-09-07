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

Usage (laptop CPU, ~369 games):
    python tools/analysis/value_head_by_phase.py --checkpoint training/checkpoints/seed.pt \\
        --out training/metrics/value_head/seed_by_phase.json [--limit 40]
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
    """Cost-weighted HP fraction, mover minus opponent, over the units
    the mover can see (wesnoth_ai.visibility, the encoder's own filter:
    own units, enemies inside the vision disc, hidden units excluded)."""
    from wesnoth_ai.visibility import units_visible_to
    ours = theirs = 0.0
    for u in units_visible_to(gs, mover):
        if u.side not in (1, 2) or u.max_hp <= 0:
            continue
        v = float(u.cost) * float(u.current_hp) / float(u.max_hp)
        if u.side == mover:
            ours += v
        else:
            theirs += v
    return ours - theirs


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


def head_values(policy, states: Sequence, batch: int = 16) -> List[float]:
    """The head's expected outcome for the side to move, batched on CPU."""
    import torch
    from tools.inference_seam import InferenceServer
    from wesnoth_ai.encoder import encode_raw
    enc = policy._inference_encoder
    server = InferenceServer(policy._inference_model, enc, device=torch.device("cpu"),
                             output_device=torch.device("cpu"), autocast_bf16=False)
    values: List[float] = []
    for i in range(0, len(states), batch):
        raws = []
        for gs in states[i:i + batch]:
            enc.register_names(gs)
            raws.append(encode_raw(gs, type_to_id=enc.unit_type_to_id,
                                   faction_to_id=enc.faction_to_id,
                                   relevant_set=bool(getattr(enc, "relevant_set_hexes", False))))
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


def summarize(rows: List[dict]) -> Dict:
    """rows: game, turn, side, winner_side, value, material."""
    by_game_turn: Dict[Tuple[str, int], Dict[int, dict]] = defaultdict(dict)
    for r in rows:
        by_game_turn[(r["game"], r["turn"])][r["side"]] = r
    per_bucket: Dict[str, dict] = {}
    for lo, hi in BUCKETS:
        name = bucket_of(lo)
        same_turn_head: Dict[str, List[float]] = defaultdict(list)
        same_turn_mat: Dict[str, List[float]] = defaultdict(list)
        pos_v, neg_v, pos_m, neg_m, brier = [], [], [], [], []
        for (game, turn), sides in by_game_turn.items():
            if not (lo <= turn <= hi):
                continue
            for r in sides.values():
                win = r["side"] == r["winner_side"]
                (pos_v if win else neg_v).append(r["value"])
                (pos_m if win else neg_m).append(r["material"])
                brier.append((((r["value"] + 1.0) / 2.0) - (1.0 if win else 0.0)) ** 2)
            if len(sides) == 2:
                w = sides[sides[1]["winner_side"]]
                ls = sides[3 - w["side"]]
                same_turn_head[game].append(
                    1.0 if w["value"] > ls["value"] else (0.5 if w["value"] == ls["value"] else 0.0))
                same_turn_mat[game].append(
                    1.0 if w["material"] > ls["material"] else
                    (0.5 if w["material"] == ls["material"] else 0.0))
        head_games = [float(np.mean(v)) for v in same_turn_head.values()]
        mat_games = [float(np.mean(v)) for v in same_turn_mat.values()]
        h, h_se = _mean_se(head_games)
        m, m_se = _mean_se(mat_games)
        per_bucket[name] = {
            "n_states": len(pos_v) + len(neg_v),
            "n_games_same_turn": len(head_games),
            "n_turn_pairs": int(sum(len(v) for v in same_turn_head.values())),
            "same_turn_auc_head": h, "same_turn_auc_head_se": h_se,
            "same_turn_auc_material": m, "same_turn_auc_material_se": m_se,
            "pooled_auc_head": pooled_auc(pos_v, neg_v),
            "pooled_auc_material": pooled_auc(pos_m, neg_m),
            "brier_head": float(np.mean(brier)) if brier else None,
            "winner_share": (len(pos_v) / (len(pos_v) + len(neg_v))
                             if pos_v or neg_v else None),
        }
    return per_bucket


def markdown(per_bucket: Dict) -> str:
    def f(v, spec=".3f"):
        return "-" if v is None else format(v, spec)
    lines = ["| turns | states | games | same-turn AUC head | same-turn AUC material | "
             "pooled AUC head | pooled AUC material | Brier head |",
             "|---|---|---|---|---|---|---|---|"]
    for name, b in per_bucket.items():
        lines.append(
            f"| {name} | {b['n_states']} | {b['n_games_same_turn']} | "
            f"{f(b['same_turn_auc_head'])} +- {f(b['same_turn_auc_head_se'])} | "
            f"{f(b['same_turn_auc_material'])} +- {f(b['same_turn_auc_material_se'])} | "
            f"{f(b['pooled_auc_head'])} | {f(b['pooled_auc_material'])} | {f(b['brier_head'])} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--limit", type=int, default=None, help="first N holdout games")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    import torch
    torch.set_num_threads(4)
    from tools.eval_sim import _load_policy
    policy = _load_policy(args.checkpoint, torch.device("cpu"), label="value_by_phase")
    manifest = [json.loads(line) for line in
                (args.dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
    games = [m for m in manifest if m.get("holdout")][:args.limit]
    t0 = time.time()
    rows: List[dict] = []
    for i, m in enumerate(games):
        data = json.load(gzip.open(args.dataset / m["file"], "rt", encoding="utf-8"))
        states = turn_start_states(data)
        vals = head_values(policy, [gs for _, _, gs in states])
        for (turn, side, gs), v in zip(states, vals):
            rows.append({"game": m["file"], "turn": turn, "side": side,
                         "winner_side": int(m["winner_side"]), "value": v,
                         "material": material(gs, side)})
        if (i + 1) % 20 == 0 or i + 1 == len(games):
            print(f"{i + 1}/{len(games)} games, {len(rows)} states, {time.time() - t0:.0f} s",
                  flush=True)
            if args.out:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.with_suffix(".partial.json").write_text(
                    json.dumps({"games": i + 1, "rows": rows}), encoding="utf-8")
    per_bucket = summarize(rows)
    report = markdown(per_bucket)
    print(report)
    if args.out:
        args.out.write_text(json.dumps({"checkpoint": str(args.checkpoint), "games": len(games),
                                        "n_states": len(rows), "buckets": per_bucket,
                                        "rows": rows}, indent=1), encoding="utf-8")
        args.out.with_suffix(".md").write_text(report + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
