"""Does the value head's reading earn a coefficient that changes with the
game's turn, or with its own size? (The follow-up to value_features.py: the
design's §4 rule, a cut of 2 SE and 5 % of the luck-only value,
cross-fitted over two folds by game.)

One coefficient alpha prices v_t at every decision today. Early in a game
v_t says less about z than late, so the coefficient that minimizes the noise
may grow with the turn; and E[R | v_t] need not be linear in v_t. Per
decision t of side s, the candidate features, every one a function of the
history before a_t (design §1.1):
    v           the value head's reading at t, the side's frame
    luck        beta^T x_t, the luck dealt before t along the labels' beta
    v_t<b>      v_t on turns of bucket b, 0 elsewhere (the turn counted
                from the end_turn commands before the decision's slot;
                bucket 1 is v's own)
    v3          v_t cubed

Per game-side the sufficient statistics of sum_t (R - c^T f_t)^2 are kept,
and each subset is fitted and read cross-fitted, its cut read over v+luck.

    <cpu python> training/metrics/luck_baseline_20261030/phase_value.py \
        --run C:/Users/momom2/wesnoth_runs/selfplay_t1_local --jobs 10 --out phase_value.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from value_features import fit, ratio_se, squared  # noqa: E402

TURN_EDGES = (4, 7, 10, 14)        # buckets: turns 1-3, 4-6, 7-9, 10-13, 14 on
FEATURES = ("v", "luck", "v_t2", "v_t3", "v_t4", "v_t5", "v3")
SUBSETS = {"v+luck": ("v", "luck"),
           "v+luck+phase": ("v", "luck", "v_t2", "v_t3", "v_t4", "v_t5"),
           "v+luck+v3": ("v", "luck", "v3"),
           "v+luck+phase+v3": FEATURES}


def bucket(turn: int) -> int:
    """1 to 5, the bucket of a 1-based turn."""
    return 1 + sum(turn >= e for e in TURN_EDGES)


def side_stats(item: Tuple[str, Tuple[float, ...]]) -> Dict[int, Dict]:
    from tools import game_record
    from tools.match_luck import COLUMNS, attack_luck
    from tools.pg_positions import decision_slots, read_record
    from wesnoth_ai.pg_records import decisions_of
    path, beta = item
    rec = read_record(Path(path))
    b = np.asarray(beta)
    attacks = []
    for k, cs, cmd in game_record.walk_core(rec, verify=False):
        if cmd[0] == "attack":
            terms = attack_luck(cs, list(cmd), 1).terms
            if terms is not None:
                attacks.append((k, float(b @ np.array([terms[c] for c in COLUMNS]))))
    ks = np.array([k for k, _ in attacks], dtype=np.int64)
    cum = np.cumsum([y for _, y in attacks]) if attacks else np.zeros(0)
    ends = np.array([k for k, c in enumerate(rec["commands"]) if c[0] == "end_turn"], dtype=np.int64)
    decisions = decisions_of(rec)
    rejections = [(int(k), int(x), int(y)) for k, x, y in rec.get("rejections", ())]
    slots = decision_slots(decisions, rejections, len(rec["commands"]))
    p = len(FEATURES)
    out = {s: {"n": 0, "sf": np.zeros(p), "sff": np.zeros((p, p)), "turns": 0} for s in (1, 2)}
    for d, (k, _r) in zip(decisions, slots):
        s = d.side
        j = int(np.searchsorted(ks, k, side="left"))
        y = (cum[j - 1] if j > 0 else 0.0) * (1.0 if s == 1 else -1.0)
        v = float(d.v) if np.isfinite(d.v) else 0.0
        turn = int(np.searchsorted(ends, k, side="left")) // 2 + 1      # side-turns ended before the slot
        f = np.zeros(p)
        f[0], f[1], f[6] = v, y, v ** 3
        bk = bucket(turn)
        if bk > 1:
            f[bk] = v                                   # v_t2 .. v_t5 sit at indices 2 .. 5
        st = out[s]
        st["n"] += 1
        st["sf"] += f
        st["sff"] += np.outer(f, f)
        st["turns"] = max(st["turns"], turn)
    return out


def main() -> int:
    import value_features
    from tools.pg_positions import RecordIndex
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    value_features.FEATURES = FEATURES             # fit() sizes its system by FEATURES
    beta = tuple(json.loads((args.run / "fits.json").read_text(encoding="utf-8"))["fits"]["train"]["beta"])
    R = {}
    for line in (args.run / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            R[(row["game"], int(row["side"]))] = float(row["R"])
    games = sorted({g for g, _ in R})
    if args.limit:
        games = games[:args.limit]
    index = RecordIndex(sorted(args.run.glob("games_s*")))
    t0 = time.time()
    with get_context("spawn").Pool(args.jobs) as pool:
        stats = pool.map(side_stats, [(str(index.path(g)), beta) for g in games], chunksize=16)
    rows = [{"fold": i % 2, "R": R[(g, s)], **st[s]} for i, (g, st) in enumerate(zip(games, stats)) for s in (1, 2)]
    n = np.array([r["n"] for r in rows], float).reshape(-1, 2).sum(1)
    luck0 = np.array([squared(r, np.zeros(len(FEATURES))) for r in rows]).reshape(-1, 2).sum(1)
    base_E = ratio_se(luck0, n)[0]
    turns = np.array([r["turns"] for r in rows])
    result = {"games": len(games), "seconds": time.time() - t0, "turn_edges": TURN_EDGES,
              "luck_alone": dict(zip(("E", "se"), ratio_se(luck0, n))),
              "side_turns": {"median": float(np.median(turns)),
                                                          "p90": float(np.percentile(turns, 90))}}
    per_game = {}
    for name, feats in SUBSETS.items():
        keep = [FEATURES.index(f) for f in feats]
        q = np.zeros(len(rows))
        coefs = []
        for fold in (0, 1):
            c = fit([r for r in rows if r["fold"] != fold], keep)
            coefs.append(c.tolist())
            for i, r in enumerate(rows):
                if r["fold"] == fold:
                    q[i] = squared(r, c)
        per_game[name] = q.reshape(-1, 2).sum(1)
        E, se = ratio_se(per_game[name], n)
        result[name] = {"E": E, "se": se, "share_cut": (base_E - E) / base_E, "coefficients_by_fold": coefs}
    ref = per_game["v+luck"]
    for name, q in per_game.items():
        if name != "v+luck":
            cut, cut_se = ratio_se(ref - q, n)
            result[name]["over_v_luck"] = {"cut": cut, "cut_se": cut_se, "share": cut / base_E,
                                           "enters": bool(cut >= 2 * cut_se and cut / base_E >= 0.05)}
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for name in SUBSETS:
        extra = result[name].get("over_v_luck")
        print(f"{name:18s} E {result[name]['E']:.4f}" + (f"  over v+luck: {extra['cut']:+.4f} +- "
                                                         f"{extra['cut_se']:.4f} ({100 * extra['share']:.2f}%)"
                                                         if extra else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
