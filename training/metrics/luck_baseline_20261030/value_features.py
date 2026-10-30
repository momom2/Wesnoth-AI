"""Which functions of the history, beside the value head's reading at the
decision and the luck dealt so far, cut the label's noise further? (The
follow-up to partial_luck.py: the design's §4 rule, a cut of 2 SE and 5 %
of the luck-only value, cross-fitted over two folds by game.)

Per decision t of side s, the candidate baseline features (every one a
function of the history before a_t, so any linear mix keeps the gradient
unbiased, design §1.1):
    v       the value head's reading at t, the side's frame
    luck    beta^T x_t, the luck dealt before t along the labels' beta
    v_opp   minus the opponent's most recent reading before t (its frame
            negated: an independent forward of the same game; 0 before the
            opponent's first decision)
    v_prev  the side's own previous reading (0 at its first decision)
    v_mean  the mean of the side's readings so far, t included

Per game-side the sufficient statistics of sum_t (R - c^T f_t)^2 are kept
(n, sum f, sum f f^T), and any subset of features is fitted and read
cross-fitted.

    <cpu python> training/metrics/luck_baseline_20261030/value_features.py \
        --run C:/Users/momom2/wesnoth_runs/selfplay_t1_local --jobs 10 --out value_features.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from itertools import combinations
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

FEATURES = ("v", "luck", "v_opp", "v_prev", "v_mean")


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
    decisions = decisions_of(rec)
    rejections = [(int(k), int(x), int(y)) for k, x, y in rec.get("rejections", ())]
    slots = decision_slots(decisions, rejections, len(rec["commands"]))
    p = len(FEATURES)
    out = {s: {"n": 0, "sf": np.zeros(p), "sff": np.zeros((p, p))} for s in (1, 2)}
    last = {1: None, 2: None}                 # each side's latest reading, its own frame
    seen = {1: [], 2: []}
    for d, (k, _r) in zip(decisions, slots):
        s, o = d.side, 3 - d.side
        j = int(np.searchsorted(ks, k, side="left"))
        y = (cum[j - 1] if j > 0 else 0.0) * (1.0 if s == 1 else -1.0)
        v = float(d.v)
        v_opp = -last[o] if last[o] is not None else 0.0
        v_prev = last[s] if last[s] is not None else 0.0
        seen[s].append(v)
        f = np.array([v, y, v_opp, v_prev, float(np.mean(seen[s]))])
        st = out[s]
        st["n"] += 1
        st["sf"] += f
        st["sff"] += np.outer(f, f)
        last[s] = v
    return out


def squared(r: Dict, c: np.ndarray) -> float:
    R = r["R"]
    return float(r["n"] * R * R - 2 * R * (c @ r["sf"]) + c @ r["sff"] @ c)


def fit(rows: Sequence[Dict], keep: Sequence[int]) -> np.ndarray:
    p = len(FEATURES)
    A, y = np.zeros((p, p)), np.zeros(p)
    for r in rows:
        A += r["sff"]
        y += r["R"] * r["sf"]
    c = np.zeros(p)
    if keep:
        k = list(keep)
        c[k] = np.linalg.lstsq(A[np.ix_(k, k)], y[k], rcond=None)[0]
    return c


def ratio_se(num: np.ndarray, den: np.ndarray) -> Tuple[float, float]:
    m = num.sum() / den.sum()
    resid = num - m * den
    G = len(num)
    return float(m), float(math.sqrt(G / (G - 1) * float((resid ** 2).sum())) / den.sum())


def main() -> int:
    from tools.pg_positions import RecordIndex
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
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
    result = {"games": len(games), "seconds": time.time() - t0, "luck_alone": dict(zip(("E", "se"),
                                                                                     ratio_se(luck0, n)))}
    per_game = {}
    subsets = [s for k in range(1, len(FEATURES) + 1) for s in combinations(range(len(FEATURES)), k)
               if 0 in s and 1 in s] + [(0,)]
    for keep in subsets:
        q = np.zeros(len(rows))
        coefs = []
        for f in (0, 1):
            c = fit([r for r in rows if r["fold"] != f], keep)
            coefs.append(c.tolist())
            for i, r in enumerate(rows):
                if r["fold"] == f:
                    q[i] = squared(r, c)
        name = "+".join(FEATURES[i] for i in keep)
        per_game[name] = q.reshape(-1, 2).sum(1)
        E, se = ratio_se(per_game[name], n)
        result[name] = {"E": E, "se": se, "share_cut": (base_E - E) / base_E, "coefficients_by_fold": coefs}
    ref = per_game["v+luck"]
    for name, q in per_game.items():
        if name != "v+luck" and name.startswith("v+luck"):
            cut, cut_se = ratio_se(ref - q, n)
            result[name]["over_v_luck"] = {"cut": cut, "cut_se": cut_se, "share": cut / base_E}
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for name in result:
        if isinstance(result[name], dict) and "E" in result[name]:
            extra = result[name].get("over_v_luck")
            print(f"{name:28s} E {result[name]['E']:.4f}" + (f"  over v+luck: {extra['cut']:+.4f} +- "
                                                              f"{extra['cut_se']:.4f} ({100 * extra['share']:.2f}%)"
                                                              if extra else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
