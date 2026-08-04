#!/usr/bin/env python3
"""T-B analyzer: pooled paired-AUC comparison of raw value head vs
32-sim search root value (docs/eval_box.md queue, 2026-08-04).

Consumes the JSONL shards written by tools/probe_teacher_advantage.py
(rows: {file, turn, side, z, ev_raw, ev_search}; error rows carry
"err" and are counted but excluded). Reports, pooled and per phase
(open turns 1-8 / mid 9-16 / late 17+):

  - AUC(ev_raw), AUC(ev_search) for predicting z=+1 vs z=-1
  - delta = AUC(search) - AUC(raw), with a game-level cluster
    bootstrap 95% CI (states within a game are correlated; rows are
    resampled by game, not by state)

Pre-registered kill bar: if pooled delta <= +0.02, the search
teacher adds nothing worth distilling into the value head at the
campaign budget -- kill the value-distillation channel.

    python tools/analyze_teacher_advantage.py tb_part*.jsonl
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

PHASES = (("open", 1, 8), ("mid", 9, 16), ("late", 17, 10**9))
BOOT = 2000


def auc(pairs):
    """Rank-based AUC of score predicting z=+1 (ties count 0.5)."""
    pos = [s for s, z in pairs if z > 0]
    neg = [s for s, z in pairs if z < 0]
    if not pos or not neg:
        return None
    ranked = sorted(pairs, key=lambda t: t[0])
    # midrank assignment for ties
    ranks = {}
    i = 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        mid = (i + j + 1) / 2.0  # 1-based midrank
        for k in range(i, j):
            ranks.setdefault(id(ranked[k]), mid)
        i = j
    rsum = sum(ranks[id(t)] for t in ranked if t[1] > 0)
    n_pos, n_neg = len(pos), len(neg)
    return (rsum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def deltas_for(rows):
    """(auc_raw, auc_search, delta) for a list of rows, or None."""
    a_r = auc([(r["ev_raw"], r["z"]) for r in rows])
    a_s = auc([(r["ev_search"], r["z"]) for r in rows])
    if a_r is None or a_s is None:
        return None
    return a_r, a_s, a_s - a_r


def cluster_bootstrap_ci(by_game, rng, n_boot=BOOT):
    """95% CI of pooled delta, resampling GAMES with replacement."""
    games = list(by_game)
    ds = []
    for _ in range(n_boot):
        sample = [by_game[rng.choice(games)] for _ in games]
        rows = [r for g in sample for r in g]
        d = deltas_for(rows)
        if d is not None:
            ds.append(d[2])
    if not ds:
        return None
    ds.sort()
    return ds[int(0.025 * len(ds))], ds[int(0.975 * len(ds))]


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    rows, n_err = [], 0
    for pat in argv[1:]:
        for p in (Path(".").glob(pat) if any(c in pat for c in "*?")
                  else [Path(pat)]):
            with p.open(encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if "err" in r:
                        n_err += 1
                    else:
                        rows.append(r)
    print(f"states: {len(rows)}  error rows: {n_err}  "
          f"games: {len({r['file'] for r in rows})}")
    if not rows:
        return 1

    for name, lo, hi in (("pooled", 1, 10**9),) + PHASES:
        sub = [r for r in rows if lo <= r["turn"] <= hi]
        d = deltas_for(sub)
        if d is None:
            print(f"{name:>6}: n={len(sub)} (one-class; no AUC)")
            continue
        a_r, a_s, delta = d
        line = (f"{name:>6}: n={len(sub)}  AUC raw={a_r:.4f}  "
                f"search={a_s:.4f}  delta={delta:+.4f}")
        if name == "pooled":
            by_game = defaultdict(list)
            for r in sub:
                by_game[r["file"]].append(r)
            ci = cluster_bootstrap_ci(dict(by_game), random.Random(7))
            if ci:
                line += f"  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]"
                verdict = ("KILL value channel (delta <= +0.02)"
                           if delta <= 0.02 else
                           "search teacher ADDS signal (delta > +0.02)"
                           + (" -- but CI crosses the bar"
                              if ci[0] <= 0.02 else ""))
                line += f"\n        verdict vs +0.02 bar: {verdict}"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
