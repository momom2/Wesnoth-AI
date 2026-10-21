"""Round 10, challenge stance: how fast does the recruit structure saturate in
games? Rules A (43 values, ridge 5,000) and B (matchup values shrunk toward A,
ridge 5,000) fitted on a fraction of batch 1's games and read on held-out
games, to see whether a step's direction needs one batch or eight.

Reads the per-match sums of round10_challenge_recruit_context.py extract.
Fractions: rules fitted on k of 10 fold-groups (folds 2g and 2g+1), read on
the 10 - k others; every split rotates the fold-groups and averages (each
reading is out of sample; the rotations share games, so the spread of the
rotations is not an SE). Reported per side-game of the games read.
numpy and the standard library only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from round10_challenge_recruit_context import ELO, MATCHES, design, fit

HERE = Path(__file__).resolve().parent


def main() -> None:
    rows, sg = [], 0
    for m in MATCHES:
        d = json.loads((HERE / f"round10_challenge_recruit_context_{m}.json").read_text())
        sg += d["side_games"]
        for key, (n, s, ss) in d["cells"].items():
            fac, u0, u1, enemy, tb, f = key.split("|")
            rows.append((fac, u0, u1, enemy, int(tb), int(f), n, s, ss))
    n = np.asarray([r[6] for r in rows], dtype=float)
    s = np.asarray([r[7] for r in rows])
    ss = np.asarray([r[8] for r in rows])
    group = np.asarray([r[5] for r in rows]) // 2
    cA = design(rows, lambda r, u: f"{r[0]}|{u}")
    cB = design(rows, lambda r, u: f"{r[0]}|{u}|{r[3]}")
    games = sg // 2
    print(f"{games} games, 10 fold-groups of about {games // 10} games")
    print("fit games  | A read on the rest, Elo per side-game (mean SE) | B read on the rest")
    for k in (1, 2, 5, 9):
        res = {"A": [], "B": []}
        ses = {"A": [], "B": []}
        for rot in range(10):
            fitg = {(rot + i) % 10 for i in range(k)}
            train = np.isin(group, list(fitg))
            test = ~train
            pA = fit(cA[0], cA[1], cA[2], n, s, train, 5000.0)
            pB = fit(cB[0], cB[1], cB[2], n, s, train, 5000.0, prior=pA)
            sg_test = sg * (10 - k) / 10
            for name, p in (("A", pA), ("B", pB)):
                sw = test & (p > 0)
                res[name].append(s[sw].sum() / sg_test * ELO)
                ses[name].append(math.sqrt(ss[sw].sum()) / sg_test * ELO)
        print(f"  {k * games // 10:6d}   | A {np.mean(res['A']):+5.1f} ({np.mean(ses['A']):.1f})"
              f"                          | B {np.mean(res['B']):+5.1f} ({np.mean(ses['B']):.1f})")


if __name__ == "__main__":
    main()
