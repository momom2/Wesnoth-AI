"""Round 10, challenge stance: choose the matchup-keyed recruit model's ridge by
the design's progressive loss (round9_synthesis_checks.py section 1), not by
welfare, then report its cross-fitted welfare at the chosen ridge.

Reads the per-match sums written by round10_challenge_recruit_context.py
extract. Model A: 43 additive (faction, unit) values at ridge 5,000 (R, the
progressive loss's choice for A). Model B: (faction, unit, enemy faction)
values shrunk toward A. Progressive loss, matches in game order b00..b06:
student j-1 is fitted on matches before j and scored on match j by
sum over cells of [-2 S_c p_c + n_c p_c^2] (lower is better; its expectation
is the squared error of p against the true cell means, up to a constant).
numpy and the standard library only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from round10_challenge_recruit_context import ELO, FOLDS, MATCHES, design, fit, welfare

HERE = Path(__file__).resolve().parent
LAM_A = 5000.0


def load_by_match():
    rows, match_idx, side_games = [], [], 0
    for j, m in enumerate(MATCHES):
        d = json.loads((HERE / f"round10_challenge_recruit_context_{m}.json").read_text())
        side_games += d["side_games"]
        for key, (n, s, ss) in d["cells"].items():
            fac, u0, u1, enemy, tb, f = key.split("|")
            rows.append((fac, u0, u1, enemy, int(tb), int(f), n, s, ss))
            match_idx.append(j)
    return rows, np.asarray(match_idx), side_games


def main() -> None:
    rows, mj, sg = load_by_match()
    n = np.asarray([r[6] for r in rows], dtype=float)
    s = np.asarray([r[7] for r in rows])
    ss = np.asarray([r[8] for r in rows])
    folds = np.asarray([r[5] for r in rows])
    cA = design(rows, lambda r, u: f"{r[0]}|{u}")
    cB = design(rows, lambda r, u: f"{r[0]}|{u}|{r[3]}")
    print("progressive loss of B's ridge (A at 5,000), matches in game order; lower is better")
    best = None
    lams = (500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0, 50000.0, 1e9)
    for lam in lams:
        loss = 0.0
        for j in range(1, len(MATCHES)):
            train, test = mj < j, mj == j
            pA = fit(cA[0], cA[1], cA[2], n, s, train, LAM_A)
            pB = fit(cB[0], cB[1], cB[2], n, s, train, lam, prior=pA)
            loss += float(np.sum(-2 * s[test] * pB[test] + n[test] * pB[test] ** 2))
        label = "inf (B = A)" if lam >= 1e9 else f"{lam:,.0f}"
        print(f"  ridge {label:>12s} labels: {loss:+.3f}")
        if best is None or loss < best[1]:
            best = (lam, loss)
    lam = best[0]
    print(f"chosen ridge for B: {lam:,.0f} labels")
    wa, wb, d = np.zeros(2), np.zeros(2), np.zeros(FOLDS)
    for f in range(FOLDS):
        train, test = folds != f, folds == f
        pA = fit(cA[0], cA[1], cA[2], n, s, train, LAM_A)
        pB = fit(cB[0], cB[1], cB[2], n, s, train, lam, prior=pA)
        a, b = np.asarray(welfare(pA, s, ss, test)), np.asarray(welfare(pB, s, ss, test))
        wa += a
        wb += b
        d[f] = (b[0] - a[0]) / sg * ELO
    for name, w in (("A (R)", wa), ("B (matchup)", wb)):
        print(f"  {name:12s} cross-fitted {w[0] / sg * ELO:+.1f} ({math.sqrt(w[1]) / sg * ELO:.1f}) Elo per side-game")
    print(f"  B - A {d.sum():+.2f} Elo, SE from the {FOLDS} per-fold differences {math.sqrt(FOLDS) * d.std(ddof=1):.2f}")
    sw = 0
    pA_all = fit(cA[0], cA[1], cA[2], n, s, np.ones(len(rows), bool), LAM_A)
    pB_all = fit(cB[0], cB[1], cB[2], n, s, np.ones(len(rows), bool), lam, prior=pA_all)
    sw = float(np.sum(n[pB_all > 0])) / sg
    swa = float(np.sum(n[pA_all > 0])) / sg
    print(f"  switches per side-game, rules fitted on all 60,000 games: A {swa:.2f}, B {sw:.2f}")


if __name__ == "__main__":
    main()
