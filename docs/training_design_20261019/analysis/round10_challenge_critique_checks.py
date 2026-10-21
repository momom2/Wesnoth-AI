"""Round 10, critique of the challenge stance: checks of its recruit readings.

Reads the per-match sums written by round10_challenge_recruit_context.py
extract (labels T = s * omega * (z - beta.S), 20 folds by SHA-256 of the game
name). numpy and the standard library only.

1. Reproduce A (43 values), B (matchup values shrunk toward A) and B - A at
   ridge 5,000, cross-fitted over 20 folds.
2. B - A by turn bin, paired on the same folds: where the matchup gain sits.
   (Against humans, about half the ladder sides choose Random, and the enemy
   faction is then unknown until the first sighting, observation parity
   2026-09-26, "Information a player does not have" item 1.)
3. The learning curve with a paired SE: for each of the 10 fold-groups g,
   the welfare read on g by the rule fitted on the 9 other groups, and by the
   rules fitted on 5 groups that exclude g (mean over those rotations);
   the difference summed over groups, SE = sqrt(10) * sd of the per-group
   differences.
4. Information per dollar of a 12,800-game screen against the label batch's
   own estimate of B (SE 3.1 at 60,000 coin games), the like-for-like
   counterpart of the stance's 42-87x.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from round10_challenge_recruit_context import ELO, FOLDS, MATCHES, design, fit, welfare

HERE = Path(__file__).resolve().parent
LAM = 5000.0
TURN_NAMES = ("turn 1", "turn 2", "turns 3-4", "turns 5-8", "turns 9+")


def load():
    rows, sg = [], 0
    for m in MATCHES:
        d = json.loads((HERE / f"round10_challenge_recruit_context_{m}.json").read_text())
        sg += d["side_games"]
        for key, (n, s, ss) in d["cells"].items():
            fac, u0, u1, enemy, tb, f = key.split("|")
            rows.append((fac, u0, u1, enemy, int(tb), int(f), n, s, ss))
    return rows, sg


def main() -> None:
    rows, sg = load()
    n = np.asarray([r[6] for r in rows], dtype=float)
    s = np.asarray([r[7] for r in rows])
    ss = np.asarray([r[8] for r in rows])
    folds = np.asarray([r[5] for r in rows])
    tb = np.asarray([r[4] for r in rows])
    cA = design(rows, lambda r, u: f"{r[0]}|{u}")
    cB = design(rows, lambda r, u: f"{r[0]}|{u}|{r[3]}")

    print("1. reproduction at ridge 5,000, 20 folds")
    per = np.zeros((FOLDS, 2, 2))
    by_turn = np.zeros((FOLDS, 5, 2))
    for f in range(FOLDS):
        train, test = folds != f, folds == f
        pA = fit(cA[0], cA[1], cA[2], n, s, train, LAM)
        pB = fit(cB[0], cB[1], cB[2], n, s, train, LAM, prior=pA)
        per[f, 0] = welfare(pA, s, ss, test)
        per[f, 1] = welfare(pB, s, ss, test)
        for b in range(5):
            t = test & (tb == b)
            by_turn[f, b, 0] = welfare(pA, s, ss, t)[0]
            by_turn[f, b, 1] = welfare(pB, s, ss, t)[0]
    for k, name in enumerate(("A", "B")):
        tot, tot2 = per[:, k].sum(0)
        print(f"  {name}: {tot / sg * ELO:+.2f} ({math.sqrt(tot2) / sg * ELO:.2f}) Elo per side-game")
    d = (per[:, 1, 0] - per[:, 0, 0]) / sg * ELO
    print(f"  B - A {d.sum():+.2f}, SE {math.sqrt(FOLDS) * d.std(ddof=1):.2f}")

    print("\n2. B - A by turn bin, paired over folds (Elo per side-game)")
    for b, name in enumerate(TURN_NAMES):
        db = (by_turn[:, b, 1] - by_turn[:, b, 0]) / sg * ELO
        print(f"  {name:10s} B {by_turn[:, b, 1].sum() / sg * ELO:+6.2f}  A {by_turn[:, b, 0].sum() / sg * ELO:+6.2f}"
              f"  B - A {db.sum():+5.2f} ({math.sqrt(FOLDS) * db.std(ddof=1):.2f})")

    print("\n3. learning curve, paired per fold-group (rules fitted on 9 groups against 5, 2 and 1 groups excluding g)")
    group = folds // 2
    for model in ("A", "B"):
        w9 = np.zeros(10)
        wk = {5: np.zeros(10), 2: np.zeros(10), 1: np.zeros(10)}
        cache = {}

        def rule(fit_groups):
            key = tuple(sorted(fit_groups))
            if key not in cache:
                tr = np.isin(group, list(key))
                pA = fit(cA[0], cA[1], cA[2], n, s, tr, LAM)
                cache[key] = pA if model == "A" else fit(cB[0], cB[1], cB[2], n, s, tr, LAM, prior=pA)
            return cache[key]

        for g in range(10):
            test = group == g
            w9[g] = welfare(rule([h for h in range(10) if h != g]), s, ss, test)[0]
            for k in wk:
                vals = []
                for rot in range(10):
                    fg = {(rot + i) % 10 for i in range(k)}
                    if g not in fg:
                        vals.append(welfare(rule(fg), s, ss, test)[0])
                wk[k][g] = np.mean(vals)
        scale = ELO / sg  # summed over the 10 groups = every side-game read once
        for k in (5, 2, 1):
            diff = (w9 - wk[k]) * scale
            print(f"  {model}: 54,000 games {w9.sum() * scale:+.2f}; {k * 6000:,} games {wk[k].sum() * scale:+.2f};"
                  f" ratio {wk[k].sum() / w9.sum():.3f}; difference {diff.sum():+.2f} (paired SE {math.sqrt(10) * diff.std(ddof=1):.2f})")

    print("\n4. information per dollar, 1/(SE^2 $)")
    screen = 1 / (ELO / math.sqrt(12800)) ** 2 / (12800 * 2.598e-5)
    labels_b = 1 / 3.1 ** 2 / (60000 * 2.849e-5)
    print(f"  screen 12,800 games {screen:.4f}; B's label estimate, 60,000 coin games, SE 3.1: {labels_b:.4f}; ratio {screen / labels_b:.1f}x")


if __name__ == "__main__":
    main()
