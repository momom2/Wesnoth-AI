"""Round 10, simplify stance: the numbers the simplified design rests on, from existing records.

Sections (cited as round10_simplify_checks.out section n):
1. R is the head with a unit one-hot feature block: the cross-fitted first-order welfare F of the
   ridge on e(a1) - e(a0) at every ridge of the round-9 grid, so the cost of one ridge shared with the
   token block can be read. Same per-cell fold sums and the same fit as round9_synthesis_checks.py.
2. The tested threshold on the one measured structure: the cross-fitted welfare curve W(tau) of R's
   fold rules, against tau = 0, with the SE of each difference.
3. The typed block's per-batch label counts and the ridge's shrinkage of a typed constant, from the
   per-side-game counts of round9_provenance_cells.out section 4.
4. The simplified path's budget at the measured prices (rates of round9_simplify_labels.out), by the
   dump's overhead o: box A (L and the smoke), rounds 1 to 3 by one code (8 batches, adoption, claim),
   E reserves, no reset; where two and three rounds fit; a null round 1; the current design's path
   (round9_synthesis_checks.out section 6) beside it.

numpy and the standard library only; no network, no simulator; a few seconds.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import round9_synthesis_checks as r9  # noqa: E402  (load, design, fit_pred: the round-9 fit, reused)

ELO = 800 / math.log(10)
LEFT = 82.34
POOL_RATE, ALLIN_RATE, TWO_RATE, PRICE = 22_209, 20_254, 6_521, 0.577
COIN_GAME = 2.849e-5
SETUP = (7 / 60 + 0.25) * PRICE
GRID = (500.0, 1_000.0, 2_000.0, 5_000.0, 10_000.0, 20_000.0, 50_000.0)


def fold_rules(X, s_all, n_all, lam):
    """Per fold f: the predicted cell values of the ridge fitted on the other folds."""
    preds = []
    for f in range(r9.FOLDS):
        so, no = s_all.sum(1) - s_all[:, f], n_all.sum(1) - n_all[:, f]
        preds.append(np.where(no > 0, r9.fit_pred(X, so, no, lam), -np.inf))
    return preds


def section1(X, s_all, ss_all, n_all, sg):
    print("1. R as the head with a unit one-hot block: cross-fitted F by ridge (labels T, 20 folds)")
    for lam in GRID:
        preds = fold_rules(X, s_all, n_all, lam)
        tot = var = sw = 0.0
        for f, p in enumerate(preds):
            ch = p > 0
            tot += s_all[ch, f].sum()
            var += ss_all[ch, f].sum()
            sw += n_all[ch, f].sum()
        print(f"   ridge {lam:>7.0f} labels: F {tot / sg * ELO:+.1f} +- {math.sqrt(var) / sg * ELO:.1f} Elo; "
              f"switches {sw / sg:.2f} per side-game")


def section2(X, s_all, ss_all, n_all, sg, lam=5_000.0):
    print("\n2. R's welfare curve over the threshold tau (cross-fitted, ridge 5,000), against tau = 0")
    preds = fold_rules(X, s_all, n_all, lam)
    taus = (-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02, 0.03, 0.05)
    base = None
    rows = []
    for tau in taus:
        tot = sw = 0.0
        masks = []
        for f, p in enumerate(preds):
            ch = p > tau
            masks.append(ch)
            tot += s_all[ch, f].sum()
            sw += n_all[ch, f].sum()
        rows.append((tau, tot, sw, masks))
        if tau == 0.0:
            base = masks
    for tau, tot, sw, masks in rows:
        dvar = sum(ss_all[masks[f] ^ base[f], f].sum() for f in range(r9.FOLDS))
        dtot = tot - sum(s_all[base[f], f].sum() for f in range(r9.FOLDS))
        print(f"   tau {tau:+.3f}: W {tot / sg * ELO:+.1f} Elo, switches {sw / sg:.2f}; "
              f"W - W(0) {dtot / sg * ELO:+.1f} +- {math.sqrt(dvar) / sg * ELO:.1f}")


def section3():
    print("\n3. Typed block: labels per 100,000-game batch and the shrinkage n / (n + lambda) of a constant")
    per_side_game = {  # round9_provenance_cells.out section 4 (own1, batch 1)
        "move>move": 30.91 + 15.99, "recruit>recruit": 4.29 + 2.10, "attack>move": 4.07,
        "end_turn>move": 3.79, "move>attack": 3.77, "move>end_turn": 2.88, "recruit>move": 2.11,
        "move>recruit": 1.95, "attack>attack": 1.85 + 1.78 + 1.07, "recruit>end_turn": 0.20,
        "recruit>attack": 0.20, "attack>recruit": 0.19, "end_turn>recruit": 0.19,
        "end_turn>attack": 0.16, "attack>end_turn": 0.15,
    }
    for lam in (5_000.0, 50_000.0):
        shr = {k: v * 200_000 / (v * 200_000 + lam) for k, v in per_side_game.items()}
        lo = min(shr, key=shr.get)
        print(f"   lambda {lam:,.0f}: smallest type {lo} n {per_side_game[lo] * 200_000:,.0f}, factor {shr[lo]:.3f}; "
              f"median factor {np.median(list(shr.values())):.3f}")


def costs(o, k=1.0):
    pool = lambda n: n / POOL_RATE * PRICE
    two = lambda n: n / TWO_RATE * PRICE
    coin = lambda n: n / ALLIN_RATE * PRICE * (1 + o) * k
    box_a = pool(37_800) + coin(2_000) + pool(1_000) + SETUP
    rnd = coin(800_000) + pool(37_800) + two(12_800) + SETUP
    e = pool(37_800)
    return dict(box_a=box_a, r1=rnd, e1=e, r2=rnd, e2=e, r3=rnd, e3=e,
                null1=box_a + coin(800_000) + pool(37_800) + SETUP)


def upto(p, rounds):
    keys = ["box_a"] + [f"{x}{i}" for i in range(1, rounds + 1) for x in ("r", "e")]
    return sum(p[k] for k in keys)


def solve(fn, lo, hi):
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fn(mid) else (lo, mid)
    return lo


def section4():
    print("\n4. The simplified path at $0.577 an hour, by the dump's overhead o (E reserves counted)")
    for o in (0.0, 0.05, 0.10, 0.20):
        p = costs(o)
        print(f"   o {o:.2f}: box A ${p['box_a']:.2f}; a round ${p['r1']:.2f}; E ${p['e1']:.2f}; "
              f"through round 1 ${upto(p, 1):.2f}; round 2 ${upto(p, 2):.2f}; round 3 ${upto(p, 3):.2f}; "
              f"a null round 1 after box A ${p['null1']:.2f}")
    for left in (LEFT, 85.0):
        for r in (2, 3):
            o_max = solve(lambda o: upto(costs(o), r) <= left, 0.0, 5.0)
            k_max = solve(lambda k: upto(costs(0.0, k), r) <= left, 0.3, 3.0)
            print(f"   with ${left:.2f}: {r} rounds fit iff o <= {o_max:.3f}, or at o 0 iff a coin game costs "
                  f"<= {k_max:.3f} x the measured (${COIN_GAME * k_max:.3e})")
    print("   current design (round9_synthesis_checks.out section 6), o 0: through round 2 $56.57; whole path "
          "$86.87 (box A $2.18, C $1.13, H1 inside round 1, S4 $3.90, round 3 adopted on two networks)")
    p = costs(0.0)
    print(f"   removed at o 0: T $0.98, C $1.13, H1 $0.98, one setup ${SETUP:.2f}, S4 $3.90; "
          f"round 3 one-network adoption saves ${(12_800 / TWO_RATE - 37_800 / POOL_RATE) * PRICE:.2f} "
          f"and its E reserve ${(12_800 / TWO_RATE - 37_800 / POOL_RATE) * PRICE:.2f}; "
          f"whole simplified path ${upto(p, 3):.2f}")


def main():
    cells, per_match, side_games = r9.load()
    X, names = r9.design(cells)
    s_all = sum(m["s_T"] for m in per_match)
    ss_all = sum(m["ss_T"] for m in per_match)
    n_all = sum(m["n"] for m in per_match)
    sg = sum(side_games)
    print(f"batch 1: {len(cells)} recruit cells, {len(names)} unit values, {sg:,} side-games\n")
    section1(X, s_all, ss_all, n_all, sg)
    section2(X, s_all, ss_all, n_all, sg)
    section3()
    section4()


if __name__ == "__main__":
    main()
