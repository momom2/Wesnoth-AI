"""Round 10 synthesis: L's lines at the capped seal, the step search's lines, and the path's cost.

Inputs (all measured, named where used):
  - capped Bd's exact coin-world value +24.1 +- 3.0 Elo and capped R's +14.9 +- 3.0
    (round10_synthesis_checks.out; leave-one-match-out rules);
  - prices of the measured host (round9_simplify_labels.out, as round9_synthesis_checks.py carries them);
  - the forecast SEs 4.88 (switched share 0.5) and 3.41 (0.25) (round9_simplify_budget.out);
  - S4's 4.8 box-hours (prereg 20261011 "Measured", as round9_synthesis_checks.py carries it).
numpy and the standard library only; under a second.
Run: python round10_synthesis_budget.py > round10_synthesis_budget.out
"""
from __future__ import annotations

import math

import numpy as np

ELO_SE_1 = 347.4          # Elo per sqrt(decisive game), one standard error: 347.4 / sqrt(N)
POOL_RATE = 22_209        # pool (one-network) games an hour, round9_simplify_labels.out
ALLIN_RATE = 20_254       # coin games an hour all-in, same
TWO_RATE = 6_521          # two-network decisive games an hour, same
PRICE = 0.577             # $ an hour, the measured host
COIN_GAME = 2.849e-5      # $ a coin game all-in, same
SETUP_H = 7 / 60 + 0.25   # 7 min measured bring-up + 15 min staging [C], to be measured at box A
LEFT = 82.34              # $85 - $2.66 (seed section 7), if the window counted against $85


def se(n):
    return ELO_SE_1 / math.sqrt(n)


def section1():
    print("1. L at 37,800 decisive games (SE %.2f), sealed at the capped rule's exact coin-world value" % se(37_800))
    for name, v, vse in (("capped Bd", 24.1, 3.0), ("capped R", 14.9, 3.0), ("full R (round 9's seal)", 21.3, 3.1)):
        s = se(37_800)
        half = 2 * math.hypot(vse, s)
        null = 2 * s
        bound = (null + 2 * s) / v
        p3 = 0.5 * math.erfc(-(0.3 * v - null) / s / math.sqrt(2))
        p5 = 0.5 * math.erfc(-(0.5 * v - null) / s / math.sqrt(2))
        z_null = (v - null) / math.hypot(vse, s)
        print(f"   {name}: {v:+.1f} +- {vse:.1f}; miss half-width {half:.1f}; null line {null:.2f}; the seal sits "
              f"{z_null:.1f} SE above it; a null bounds the transfer ratio below {bound:.2f}; "
              f"P(lower bound > 0) {p3:.2f} at ratio 0.3, {p5:.2f} at 0.5")
    s = se(37_800)
    print(f"   T against L (both 37,800): difference SE {math.hypot(s, s):.2f}, line {2 * math.hypot(s, s):.1f}")


def section2():
    print("\n2. The step search at S2: three screens of 12,800 one-network games, the best confirmed on fresh seeds")
    s12, s37 = se(12_800), se(37_800)
    print(f"   screen SE {s12:.2f}; two screens' difference SE {math.hypot(s12, s12):.2f}; confirmation SE {s37:.2f}")
    for share, fse in ((0.5, 4.88), (0.25, 3.41)):
        print(f"   forecast SE {fse:.2f} (switched share {share}): a screen's miss line {2 * math.hypot(fse, s12):.1f}; "
              f"the confirmation's {2 * math.hypot(fse, s37):.1f}")
    rng = np.random.default_rng(20261021)
    n = 400_000
    for truth in ((0, 0, 0), (12, 8, 5), (2, 6, 5), (-10, -2, 3), (20, 15, 10)):
        reads = np.array(truth) + s12 * rng.standard_normal((n, 3))
        pick = reads.argmax(1)
        conf = np.array(truth)[pick] + s37 * rng.standard_normal(n)
        adopt = conf > 2 * s37
        best = int(np.argmax(truth))
        regret = (max(truth) - np.array(truth)[pick]).mean()
        print(f"   true values of the three screens {truth}: P(adopt) {adopt.mean():.3f}; P(best picked) {np.mean(pick == best):.3f};"
              f" mean regret {regret:.2f} Elo")


def prices(o, k=1.0):
    pool = lambda n: n / POOL_RATE * PRICE
    two = lambda n: n / TWO_RATE * PRICE
    coin = lambda n, dumped=True: n / ALLIN_RATE * PRICE * (1 + (o if dumped else 0.0)) * k
    setup = SETUP_H * PRICE
    return pool, two, coin, setup


def path(o, k=1.0, with_t=True, third="reset"):
    """Box A: L, T iff ruling (h), the smoke (3,000 dumped own2 coin games). C iff T adopts a rule into pi_0.
    A round: 8 batches of 100,000 coin games (batch 1's first match of 10,000 undumped, for o), three screens
    of 12,800 one-network games, a confirmation of 37,800, a claim of 12,800 two-network games, a setup.
    The third slot: the reset (4.8 box-hours, N against pi_k two-network) then round 3 with its confirmation
    against pi_k on two networks (12,800); or a third refit of the frozen network (as round 2)."""
    pool, two, coin, setup = prices(o, k)
    box_a = pool(37_800) * (2 if with_t else 1) + coin(3_000) + setup
    c = two(12_800) if with_t else 0.0
    gen = coin(790_000) + coin(10_000, dumped=False)
    search = 3 * pool(12_800) + pool(37_800)
    r1 = gen + search + two(12_800) + setup
    r2 = r1
    if third == "reset":
        r3 = 4.8 * PRICE + two(12_800) + gen + 3 * pool(12_800) + two(12_800) + two(12_800) + setup
    else:
        r3 = r1
    null1 = gen + search + setup
    return dict(box_a=box_a, c=c, r1=r1, r2=r2, r3=r3, null1=null1)


def solve(fn, lo, hi):
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fn(mid) else (lo, mid)
    return lo


def section3():
    print("\n3. The path at $0.577 an hour, by the dump overhead o")
    for with_t in (True, False):
        lab = "ruling (h) yes: L, T, C" if with_t else "ruling (h) no: L only"
        for o in (0.0, 0.1, 0.2):
            p = path(o, with_t=with_t)
            p3 = path(o, with_t=with_t, third="refit")
            two = p["box_a"] + p["c"] + p["r1"] + p["r2"]
            print(f"   {lab}, o {o:.2f}: box A ${p['box_a']:.2f}; C ${p['c']:.2f}; a round ${p['r1']:.2f}; "
                  f"through round 2 ${two:.2f} (left ${LEFT - two:.2f}); with reset and round 3 ${two + p['r3']:.2f}; "
                  f"with a third refit ${two + p3['r3']:.2f}; a null round 1 after box A ${p['box_a'] + p['null1']:.2f}")
        for left in (LEFT, 85.0):
            fit2 = lambda o: (lambda p: p["box_a"] + p["c"] + p["r1"] + p["r2"])(path(o, with_t=with_t)) <= left
            o2 = solve(fit2, 0.0, 3.0)
            for third in ("reset", "refit"):
                fit3 = lambda o: sum(v for kk, v in path(o, with_t=with_t, third=third).items() if kk != "null1") <= left
                o3 = solve(fit3, -1.0, 3.0)
                kmax = solve(lambda k: sum(v for kk, v in path(0.0, k, with_t, third).items() if kk != "null1") <= left,
                             0.3, 1.5)
                print(f"   {lab}, ${left:.2f}: two rounds iff o <= {o2:.2f}; three with the {third} iff o <= {o3:.3f}, "
                      f"or at o 0 iff a coin game costs <= {kmax:.3f} x the measured (${COIN_GAME * kmax:.3e})")
    pool, two, coin, setup = prices(0.0)
    print(f"   unit costs: screen ${pool(12_800):.3f}; confirmation ${pool(37_800):.3f}; claim ${two(12_800):.3f}; "
          f"batch of 100,000 coin games ${coin(100_000):.3f}; setup ${setup:.3f}; the step search ${3 * pool(12_800) + pool(37_800):.2f} "
          f"against round 9's S2 plus E reserve ${2 * pool(37_800):.2f}")


if __name__ == "__main__":
    section1()
    section2()
    section3()
