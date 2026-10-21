"""Round 10, challenge stance: costs, standard errors and decision lines of the
label-directed, play-sized step regimen (LDPS), from measured prices and the
round-10 recruit readings. numpy and the standard library only.

Measured inputs (named where used):
- prices, round9_simplify_labels.out [prices]: pool (one-network) game
  $2.598e-5; coin game all-in $2.849e-5; two-network decisive game $8.848e-5;
- the match SE: 347.4 / sqrt(N) Elo on raw z of decisive games (Var z <= 1),
  as round9_synthesis_checks.out uses (12,800 -> 3.07, 37,800 -> 1.79);
- B's in-band first-order curve and its reversed direction,
  round10_challenge_shift_curve.out; B's welfare +28.5 +- 3.1,
  round10_challenge_recruit_context_ridge.out;
- B's learning curve, round10_challenge_recruit_halves.out;
- the current design's round and stage costs, round9_synthesis_checks.out s6;
- the student-1 forecast SE 6.8-9.8 Elo on a 100,000-game batch,
  round9_synthesis_checks.out s3.
"""
from __future__ import annotations

import math

import numpy as np

ELO = 800 / math.log(10)
POOL, COIN, TWO = 2.598e-5, 2.849e-5, 8.848e-5
BUDGET = 82.34


def se(n: int) -> float:
    return ELO / math.sqrt(n)


def main() -> None:
    s12, s38 = se(12800), se(37800)
    screen, confirm, claim = 12800 * POOL, 37800 * POOL, 12800 * TWO
    print("1. unit prices and SEs")
    print(f"  screen 12,800 one-network games ${screen:.3f}, SE {s12:.2f} Elo; confirmation 37,800 ${confirm:.3f}, SE {s38:.2f};"
          f" claim 12,800 two-network ${claim:.2f}")
    for n in (30000, 60000, 100000):
        print(f"  coin batch {n:,} games ${n * COIN:.3f}")

    print("\n2. information per dollar about one fixed step's play value")
    for name, s_, c in (("one-network screen, 12,800 games", s12, screen),
                        ("student-1 welfare forecast, 100,000-game batch, SE 6.8", 6.8, 100000 * COIN),
                        ("student-1 welfare forecast, 100,000-game batch, SE 9.8", 9.8, 100000 * COIN)):
        print(f"  {name:58s} 1/(SE^2 $) = {1 / (s_ * s_ * c):.4f}")
    print(f"  ratio screen / forecast: {(1 / (s12**2 * screen)) / (1 / (6.8**2 * 100000 * COIN)):.0f}x to"
          f" {(1 / (s12**2 * screen)) / (1 / (9.8**2 * 100000 * COIN)):.0f}x")

    print("\n3. box A'' (stage 1)")
    stages = [("own1: B band-only, 37,800 (L's analog)", confirm),
              ("own1: the S0(a) attack hook through the shift path, 12,800 (known answer)", screen),
              ("own2: three screens (B band-only, shift eta 8, shift eta 40), 12,800 each", 3 * screen),
              ("own2: confirmation of the best screen, 37,800 fresh seeds", confirm),
              ("claim: own2 + step against own1, two networks", claim)]
    tot = 0.0
    for name, c in stages:
        tot += c
        print(f"  {name:76s} ${c:.2f}")
    print(f"  total ${tot:.2f} (the current design's box A + C: $2.18 + $1.13 = $3.31)")
    box = tot

    print("\n4. lines of stage 1")
    F, F_se = 28.5, 3.1
    print(f"  own1 B band-only: sealed F {F:+.1f} +- {F_se}; miss half-width 2 sqrt(F_se^2 + SE^2) = {2 * math.hypot(F_se, s38):.1f};"
          f" null line 2 SE = {2 * s38:.2f}, bounding the ratio rho below {2 * s38 / F + 2 * F_se / F * 0:.2f}"
          f" (point) [with F's own SE: {(2 * s38) / (F - 2 * F_se):.2f}]")
    hook, hook_se = -41.3, 4.4
    half = 2 * math.hypot(hook_se, s12)
    print(f"  known answer for the shift path: f = 1[attack], eta 1 on one seat of own1 is S0(a)'s hook,"
          f" measured {hook:+.1f} +- {hook_se} by the coin-hook path; window {hook - half:+.1f} to {hook + half:+.1f}"
          f" (half-width 2 sqrt({hook_se}^2 + {s12:.2f}^2) = {half:.1f}); a seat swap reads about +41, an unapplied shift 0")
    print(f"  T's analog, own2 band-only screen against own1's band-only reading: SE {math.hypot(s12, s38):.2f}, line {2 * math.hypot(s12, s38):.1f}")
    w_rev, w_rev_se, w_inf = -11.7, 1.7, 28.5
    ratio = w_rev / w_inf
    r_se = abs(ratio) * math.hypot(w_rev_se / w_rev, F_se / w_inf)
    line_se = math.sqrt(s12**2 + (ratio * s38) ** 2 + (F * r_se) ** 2)
    print(f"  (rejected alternative) reversed shift eta 3: J_rev = {ratio:+.3f} (+- {r_se:.3f}) x J_fwd; 3-SE line {3 * line_se:.1f} Elo:"
          f" a sign error is caught only if rho >= {3 * line_se / (2 * abs(w_rev)):.2f}")
    print(f"  beyond-band reading J(eta 40) - J(band-only), two screens: SE {math.sqrt(2) * s12:.2f};"
          f" on confirmations it would be {math.sqrt(2) * s38:.2f}")
    print("  in-band first-order parts (own1 labels): band-only +28.5 (3.1), eta 8 +24.3 (2.4), eta 40 +28.4 (3.0)")

    print("\n5. the line search's selection, Monte Carlo (3 screens SE 3.07, best confirmed at SE 1.79, adopt iff lower 2-SE bound > 0)")
    rng = np.random.default_rng(20261021)
    reps = 200000
    for truth in ((0, 0, 0), (20, 20, 20), (20, 25, 15), (20, 15, 25), (20, 24, 30), (5, 5, 5), (3, -5, -15)):
        t = np.asarray(truth, dtype=float)
        reads = t + rng.normal(0, s12, (reps, 3))
        pick = reads.argmax(1)
        conf = t[pick] + rng.normal(0, s38, reps)
        adopt = conf - 2 * s38 > 0
        regret = t.max() - t[pick]
        print(f"  true J {truth}: P(adopt) {adopt.mean():.3f}; P(best picked) {np.mean(t[pick] == t.max()):.3f};"
              f" mean regret {regret.mean():.2f} Elo; screen winner's bias {np.mean(reads.max(1) - t[pick]):+.2f}")

    print("\n6. iterations after stage 1, and the current design's steps for the same money")
    left = BUDGET - box
    it_rec = 30000 * COIN + 2 * screen + confirm + claim
    it_three = 30000 * COIN + 6 * screen + confirm + claim
    it_head = 100000 * COIN + 8 * screen + confirm + claim
    print(f"  left after box A'': ${left:.2f}")
    print(f"  one iteration, recruit class, 30,000-game batch, 2 screens: ${it_rec:.2f}; fits {int(left // it_rec)}")
    print(f"  one iteration, three classes, 30,000-game batch, 6 screens: ${it_three:.2f}; fits {int(left // it_three)}")
    print(f"  one iteration, head dump over all classes, 100,000 games (o 0), 8 screens: ${it_head:.2f}; fits {int(left // it_head)}")
    print(f"  with a reset ($3.90) after every third adoption: recruit-class iterations fit {int(left // (it_rec + 3.90 / 3))}")
    print("  the current design: box A + C $3.31, round 1 $26.18, round 2 $25.12 (o 0), per adoption about $25-26")

    print("\n7. the learning curve (round10_challenge_recruit_halves.out): B out of sample by games fitted")
    curve = {6000: 17.7, 12000: 20.6, 30000: 24.7, 54000: 27.2}
    for g, w in curve.items():
        print(f"  {g:6,d} games ${g * COIN:.2f}: {w:+.1f} Elo = {w / curve[54000]:.2f} of the 54,000-game rule")


if __name__ == "__main__":
    main()
