"""Round 9, simplify stance: one fixed switch rule in place of the tested threshold and fallback E.

The current design carries three switch rules: the sign rule f > 0, a threshold tau chosen on held-out
labels behind a 2-SE pre-test (with its own plant), and fallback E (the top half of the switches by f,
played only after a forecast miss, with its own forecast and match). This script compares two FIXED
rules, each chosen before any label is read, against the oracle threshold, in round 8's own model
(round8_provenance_rules.evaluate: a = c + r, f = c + kappa (r + e), e ~ N(0, eta_8)), with its inputs
moved to section 7's measurements:
  - the typed mean c restricted to the measured pooled mean +0.0003 +- 0.0007 (readout_b_offline.txt):
    c in {-0.0011, +0.0003, +0.0017}, its 2-SE range and point;
  - the label variance s2 = 4.01 (round9_simplify_labels.out, frozen coefficients) and 77.6 band
    decisions per side-game (the census), which set eta_8.
Rules: sign (f > 0); top half (the threshold at which the switch count is half the sign rule's; E's rule,
here played as the only rule). For each shape, over spreads sd where the oracle's first-order gain is at
least 5.1 Elo, the worst and median share of the oracle each rule keeps, and the Elo it leaves.
The top half's curvature property (a loss quadratic in the switch count falls to a quarter) is
round 6's derivation and is not simulated here.
numpy and the standard library only.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round8_provenance_interim_shapes as I  # noqa: E402

I.S2, I.BAND = 4.01, 77.6
I.N_BATCH = 2 * I.BAND * 100_000
import round8_provenance_rules as RU  # noqa: E402

ELO_BAND = I.ELO * I.BAND
CS = (-0.0011, 0.0003, 0.0017)
SDS = np.exp(np.linspace(math.log(0.0005), math.log(0.02), 24))
SHAPES = ["gaussian", "laplace"] + [f"two_{q}" for q in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95)]


def main():
    print(f"eta_8 {I.eta(8):.3e} (s2 {I.S2}, band {I.BAND}); Elo per unit mean gain per band decision {ELO_BAND:.0f}")
    print("shape     | sign/oracle worst (Elo left), median | top half/oracle worst (Elo left), median | cases")
    worst_sign, worst_top = (1.0, ""), (1.0, "")
    for name in SHAPES:
        rs, rt, ls, lt = [], [], [], []
        for c in CS:
            for sd in SDS:
                g0, best, _, g_e = RU.evaluate(c, sd, name)
                if ELO_BAND * best < 5.1:
                    continue
                rs.append(g0 / best)
                rt.append(g_e / best)
                ls.append(ELO_BAND * (best - g0))
                lt.append(ELO_BAND * (best - g_e))
        if not rs:
            print(f"  {name:9s} no case with an oracle gain of 5.1 Elo")
            continue
        i, j = int(np.argmin(rs)), int(np.argmin(rt))
        print(f"  {name:9s} | {rs[i]:.3f} ({ls[i]:5.1f}), {np.median(rs):.3f} | {rt[j]:.3f} ({lt[j]:5.1f}), "
              f"{np.median(rt):.3f} | {len(rs)}")
        if rs[i] < worst_sign[0]:
            worst_sign = (rs[i], name)
        if rt[j] < worst_top[0]:
            worst_top = (rt[j], name)
    print(f"\nworst share of the oracle over shapes: sign {worst_sign[0]:.3f} ({worst_sign[1]}), "
          f"top half {worst_top[0]:.3f} ({worst_top[1]})")


if __name__ == "__main__":
    main()
