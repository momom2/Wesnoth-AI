"""Round 10, critique of the simplify stance: the claim check's two tails.

round9_synthesis_checks.py section 5 counts a claim miss as |D - A| > 2 sqrt(SE_D^2 + 1.79^2), two-sided;
round10_simplify.md stops the chain on any miss. Split the false-miss rate into its tails (D below A:
the gain did not reach own1; D above A: it reached own1 with more than its adoption reading), with the
same model: A ~ N(step, 1.79^2) kept iff A > 2 x 1.79; D ~ N(step, SE_D^2). numpy only; a second.
"""
import math

import numpy as np

rng = np.random.default_rng(20261021)
SE_A = 1.79
N = 2_000_000
for label, se_d in (("claim_(k-1) a 12,800-game claim", math.sqrt(2) * 3.07),
                    ("claim_0 = S0(b)", math.sqrt(3.07 ** 2 + 4.35 ** 2))):
    line = 2 * math.sqrt(se_d ** 2 + SE_A ** 2)
    print(f"{label}: SE of D {se_d:.2f}, line {line:.1f} Elo")
    for step in (2.8, 4.0, 6.0, 10.0, 21.3):
        a = rng.normal(step, SE_A, N)
        keep = a > 2 * SE_A
        d = rng.normal(step, se_d, N)[keep]
        a = a[keep]
        low, high = np.mean(d - a < -line), np.mean(d - a > line)
        print(f"   true step {step:5.1f}: adopted {keep.mean():.3f}; false miss below {low:.3f}, above {high:.3f}, "
              f"two-sided {low + high:.3f}")
