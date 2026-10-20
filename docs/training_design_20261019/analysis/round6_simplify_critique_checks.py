"""Round 6, critique of the simplify stance: three checks from its own model and inputs.

1. The early stop after batch 4 at mu = -0.0018, near and above the match-only floor
   (round6_simplify_power.out prints it only for sd <= 0.003). Reuses round5_synthesis_power.py
   with the paired PURE match's kept share 0.847, as round6_simplify_power.py does.

2. P-coin's ridge reading, w_hat . u / 0.01, predicted "1 +- 0.10" (round6_simplify.md 2.7).
   Model: round6_simplify_rule.py's Gaussian model on batch 1 alone (15.22 M labels, sigma^2
   3.91, d 776, isotropic features with per-coordinate variance s, so Var(u . x) = s). Real
   advantages w'x with spread sd (w ~ N(0, sd^2 / (d s) I)) plus the plant 0.01 u. Lambda chosen
   as student 1's is: progressive loss of one half of batch 1 on the other, on the rule script's
   grid widened to 1e0..1e12. Reported: the ridge reading's mean and 2.5-97.5 percentiles, and
   the OLS (lambda 0) reading's SE, for s = 1 (the variance round 5 assumed for u . x,
   round5_simplify_critique_checks.py) and s = 1/776 (trace 1, round6_simplify_rule.py).

3. The slope reading's relative SE, sqrt(sigma^2 / (N s)) / 0.01, as a function of s: the plant
   can fail only if 0 lies outside its 2-SE band.
numpy and the standard library only.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round5_synthesis_power as S  # noqa: E402

D, SIGMA2, DELTA = 776, 3.91, 0.01
N = 2 * 76.1 * 100_000
LAMBDAS = np.geomspace(1e0, 1e12, 97)
REPS = 400
RNG = np.random.default_rng(20261020)


def early_stop():
    S.MATCH_SE_Z = math.sqrt(0.847 / 12_800)
    S.TWO_MATCH_SE_ELO = 2 * S.P.ELO_PER_Z * S.MATCH_SE_Z
    inputs = S.P.read_inputs()
    print("1. early stop after batch 4, mu -0.0018 (paired PURE match, 2 SE "
          f"{S.TWO_MATCH_SE_ELO:.1f} Elo)")
    for d_eff in (200, 776):
        for sd in (0.003, 0.0035, 0.0039, 0.0045, 0.005, 0.006):
            ps, m, se = S.early_stop_probability(inputs, sd, -0.0018, d_eff, 0.25)
            g = S.round_readout(inputs, sd, -0.0018, d_eff, 8, 0.25)["gain_elo"]
            print(f"  d_eff {d_eff:3d} sd {sd:.4f}: P(stop) {ps:.2f}; final gain had it run {g:5.1f} Elo")


def plant_reading(s, sd):
    u = RNG.normal(size=D)
    u /= np.linalg.norm(u)
    w_real = RNG.normal(0, sd / math.sqrt(D * s), D) if sd > 0 else np.zeros(D)
    w = w_real + DELTA * u
    g_half = N / 2 * s
    h1 = g_half * w + math.sqrt(SIGMA2 * g_half) * RNG.normal(size=D)
    h2 = g_half * w + math.sqrt(SIGMA2 * g_half) * RNG.normal(size=D)
    loss = [-2 * (h1 / (g_half + lam)) @ h2 + g_half * np.sum((h1 / (g_half + lam)) ** 2)
            for lam in LAMBDAS]
    lam = LAMBDAS[int(np.argmin(loss))]
    r = h1 + h2
    ridge = (r / (N * s + lam)) @ u / DELTA
    ols = (r / (N * s)) @ u / DELTA
    return ridge, ols


def plants():
    print("\n2. P-coin ridge reading w_hat . u / 0.01 on batch 1 (prediction in the stance: 1 +- 0.10)")
    for s, name in ((1.0, "Var(u.x) = 1"), (1.0 / D, "trace 1")):
        for sd in (0.0, 0.002, 0.004):
            rid, ols = zip(*(plant_reading(s, sd) for _ in range(REPS)))
            rid, ols = np.array(rid), np.array(ols)
            print(f"  {name:13s} real sd {sd:.3f}: ridge mean {rid.mean():.3f} "
                  f"(2.5-97.5%: {np.percentile(rid, 2.5):.3f} to {np.percentile(rid, 97.5):.3f}); "
                  f"OLS mean {ols.mean():.3f}, sd {ols.std():.3f}")


def slope_se():
    print("\n3. slope reading's relative SE by Var(u.x)")
    for s in (1.0, 0.1, 0.01, 1.0 / D):
        rel = math.sqrt(SIGMA2 / (N * s)) / DELTA
        print(f"  Var(u.x) {s:.5f}: relative SE {rel:.3f}; 2-SE band 1 +- {2 * rel:.2f}; "
              f"contains 0: {2 * rel >= 1}")


if __name__ == "__main__":
    early_stop()
    plants()
    slope_se()
