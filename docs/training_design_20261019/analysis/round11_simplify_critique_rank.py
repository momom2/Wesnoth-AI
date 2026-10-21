"""Round 11, critique of the simplify stance: is the z at which |G|^2 is
resolved a function of n / B_simple alone, as S0's gate "|G|^2 at z >= 2"
and its comparison with "the coin analog's strength" assume?

For the unbiased estimator U = sum_{a != b} v_a . v_b / (n (n - 1)), under
G = 0, Var(U) = 2 tr(Sigma^2) / (n (n - 1)), so

    z = |G|^2 / SD(U) ~= (n / B_simple) * sqrt(r_eff / 2),
    r_eff = tr(Sigma)^2 / tr(Sigma^2)  (the noise's effective rank).

The z therefore grows with the effective rank of the noise, which depends on
the parametrisation: 43 columns here, 15.6 M parameters for the network.

Reads the same rows as round11_simplify_noise.py (batch 1's recruit-pair
rows, 60,000 decisive coin games) and reports, for its R43 gradient: tr Sigma,
tr Sigma^2, r_eff, the null SD of U from the formula against the jackknife
SE that script printed, and the z the formula gives. Then, derived only, the
z of S0's gate at 40,000 games for B_simple from 1e4 to 1e6 and r_eff from
1 to 1e4. Numpy and the standard library only.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from round11_simplify_noise import load, per_game_vectors  # noqa: E402


def main():
    cols, n_games = load()
    vecs, _, _, _ = per_game_vectors(cols, n_games)
    V = vecs["R43"]
    n = V.shape[0]
    mean = V.mean(0)
    C = (V - mean).T @ (V - mean) / (n - 1)
    tr = float(np.trace(C))
    tr2 = float(np.sum(C * C))
    r_eff = tr * tr / tr2
    s = V.sum(0)
    g2 = (float(s @ s) - float((V * V).sum())) / (n * (n - 1))
    sd_null = math.sqrt(2.0 * tr2 / (n * (n - 1)))
    sd_signal = math.sqrt(4.0 * float(mean @ C @ mean) / n)
    print(f"R43: n {n}; tr Sigma {tr:.4e}; tr Sigma^2 {tr2:.4e}; r_eff {r_eff:.2f}")
    print(f"  |G|^2 {g2:.4e}; B_simple {tr / g2:,.0f}")
    print(f"  SD(U) under G = 0 {sd_null:.4e}; signal term {sd_signal:.4e}; "
          f"combined {math.hypot(sd_null, sd_signal):.4e} (jackknife SE printed by the noise script 1.6015e-03)")
    print(f"  z by formula {g2 / math.hypot(sd_null, sd_signal):.2f} (jackknife 4.13); "
          f"(n / B) sqrt(r_eff / 2) = {n / (tr / g2) * math.sqrt(r_eff / 2):.2f}")
    eig = np.sort(np.linalg.eigvalsh(C))[::-1]
    print("  top eigenvalue shares of Sigma: " + ", ".join(f"{x / tr:.3f}" for x in eig[:5]))
    print(f"  z against a sign-flip (G = 0) null: {g2 / sd_null:.2f}; against the full SE: "
          f"{g2 / math.hypot(sd_null, sd_signal):.2f}")
    print("\nDerived: z of S0's gate at n = 40,000 under G = 0's SD, z = (n / B) sqrt(r_eff / 2)")
    for b in (1e4, 1e5, 1e6):
        row = "; ".join(f"r_eff {r:>6,.0f}: z {40_000 / b * math.sqrt(r / 2):6.2f}" for r in (1, 100, 1e4))
        print(f"  B_simple {b:>9,.0f}: {row}")


if __name__ == "__main__":
    main()
