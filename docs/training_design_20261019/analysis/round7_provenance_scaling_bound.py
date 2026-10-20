"""Round 7, provenance stance: what more coin labels on the same frozen span can buy, without
assuming the shape of the near-tie advantages.

Claim. A student fitted on N labels predicts f = a + e at a band decision, a the in-span advantage
of the runner-up, e Gaussian with variance eta proportional to 1/N (OLS on a fixed span; random
design, Hsu, Kakade and Zhang 2012). The sign rule switches where f > 0 and gains, per decision,
    g(eta) = E[a Phi(a / sqrt(eta))].
With s = sqrt(eta) and t = a / s, d(s g)/ds = E[a w(t)], w(t) = Phi(t) - t phi(t), and w is
increasing (w' = t^2 phi >= 0) with values in (0, 1). If E[a] >= 0, then
E[a w] = E[a] E[w] + Cov(a, w(a / s)) >= 0 (Chebyshev's association inequality: a and w(a / s) are
both increasing in a). So s g(s) is nondecreasing, and for N' >= N:
    g(N') <= sqrt(N' / N) g(N)            (any shape of a with mean >= 0)
and, read the other way, g(N) >= sqrt(N / N') g(N'). The same holds when the noise scale varies
across decisions independently of a (apply the argument at each scale). With E[a] < 0 it can fail:
the value of a noisy signal is then convex at low precision (Radner and Stiglitz 1984).

This script checks the claim numerically on several shapes (each with sd 0.002 and 0.003), with
homoscedastic and heteroscedastic noise, then shows the failure at a negative mean under both the
sign rule and the Gaussian posterior-mean rule of round5_synthesis_power.py, and finally states the
consequences at the design's numbers:
  - eta_k = sigma^2 d / (k x labels per batch): sigma^2 3.67, d 776, 2 x 76.1 x 100,000 labels a
    batch (current_design.md section 2; round6_synthesis_power.out);
  - interim match after batch 3 (SE 1.79 Elo, stop iff the estimate is below 0): the stop
    probability when the batch-8 student would be adoptable at 80% power (5.1 Elo);
  - a second round on the same band (16 batches pooled): the ceiling on its first-order increment,
    and the chance that pi_2 against pi_1 (SE 1.79) is adopted at that ceiling.
numpy and the standard library only.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round5_synthesis_power as S  # noqa: E402

ELO = 800 / math.log(10)
SIGMA2, D, N_BAND, BATCH = 3.67, 776, 76.1, 100_000
LABELS = 2 * N_BAND * BATCH
SE_MATCH = 1.79
RNG = np.random.default_rng(20261020)


def Phi(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x) / math.sqrt(2.0)))


def eta(k):
    return SIGMA2 * D / (k * LABELS)


def shapes(sd, n=400_000):
    """Mean-zero draws of a with standard deviation sd."""
    z = RNG.standard_normal(n)
    lap = RNG.laplace(size=n)
    t3 = RNG.standard_t(3, size=n)
    out = {"gaussian": z, "laplace": lap, "student_t3": t3,
           "uniform": RNG.uniform(-1, 1, n),
           "skewed (exp - 1)": RNG.exponential(size=n) - 1.0,
           "skewed (1 - exp)": 1.0 - RNG.exponential(size=n)}
    for eps in (0.2, 0.04):
        x = np.where(RNG.random(n) < eps, np.where(RNG.random(n) < 0.5, 1.0, -1.0), 0.0)
        out[f"sparse two-point eps {eps}"] = x
    return {k: (v - v.mean()) / v.std() * sd for k, v in out.items()}


def gain(a, e2, scale=None):
    s = np.sqrt(e2 if scale is None else e2 * scale)
    return float(np.mean(a * Phi(a / s)))


def main():
    print("eta per batch count: " + ", ".join(f"k {k}: {eta(k):.2e} (sd {math.sqrt(eta(k)):.4f})" for k in (3, 8, 16)))
    print(f"bounds: g8/g3 <= sqrt(8/3) = {math.sqrt(8 / 3):.3f}; g16/g8 <= sqrt(2) = {math.sqrt(2):.3f}\n")
    print("1. mean-zero shapes, sign rule (gains in Elo per side-game = 347.4 x 76.1 x g)")
    for sd in (0.002, 0.003):
        for name, a in shapes(sd).items():
            g3, g8, g16 = (gain(a, eta(k)) for k in (3, 8, 16))
            scale = np.exp(0.5 * RNG.standard_normal(a.size))       # heteroscedastic, independent of a
            h3, h8, h16 = (gain(a, eta(k), scale) for k in (3, 8, 16))
            print(f"  sd {sd} {name:26s} g8 {ELO * N_BAND * g8:5.1f} Elo; g8/g3 {g8 / g3:.3f}, g16/g8 {g16 / g8:.3f}"
                  f" | heteroscedastic g8/g3 {h8 / h3:.3f}, g16/g8 {h16 / h8:.3f}")
    print("\n2. negative mean: the bound fails (sign rule, and the Gaussian posterior-mean rule of the power model)")
    for mu in (-0.0005, -0.0018):
        for sd in (0.002, 0.003, 0.004):
            a = RNG.standard_normal(400_000) * sd + mu
            g3, g8, g16 = (gain(a, eta(k)) for k in (3, 8, 16))
            p = [S.per_decision(sd * sd, eta(k), mu)[0] for k in (3, 8, 16)]
            print(f"  mu {mu:+.4f} sd {sd}: sign rule g8 {ELO * N_BAND * g8:+6.1f} Elo, g8/g3 "
                  f"{g8 / g3 if g3 > 0 else float('nan'):6.2f}, g16/g8 {g16 / g8 if g8 > 0 else float('nan'):5.2f} | "
                  f"posterior-mean rule g8 {ELO * N_BAND * p[1]:5.1f} Elo, g8/g3 "
                  f"{p[1] / p[0] if p[0] > 0 else float('inf'):6.2f}, g16/g8 {p[2] / p[1] if p[1] > 0 else float('inf'):5.2f}")
    print("\n3. consequences at the design's numbers (mean >= 0 regime; first-order gains)")
    adopt80 = 2.84 * SE_MATCH
    g3_min = math.sqrt(3 / 8) * adopt80
    print(f"  interim: if student 8 gains >= {adopt80:.1f} Elo (adoptable at 80% power), student 3 gains >= "
          f"{g3_min:.2f}, so P(stop) <= {float(Phi(-g3_min / SE_MATCH)):.3f}")
    for g8 in (5.0, 8.0, 10.0, 15.0, 20.0, 30.0):
        inc = (math.sqrt(2) - 1) * g8
        p_adopt = float(Phi(inc / SE_MATCH - 2))
        print(f"  round 1 first-order gain {g8:4.1f} Elo -> same-band round 2 increment <= {inc:4.1f} Elo; "
              f"pi_2 vs pi_1 adopted with P <= {p_adopt:.2f} at that ceiling")
    print("  Gaussian model's own increment (mu 0, d 776), for comparison:")
    for sd in (0.0015, 0.002, 0.003, 0.004):
        g8, g16 = (ELO * N_BAND * S.per_decision(sd * sd, eta(k), 0.0)[0] for k in (8, 16))
        print(f"    sd {sd}: g8 {g8:5.1f}, g16 {g16:5.1f}, increment {g16 - g8:4.1f} Elo ({(g16 - g8) / g8:.2f} of g8)")
    print(f"  the same-band increment reaches the adoption line (2 SE = {2 * SE_MATCH:.1f} Elo) only if g8 >= "
          f"{2 * SE_MATCH / (math.sqrt(2) - 1):.1f}, and 80% power ({adopt80:.1f}) only if g8 >= {adopt80 / (math.sqrt(2) - 1):.1f} Elo")
    print("\n4. batch 1's reading of the mean label (which decides whether section 3 applies), per margin bin")
    band = json.loads((Path(__file__).resolve().parent / "round5_simplify_band.json").read_text())
    scale = 224.6 / 271.7          # own1's decisions per side-game over parity3's readings (decision units)
    dens = band["density_per_nat_per_side_game"]
    for key, width in (("[0.0,0.1)", 0.1), ("[0.1,0.2)", 0.1), ("[0.2,0.3)", 0.1), ("[0.3,0.4)", 0.05)):
        per_sg = dens[key] * width * scale
        n = per_sg * 2 * BATCH
        print(f"  bin {key[:9]} (width {width}): {per_sg:5.1f} band decisions per side-game, {n / 1e6:.2f} M labels, "
              f"SE of the mean label {math.sqrt(SIGMA2 / n):.5f}")
    print(f"  whole band: SE {math.sqrt(SIGMA2 / LABELS):.5f}; prediction noise of student 8 sqrt(eta_8) {math.sqrt(eta(8)):.4f}")


if __name__ == "__main__":
    main()
