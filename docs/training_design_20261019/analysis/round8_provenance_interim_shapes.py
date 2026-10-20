"""Round 8, provenance stance: is the interim stop's error bounded without the Gaussian shape?

The design stops round 1 after batch 3 iff student 3's match estimate J3 is below 0, and counts the
stop as a null that exhausts the band. It states the false-stop probability at an adoptable round
as <= 0.041 where the sqrt(N) ceiling's premise (mean >= 0 per cell) holds, and up to 0.15 at
mu = -0.0018 under the Gaussian model.

Model: the design's own (round5_synthesis_power.per_decision), with only the shape of the in-span
residual changed. In one cell, the advantage is a = c + r, c the typed mean (estimated precisely,
unpenalised), r the in-span residual with mean 0 and variance v. The ridge at held-out lambda is
the best linear shrinkage, kappa = v / (v + eta); the student switches iff c + kappa (r + e) > 0,
e ~ N(0, eta), eta_k = s2 d / (k n). Gain per band decision g_k = E[(c + r) Phi((r + c / kappa) / sqrt(eta))];
J_k = 347.4 x 76.1 x g_k Elo; P(stop) = Phi(-J3 / 1.79).

For each shape, over a grid of c and sd(r), among cases where student 8 is adoptable at 80% power
(J8 >= 5.1 Elo), report the largest P(stop), separately for c >= 0 and c < 0. A single cell is the
worst case of a mixture of cells. numpy and the standard library only.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round5_synthesis_power as S  # noqa: E402

ELO = 800 / math.log(10)
BAND = 76.1
S2, D, N_BATCH = 3.67, 772, 2 * 76.1 * 100_000
MATCH_SE = 1.79
ADOPTABLE = 5.1
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(80)    # nodes for E over a standard normal
GH_W = GH_W / GH_W.sum()


def Phi(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x) / math.sqrt(2.0)))


def eta(k):
    return S2 * D / (k * N_BATCH)


def shape_points(name, sd):
    """Support points and weights of r, mean 0, sd sd."""
    if name == "gaussian":
        return sd * GH_X, GH_W
    if name == "laplace":
        x = np.linspace(-12, 12, 2401)
        w = np.exp(-np.abs(x) * math.sqrt(2.0))
        w /= w.sum()
        return sd * x / math.sqrt(np.sum(w * x * x)), w
    q = float(name.split("_")[1])                       # two-point: +A with prob q
    a = sd * math.sqrt((1 - q) / q)
    return np.array([a, -a * q / (1 - q)]), np.array([q, 1 - q])


def gain(c, sd, name, k):
    r, w = shape_points(name, sd)
    v, e = sd * sd, eta(k)
    kappa = v / (v + e)
    p_switch = Phi((r + c / kappa) / math.sqrt(e))
    return ELO * BAND * float(np.sum(w * (c + r) * p_switch))


def main():
    print(f"eta_3 {eta(3):.3e}, eta_8 {eta(8):.3e} (s2 {S2}, d {D}, {N_BATCH:.3e} labels a batch)")
    # cross-check the Gaussian against the design's own function
    for c, sd in ((0.0, 0.002), (-0.0018, 0.0035)):
        mine = gain(c, sd, "gaussian", 8)
        theirs = ELO * BAND * S.per_decision(sd * sd, eta(8), c)[0]
        print(f"  check c {c:+.4f} sd {sd}: this script J8 {mine:.2f} Elo, round5_synthesis_power.per_decision {theirs:.2f}")
    shapes = ["gaussian", "laplace"] + [f"two_{q}" for q in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95)]
    cs = np.round(np.arange(-0.008, 0.0041, 0.0002), 6) + 0.0
    sds = np.exp(np.linspace(math.log(0.0005), math.log(0.02), 45))
    print("\nlargest P(stop) among cases with J8 >= 5.1 Elo (student 3's J3 and J8 at that case)")
    overall = {"c>=0": (0, None), "c<0": (0, None)}
    for name in shapes:
        best = {"c>=0": (0.0, None), "c<0": (0.0, None)}
        for c in cs:
            for sd in sds:
                j8 = gain(c, sd, name, 8)
                if j8 < ADOPTABLE:
                    continue
                j3 = gain(c, sd, name, 3)
                ps = float(Phi(-j3 / MATCH_SE))
                key = "c>=0" if c >= 0 else "c<0"
                if ps > best[key][0]:
                    best[key] = (ps, (c, sd, j3, j8))
        line = []
        for key in ("c>=0", "c<0"):
            ps, case = best[key]
            if case is None:
                line.append(f"{key}: none adoptable")
            else:
                c, sd, j3, j8 = case
                line.append(f"{key}: P(stop) {ps:.3f} at c {c:+.4f} sd {sd:.4f} (J3 {j3:+.1f}, J8 {j8:.1f})")
                if ps > overall[key][0]:
                    overall[key] = (ps, (name, c, sd, j3, j8))
        print(f"  {name:9s} " + "; ".join(line))
    print("\nworst over shapes:")
    for key, (ps, case) in overall.items():
        print(f"  {key}: P(stop) {ps:.3f}, shape {case[0]}, c {case[1]:+.4f}, sd {case[2]:.4f}, J3 {case[3]:+.1f}, J8 {case[4]:.1f}")
    print(f"  design statement: <= {float(Phi(-math.sqrt(3 / 8) * ADOPTABLE / MATCH_SE)):.3f} under the ceiling's premise; "
          f"0.15 Gaussian at mu -0.0018")


if __name__ == "__main__":
    main()
