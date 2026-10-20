"""Round 8, provenance stance: do the design's decision rules keep their stated properties when
only the shape of the near-tie advantage changes?

Two rules carry statements derived in round 5's Gaussian model (round6_simplify_rule.out):
  - the sign rule (switch where the student's prediction f > 0) is within 0.6 Elo of the best
    threshold;
  - fallback E (the same student switching only above the median of f among its switches) keeps
    0.80-0.87 of the first-order gain.
(The interim stop is covered by round8_provenance_interim_shapes.py.)
Model: the design's own, as in round8_provenance_interim_shapes.py (eta, shapes and constants
imported): in one cell the runner-up's advantage is a = c + r, c the typed mean (unpenalised,
known), r the in-span residual (mean 0, variance v, shape varied); the ridge at held-out lambda
predicts f = c + kappa (r + e), kappa = v / (v + eta), e ~ N(0, eta), eta = eta_8 (student 8).
For every shape and every (c, sd) on the grid with J8 >= 5.1 Elo (adoptable at 80% power), report:
  1. the sign rule's first-order gain over the best threshold's (tau chosen with knowledge of the
     shape: an oracle), worst case and the Elo it leaves;
  2. the best rule of the lambda family (threshold 0, the shrinkage chosen for first-order gain
     instead of squared loss) against the same oracle, worst case;
  3. E's first-order share of the sign rule's gain, smallest case;
  4. the better of the sign rule and E against the oracle, worst case. With e Gaussian (log-concave),
     the location family has a monotone likelihood ratio, so E[a | f] is nondecreasing in f for
     any distribution of r, and the top half of any switch set by f keeps at least half its sum:
     the floor is 0.5 for every shape.
numpy and the standard library only; the normal CDF by Abramowitz and Stegun 7.1.26.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round8_provenance_interim_shapes as I  # noqa: E402

ADOPTABLE = 5.1
ELO_BAND = I.ELO * I.BAND


def phi_cdf(x):
    """Standard normal CDF (absolute error below 1.5e-7), vectorised."""
    x = np.asarray(x, float)
    z = np.abs(x) / math.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return 0.5 * (1.0 + np.sign(x) * (1.0 - poly * np.exp(-z * z)))


def shape(name, sd):
    if name == "laplace":                                   # coarser support than the import's
        x = np.linspace(-8, 8, 401)
        w = np.exp(-np.abs(x) * math.sqrt(2.0))
        w /= w.sum()
        return sd * x / math.sqrt(np.sum(w * x * x)), w
    return I.shape_points(name, sd)


def evaluate(c, sd, name):
    r, w = shape(name, sd)
    v, e = sd * sd, I.eta(8)
    kappa = v / (v + e)
    se = math.sqrt(e)
    aw = w * (c + r)
    span = 8 * kappa * math.sqrt(v + e) + abs(c)
    taus = np.concatenate([np.linspace(-span, span, 241), [0.0]])
    # P(f > tau | r) for every tau (rows) and support point (columns)
    p = phi_cdf((r[None, :] - (taus[:, None] - c) / kappa) / se)
    gains = p @ aw
    g0, n0 = gains[-1], float(p[-1] @ w)
    best = gains.max()
    # the lambda family: shrinkage kappa' in place of kappa, threshold 0 (switch iff r + e > -c / kappa')
    kps = kappa * np.exp(np.linspace(math.log(0.02), math.log(50.0), 161))
    pk = phi_cdf((r[None, :] + c / kps[:, None]) / se)
    best_lambda = (pk @ aw).max()
    # E: the threshold on f at which the switch count is half the sign rule's (bisection)
    lo, hi = 0.0, span + 10 * kappa * se
    for _ in range(45):
        mid = (lo + hi) / 2
        if float(phi_cdf((r - (mid - c) / kappa) / se) @ w) > n0 / 2:
            lo = mid
        else:
            hi = mid
    g_e = float(phi_cdf((r - (lo - c) / kappa) / se) @ aw)
    return g0, best, best_lambda, g_e


def main():
    shapes = ["gaussian", "laplace"] + [f"two_{q}" for q in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95)]
    cs = np.round(np.arange(-0.008, 0.0041, 0.0004), 6) + 0.0
    sds = np.exp(np.linspace(math.log(0.0005), math.log(0.02), 24))
    print(f"student 8: eta {I.eta(8):.3e}; adoptable: J8 >= {ADOPTABLE} Elo; Elo per unit mean gain per band decision {ELO_BAND:.0f}")
    print("per shape, over adoptable cases: sign rule at the squared-loss lambda / oracle threshold, worst case "
          "(c, sd; J8 sign rule, J8 oracle); best over the lambda family / oracle, worst; E share min, max")
    worst_all, e_min_all, lam_all, two_all = (1.0, None), 1.0, 1.0, 1.0
    for name in shapes:
        worst, lam_worst, two_worst, e_min, e_max, n = (1.0, None), 1.0, 1.0, 1.0, 0.0, 0
        for c in cs:
            for sd in sds:
                g0, best, best_lambda, g_e = evaluate(c, sd, name)
                if ELO_BAND * g0 < ADOPTABLE:
                    continue
                n += 1
                if g0 / best < worst[0]:
                    worst = (g0 / best, (c, sd, ELO_BAND * g0, ELO_BAND * best))
                lam_worst = min(lam_worst, best_lambda / best)
                two_worst = min(two_worst, max(g0, g_e) / best)
                e_min, e_max = min(e_min, g_e / g0), max(e_max, g_e / g0)
        if n == 0:
            print(f"  {name:9s} no adoptable case")
            continue
        ratio, (c, sd, j, jb) = worst
        print(f"  {name:9s} {n:4d} cases: sign {ratio:.3f} (c {c:+.4f}, sd {sd:.4f}; {j:5.1f} vs {jb:5.1f} Elo); "
              f"lambda family {lam_worst:.3f}; better of sign and E {two_worst:.3f}; E {e_min:.3f}-{e_max:.3f}")
        if ratio < worst_all[0]:
            worst_all = (ratio, name, j, jb)
        e_min_all, lam_all, two_all = min(e_min_all, e_min), min(lam_all, lam_worst), min(two_all, two_worst)
    print(f"\nworst sign-rule ratio over shapes: {worst_all[0]:.3f} ({worst_all[1]}; {worst_all[2]:.1f} against "
          f"{worst_all[3]:.1f} Elo, {worst_all[3] - worst_all[2]:.1f} Elo left to an oracle threshold)")
    print(f"worst ratio of the best lambda (threshold 0, shrinkage chosen for gain) to the oracle: {lam_all:.3f}")
    print(f"worst ratio of the better of the sign rule and E to the oracle: {two_all:.3f}")
    print(f"smallest E share over shapes: {e_min_all:.3f} (shape-free floor 0.5)")


if __name__ == "__main__":
    main()
