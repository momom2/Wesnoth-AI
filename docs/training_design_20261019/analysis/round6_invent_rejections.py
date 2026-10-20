"""Round 6 (invent stance): two ways of buying coin-label information that this stance rejects,
with the derivations that reject them. Standard library only.

1. Branching continuations. Play a prefix (a fraction f of a game's cost, holding a fraction f of
   its coins) once, then K independent continuations. A prefix coin's label uses the mean of the K
   results, so its variance is r (f + (1 - f) / K) when the unexplained variance r accrues in
   proportion to cost; a suffix coin's label keeps r. Labels of different continuations are
   uncorrelated (their coins are independent and mean zero). Information per unit cost, relative
   to independent games at equal cost:
       ratio(f, K) = [f / (f + (1 - f)/K) + K (1 - f)] / [f + K (1 - f)].
2. Horizon-truncated labels with a rectifier (prediction-powered inference, Angelopoulos et al.
   2023). Cheap labels T_H read a value at a horizon costing a fraction h of a game; a subset of
   games is played on and gives T_F; the estimator mean_all(T_H) + mean_full(T_F - T_H) is
   unbiased. With Var(T_H) = a s2, Var(T_F - T_H) = (1 - a) s2 (a calibrated value), the optimal
   allocation's variance at budget B is (sqrt(a h) + sqrt((1 - a)(1 - h)))^2 s2 / B against s2 / B
   for full games alone, when a > h; when a <= h the constraint N_F <= N_T binds and every game
   is played on (factor 1). The gain is 1 minus that factor.
"""
import math


def branching_ratio(f, k):
    return (f / (f + (1 - f) / k) + k * (1 - f)) / (f + k * (1 - f))


def ppi_factor(a, h):
    """Optimal allocation N_T ~ sqrt(a / h), N_F ~ sqrt((1 - a) / (1 - h)); a game played on must
    first be truncated, so N_F <= N_T, which binds when a <= h: then every game is played on and
    the factor is a + (1 - a) = 1."""
    if a <= h:
        return 1.0
    return (math.sqrt(a * h) + math.sqrt((1 - a) * (1 - h))) ** 2


def main():
    best = max((branching_ratio(f / 100, k), f / 100, k) for f in range(1, 100) for k in range(2, 33))
    print(f"1. branching: best information ratio {best[0]:.3f} at prefix share {best[1]:.2f}, K {best[2]}")
    for f, k in ((0.5, 2), (0.5, 4), (0.3, 2), (0.8, 4)):
        print(f"   f {f:.1f} K {k}: ratio {branching_ratio(f, k):.3f}")
    print("2. horizon labels with a rectifier: variance factor against full games (1 = no gain)")
    for a in (0.5, 0.6, 0.7, 0.8):
        row = ", ".join(f"h {h:.1f}: {ppi_factor(a, h):.3f}" for h in (0.3, 0.5, 0.7))
        print(f"   pre-horizon share of variance a {a:.1f} -> {row}")
    print("   a gain needs a > h: the result's variance front-loaded relative to cost, and a calibrated value at h")


if __name__ == "__main__":
    main()
