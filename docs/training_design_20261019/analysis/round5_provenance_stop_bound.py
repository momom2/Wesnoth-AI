"""Round 5 (provenance stance): what a learned stop rule can gain in the window the record measured.

Inputs, measured (round5_provenance_stop_acts.out): own1 at offset -2.5 against -1.5 plays
N = 19.4 more acts per side-game, and the mean advantage of one such act is m = -0.0018
(95% interval -0.0061 to +0.0017). Inputs, measured (round5_provenance_match_variance.out):
a corrected match keeps r = 0.747 of the result's variance.

Model, stated: within the window the advantage A of continuing is normal with mean m and sd tau
across decisions; a rule acting on a prediction P = E[A | h] continues iff P > 0, where P is normal
with mean m and sd sqrt(q) * tau (q: the share of the heterogeneity the observation predicts).
Its gain over always stopping in the window is N * E[P * 1(P > 0)]
  = N * (s * phi(m / s) + m * Phi(m / s)),  s = sqrt(q) * tau,
in result units per side-game; Elo = 347.4 * gain near an even match.
The table gives the gain, and the tau a stop-only match must find for its gain to reach 2 SE.
Standard library only.
"""
import math

N, M = 19.4, -0.0018
ELO_PER_Z = 400 / math.log(10) * 2
R = 0.747


def phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def gain(tau, q, m=M):
    s = math.sqrt(q) * tau
    return N * (s * phi(m / s) + m * Phi(m / s))


def main():
    print("Elo gain of the stop rule in the [1.5, 2.5) window, by tau (rows) and q (columns), m = -0.0018")
    qs = (0.1, 0.3, 1.0)
    print("tau      " + "".join(f"q={q:<8}" for q in qs))
    for tau in (0.002, 0.005, 0.01, 0.02, 0.04):
        print(f"{tau:<8} " + "".join(f"{ELO_PER_Z * gain(tau, q):<10.1f}" for q in qs))
    for n in (3200, 12800):
        se = ELO_PER_Z * math.sqrt(R / n)
        target = 2 * se / ELO_PER_Z
        for q in qs:
            lo, hi = 1e-5, 1.0
            for _ in range(100):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if gain(mid, q) < target else (lo, mid)
            print(f"corrected match of {n} decisive games: SE {se:.1f} Elo; a gain of 2 SE needs "
                  f"sqrt(q)*tau >= {math.sqrt(q) * hi:.4f} (tau >= {hi:.4f} at q = {q})")
    for m in (-0.0061, 0.0017):
        print(f"at the interval's end m = {m}: gain at sqrt(q)*tau = 0.005 is {ELO_PER_Z * gain(0.005, 1.0, m):.1f} Elo")


if __name__ == "__main__":
    main()
