"""Round 7, critique of the provenance stance: four checks of its numbers.

1. Price stationarity. Completion lines of the 120- and 160-job arms in
   training/metrics/generation_speed_r5_20261015/: the rate in three equal sub-windows of the
   steady window (completions n/5 to 4n/5) and the mean "in flight" count there, to test the
   claim that the window is steady and that all slots are busy.

2. The ceiling's premise against batch 1's regime reading. Same-kind pair means follow
   m(G) = b * G (continuity at a tie, b < 0 when the logit gap is informative), the band's margin
   density from round5_simplify_band.json (bins of 0.1 nat up to w = 0.35), the per-bin SEs of
   round7_provenance_scaling_bound.out section 4 (0.00085, 0.00091, 0.00098, 0.00145) and the
   band SE 0.00049. For each b: the chance that batch 1 flags at least one bin (or the band)
   at -2 SE, and the gains of the sign rule (OLS, f = a + e) and of the Gaussian posterior-mean
   rule (round5_synthesis_power.per_decision), with a in-span Gaussian of sd v^0.5 around m in
   each bin; ratios g8/g3 and g16/g8 against the ceiling sqrt(8/3) and sqrt(2).

3. The posterior-mean rule at mean >= 0, the rule the design actually uses (ridge with held-out
   lambda): does it respect the ceiling?

4. D2 at the ceiling: P(adopt pi_2 against pi_1) when round 1's first-order gain is below 8.6.

Closed forms: for a ~ N(m, v), f = a + e, e ~ N(0, eta):
  sign rule   E[a 1{f > 0}] = m Phi(m / s) + (v / s) phi(m / s), s = sqrt(v + eta);
  posterior-mean rule: round5_synthesis_power.per_decision(v, eta, m).
numpy and the standard library only.
"""
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import round5_synthesis_power as S  # noqa: E402

RUN = HERE.parents[2] / "training/metrics/generation_speed_r5_20261015"
ELO = 800 / math.log(10)
SIGMA2, D, N_BAND, BATCH = 3.67, 776, 76.1, 100_000
LABELS = 2 * N_BAND * BATCH
SE_MATCH = 1.79
BIN_SE = (0.00085, 0.00091, 0.00098, 0.00145)
BAND_SE = 0.00049


def Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def eta(k):
    return SIGMA2 * D / (k * LABELS)


def stationarity():
    print("1. steady-window stationarity")
    for arm in ("ref_p120x6", "priors_p120x6", "ref_p160x6"):
        times, flight = {}, {}
        for line in (RUN / f"arm_{arm}.log").read_text().splitlines():
            m = re.match(r"(\d\d):(\d\d):(\d\d) INFO game .* done in .*results=(\d+)/\d+.* (\d+) in flight", line)
            if m:
                h, mi, s, k, f = (int(x) for x in m.groups())
                times.setdefault(k, 3600 * h + 60 * mi + s)
                flight[k] = f
        n = max(times)
        lo, hi = n // 5, 4 * n // 5
        cuts = [lo + (hi - lo) * j // 3 for j in range(4)]
        rates = [(cuts[j + 1] - cuts[j]) / max(times[cuts[j + 1]] - times[cuts[j]], 1) * 3600 for j in range(3)]
        fl = [flight[k] for k in range(lo, hi + 1) if k in flight]
        slots = int(re.search(r"p(\d+)x", arm).group(1))
        print(f"  {arm}: sub-window rates " + ", ".join(f"{r:,.0f}" for r in rates)
              + f" games/h; in flight mean {sum(fl) / len(fl):.1f} (min {min(fl)}, max {max(fl)}) of {slots}")


def band_bins():
    dens = json.loads((HERE / "round5_simplify_band.json").read_text())["density_per_nat_per_side_game"]
    bins = [("[0.0,0.1)", 0.05, 0.1), ("[0.1,0.2)", 0.15, 0.1), ("[0.2,0.3)", 0.25, 0.1), ("[0.3,0.4)", 0.325, 0.05)]
    weights = [dens[k] * w for k, _, w in bins]
    total = sum(weights)
    return [(c, wt / total) for (_, c, _), wt in zip(bins, weights)]


def sign_gain(m, v, e):
    s = math.sqrt(v + e)
    return m * Phi(m / s) + (v / s) * phi(m / s)


def band_gain(rule, b, v, k):
    g = 0.0
    for centre, weight in band_bins():
        m = b * centre
        g += weight * (sign_gain(m, v, eta(k)) if rule == "sign" else S.per_decision(v, eta(k), m)[0])
    return g


def regime():
    print("\n2. same-kind means m(G) = b G: batch 1's flag rate and the ceiling")
    bins = band_bins()
    for b in (-0.001, -0.002, -0.004, -0.008):
        means = [b * c for c, _ in bins]
        band_mean = sum(w * m for (_, w), m in zip(bins, means))
        p_none = 1.0
        for m, se in zip(means, BIN_SE):
            p_none *= 1.0 - Phi(-2.0 - m / se)
        p_band = Phi(-2.0 - band_mean / BAND_SE)
        print(f"  b {b:+.3f}/nat: bin means " + ", ".join(f"{m:+.5f}" for m in means)
              + f"; band mean {band_mean:+.5f}; P(some bin flagged) {1 - p_none:.2f}, P(band flagged) {p_band:.2f}")
        for sd in (0.0015, 0.002, 0.003):
            v = sd * sd
            row = []
            for rule in ("sign", "posterior"):
                g3, g8, g16 = (band_gain(rule, b, v, k) for k in (3, 8, 16))
                r83 = g8 / g3 if g3 > 0 else float("nan")
                r168 = g16 / g8 if g8 > 0 else float("nan")
                row.append(f"{rule:9s} g8 {ELO * N_BAND * g8:+6.1f} Elo, g8/g3 {r83:5.2f}, g16/g8 {r168:5.2f}")
            print(f"    sd {sd}: " + " | ".join(row))
    print(f"  ceiling: g8/g3 <= {math.sqrt(8 / 3):.3f}, g16/g8 <= {math.sqrt(2):.3f}")


def posterior_nonnegative():
    print("\n3. posterior-mean rule at mean >= 0")
    for mu in (0.0, 0.0002, 0.0005, 0.001):
        for sd in (0.0015, 0.002, 0.003):
            v = sd * sd
            g3, g8, g16 = (S.per_decision(v, eta(k), mu)[0] for k in (3, 8, 16))
            print(f"  mu {mu:+.4f} sd {sd}: g8 {ELO * N_BAND * g8:5.1f} Elo, g8/g3 {g8 / g3:.3f}, g16/g8 {g16 / g8:.3f}")


def d2():
    print("\n4. D2: same-band round 2 adopted against pi_1 at the ceiling increment (SE 1.79, line 2 SE)")
    for g8 in (4.0, 5.1, 6.0, 7.0, 8.0, 8.6):
        inc = (math.sqrt(2) - 1) * g8
        print(f"  round-1 first-order gain {g8:4.1f}: increment <= {inc:.2f} Elo, P(adopt) <= {Phi(inc / SE_MATCH - 2):.2f}")


def main():
    stationarity()
    regime()
    posterior_nonnegative()
    d2()


if __name__ == "__main__":
    main()
