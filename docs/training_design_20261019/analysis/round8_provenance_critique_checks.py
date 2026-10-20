"""Round 8, critique of the provenance stance: three checks of its derived numbers.

A. Is the near-tie part of the attack move identified by the knob record? round8_provenance_knob_checks.py
   reads the move at a = 0.35 from fits linear or quadratic in a+ over 13 vectors (a+ 0.09 to 1.40).
   Here a+ is split at 0.35 into min(a+, 0.35) and max(a+ - 0.35, 0), the other columns unchanged,
   so the move at 0.35 is read from its own coefficient: what the record says about switches at
   gaps below 0.35 without borrowing the slope of the larger gaps.
B. The tested threshold as the design specifies it: tau-tilde_8 is chosen on the welfare of
   students 4-7 (batches 5-8), each at its own noise eta_k, then applied to student 8's f.
   round8_provenance_threshold.py selected at student 8's own f. Here, in expectation (no
   selection noise), the threshold maximising the summed welfare of students 4-7 is applied
   to student 8, against student 8's oracle threshold and tau 0. Model and cases imported.
C. The no-variate label toy (round8_provenance_label_variance.py): its recorded run shows all six plain
   label means above the forced-arm truth. Compared here with the exact advantages over 8 seeds.
numpy and the standard library only.
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import round8_provenance_interim_shapes as I  # noqa: E402
import round8_provenance_knob_checks as KC  # noqa: E402
import round8_provenance_label_variance as LV  # noqa: E402
import round8_provenance_rules as RU  # noqa: E402
import round8_provenance_threshold as TH  # noqa: E402


def check_a():
    n, z, e, a, rr, t = KC.stage1_rows()
    ap = np.maximum(a, 0)
    near, far = np.minimum(ap, 0.35), np.maximum(ap - 0.35, 0)
    X = np.column_stack([near, far, np.minimum(a, 0), np.minimum(e + 1.5, 0), np.maximum(e + 1.5, 0), rr, t])
    c, cov, disp = KC.wls(X, z, n)
    se = np.sqrt(np.diag(cov))
    print("A. hinge at 0.35 in a+ (stage-1 games, weights n, dispersion %.2f)" % disp)
    print(f"   vectors with 0 < a+ < 0.35: {int(np.sum((a > 0) & (a < 0.35)))}; with a+ <= 0.7: {int(np.sum((a > 0) & (a <= 0.7)))}")
    move, move_se = 0.35 * c[0], 0.35 * se[0]
    print(f"   slope on [0, 0.35]: {c[0]:+.4f} +- {se[0]:.4f} per nat -> move at 0.35 {move:+.4f} +- {move_se:.4f} z "
          f"(upper 2-SE {move + 2 * move_se:+.4f}); slope above 0.35: {c[1]:+.4f} +- {se[1]:.4f}")
    for f_up in (16.5,):
        print(f"   per switch at gaps < 0.35 (F 16.5): point {move / f_up:+.4f}, upper 2-SE {(move + 2 * move_se) / f_up:+.4f}")
    Xl = KC.base_design(e, a, rr, t)
    cl, covl, _ = KC.wls(Xl, z, n)
    print(f"   for reference, the linear fit's move at 0.35: {0.35 * cl[0]:+.4f} +- {0.35 * math.sqrt(covl[0, 0]):.4f}")


def welfare_curve(name, c, sd, k, taus):
    r, w = RU.shape(name, sd)
    v, eta = sd * sd, I.eta(k)
    kappa = v / (v + eta)
    p = RU.phi_cdf((r[None, :] - (taus[:, None] - c) / kappa) / math.sqrt(eta))
    return p @ (w * (c + r))


def check_b():
    print("\nB. threshold chosen on students 4-7, applied to student 8 (expectation, no selection noise), Elo")
    print("   case                        tau 0 | student-8 oracle | pooled 4-7 choice applied to 8 | tau* (8) vs pooled tau")
    for name, c, sd in TH.CASES:
        span = 0.06
        taus = np.concatenate([np.linspace(-span, span, 6001), [0.0]])
        g8 = welfare_curve(name, c, sd, 8, taus)
        pooled = sum(welfare_curve(name, c, sd, k, taus) for k in (4, 5, 6, 7))
        j8, jp = int(np.argmax(g8)), int(np.argmax(pooled))
        e = RU.ELO_BAND
        print(f"   {name:9s} c {c:+.4f} sd {sd:.4f}: {e * g8[-1]:5.1f} | {e * g8[j8]:5.1f} | {e * g8[jp]:5.1f} | "
              f"{taus[j8]:+.5f} vs {taus[jp]:+.5f}")


def check_c():
    """Exact advantages of the toy (its latent is Gaussian given the coins), against the plain label over 8 seeds."""
    import itertools
    sd_tot = math.sqrt(0.8 ** 2 + LV.K * LV.LUCK_SD ** 2 + 0.5 ** 2)
    truth = []
    for k in range(LV.K):
        ez = {0: 0.0, 1: 0.0}
        for d in (0, 1):
            for rest in itertools.product((0, 1), repeat=LV.K - 1):
                coins = list(rest[:k]) + [d] + list(rest[k:])
                p = float(np.prod([LV.P if cj else 1 - LV.P for j, cj in enumerate(coins) if j != k]))
                mu = sum(LV.MOVER[j] * coins[j] * LV.DELTA[j] for j in range(LV.K))
                ez[d] += p * math.erf(mu / sd_tot / math.sqrt(2))
        truth.append(LV.MOVER[k] * (ez[1] - ez[0]))
    truth = np.array(truth)
    print("\nC. no-variate label toy: plain label mean minus the exact advantage, 8 seeds of 1,000,000 games")
    print("   exact advantages " + " ".join(f"{x:+.4f}" for x in truth) + " (the toy's forced-arm Monte Carlo read +0.0684 -0.0465 +0.1129 +0.0226 -0.0901 +0.1361)")
    total = np.zeros(LV.K)
    for s in range(8):
        coins, _, z = LV.play(rng=np.random.default_rng(7000 + s), n=1_000_000)
        w = (coins - LV.P) / (LV.P * (1 - LV.P))
        diff = (LV.MOVER * w * z[:, None]).mean(axis=0) - truth
        total += diff
    se = math.sqrt(1 / (LV.P * (1 - LV.P)) / 8_000_000)
    grand = total / 8
    print("   mean difference per decision " + " ".join(f"{x:+.4f}" for x in grand) +
          f" (SE {se:.4f} each); all six pooled: z {grand.mean() / (se / math.sqrt(LV.K)):+.2f}")


def main():
    check_a()
    check_b()
    check_c()


if __name__ == "__main__":
    main()
