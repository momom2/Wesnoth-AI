"""Round 5, critique of the simplify stance: the switch rule under a negative mean.

Reuses round5_simplify_power.py's stated Gaussian model and inputs unchanged
(case(), read_inputs()): at band decisions the runner-up's advantage is
a ~ N(mu, v), the frozen student predicts f = a + e, e ~ N(0, eta). Compares

  sign rule        switch where f > 0                 (round5_simplify.md 2.4)
  calibrated rule  switch where E[a | f] > 0, i.e.     (the posterior mean's sign)
                   mu + beta (f - mu) > 0, beta = v / (v + eta)

at 8 batches (4 train, 4 forecast), the derived own1 band (62.0) and the
unit-corrected band (76.1 = 0.339 x 224.6, round5_simplify_critique_decision_units.out),
d_eff 776, for mu in {0, -0.0018}; mu = -0.0018 is the case the power script
itself carries (the stop window's mean per act, round5_provenance_stop_acts.out).
Standard library only.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round5_simplify_power as P  # noqa: E402


def gain_rule(n_band, v, eta, mu, rule):
    s = math.sqrt(v + eta)
    if rule == "sign":
        t = (0.0 - mu) / s
    else:
        beta = v / (v + eta)
        threshold = mu * (1.0 - 1.0 / beta)  # f above which mu + beta (f - mu) > 0
        t = (threshold - mu) / s
    # E[a 1(f > c)] for (a, f) jointly normal: mu (1 - Phi(t)) + (v / s) phi(t)
    per_decision = mu * (1.0 - P.Phi(t)) + (v / s) * P.phi(t)
    return P.ELO_PER_Z * n_band * per_decision, n_band * (1.0 - P.Phi(t))


def main():
    inputs = P.read_inputs()
    sigma_eps2 = inputs["var_z"] * inputs["r_R_own1"] / (P.P_COIN * (1 - P.P_COIN))
    for n_band in (inputs["n_band_own1"][P.BAND], 0.339 * 224.6):
        n_train = 4 * P.BATCH * 2 * n_band
        eta = sigma_eps2 * P.D_FEATURES / n_train
        print(f"band {n_band:.1f} per side-game, eta {eta:.3e}")
        for mu in (0.0, -0.0018):
            for sd in (0.002, 0.003, 0.005, 0.010):
                v = sd * sd
                g_sign, k_sign = gain_rule(n_band, v, eta, mu, "sign")
                g_cal, k_cal = gain_rule(n_band, v, eta, mu, "calibrated")
                print(f"  mu {mu:+.4f} sd {sd:.3f}: sign rule {g_sign:+6.1f} Elo ({k_sign:4.1f} switches), "
                      f"calibrated {g_cal:+6.1f} Elo ({k_cal:4.1f} switches)")


if __name__ == "__main__":
    main()
