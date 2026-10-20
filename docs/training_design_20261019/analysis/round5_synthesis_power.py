"""Round 5 synthesis: what one coin round can detect and gain, under the repaired protocol.

Repairs over round5_simplify_power.py (its model and inputs are reused unchanged):
- the band count in the decision unit of the band share: 0.339 x 224.6 = 76.1 band decisions per
  side-game (round5_simplify_critique_decision_units.out), not 62.0;
- the calibrated rule: switch where E[a | f] = mu + beta (f - mu) > 0, the student having an
  unpenalised intercept (round5_simplify_critique_rule.py's gain formula);
- the progressive protocol: every batch trains the student; batch k+1 is generated with the
  student fitted on batches 1..k logging its prediction f at every band decision, so batch k+1's
  labels read that student's switches out of sample (progressive validation, Blum, Kalai and
  Langford 1999). The forecast is the model-free sum of those labels over the switched decisions,
  pooled over batches 5..B (students 4..B-1). The match reads the student fitted on all B batches.

Model (stated, round5_simplify_power.py): at band decisions the runner-up's advantage, as far as
the student's span reaches, is a ~ N(mu, v); a student fitted on N labels predicts f = a + e,
e ~ N(0, eta), eta = sigma^2 d_eff / N, sigma^2 = Var(z) r / (p (1 - p)).
Gain per band decision of a student with error variance eta, rule threshold c on f:
    E[a 1(f > c)] = mu (1 - Phi(t)) + (v / s) phi(t),  s = sqrt(v + eta),  t = (c - mu) / s,
    c = mu (1 - 1 / beta), beta = v / (v + eta)   (c = 0 when mu = 0).
Model-free forecast of one batch, per side-game in z: mean n_band * gain, SE sqrt(k sigma^2 / n_sg),
k = switches per side-game, n_sg = 2 x games (both sides draw coins).

Also derived here:
- the early stop after batch 4: stop when the pooled progressive forecast of batches 2..4
  (students 1..3) has its upper 2-SE bound under 2 match SEs; P(stop) by sd;
- the critic column: for a critic contrast X entered as one more ridge column, the SE of
  kappa * sd(X) on N labels is sqrt(sigma^2 / N); the switching gain a column of strength
  kappa * sd(X) = g carries is about n_band * 0.399 * g per side-game (mu = 0, the column alone);
  the g that gives 2 match SEs, and the games needed to read g at 2.84 SE.
Standard library only.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round5_simplify_power as P  # noqa: E402

N_BAND = 0.339 * 224.6          # band decisions per side-game, decision unit of the share
BATCH = 100_000
MATCH_SE_Z = math.sqrt(0.747 / 12800)          # 12,800 corrected games
TWO_MATCH_SE_ELO = 2 * P.ELO_PER_Z * MATCH_SE_Z


def sigma2(inputs, p):
    return inputs["var_z"] * inputs["r_R_own1"] / (p * (1 - p))


def per_decision(v, eta, mu):
    s = math.sqrt(v + eta)
    if v == 0.0:
        # a true null: the estimated slope is noise around 0; with mu = 0 the estimated rule
        # switches about half the band at zero gain, with mu < 0 the intercept (SE about
        # sqrt(sigma^2 / N), far below |mu|) switches none, with mu > 0 all
        if mu == 0.0:
            return 0.0, 0.5
        return (0.0, 0.0) if mu < 0 else (mu, 1.0)
    c = 0.0 if mu == 0.0 else mu * (1.0 - 1.0 / (v / (v + eta)))
    t = (c - mu) / s
    gain = mu * (1.0 - P.Phi(t)) + (v / s) * P.phi(t)
    return gain, 1.0 - P.Phi(t)


def round_readout(inputs, sd, mu, d_eff, batches, p):
    s2 = sigma2(inputs, p)
    labels_per_batch = 2 * N_BAND * BATCH
    v = sd * sd

    def eta(k):
        return s2 * d_eff / (k * labels_per_batch)

    final_gain, final_sw = per_decision(v, eta(batches), mu)
    # progressive forecast over batches 5..B, read by students 4..B-1
    readers = range(4, batches) if batches > 4 else range(1, batches)
    means, variances = [], []
    for k in readers:
        g, sw = per_decision(v, eta(k), mu)
        means.append(N_BAND * g)
        variances.append(max(sw, 1e-12) * N_BAND * s2 / (2 * BATCH))
    w = [1 / x for x in variances]
    pooled = sum(m * wi for m, wi in zip(means, w)) / sum(w)
    pooled_se = math.sqrt(1 / sum(w))
    return {"gain_elo": P.ELO_PER_Z * N_BAND * final_gain, "switches": N_BAND * final_sw,
            "forecast_elo": P.ELO_PER_Z * pooled, "forecast_se_elo": P.ELO_PER_Z * pooled_se,
            "cost": batches * BATCH * P.PRICE_PER_GAME}


def early_stop_probability(inputs, sd, mu, d_eff, p):
    s2 = sigma2(inputs, p)
    labels_per_batch = 2 * N_BAND * BATCH
    v = sd * sd
    means, variances = [], []
    for k in (1, 2, 3):
        g, sw = per_decision(v, s2 * d_eff / (k * labels_per_batch), mu)
        means.append(N_BAND * g)
        variances.append(max(sw, 1e-12) * N_BAND * s2 / (2 * BATCH))
    w = [1 / x for x in variances]
    mean = sum(m * wi for m, wi in zip(means, w)) / sum(w)
    se = math.sqrt(1 / sum(w))
    threshold = TWO_MATCH_SE_ELO / P.ELO_PER_Z
    # stop when estimate + 2 se < threshold
    return P.Phi((threshold - 2 * se - mean) / se), P.ELO_PER_Z * mean, P.ELO_PER_Z * se


def floor_sd(inputs, mu, d_eff, batches, p):
    lo, hi = 1e-5, 0.05
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        r = round_readout(inputs, mid, mu, d_eff, batches, p)
        ok = r["gain_elo"] >= TWO_MATCH_SE_ELO and r["forecast_elo"] >= 2.84 * r["forecast_se_elo"]
        lo, hi = (lo, mid) if ok else (mid, hi)
    return hi


def floor_sd_match(inputs, mu, d_eff, batches, p):
    """Smallest sd at which the final student's gain is 2.84 match SEs (80% power at 2 SE)."""
    lo, hi = 1e-5, 0.05
    target = 2.84 * TWO_MATCH_SE_ELO / 2
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        ok = round_readout(inputs, mid, mu, d_eff, batches, p)["gain_elo"] >= target
        lo, hi = (lo, mid) if ok else (mid, hi)
    return hi


def main():
    inputs = P.read_inputs()
    print(f"band {N_BAND:.1f} per side-game; Var(z) {inputs['var_z']:.4f}; r {inputs['r_R_own1']:.3f}; "
          f"sigma^2 at p 0.25 {sigma2(inputs, 0.25):.2f}, at p 0.5 {sigma2(inputs, 0.5):.2f}; "
          f"2 match SEs {TWO_MATCH_SE_ELO:.1f} Elo; batch ${BATCH * P.PRICE_PER_GAME:.2f}")
    print("\n1. floors: smallest sd where the final student gains 2 match SEs AND the progressive "
          "forecast sits 2.84 SE above 0; then the match-only floor (gain 2.84 match SEs, 7.5 Elo)")
    for p in (0.25, 0.5):
        for batches in (4, 8):
            for d_eff in (200, 776):
                for mu in (0.0, -0.0018):
                    sd = floor_sd(inputs, mu, d_eff, batches, p)
                    r = round_readout(inputs, sd, mu, d_eff, batches, p)
                    sd_m = floor_sd_match(inputs, mu, d_eff, batches, p)
                    rm = round_readout(inputs, sd_m, mu, d_eff, batches, p)
                    print(f"  p {p:.2f} {batches} batches (${r['cost']:.1f}) d_eff {d_eff:3d} mu {mu:+.4f}: floor sd "
                          f"{sd:.4f}; there gain {r['gain_elo']:.1f} Elo, {r['switches']:.0f} switches, "
                          f"forecast {r['forecast_elo']:.1f} +- {r['forecast_se_elo']:.1f} Elo | match-only floor sd "
                          f"{sd_m:.4f} (gain {rm['gain_elo']:.1f}, forecast {rm['forecast_elo']:.1f} +- {rm['forecast_se_elo']:.1f})")
    print("\n2. readouts at 8 batches, p 0.25, d_eff 776")
    for mu in (0.0, -0.0018):
        for sd in (0.0, 0.001, 0.002, 0.003, 0.005, 0.010):
            r = round_readout(inputs, sd, mu, 776, 8, 0.25)
            ps, m, se = early_stop_probability(inputs, sd, mu, 776, 0.25)
            print(f"  mu {mu:+.4f} sd {sd:.3f}: final gain {r['gain_elo']:5.1f} Elo ({r['switches']:4.1f} switches); "
                  f"progressive forecast {r['forecast_elo']:5.1f} +- {r['forecast_se_elo']:.1f}; "
                  f"early stop after batch 4: P {ps:.2f} (its reading {m:.1f} +- {se:.1f})")
    print("\n3. a critic contrast X as one more ridge column (mu 0, the column alone)")
    s2 = sigma2(inputs, 0.25)
    g_needed = (TWO_MATCH_SE_ELO / P.ELO_PER_Z) / (N_BAND * P.phi(0.0))
    print(f"  kappa * sd(X) giving 2 match SEs of switching gain: {g_needed:.5f} per band decision")
    for games in (20_000, 50_000, 200_000, 800_000):
        n = games * 2 * N_BAND
        se = math.sqrt(s2 / n)
        g_read = 2.84 * se
        print(f"  {games:7d} coin games: SE of kappa * sd(X) {se:.5f}; reads at 2.84 SE a column worth "
              f"{P.ELO_PER_Z * N_BAND * P.phi(0.0) * g_read:.1f} Elo of switching gain")
    print("\n4. information per dollar, the adoption match against one forecast batch")
    r = round_readout(inputs, 0.003, 0.0, 776, 8, 0.25)
    batch_se = math.sqrt(r["switches"] * N_BAND / N_BAND * s2 / (2 * BATCH)) * P.ELO_PER_Z
    match_cost = 12800 * 0.42 * 17 / 60 / 800
    print(f"  one 100,000-game batch read model-free at {r['switches']:.0f} switches: SE {batch_se:.1f} Elo for "
          f"${BATCH * P.PRICE_PER_GAME:.2f}; a 12,800-game corrected match: SE {TWO_MATCH_SE_ELO / 2:.2f} Elo for ${match_cost:.2f}")


if __name__ == "__main__":
    main()
