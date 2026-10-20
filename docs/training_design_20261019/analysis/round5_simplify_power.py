"""Round 5, simplify stance: what the near-tie coin student can detect and gain, per budget.

Every input is read from a record or is a stated choice; every output is derived.

Measured inputs (read here from the files that hold them):
- r_R, the share of a result's variance the critic-free correction keeps (setup baseline plus the
  seven luck columns, fitted on other matches): own1 self-play matches gen_own1_selfplay and
  own64_selfplay in round5_provenance_match_variance.out (0.730, 0.734); pooled 0.747.
  Var(z) = 1.0004 for own1 self-play (round5_simplify_variance.json).
- The band share: decisions whose top-two joint log-prior margin (after the end_turn offset) is
  under w, over multi-legal decisions: round5_simplify_band.json (parity3, 100 side-games).
- own1's decisions per side-game: round5_provenance_stop_acts.out, gen_own1_selfplay (decisions per
  side-turn without end_turn, A and B, and side-turns per game), plus one end_turn per side-turn.
  own1's band count = band share x own1's decisions (derived, assuming parity3's margin
  distribution per decision; S0 reads own1's own).

Choices: coin probability p, band width w, batch size, the price of a game ($0.42 an hour over
17,000 games an hour, the seed's throughput), the feature count d = 776 (two 384-wide token
contrasts, four kind indicators, four factored log-probabilities), the protocol (train on the
first half of the batches, freeze the student, read its forecast on the second half).

Model (stated): at band decisions the true advantage of the runner-up over the top, as far as the
student's span can express it, is a ~ N(mu, v). A ridge student fitted on N_train labels predicts
f = a + e, e ~ N(0, eta), eta = sigma_eps^2 d_eff / N_train (random-design ridge, Hsu, Kakade and
Zhang 2012, bias ignored), where sigma_eps^2 = Var(z) r_R / (p (1 - p)) is the label noise. Then
  calibration slope    beta = v / (v + eta),  SE(beta_hat) = sqrt(sigma_eps^2 / (N_test (v + eta)))
  gain per decision    E[a 1(f > 0)] = mu Phi(mu / s) + (v / s) phi(mu / s),  s = sqrt(v + eta)
  forecast SEs         G_DR (sum of switched held-out labels): sqrt(sigma_eps^2 k / n_sg)
                       G_plug (beta_hat times the summed switched predictions): SE(beta_hat) * sum_switch f
  Elo = 347.4 z near an even match (800 / ln 10).
Standard library only.
"""
import json
import math
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
ELO_PER_Z = 800 / math.log(10)
PRICE_PER_GAME = 0.42 / 17000
D_FEATURES = 776
MATCH_GAMES = 12800
BATCH = 100_000
P_COIN, BAND = 0.25, "0.35"


def phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def read_inputs():
    text = (HERE / "round5_provenance_match_variance.out").read_text()
    both = {m: float(b) for m, b in re.findall(r"^\s+(\S+)\s+\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)$", text, re.M)}
    pooled = float(re.search(r"both\s+pooled Var\(G\)/Var\(z\) = ([\d.]+)", text).group(1))
    band = json.loads((HERE / "round5_simplify_band.json").read_text())
    var_z = json.loads((HERE / "round5_simplify_variance.json").read_text())["own1_selfplay"]["var_z"]
    acts = (HERE / "round5_provenance_stop_acts.out").read_text()
    m = re.search(r"gen_own1_selfplay: .*decisions per side-turn A ([\d.]+) B ([\d.]+), "
                  r"A side-turns per game ([\d.]+)", acts)
    per_turn = (float(m.group(1)) + float(m.group(2))) / 2 + 1.0   # plus the end_turn decision
    own1_decisions = per_turn * float(m.group(3))
    share = {w: band["bands"][w]["decisions_in_band"] / band["multi_legal_decisions_per_side_game"]
             for w in band["bands"]}
    return {"r_R_own1": (both["gen_own1_selfplay"] + both["own64_selfplay"]) / 2, "r_R_pooled": pooled,
            "var_z": var_z, "band_share": share, "own1_decisions_per_side_game": own1_decisions,
            "n_band_own1": {w: s * own1_decisions for w, s in share.items()},
            "n_band_parity3": {w: band["bands"][w]["decisions_in_band"] for w in band["bands"]}}


def case(inputs, n_band, d_eff, train_games, test_games, v, mu=0.0, p=P_COIN):
    sigma_eps2 = inputs["var_z"] * inputs["r_R_own1"] / (p * (1 - p))
    n_train, n_test = train_games * 2 * n_band, test_games * 2 * n_band
    eta = sigma_eps2 * d_eff / n_train
    s = math.sqrt(v + eta)
    se_beta = math.sqrt(sigma_eps2 / (n_test * (v + eta)))
    switches = n_band * Phi(mu / s)
    sum_f = n_band * (mu * Phi(mu / s) + s * phi(mu / s))
    gain = n_band * (mu * Phi(mu / s) + (v / s) * phi(mu / s))
    return {"beta": v / (v + eta), "beta_over_se": v / (v + eta) / se_beta, "gain_elo": ELO_PER_Z * gain,
            "switches_per_side_game": switches,
            "se_forecast_plug_elo": ELO_PER_Z * se_beta * sum_f,
            "se_forecast_dr_elo": ELO_PER_Z * math.sqrt(sigma_eps2 * switches / (2 * test_games))}


def smallest_v(inputs, n_band, d_eff, train_games, test_games, mu, target_elo):
    """Smallest v at which the gain reaches target_elo and beta_hat is 2 SE above 0."""
    lo, hi = 1e-9, 1e-2
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        c = case(inputs, n_band, d_eff, train_games, test_games, mid, mu)
        ok = c["gain_elo"] >= target_elo and c["beta_over_se"] >= 2
        lo, hi = (lo, mid) if ok else (mid, hi)
    return hi


def main():
    inputs = read_inputs()
    match_se = ELO_PER_Z * math.sqrt(inputs["r_R_pooled"] / MATCH_GAMES)
    out = {"inputs": inputs, "match_se_elo": match_se,
           "match_cost_dollars": MATCH_GAMES * 0.42 * 17 / 60 / 800, "thresholds": [], "gain_table": []}
    sigma_eps2 = inputs["var_z"] * inputs["r_R_own1"] / (P_COIN * (1 - P_COIN))
    print(f"inputs: r_R own1 {inputs['r_R_own1']:.3f}, pooled {inputs['r_R_pooled']:.3f}, Var(z) {inputs['var_z']:.4f}")
    print(f"band share of decisions {dict((w, round(s, 3)) for w, s in inputs['band_share'].items())}; "
          f"own1 decisions per side-game {inputs['own1_decisions_per_side_game']:.1f}; band decisions per side-game "
          f"own1 (derived) {dict((w, round(n, 1)) for w, n in inputs['n_band_own1'].items())}, "
          f"parity3 (measured) {inputs['n_band_parity3']}")
    print(f"adoption match: {MATCH_GAMES} corrected games, SE {match_se:.2f} Elo, ${out['match_cost_dollars']:.2f}")
    print(f"label noise sigma_eps^2 at p = {P_COIN}: {sigma_eps2:.2f}; batch of {BATCH} games ${BATCH * PRICE_PER_GAME:.2f}")
    target = 2 * match_se
    for label, n_band in (("own1 derived", inputs["n_band_own1"][BAND]), ("parity3 measured", inputs["n_band_parity3"][BAND])):
        print(f"\nband count {label}: {n_band:.1f} per side-game at w = {BAND}")
        for batches in (2, 4, 8, 16):
            train = test = batches // 2 * BATCH
            cost = batches * BATCH * PRICE_PER_GAME
            for d_eff in (200, D_FEATURES):
                for mu in (0.0, -0.0018):
                    v = smallest_v(inputs, n_band, d_eff, train, test, mu, target)
                    c = case(inputs, n_band, d_eff, train, test, v, mu)
                    rec = {"band": label, "batches": batches, "cost": cost, "d_eff": d_eff, "mu": mu,
                           "v_min": v, "sd_min": math.sqrt(v), **c}
                    out["thresholds"].append(rec)
                    print(f"  {batches:2d} batches (${cost:4.1f}, half train, half forecast), d_eff {d_eff:3d}, mu {mu:+.4f}: "
                          f"smallest sd {math.sqrt(v):.4f}; there beta {c['beta']:.2f}, gain {c['gain_elo']:.1f} Elo, "
                          f"{c['switches_per_side_game']:.0f} switches/side-game, forecast SE plug {c['se_forecast_plug_elo']:.1f} "
                          f"DR {c['se_forecast_dr_elo']:.1f} Elo")
    n_band = inputs["n_band_own1"][BAND]
    print(f"\ngain at 8 batches ($19.8: 4 train, 4 forecast), own1 band {n_band:.1f}, d_eff {D_FEATURES}, mu 0:")
    for sd in (0.001, 0.002, 0.003, 0.005, 0.01):
        c = case(inputs, n_band, D_FEATURES, 4 * BATCH, 4 * BATCH, sd * sd)
        out["gain_table"].append({"sd": sd, **c})
        print(f"  sd {sd:.3f}: gain {c['gain_elo']:6.1f} Elo, beta {c['beta']:.2f} ({c['beta_over_se']:.1f} SE), "
              f"forecast SE plug {c['se_forecast_plug_elo']:.1f} Elo")
    n1 = BATCH * 2 * n_band
    out["plant_se_even_hex"] = math.sqrt(sigma_eps2 / (n1 * 0.5))
    out["plant_se_unit_direction"] = math.sqrt(sigma_eps2 / n1)
    out["mean_label_se_batch1"] = math.sqrt(sigma_eps2 / n1)
    print(f"\nbatch 1 ({n1 / 1e6:.1f}M labels): planted even-hex coefficient SE {out['plant_se_even_hex']:.5f} "
          f"(contrast variance 0.5 assumed, read blind from the logs); planted unit-variance direction SE "
          f"{out['plant_se_unit_direction']:.5f}; mean label (coin cost per deviation) SE {out['mean_label_se_batch1']:.5f}")
    print(f"coins against fork pairs, information per game-equivalent: {2 * n_band * P_COIN * (1 - P_COIN) / (inputs['var_z'] * inputs['r_R_own1']):.1f} "
          f"against {1 / (2 * 0.608):.2f} (fork pair, playout noise 0.608 at temperature 0.5, round5_provenance_playout_noise.out)")
    (HERE / "round5_simplify_power.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
