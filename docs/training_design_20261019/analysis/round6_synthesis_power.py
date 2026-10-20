"""Round 6 synthesis: what round 1 detects and costs under the round-6 instruments.

Reuses the stated Gaussian model of round5_synthesis_power.py (through round5_simplify_power.py)
unchanged: at band decisions the in-span advantage of the runner-up is a ~ N(mu, v); a student
fitted on N labels predicts f = a + e, e ~ N(0, eta), eta = sigma^2 d_eff / N; the switch rule's
gain per band decision is per_decision(v, eta, mu). With an unpenalised intercept and lambda chosen
by held-out loss, the ridge sign rule is the model's posterior-mean rule, the same as the calibrated
rule in-model (round6_simplify_rule.out). The model states what each outcome would mean; it is not
a forecast.

Measured inputs, each read in its record:
- label residual: the full-cell setup baseline plus the seven columns keeps 0.688 in own1 self-play
  (round6_provenance_cells.out); Var(z) 1.0004 (round5_simplify_variance.json); so
  sigma^2 = 1.0004 x 0.688 / 0.1875 at p 0.25;
- raw full-cell share 0.157 (round6_simplify_pairs_raw.out): the cell alone keeps 0.843, the
  residual if ruling (b) is "no";
- matches: PURE side-swapped pairs keep 0.847 of a game's variance (round6_simplify_pairs_raw.out);
  a one-network match runs in run_elo_batch at --jobs 120 at $2.51e-5 a game (ref_p120x6,
  round6_provenance_prices.out); a two-network match is priced at the eval driver's median
  $1.17e-4 a decisive game (same file), an upper bound until timed;
- replay at 2,440 decisions a second (BACKLOG.md, via round5_simplify_critique_extra.py); the
  18,800 walked decisive games take 8.86 M forwards (round6_invent_critique_checks.out section 3);
- distillation recipe $2.1 (docs/memory_own_trajectories_prereg_20261011.md); P-distill $1.09
  (round6_provenance.md provenance table, rederived there from measured rates).

Derived here:
1. PURE paired match SE and price by size;
2. match-only floors at 8 batches, adoption at 32,000 games (the size chosen in the design);
3. the interim match after batch 3 (student 3 against own1, 32,000 games, stop iff its estimate
   is below 0), and (3b) where to place it: P(stop), and the combined power (continue) x (adopt at batch 8), whose 0.8 line is
   what a null round bounds with the stop counted;
4. the forecast's SE and the miss line;
5. the dice-priced corrector's admission line: label information is p(1-p)/E[R^2], eta scales with
   r, so a corrector keeping r holds the floors with ceil(8 r / 0.688) batches;
6. the budget and the price lines for three and two rounds;
7. the SE of reading the switches' interaction with the opponent's coins from products of two
   coin scores (unbiased, since the coins are independent).
Standard library only.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round5_simplify_power as P  # noqa: E402
import round5_synthesis_power as S  # noqa: E402

LABEL_R = 0.688
CELL_ONLY_R = 1 - 0.157
PAIRS_RAW = 0.847
GEN_PRICE = 2.51e-5
EVAL_PRICE = 1.17e-4
BATCH = 100_000
BATCHES = 8
MATCH = 32_000
INTERIM = 3
N_BAND = S.N_BAND
REPLAY_RATE = 2440
HOUR = 0.42


def match_se(games, kept=PAIRS_RAW):
    return P.ELO_PER_Z * math.sqrt(kept / games)


def inputs_at(r):
    base = P.read_inputs()
    return dict(base, r_R_own1=r)


def gain(inputs, sd, mu, d_eff, batches, p=0.25):
    s2 = S.sigma2(inputs, p)
    eta = s2 * d_eff / (batches * 2 * N_BAND * BATCH)
    g, sw = S.per_decision(sd * sd, eta, mu)
    return P.ELO_PER_Z * N_BAND * g, N_BAND * sw


def bisect(fn, lo=1e-5, hi=0.05):
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        lo, hi = (lo, mid) if fn(mid) else (mid, hi)
    return hi


def combined_power(inputs, sd, mu, d_eff, se):
    g4, _ = gain(inputs, sd, mu, d_eff, INTERIM)
    g8, _ = gain(inputs, sd, mu, d_eff, BATCHES)
    p_stop = P.Phi(-g4 / se)
    return (1 - p_stop) * P.Phi(g8 / se - 2), p_stop, g4, g8


def main():
    inp = inputs_at(LABEL_R)
    s2 = S.sigma2(inp, 0.25)
    print(f"label sigma^2 at p 0.25: {s2:.2f} (r {LABEL_R}); band {N_BAND:.1f} per side-game")

    print("\n1. PURE paired matches in the pool (pairs keep 0.847)")
    for games in (12_800, 25_600, 32_000, 51_200):
        se = match_se(games)
        print(f"  {games:6,} games: SE {se:.2f} Elo, 2 SE {2 * se:.1f}, 80%-power gain {2.84 * se:.1f}, "
              f"${games * GEN_PRICE:.2f}; same SE corrected (0.700) needs {games * 0.700 / PAIRS_RAW:,.0f} games, "
              f"saving ${games * (1 - 0.700 / PAIRS_RAW) * GEN_PRICE:.2f}")
    se_m = match_se(MATCH)

    print(f"\n2. match-only floors, {BATCHES} batches, adoption at {MATCH:,} PURE paired games "
          f"(gain 2.84 SE = {2.84 * se_m:.1f} Elo)")
    floors = {}
    for d_eff in (200, 776):
        for mu in (0.0, -0.0018):
            sd = bisect(lambda x: gain(inp, x, mu, d_eff, BATCHES)[0] >= 2.84 * se_m)
            floors[(d_eff, mu)] = sd
            print(f"  d_eff {d_eff:3d} mu {mu:+.4f}: floor sd {sd:.4f}")

    print(f"\n3. interim match after batch {INTERIM} ({MATCH:,} games, SE {se_m:.2f}): stop iff the estimate is below 0")
    for d_eff in (200, 776):
        for mu in (0.0, -0.0018):
            print(f"  d_eff {d_eff} mu {mu:+.4f}")
            for sd in (0.0, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005):
                pw, ps, g4, g8 = combined_power(inp, sd, mu, d_eff, se_m)
                print(f"    sd {sd:.4f}: student {INTERIM} {g4:5.1f} Elo, student 8 {g8:5.1f}; P(stop) {ps:.2f}; "
                      f"power with stop {pw:.2f} (without {P.Phi(g8 / se_m - 2):.2f})")
            sd80 = bisect(lambda x: combined_power(inp, x, mu, d_eff, se_m)[0] >= 0.8)
            print(f"    null bound with the stop counted (combined power 0.8): sd {sd80:.4f} "
                  f"(no-stop floor {floors[(d_eff, mu)]:.4f})")

    print("\n3b. where to place the interim match: after batch b, stop iff student b's estimate is below 0")
    for b in (2, 3, 4, 5):
        cells = []
        for d_eff in (200, 776):
            for mu in (0.0, -0.0018):
                def pw(x, b=b, d_eff=d_eff, mu=mu):
                    gb, _ = gain(inp, x, mu, d_eff, b)
                    g8, _ = gain(inp, x, mu, d_eff, BATCHES)
                    return (1 - P.Phi(-gb / se_m)) * P.Phi(g8 / se_m - 2)
                cells.append(f"d{d_eff} mu{mu:+.4f} {bisect(lambda x: pw(x) >= 0.8):.4f}")
        cost_null = b * BATCH * GEN_PRICE + MATCH * GEN_PRICE + 0.5 * (BATCHES - b) * BATCH * GEN_PRICE
        print(f"  b {b}: null bound with the stop counted " + "; ".join(cells) +
              f"; expected coin spend at a null ${cost_null:.2f} (no interim ${BATCHES * BATCH * GEN_PRICE:.2f})")

    print("\n4. forecast SE (progressive, students 4-7 on batches 5-8) and the miss line")
    for mu in (0.0, -0.0018):
        for sd in (0.002, 0.003, 0.005):
            r = S.round_readout(inp, sd, mu, 776, BATCHES, 0.25)
            line = 2 * math.sqrt(r["forecast_se_elo"] ** 2 + se_m ** 2)
            print(f"  mu {mu:+.4f} sd {sd:.3f}: final gain {r['gain_elo']:.1f}, forecast {r['forecast_elo']:.1f} "
                  f"+- {r['forecast_se_elo']:.1f} Elo; miss line {line:.1f} Elo")

    print("\n5. dice-priced corrector: batches holding the floors at kept ratio r (eta scales with r)")
    for b in (8, 7, 6, 5, 4):
        r_line = LABEL_R * b / BATCHES
        print(f"  {b} batches suffice iff r <= {r_line:.3f}; saves ${(BATCHES - b) * BATCH * GEN_PRICE:.2f} a round")
    print(f"  ruling (b) 'no' (cell alone, r {CELL_ONLY_R:.3f}): batches x{CELL_ONLY_R / LABEL_R:.3f} for the same floors, "
          f"or floors x{math.sqrt(CELL_ONLY_R / LABEL_R):.3f} at 8 batches")
    for d_eff in (200, 776):
        for mu in (0.0, -0.0018):
            sd = bisect(lambda x: gain(inputs_at(CELL_ONLY_R), x, mu, d_eff, BATCHES)[0] >= 2.84 * se_m)
            print(f"    cell alone, d_eff {d_eff} mu {mu:+.4f}: floor sd {sd:.4f}")

    print("\n6. budget at measured prices")
    # the dice-priced fit regresses one result per game: it needs the 18,800 walked decisive games,
    # 8.86 M forwards at their own turn counts (round6_invent_critique_checks.out section 3)
    replay = 8.86e6 / REPLAY_RATE / 3600 * HOUR
    s0 = {"replay of the 18,800 walked decisive games": replay, "mirror null 3,200 games": 3_200 * GEN_PRICE,
          "attack known answer 6,400 games": 6_400 * GEN_PRICE, "coin smoke 2,000 games": 2_000 * GEN_PRICE,
          "box setup (median)": 0.018}
    for k, v in s0.items():
        print(f"  S0 {k}: ${v:.3f}")
    S0 = sum(s0.values())
    round_games = BATCHES * BATCH * GEN_PRICE
    m = MATCH * GEN_PRICE
    s4 = 1.09 + 2.1 + 2 * 12_800 * EVAL_PRICE
    stages = {"S0": S0, "S1 (8 batches + interim)": round_games + m, "S2 adoption": m,
              "S3 round 2 (+ interim, direct, claim)": round_games + 3 * m, "S4 reset": s4,
              "S5 round 3": round_games + 3 * m}
    for k, v in stages.items():
        print(f"  {k}: ${v:.2f}")
    reserve = 2 * m
    total = sum(stages.values())
    print(f"  total ${total:.2f}; miss reserve (E3 and E, one round) ${reserve:.2f}; left of $85 ${85 - total - reserve:.2f}")
    fixed3 = total - 3 * round_games + reserve
    fixed2 = fixed3 - stages["S5 round 3"] + round_games
    print(f"  three rounds fit iff price x (1 + overhead) <= ${(85 - fixed3) / (3 * BATCHES * BATCH):.2e} a coin game "
          f"(fixed ${fixed3:.2f}); two rounds iff <= ${(85 - fixed2) / (2 * BATCHES * BATCH):.2e} (fixed ${fixed2:.2f})")
    print(f"  at the measured $2.51e-5: overhead up to {(85 - fixed3) / (3 * BATCHES * BATCH) / GEN_PRICE - 1:.3f} "
          f"fits three rounds; the 1.41x host spread does not")
    print(f"  a null round 1 stopped at batch {INTERIM}: ${S0 + INTERIM * BATCH * GEN_PRICE + m:.2f}; run through: ${S0 + round_games + 2 * m:.2f}")

    print("\n7. reading the interaction term from the coins themselves")
    # For own switch set S and the opponent's band states O, the product
    # (sum_{i in S} s_i w_i)(sum_{j in O} s_j w_j) R is unbiased for sum_{i,j} H_ij (the coins are
    # independent, so each pair's score product isolates the mixed partial of the multilinear J).
    # Its per-game variance is about Var(R) E[(sum w_i)^2] E[(sum w_j)^2], E[w^2] = 1/(p(1-p)).
    w2 = 1 / 0.1875
    for n_s in (19.0, 38.1):
        var_game = LABEL_R * (n_s * w2) * (N_BAND * w2)
        for games in (800_000, 2_400_000):
            se = P.ELO_PER_Z * math.sqrt(var_game / games)
            print(f"  {n_s:4.1f} switches against {N_BAND:.1f} opponent band states: SE {se:.0f} Elo at {games:,} coin games")


if __name__ == "__main__":
    main()
