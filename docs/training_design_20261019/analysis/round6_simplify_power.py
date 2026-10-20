"""Round 6, simplify stance: the round-1 floors and early-stop rates when the adoption match is a
PURE side-swapped paired match (no luck columns), and the plan's budget.

Reuses round5_synthesis_power.py unchanged (its model, inputs and functions), overriding only the
match's kept share: pairs alone keep 0.847 (round6_simplify_pairs_raw.out) instead of the
corrected match's 0.747. The switch rule is the ridge sign rule with lambda chosen by progressive
loss; in the scalar model this is the posterior-mean rule the synthesis already evaluates (its
c = mu (1 - 1/beta) at the true beta), and round6_simplify_rule.out shows the simulated rule within
0.6 Elo of that oracle. Standard library only.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round5_synthesis_power as S  # noqa: E402

KEPT_PAIRS = 0.847
MATCH_GAMES = 12_800
MATCH_PRICE = MATCH_GAMES * 0.42 * 17 / 60 / 800
BATCH_PRICE = S.BATCH * S.P.PRICE_PER_GAME


def main():
    S.MATCH_SE_Z = math.sqrt(KEPT_PAIRS / MATCH_GAMES)
    S.TWO_MATCH_SE_ELO = 2 * S.P.ELO_PER_Z * S.MATCH_SE_Z
    inputs = S.P.read_inputs()
    print(f"paired PURE match: SE {S.TWO_MATCH_SE_ELO / 2:.2f} Elo, 2 SE {S.TWO_MATCH_SE_ELO:.1f} Elo, "
          f"80%-power line {2.84 * S.TWO_MATCH_SE_ELO / 2:.1f} Elo, ${MATCH_PRICE:.2f}")
    for d_eff in (200, 776):
        for mu in (0.0, -0.0018):
            sd = S.floor_sd_match(inputs, mu, d_eff, 8, 0.25)
            r = S.round_readout(inputs, sd, mu, d_eff, 8, 0.25)
            print(f"  d_eff {d_eff:3d} mu {mu:+.4f}: match-only floor sd {sd:.4f} "
                  f"(gain {r['gain_elo']:.1f} Elo, forecast {r['forecast_elo']:.1f} +- {r['forecast_se_elo']:.1f})")
    print("early stop after batch 4 (students 1-3's progressive upper bound under 2 match SEs), d_eff 776:")
    for mu in (0.0, -0.0018):
        for sd in (0.0, 0.001, 0.002, 0.003):
            ps, m, se = S.early_stop_probability(inputs, sd, mu, 776, 0.25)
            g = S.round_readout(inputs, sd, mu, 776, 8, 0.25)["gain_elo"]
            print(f"  mu {mu:+.4f} sd {sd:.3f}: P(stop) {ps:.2f}; final gain had it run {g:.1f} Elo")
    smoke = 2_000 * S.P.PRICE_PER_GAME
    rnd = 8 * BATCH_PRICE
    p_distill = 0.6
    recipe = 2.1
    stages = {
        "S1 smoke + round 1": smoke + rnd,
        "S2 adoption match": MATCH_PRICE,
        "S2 fallback match (only on a miss)": MATCH_PRICE,
        "S3 round 2 + direct match + own1 match": rnd + 2 * MATCH_PRICE,
        "S4 P-distill + recipe + 1-2 matches": p_distill + recipe + 2 * MATCH_PRICE,
        "S5 round 3 + direct match + own1 match": rnd + 2 * MATCH_PRICE,
    }
    total = sum(stages.values())
    for k, v in stages.items():
        print(f"  {k}: ${v:.2f}")
    print(f"  total at most ${total:.1f}; left of $85: ${85 - total:.1f}; "
          f"overhead share that still fits: {(85 - total) / (3 * rnd):.3f} of generation")


if __name__ == "__main__":
    main()
