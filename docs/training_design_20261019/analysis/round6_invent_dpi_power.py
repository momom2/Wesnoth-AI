"""Round 6 (invent stance): what a lower kept-variance ratio r buys the coin round and the match.

The coin label's noise is sigma^2 = Var(z) r / (p (1 - p)), where r is the share of the result's
variance the control variates leave (0.732 for the setup baseline plus the seven hand columns in
own1 self-play, round5_provenance_match_variance.out). Dice-priced innovations (round6_invent.md)
would change only r. This script reruns the round-5 synthesis power model
(round5_synthesis_power.py, its Gaussian model and inputs unchanged) with r replaced, and reads:
- the match-only floor (smallest near-tie spread sd whose final student gains 2.84 match SEs) and
  the full floor (gain >= 2 match SEs and progressive forecast >= 2.84 SE) at 8 batches;
- the progressive forecast's SE at sd 0.003;
- the batches (of 100,000 games, $2.47 each) needed for the match-only floor to reach the value
  it has at r = 0.732 with 8 batches;
- the corrected match's SE at 12,800 games and the games giving SE 2.57 Elo, if the same factor
  applied to the match's kept ratio 0.700 (side-swapped pairs plus columns).
Every output is derived from the stated model; none is a prediction of r.
Standard library only.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round5_simplify_power as P  # noqa: E402
import round5_synthesis_power as S  # noqa: E402

R_NOW = 0.732
R_MATCH_NOW = 0.700
D_EFF = 776
BATCHES = 8


def with_r(inputs, r):
    out = dict(inputs)
    out["r_R_own1"] = r
    return out


def batches_to_floor(inputs, mu, target_sd):
    for b in range(2, 41):  # the progressive forecast needs at least 2 batches
        if S.floor_sd_match(inputs, mu, D_EFF, b, 0.25) <= target_sd:
            return b
    return None


def main():
    base = P.read_inputs()
    print(f"inputs: r now {base['r_R_own1']:.3f}, Var(z) {base['var_z']:.4f}, band {S.N_BAND:.1f} per side-game, "
          f"d_eff {D_EFF}, p 0.25, batch ${S.BATCH * P.PRICE_PER_GAME:.2f}")
    targets = {mu: S.floor_sd_match(with_r(base, base["r_R_own1"]), mu, D_EFF, BATCHES, 0.25) for mu in (0.0, -0.0018)}
    print(f"match-only floors at r now, 8 batches: mu 0 {targets[0.0]:.4f}, mu -0.0018 {targets[-0.0018]:.4f}")
    for r in (0.732, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
        inp = with_r(base, r)
        line = [f"r {r:.3f} (information x{base['r_R_own1'] / r:.2f}):"]
        for mu in (0.0, -0.0018):
            fm = S.floor_sd_match(inp, mu, D_EFF, BATCHES, 0.25)
            ff = S.floor_sd(inp, mu, D_EFF, BATCHES, 0.25)
            b = batches_to_floor(inp, mu, targets[mu])
            line.append(f"mu {mu:+.4f}: floors sd {fm:.4f} (match-only) {ff:.4f} (full); "
                        f"batches for today's floor {b} (${b * S.BATCH * P.PRICE_PER_GAME:.1f})")
        rd = S.round_readout(inp, 0.003, 0.0, D_EFF, BATCHES, 0.25)
        line.append(f"at sd 0.003, mu 0: gain {rd['gain_elo']:.1f} Elo, forecast SE {rd['forecast_se_elo']:.1f}")
        r_match = R_MATCH_NOW * r / R_NOW
        se = P.ELO_PER_Z * math.sqrt(r_match / 12800)
        games = 12800 * r_match / R_MATCH_NOW
        line.append(f"match at the same factor: kept {r_match:.3f}, SE {se:.2f} Elo at 12,800 games; "
                    f"SE 2.57 at {games:,.0f} games (${games * 0.42 * 17 / 60 / 800:.2f})")
        # the same 12,800-game match read with the match's ratio scaled too: its 2-SE line drops,
        # so the floor drops further (round5_synthesis_power's line uses Var ratio 0.747)
        saved = S.TWO_MATCH_SE_ELO
        S.TWO_MATCH_SE_ELO = 2 * P.ELO_PER_Z * math.sqrt(0.747 * r / R_NOW / 12800)
        fm0 = S.floor_sd_match(inp, 0.0, D_EFF, BATCHES, 0.25)
        fm1 = S.floor_sd_match(inp, -0.0018, D_EFF, BATCHES, 0.25)
        S.TWO_MATCH_SE_ELO = saved
        line.append(f"match-only floors with the match also corrected: mu 0 {fm0:.4f}, mu -0.0018 {fm1:.4f}")
        print("\n  ".join(line))


if __name__ == "__main__":
    main()
