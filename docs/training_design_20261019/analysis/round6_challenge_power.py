"""Round 6 (challenge): sizes, resolutions and prices of the dice-first regimen, derived from
measured inputs. Every input is printed with its source; nothing here reads a record that the
other round-6 challenge scripts do not already cite.

Inputs (measured, with their files):
- r_setup = 0.897: share of Var(z) the setup baseline keeps (round5_provenance_match_variance.out).
- r_pairs = 0.700: side-swapped pairs plus the seven columns (round5_provenance_critique_pairs.out),
  used only if ruling (b) allows the columns; else r_setup.
- r_cols_setup = 0.732: setup plus columns in own1 self-play (round5_synthesis_power.out).
- fights per game 109.7 (round5_synthesis_outcomes.out).
- P(defender dies) Bernoulli variance 0.0878 on the counter-weapon subsample
  (round6_challenge_dice_share.out; biased sample, stated).
- attack-involving band pairs per own1 side-game: 8.0 (lower) to 24.3 (upper) at w = 0.35;
  21.5 to 58.5 at w = 1.5 (round6_challenge_band_attacks.out).
- prices: generation game $2.47e-5, match game $1.49e-4 (current_design.md section 4).
- Elo per unit of mean z at an even match: 800 / ln 10 / 2 * ... = 347.4.

Outputs:
1. D1: excess variance of the consequence model's fit, sigma^2 d / N, against the share the fixed
   columns already explain (0.161, round6_challenge_dice_share.out), for d = 455 and N games.
2. P-dice: the SE of a planted coefficient, and the band its recovery must fall in.
3. D2: paired-arm match resolution, per arm and real minus anti, in Elo and per switch.
4. D3: targeted coins' model-free forecast SE at N coin games.
5. Round costs.
numpy and the standard library only.
"""
import math
import sys

ELO = 347.4
R_SETUP, R_PAIRS, R_COLS = 0.897, 0.700, 0.732
FIGHTS = 109.7
VAR_KILL = 0.0878
PAIRS_W035 = (8.0, 24.3)
PAIRS_W150 = (21.5, 58.5)
GEN, MATCH = 2.47e-5, 1.49e-4
EXPLAINED_FIXED = 0.161
D_PSI = 7 * (1 + 2 * 32)   # 7 outcome features x (constant + two fighters' 32-dim token projections)


def main():
    print(f"1. D1 consequence-model fit, d = {D_PSI}, residual noise r_setup = {R_SETUP}")
    for n in (10_000, 30_000, 100_000, 300_000):
        excess = R_SETUP * D_PSI / n
        print(f"   N {n:>7,}: excess variance {excess:.4f} of Var(z) = {excess / EXPLAINED_FIXED:.3f} "
              f"of what the fixed columns explain; generation ${n * GEN:.2f}")

    print("\n2. P-dice: delta * s_e * (1[defender dies] - P_law), s_e = +-1 on half the fights")
    for n in (100_000,):
        info = n * FIGHTS * 0.5 * VAR_KILL
        se = math.sqrt(R_SETUP / info)
        for delta in (0.01, 0.02):
            print(f"   N {n:,}, delta {delta}: SE {se:.5f}; recovered coefficient 1 +- {2 * se / delta:.3f} (2 SE)")

    print("\n3. D2 paired arms, 3,200 corrected decisive games each (side A only switches)")
    for label, r in (("pairs + columns (ruling b yes)", R_PAIRS), ("setup only (ruling b no)", R_SETUP)):
        se_arm = ELO * math.sqrt(r / 3200)
        se_diff = math.sqrt(2) * se_arm
        print(f"   {label}: SE per arm {se_arm:.2f} Elo; real - anti SE {se_diff:.2f} Elo")
        for m in PAIRS_W035:
            # real switches where X prefers a1, anti where X prefers a0; on a symmetric split each arm
            # changes m / 2 decisions per side-game; real - anti = ELO * (m / 2) * (a_real - a_anti).
            a_min = 2 * se_diff / (ELO * m / 2)
            print(f"     m = {m:4.1f} attack pairs per side-game: 2-SE line on a_real - a_anti {a_min:.4f}, "
                  f"on the alignment (half of it) {a_min / 2:.4f}")
        print(f"     cost of both arms ${2 * 3200 * MATCH:.2f}")

    print("\n4. D3 targeted coins (p = 0.25 only where X prefers the runner-up), model-free forecast")
    for r in (R_COLS, R_SETUP):
        sigma2 = r / (0.25 * 0.75)
        for n in (100_000, 200_000):
            for ls in (8.0, 24.3, 58.5 / 2):
                se = ELO * math.sqrt(ls * sigma2 / (2 * n))
                print(f"   r {r:.3f} (sigma^2 {sigma2:.2f}), N {n:,} (${n * GEN:.2f}), "
                      f"{ls:4.1f} switches per side-game: forecast SE {se:.1f} Elo")

    print("\n5. round costs ($)")
    d1, d2, d3 = 100_000 * GEN, 2 * 3200 * MATCH, 200_000 * GEN
    adopt = 12_800 * MATCH + 800 * MATCH
    distill = 2.1 + 0.6
    print(f"   D1 {d1:.2f}  D2 {d2:.2f}  D3 {d3:.2f}  D4 adoption {adopt:.2f}  D5 distill {distill:.2f}")
    print(f"   round 1 {d1 + d2 + d3 + adopt + distill:.2f}; first decision read (D1 + D2) {d1 + d2:.2f}")
    later = d3 + d2 + adopt + distill
    print(f"   later round (dice refit inside D3's coin games) {later:.2f}")


if __name__ == "__main__":
    sys.exit(main())
