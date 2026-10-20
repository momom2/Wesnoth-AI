"""Round 6, provenance stance: what round 1 detects when matches run in the generation pool.

Reuses the stated Gaussian model of round5_synthesis_power.py (and through it
round5_simplify_power.py) unchanged: at band decisions the in-span advantage of the runner-up is
a ~ N(mu, v); the student fitted on N labels predicts f = a + e, e ~ N(0, eta),
eta = sigma^2 d_eff / N; the calibrated rule's gain per band decision is per_decision(v, eta, mu).
The model is used only to state what each outcome would mean, never as a forecast.

Changed inputs, each measured:
- match price: a corrected match of the candidate (own1 + its switch head: one network) against
  own1 runs in the generation pool at $2.51e-5 a game (ref_p120x6) instead of the eval driver's
  $1.17e-4 median (round6_provenance_prices.out), so a 25,600-game match costs $0.64;
- match SE: side-swapped pairs + seven columns keep 0.700 (round5_provenance_critique_pairs.out);
- label residual: the full-cell setup baseline keeps 0.688 in own1 self-play against the additive
  baseline's 0.732 (round6_provenance_cells.out).

Derived here:
1. floors (match-only: the final student's gain reaches 2.84 match SEs) at 12,800 / 25,600 /
   51,200-game adoption matches, against the design's 12,800-game eval-driver match;
2. the interim match after batch 4 (student 4 against own1, generation pool) and the stop rule
   "stop iff its estimate is below 0", P(stop) by spread, against the design's progressive-forecast
   stop (P 0.15 at a mu-0 null, round5_synthesis_power.out);
3. what the matches identify when the forecast misses. With J quadratic along the switch set,
   J(alpha) = g alpha + A alpha^2 (g the gradient at own1, A the curvature), and the label sum
   L = g + p 1_band'H 1_S + p 1_opp'H 1_S (labels are partial derivatives of the multilinear
   value at the coin point, round6_provenance.md section 2.4): from J(1) and J(1/2),
   g = 4 J(1/2) - J(1), A = 2 J(1) - 4 J(1/2), and the labels' offset C = L - g; their SEs at
   25,600-game matches and the forecast's SE 4.6 (4.7 at r 0.732, round5_synthesis_power.out,
   scaled by sqrt(0.688 / 0.732));
4. the correction term's SE at 25,600 games (1.56 Elo at 12,800, round5_simplify_critique_extra.out).
Standard library only.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import round5_simplify_power as P  # noqa: E402
import round5_synthesis_power as S  # noqa: E402

PAIRS_R = 0.700
LABEL_R = 0.688
GEN_PRICE = 2.51e-5
EVAL_PRICE_MEDIAN = 1.17e-4
BATCH = 100_000
N_BAND = S.N_BAND


def match_se_elo(games):
    return P.ELO_PER_Z * math.sqrt(PAIRS_R / games)


def gain_elo(inputs, sd, mu, d_eff, batches, p=0.25):
    s2 = inputs["var_z"] * inputs["r_R_own1"] / (p * (1 - p))
    eta = s2 * d_eff / (batches * 2 * N_BAND * BATCH)
    g, sw = S.per_decision(sd * sd, eta, mu)
    return P.ELO_PER_Z * N_BAND * g, N_BAND * sw


def floor_sd(inputs, mu, d_eff, batches, target_elo):
    lo, hi = 1e-6, 0.05
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        ok = gain_elo(inputs, mid, mu, d_eff, batches)[0] >= target_elo
        lo, hi = (lo, mid) if ok else (mid, hi)
    return hi


def main():
    base = P.read_inputs()
    old = dict(base)
    new = dict(base, r_R_own1=LABEL_R)
    print(f"label residual: design {old['r_R_own1']:.3f}, full cell {LABEL_R:.3f}; "
          f"sigma^2 at p 0.25: {old['var_z'] * old['r_R_own1'] / 0.1875:.2f} -> {new['var_z'] * LABEL_R / 0.1875:.2f}")
    print("\n1. match-only floors, 8 batches (sd of the in-span near-tie spread whose gain reaches 2.84 match SEs)")
    rows = [("design: 12,800 eval-driver games, r 0.732", old, 12_800, EVAL_PRICE_MEDIAN)]
    for games in (12_800, 25_600, 51_200):
        rows.append((f"{games:,} generation-pool games, r 0.688", new, games, GEN_PRICE))
    for name, inp, games, price in rows:
        se = match_se_elo(games)
        target = 2.84 * se
        cells = []
        for d_eff in (200, 776):
            for mu in (0.0, -0.0018):
                cells.append(f"d{d_eff} mu{mu:+.4f}: {floor_sd(inp, mu, d_eff, 8, target):.4f}")
        print(f"  {name}: SE {se:.2f} Elo, ${games * price:.2f}, adoptable gain {target:.1f} Elo; " + "; ".join(cells))
    print("\n2. interim match after batch 4 (student 4 vs own1, generation pool), d_eff 776")
    for games in (12_800, 25_600):
        se = match_se_elo(games)
        print(f"  {games:,} games, SE {se:.2f} Elo, ${games * GEN_PRICE:.2f}: stop iff the estimate is below 0")
        for mu in (0.0, -0.0018):
            for sd in (0.0, 0.001, 0.0015, 0.002, 0.003, 0.005):
                g4, sw4 = gain_elo(new, sd, mu, 776, 4)
                g8, sw8 = gain_elo(new, sd, mu, 776, 8)
                p_stop = P.Phi(-g4 / se)
                print(f"    mu {mu:+.4f} sd {sd:.4f}: student 4 gains {g4:5.1f} Elo ({sw4:4.1f} switches), student 8 "
                      f"{g8:5.1f}; P(stop) {p_stop:.2f}; expected batches bought {4 + 4 * (1 - p_stop):.1f}")
    print("  design's stop (progressive forecast of students 1-3, upper bound under 5.3 Elo): "
          "P 0.15 at mu 0 sd 0, 0.04 at sd 0.002 (round5_synthesis_power.out)")
    print("\n3. identification from play when the forecast misses (25,600-game matches)")
    se_m = match_se_elo(25_600)
    se_f = 4.7 * math.sqrt(LABEL_R / 0.732)
    print(f"  match SE {se_m:.2f}, forecast SE {se_f:.2f} Elo")
    print(f"  g = 4 J(1/2) - J(1): SE sqrt(17) se_m = {math.sqrt(17) * se_m:.1f}")
    print(f"  A = 2 J(1) - 4 J(1/2): SE sqrt(20) se_m = {math.sqrt(20) * se_m:.1f}")
    print(f"  C = L - g (labels' offset: opponent coins, non-exchangeable curvature): "
          f"SE {math.sqrt(se_f ** 2 + 17 * se_m ** 2):.1f}")
    print(f"  the design's alpha* = -G1 / (2 G2) with G1 = L: L estimates g + A when p = q/2 "
          f"(midpoint), so it reads -(g + A) / (2 A); e.g. g 20, A -15 Elo: true {20 / 30:.2f}, "
          f"design {-(20 - 15) / (2 * -15):.2f}")
    print("\n4. correction term's SE: 1.56 Elo at 12,800 games -> "
          f"{1.56 * math.sqrt(12_800 / 25_600):.2f} at 25,600, {1.56 * math.sqrt(12_800 / 51_200):.2f} at 51,200")


if __name__ == "__main__":
    main()
