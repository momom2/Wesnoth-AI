"""Round 7, simplify stance: what each removal costs, under the round-6 Gaussian model and prices.

Reuses round6_synthesis_power.py (and through it round5_synthesis_power.py and
round5_simplify_power.py) unchanged: at band decisions the in-span advantage of the runner-up is
a ~ N(mu, v); a student fitted on labels of total information I predicts with error variance
eta = d_eff / I; Elo = 347.4 z near an even match. The model states what each outcome would mean;
it is not a forecast.

Measured inputs, read here from the outputs that hold them:
- round6_provenance_cells.out: the label residual of own1 self-play keeps 0.688 with the full
  setup cell known exactly (pooled cell share) and 0.839 with the seven columns alone (fitted
  leave-one-match-out, so on other matches); the pooled within-cell share 0.696, and the
  finite-sample term of a cell baseline fitted on N games, within x 1,512 / N;
- round6_simplify_pairs_raw.out: side-swapped pairs keep 0.847 of a game's variance;
- round6_provenance_prices.out: $2.51e-5 a one-network game in the pool (run_elo_batch at
  --jobs 120), $1.17e-4 a decisive game in the eval driver (median of 18 matches), box setup $0.018;
- round5_provenance_critique_knobs.out: z per unit of end_turn offset above -1.5 and below it,
  on all knob games and on the unselected stage-1 games;
- round6_provenance_closure.out: the recipe's tempo closure and its 95% bootstrap bands.

Derived here:
1. independent-game matches at the paired match's SE, their price and the known answer's window;
2. label information by batch when every coefficient is frozen before its batch is generated
   (batch 1: no cell baseline, columns fitted on the record) against the round-6 design (batch 1
   cross-fits its halves); effective batch counts, floors, the interim's null bound;
3. the budget of the simplified plan, the three-round price line, a null round's cost;
4. what deferring DPI costs: batches per round by kept ratio r, and whether a fourth round fits;
5. the coin plant read as a difference (planted minus unplanted labels on the same games): its SE
   in closed form, checked by simulation, with a wrong mover sign and a misaligned coin;
6. the end_turn closure's price in Elo when the offset is baked into the distillation target.
numpy and the standard library only.
"""
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import round5_simplify_power as P  # noqa: E402
import round6_synthesis_power as R6  # noqa: E402

RNG = np.random.default_rng(20261020)
CELLS = 1512
BATCH = R6.BATCH
BATCHES = R6.BATCHES
N_BAND = R6.N_BAND
GEN = R6.GEN_PRICE
EVAL = R6.EVAL_PRICE
SETUP = 0.018
DISTILL = 2.1
MATCH_PAIRED = R6.MATCH
P_COIN = 0.25


def read_cells():
    text = (HERE / "round6_provenance_cells.out").read_text()
    pooled = re.search(r"pooled, .*columns keep ([\d.]+), full-cell share ([\d.]+), within-cell ([\d.]+)", text)
    own = re.search(r"own1 self-play, .*columns keep ([\d.]+),", text)
    own_pooled = re.search(r"own1 self-play with the pooled cell share: within-cell ([\d.]+)", text)
    return {"within_pooled": float(pooled.group(3)), "own_columns_only": float(own.group(1)),
            "own_within": float(own_pooled.group(1))}


def read_knobs():
    text = (HERE / "round5_provenance_critique_knobs.out").read_text()
    blocks = re.findall(r"reduced form: z per unit of offset below -1.5 ([+-][\d.]+) \+- ([\d.]+), "
                        r"above ([+-][\d.]+) \+- ([\d.]+)", text)
    first, stage1 = blocks[0], blocks[-1]
    return {"all": tuple(float(x) for x in first), "stage1": tuple(float(x) for x in stage1)}


def read_closure():
    text = (HERE / "round6_provenance_closure.out").read_text()
    bands = re.findall(r"closure, .*: ([\d.]+) \(95% bootstrap ([\d.]+) to ([\d.]+)\)", text)
    return [tuple(float(x) for x in b) for b in bands]


def label_r(cells, batch_index, frozen):
    """Kept ratio of batch k's labels (k from 1)."""
    within, own = cells["within_pooled"], cells["own_within"]
    if batch_index == 1:
        return cells["own_columns_only"] if frozen else own + within * CELLS / (BATCH / 2)
    return own + within * CELLS / (BATCH * (batch_index - 1))


def effective_batches(cells, k, frozen, r_ref):
    return sum(r_ref / label_r(cells, j, frozen) for j in range(1, k + 1))


def bisect(fn):
    return R6.bisect(fn)


def section1():
    print("1. matches as independent games in the existing driver, sized to the paired match's SE")
    se_paired = R6.match_se(MATCH_PAIRED)
    n_ind = MATCH_PAIRED / R6.PAIRS_RAW
    n_round = int(math.ceil(n_ind / 100.0) * 100)
    se_ind = P.ELO_PER_Z / math.sqrt(n_round)
    extra = (n_round - MATCH_PAIRED) * GEN
    print(f"  paired {MATCH_PAIRED:,} games: SE {se_paired:.2f} Elo, ${MATCH_PAIRED * GEN:.2f}")
    print(f"  independent at equal SE: {n_ind:,.0f} -> {n_round:,} games, SE {se_ind:.2f} Elo, "
          f"${n_round * GEN:.2f} (+${extra:.3f} a match)")
    print(f"  independent at {MATCH_PAIRED:,} games: SE {P.ELO_PER_Z / math.sqrt(MATCH_PAIRED):.2f} Elo")
    n_matches = 8
    print(f"  one-network matches in the plan: {n_matches} (interim and adoption in round 1; interim, "
          f"adoption, claim in rounds 2 and 3): +${n_matches * extra:.2f} over the plan, "
          f"against a mirror null of 3,200 games (${3200 * GEN:.2f}) no longer run")
    se_known = P.ELO_PER_Z / math.sqrt(6400)
    print(f"  attack known answer, 6,400 independent games: SE {se_known:.2f} Elo, window "
          f"+-{2 * math.sqrt(16 ** 2 + se_known ** 2):.1f} Elo around -78 (16 = the prediction's SE)")
    return n_round, se_ind


def section2(cells, se_m):
    r_ref = R6.LABEL_R
    print(f"\n2. label information by batch (r_ref {r_ref}), coefficients frozen before each batch "
          f"against batch 1 cross-fitted by halves")
    for frozen in (False, True):
        rs = [label_r(cells, k, frozen) for k in range(1, BATCHES + 1)]
        name = "frozen (batch 1: b = 0, beta from the record)" if frozen else "round 6 (batch 1 cross-fits halves)"
        print(f"  {name}: r by batch " + " ".join(f"{r:.3f}" for r in rs) +
              f"; effective batches at 3: {effective_batches(cells, 3, frozen, r_ref):.2f}, "
              f"at 8: {effective_batches(cells, BATCHES, frozen, r_ref):.2f}")
    inp = R6.inputs_at(r_ref)
    print(f"  floors (gain 2.84 SE = {2.84 * se_m:.1f} Elo at batch 8) and the interim's null bound "
          f"(stop iff student 3 reads below 0), match SE {se_m:.2f}:")
    for d_eff in (200, 776):
        for mu in (0.0, -0.0018):
            row = []
            for label, b3, b8 in (("idealised 8", 3.0, 8.0),
                                  ("round 6", effective_batches(cells, 3, False, r_ref),
                                   effective_batches(cells, 8, False, r_ref)),
                                  ("frozen", effective_batches(cells, 3, True, r_ref),
                                   effective_batches(cells, 8, True, r_ref))):
                floor = bisect(lambda x: R6.gain(inp, x, mu, d_eff, b8)[0] >= 2.84 * se_m)

                def power(x):
                    g3, _ = R6.gain(inp, x, mu, d_eff, b3)
                    g8, _ = R6.gain(inp, x, mu, d_eff, b8)
                    return (1 - P.Phi(-g3 / se_m)) * P.Phi(g8 / se_m - 2)
                bound = bisect(lambda x: power(x) >= 0.8)
                row.append(f"{label}: floor {floor:.4f}, bound {bound:.4f}")
            print(f"    d_eff {d_eff} mu {mu:+.4f}: " + "; ".join(row))


def plan_cost(match_games, batches_per_round, extra_s0=0.0, rounds=3):
    m = match_games * GEN
    s0 = SETUP + 6400 * GEN + extra_s0
    r1 = batches_per_round[0] * BATCH * GEN + 2 * m
    later = [b * BATCH * GEN + 3 * m for b in batches_per_round[1:rounds]]
    s4 = DISTILL + 12_800 * EVAL
    return s0, r1, later, s4


def section3(match_games):
    print("\n3. budget of the simplified plan")
    m = match_games * GEN
    s0, r1, later, s4 = plan_cost(match_games, [8, 8, 8])
    total = s0 + r1 + sum(later) + s4
    print(f"  S0 (setup, attack known answer 6,400 games): ${s0:.2f}")
    print(f"  S1 + S2 (8 batches, interim, adoption): ${r1:.2f}")
    print(f"  S3, S5 (8 batches, interim, adoption, claim): ${later[0]:.2f} each")
    print(f"  S4 (distillation ${DISTILL}, one two-network match of 12,800 games at the eval price): ${s4:.2f}")
    print(f"  total ${total:.2f}; left of $85 ${85 - total:.2f}")
    fixed = total - 3 * BATCHES * BATCH * GEN
    line = (85 - fixed) / (3 * BATCHES * BATCH)
    print(f"  three rounds fit iff price x (1 + overhead) <= ${line:.2e} a coin game; at $2.51e-5 overhead up to "
          f"{line / GEN - 1:.3f}; the 1.41x host spread {'fits' if line >= 1.41 * GEN else 'does not fit'}")
    stop = s0 + 3 * BATCH * GEN + m
    through = s0 + BATCHES * BATCH * GEN + 2 * m
    expected = stop + 0.5 * ((BATCHES - 3) * BATCH * GEN + m)
    print(f"  a null round 1 stopped at the interim ${stop:.2f}; run through ${through:.2f}; "
          f"expected at a null (P(stop) 0.50) ${expected:.2f}; with no interim ${s0 + BATCHES * BATCH * GEN + m:.2f}")
    print("  round 6 design for comparison: $73.58 plus a $1.61 reserve (round6_synthesis_power.out)")
    return total


def section4(match_games):
    print("\n4. deferring DPI: batches a round needs at kept ratio r (ceil(8 r / 0.688)) and a fourth round")
    m = match_games * GEN
    for r in (0.688, 0.602, 0.516, 0.430, 0.344):
        b = math.ceil(round(8 * r / R6.LABEL_R, 6))
        s0, r1, later, s4 = plan_cost(match_games, [b, b, b], extra_s0=0.424)
        total = s0 + r1 + sum(later) + s4
        fourth = b * BATCH * GEN + 3 * m
        # overhead at which the fourth round still fits: coin games scale with (1 + o)
        coin = 4 * b * BATCH * GEN
        fixed = total + fourth - coin
        o_max = (85 - fixed) / coin - 1
        s0d, r1d, laterd, s4d = plan_cost(match_games, [8, 8, b], extra_s0=0.424)
        total_d = s0d + r1d + sum(laterd) + s4d
        coin_d = (8 + 8 + 2 * b) * BATCH * GEN
        o_max_d = (85 - (total_d + fourth - coin_d)) / coin_d - 1
        print(f"  r <= {r:.3f}: {b} batches; DPI from round 1: three rounds ${total:.2f}, a fourth fits at overhead "
              f"<= {o_max:+.3f}; DPI from round 3 on: a fourth fits at overhead <= {o_max_d:+.3f}")


def plant_readings(n_games, n_side, p, rng, wrong_sign_b=False, misalign=False):
    """Slope of (T' - T)/delta on g per side; T' - T = s_i w_i (z' - z), z' - z = sum_j s_j (D_j - P) g_j."""
    s = np.concatenate([np.ones(n_side), -np.ones(n_side)])
    g = rng.standard_normal((n_games, 2 * n_side))
    d = (rng.random((n_games, 2 * n_side)) < p).astype(float)
    inc = ((d - p) * g * s).sum(1, keepdims=True)                       # built from the record
    d_label = np.roll(d, 1, axis=1) if misalign else d
    s_label = s.copy()
    if wrong_sign_b:
        s_label[n_side:] = 1.0
    w = (d_label - p) / (p * (1 - p))
    y = s_label * w * inc
    out = []
    for side in (slice(0, n_side), slice(n_side, 2 * n_side)):
        gs, ys = g[:, side], y[:, side]
        out.append(float((gs * ys).sum() / (gs * gs).sum()))
    pooled = float((g * y).sum() / (g * g).sum())
    return out + [pooled]


def section5():
    print("\n5. the coin plant read as a difference (game noise cancels; delta cancels)")
    n = BATCH
    print(f"  closed form for Gaussian W_g: SE per side sqrt(3/n) = {math.sqrt(3 / n):.4f}, "
          f"pooled sqrt(2/n) = {math.sqrt(2 / n):.4f} at n = {n:,} games")
    n_sim, n_side, reps = 2_000, 76, 200
    sims = np.array([plant_readings(n_sim, n_side, P_COIN, RNG) for _ in range(reps)])
    scale = math.sqrt(n_sim / n)
    print(f"  simulation ({reps} x {n_sim:,} games, {n_side} band decisions a side, p {P_COIN}): mean "
          f"side A {sims[:, 0].mean():.3f}, side B {sims[:, 1].mean():.3f}, pooled {sims[:, 2].mean():.3f}; "
          f"SE at {n:,} games: side {sims[:, 0].std() * scale:.4f}, pooled {sims[:, 2].std() * scale:.4f}")
    wrong = np.array([plant_readings(n_sim, n_side, P_COIN, RNG, wrong_sign_b=True) for _ in range(20)])
    mis = np.array([plant_readings(n_sim, n_side, P_COIN, RNG, misalign=True) for _ in range(20)])
    print(f"  side B's sign wrong in the label code: side A {wrong[:, 0].mean():.3f}, side B {wrong[:, 1].mean():.3f}")
    print(f"  coin read one decision off: side A {mis[:, 0].mean():.3f}, side B {mis[:, 1].mean():.3f}")


def section6():
    print("\n6. end_turn offset baked into the distillation target: Elo price of the closure c "
          "(effective offset -1.5 c), from the knob slopes")
    knobs = read_knobs()
    closure = read_closure()
    print("  tempo closure bands read: " + "; ".join(f"{c:.2f} ({lo:.2f} to {hi:.2f})" for c, lo, hi in closure))
    for name, (below, below_se, above, above_se) in knobs.items():
        cells = []
        for c in (0.60, 0.82, 1.00, 1.30):
            units = 1.5 * (1 - c)
            if units > 0:
                z, se = above * units, above_se * units
            else:
                z, se = below * (-units), below_se * (-units)
            cells.append(f"c {c:.2f}: {P.ELO_PER_Z * z:+.1f} +- {P.ELO_PER_Z * abs(se):.1f}")
        print(f"  {name} knob games (z per unit above -1.5 {above:+.4f} +- {above_se}, below {below:+.4f} "
              f"+- {below_se}): " + "; ".join(cells) + " Elo")


def main():
    cells = read_cells()
    print(f"inputs: own1 within-cell (pooled cell share) {cells['own_within']}, columns only (record-fitted) "
          f"{cells['own_columns_only']}, pooled within-cell {cells['within_pooled']}; pairs keep {R6.PAIRS_RAW}; "
          f"${GEN:.2e} a pool game, ${EVAL:.2e} an eval game")
    match_games, se_m = section1()
    section2(cells, se_m)
    section3(match_games)
    section4(match_games)
    section5()
    section6()


if __name__ == "__main__":
    main()
