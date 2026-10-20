"""Round 8, simplify stance: what removing the interim match, the S0(a) replay and gamma-V costs,
and what reading the coin plant at the smoke buys.

Model: round 5's stated Gaussian model, reused unchanged through round6_synthesis_power.py
(gain, inputs_at, bisect) and round7_simplify_power.py (label kept ratio by batch under frozen
coefficients, effective batch counts). At band decisions the in-span advantage of the runner-up is
a ~ N(mu, v); a student on labels of total information I predicts with error variance d_eff / I.
The model states what each outcome would mean; it is not a forecast.

Measured inputs, read where they are recorded:
- steady pool price $1.912e-5 a one-network game (round7_provenance_price_tail.out, ref_p120x6),
  host spread 1.41x, eval driver $1.17e-4 a decisive game, box setup $0.018
  (round6_provenance_prices.out);
- label residual 0.688 (cell plus columns), 0.839 (columns alone, record-fitted), within-cell
  terms (round6_provenance_cells.out, read by round7_simplify_power.read_cells);
- distillation recipe $2.1 (docs/memory_own_trajectories_prereg_20261011.md, an upper bound);
- Elo per unit z 800 / ln 10.
Choices [C], unchanged from current_design.md: 8 batches of 100,000 coin games, 37,800-game
one-network matches (SE 1.79), 12,800-game two-network matches, p 0.25, d 772.
numpy and the standard library only.
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import round5_simplify_power as P  # noqa: E402
import round6_synthesis_power as R6  # noqa: E402
import round7_simplify_power as R7S  # noqa: E402

POOL = 1.912e-5
SPREAD = 1.41
EVAL = 1.17e-4
SETUP = 0.018
DISTILL = 2.1
BUDGET = 85.0
BATCH, BATCHES = 100_000, 8
MATCH = 37_800
SE_M = P.ELO_PER_Z / math.sqrt(MATCH)
SMOKE = 2_000
ATTACK = 6_400
TWO_NET = 12_800
D_EFF = 772
R_REF = R6.LABEL_R          # 0.688
RNG = np.random.default_rng(20261020)


def eff(cells, k):
    return R7S.effective_batches(cells, k, True, R_REF)


def powers(inp, sd, mu, b3, b8):
    g3, _ = R6.gain(inp, sd, mu, D_EFF, b3)
    g8, _ = R6.gain(inp, sd, mu, D_EFF, b8)
    p_stop = P.Phi(-g3 / SE_M)
    without = P.Phi(g8 / SE_M - 2)
    return without, (1 - p_stop) * without, p_stop, g3, g8


def section1(cells):
    inp = R6.inputs_at(R_REF)
    b3, b8 = eff(cells, 3), eff(cells, 8)
    print(f"1. the interim match (student 3 against the incumbent, stop iff its estimate < 0), "
          f"frozen coefficients: effective batches {b3:.2f} at 3, {b8:.2f} at 8; d {D_EFF}; match SE {SE_M:.2f}")
    for mu in (0.0, -0.0018):
        floor = R6.bisect(lambda x: powers(inp, x, mu, b3, b8)[0] >= 0.8)
        bound = R6.bisect(lambda x: powers(inp, x, mu, b3, b8)[1] >= 0.8)
        print(f"  mu {mu:+.4f}: 0.8-power sd without the interim {floor:.4f}, with it {bound:.4f}")
        for sd in (0.0, 0.5 * floor, 0.8 * floor, floor, 1.2 * floor, 1.5 * floor):
            w0, w1, ps, g3, g8 = powers(inp, sd, mu, b3, b8)
            print(f"    sd {sd:.4f}: student 3 {g3:5.1f} Elo, student 8 {g8:5.1f}; P(stop) {ps:.3f}; "
                  f"adoption power without {w0:.3f}, with {w1:.3f} (lost {w0 - w1:.3f})")
    shape_free = P.Phi(-math.sqrt(b3 / b8) * 2.84)
    print(f"  shape-free false stop where the ceiling's premise holds (student 8 at 80% power, "
          f"gain >= sqrt({b3:.2f}/{b8:.2f}) of it at student 3): P <= {shape_free:.3f}")


def plan(pool, interim):
    m = MATCH * pool
    coin = BATCHES * BATCH * pool
    two = TWO_NET * EVAL
    k = 1 if interim else 0
    st = {
        "S0 setup + attack known answer + coin smoke (smoke counted in batch 1)": SETUP + ATTACK * pool,
        "S1 round 1 coins (8 batches)" + (" + interim" if interim else ""): coin + k * m,
        "S2 adoption = claim": m,
        "E reserve": (2 if interim else 3) * m,
        "S3 round 2 (8 batches, adoption vs pi_1, claim vs own1)" + (" + interim" if interim else ""):
            coin + (2 + k) * m,
        "S4 reset (distillation, two-network match)": DISTILL + two,
        "S5 round 3 (8 batches, adoption vs N, two-network claim)" + (" + interim" if interim else ""):
            coin + (1 + k) * m + two,
    }
    return st, m, coin


def section2():
    print("\n2. budget, steady price; the current design's S0(a) replay ($0.043) is gone in both columns "
          "so the difference is the interim alone")
    for pool, label in ((POOL, "measured host"), (POOL * SPREAD, "host 1.41x slower")):
        for interim in (True, False):
            st, m, coin = plan(pool, interim)
            total = sum(st.values())
            fixed = total - 3 * coin
            line = (BUDGET - fixed) / (3 * BATCHES * BATCH)
            s0 = st["S0 setup + attack known answer + coin smoke (smoke counted in batch 1)"]
            name = "with interim" if interim else "no interim  "
            if interim:
                stop = s0 + 3 * BATCH * pool + m
                through = s0 + coin + 2 * m
                null = f"null round 1: stopped ${stop:.2f}, through ${through:.2f}, expected ${0.5 * (stop + through):.2f}"
            else:
                null = f"null round 1: ${s0 + coin + m:.2f}"
            print(f"  {label}, {name}: total ${total:.2f}, left ${BUDGET - total:.2f}; three rounds fit iff "
                  f"price x (1 + overhead) <= ${line:.3e} (room {line / pool - 1:+.3f}); {null}")
            s5 = [v for k2, v in st.items() if k2.startswith("S5")][0]
            fixed2 = total - s5 - 2 * coin
            print(f"    two rounds (S5 dropped, S4 kept) fit iff price x (1 + overhead) <= "
                  f"${(BUDGET - fixed2) / (2 * BATCHES * BATCH):.3e}")
            if not interim and pool == POOL:
                print("    " + "; ".join(f"{k.split(' (')[0]} ${v:.2f}" for k, v in st.items()))


def section3(cells):
    print("\n3. gamma-V deferred: every floor the design states is computed at r 0.688, without V")
    inp0 = R6.inputs_at(R_REF)
    b8 = eff(cells, 8)
    st, m, coin = plan(POOL, False)
    left = BUDGET - sum(st.values())
    extra = left / (BATCH * POOL)
    print(f"  the slack ${left:.2f} buys {extra:.1f} more batches, {extra / 3:.1f} a round over three rounds")
    for k in (1.0, 0.875, 0.75, 0.5):
        inp = R6.inputs_at(R_REF * k)
        cells_txt = []
        for mu in (0.0, -0.0018):
            f = R6.bisect(lambda x: P.Phi(R6.gain(inp, x, mu, D_EFF, b8)[0] / SE_M - 2) >= 0.8)
            cells_txt.append(f"mu {mu:+.4f} floor {f:.4f}")
        same = BATCHES * (1 / k - 1)
        print(f"  if V kept a share {k:.3f} of the residual: " + ", ".join(cells_txt) +
              f"; the same floors from {same:.1f} more batches a round without V")
    f0 = R6.bisect(lambda x: P.Phi(R6.gain(inp0, x, 0.0, D_EFF, b8)[0] / SE_M - 2) >= 0.8)
    b_more = b8 * (1 + extra / 3 / BATCHES)
    f1 = R6.bisect(lambda x: P.Phi(R6.gain(inp0, x, 0.0, D_EFF, b_more)[0] / SE_M - 2) >= 0.8)
    print(f"  spending the slack on batches instead (mu 0): floor {f0:.4f} -> {f1:.4f}; "
          f"V matches that iff it keeps <= {BATCHES / (BATCHES + extra / 3):.3f}")


def plant_partial(n_games, n_side, p, frac, rng):
    """Slope per side when a fraction frac of band decisions has its coin read one decision off."""
    s = np.concatenate([np.ones(n_side), -np.ones(n_side)])
    g = rng.standard_normal((n_games, 2 * n_side))
    d = (rng.random((n_games, 2 * n_side)) < p).astype(float)
    inc = ((d - p) * g * s).sum(1, keepdims=True)
    bad = rng.random((n_games, 2 * n_side)) < frac
    d_label = np.where(bad, np.roll(d, 1, axis=1), d)
    w = (d_label - p) / (p * (1 - p))
    y = s * w * inc
    side = slice(0, n_side)
    return float((g[:, side] * y[:, side]).sum() / (g[:, side] ** 2).sum())


def section4():
    print("\n4. the coin plant read on the smoke's games as well as on batch 1 (same offline code)")
    n_side = 76
    for n in (SMOKE, BATCH):
        print(f"  {n:,} games: closed-form SE per side sqrt(3/n) = {math.sqrt(3 / n):.4f}")
    reps = 200
    sims = np.array([R7S.plant_readings(SMOKE, n_side, 0.25, RNG)[0] for _ in range(reps)])
    se = sims.std()
    print(f"  simulated at {SMOKE:,} games ({reps} reps): mean {sims.mean():.3f}, SE {se:.4f}")
    print(f"  at the smoke a wrong side sign (-1.005) sits {(1 + 1.005) / se:.0f} SE from 1, "
          f"a coin read one decision off (0.016) {(1 - 0.016) / se:.0f} SE")
    for frac in (0.02, 0.05, 0.10, 0.20):
        r = np.mean([plant_partial(SMOKE, n_side, 0.25, frac, RNG) for _ in range(100)])
        print(f"  coins misread on a fraction {frac:.2f} of decisions: slope {r:.3f}; "
              f"{(1 - r) / se:.1f} SE at the smoke, {(1 - r) / math.sqrt(3 / BATCH):.1f} SE on batch 1")


def main():
    cells = R7S.read_cells()
    print(f"inputs: steady price ${POOL:.3e}; r {R_REF} (cell + columns), batch 1 {cells['own_columns_only']} "
          f"(columns alone); match {MATCH:,} games, SE {SE_M:.2f} Elo")
    section1(cells)
    section2()
    section3(cells)
    section4()


if __name__ == "__main__":
    main()
