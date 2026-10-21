"""Round 9, challenge stance: the numbers the factorial self-play regimen rests on.

Every game is a mirrored factorial over K policy blocks: seat A plays block k
at level eps_k in {-1, +1} (on / off), seat B at -eps_k, the signs taken from a
Hadamard design keyed by the game index. The block effect is read as

    delta_hat_k = (1/N) sum_g R_g eps_gk,   R_g = z_g - b(cell_g) - beta . S_g,

with b and beta frozen before the batch (round9_simplify_residual.out).

Sections:
 1. Inputs, each measured, with its record.
 2. SE of one block's effect at the batch sizes used, all K blocks at once.
 3. Games for equal SE on a block's play effect: coin welfare against the
    factorial, n V_T / (2 sigma_R^2) for a block switching n decisions per
    side-game; the blocks of round 1.
 4. Windows of the first factorial's checks: the attack hook (S0(a)'s
    -41.3 +- 4.4) and the per-faction recruit blocks' first-order predictions.
 5. Costs per stage and the budget: rounds that fit.
 6. Monte Carlo of the estimator with binary results: bias, SE, the aliasing
    of within-seat two-block interactions (predicted a_jk / 2 into each main
    effect), the planted copy's exact recovery under a Hadamard design, and the
    split-half selection followed by a fresh confirmation match: its reading
    against its prediction, and false adoption when every block is null.

numpy and the standard library only; a few seconds.
"""
from __future__ import annotations

import math

import numpy as np

ELO_PER_Z = 347.4  # current_design.md: Elo per unit of mean z near 0

# 1. Inputs -----------------------------------------------------------------
SIGMA_R2 = 0.699     # round9_simplify_residual.out: frozen beta + shrunk cell, out of sample, coin games
SIGMA_R2_OWN2 = 0.692  # same file: the frozen coefficients on own2_64 vs own2_0 (800 games, SE ~0.04)
V_T = 4.01           # round9_simplify_labels.out: E[T^2] with the frozen baselines
PRICE_COIN = 2.849e-5   # round9_simplify_labels.out: coin games all-in (walk, rows, plant, readout), $0.577/h
PRICE_POOL = 2.598e-5   # same: one-network pool games, $0.577/h
PRICE_TWO = 8.848e-5    # same: two-network decisive game (own2 vs own1 box)
DISTILL_HOURS = 4.8     # round9_provenance_budget.out: own-trajectory distillation box-hours
RATE = 0.577            # $/h of the measured host
BUDGET_LEFT = 82.34     # seed section 7: $85 - $2.66
HOOK = (-41.3, 4.4)     # seed section 7: S0(a), hooked seat, 6,400 games, raw z
# round9_provenance_recruit_cells.out section 6: rule (a) (cell mean > 0) cross-fitted welfare by
# mover faction, Elo per side-game of all factions, and that faction's recruit-pair band decisions
# per side-game; rule (a) switches 0.602 of these decisions (section 3).
FACTION_RULE = {
    "Drakes": (3.6, 1.4, 0.90), "Knalgan Alliance": (3.4, 1.6, 0.76), "Loyalists": (3.0, 1.0, 0.42),
    "Northerners": (5.5, 1.3, 0.78), "Rebels": (-1.7, 1.2, 0.61), "Undead": (9.2, 1.7, 0.82),
}
RULE_A_TOTAL = (23.1, 3.4)
SWITCH_SHARE = 0.602
R_ADDITIVE = (22.9, 3.2, 2.21)  # the additive 43-value rule: Elo, SE, switches per side-game


def se_elo(n_games: int, s2: float = SIGMA_R2) -> float:
    return math.sqrt(s2 / n_games) * ELO_PER_Z


def section2():
    print("2. SE of one block's play effect (one seat on against off), all blocks of the design at once")
    for n in (10_000, 37_800, 100_000):
        print(f"   {n:7d} games: {se_elo(n):.2f} Elo (luck- and cell-corrected); a dedicated PURE match of"
              f" the same size, raw z: {math.sqrt(1 / n) * ELO_PER_Z:.2f} Elo for one block")


def section3():
    print("\n3. Games for equal SE on a block's play effect: coin welfare / factorial = n V_T / (2 sigma_R^2)")
    k = V_T / (2 * SIGMA_R2)
    print(f"   n V_T / (2 sigma_R^2) = {k:.2f} n  (n = decisions the block switches per side-game)")
    name, (e, s, n) = "additive recruit rule R", R_ADDITIVE  # noqa: F841 (an analysis script kept as run)
    print(f"   {name:28s} n {n:.2f}: ratio {k * n:.2f}")
    for f, (w, ws, d) in FACTION_RULE.items():
        n = SWITCH_SHARE * d
        print(f"   rule (a), {f:18s} n {n:.2f}: ratio {k * n:.2f}")
    print("   (the factorial reads every block of a batch at that SE; coins read every rule at once too,"
          " but first order and under the coin policy)")


def section4():
    print("\n4. Checks of the first factorials")
    n_smoke = 10_000
    s = se_elo(n_smoke)
    lo, hi = HOOK[0] - 3 * math.hypot(HOOK[1], s), HOOK[0] + 3 * math.hypot(HOOK[1], s)
    print(f"   attack hook as one block of a {n_smoke}-game factorial on own1: SE {s:.2f}; window at 3 SE of the"
          f" difference [{lo:.1f}, {hi:.1f}] Elo; a seat swap reads about +41")
    n1 = 37_800
    s1 = se_elo(n1)
    print(f"   per-faction recruit blocks on own1, {n1} games, block SE {s1:.2f} Elo; first-order prediction and"
          " the transfer test |reading - prediction| < 3 SE of the difference:")
    tot_var = 0.0
    for f, (w, ws, d) in FACTION_RULE.items():
        sd = math.hypot(ws, s1)
        print(f"     {f:18s} predicted {w:+.1f} +- {ws:.1f}; SE of the difference {sd:.2f}; miss line +-{3 * sd:.1f}")
        tot_var += s1 ** 2
    sd_sum = math.sqrt(RULE_A_TOTAL[1] ** 2 + tot_var)
    print(f"   sum of the six readings against rule (a)'s {RULE_A_TOTAL[0]:+.1f} +- {RULE_A_TOTAL[1]:.1f}:"
          f" SE of the difference {sd_sum:.2f}, miss line +-{2 * sd_sum:.1f} at 2 SE")
    print(f"   a reading of 0 for all six (labels do not carry to play) sits {RULE_A_TOTAL[0] / sd_sum:.1f} SE"
          " below the prediction")


def section5():
    print("\n5. Costs (measured host, $0.577/h) and the budget")
    smoke = 10_000 * PRICE_POOL
    f1 = 37_800 * PRICE_POOL
    fact = 100_000 * PRICE_COIN
    conf = 37_800 * PRICE_POOL
    claim = 12_800 * PRICE_TWO
    distill = DISTILL_HOURS * RATE + claim
    round_cost = fact + conf
    print(f"   F1a smoke 10,000 games ${smoke:.2f}; F1b own1 factorial 37,800 games ${f1:.2f}")
    print(f"   one round: factorial with band coins 100,000 games ${fact:.2f} + confirmation 37,800 pool games"
          f" ${conf:.2f} = ${round_cost:.2f} ({100_000 / 20_254 + 37_800 / 22_341:.1f} h)")
    print(f"   PURE claim against own1, two networks, 12,800 games ${claim:.2f}; distillation and its"
          f" closure match ${distill:.2f}")
    fixed = smoke + f1 + 3 * claim + 3 * distill
    rounds = int((BUDGET_LEFT - fixed) // round_cost)
    print(f"   fixed: F1 + 3 claims + 3 distillations ${fixed:.2f}; rounds that fit in ${BUDGET_LEFT:.2f}: {rounds}"
          f" (${fixed + rounds * round_cost:.2f})")
    print(f"   without distillation (the optimizer ruling pending): rounds that fit"
          f" {int((BUDGET_LEFT - smoke - f1 - claim) // round_cost)}")
    print("   current design's round at the same host: $23.98 (round9_simplify_budget.out)"
          f" = {23.98 / round_cost:.1f} factorial rounds")


def hadamard(m: int) -> np.ndarray:
    h = np.array([[1]])
    while h.shape[0] < m:
        h = np.block([[h, h], [h, -h]])
    return h


def simulate(rng, deltas, inter, n_games, luck_share=0.30):
    """Binary results; side A's mean score m = sum_k delta_k eps_k + interactions
    (within-seat: seat A's term a_jk x_j x_k minus seat B's, x = (1 + eps)/2 for A,
    (1 - eps)/2 for B), plus a known-law luck variable L that the variate removes.
    Signs from a Hadamard design (rows cycled, columns 1..K)."""
    K = len(deltas)
    H = hadamard(1 << max(1, math.ceil(math.log2(K + 1))))[:, 1:K + 1]
    eps = H[np.arange(n_games) % H.shape[0]]
    xa, xb = (1 + eps) / 2, (1 - eps) / 2
    m = eps @ deltas
    for (j, k), a in inter.items():
        m = m + a * (xa[:, j] * xa[:, k] - xb[:, j] * xb[:, k])
    # luck: a mean-zero shift with a known law, then the result
    L = rng.normal(0, 1, n_games)
    c = math.sqrt(luck_share)
    p = np.clip(0.5 * (1 + m + c * 0.8 * np.tanh(L)), 0.01, 0.99)
    z = np.where(rng.random(n_games) < p, 1.0, -1.0)
    R = z - c * 0.8 * np.tanh(L)  # the variate: its law is known, its mean zero
    return eps, z, R


def section6(rng):
    print("\n6. Monte Carlo of the mirrored factorial (binary results)")
    K, N = 8, 37_800
    deltas = np.array([0.03, 0.015, 0.0, -0.01, 0.0, 0.0, 0.02, 0.0])
    inter = {(0, 1): 0.01}
    reps = 400
    est = np.zeros((reps, K))
    for r in range(reps):
        eps, z, R = simulate(rng, deltas, inter, N)
        est[r] = (R[:, None] * eps).mean(0)
    pred = deltas.copy()
    pred[0] += 0.005
    pred[1] += 0.005
    se_sim = est.std(0).mean()
    var_r = np.var(R)
    print(f"   K {K}, N {N}: mean bias against delta + a_jk/2 (max over blocks) {np.max(np.abs(est.mean(0) - pred)):.5f};"
          f" SE simulated {se_sim:.5f}, formula sqrt(Var(R)/N) {math.sqrt(var_r / N):.5f}")
    print(f"   block 0, 1 with a within-seat interaction 0.01: read {est.mean(0)[0]:+.4f}, {est.mean(0)[1]:+.4f}"
          f" against delta {deltas[0]:+.4f}, {deltas[1]:+.4f} and the predicted +a/2 aliasing")
    # planted copy: exact under Hadamard balance
    eps, z, R = simulate(rng, deltas, inter, 37_888)
    u = np.zeros(K)
    u[[2, 5]] = 0.02
    Rp = R + eps @ u
    rec = (Rp[:, None] * eps).mean(0) - (R[:, None] * eps).mean(0)
    print(f"   planted copy (+0.02 on blocks 2 and 5), recovered {np.round(rec, 6).tolist()} (exact identity when"
          " N is a multiple of the design's rows)")

    # split-half selection, then a fresh confirmation match of the selected step
    def step_trial(deltas, n_fact=100_000, n_conf=37_800):
        eps, z, R = simulate(rng, deltas, {}, n_fact)
        # halves made of whole design cycles (16 rows): a parity split would hold some columns constant
        half = (np.arange(n_fact) // 16) % 2 == 0
        d1 = (R[half, None] * eps[half]).mean(0)
        d2 = (R[~half, None] * eps[~half]).mean(0)
        se_half = math.sqrt(np.var(R) / half.sum())
        sel = d1 > 2 * se_half
        if not sel.any():
            return None
        predicted = d2[sel].sum()
        true = deltas[sel].sum()
        # confirmation: one seat with the selected blocks on, the other off, raw z
        zc = np.where(rng.random(n_conf) < 0.5 * (1 + true), 1.0, -1.0)
        read, se = zc.mean(), zc.std() / math.sqrt(n_conf)
        return predicted, true, read, se, se_half * math.sqrt(sel.sum())

    out = [step_trial(np.array([0.03, 0.015, 0.0, -0.01, 0.0, 0.0, 0.02, 0.0])) for _ in range(200)]
    out = [o for o in out if o is not None]
    pr = np.array([o[0] for o in out])
    tr = np.array([o[1] for o in out])
    rd = np.array([o[2] for o in out])
    print(f"   selection on one half, prediction on the other: prediction - true mean {np.mean(pr - tr):+.5f}"
          f" (SE {np.std(pr - tr) / math.sqrt(len(pr)):.5f}); confirmation - prediction mean {np.mean(rd - pr):+.5f}")
    miss = [abs(o[2] - o[0]) > 2 * math.hypot(o[3], o[4]) for o in out]
    print(f"   confirmation outside its 2-SE miss line: {np.mean(miss):.3f} of {len(out)} trials (nominal 0.046)")
    null = [step_trial(np.zeros(8)) for _ in range(400)]
    fired = [o for o in null if o is not None]
    adopted = [o for o in fired if o[2] - 2 * o[3] > 0]
    print(f"   all blocks null: a step proposed in {len(fired)} of 400, adopted by the confirmation's lower bound"
          f" in {len(adopted)} ({len(adopted) / 400:.3f}; the bound is 0.023 per proposed step)")


def main():
    print("1. Inputs: sigma_R^2 0.699 (own2: 0.692), V_T 4.01, prices $2.849e-5 / $2.598e-5 / $8.848e-5 a game,"
          " distillation 4.8 h, $82.34 left, S0(a) -41.3 +- 4.4 Elo")
    section2()
    section3()
    section4()
    section5()
    section6(np.random.default_rng(20261021))


if __name__ == "__main__":
    main()
