"""Round 11 synthesis: the derived numbers of current_design.md and optimizer_discussion.md.

Every input below is a measured record (named beside it) or a stated choice; every output is
arithmetic on them, or a Monte Carlo of a rule under stated truths. Outputs marked
"illustration" carry no decision. numpy and the standard library only; about 10 s of CPU.
"""
import math

import numpy as np

# ---- measured inputs
HOST_RATE = 0.577          # $/h, the coin box (seed section 7; round9_simplify_labels.out)
GEN_GAME = 2.598e-5        # $ a one-network pool game at HOST_RATE (round9_simplify_labels.out [prices])
TWO_NET_GAME = 8.848e-5    # $ a decisive two-network game (round9_simplify_labels.out [prices])
TRAIN_RATE = 1477.0        # positions/s, configuration B (training/metrics/trainer_calibration_20261015)
DECISIONS = 475.7          # a game, own2 v own1 at raw:t0+eo-1.5 (round11_provenance_census.out)
SIDE_DECISIONS = 237.9     # a side-game (same)
WALK_WALL = 0.0094         # s of wall a game for the luck walk on 31 cores (round11_simplify_budget.out [misc])
RESID = 0.6988             # Var(z - b - beta S) / Var(z) at argmax (round9_simplify_residual.out)
SETUP_MIN = 22.0           # box staging minutes (current_design.md round 10 table; 15 of them a choice)
BUDGET_COUNTED = 82.34 - 0.90    # current_design.md section 4 less match L (BACKLOG 0.30.5)
BUDGET_NOT_COUNTED = 85.00 - 0.90

ELO_PER_Z = 400 / math.log(10) * 2   # d Elo / d E[z] at E[z] = 0 (z = +1 win, -1 loss): 347.4

TRAIN_GAME = DECISIONS / TRAIN_RATE / 3600 * HOST_RATE
FWD_UPPER = TRAIN_GAME               # a forward costs at most a trained position (bound)
FWD_LOWER = TRAIN_GAME / 3           # illustration: forward-only at three times forward-backward (unmeasured)
SETUP = SETUP_MIN / 60 * HOST_RATE


def match_se_elo(n, p):
    """SE in Elo of a match of n decisive games read at win rate p (logistic Elo)."""
    return 400 / math.log(10) / math.sqrt(n * p * (1 - p))


def is_se_z(sigma2, k_side, n_games):
    """SE (z units) of the held-out reading of one seat switched against pi_k, averaged over both
    seats: sigma sqrt((exp(2K) - 1) / (2 n)) (round11_provenance_derived.py section 5)."""
    return math.sqrt(sigma2) * math.sqrt((math.exp(2 * k_side) - 1) / (2 * n_games))


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def section_prices():
    print("1. prices at $%.3f/h (generation and training at the argmax game length; the T1 length is unmeasured"
          " and scales every line)" % HOST_RATE)
    print(f"   generation ${GEN_GAME:.3e} a game; training every decision ${TRAIN_GAME:.3e} a game"
          f" ({TRAIN_GAME / GEN_GAME:.2f}x generation); two-network game ${TWO_NET_GAME:.3e}")
    print(f"   forward over a held-out game: at most ${FWD_UPPER:.3e}; illustration at 3x a trained position's"
          f" speed ${FWD_LOWER:.3e}")
    print(f"   box setup {SETUP_MIN:.0f} min ${SETUP:.3f}; luck walk ${WALK_WALL / 3600 * HOST_RATE:.2e} a game")


def stage1():
    selfplay = 30000 * GEN_GAME
    d0 = 12800 * GEN_GAME
    mem = 3200 * TWO_NET_GAME
    walk = 30000 * WALK_WALL / 3600 * HOST_RATE
    total = selfplay + d0 + mem + walk + SETUP
    print("2a. stage 1 (no training): 30,000 own2 T1 self-play games (20,000 to train on, 10,000 held out)"
          f" ${selfplay:.2f}; D0 own2@T1 v own2@reference, 12,800 decisive, one network ${d0:.2f};"
          f" memory 64 v 0 slots at T1, 3,200 decisive, two networks ${mem:.2f}; luck walk ${walk:.3f};"
          f" setup ${SETUP:.2f}; total ${total:.2f}")
    return total


def stage_c_forwards(trimmed=False):
    """Forward-equivalents over the 10,000 held-out games H (half A and half B 5,000 each).
    pi_k on all of H (ratio denominators, replay identity): 1. On half A: the five arms at K 1/4
    (SGD, reduced AdamW, Muon, heads-only, probe span), the winner's two other radii, reduced AdamW
    at N 5,000 and 10,000, the planted step: 10 candidates, 5 forwards. On half B: the winner, 0.5.
    On a 5% subsample: KL searches, 3 trials a candidate; the critical batch's 9 KL readings; the
    4 head-temperature Fisher readings."""
    candidates = 10 - (2 if trimmed else 0)       # trimmed: no heads-only arm, no 5,000-game point
    return 1 + candidates * 0.5 + 0.5 + (candidates * 3 + 9 + 4) * 0.05


def stage_c():
    grad = 20000 * TRAIN_GAME
    n_fwd = stage_c_forwards()
    lo = grad + n_fwd * 10000 * FWD_LOWER + SETUP
    hi = grad + n_fwd * 10000 * FWD_UPPER + SETUP
    print(f"2b. stage C (gradients and candidate steps, nothing played): gradient pass over 20,000 games ${grad:.2f};"
          f" {n_fwd:.2f} forward-equivalents over 10,000 held-out games; total ${lo:.2f} (illustrative forward"
          f" speed) to ${hi:.2f} (bound)")
    t = stage_c_forwards(trimmed=True)
    print(f"    trimmed (no heads-only arm, no 5,000-game point), forwards at the bound: {t:.2f} forward-equivalents,"
          f" ${grad + t * 10000 * FWD_UPPER + SETUP:.2f}")
    return lo, hi


def stage_p():
    xplay = 5000 * TWO_NET_GAME
    link = 12800 * GEN_GAME
    total = xplay + SETUP
    print(f"2c. stage P, the planted step played (its gradient and offline readout are in stage C): 5,000"
          f" two-network games ${xplay:.2f} + setup = ${total:.2f}; the sharpening link match, iff its slope"
          f" is resolved: 12,800 decisive one-network games ${link:.2f}")
    return total


def round_cost(n_train, n_held, fwd, claim_every=3):
    gen = (n_train + n_held) * GEN_GAME
    grad = n_train * TRAIN_GAME
    # pi_k on all of H; three radii on half A; the winner on half B; KL searches (3 x 3 trials, 5%)
    forwards = (1 + 1.5 + 0.5 + 9 * 0.05) * n_held * fwd
    claim = 12800 * TWO_NET_GAME / claim_every
    return gen + grad + forwards + claim, (gen, grad, forwards, claim)


def section_rounds(s1, c_lo, c_hi, p):
    print("2d. a round: N self-play games trained once + |H| held out, three radii read on H, a 12,800-game claim"
          " every third round")
    for n, h in ((10000, 5000), (20000, 10000), (40000, 10000), (80000, 20000)):
        lo, parts_lo = round_cost(n, h, FWD_LOWER)
        hi, parts_hi = round_cost(n, h, FWD_UPPER)
        print(f"   N {n:6,} |H| {h:6,}: ${lo:5.2f} to ${hi:5.2f} (generation ${parts_lo[0]:.2f}, gradient ${parts_lo[1]:.2f},"
              f" forwards ${parts_lo[2]:.2f}-{parts_hi[2]:.2f}, claim share ${parts_lo[3]:.2f})")
    for label, budget in (("if the $5 window counted", BUDGET_COUNTED), ("if not", BUDGET_NOT_COUNTED)):
        left_lo = budget - s1 - c_hi - p
        left_hi = budget - s1 - c_lo - p
        r20_lo = round_cost(20000, 10000, FWD_UPPER)[0]
        r20_hi = round_cost(20000, 10000, FWD_LOWER)[0]
        print(f"   budget ${budget:.2f} {label}: after stages 1, C, P ${left_lo:.2f} to ${left_hi:.2f};"
              f" rounds at N 20,000: {left_lo / r20_lo:.0f} to {left_hi / r20_hi:.0f}")


GEN_RATE = 22209.0          # pool games an hour (round11_invent_budget.out section 2)
TWO_NET_RATE = 6521.0       # decisive two-network games an hour (round9_simplify_labels.out [prices])


def section_wall():
    gen_h = 30000 / GEN_RATE
    grad_h = 20000 * DECISIONS / TRAIN_RATE / 3600
    fwd_pos = (1 + 1.5 + 0.5 + 9 * 0.05) * 10000 * DECISIONS
    fwd_lo_h = fwd_pos / (3 * TRAIN_RATE) / 3600
    fwd_hi_h = fwd_pos / TRAIN_RATE / 3600
    claim_h = 12800 / TWO_NET_RATE / 3
    print(f"2e. a round's wall at N 20,000, |H| 10,000: generation {gen_h:.2f} h, gradient pass {grad_h:.2f} h,"
          f" forwards {fwd_lo_h:.2f}-{fwd_hi_h:.2f} h, claim share {claim_h:.2f} h:"
          f" {gen_h + grad_h + fwd_lo_h + claim_h:.1f} to {gen_h + grad_h + fwd_hi_h + claim_h:.1f} box-hours")


def section_se():
    print("3. standard errors")
    print("   per-decision KL eps and per-side-game KL K = eps x 237.9 decisions (argmax length):")
    for k in (0.125, 0.25, 0.5):
        print(f"     K {k:.3f} per side-game = eps {k / SIDE_DECISIONS:.5f} per decision")
    print("   match SE (Elo) by decisive games n and win rate p:")
    for n in (3200, 6400, 12800):
        cells = ", ".join(f"p {p:.2f}: {match_se_elo(n, p):4.1f}" for p in (0.5, 0.25, 0.09, 0.05))
        print(f"     n {n:6,}: {cells}")
    print("   held-out reading of a step (one seat switched against pi_k), on half of H, in Elo:")
    for sigma2 in (RESID, 1.0):
        for n in (2500, 5000, 10000):
            cells = ", ".join(f"K {k:.3f}: {is_se_z(sigma2, k, n) * ELO_PER_Z:4.2f}" for k in (0.125, 0.25, 0.5))
            print(f"     sigma2 {sigma2:.3f}, {n:6,} games: {cells}")


def section_rules():
    print("4. operating characteristics of the round rules (s = SE of the confirmed reading on half B)")
    print("   revert iff the confirmed reading < -2 s: P(revert | true gain g):")
    print("     " + ", ".join(f"g/s {r:.1f}: {phi(-2 - r):.3f}" for r in (0.0, 0.5, 1.0, 2.0)))
    print("   pause for the user iff the sum of the last three confirmed readings is not above 2 SE"
          " (SE sqrt(3) s): P(pause at a check | per-round gain g):")
    print("     " + ", ".join(f"g/s {r:.1f}: {phi(2 - math.sqrt(3) * r):.3f}" for r in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)))
    for k, sig in ((0.25, RESID), (0.5, RESID)):
        s = is_se_z(sig, k, 5000) * ELO_PER_Z
        g80 = (2 + 0.8416) / math.sqrt(3) * s
        print(f"     at K {k}, half H 5,000, sigma2 {sig:.3f}: s {s:.2f} Elo; a per-round gain of {g80:.1f} Elo passes"
              f" a check with probability 0.8")
    print("   cumulative transfer check at round k: claim_k - claim_0 against the sum of k confirmed readings;"
          " SE sqrt(SE_claim(p_0)^2 + SE_claim(p_k)^2 + k s^2), claims of 12,800 decisive games:")
    s = is_se_z(RESID, 0.25, 5000) * ELO_PER_Z
    for p0, pk in ((0.09, 0.09), (0.09, 0.25), (0.25, 0.5), (0.5, 0.5)):
        cells = ", ".join(
            f"k {k:2d}: {math.sqrt(match_se_elo(12800, p0) ** 2 + match_se_elo(12800, pk) ** 2 + k * s * s):4.1f}"
            for k in (3, 6, 9, 12))
        print(f"     p_0 {p0:.2f}, p_k {pk:.2f}, s {s:.2f}: {cells}")


def section_batch():
    print("5. cost-optimal round size N* = sqrt(c_f B / c_g) (McCandlish et al. 2018 with a fixed per-round cost);"
          " B in games under the step's metric is unmeasured (stage C)")
    c_g = GEN_GAME + TRAIN_GAME
    for fwd, label in ((FWD_LOWER, "illustrative forward"), (FWD_UPPER, "forward bound")):
        c_f = 10000 * GEN_GAME + (1 + 1.5 + 0.5 + 9 * 0.05) * 10000 * fwd + 12800 * TWO_NET_GAME / 3
        cells = ", ".join(f"B {b:,}: {math.sqrt(c_f * b / c_g):9,.0f}" for b in (1000, 10000, 100000, 1000000))
        print(f"   c_f ${c_f:.2f} ({label}), c_g ${c_g:.2e}: {cells}")


def ladder_mc(z_star, k0, rounds=12, corr=0.9, trials=20000, seed=7):
    """The radius ladder under the quadratic step model: true gain G*(2 sqrt(k) - k) at k = eps/eps*.
    Each round reads three radii eps x {1/2, 1, 2} on half A (SE s* sqrt(k), correlation corr between
    them), adopts the best, confirms on half B (independent, same SE); a confirmed reading below -2 SE
    reverts (gain 0) and halves eps. z_star = G*/s*, the z of the optimum's reading on half A."""
    rng = np.random.default_rng(seed)
    mults = np.array([0.5, 1.0, 2.0])
    k = np.full(trials, k0)
    share = np.zeros(trials)
    high = 0
    cov = corr * np.ones((3, 3)) + (1 - corr) * np.eye(3)
    chol = np.linalg.cholesky(cov)
    for _ in range(rounds):
        ks = k[:, None] * mults[None, :]
        true = 2 * np.sqrt(ks) - ks
        se = np.sqrt(ks) / z_star
        noise = (rng.standard_normal((trials, 3)) @ chol.T) * se
        pick = np.argmax(true + noise, axis=1)
        idx = np.arange(trials)
        g = true[idx, pick]
        conf = g + rng.standard_normal(trials) * se[idx, pick]
        revert = conf < -2 * se[idx, pick]
        share += np.where(revert, 0.0, g)
        k_next = ks[idx, pick]
        k = np.where(revert, k / 2, k_next)         # a revert keeps pi_k and halves its radius
        high += int(np.sum(k >= 4))
    return share.mean() / rounds, high / (rounds * trials)


def section_ladder():
    print("6. the radius ladder against the fixed step and the ratio rule (round11_challenge_critique_rule.out),"
          " 12 stationary rounds, mean share of the best round's gain")
    for k0 in (1 / 16, 1 / 100):
        for z in (0.5, 1.0, 2.0, 3.0):
            for corr in (0.9, 0.0):
                sh, hi = ladder_mc(z, k0, corr=corr)
                print(f"   k0 {k0:.4f}, z* {z:.1f}, correlation {corr:.1f}: share {sh:.3f}; rounds at k >= 4: {hi:.3f}")
    print("   fixed step at k0 1/16: share 2 sqrt(k0) - k0 = 0.438; at k0 1/100: 0.190")


def main():
    section_prices()
    s1 = stage1()
    c_lo, c_hi = stage_c()
    p = stage_p()
    section_rounds(s1, c_lo, c_hi, p)
    section_wall()
    section_se()
    section_rules()
    section_batch()
    section_ladder()


if __name__ == "__main__":
    main()
