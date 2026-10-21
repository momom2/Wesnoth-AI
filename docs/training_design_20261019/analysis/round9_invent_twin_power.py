"""Round 9, invent stance: chance-coupled twins, the derived numbers.

1. Variance factor of a twin-centred label: K exchangeable twins on one seed
   (same setup, same keyed dice, independent coins) have residuals with a
   common correlation rho (compound symmetry, exact by exchangeability). The
   label residual X_k = R_k - kappa * mean_{j != k} R_j, at the optimal kappa,
   has E[X^2]/E[R^2] = 1 - rho^2 (K-1) / (1 + (K-2) rho). Table over rho
   (no decision rests on the grid; the pilot measures rho).
2. The pilot's resolution, simulated: binary results z = sign(mu_seed + e),
   kappa and the ratio cross-fitted on halves of the seeds, at the pilot's
   1,000 seeds x 4 twins. SE of the measured ratio and the smallest rho whose
   ratio's upper 2-SE bound is below 0.95.
3. The planted shared component: a copy of every twin's result shifted by
   a * eta_seed (eta = +-1 keyed by seed) must move the measured ratio to the
   derived value r'(rho, a); simulated recovery.
4. Budget at the measured price, as a function of the measured ratio r: the
   coin batches of S1, S3 and S5 scale by r (same floors), each twin index is
   its own match (one start-up each). Stage list and prices from
   round9_provenance_budget.out (host 56955777, $0.577 an hour).

numpy and the standard library only; under a minute.
"""
import math

import numpy as np

# measured: round9_provenance_budget.out (effective 22,341 games/h one network,
# 6,523 games/h two networks, start-up 69 s, $0.577/h, $82.34 left, setup 0.1 h,
# distillation 2.1/0.44 box-hours)
RATE, ONE_PER_H, TWO_PER_H, STARTUP_S = 0.577, 22_341, 6_523, 69
LEFT, SETUP_H, DISTILL_H = 82.34, 0.1, 2.1 / 0.44
BATCH, BATCHES, MATCH, TWO_NET, LINK = 100_000, 8, 37_800, 12_800, 12_800
K_TWINS, PILOT_SEEDS = 4, 1_000


def ratio(rho, k):
    return 1 - rho * rho * (k - 1) / (1 + (k - 2) * rho)


def section1():
    print("1. E[X^2]/E[R^2] at the optimal kappa, by rho and K")
    print("   rho    K=2    K=4    K=8   K->inf")
    for rho in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7):
        print(f"   {rho:.2f}  {ratio(rho, 2):.3f}  {ratio(rho, 4):.3f}  {ratio(rho, 8):.3f}  {1 - rho:.3f}")


def latent_for(rho_target):
    # z = sign(mu + e), mu ~ N(0, t^2), e ~ N(0, 1): corr(z_j, z_k) = (2/pi) asin(t^2/(1+t^2))
    lo, hi = 0.0, 50.0
    for _ in range(100):
        t2 = (lo + hi) / 2
        r = 2 / math.pi * math.asin(t2 / (1 + t2))
        lo, hi = (t2, hi) if r < rho_target else (lo, t2)
    return math.sqrt(lo)


def measured_ratio(z):
    """Cross-fitted: kappa fitted on one half of the seeds, ratio read on the other."""
    n, k = z.shape
    loo = (z.sum(axis=1, keepdims=True) - z) / (k - 1)
    halves = (np.arange(n) % 2 == 0, np.arange(n) % 2 == 1)
    num = den = 0.0
    for fit, read in (halves, halves[::-1]):
        kappa = (z[fit] * loo[fit]).sum() / (loo[fit] ** 2).sum()
        num += ((z[read] - kappa * loo[read]) ** 2).sum()
        den += (z[read] ** 2).sum()
    return num / den


def section2(rng, reps=400):
    print(f"\n2. pilot resolution: {PILOT_SEEDS} seeds x {K_TWINS} twins, binary results, {reps} replicates")
    print("   rho   derived r  mean r_hat   SE(r_hat)  P(upper 2-SE bound < 0.95)")
    for rho in (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5):
        t = latent_for(rho)
        rs = []
        for _ in range(reps):
            mu = rng.normal(0, t, size=(PILOT_SEEDS, 1))
            z = np.sign(mu + rng.normal(size=(PILOT_SEEDS, K_TWINS)))
            rs.append(measured_ratio(z))
        rs = np.array(rs)
        se = rs.std()
        print(f"   {rho:.2f}  {ratio(rho, K_TWINS):.3f}      {rs.mean():.3f}       {se:.4f}     "
              f"{np.mean(rs + 2 * se < 0.95):.2f}")


def section3(rng, reps=200, a=0.5):
    print(f"\n3. planted shared component a * eta_seed, a = {a}: the ratio on the planted copy")
    print("   rho   derived r'  mean r'_hat  SE    (r' from rho' = (rho + a^2)/(1 + a^2), unit-variance z)")
    for rho in (0.0, 0.2):
        t = latent_for(rho)
        rho_p = (rho + a * a) / (1 + a * a)
        rs = []
        for _ in range(reps):
            mu = rng.normal(0, t, size=(PILOT_SEEDS, 1))
            z = np.sign(mu + rng.normal(size=(PILOT_SEEDS, K_TWINS)))
            eta = rng.choice((-1.0, 1.0), size=(PILOT_SEEDS, 1))
            rs.append(measured_ratio(z + a * eta))
        rs = np.array(rs)
        print(f"   {rho:.2f}  {ratio(rho_p, K_TWINS):.3f}       {rs.mean():.3f}       {rs.std():.4f}")
    print("   a twin joined to the wrong seed reads r' = 1 (the plant is not shared by the join): caught")


def section4():
    p1, p2 = RATE / ONE_PER_H, RATE / TWO_PER_H
    setup = SETUP_H * RATE
    startup = STARTUP_S / 3600 * RATE
    print(f"\n4. budget at ${RATE}/h: one network ${p1:.3e} a game, two networks ${p2:.3e}; ${LEFT} left")
    pilot = setup + 2 * K_TWINS * PILOT_SEEDS * p1 + 2 * K_TWINS * startup
    print(f"   T0 twin pilot, 2 keyings x {K_TWINS} twins x {PILOT_SEEDS} seeds, one match per twin index: ${pilot:.2f}")
    fixed = {
        "A: L+T (own1+R vs own1, own2+R vs own2), 12,800 games each": setup + 2 * LINK * p1,
        "A: claim own2+R vs own1, two networks, if T adopts": TWO_NET * p2,
        "S2: adoption + claim vs own1": MATCH * p1 + TWO_NET * p2,
        "E reserve": 2 * MATCH * p1 + TWO_NET * p2,
        "S3 matches: adoption + claim": MATCH * p1 + TWO_NET * p2,
        "S4: distillation + N vs pi_k": DISTILL_H * RATE + TWO_NET * p2,
        "S5 matches: adoption vs pi_k + claim, two networks each": 2 * TWO_NET * p2,
    }
    for k, v in fixed.items():
        print(f"   {k}: ${v:.2f}")
    fixed_total = sum(fixed.values()) + pilot
    print("   coin rounds at ratio r (8 batches of 100,000 r games, K matches a batch):")
    print("   r      a round's batches   three rounds + all else   left    extra rounds (batches + S3's matches) that fit")
    for r in (1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5):
        batches = 3 * setup + 3 * BATCHES * (BATCH * r * p1 + K_TWINS * startup)
        one_round = BATCHES * (BATCH * r * p1 + K_TWINS * startup) + setup
        total = fixed_total + batches
        left = LEFT - total
        extra = max(0.0, left) / (one_round + fixed["S3 matches: adoption + claim"])
        print(f"   {r:.2f}   ${one_round:6.2f}            ${total:6.2f}                  ${left:+6.2f}  {extra:.2f}")
    plain = 3 * setup + 3 * BATCHES * BATCH * p1 + fixed_total - pilot
    print(f"   reference, no twins and no pilot: ${plain:.2f}, left ${LEFT - plain:+.2f}")


def main():
    rng = np.random.default_rng(20261021)
    section1()
    section2(rng)
    section3(rng)
    section4()


if __name__ == "__main__":
    main()
