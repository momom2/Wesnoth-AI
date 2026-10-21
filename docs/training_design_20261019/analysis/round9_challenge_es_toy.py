"""Round 9, challenge stance: checks by simulation of the two formulas the
antithetic weight-space design rests on.

A. The signal estimator. K perturbations, M antithetic games each, results
   z in {-1, +1} with P(z = +1 | eps) = (1 + m(eps)) / 2, m(eps) = 2 sigma g.eps.
   s^2 := Var_eps(m(eps)). Estimator: the variance of the K per-perturbation
   means minus the mean binomial variance / M. Checked: bias and SE against
   SE(s2_hat) ~ sqrt(2 / (K - 1)) (s^2 + sigma_R^2 / M).

B. The step's gain. A quadratic f(theta) = g.theta - theta' H theta / 2 in
   D dimensions, most of them flat; antithetic games read
   R = f(theta + sigma eps) - f(theta - sigma eps) + noise, one-sided games
   R = f(theta + sigma eps) - f(theta) + noise. From the one-sided mean loss L
   and the antithetic signal s^2 (both measured in the simulation as E0
   would), the step alpha* and its gain are predicted:
       |g|^2 = s^2 / (4 sigma^2),  tr H = 2 L / sigma^2,
       nu = (sigma_R^2 + s^2) / (4 sigma^2 N)   (per-component noise of g_hat),
       G_pred = |g|^4 / (2 (g'Hg + nu tr H))  ~  N s^4 / (16 L (sigma_R^2 + s^2))
   when the noise term dominates (g'Hg is not measured by E0; the formula
   without it is an upper bound on the gain). The realised gain
   f(theta + alpha* g_hat) - f(theta) is averaged over replications.

numpy only; a few seconds.
"""
from __future__ import annotations

import math

import numpy as np

SIGMA_R2 = 0.824  # E[(z - beta.S)^2], round9_challenge_coherence.out section 1


def part_a(rng):
    print("A. Signal estimator, binary results")
    print("   s      K    M     mean s2_hat   true s2     SE sim     SE formula")
    for s in (0.0, 0.02, 0.04):
        for K, M in ((100, 400), (40, 1000)):
            reps = 300
            est = []
            for _ in range(reps):
                m = rng.normal(0.0, s, size=K) if s > 0 else np.zeros(K)
                m = np.clip(m, -0.99, 0.99)
                wins = rng.binomial(M, (1 + m) / 2)
                means = 2 * wins / M - 1
                binom = np.mean(1 - means ** 2) / (M - 1)
                est.append(np.var(means, ddof=1) - binom)
            est = np.array(est)
            se_f = math.sqrt(2 / (K - 1)) * (s * s + 1.0 / M)
            print(f"   {s:<5}  {K:<4} {M:<5} {est.mean():+.6f}   {s * s:.6f}   {est.std():.6f}   {se_f:.6f}")


def simulate_step(rng, D, n_curved, h, g_norm, sigma, N, reps=20):
    """One configuration: returns measured s2, L, predicted gains, realised gain."""
    g = np.zeros(D)
    g[:n_curved] = rng.normal(size=n_curved)
    g *= g_norm / np.linalg.norm(g)
    Hdiag = np.zeros(D)
    Hdiag[:n_curved] = h
    trH = Hdiag.sum()
    gHg = float(np.sum(g * g * Hdiag))

    def f(theta):
        return theta @ g - 0.5 * np.sum(Hdiag * theta * theta, axis=-1)

    # E0 as the box would read it: s2 from K x M antithetic, L from one-sided games.
    K, M = 100, 400
    eps = rng.normal(size=(K, D))
    m = 2 * sigma * eps @ g
    means = m + rng.normal(0, math.sqrt(SIGMA_R2 / M), size=K)
    s2_hat = max(np.var(means, ddof=1) - SIGMA_R2 / M, 1e-12)
    n_one = 20000
    eps1 = rng.normal(size=(n_one, D))
    R1 = sigma * eps1 @ g - 0.5 * sigma ** 2 * np.sum(Hdiag * eps1 * eps1, axis=1) + rng.normal(0, math.sqrt(SIGMA_R2), n_one)
    L_hat = max(-R1.mean(), 1e-9)
    s2_true = 4 * sigma ** 2 * g_norm ** 2
    L_true = 0.5 * sigma ** 2 * trH

    g2_hat = s2_hat / (4 * sigma ** 2)
    trH_hat = 2 * L_hat / sigma ** 2
    nu_hat = (SIGMA_R2 + s2_hat) / (4 * sigma ** 2 * N)
    alpha = g2_hat / (nu_hat * trH_hat)  # gHg unknown to E0: the noise-dominated step
    G_upper = N * s2_hat ** 2 / (16 * L_hat * (SIGMA_R2 + s2_hat))
    nu = (SIGMA_R2 + s2_true) / (4 * sigma ** 2 * N)
    G_exact_at_alpha = alpha * g_norm ** 2 - 0.5 * alpha ** 2 * (gHg + nu * trH)

    gains = []
    for _ in range(reps):
        acc = np.zeros(D)
        for start in range(0, N, 10000):
            n = min(10000, N - start)
            e = rng.standard_normal(size=(n, D), dtype=np.float32)
            R = 2 * sigma * (e @ g.astype(np.float32)) + rng.normal(0, math.sqrt(SIGMA_R2), n).astype(np.float32)
            acc += R @ e
        g_hat = acc / (2 * sigma * N)
        gains.append(f(alpha * g_hat) - f(np.zeros(D)))
    gains = np.array(gains)
    return dict(s2_true=s2_true, s2_hat=s2_hat, L_true=L_true, L_hat=L_hat, gHg_share=gHg / (gHg + nu * trH),
                G_upper=G_upper, G_model=G_exact_at_alpha, G_real=gains.mean(), G_real_se=gains.std() / math.sqrt(reps))


def part_b(rng):
    print("\nB. Step gain: prediction from measured s2 and L against the realised gain (z units)")
    print("   D     curved  N       s2 true/hat         L true/hat       gHg share  G upper   G model   G realised")
    configs = [
        (500, 20, 1.0, 0.015, 1.0, 20000),
        (500, 20, 1.0, 0.015, 1.0, 100000),
        (500, 200, 0.1, 0.015, 1.0, 100000),
        (500, 20, 1.0, 0.05, 1.0, 20000),
        (500, 20, 10.0, 0.05, 1.0, 100000),
    ]
    for D, n_curved, h, g_norm, sigma, N in configs:
        r = simulate_step(rng, D, n_curved, h, g_norm, sigma, N)
        print(f"   {D:<5} {n_curved:<7} {N:<7} {r['s2_true']:.5f}/{r['s2_hat']:.5f}   {r['L_true']:.3f}/{r['L_hat']:.3f}"
              f"      {r['gHg_share']:.3f}     {r['G_upper']:.4f}    {r['G_model']:.4f}    {r['G_real']:.4f} +- {r['G_real_se']:.4f}")
    print("   G upper: N s^4 / (16 L (sigma_R^2 + s^2)) from the measured s2, L; G model: the same step's exact")
    print("   quadratic gain including g'Hg; G realised: the step applied, averaged over replications.")


def main():
    rng = np.random.default_rng(20261021)
    part_a(rng)
    part_b(rng)


if __name__ == "__main__":
    main()
