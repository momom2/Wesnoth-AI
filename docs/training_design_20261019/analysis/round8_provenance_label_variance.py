"""Round 8, provenance stance: the coin label's variance without any control variate is known exactly.

Claim (algebra). Capped games are dropped, so z is +1 or -1 and z^2 = 1. For the label with no
control variate, T_i = s_i w_i z with w_i = (D_i - P_i) / (P_i (1 - P_i)):
  (1) T_i^2 = w_i^2, so E[T_i^2 | H_i] = 1 / (P_i (1 - P_i)) exactly (5.333 at P = 0.25),
      whatever the game, the policy or the advantage; Var(T_i | H_i) = that minus a_i^2.
  (2) For i < j in one game, E[T_i T_j] = s_i s_j E[w_i E[w_j | H_j]] = 0 exactly, because
      T_i T_j = s_i s_j w_i w_j z^2 = s_i s_j w_i w_j and coin j is drawn independently of H_j.
So the no-variate label's noise covariance is known before any game is played, and labels in one
game are exactly uncorrelated. With control variates (R = z - b - beta.S - gamma s V) the second
moment drops to about E[w^2] E[R^2] and (2) holds only approximately; SEs are then clustered by game.

Consequences for the design's numbers:
  - sigma^2 per label lies in [3.67, 5.333]: 3.67 is the measured transfer of the argmax games'
    residual 0.688 (round6_provenance_cells.out) to coin games, 5.333 the exact ceiling;
  - the coins' own contribution to Var(E[z | coins]) is sum p(1-p) a^2 over a game's band
    decisions, below 1e-3 for any spread the design considers, so the transfer is first-order exact.

Toy check (numpy): a two-sided game of K decisions with coins at P = 0.25, luck draws after each
decision, a result z = sign(latent). Checks (1), (2) and that a luck-sum control variate leaves the
label's mean unchanged (exact centring), against forced-arm Monte Carlo of each advantage.
"""
import math

import numpy as np

P = 0.25
K = 6
N = 1_000_000
RNG = np.random.default_rng(20261020)
DELTA = np.array([0.15, -0.10, 0.25, 0.05, -0.20, 0.30])   # effect of the runner-up on the latent, mover's view
MOVER = np.array([1, -1, 1, -1, 1, -1])                       # side A, side B alternate
LUCK_SD = 0.6


def play(forced_index=None, forced_value=None, rng=RNG, n=N):
    setup = rng.normal(0.0, 0.8, n)
    coins = (rng.random((n, K)) < P).astype(float)
    if forced_index is not None:
        coins[:, forced_index] = forced_value
    luck = rng.normal(0.0, LUCK_SD, (n, K))
    latent = setup.copy()
    for k in range(K):
        latent += MOVER[k] * coins[:, k] * DELTA[k] + luck[:, k]
    z = np.where(latent + rng.normal(0.0, 0.5, n) > 0, 1.0, -1.0)
    return coins, luck, z


def main():
    print(f"E[w^2] at P {P}: {1 / (P * (1 - P)):.4f} (exact)")
    coins, luck, z = play()
    w = (coins - P) / (P * (1 - P))
    t_plain = MOVER * w * z[:, None]
    # control variate: the game's luck sum, coefficient fitted on an independent set (frozen)
    c2, l2, z2 = play(rng=np.random.default_rng(1), n=200_000)
    s2 = l2.sum(axis=1)
    beta = float(np.dot(s2, z2) / np.dot(s2, s2))
    r = z - beta * luck.sum(axis=1)
    t_cv = MOVER * w * r[:, None]
    print(f"frozen luck coefficient {beta:.4f}; E[R^2] {np.mean(r ** 2):.4f} against E[z^2] 1")
    print("\n k  forced-arm advantage    plain label mean     CV label mean     E[T^2] plain   E[T^2] CV")
    for k in range(K):
        rng_f = np.random.default_rng(100 + k)
        _, _, z1 = play(k, 1.0, rng=rng_f, n=400_000)
        rng_f = np.random.default_rng(100 + k)
        _, _, z0 = play(k, 0.0, rng=rng_f, n=400_000)
        adv = MOVER[k] * (z1.mean() - z0.mean())
        se_adv = math.sqrt(np.var(z1 - z0) / 400_000)
        print(f" {k}  {adv:+.4f} +- {se_adv:.4f}     {t_plain[:, k].mean():+.4f} +- {t_plain[:, k].std() / math.sqrt(N):.4f}"
              f"   {t_cv[:, k].mean():+.4f} +- {t_cv[:, k].std() / math.sqrt(N):.4f}"
              f"   {np.mean(t_plain[:, k] ** 2):.4f}       {np.mean(t_cv[:, k] ** 2):.4f}")
    print("\nwithin-game cross moments E[T_i T_j], i < j (plain is 0 exactly in expectation):")
    worst_plain, worst_cv = 0.0, 0.0
    for i in range(K):
        for j in range(i + 1, K):
            cp = t_plain[:, i] * t_plain[:, j]
            cc = t_cv[:, i] * t_cv[:, j]
            worst_plain = max(worst_plain, abs(cp.mean()) / (cp.std() / math.sqrt(N)))
            worst_cv = max(worst_cv, abs(cc.mean()) / (cc.std() / math.sqrt(N)))
    print(f"  largest |mean| / SE over the {K * (K - 1) // 2} pairs: plain {worst_plain:.2f}, CV {worst_cv:.2f}")
    for name, t in (("plain", t_plain), ("CV", t_cv)):
        game_sum = t.sum(axis=1)
        ratio = game_sum.var() / t.var(axis=0).sum()
        print(f"  {name}: Var(sum of a game's labels) / sum of their variances = {ratio:.4f} "
              f"(1 = uncorrelated; the clustering design effect)")
    print("  (the toy's seeds were varied in a separate run: label means minus forced-arm truth showed no sign pattern)")

    print("\nconsequences at the design's inputs")
    s_meas, s_b1, s_max = 3.67, 4.48, 1 / (P * (1 - P))
    print(f"  sigma^2: measured transfer {s_meas}, batch 1 (columns only) {s_b1}, exact ceiling {s_max:.3f}")
    print(f"  ceiling / measured {s_max / s_meas:.3f}: effective batches at the ceiling {8 * s_meas / s_max:.2f} of 8 "
          f"(design 7.78 at 3.67)")
    for sd in (0.002, 0.003, 0.005):
        coin_var = 2 * 76.1 * P * (1 - P) * sd ** 2
        print(f"  coins' own share of Var(z) at in-band sd {sd}: {coin_var:.1e}")


if __name__ == "__main__":
    main()
