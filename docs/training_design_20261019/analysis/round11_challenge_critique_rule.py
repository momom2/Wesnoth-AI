"""Round 11, critique of the challenge stance: the certified trust region's radius rule over many rounds.

The stance's own model (round11_challenge.md 1.3; analysis/round11_challenge_budget.py [step size]):
at radius eps = k eps*, the first-order gain is F1 sqrt(k) and the actual gain F1 (sqrt(k) - k / 2),
F1 being the first-order gain at the best radius eps*. Its forecast SE (budget [standard errors]) is
proportional to sqrt(eps) at fixed held-out size, so it is sG1 sqrt(k), and the forecast's z-score
zG = F1 / sG1 does not depend on the radius. The cross-play reading R has SE sR = 3.47 Elo (10,000 games).

The rule, as written: move only when the forecast G_hat > 2 SE; rho_hat = R / G_hat; next eps =
eps / (2 (1 - rho_hat))^2, capped at x4 and /4; revert (retrain at eps / 4) when R < -2 sR.
Stationary a and C across rounds (the stance's assumption). Reported per (F1, zG):
  - the share of rounds the rule moves at all (it is Phi(zG - 2) at every radius);
  - the mean share of the best gain over 12 rounds from k0 = 1/16 and from k0 = 1/100;
  - from the optimum k = 1, the share of rounds sent to k >= 4 (zero or negative gain next round);
  - the probability that 12 rounds pass with no revert and no stop when F1 = 0 (a null update).
numpy and the standard library only; 20,000 replicate chains a cell.
"""
import math

import numpy as np

RNG = np.random.default_rng(20261021)
SR = 3.47
ROUNDS = 12
REPS = 20000


def chain(F1, zG, k0, sR=SR, rounds=ROUNDS, reps=REPS):
    sG1 = F1 / zG if F1 > 0 else 1.0
    k = np.full(reps, k0)
    share = np.zeros(reps)
    moved = 0
    for _ in range(rounds):
        first = F1 * np.sqrt(k)
        actual = F1 * (np.sqrt(k) - k / 2)
        g_hat = first + sG1 * np.sqrt(k) * RNG.standard_normal(reps)
        r = actual + sR * RNG.standard_normal(reps)
        resolved = g_hat > 2 * sG1 * np.sqrt(k)
        moved += resolved.mean()
        revert = r < -2 * sR
        rho = np.clip(r / np.maximum(g_hat, 1e-9), -10, 0.99)
        mult = np.clip(1.0 / (2 * (1 - rho)) ** 2, 0.25, 4.0)
        mult = np.where(resolved, mult, 1.0)
        gain_k = np.where(revert, k / 4, k)          # a revert retrains the step at eps / 4
        if F1 > 0:
            share += 2 * np.sqrt(gain_k) - gain_k
        k = np.where(revert, k / 4, k * mult)
    return share / rounds, moved / rounds


def from_optimum(F1, zG, reps=200000):
    sG1 = F1 / zG
    g_hat = F1 + sG1 * RNG.standard_normal(reps)
    r = F1 / 2 + SR * RNG.standard_normal(reps)
    resolved = g_hat > 2 * sG1
    rho = np.clip(r / np.maximum(g_hat, 1e-9), -10, 0.99)
    mult = np.where(resolved, np.clip(1.0 / (2 * (1 - rho)) ** 2, 0.25, 4.0), 1.0)
    nxt = mult
    return float(np.mean(nxt >= 4.0 - 1e-9)), float(np.mean(2 * np.sqrt(nxt) - nxt))


def null_update(rounds=ROUNDS, reps=REPS):
    """F1 = 0: R ~ N(0, sR) each round. The stance's stops: two reverts in a row; claims compare a
    chain of zeros with a claim of zero, so they do not fire either."""
    r = SR * RNG.standard_normal((reps, rounds))
    rev = r < -2 * SR
    two_in_row = np.any(rev[:, 1:] & rev[:, :-1], axis=1)
    return float(np.mean(~two_in_row)), float(rev.mean())


def main():
    print("[the gate: P(rule moves) = Phi(zG - 2), at every radius]")
    for zG in (1, 2, 3, 5):
        print(f"  zG {zG}: {0.5 * (1 + math.erf((zG - 2) / math.sqrt(2))):.3f}")
    print("\n[12 rounds, stationary a and C: mean share of the best gain; share of rounds the rule moved]")
    for F1 in (5.0, 10.0, 20.0, 40.0):
        for zG in (1.0, 2.0, 3.0, 5.0):
            row = []
            for k0 in (1 / 16, 1 / 100):
                s, m = chain(F1, zG, k0)
                row.append(f"k0 {k0:.4f}: share {s.mean():.3f} (moved {m:.2f})")
            print(f"  F1 {F1:4.0f} Elo, zG {zG:.0f}: " + "; ".join(row))
    print("\n[from the optimum k = 1: share of rounds sent to k >= 4; mean share of the best next round]")
    for F1 in (5.0, 10.0, 20.0, 40.0):
        for zG in (2.0, 3.0, 5.0):
            p4, nxt = from_optimum(F1, zG)
            print(f"  F1 {F1:4.0f}, zG {zG:.0f}: to k>=4 {p4:.3f}; next-round share {nxt:+.3f}")
    keep, rate = null_update()
    print(f"\n[null update, F1 = 0, 12 rounds]: no 'two reverts in a row' stop in {keep:.3f} of chains; "
          f"revert rate {rate:.4f} a round")


if __name__ == "__main__":
    main()
