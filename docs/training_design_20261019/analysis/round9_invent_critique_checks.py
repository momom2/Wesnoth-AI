"""Round 9, critique of the invent stance (chance-coupled twins): checks.

1. Self-inclusion bug. If the sibling mean wrongly includes the game itself,
   the label shrinks by (1 - kappa/K) and r_hat reads low (a spurious twin
   gain). Which of the stance's checks see it? Simulated: K = 4 twins, a
   residual with common correlation rho, one coin per game (P 0.25) with a
   planted effect. Readings: label mean, r_hat, the coin plant added to the
   game's own residual only versus to every twin's z, and the planted shared
   component (a = 0.5) against r'(rho_hat) with rho_hat inverted from r_hat
   versus rho_hat from pairwise covariances.
2. Cross-twin covariance. A toy where the shared chance U also moves the
   coin's realized effect (delta_U, sd s_d). The true sampling variance of the
   twin estimator of the mean effect, relative to K times as many independent
   games with plain labels (r_eff), against the residual ratio r_hat the
   stance sizes batches on.
3. Arithmetic: r at rho 0.030 for K = 4; the 0.95 line's saving against the
   plain plan; stage A's price in the regimen against the totals.

numpy and the standard library only; well under a minute.
"""
import numpy as np

K, P = 4, 0.25


def ratio(rho, k=K):
    return 1 - rho * rho * (k - 1) / (1 + (k - 2) * rho)


def twins(rng, n, rho, delta, s_d=0.0):
    g = rng.normal(size=(n, 1))
    d_u = delta + s_d * rng.normal(size=(n, 1))
    e = rng.normal(size=(n, K))
    coin = (rng.random(size=(n, K)) < P).astype(float)
    r = np.sqrt(rho) * g + np.sqrt(1 - rho) * e + d_u * coin
    omega = (coin - P) / (P * (1 - P))
    return r, coin, omega


def kappa_fit(r, include_self):
    sib = r.mean(axis=1, keepdims=True) * np.ones_like(r) if include_self else (r.sum(axis=1, keepdims=True) - r) / (K - 1)
    return (r * sib).sum() / (sib ** 2).sum(), sib


def residual(r, include_self):
    kappa, sib = kappa_fit(r, include_self)
    return r - kappa * sib, kappa


def section1(rng, n=200_000, delta=0.10):
    print("1. self-inclusion bug, K 4, P 0.25, planted coin effect 0.10, 200,000 seeds")
    for rho in (0.0, 0.2):
        r, coin, omega = twins(rng, n, rho, delta)
        for bug in (False, True):
            x, kappa = residual(r, bug)
            r_hat = (x ** 2).mean() / (r ** 2).mean()
            label = (omega * x).mean()
            # coin plant, slope 1 intended: plant = (D - P) * u, u = 1
            plant = (coin - P)
            own_only = r.copy()
            xo, _ = residual(own_only, bug)
            # own residual only: the plant enters X_k's own term, the sibling vector stays unplanted
            sib_unplanted = r.mean(axis=1, keepdims=True) if bug else (r.sum(axis=1, keepdims=True) - r) / (K - 1)
            x_own = (r + plant) - kappa * sib_unplanted
            slope_own = (omega * (x_own - xo)).mean()
            planted = r + plant
            sib_pl = planted.mean(axis=1, keepdims=True) if bug else (planted.sum(axis=1, keepdims=True) - planted) / (K - 1)
            x_all = planted - kappa * sib_pl
            slope_all = (omega * (x_all - xo)).mean()
            # planted shared component a * eta
            a = 0.5
            eta = rng.choice((-1.0, 1.0), size=(n, 1))
            rp = r + a * eta
            xp, _ = residual(rp, bug)
            r_hat_p = (xp ** 2).mean() / (rp ** 2).mean()
            # rho_hat two ways: inverted from r_hat (grid), pairwise covariance (separate code)
            grid = np.linspace(0, 0.99, 991)
            rho_inv = grid[np.argmin(np.abs(ratio(grid) - r_hat))]
            c = np.cov(r.T)
            rho_pair = (c.sum() - np.trace(c)) / (K * (K - 1)) / (np.trace(c) / K)
            pred_inv = ratio((rho_inv + a * a * 1.0 / (r ** 2).mean()) / (1 + a * a / (r ** 2).mean()))
            pred_pair = ratio((rho_pair + a * a / (r ** 2).mean()) / (1 + a * a / (r ** 2).mean()))
            tag = "self included" if bug else "correct      "
            print(f"   rho {rho:.1f} {tag}: kappa {kappa:.3f}  label mean {label:+.4f} (truth +{delta:.2f})  "
                  f"r_hat {r_hat:.3f} (derived {ratio(rho):.3f})")
            print(f"      coin plant slope: own residual only {slope_own:.3f}; in every twin's z {slope_all:.3f}")
            print(f"      shared plant: measured r' {r_hat_p:.3f}; predicted from rho_hat inverted {pred_inv:.3f} "
                  f"(rho_hat {rho_inv:.2f}); from pairwise rho_hat {pred_pair:.3f} (rho_hat {rho_pair:.2f})")


def section2(rng, n=1_000_000, rho=0.3, delta=0.02):
    print(f"\n2. cross-twin covariance: rho {rho}, mean effect {delta}, one shared-prefix coin per twin,"
          f" {n} seeds x {K} twins")
    print("   r_eff = Var(a seed's sum of K twin labels) / (K Var(a plain label in an independent game)):")
    print("   the twin design's variance for the mean effect relative to K times as many independent games")
    print("   s_d (sd of the realized effect across shared chance)   r_hat   r_eff   cross-twin label corr")
    for s_d in (0.0, 0.1, 0.2, 0.3, 0.5):
        r, coin, omega = twins(rng, n, rho, delta, s_d)
        x, _ = residual(r, False)
        t = omega * x
        r_hat = (x ** 2).mean() / (r ** 2).mean()
        seed_var = t.sum(axis=1).var()
        # independent games: every game its own shared chance, plain label omega * R
        ri, _, oi = twins(rng, n, rho, delta, s_d)
        plain_var = (oi[:, 0] * ri[:, 0]).var()
        corr = (seed_var - t.var(axis=0).sum()) / (K * (K - 1)) / t.var()
        print(f"   {s_d:.1f}                                                 {r_hat:.3f}   "
              f"{seed_var / (K * plain_var):.3f}   {corr:+.4f}")


def section3():
    print("\n3. arithmetic")
    print(f"   r at rho 0.030 (cell-baseline noise shared by twins), K 4: {ratio(0.030):.4f}; K -> inf: {1 - 0.030:.3f}")
    rate, one_h, startup_s = 0.577, 22_341, 69
    p1 = rate / one_h
    setup = 0.1 * rate
    startup = startup_s / 3600 * rate
    plain_round = setup + 8 * 100_000 * p1
    pilot = 0.35
    for r in (0.95, 0.9, 0.8):
        twin_round = setup + 8 * (100_000 * r * p1 + K * startup)
        save1 = plain_round - twin_round
        print(f"   r {r:.2f}: a round saves ${save1:.2f} against the plain plan; round 1 net of the pilot "
              f"${save1 - pilot:+.2f}; three rounds net ${3 * save1 - pilot:+.2f}")
    for horizon in (1, 3):
        lo, hi = 0.5, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            gain = horizon * (plain_round - (setup + 8 * (100_000 * mid * p1 + K * startup))) - pilot
            lo, hi = (mid, hi) if gain > 0 else (lo, mid)
        print(f"   break-even r over {horizon} round(s): {lo:.3f}")
    a_table = setup + 12_800 * p1 + startup
    a_totals = 0.72 + 1.13
    print(f"   stage A: regimen's one arm (12,800 one-network games) ${a_table:.2f}; "
          f"totals carry L+T and the two-network claim ${a_totals:.2f}: difference ${a_totals - a_table:.2f}")


def main():
    rng = np.random.default_rng(20261021)
    section1(rng)
    section2(rng)
    section3()


if __name__ == "__main__":
    main()
