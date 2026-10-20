"""Round 5, critique of the simplify stance: four more checks of its numbers.

1. Where the replay's argmax disagreements sit by gap. Per side-game of
   training/metrics/critic_step1_20261009/prior_gaps.games.jsonl (parity3, 100
   side-games, replayed on another box): disagreements = decisions - agree -
   premoves, regressed without intercept on the game's counts of readings with
   top-two gap in [0, 0.0136), [0.0136, 0.1), [0.1, 0.35), [0.35, inf).
   The last coefficient bounds the flip rate per decision at G >= 0.35, the
   quantity S0's "predicted 0" stands for. Bootstrap over side-games.

2. S1's early stop under a true null. The proposal skips batches 5-8 when
   batches 2-4's progressive readings put batch 8's gain below 5.3 Elo at its
   upper bound. Under its own Gaussian model (round5_simplify_power.py: label
   noise 3.91, d_eff 776 taken as known, 62 band decisions per side-game,
   100,000-game batches), each progressive slope reads v / eta_b with SE
   eta_b * SE(beta_b); combined by inverse variance, the 2-SE upper bound on v
   is compared with the v that gives 5.3 Elo at batch 8. Reports the chance the
   skip fires at sd 0 (null), 0.0015, and the floor.

3. The adoption rule's "corrected and PURE agree within 2 combined SE": the
   SE of the paired difference (the correction's own variance over the same
   games) against the root-sum-square the wording implies. The correction's
   variance per game is recomputed leave-one-match-out on
   training/metrics/match_luck_20261019/rows.jsonl exactly as
   round5_provenance_match_variance.py does (setup + seven columns, ridge 1).

4. Critic re-entry "with no new games": the network's token outputs x_i at
   every band decision are not stored, so a student refit replays the
   training half and a re-read of the refitted student replays the forecast
   half, at 2,440 decisions a second and 465 decisions a game (BACKLOG.md,
   generation round 5), $0.42 an hour.

numpy and the standard library only; under a minute.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GAPS = ROOT / "training/metrics/critic_step1_20261009/prior_gaps.games.jsonl"
ROWS = ROOT / "training/metrics/match_luck_20261019/rows.jsonl"
ELO_PER_Z = 800 / math.log(10)
RNG = np.random.default_rng(20261020)


def flips_by_gap():
    games = [json.loads(line) for line in open(GAPS)]
    edges = (0.0, 0.0136, 0.1, 0.35, np.inf)
    X, y = [], []
    for g in games:
        gaps = np.array([r["gaps"][0] for r in g["readings"] if r["gaps"]])
        X.append([np.sum((gaps >= lo) & (gaps < hi)) for lo, hi in zip(edges[:-1], edges[1:])])
        y.append(g["decisions"] - g["agree"] - g.get("premoves", 0))
    X, y = np.array(X, float), np.array(y, float)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    boots = []
    for _ in range(2000):
        i = RNG.integers(0, len(y), len(y))
        boots.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    print("1. disagreements per reading, by top-two gap bin (no intercept, 100 side-games):")
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        print(f"   [{a}, {b}): readings {X[:, k].sum():6.0f}, rate {coef[k]:+.5f} (95% bootstrap {lo[k]:+.5f} to {hi[k]:+.5f})")
    print(f"   total disagreements {y.sum():.0f}. A negative rate is a misfit of the additive model "
          f"(per-decision flags are not in the record), so the record neither confirms nor bounds "
          f"zero flips at G >= 0.35.")


def early_stop(n_band=62.0, d_eff=776, sigma2=3.91, batch=100_000):
    n_b = 2 * n_band * batch                       # labels per batch
    eta = lambda k: sigma2 * d_eff / (k * n_b)     # student after k training batches

    def gain_elo(v):
        s = math.sqrt(v + eta(4))
        return ELO_PER_Z * n_band * (v / s) * 0.3989422804

    lo, hi = 1e-10, 1e-3
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if gain_elo(mid) < 5.3 else (lo, mid)
    v_thr = hi
    print(f"\n2. early stop: v giving 5.3 Elo at batch 8 = {v_thr:.3e} (sd {math.sqrt(v_thr):.4f})")
    for sd in (0.0, 0.0015, math.sqrt(v_thr)):
        v = sd * sd
        # batches 2, 3, 4 read by students trained on 1, 2, 3 batches; small-v linearisation v_hat = beta_hat * eta
        prec = 0.0
        for k in (1, 2, 3):
            se_beta = math.sqrt(sigma2 / (n_b * (v + eta(k))))
            se_v = se_beta * (v + eta(k))
            prec += 1 / se_v ** 2
        se_comb = math.sqrt(1 / prec)
        # fires when v_hat + 2 se < v_thr, v_hat ~ N(v, se_comb)
        z = (v_thr - 2 * se_comb - v) / se_comb
        p_fire = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        print(f"   true sd {sd:.4f}: combined SE of v_hat {se_comb:.2e}; upper bound at v_hat = v reads "
              f"{gain_elo(v + 2 * se_comb):.1f} Elo; P(skip fires) {p_fire:.2f}")


def paired_se(n_games=12800):
    rows = [json.loads(line) for line in open(ROWS)]
    rows = [r for r in rows if r.get("decisive")]
    cols = ("hp", "material", "material_nonterminal", "terminal", "value_50", "value_100", "value_200")
    maps = {m: i for i, m in enumerate(sorted({r["scenario_id"] for r in rows}))}
    facs = {f: i for i, f in enumerate(sorted({r["faction_a"] for r in rows} | {r["faction_b"] for r in rows}))}
    nm, nf = len(maps) * 2, len(facs) ** 2
    D = np.zeros((len(rows), nm + nf))
    for i, r in enumerate(rows):
        D[i, maps[r["scenario_id"]] * 2 + r["side_a"] - 1] = 1
        D[i, nm + facs[r["faction_a"]] * len(facs) + facs[r["faction_b"]]] = 1
    match = np.array([r["match"] for r in rows])
    z = np.array([float(r["z"]) for r in rows])
    S = np.array([[r["S"][c] for c in cols] for r in rows])
    zc = z.copy()
    for m in set(match):
        zc[match == m] -= z[match == m].mean()
    corr = np.zeros_like(z)
    J = np.hstack([D, S])
    for m in sorted(set(match)):
        te, tr = match == m, match != m
        c = np.linalg.solve(J[tr].T @ J[tr] + np.eye(J.shape[1]), J[tr].T @ zc[tr])
        cs = c[: D.shape[1]]
        base = D[te] @ cs - (cs[:nm].mean() + cs[nm:].mean())
        corr[te] = base + S[te] @ c[D.shape[1]:]
    var_corr = np.mean([np.var(corr[match == m]) for m in set(match)])
    var_g = np.mean([np.var(z[match == m] - corr[match == m]) for m in set(match)])
    var_z = np.mean([np.var(z[match == m]) for m in set(match)])
    se_pair = ELO_PER_Z * math.sqrt(var_corr / n_games)
    se_corr = ELO_PER_Z * math.sqrt(var_g / n_games)
    se_pure = ELO_PER_Z * math.sqrt(var_z / n_games)
    print(f"\n3. per game: Var(z) {var_z:.3f}, Var(corrected) {var_g:.3f}, Var(correction) {var_corr:.3f}; "
          f"at {n_games} games SE corrected {se_corr:.2f}, PURE {se_pure:.2f}, "
          f"root-sum-square {math.hypot(se_corr, se_pure):.2f}, paired difference {se_pair:.2f} Elo")


def reentry_cost():
    hours_half = 400_000 * 465 / 2440 / 3600
    print(f"\n4. replay of one 400,000-game half: {hours_half:.1f} box-hours, ${hours_half * 0.42:.1f}; "
          f"both halves {2 * hours_half:.1f} h, ${2 * hours_half * 0.42:.1f}")


def main():
    flips_by_gap()
    early_stop()
    paired_se()
    reentry_cost()


if __name__ == "__main__":
    sys.exit(main())
