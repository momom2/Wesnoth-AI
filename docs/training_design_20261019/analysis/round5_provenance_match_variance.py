"""Round 5 (provenance stance): the variance a corrected match keeps without any critic.

Reads training/metrics/match_luck_20261019/rows.jsonl (23 argmax matches, one row per game,
side A's view) and measures, leave-one-match-out, how much of a match result's variance three
critic-free corrections remove:

  setup    an additive baseline b(map, side_a) + g(faction_a, faction_b) fitted on the other 22
           matches (each match's mean removed first), centred on the uniform setup draw, so the
           corrected mean is unchanged;
  columns  the seven luck columns' sums S (hp, material, material_nonterminal, terminal,
           value_50, value_100, value_200), slope fitted on the other 22 matches;
  both     the two together.

Decisive games only. Output: pooled Var(G)/Var(z) with a game bootstrap interval, per-match
ratios, and the setup share of variance. numpy and the standard library only.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROWS = Path(__file__).resolve().parents[3] / "training/metrics/match_luck_20261019/rows.jsonl"
COLUMNS = ("hp", "material", "material_nonterminal", "terminal", "value_50", "value_100", "value_200")
RNG = np.random.default_rng(20261020)


def load_rows():
    rows = [json.loads(line) for line in open(ROWS)]
    return [r for r in rows if r.get("decisive")]


def design(rows, maps, factions):
    """Setup dummies: map x side_a (one-hot), faction_a x faction_b (one-hot)."""
    n_ms, n_ff = len(maps) * 2, len(factions) ** 2
    X = np.zeros((len(rows), n_ms + n_ff))
    for i, r in enumerate(rows):
        X[i, maps[r["scenario_id"]] * 2 + (r["side_a"] - 1)] = 1.0
        X[i, n_ms + factions[r["faction_a"]] * len(factions) + factions[r["faction_b"]]] = 1.0
    return X


def uniform_mean(coef, n_maps, n_factions):
    """Mean of the additive baseline over the uniform draw of map, side and faction pair."""
    n_ms = n_maps * 2
    return coef[:n_ms].mean() + coef[n_ms:].mean()


def ridge(X, y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def main():
    rows = load_rows()
    maps = {m: i for i, m in enumerate(sorted({r["scenario_id"] for r in rows}))}
    factions = {f: i for i, f in enumerate(sorted({r["faction_a"] for r in rows} | {r["faction_b"] for r in rows}))}
    match = np.array([r["match"] for r in rows])
    z = np.array([float(r["z"]) for r in rows])
    S = np.array([[r["S"][c] for c in COLUMNS] for r in rows])
    D = design(rows, maps, factions)
    names = sorted(set(match))
    # Each match's own mean removed before fitting (matches differ in strength).
    zc = z.copy()
    for m in names:
        zc[match == m] -= z[match == m].mean()

    out = {k: np.zeros_like(z) for k in ("raw", "setup", "columns", "both")}
    for m in names:
        test, train = match == m, match != m
        # setup alone; small ridge keeps the 1,512-cell-free additive model stable
        cs = ridge(D[train], zc[train], 1.0)
        base = D[test] @ cs - uniform_mean(cs, len(maps), len(factions))
        # columns alone
        Sc = S[train] - S[train].mean(0)
        bc = np.linalg.lstsq(Sc, zc[train], rcond=None)[0]
        # both, fitted jointly
        J = np.hstack([D[train], S[train]])
        cj = ridge(J, zc[train], 1.0)
        cj_setup, cj_cols = cj[: D.shape[1]], cj[D.shape[1]:]
        base_j = D[test] @ cj_setup - uniform_mean(cj_setup, len(maps), len(factions))
        out["raw"][test] = z[test]
        out["setup"][test] = z[test] - base
        out["columns"][test] = z[test] - S[test] @ bc
        out["both"][test] = z[test] - base_j - S[test] @ cj_cols

    def pooled(g, idx=None):
        idx = np.arange(len(z)) if idx is None else idx
        num = den = 0.0
        for m in names:
            sel = idx[match[idx] == m]
            if len(sel) < 2:
                continue
            num += np.var(g[sel]) * len(sel)
            den += np.var(z[sel]) * len(sel)
        return num / den

    print(f"decisive games {len(z)}, matches {len(names)}, maps {len(maps)}, factions {len(factions)}")
    boots = {k: [] for k in ("setup", "columns", "both")}
    for _ in range(200):
        idx = RNG.integers(0, len(z), len(z))
        for k in boots:
            boots[k].append(pooled(out[k], idx))
    for k in ("setup", "columns", "both"):
        lo, hi = np.percentile(boots[k], [2.5, 97.5])
        print(f"{k:8s} pooled Var(G)/Var(z) = {pooled(out[k]):.3f}  (95% bootstrap {lo:.3f}-{hi:.3f})")
    print("\nper match (n, setup, columns, both):")
    for m in names:
        sel = match == m
        v = np.var(z[sel])
        print(f"  {m:24s} {sel.sum():5d}  {np.var(out['setup'][sel]) / v:.3f}  "
              f"{np.var(out['columns'][sel]) / v:.3f}  {np.var(out['both'][sel]) / v:.3f}")
    # Mean preservation: per match, corrected minus raw mean, with the SE of that difference.
    t2 = []
    for m in names:
        sel = match == m
        diff = out["both"][sel] - z[sel]
        t = diff.mean() / (diff.std(ddof=1) / np.sqrt(sel.sum()))
        t2.append(t * t)
    t2 = np.array(t2)
    print(f"\nmean preservation (both): sum of t^2 over {len(t2)} matches = {t2.sum():.1f} "
          f"(chi-square {len(t2)} df: mean {len(t2)}, 95th percentile about {len(t2) + 1.645 * np.sqrt(2 * len(t2)):.0f}); "
          f"largest |t| {np.sqrt(t2.max()):.2f}")
    # The setup share, in-sample (an upper bound on what the fitted baseline can remove).
    cs_all = ridge(D, zc, 1.0)
    share_in = 1 - np.var(zc - D @ cs_all) / np.var(zc)
    print(f"\nsetup share of variance, in-sample additive fit (upper side): {share_in:.3f}")
    # Elo per unit of mean z at an even match: 400 / ln 10 * 2 = 347.4.
    for k in ("columns", "both"):
        r = pooled(out[k])
        n5 = r * (347.4 / 5.0) ** 2
        print(f"{k}: decisive games for SE 5 Elo {n5:.0f}; uncorrected {(347.4 / 5.0) ** 2:.0f}")


if __name__ == "__main__":
    sys.exit(main())
