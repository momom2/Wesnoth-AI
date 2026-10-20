"""Round 5 (provenance stance): what a side-swapped paired design would leave of a match's variance.

Reads training/metrics/match_luck_20261019/rows.jsonl (decisive games). Each match's mean is
removed; the seven luck columns are fitted leave-one-match-out (as in
round5_provenance_match_variance.py); then, on the column-corrected residual, the method of
moments of round5_provenance_setup_cells.py estimates the variance of the true means of the
full setup cell (map, side of A, faction of A, faction of B).

A pair plays one setup twice with the players' sides and factions swapped. If the setup's effect
on A's result is a property of the setup (the swapped game hands A exactly what B held), the
pair's mean removes it entirely; what stays per game is the within-cell variance. The script
reports, as a share of Var(z):
  columns only          what the corrected match keeps with the columns alone;
  columns + additive    the fitted additive baseline (the corrected match of the design);
  columns + pairs       the paired design: columns-corrected variance minus the full cell share.
The last is an estimate under the stated symmetry; player-by-setup interactions are signal, not
removed, and are inside the cell share here, so it is an upper side of what pairing removes.
numpy and the standard library only.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROWS = Path(__file__).resolve().parents[3] / "training/metrics/match_luck_20261019/rows.jsonl"
COLUMNS = ("hp", "material", "material_nonterminal", "terminal", "value_50", "value_100", "value_200")
RNG = np.random.default_rng(20261020)


def cell_share(keys, y, total):
    groups = collections.defaultdict(list)
    for k, v in zip(keys, y):
        groups[k].append(v)
    means, noise, weights = [], [], []
    for vals in groups.values():
        if len(vals) < 2:
            continue
        a = np.array(vals)
        means.append(a.mean())
        noise.append(a.var(ddof=1) / len(a))
        weights.append(len(a))
    w = np.array(weights, float)
    m = np.array(means)
    mu = np.average(m, weights=w)
    return (np.average((m - mu) ** 2, weights=w) - np.average(noise, weights=w)) / total


def main():
    rows = [json.loads(line) for line in open(ROWS)]
    rows = [r for r in rows if r.get("decisive")]
    z = np.array([float(r["z"]) for r in rows])
    match = np.array([r["match"] for r in rows])
    S = np.array([[r["S"][c] for c in COLUMNS] for r in rows])
    zc = z.copy()
    for m in set(match):
        zc[match == m] -= z[match == m].mean()
    resid = np.zeros_like(z)
    for m in set(match):
        test, train = match == m, match != m
        Sc = S[train] - S[train].mean(0)
        b = np.linalg.lstsq(Sc, zc[train], rcond=None)[0]
        resid[test] = zc[test] - S[test] @ b
    total = np.var(zc)
    cols = np.var(resid) / total
    keys = [(r["scenario_id"], r["side_a"], r["faction_a"], r["faction_b"]) for r in rows]
    share = cell_share(keys, resid, total)
    halves = []
    for _ in range(50):
        perm = RNG.permutation(len(rows))
        h1, h2 = perm[: len(rows) // 2], perm[len(rows) // 2:]
        halves.append(cell_share([keys[i] for i in h1], resid[h1], np.var(zc[h1]))
                      - cell_share([keys[i] for i in h2], resid[h2], np.var(zc[h2])))
    se = np.std(halves) / 2
    print(f"decisive games {len(z)}; Var(z - match mean) {total:.4f}")
    print(f"columns only (leave-one-match-out): {cols:.3f}")
    print(f"full setup cell share of the column-corrected residual: {share:.3f} +- {se:.3f} (split-half)")
    print(f"columns + pairs (upper side of what pairing removes): {cols - share:.3f}")
    print("columns + additive baseline: 0.747 (round5_provenance_match_variance.out)")


if __name__ == "__main__":
    sys.exit(main())
