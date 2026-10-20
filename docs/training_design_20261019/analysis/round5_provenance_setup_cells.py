"""Round 5 (provenance stance): the setup draw's full share of a match result's variance.

Reads training/metrics/match_luck_20261019/rows.jsonl (decisive games). After removing each
match's mean, estimates by method of moments the variance of the true cell means over the full
setup cell (map, side of player A, faction of A, faction of B), interactions included:

    share = [Var_between(cell means) - mean(within-cell variance / n_cell)] / Var(z),

pooling cells with at least 2 games, and the same for the additive margins. This is the variance a
paired design that replays each setup with the players' sides swapped (Fishtest's pentanomial
pairs) could remove, against what the additive baseline of round5_provenance_match_variance.py
removes. numpy and the standard library only.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROWS = Path(__file__).resolve().parents[3] / "training/metrics/match_luck_20261019/rows.jsonl"
RNG = np.random.default_rng(20261020)


def moments_share(keys, zc):
    groups = collections.defaultdict(list)
    for k, v in zip(keys, zc):
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
    between = np.average((m - mu) ** 2, weights=w)
    return (between - np.average(noise, weights=w)) / np.var(zc), len(m), float(w.mean())


def main():
    rows = [json.loads(line) for line in open(ROWS)]
    rows = [r for r in rows if r.get("decisive")]
    z = np.array([float(r["z"]) for r in rows])
    match = np.array([r["match"] for r in rows])
    zc = z.copy()
    for m in set(match):
        zc[match == m] -= z[match == m].mean()
    specs = {
        "full cell (map, side, faction A, faction B)": lambda r: (r["scenario_id"], r["side_a"], r["faction_a"], r["faction_b"]),
        "map x side": lambda r: (r["scenario_id"], r["side_a"]),
        "faction A x faction B": lambda r: (r["faction_a"], r["faction_b"]),
        "side alone": lambda r: r["side_a"],
    }
    for name, key in specs.items():
        keys = [key(r) for r in rows]
        share, cells, per = moments_share(keys, zc)
        # Resampling with replacement biases the within-cell variance down; use random halves
        # instead: each half's estimate is unbiased, and half the spread of their differences
        # estimates the full sample's standard error.
        diffs = []
        for _ in range(50):
            perm = RNG.permutation(len(rows))
            h1, h2 = perm[: len(rows) // 2], perm[len(rows) // 2:]
            s1 = moments_share([keys[i] for i in h1], zc[h1])[0]
            s2 = moments_share([keys[i] for i in h2], zc[h2])[0]
            diffs.append(s1 - s2)
        se = np.std(diffs) / 2
        print(f"{name:45s} share {share:+.3f} +- {se:.3f} (split-half), cells {cells}, games per cell {per:.1f}")


if __name__ == "__main__":
    sys.exit(main())
