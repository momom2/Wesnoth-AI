"""Round 5, critique of the provenance stance: how much of the setup's variance would
side-swapped pairs actually remove?

Reads training/metrics/match_luck_20261019/rows.jsonl (decisive games), removes each match's
mean, and corrects by the seven luck columns leave-one-match-out exactly as
round5_provenance_paired_setups.py does (its function is imported).

A pair plays cell c = (map, side of A, faction of A, faction of B) and its swap
c' = (map, other side, faction of B, faction of A). With f the cell's true effect on A's result
and tau2 = Var(f), the setup's per-game variance is tau2 unpaired and tau2 + Cov(f(c), f(c'))
paired, so pairing removes -Cov(f(c), f(c')). The provenance stance assumes exact antisymmetry,
Cov = -tau2, which makes its 0.696 a lower bound of what pairs keep. Here Cov is estimated
directly: the product of the two cells' means (grand mean removed) is unbiased for Cov, since
their noises are independent. Each unordered pair {c, c'} counts once, weight
n_c n_c' / (n_c + n_c'); the SE is a bootstrap over cell pairs.

numpy and the standard library only.
"""
import collections
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ps", HERE / "round5_provenance_paired_setups.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)
RNG = np.random.default_rng(20261020)


def swap(key):
    scenario, side, fa, fb = key
    return (scenario, 3 - side, fb, fa)


def cross_cov(keys, y):
    groups = collections.defaultdict(list)
    for k, v in zip(keys, y):
        groups[k].append(v)
    mu = float(np.mean(y))
    pairs, seen = [], set()
    for k in groups:
        k2 = swap(k)
        if k2 not in groups or (k2, k) in seen:
            continue
        seen.add((k, k2))
        n1, n2 = len(groups[k]), len(groups[k2])
        pairs.append(((np.mean(groups[k]) - mu) * (np.mean(groups[k2]) - mu), n1 * n2 / (n1 + n2)))
    prod = np.array([p for p, _ in pairs])
    w = np.array([w for _, w in pairs])
    est = float(np.average(prod, weights=w))
    boots = []
    for _ in range(500):
        i = RNG.integers(0, len(prod), len(prod))
        boots.append(np.average(prod[i], weights=w[i]))
    return est, float(np.std(boots)), len(prod)


def main():
    rows = [json.loads(line) for line in open(ps.ROWS)]
    rows = [r for r in rows if r.get("decisive")]
    z = np.array([float(r["z"]) for r in rows])
    match = np.array([r["match"] for r in rows])
    S = np.array([[r["S"][c] for c in ps.COLUMNS] for r in rows])
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
    keys = [(r["scenario_id"], r["side_a"], r["faction_a"], r["faction_b"]) for r in rows]
    tau2 = ps.cell_share(keys, resid, total) * total
    cov, se, npairs = cross_cov(keys, resid)
    cols = np.var(resid) / total
    print(f"decisive games {len(z)}; cell pairs with both cells observed {npairs}")
    print(f"column-corrected share {cols:.3f}; full-cell tau2 share {tau2 / total:.3f}")
    print(f"Cov(f(c), f(swap c)) share {cov / total:+.3f} +- {se / total:.3f} (exact antisymmetry: {-tau2 / total:+.3f})")
    print(f"pairs keep {cols + cov / total:.3f} +- {se / total:.3f}; the additive baseline keeps 0.747 "
          f"(round5_provenance_match_variance.out); the stance's lower bound 0.696")
    for name, keyf in (("faction pair only", lambda r: ("", r["side_a"], r["faction_a"], r["faction_b"])),):
        k2 = [keyf(r) for r in rows]
        # side kept in the key so the swap maps (side, fa, fb) to (other side, fb, fa)
        c2, s2, n2 = cross_cov(k2, resid)
        t2 = ps.cell_share(k2, resid, total)
        print(f"{name}: tau2 share {t2:.3f}, swap covariance share {c2 / total:+.3f} +- {s2 / total:.3f} ({n2} pairs)")


if __name__ == "__main__":
    sys.exit(main())
