"""Round 9, provenance stance: the coin label's residual on batch 1's own games, and
whether the setup-cell baseline b(cell) earns its place.

Reads training/metrics/near_tie_coins_20261020/rows_b00..b06.jsonl (z, side A,
map, factions and the seven luck sums S per game; decisive games only) and the
batch's frozen beta (readout_b.json). Every share is E[R^2] / E[z^2], z = +-1,
so E[z^2] = 1 exactly; the label's variance is E[omega^2] E[R^2] up to the
weak dependence of R on the coins (round9_provenance_cells.out section 1
measures E[T^2] directly).

Residuals, each cross-fitted where it is fitted (leave one match out, seven matches):
  R_z      = z                                   (no variate)
  R_beta   = z - beta . S                         (the design's batch-1 label; beta from the luck walk)
  R_refit  = z - beta_k . S                       (beta refitted on the other six matches)
  R_cell   = z - beta . S - b_k(cell)             (cell = map, side of A, faction of A, faction of B)
  R_add    = z - beta . S - b_k(map, side) - b_k(faction pair)   (additive baseline)
and the method-of-moments within-cell share (the asymptote of R_cell with
infinitely many earlier games), by a U-statistic over the seven matches.
numpy and the standard library only.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REC = HERE.parents[2] / "training" / "metrics" / "near_tie_coins_20261020"
MATCHES = ("b00", "b01", "b02", "b03", "b04", "b05", "b06")
COLUMNS = ("hp", "material", "material_nonterminal", "terminal", "value_50", "value_100", "value_200")


def load():
    rows = []
    for m in MATCHES:
        with open(REC / f"rows_{m}.jsonl", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if "game" in r and r.get("decisive") and "S" in r:
                    rows.append(r)
    z = np.array([float(r["z"]) for r in rows])
    S = np.array([[r["S"][c] for c in COLUMNS] for r in rows])
    match = np.array([r["match"] for r in rows])
    cell = [(r["scenario_id"], r["side_a"], r["faction_a"], r["faction_b"]) for r in rows]
    mapside = [(r["scenario_id"], r["side_a"]) for r in rows]
    facs = [(r["faction_a"], r["faction_b"]) for r in rows]
    return z, S, match, cell, mapside, facs


def cell_means_cf(y, keys, match):
    """Leave-one-match-out cell means of y (0 for a cell unseen elsewhere)."""
    out = np.zeros_like(y)
    for m in MATCHES:
        test = match == m
        acc = defaultdict(lambda: [0.0, 0])
        for v, k, t in zip(y, keys, test):
            if not t:
                a = acc[k]
                a[0] += v
                a[1] += 1
        for i in np.nonzero(test)[0]:
            a = acc.get(keys[i])
            out[i] = a[0] / a[1] if a and a[1] else 0.0
    return out


def additive_cf(y, keys1, keys2, match, sweeps=4):
    """Leave-one-match-out additive baseline b1(k1) + b2(k2), by backfitting."""
    out = np.zeros_like(y)
    for m in MATCHES:
        train = match != m
        b1, b2 = defaultdict(float), defaultdict(float)
        for _ in range(sweeps):
            for keys, b, other, ob in ((keys1, b1, keys2, b2), (keys2, b2, keys1, b1)):
                acc = defaultdict(lambda: [0.0, 0])
                for i in np.nonzero(train)[0]:
                    a = acc[keys[i]]
                    a[0] += y[i] - ob[other[i]]
                    a[1] += 1
                b.clear()
                b.update({k: s / n for k, (s, n) in acc.items()})
        for i in np.nonzero(~train)[0]:
            out[i] = b1.get(keys1[i], 0.0) + b2.get(keys2[i], 0.0)
    return out


def within_cell_mom(y, keys, match):
    """Method of moments: E[y^2] - sum over cells of share x (unbiased mu_c^2 from
    pairs of distinct matches)."""
    by = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for v, k, m in zip(y, keys, match):
        a = by[k][m]
        a[0] += v
        a[1] += 1
    n_tot = len(y)
    between = 0.0
    for k, per in by.items():
        s = np.array([a[0] for a in per.values()])
        n = np.array([a[1] for a in per.values()], dtype=float)
        den = n.sum() ** 2 - (n * n).sum()
        if den > 0:
            between += n.sum() / n_tot * (s.sum() ** 2 - (s * s).sum()) / den
    return float(np.mean(y * y)) - between


def main():
    z, S, match, cell, mapside, facs = load()
    beta = np.asarray(json.loads((REC / "readout_b.json").read_text())["beta"]["values"])
    n = len(z)
    print(f"{n} decisive games of batch 1 (own1 coin self-play), E[z^2] = {np.mean(z * z):.4f}, mean z {z.mean():+.4f}")
    r_beta = z - S @ beta
    refit = np.zeros_like(z)
    for m in MATCHES:
        tr, te = match != m, match == m
        b = np.linalg.lstsq(S[tr], z[tr], rcond=None)[0]
        refit[te] = z[te] - S[te] @ b
    b_all = np.linalg.lstsq(S, z, rcond=None)[0]
    print(f"  in-sample beta on batch 1: {np.round(b_all, 5).tolist()}")
    print(f"  frozen beta (luck walk):   {np.round(beta, 5).tolist()}")
    r_cell = r_beta - cell_means_cf(r_beta, cell, match)
    r_add = r_beta - additive_cf(r_beta, mapside, facs, match)
    mom = within_cell_mom(r_beta, cell, match)
    se = lambda r: np.std(r * r) / np.sqrt(n)  # noqa: E731
    for name, r in (("R_z (no variate)", z), ("R_beta (design, batch 1)", r_beta), ("R_refit (beta cross-fitted)", refit),
                    ("R_cell (cell baseline, cross-fitted)", r_cell), ("R_add (additive baseline, cross-fitted)", r_add)):
        print(f"  {name:40s} share {np.mean(r * r):.4f} (+- {se(r):.4f})")
    print(f"  within-cell share, method of moments (asymptote of R_cell): {mom:.4f}; cells observed {len(set(cell))}")
    print(f"  games for the same label SE, relative to R_beta: R_cell {np.mean(r_cell ** 2) / np.mean(r_beta ** 2):.3f}, "
          f"asymptote {mom / np.mean(r_beta ** 2):.3f}")
    print(f"  mean games per cell {n / len(set(cell)):.1f}; first mover's mean z (side A = side 1): "
          f"{np.mean([zz for zz, c in zip(z, cell) if c[1] == 1]):+.4f}")
    weighted(z, r_beta, r_cell)


def weighted(z, r_beta, r_cell):
    """The same shares weighted by each game's band decisions (the label's own
    weighting), on the matches whose coin rows are on disk (b00, b06)."""
    games = []
    for m in MATCHES:
        with open(REC / f"rows_{m}.jsonl", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if "game" in r and r.get("decisive") and "S" in r:
                    games.append((m, r["game"]))
    at = {g: i for i, g in enumerate(games)}
    idx, cnt = [], []
    for m in ("b00", "b06"):
        with open(REC / f"coins_{m}.jsonl", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                k = (r.get("match"), r.get("game"))
                if k in at and "band" in r:
                    idx.append(at[k])
                    cnt.append(len(r["band"]["d"]) - sum(r["band"].get("refused", [])))
    idx, w = np.array(idx), np.array(cnt, dtype=float)
    for name, r in (("R_z", z), ("R_beta", r_beta), ("R_cell", r_cell)):
        rr = r[idx] ** 2
        print(f"  weighted by band decisions ({len(idx)} games of b00 and b06): {name} share {np.sum(w * rr) / w.sum():.4f}, "
              f"unweighted on the same games {rr.mean():.4f}")


if __name__ == "__main__":
    main()
