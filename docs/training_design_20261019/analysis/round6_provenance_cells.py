"""Round 6, provenance stance: the label's pre-coin baseline when the setup term is the full cell.

The coin label's residual R = z - c - beta*S keeps 0.732 of Var(z) in own1 self-play with the
additive setup baseline (map x side + faction pair) and the seven columns
(round5_provenance_match_variance.out). A coin round has 100,000 games a batch, so the full cell
(map, side of A, faction of A, faction of B; at most 21 x 2 x 36 = 1,512 cells) can be fitted on
earlier batches with hundreds of games a cell. Its asymptotic kept ratio is the within-cell
variance of the column-corrected result.

Reads training/metrics/match_luck_20261019/rows.jsonl (decisive games); columns fitted
leave-one-match-out on match-centred results (round5_provenance_paired_setups.py's procedure).
Reports, as a share of Var(z) (match-centred, as round5_provenance_paired_setups.py):
  pooled over the 23 matches: column-corrected share, full-cell share (method of moments),
    within-cell share = their difference;
  own1 self-play only (gen_own1_selfplay, own64_selfplay): the same, cells from those 1,600
    games alone (noisy: about 1.6 games a cell), with a bootstrap over cells;
  the finite-sample term a cell fitted on 700,000 earlier games adds (sigma_e^2 x cells / games).
numpy and the standard library only.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ps", HERE / "round5_provenance_paired_setups.py")
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)
RNG = np.random.default_rng(20261020)
SELF_PLAY = ("gen_own1_selfplay", "own64_selfplay")


def corrected():
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
    keys = [(r["scenario_id"], r["side_a"], r["faction_a"], r["faction_b"]) for r in rows]
    return zc, match, resid, keys


def shares(resid, keys, var_z):
    col = float(np.var(resid)) / var_z
    cell = float(ps.cell_share(keys, resid, var_z * 1.0))
    return col, cell, col - cell


def main():
    z, match, resid, keys = corrected()
    var_z = float(np.var(z))
    col, cell, within = shares(resid, keys, var_z)
    n_cells = len(set(keys))
    print(f"pooled, {len(z)} decisive games, {n_cells} cells observed: columns keep {col:.3f}, "
          f"full-cell share {cell:.3f}, within-cell {within:.3f}")
    sp = np.isin(match, SELF_PLAY)
    r_sp = resid[sp]
    k_sp = [k for k, s in zip(keys, sp) if s]
    vz_sp = float(np.var(z[sp]))
    col_sp, cell_sp, within_sp = shares(r_sp, k_sp, vz_sp)
    by_cell = {}
    for j, k in enumerate(k_sp):
        by_cell.setdefault(k, []).append(j)
    cells = list(by_cell)
    boots = []
    for _ in range(300):
        picked = RNG.integers(0, len(cells), len(cells))
        i, kk = [], []
        for n, c in enumerate(picked):
            for j in by_cell[cells[c]]:
                i.append(j)
                kk.append((n,) + cells[c])   # a resampled cell keeps its games together, under a fresh key
        i = np.array(i)
        boots.append(shares(r_sp[i], kk, vz_sp)[2])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"own1 self-play, {sp.sum()} games, {len(set(k_sp))} cells: columns keep {col_sp:.3f}, "
          f"full-cell share {cell_sp:.3f}, within-cell {within_sp:.3f} (95% bootstrap {lo:.3f} to {hi:.3f})")
    print(f"own1 self-play with the pooled cell share: within-cell {col_sp - cell:.3f}")
    for games in (100_000, 700_000):
        print(f"finite-sample term of a cell baseline fitted on {games} games: "
              f"{within * 1512 / games:.4f} (1,512 cells)")
    print("against the additive baseline's 0.730 / 0.734 in own1 self-play (round5_provenance_match_variance.out)")


if __name__ == "__main__":
    main()
