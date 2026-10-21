"""Round 10, critique of the simplify stance: is R "the head with a unit block and zero token weights"?

round10_simplify.md defines the head's unit block per candidate: phi(a) carries a 43-wide faction|unit
one-hot when a is a recruit. Then x = phi(a1) - phi(a0) is non-zero on EVERY band pair with one recruit
candidate (recruit>move, move>recruit, recruit>end_turn, ...), not only on recruit>recruit pairs of two
different units, where R was fitted (round9_synthesis_checks.py, 262 cells). This script reads, from batch
1's existing per-cell fold sums (labels T, b = 0, the same fold hash and beta for both partitions):

1. how many band decisions per side-game carry a non-zero unit block outside R's cells;
2. what R's 43 values do there when played through the per-candidate block (switch iff w'x > 0):
   switches per side-game and cross-fitted first-order welfare (rule fitted on 19 folds of the rr cells,
   read on the 20th fold's mixed cells);
3. what the processor would fit with x = the per-candidate unit block over all pairs (rr + mixed) at
   ridge 5,000: its distance from R's values, and its cross-fitted welfare on rr cells, on mixed cells and
   in total. This is R0's known answer as the proposal specifies it.
4. R's cross-fitted F at ridges beyond the proposal's grid, up to lambda -> infinity (one shared ridge's worst case).

Reads round9_provenance_cells_b0*.json (rr_units) and round9_provenance_unit_cells_b0*.json.gz.
numpy and the standard library only; no network, no simulator; a few seconds.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import round9_synthesis_checks as r9  # noqa: E402

_spec = importlib.util.spec_from_file_location("uc", HERE / "round9_provenance_unit_cells.py")
uc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(uc)

LAM = 5_000.0
ELO = r9.ELO
FOLDS = r9.FOLDS
MIXED = ("recruit>move", "move>recruit", "recruit>end_turn", "end_turn>recruit",
         "recruit>attack", "attack>recruit")


def r_weights(X, s, n, lam=LAM):
    A = X.T @ (X * n[:, None]) + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ s)


def mixed_design(cells, names):
    """Per-candidate unit block difference for mixed pairs: +e(u1) if a1 recruits, -e(u0) if a0 recruits."""
    at = {u: i for i, u in enumerate(names)}
    X = np.zeros((len(cells), len(names)))
    missing = 0
    for i, c in enumerate(cells):
        sub, fac, pair = c.split("|", 2)
        d0, d1 = pair.split(">", 1)
        k0, k1 = sub.split(">")
        if k0 == "recruit":
            j = at.get(f"{fac}|{d0}")
            if j is None:
                missing += 1
            else:
                X[i, j] -= 1.0
        if k1 == "recruit":
            j = at.get(f"{fac}|{d1}")
            if j is None:
                missing += 1
            else:
                X[i, j] += 1.0
    return X, missing


def main():
    cells, per_match, side_games = r9.load()
    Xr, names = r9.design(cells)
    s_r = sum(m["s_T"] for m in per_match)
    ss_r = sum(m["ss_T"] for m in per_match)
    n_r = sum(m["n"] for m in per_match)
    sg = sum(side_games)

    _data, ucells, acc = uc.merged()
    subs = np.array([c.split("|", 1)[0] for c in ucells])
    mask = np.isin(subs, MIXED)
    mcells = [c for c, m in zip(ucells, mask) if m]
    s_m, ss_m, n_m = acc["s_T"][mask], acc["ss_T"][mask], acc["n"][mask]
    Xm, missing = mixed_design(mcells, names)
    nonzero = np.abs(Xm).sum(1) > 0
    print(f"batch 1: {sg:,} side-games; R: {len(cells)} rr cells, {len(names)} unit values, "
          f"{n_r.sum() / sg:.2f} labels per side-game")
    print("\n1. Band pairs with one recruit candidate (the per-candidate unit block is non-zero there)")
    for sub in MIXED:
        k = subs[mask] == sub
        print(f"   {sub:18s} {n_m[k].sum() / sg:.2f} per side-game")
    print(f"   total {n_m.sum() / sg:.2f} per side-game, of which {n_m[nonzero].sum() / sg:.2f} with a unit "
          f"R knows ({missing} cell-sides with a recruit unit outside R's 43)")

    print("\n2. R's 43 values played through the per-candidate block on those pairs (switch iff w'x > 0), "
          "cross-fitted")
    tot = var = sw = 0.0
    by_sub = {s: [0.0, 0.0, 0.0] for s in MIXED}
    msubs = subs[mask]
    for f in range(FOLDS):
        so, no = s_r.sum(1) - s_r[:, f], n_r.sum(1) - n_r[:, f]
        w = r_weights(Xr, so, no)
        ch = (Xm @ w) > 0
        tot += s_m[ch, f].sum()
        var += ss_m[ch, f].sum()
        sw += n_m[ch, f].sum()
        for sub in MIXED:
            k = ch & (msubs == sub)
            by_sub[sub][0] += s_m[k, f].sum()
            by_sub[sub][1] += ss_m[k, f].sum()
            by_sub[sub][2] += n_m[k, f].sum()
    print(f"   extra switches {sw / sg:.2f} per side-game (R's own: 2.19); first-order welfare "
          f"{tot / sg * ELO:+.1f} +- {math.sqrt(var) / sg * ELO:.1f} Elo per side-game")
    for sub, (t, v, n) in by_sub.items():
        if n:
            print(f"     {sub:18s} switches {n / sg:.2f}; {t / sg * ELO:+.1f} +- {math.sqrt(v) / sg * ELO:.1f} Elo")
    w_all = r_weights(Xr, s_r.sum(1), n_r.sum(1))
    print(f"   R fitted on all folds: per-faction sums of w (minimum-norm level) max |sum| "
          f"{max(abs(sum(w_all[i] for i, u in enumerate(names) if u.startswith(fac + '|'))) for fac in {u.split('|')[0] for u in names}):.2e}")

    print("\n3. The processor's fit with x = the per-candidate unit block on rr + mixed pairs, ridge 5,000")
    X_all = np.vstack([Xr, Xm])
    s_all = np.vstack([s_r, s_m])
    ss_all = np.vstack([ss_r, ss_m])
    n_all = np.vstack([n_r, n_m])
    w_proc = r_weights(X_all, s_all.sum(1), n_all.sum(1))
    d = w_proc - w_all
    print(f"   max |w_proc - w_R| {np.abs(d).max():.4f}; rms {np.sqrt(np.mean(d ** 2)):.4f}; "
          f"rms of w_R {np.sqrt(np.mean(w_all ** 2)):.4f}; correlation {np.corrcoef(w_proc, w_all)[0, 1]:.3f}")
    nr = len(cells)
    t_rr = v_rr = t_mx = v_mx = sw_rr = sw_mx = 0.0
    for f in range(FOLDS):
        so, no = s_all.sum(1) - s_all[:, f], n_all.sum(1) - n_all[:, f]
        w = r_weights(X_all, so, no)
        ch = (X_all @ w) > 0
        cr, cm = ch[:nr], ch[nr:]
        t_rr += s_all[:nr][cr, f].sum()
        v_rr += ss_all[:nr][cr, f].sum()
        sw_rr += n_all[:nr][cr, f].sum()
        t_mx += s_all[nr:][cm, f].sum()
        v_mx += ss_all[nr:][cm, f].sum()
        sw_mx += n_all[nr:][cm, f].sum()
    print(f"   cross-fitted: rr cells {t_rr / sg * ELO:+.1f} +- {math.sqrt(v_rr) / sg * ELO:.1f} Elo "
          f"({sw_rr / sg:.2f} switches); mixed {t_mx / sg * ELO:+.1f} +- {math.sqrt(v_mx) / sg * ELO:.1f} Elo "
          f"({sw_mx / sg:.2f} switches); total {(t_rr + t_mx) / sg * ELO:+.1f} +- "
          f"{math.sqrt(v_rr + v_mx) / sg * ELO:.1f} Elo")


def section4():
    """R's cross-fitted F at ridges beyond round10_simplify_checks.out section 1's grid, up to the limit
    lambda -> infinity, where the switch rule is sign(X X' s): the worst case of one ridge shared with a
    token block that the progressive loss shrinks hard."""
    cells, per_match, side_games = r9.load()
    X, _names = r9.design(cells)
    s_all = sum(m["s_T"] for m in per_match)
    ss_all = sum(m["ss_T"] for m in per_match)
    n_all = sum(m["n"] for m in per_match)
    sg = sum(side_games)
    print("\n4. R's cross-fitted F beyond the measured ridge grid (60,000 games)")
    for lam in (50_000.0, 200_000.0, 1_000_000.0, 10_000_000.0, math.inf):
        tot = var = sw = 0.0
        for f in range(FOLDS):
            so, no = s_all.sum(1) - s_all[:, f], n_all.sum(1) - n_all[:, f]
            p = X @ (X.T @ so) if math.isinf(lam) else r9.fit_pred(X, so, no, lam)
            ch = (no > 0) & (p > 0)
            tot += s_all[ch, f].sum()
            var += ss_all[ch, f].sum()
            sw += n_all[ch, f].sum()
        print(f"   ridge {lam:>12,.0f}: F {tot / sg * ELO:+.1f} +- {math.sqrt(var) / sg * ELO:.1f} Elo; "
              f"switches {sw / sg:.2f}")


if __name__ == "__main__":
    main()
    section4()
