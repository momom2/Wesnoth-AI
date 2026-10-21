"""Round 9, provenance stance: the recruit type-pair cells, the one partition of
batch 1's band decisions where round9_provenance_cells.py finds a spread of
cell means. Reads the per-cell fold sums written by round9_provenance_cells.py
extract (partition rr_units: (mover's faction, a0's unit > a1's unit) for
recruit>recruit decisions whose two candidates are different units).
Standard library and numpy only.

Readings, restricted to those decisions (the "rest" cell left out):
1. Excess dispersion: sum over cells of (mean/SE)^2 against the cell count
   (its SD sqrt(2K) under no spread), cells with >= 200 labels.
2. V_rr: the variance of the true cell means among these decisions, by the
   fold U-statistic, jackknife SE; and the same on even and on odd folds.
3. Cross-fitted welfare of three rules chosen on 19 folds, read on the 20th:
   switch where (a) the cell mean > 0, (b) the cell mean > 0 with >= 1,000
   out-of-fold labels, (c) an additive per-(faction, unit) value v, switch iff
   v(a1) - v(a0) > 0, fitted by ridge on cell means (lambda in labels).
   Per side-game in z and Elo; SE from the labels' second moment.
4. The ten largest cells.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MATCHES = ("b00", "b01", "b02", "b03", "b04", "b05", "b06")
FOLDS = 20
ELO = 800 / math.log(10)


def load():
    data = [json.loads((HERE / f"round9_provenance_cells_{m}.json").read_text()) for m in MATCHES]
    sg = sum(d["side_games"] for d in data)
    cells = sorted({c for d in data for c in d["partitions"]["rr_units"]["cells"]} - {"rest"})
    at = {c: i for i, c in enumerate(cells)}
    acc = {k: np.zeros((len(cells), FOLDS)) for k in ("n", "s_T", "ss_T", "s_T0", "ss_T0")}
    for d in data:
        p = d["partitions"]["rr_units"]
        for j, c in enumerate(p["cells"]):
            if c == "rest":
                continue
            for k in acc:
                acc[k][at[c]] += np.asarray(p[k][j])
    return cells, acc, sg


def u_square(s, n):
    num = s.sum(1) ** 2 - (s * s).sum(1)
    den = n.sum(1) ** 2 - (n * n).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, 0.0)


def v_within(s, n):
    pi = n.sum(1) / n.sum()
    tot = s.sum(0, keepdims=True)
    return float(np.sum(pi * u_square(s, n)) - u_square(tot, n.sum(0, keepdims=True))[0])


def jack(fn, s, n, folds):
    full = fn(s[:, folds], n[:, folds])
    reps = np.asarray([fn(s[:, [f for f in folds if f != g]], n[:, [f for f in folds if f != g]]) for g in folds])
    k = len(folds)
    return full, math.sqrt((k - 1) / k * np.sum((reps - reps.mean()) ** 2))


def units_of(cell):
    fac, pair = cell.split("|")
    u0, u1 = pair.split(">")
    return f"{fac}|{u0}", f"{fac}|{u1}"


def additive_rule(cells, s, n, lam):
    """Ridge fit of the cell mean on e(a1) - e(a0) over (faction, unit) values,
    weighted by label counts; returns the predicted mean per cell."""
    names = sorted({u for c in cells for u in units_of(c)})
    at = {u: i for i, u in enumerate(names)}
    X = np.zeros((len(cells), len(names)))
    for i, c in enumerate(cells):
        u0, u1 = units_of(c)
        X[i, at[u1]] += 1.0
        X[i, at[u0]] -= 1.0
    w = n
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(n > 0, s / n, 0.0)
    A = X.T @ (X * w[:, None]) + lam * np.eye(len(names))
    beta = np.linalg.solve(A, X.T @ (w * m))
    return X @ beta


def welfare(cells, a, sg, rule, label="T"):
    s, ss, n = a[f"s_{label}"], a[f"ss_{label}"], a["n"]
    total = var = switched = 0.0
    for f in range(FOLDS):
        so, no = s.sum(1) - s[:, f], n.sum(1) - n[:, f]
        choose = rule(so, no)
        total += s[choose, f].sum()
        var += ss[choose, f].sum()
        switched += n[choose, f].sum()
    return total / sg, math.sqrt(var) / sg, switched / n.sum()


def main():
    cells, a, sg = load()
    n_tot = a["n"].sum()
    print(f"recruit>recruit, different units: {len(cells)} cells (faction, a0 > a1), {int(n_tot)} labels, "
          f"{n_tot / sg:.2f} per side-game")
    for label in ("T", "T0"):
        s, ss, n = a[f"s_{label}"], a[f"ss_{label}"], a["n"]
        ns, sums, sss = n.sum(1), s.sum(1), ss.sum(1)
        big = ns >= 200
        zc = sums[big] / np.sqrt(sss[big])
        K = int(big.sum())
        print(f"\n[{label}] 1. excess dispersion over {K} cells with >= 200 labels: sum z^2 {np.sum(zc ** 2):.1f} "
              f"against {K} (SD {math.sqrt(2 * K):.1f}): excess {(np.sum(zc ** 2) - K) / math.sqrt(2 * K):+.2f} SD")
        mean = sums.sum() / ns.sum()
        mse = math.sqrt(sss.sum()) / ns.sum()
        print(f"[{label}] pooled mean of these decisions {mean:+.4f} ({mse:.4f})")
        v, se = jack(v_within, s, n, list(range(FOLDS)))
        ve, see = jack(v_within, s, n, list(range(0, FOLDS, 2)))
        vo, seo = jack(v_within, s, n, list(range(1, FOLDS, 2)))
        print(f"[{label}] 2. V_rr {v:.3e} ({se:.2e}), z {v / se:+.2f}: sd of cell means {math.sqrt(max(v, 0)):.4f}, "
              f">= {math.sqrt(max(v - 2 * se, 0)):.4f} at 2 SE; even folds {ve:.3e} ({see:.2e}), odd folds {vo:.3e} ({seo:.2e})")
        rules = {
            "(a) cell mean > 0": lambda so, no: (no > 0) & (so > 0),
            "(b) cell mean > 0, >= 1,000 labels": lambda so, no: (no >= 1000) & (so > 0),
        }
        for lam in (2_000.0, 20_000.0):
            def rule(so, no, lam=lam):
                pred = additive_rule(cells, so, no, lam)
                return (no > 0) & (pred > 0)
            rules[f"(c) additive unit values, ridge {lam:.0f} labels"] = rule
        for name, rule in rules.items():
            wz, wse, share = welfare(cells, a, sg, rule, label)
            print(f"[{label}] 3. {name:40s} {wz:+.5f} ({wse:.5f}) z per side-game = {wz * ELO:+.1f} ({wse * ELO:.1f}) Elo; "
                  f"switches {share:.3f} of these decisions")
    print("\n5. honest halves: the rule chosen on the even folds' games, read on the odd folds' games, and the "
          "reverse (each reading independent of its rule; per side-game of the half read)")
    even, odd = list(range(0, FOLDS, 2)), list(range(1, FOLDS, 2))
    half_sg = sg / 2
    for label in ("T", "T0"):
        for name, (pick, read) in (("even -> odd", (even, odd)), ("odd -> even", (odd, even))):
            sp, npk = a[f"s_{label}"][:, pick].sum(1), a["n"][:, pick].sum(1)
            sr, ssr = a[f"s_{label}"][:, read].sum(1), a[f"ss_{label}"][:, read].sum(1)
            for rname, choose in (("cell mean > 0", (npk > 0) & (sp > 0)),
                                  ("additive, ridge 2000", additive_rule(cells, sp, npk, 2000.0) > 0)):
                wz, wse = sr[choose].sum() / half_sg, math.sqrt(ssr[choose].sum()) / half_sg
                print(f"  [{label}] {name}, {rname:22s}: {wz:+.5f} ({wse:.5f}) z = {wz * ELO:+.1f} ({wse * ELO:.1f}) Elo")
    print("\n6. cross-fitted welfare of rule (a) by mover faction (T), Elo per side-game of all factions")
    facs = sorted({c.split("|")[0] for c in cells})
    s_all, ss_all, n_all = a["s_T"], a["ss_T"], a["n"]
    for fac in facs:
        rows = np.array([c.split("|")[0] == fac for c in cells])
        tot = var = 0.0
        for f in range(FOLDS):
            so, no = s_all.sum(1) - s_all[:, f], n_all.sum(1) - n_all[:, f]
            choose = rows & (no > 0) & (so > 0)
            tot += s_all[choose, f].sum()
            var += ss_all[choose, f].sum()
        print(f"  {fac:18s} {tot / sg * ELO:+.1f} ({math.sqrt(var) / sg * ELO:.1f}) Elo; {n_all[rows].sum() / sg:.2f} "
              f"decisions per side-game")

    print("\n7. the additive head's ridge, cross-fitted (rule fitted on T, read with T and with T0)")
    names = sorted({u for c in cells for u in units_of(c)})
    print(f"  {len(names)} (faction, unit) values")
    for lam in (200.0, 2_000.0, 20_000.0, 200_000.0):
        out = []
        for read in ("T", "T0"):
            tot = var = 0.0
            for f in range(FOLDS):
                so, no = a["s_T"].sum(1) - a["s_T"][:, f], a["n"].sum(1) - a["n"][:, f]
                choose = (no > 0) & (additive_rule(cells, so, no, lam) > 0)
                tot += a[f"s_{read}"][choose, f].sum()
                var += a[f"ss_{read}"][choose, f].sum()
            out.append(f"{read} {tot / sg * ELO:+.1f} ({math.sqrt(var) / sg * ELO:.1f})")
        print(f"  ridge {lam:>9.0f} labels: " + "; ".join(out) + " Elo per side-game")

    s, ss, n = a["s_T"].sum(1), a["ss_T"].sum(1), a["n"].sum(1)
    order = np.argsort(-n)
    print("\n4. the largest cells (T): n, mean (SE), z")
    for i in order[:12]:
        print(f"  {cells[i]:45s} {int(n[i]):7d} {s[i] / n[i]:+.4f} ({math.sqrt(ss[i]) / n[i]:.4f}) z {s[i] / math.sqrt(ss[i]):+.2f}")
    zs = s / np.sqrt(np.maximum(ss, 1e-12))
    print("  largest |z| among cells with >= 1,000 labels:")
    for i in [j for j in np.argsort(-np.abs(zs)) if n[j] >= 1000][:8]:
        print(f"  {cells[i]:45s} {int(n[i]):7d} {s[i] / n[i]:+.4f} ({math.sqrt(ss[i]) / n[i]:.4f}) z {zs[i]:+.2f}")


if __name__ == "__main__":
    main()
