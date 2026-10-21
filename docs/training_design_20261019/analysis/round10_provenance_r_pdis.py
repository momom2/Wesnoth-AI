"""Round 10, provenance stance: R's all-order value in the coin world, exactly, from batch 1's labels.

round10_provenance_r_link.out section 1 found that R's later switches are worth about nothing once an
earlier R decision of the same side was switched (-0.0048 against +0.0284 per switch, z -3.16). F, the
design's sealed prediction for L, sums first-order labels as if every switch were made alone. This script
replaces that sum by an exact estimate of what imposing R on one side is worth in the coin world.

Performance-difference lemma (Kakade & Langford 2002): J(pi') - J(pi) = E_pi'[sum_t A^pi(s_t, pi'(s_t))].
Take pi = the coin policy pi_c (P(a1) = 1/4 at every band decision), pi'_R = pi_c with R's switch decisions
of one side played a1 for sure, pi'_0 = pi_c with them played a0 for sure. At such a decision j,
A^pi_c(s, a1) = 0.75 Delta_j and A^pi_c(s, a0) = -0.25 Delta_j, Delta_j = Q^pi_c(a1) - Q^pi_c(a0); elsewhere
pi' = pi_c, so the advantage has mean 0. Hence

    J(pi'_R) - J(pi'_0) = 0.75 E_pi'_R[sum_j Delta_j] + 0.25 E_pi'_0[sum_j Delta_j].

Each expectation is read from coin games by per-decision importance sampling over the side's earlier R
decisions only (the coin at each is independent of the past, P logged):
    W_R(j) = prod_{i<j} 1[d_i = 1] / 0.25,   W_0(j) = prod_{i<j} 1[d_i = 0] / 0.75,
and the label T_j is unbiased for Delta_j given the history. Both are exact; no model of the interaction.
With no interaction, E[W T] equals the first-order sum and the estimate reduces to F.

What it is not: L plays uncoined, so the states (other coins) and the continuation (coined after j) still
differ from L's. This estimate isolates R's own interactions; L then reads only the coin-to-uncoined shift.

Rules: each match's R fitted on the other six (round10_provenance_r_link.py), labels as there. Also:
the SE of the difference between the design's 20-fold F (+21.3) and the leave-one-match-out F, both
out of sample, from round9_provenance_cells_*.json's per-fold cell sums. numpy and the standard library.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ELO = 800 / math.log(10)
FOLDS = 20

spec = importlib.util.spec_from_file_location("rl", HERE / "round10_provenance_r_link.py")
rl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl)


def grouped(a):
    """Sort R-switch rows by (game, side, rank); return the rows and n_played_before per row."""
    keys = a[:, 0] * 4 + a[:, 1]
    order = np.lexsort((a[:, 9], keys))
    a = a[order]
    k = keys[order]
    starts = np.r_[True, k[1:] != k[:-1]]
    d = a[:, 3]
    cum = np.cumsum(d)
    group_start_cum = np.maximum.accumulate(np.where(starts, cum - d, 0))
    played_before = cum - d - group_start_cum
    return a, played_before


def cluster_se(values, games):
    order = np.argsort(games, kind="stable")
    g, v = games[order], values[order]
    starts = np.nonzero(np.r_[True, g[1:] != g[:-1]])[0]
    sums = np.add.reduceat(v, starts)
    return math.sqrt(float(np.sum(sums ** 2)))


def section1(a, sg):
    a, pb = grouped(a)
    T, d, rank, game = a[:, 2], a[:, 3], a[:, 9].astype(int), a[:, 0]  # noqa: F841 (an analysis script kept as run)
    all_played = pb == rank
    none_played = pb == 0
    wR = np.where(all_played, 4.0 ** rank, 0.0)
    w0 = np.where(none_played, (4.0 / 3.0) ** rank, 0.0)
    est = 0.75 * wR * T + 0.25 * w0 * T
    J = est.sum() / sg * ELO
    J_se = cluster_se(est, game) / sg * ELO
    F = T.sum() / sg * ELO
    F_se = cluster_se(T, game) / sg * ELO
    diff = est - T
    D_se = cluster_se(diff, game) / sg * ELO
    print("1. R imposed on one side in the coin world, exact (performance-difference lemma + per-decision IS)")
    print(f"   first-order F_LOMO {F:+.1f} +- {F_se:.1f} Elo per side-game (clustered by game)")
    print(f"   all-order J_c(R) {J:+.1f} +- {J_se:.1f} Elo; J_c(R) - F {J - F:+.1f} +- {D_se:.1f} (z {(J - F) / D_se:+.2f})")
    print("   by rank (earlier R decisions of the same side): switches per side-game; first-order; all-order; "
          "share of rows with all / none earlier played")
    for r in range(0, 6):
        m = rank == r if r < 5 else rank >= r
        lab = f"{r}" if r < 5 else "5+"
        print(f"     rank {lab}: {m.sum() / sg:.3f}; F part {T[m].sum() / sg * ELO:+.1f} +- {cluster_se(T[m], game[m]) / sg * ELO:.1f}; "
              f"J part {est[m].sum() / sg * ELO:+.1f} +- {cluster_se(est[m], game[m]) / sg * ELO:.1f}; "
              f"all {all_played[m].mean():.3f}, none {none_played[m].mean():.3f}")
    for cap in (1, 2, 3):
        m = rank <= cap
        Jc = est[m].sum() / sg * ELO
        print(f"   ranks 0-{cap}: the PDL terms of each side's first {cap + 1} R decisions, exact for them (not the value of a capped rule): "
              f"J {Jc:+.1f} +- {cluster_se(est[m], game[m]) / sg * ELO:.1f}; F {T[m].sum() / sg * ELO:+.1f}; "
              f"J - F {(est[m] - T[m]).sum() / sg * ELO:+.1f} +- {cluster_se(est[m] - T[m], game[m]) / sg * ELO:.1f}; "
              f"{m.sum() / sg:.2f} of {len(T) / sg:.2f} switches per side-game")
    # per-rank-1 randomized contrast: label at rank 1 by the rank-0 coin
    m1 = rank == 1
    x1, x0 = T[m1 & (pb == 1)], T[m1 & (pb == 0)]
    dd = x1.mean() - x0.mean()
    se = math.sqrt(x1.var() / len(x1) + x0.var() / len(x0))
    print(f"   rank 1 only, by the coin at rank 0: played {x1.mean():+.4f} (n {len(x1)}), not played {x0.mean():+.4f} "
          f"(n {len(x0)}): difference {dd:+.4f} +- {se:.4f} (z {dd / se:+.2f})")
    return J, J_se, F, F_se


def section2():
    print("\n2. the design's 20-fold F against the leave-one-match-out F: both out of sample, same labels")
    data = {m: json.loads((HERE / f"round9_provenance_cells_{m}.json").read_text()) for m in rl.MATCHES}
    cells = sorted({c for d in data.values() for c in d["partitions"]["rr_units"]["cells"]} - {"rest"})
    at = {c: i for i, c in enumerate(cells)}
    s = np.zeros((len(rl.MATCHES), len(cells), FOLDS))
    ss = np.zeros_like(s)
    n = np.zeros_like(s)
    for mi, m in enumerate(rl.MATCHES):
        p = data[m]["partitions"]["rr_units"]
        for j, c in enumerate(p["cells"]):
            if c != "rest":
                s[mi, at[c]] += p["s_T"][j]
                ss[mi, at[c]] += p["ss_T"][j]
                n[mi, at[c]] += p["n"][j]
    sg = sum(data[m]["side_games"] for m in rl.MATCHES)
    X = rl.design(cells)
    fold_rule = [rl.fit_pred(X, s.sum((0, 2)) - s[:, :, f].sum(0), n.sum((0, 2)) - n[:, :, f].sum(0)) > 0
                 for f in range(FOLDS)]
    match_rule = [rl.fit_pred(X, s.sum((0, 2)) - s[mi].sum(1), n.sum((0, 2)) - n[mi].sum(1)) > 0
                  for mi in range(len(rl.MATCHES))]
    full_rule = rl.fit_pred(X, s.sum((0, 2)), n.sum((0, 2))) > 0
    F20 = FL = D = V = 0.0
    dis_n = 0.0
    for mi in range(len(rl.MATCHES)):
        for f in range(FOLDS):
            a, b = fold_rule[f], match_rule[mi]
            F20 += s[mi, a, f].sum()
            FL += s[mi, b, f].sum()
            dis = a != b
            D += (s[mi, a & ~b, f].sum() - s[mi, b & ~a, f].sum())
            V += ss[mi, dis, f].sum()
            dis_n += n[mi, dis, f].sum()
    print(f"   20-fold {F20 / sg * ELO:+.1f}, leave-one-match-out {FL / sg * ELO:+.1f}; difference {D / sg * ELO:+.1f} "
          f"+- {math.sqrt(V) / sg * ELO:.1f} Elo (labels where the two rules disagree: {dis_n / sg:.3f} per side-game)")
    agree = [int((fold_rule[f] == full_rule).sum()) for f in range(FOLDS)]
    agree_m = [int((match_rule[mi] == full_rule).sum()) for mi in range(len(rl.MATCHES))]
    print(f"   cells (of {len(cells)}) where a fold rule agrees with the all-data rule: {min(agree)}-{max(agree)}; "
          f"a match rule: {min(agree_m)}-{max(agree_m)}")
    sw_full = (n.sum((0, 2)) * full_rule).sum() / sg
    print(f"   the all-data rule (the one L plays) switches {sw_full:.2f} decisions per side-game in batch 1's states")


def main():
    a, sg = rl.load_rows()
    J, J_se, F, F_se = section1(a, sg)
    section2()
    match_se = ELO / math.sqrt(37_800)
    print("\n3. L's sealed prediction, restated")
    for name, val, se in (("first-order F (current design)", 21.3, 3.1), ("all-order J_c(R), this script", J, J_se)):
        miss = 2 * math.sqrt(se ** 2 + match_se ** 2)
        p_null = 0.5 * math.erfc((val - 2 * match_se) / math.sqrt(se ** 2 + match_se ** 2) / math.sqrt(2))
        print(f"   {name}: {val:+.1f} +- {se:.1f}; miss half-width {miss:.1f}; P(L reads no gain at 2 SE | the "
              f"prediction holds) {p_null:.3f}")


if __name__ == "__main__":
    main()
