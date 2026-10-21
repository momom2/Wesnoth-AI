"""Round 10, critique of the provenance stance: checks of its own-interaction, state-shift and learning-curve readings.

Reads the stance's per-match rows (round10_provenance_r_link_b0*.json, reproduced for b00 by re-extraction) and
round 9's cell sums (round9_provenance_cells_*.json). numpy and the standard library only.

1. The first-coin correction's coefficient. The stance's exact identity (round10_provenance_r_pdis.py) is
   J(pi'_R) - J(pi'_0) = 0.75 E_R[sum Delta] + 0.25 E_0[sum Delta]. Weighting only the side's first R coin, a later
   label gets 0.75 * 4 * 1[D1 = 1] + 0.25 * (4/3) * 1[D1 = 0]; its mean minus the unweighted mean is
   0.5 (E1 - E0) at P(D1 = 1) = 1/4, not 0.75 (E1 - E0) as round10_provenance_r_link.py section 1 uses.
2. The state shift at rank-0 switches, exactly: W = 1[no prior deviation by either side] / 0.75^K (K prior band
   decisions, P 0.25 at each) reweights the coin world's states to uncoined prior play, which is L's state
   distribution for a side's first R decision (the opponent in L plays own1 without R). Continuation still coined.
3. R's learning curve, paired: for each held-out match, the rule fitted on all six others against rules fitted
   on k of them, read on the same held-out labels (the variance of a mean over subsets is bounded by the mean of
   their variances, so the SE is conservative).
4. The stance's "turn-1 part 3.9 SE above the null line", with L's own match noise included.
5. The second-order term of J(pi'_R) - J(pi'_0) around the coin policy (Tang, Valko & Munos 2020, TayPO-2 form):
   pairs of a side's R decisions, variance growing with the pair count rather than as 4^k.
Run: python round10_provenance_critique_checks.py (about 2 s; output in round10_provenance_critique_checks.out).
"""
from __future__ import annotations

import importlib.util
import itertools
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ELO = 800 / math.log(10)
spec = importlib.util.spec_from_file_location("rl", HERE / "round10_provenance_r_link.py")
rl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl)


def cluster_se(values, games):
    order = np.argsort(games, kind="stable")
    g, v = games[order], values[order]
    starts = np.nonzero(np.r_[True, g[1:] != g[:-1]])[0]
    return math.sqrt(float(np.sum(np.add.reduceat(v, starts) ** 2)))


def first_rank_coin(a):
    """Per row, the coin d at the side's rank-0 R decision."""
    keys = a[:, 0] * 4 + a[:, 1]
    order = np.lexsort((a[:, 9], keys))
    a = a[order]
    k = keys[order]
    starts = np.r_[True, k[1:] != k[:-1]]
    idx = np.maximum.accumulate(np.where(starts, np.arange(len(a)), 0))
    return a, a[idx, 3]


def section1(a, sg):
    a, d1 = first_rank_coin(a)
    T, rank, game = a[:, 2], a[:, 9], a[:, 0]
    later = rank > 0
    w = np.where(later, np.where(d1 == 1, 3.0, 1.0 / 3.0), 1.0)
    est = w * T
    diff = (est - T) / sg * ELO
    print("1. first-coin correction, weighted consistently with the stance's PDL identity")
    print(f"   exact first-coin-only estimate minus F: {diff.sum():+.2f} +- {cluster_se(est - T, game) / sg * ELO:.2f} Elo "
          f"(the stance's 0.75 x difference reads -3.5 +- 4.2; 0.5 x the same difference is "
          f"{-3.5 * 0.5 / 0.75:+.2f} +- {4.2 * 0.5 / 0.75:.2f})")


def section2(a, sg):
    T, game, rank, turn = a[:, 2], a[:, 0], a[:, 9], a[:, 8]
    K = a[:, 4] + a[:, 5]
    dev = a[:, 6] + a[:, 7]
    print("\n2. state shift at rank-0 R switches, exact importance weights to uncoined prior play")
    for lab, m in (("rank 0, turn 1", (rank == 0) & (turn == 1)), ("rank 0, turns 1-2", (rank == 0) & (turn <= 2)),
                   ("rank 0, all turns", rank == 0)):
        Km = K[m]
        W = np.where(dev[m] == 0, 0.75 ** (-Km), 0.0)
        est = W * T[m]
        ess = W.sum() ** 2 / np.sum(W ** 2) if np.any(W > 0) else 0.0
        print(f"   {lab}: {m.sum() / sg:.3f} switches per side-game, mean K {Km.mean():.1f} (max {Km.max():.0f}); "
              f"coin-world F part {T[m].sum() / sg * ELO:+.2f} +- {cluster_se(T[m], game[m]) / sg * ELO:.2f}; "
              f"at uncoined prior states {est.sum() / sg * ELO:+.2f} +- {cluster_se(est, game[m]) / sg * ELO:.2f}; "
              f"mean weight {W.mean():.3f}, ESS share {ess / m.sum():.3f}; shift {(est - T[m]).sum() / sg * ELO:+.2f} "
              f"+- {cluster_se(est - T[m], game[m]) / sg * ELO:.2f} (paired)")


def section3():
    cells, per = rl.cell_sums()
    X = rl.design(cells)
    names = rl.MATCHES
    ss = {}
    import json
    for m in names:
        d = json.loads((HERE / f"round9_provenance_cells_{m}.json").read_text())
        p = d["partitions"]["rr_units"]
        at = {c: i for i, c in enumerate(cells)}
        v = np.zeros(len(cells))
        for j, c in enumerate(p["cells"]):
            if c != "rest":
                v[at[c]] += sum(p["ss_T"][j])
        ss[m] = v
    sg = sum(per[m][3] for m in names)
    print("\n3. R's learning curve, paired on the same held-out labels: the 6-match rule minus k-match rules")
    for k in (1, 2, 3, 4, 5):
        tot, var = 0.0, 0.0
        for m in names:
            rest = [x for x in names if x != m]
            s6 = sum(per[x][0] for x in rest)
            n6 = sum(per[x][2] for x in rest)
            full = rl.fit_pred(X, s6, n6) > 0
            ds, vs = [], []
            for sub in itertools.combinations(rest, k):
                rk = rl.fit_pred(X, sum(per[x][0] for x in sub), sum(per[x][2] for x in sub)) > 0
                ds.append(per[m][0][full].sum() - per[m][0][rk].sum())
                vs.append(ss[m][full != rk].sum())
            tot += np.mean(ds)
            var += np.mean(vs)
        print(f"   k {k} (about {k * 8_571:,} training games) to 6 (about 51,400): {tot / sg * ELO:+.2f} "
              f"+- {math.sqrt(var) / sg * ELO:.2f} Elo (conservative SE)")


def section4():
    null_line = 2 * ELO / math.sqrt(37_800)
    lo, se_pred, se_match = 12.7 - 0.3, math.hypot(1.5, 1.7), ELO / math.sqrt(37_800)
    print("\n4. the turn-1 part against L's null line")
    print(f"   without L's noise (the stance): {(lo - null_line) / se_pred:.2f} SE; with it: "
          f"{(lo - null_line) / math.hypot(se_pred, se_match):.2f} SE")


def main():
    a, sg = rl.load_rows()
    section1(a, sg)
    section2(a, sg)
    section3()
    section4()


def section5(a, sg):
    """Second-order (TayPO-2-style) term of J(pi'_R) - J(pi'_0) around the coin policy: each later R label times
    the sum over the side's earlier R coins of 0.75 (rho_R - 1) + 0.25 (rho_0 - 1), i.e. +2 if played, -2/3 if not.
    Its variance grows with the number of earlier coins, not as 4^k."""
    spec2 = importlib.util.spec_from_file_location("pd", HERE / "round10_provenance_r_pdis.py")
    pd = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(pd)
    a, pb = pd.grouped(a)
    T, rank, game = a[:, 2], a[:, 9], a[:, 0]
    c = 2.0 * pb - (2.0 / 3.0) * (rank - pb)
    v = T * c
    print("\n5. second-order term over all ranks (pairs of a side's R decisions), Tang, Valko & Munos 2020 form")
    print(f"   second-order correction {v.sum() / sg * ELO:+.2f} +- {cluster_se(v, game) / sg * ELO:.2f} Elo "
          f"(exact all-order J - F reads +31.0 +- 40.4; ranks 0-1 exact -0.3 +- 1.7)")
    tot = T * (1.0 + c)
    print(f"   F_LOMO plus the second-order term: {tot.sum() / sg * ELO:+.2f} +- {cluster_se(tot, game) / sg * ELO:.2f} Elo "
          f"(L's null line {2 * ELO / math.sqrt(37_800):.2f})")
    for lab, m in (("rank 1", rank == 1), ("rank 2", rank == 2), ("ranks 1-2", (rank >= 1) & (rank <= 2)),
                   ("ranks 3+", rank >= 3)):
        print(f"   {lab}: {v[m].sum() / sg * ELO:+.2f} +- {cluster_se(v[m], game[m]) / sg * ELO:.2f}")


if __name__ == "__main__":
    main()
    a_, sg_ = rl.load_rows()
    section5(a_, sg_)
