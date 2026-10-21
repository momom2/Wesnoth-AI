"""Round 10 synthesis: own interactions of each recruit rule on record, and the capped rule L can seal.

The provenance critique (round10_provenance_critique_checks.out section 5) read R's second-order term
(Tang, Valko & Munos 2020, TayPO-2 form around the coin policy) at -12.3 +- 4.5 Elo, so R's first-order F
is not a sealable prediction of R in play. The invent stance found a count gradient inside unit identity
(round10_invent_recruit_rules.out section 7): a rule that switches toward the type the side has fewer of.
If R's negative interactions are its later switches recruiting the same unit again, a rule that carries
the count d = n0 - n1 should show a smaller second-order term. This script reads, for each rule, on the
same rows and the same leave-one-match-out (LOMO) folds:

  F       first-order welfare, cross-fitted (each match's rule fitted on the other six);
  cap2    the exact coin-world value of the rule capped at each side's first two switches a game:
          the performance-difference terms of ranks 0-1, rank 1 weighted by 3 if the rank-0 coin played a1
          and 1/3 if not (round10_provenance_r_pdis.py's identity; the critique showed these terms are the
          capped rule's value, since ranks 0-1 states do not depend on later ranks);
  SO      the second-order term over all ranks: each later switch label times the sum, over the side's
          earlier switches, of +2 (earlier coin played a1) or -2/3 (not);
  by rank, by turn.

Rules (rows: every different-unit recruit near-tie of batch 1, round10_invent_recruit_rows_*.json.gz):
  R   43 (faction, unit) values, ridge 5,000 labels;
  Rd  R plus the count d = n0 - n1 (ridge 1 on d), the invent stance's rule (e);
  B   R plus (faction, unit, enemy faction) deviations, ridge 5,000 on both blocks; the enemy faction
      is read from the other side's rows of the same game (rows without it use R's block only);
  Bd  B plus d;
  C   switch iff d > 0 (no parameter; reported as a diagnostic only: it is a hand-written rule).
Also: R's 20-fold reading under the invent stance's crc32 folds against LOMO, to show the fold spread.

numpy and the standard library only; about 30 s.
Run: python round10_synthesis_checks.py > round10_synthesis_checks.out
"""
from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MATCHES = ("b00", "b01", "b02", "b03", "b04", "b05", "b06")
ELO = 800 / math.log(10)
SG = 2 * 60_000
RIDGE = 5_000.0


def load():
    cols = defaultdict(list)
    for m in MATCHES:
        d = json.loads(gzip.decompress((HERE / f"round10_invent_recruit_rows_{m}_p0.json.gz").read_bytes()))
        for k, v in d["rows"].items():
            cols[k].extend(v)
    games = np.array(cols["game"])
    _, gid = np.unique(games, return_inverse=True)
    match = np.array([g.split("/")[0] for g in games])
    side = np.array(cols["side"], dtype=int)
    fac = np.array(cols["fac"])
    # enemy faction: the other side's faction in the same game, when that side has a row
    side_fac = {}
    for g, s, f in zip(gid, side, fac):
        side_fac[(int(g), int(s))] = f
    enemy = np.array([side_fac.get((int(g), 3 - int(s)), "") for g, s in zip(gid, side)])
    return dict(gid=gid, match=match, side=side, cmd=np.array(cols["cmd"], dtype=int),
                D=np.array(cols["d"], dtype=int), fac=fac, enemy=enemy,
                u0=np.array([f + "|" + u for f, u in zip(fac, cols["u0"])]),
                u1=np.array([f + "|" + u for f, u in zip(fac, cols["u1"])]),
                dd=np.array(cols["n0"], dtype=float) - np.array(cols["n1"], dtype=float),
                st=np.array(cols["st"], dtype=int), T=np.array(cols["T"], dtype=float))


def sparse_design(d, with_b, with_d):
    """Per row, column indices and values (at most 5 nonzeros), and the penalty per column."""
    units = sorted(set(d["u0"]) | set(d["u1"]))
    at_u = {u: i for i, u in enumerate(units)}
    idx, val, pen = [], [], [RIDGE] * len(units)
    idx.append(np.array([at_u[u] for u in d["u1"]]))
    val.append(np.ones(len(d["T"])))
    idx.append(np.array([at_u[u] for u in d["u0"]]))
    val.append(-np.ones(len(d["T"])))
    k = len(units)
    if with_b:
        known = d["enemy"] != ""
        keys = sorted({f"{u}@{e}" for u, e in zip(np.r_[d["u0"][known], d["u1"][known]],
                                                  np.r_[d["enemy"][known], d["enemy"][known]])})
        at_b = {key: k + i for i, key in enumerate(keys)}
        dummy = k + len(keys)  # a zero column for rows with no enemy faction
        idx.append(np.array([at_b[f"{u}@{e}"] if e else dummy for u, e in zip(d["u1"], d["enemy"])]))
        val.append(known.astype(float))
        idx.append(np.array([at_b[f"{u}@{e}"] if e else dummy for u, e in zip(d["u0"], d["enemy"])]))
        val.append(-known.astype(float))
        pen += [RIDGE] * len(keys) + [1.0]
        k = dummy + 1
    if with_d:
        idx.append(np.full(len(d["T"]), k))
        val.append(d["dd"])
        pen.append(1.0)
        k += 1
    return np.column_stack(idx), np.column_stack(val), np.array(pen)


def stats(idx, val, T, mask, k):
    """X'X and X'T over the masked rows, from the sparse rows."""
    I, V, t = idx[mask], val[mask], T[mask]  # noqa: E741 (an analysis script kept as run)
    nnz = I.shape[1]  # noqa: F841 (an analysis script kept as run)
    flat = (I[:, :, None] * k + I[:, None, :]).reshape(-1)
    w = (V[:, :, None] * V[:, None, :]).reshape(-1)
    XtX = np.bincount(flat, weights=w, minlength=k * k).reshape(k, k)
    XtT = np.bincount(I.reshape(-1), weights=(V * t[:, None]).reshape(-1), minlength=k)
    return XtX, XtT


def lomo_switch(d, with_b, with_d):
    idx, val, pen = sparse_design(d, with_b, with_d)
    k = len(pen)
    per = {m: stats(idx, val, d["T"], d["match"] == m, k) for m in MATCHES}
    tot_A = sum(p[0] for p in per.values())
    tot_b = sum(p[1] for p in per.values())
    f = np.zeros(len(d["T"]))
    for m in MATCHES:
        A = tot_A - per[m][0] + np.diag(pen)
        w = np.linalg.solve(A, tot_b - per[m][1])
        held = d["match"] == m
        f[held] = (val[held] * w[idx[held]]).sum(1)
    return f > 0


def progressive(d, with_b, with_d, lam):
    """Progressive loss over the matches in game order: student j-1 (fitted on matches before j) read on
    match j, sum of -2 w'b_j + w'G_j w; the empty student (w = 0) reads 0. Lower is better."""
    idx, val, pen = sparse_design(d, with_b, with_d)
    pen = np.where(pen == RIDGE, lam, pen)
    k = len(pen)
    per = [stats(idx, val, d["T"], d["match"] == m, k) for m in MATCHES]
    loss, A, b = 0.0, np.zeros((k, k)), np.zeros(k)
    for j, (G, bj) in enumerate(per):
        if j:
            w = np.linalg.solve(A + np.diag(pen), b)
            loss += -2 * w @ bj + w @ G @ w
        A, b = A + G, b + bj
    return loss


def cluster_se(v, gid):
    s = np.bincount(gid, weights=v, minlength=gid.max() + 1)
    return math.sqrt(float(np.sum(s * s)))


def elo(v, gid):
    return v.sum() / SG * ELO, cluster_se(v, gid) / SG * ELO


def ranked(d, sw):
    """For the switch rows of one rule: rank among the side-game's switches, earlier switches played,
    and the rank-0 coin of the side-game."""
    rows = np.nonzero(sw)[0]
    key = d["gid"][rows] * 4 + d["side"][rows]
    order = np.lexsort((d["cmd"][rows], key))
    rows, key = rows[order], key[order]
    start = np.r_[True, key[1:] != key[:-1]]
    pos = np.arange(len(rows))
    first = np.maximum.accumulate(np.where(start, pos, 0))
    rank = pos - first
    Dv = d["D"][rows]
    cum = np.cumsum(Dv)
    pb = cum - Dv - (cum - Dv)[first]
    return rows, rank, pb, Dv[first]


def report(name, d, sw):
    T, gid = d["T"], d["gid"]
    rows, rank, pb, d0 = ranked(d, sw)
    t, g = T[rows], gid[rows]
    F, Fse = elo(t, g)
    cap = rank <= 1
    w1 = np.where(rank == 1, np.where(d0 == 1, 3.0, 1.0 / 3.0), 1.0)
    capJ, capJse = elo((w1 * t)[cap], g[cap])
    capF, _ = elo(t[cap], g[cap])
    capD, capDse = elo(((w1 - 1.0) * t)[cap], g[cap])
    c = 2.0 * pb - (2.0 / 3.0) * (rank - pb)
    so, sose = elo(t * c, g)
    tot, totse = elo(t * (1.0 + c), g)
    print(f"{name}: {sw.sum() / SG:.2f} switches per side-game; F {F:+.1f} +- {Fse:.1f} Elo")
    print(f"   capped at 2 a side-game: {cap.sum() / SG:.2f} switches; exact coin-world value {capJ:+.1f} +- {capJse:.1f}"
          f" (first order {capF:+.1f}; interaction {capD:+.1f} +- {capDse:.1f})")
    print(f"   second-order term, all ranks: {so:+.1f} +- {sose:.1f}; F plus it {tot:+.1f} +- {totse:.1f}")
    for lab, m in (("rank 1", rank == 1), ("rank 2", rank == 2), ("ranks 3+", rank >= 3)):
        v, vse = elo((t * c)[m], g[m])
        print(f"     {lab:8s} {m.sum() / SG:.2f} switches; first order {elo(t[m], g[m])[0]:+.1f}; second-order {v:+.1f} +- {vse:.1f}")
    t1 = d["st"][rows] == 1
    print(f"   side-turn 1: {t1.sum() / SG:.2f} switches, F {elo(t[t1], g[t1])[0]:+.1f}; capped part "
          f"{elo((w1 * t)[cap & t1], g[cap & t1])[0]:+.1f}")
    return rows, rank, w1, cap


def paired(d, sa, sb):
    v = d["T"] * (sa.astype(float) - sb.astype(float))
    return elo(v, d["gid"])


def main():
    d = load()
    n = len(d["T"])
    known = d["enemy"] != ""
    print(f"rows {n} ({n / SG:.2f} per side-game); enemy faction known on {known.mean():.3f} of rows")
    print("all readings leave one match out (each match's rule fitted on the other six), game-clustered SEs, "
          "Elo per side-game\n")
    print("progressive loss (matches in game order; lower is better), by rule and ridge on the unit and matchup blocks")
    best = None
    for name, wb, wd in (("R", False, False), ("Rd", False, True), ("B", True, False), ("Bd", True, True)):
        row = []
        for lam in (2_000.0, 5_000.0, 10_000.0, 20_000.0):
            v = progressive(d, wb, wd, lam)
            row.append(f"{lam:,.0f}: {v:+.1f}")
            if best is None or v < best[0]:
                best = (v, name, lam)
        print(f"   {name:3s} " + "; ".join(row))
    print(f"   lowest: {best[1]} at ridge {best[2]:,.0f} ({best[0]:+.1f})\n")
    rules = {"R": lomo_switch(d, False, False), "Rd": lomo_switch(d, False, True),
             "B": lomo_switch(d, True, False), "Bd": lomo_switch(d, True, True), "C": d["dd"] > 0}
    capped = {}
    for name, sw in rules.items():
        capped[name] = report(name, d, sw)
        print()
    print("paired first-order differences, same labels (LOMO)")
    for a, b in (("Rd", "R"), ("B", "R"), ("Bd", "R"), ("Bd", "B"), ("Bd", "Rd")):
        v, se = paired(d, rules[a], rules[b])
        print(f"   {a} - {b}: {v:+.1f} +- {se:.1f}")
    print("\npaired capped (exact, ranks 0-1) differences against capped R")
    T, gid = d["T"], d["gid"]
    base = np.zeros(n)
    rows, rank, w1, cap = capped["R"]
    base[rows[cap]] = (w1 * T[rows])[cap]
    for name in ("Rd", "B", "Bd"):
        rows, rank, w1, cap = capped[name]
        x = np.zeros(n)
        x[rows[cap]] = (w1 * T[rows])[cap]
        v, se = elo(x - base, gid)
        print(f"   cap2 {name} - cap2 R: {v:+.1f} +- {se:.1f}")
    print("\ncapped R by mover faction (exact, ranks 0-1), for L's per-faction report")
    rows, rank, w1, cap = capped["R"]
    for fac in sorted(set(d["fac"])):
        m = cap & (d["fac"][rows] == fac)
        v, se = elo((w1 * T[rows])[m], gid[rows][m])
        print(f"   {fac:18s} {v:+.1f} +- {se:.1f}; switches {m.sum() / SG:.2f}")


if __name__ == "__main__":
    main()
