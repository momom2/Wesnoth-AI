"""Round 10, invent stance: is batch 1's recruit spread unit identity or the
argmax player's collapsed mixture?

Reads round10_invent_recruit_rows_<match>_p0.json.gz (round10_invent_recruit_extract.py):
every band decision of batch 1 whose candidates are recruits of two different
types, 514,281 of them, with the counts n0, n1 of each type the mover had
recruited before, and the label T.

Readings (game-clustered SEs; 20 folds by a hash of the game key; a rule
fitted on 19 folds is read on the 20th, so each reading is out of sample):
1. mean label by d = n0 - n1 (a0's count less a1's);
2. cross-fitted first-order welfare, per side-game in z and Elo, of
   (a) R: additive (faction, unit) values, ridge 5,000 labels (the design's R);
   (b) C: switch iff n1 < n0 (no parameter);
   (c) C2: switch iff n0 - n1 >= 2;
   (d) a fitted line in d (intercept and slope), switch iff it is > 0;
   (e) R plus d: the 43 unit columns (ridge 5,000) and d (ridge 1);
   paired differences against R;
3. the overlap of R's and C's switch sets, and each rule's welfare on the
   decisions where the other does not switch;
4. C's welfare by mover faction and by side-turn;
5. the unit values R fits, against own1's whole-game recruit share of each
   unit relative to the human corpus's (round10_invent_composition.py measures
   both; this script recomputes own1's shares from batch 1's recruit counts at
   the decisions it reads, so it is reported only as a rank correlation).
Standard library and numpy only.
"""
from __future__ import annotations

import gzip
import json
import math
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MATCHES = ("b00", "b01", "b02", "b03", "b04", "b05", "b06")
FOLDS = 20
ELO = 800 / math.log(10)
SIDE_GAMES = 2 * 60_000
RIDGE = 5_000.0


def load():
    cols = defaultdict(list)
    for m in MATCHES:
        d = json.loads(gzip.decompress((HERE / f"round10_invent_recruit_rows_{m}_p0.json.gz").read_bytes()))
        for k, v in d["rows"].items():
            cols[k].extend(v)
    games = np.array(cols["game"])
    _, gid = np.unique(games, return_inverse=True)
    fold = np.array([zlib.crc32(g.encode()) % FOLDS for g in games])
    fac = np.array(cols["fac"])
    u0 = np.array([f + "|" + u for f, u in zip(cols["fac"], cols["u0"])])
    u1 = np.array([f + "|" + u for f, u in zip(cols["fac"], cols["u1"])])
    n0, n1 = np.array(cols["n0"]), np.array(cols["n1"])
    return dict(gid=gid, fold=fold, fac=fac, u0=u0, u1=u1, n0=n0, n1=n1, st=np.array(cols["st"]),
                gap=np.array(cols["gap"]), T=np.array(cols["T"]), tot=np.array(cols["tot"]))


def unit_design(d):
    names = sorted(set(d["u0"]) | set(d["u1"]))
    at = {u: i for i, u in enumerate(names)}
    rows = np.arange(len(d["T"]))
    X = np.zeros((len(d["T"]), len(names)))
    X[rows, [at[u] for u in d["u1"]]] += 1.0
    X[rows, [at[u] for u in d["u0"]]] -= 1.0
    return X, names


def clustered(T, mask, gid):
    s = np.bincount(gid[mask], weights=T[mask], minlength=gid.max() + 1)
    return s.sum(), math.sqrt(np.sum(s * s))


def welfare(d, choose_fold):
    """choose_fold(train_mask) -> boolean switch array over all rows (read on the held-out fold)."""
    switch = np.zeros(len(d["T"]), dtype=bool)
    for f in range(FOLDS):
        held = d["fold"] == f
        sw = choose_fold(~held)
        switch[held] = sw[held]
    tot, se = clustered(d["T"], switch, d["gid"])
    return switch, tot / SIDE_GAMES, se / SIDE_GAMES


def ridge_fit(X, T, train, pen):
    Xt = X[train]
    A = Xt.T @ Xt + np.diag(pen)
    return np.linalg.solve(A, Xt.T @ T[train])


def report(name, d, sw, wz, wse):
    print(f"  {name:44s} {wz:+.5f} ({wse:.5f}) z = {wz * ELO:+.1f} ({wse * ELO:.1f}) Elo per side-game; "
          f"switches {sw.sum() / SIDE_GAMES:.2f} per side-game ({sw.mean():.3f} of these decisions)")


def paired(d, sa, sb, label):
    diff = d["T"] * (sa.astype(float) - sb.astype(float))
    s = np.bincount(d["gid"], weights=diff, minlength=d["gid"].max() + 1)
    print(f"  {label:44s} {s.sum() / SIDE_GAMES * ELO:+.1f} ({math.sqrt(np.sum(s * s)) / SIDE_GAMES * ELO:.1f}) Elo")


def main():
    d = load()
    T, n0, n1 = d["T"], d["n0"], d["n1"]
    dd = n0 - n1
    N = len(T)
    print(f"rows {N}, {N / SIDE_GAMES:.2f} per side-game; pooled mean {T.mean():+.4f} "
          f"({clustered(T, np.ones(N, bool), d['gid'])[1] / N:.4f})")

    print("\n1. mean label by d = n0 - n1 (positive: the side has more of the argmax's type)")
    for lo, hi, lab in ((-99, -2, "<= -2"), (-1, -1, "-1"), (0, 0, "0"), (1, 1, "+1"), (2, 2, "+2"),
                        (3, 4, "+3..+4"), (5, 99, ">= +5")):
        m = (dd >= lo) & (dd <= hi)
        tot, se = clustered(T, m, d["gid"])
        print(f"  d {lab:7s} n {m.sum():7d} ({m.sum() / SIDE_GAMES:.2f}/side-game)  mean {tot / m.sum():+.4f} ({se / m.sum():.4f})  "
              f"z {tot / se:+.2f}")
    for st_lab, sm in (("side-turn 1", d["st"] == 1), ("side-turn >= 2", d["st"] >= 2)):
        print(f"  [{st_lab}]")
        for lo, hi, lab in ((-99, -1, "<= -1"), (0, 0, "0"), (1, 1, "+1"), (2, 99, ">= +2")):
            m = sm & (dd >= lo) & (dd <= hi)
            tot, se = clustered(T, m, d["gid"])
            print(f"    d {lab:6s} n {m.sum():7d}  mean {tot / max(m.sum(), 1):+.4f} ({se / max(m.sum(), 1):.4f})  z {tot / se:+.2f}")

    X, names = unit_design(d)
    k = X.shape[1]
    print(f"\n2. cross-fitted welfare ({FOLDS} folds by game; {k} (faction, unit) columns)")
    rules = {}

    def rule_R(train):
        beta = ridge_fit(X, T, train, np.full(k, RIDGE))
        return X @ beta > 0
    rules["(a) R: unit values, ridge 5,000"] = rule_R
    rules["(b) C: switch iff n1 < n0"] = lambda train: dd > 0
    rules["(c) C2: switch iff n0 - n1 >= 2"] = lambda train: dd >= 2

    D1 = np.column_stack([np.ones(N), dd.astype(float)])

    def rule_line(train):
        beta = ridge_fit(D1, T, train, np.array([1.0, 1.0]))
        return D1 @ beta > 0
    rules["(d) line in d, fitted"] = rule_line

    XD = np.column_stack([X, dd.astype(float)])

    def rule_Rd(train):
        beta = ridge_fit(XD, T, train, np.r_[np.full(k, RIDGE), 1.0])
        return XD @ beta > 0
    rules["(e) R plus d"] = rule_Rd
    sw = {}
    for name, rule in rules.items():
        s, wz, wse = welfare(d, rule)
        sw[name] = s
        report(name, d, s, wz, wse)
    print("  paired differences (same labels):")
    names_r = list(rules)
    paired(d, sw[names_r[1]], sw[names_r[0]], "C - R")
    paired(d, sw[names_r[4]], sw[names_r[0]], "(R plus d) - R")
    paired(d, sw[names_r[4]], sw[names_r[1]], "(R plus d) - C")

    sR, sC = sw[names_r[0]], sw[names_r[1]]
    print("\n3. overlap of R's and C's switch sets")
    for lab, m in (("both switch", sR & sC), ("R only", sR & ~sC), ("C only", ~sR & sC), ("neither", ~sR & ~sC)):
        tot, se = clustered(T, m, d["gid"])
        print(f"  {lab:12s} {m.sum() / SIDE_GAMES:.2f} per side-game  welfare {tot / SIDE_GAMES * ELO:+.1f} ({se / SIDE_GAMES * ELO:.1f}) Elo"
              f"  mean label {tot / max(m.sum(), 1):+.4f}")

    print("\n4. C by mover faction and by side-turn (welfare in Elo per side-game of all factions)")
    for fac in sorted(set(d["fac"])):
        m = sC & (d["fac"] == fac)
        tot, se = clustered(T, m, d["gid"])
        print(f"  {fac:18s} {tot / SIDE_GAMES * ELO:+.1f} ({se / SIDE_GAMES * ELO:.1f}); switches {m.sum() / SIDE_GAMES:.2f}")
    for lab, m0 in (("side-turn 1", d["st"] == 1), ("side-turn 2-4", (d["st"] >= 2) & (d["st"] <= 4)),
                    ("side-turn 5+", d["st"] >= 5)):
        m = sC & m0
        tot, se = clustered(T, m, d["gid"])
        print(f"  {lab:18s} {tot / SIDE_GAMES * ELO:+.1f} ({se / SIDE_GAMES * ELO:.1f}); switches {m.sum() / SIDE_GAMES:.2f}")

    print("\n5. R's unit values (fitted on all rows) against how often own1 recruits the unit at these decisions")
    beta = ridge_fit(X, T, np.ones(N, bool), np.full(k, RIDGE))
    beta_d = ridge_fit(XD, T, np.ones(N, bool), np.r_[np.full(k, RIDGE), 1.0])
    owned = defaultdict(float)
    seen = defaultdict(int)
    for u, n in zip(np.r_[d["u0"], d["u1"]], np.r_[n0, n1]):
        owned[u] += n
        seen[u] += 1
    mean_owned = np.array([owned[u] / max(seen[u], 1) for u in names])
    rank = lambda v: np.argsort(np.argsort(v))
    rho = np.corrcoef(rank(beta), rank(mean_owned))[0, 1]
    rho_d = np.corrcoef(rank(beta_d[:k]), rank(mean_owned))[0, 1]
    print(f"  Spearman(R's value, mean count owned when the unit is a candidate) {rho:+.3f}; "
          f"after d enters the fit {rho_d:+.3f}; slope on d {beta_d[k]:+.4f} per unit of d")
    print(f"  sd of R's values {beta.std():.4f}; of the unit values with d in the fit {beta_d[:k].std():.4f}")

    print("\n6. the label of C's switches (d >= 1) by margin G: does the network's own preference carry the value?")
    for lo, hi in ((0.0, 0.1), (0.1, 0.2), (0.2, 0.35)):
        for lab, m0 in (("d 1-2", (dd >= 1) & (dd <= 2)), ("d >= 3", dd >= 3), ("d <= 0", dd <= 0)):
            m = m0 & (d["gap"] >= lo) & (d["gap"] < hi)
            tot, se = clustered(T, m, d["gid"])
            print(f"  G [{lo:.2f},{hi:.2f}) {lab:7s} n {m.sum():7d}  mean {tot / max(m.sum(), 1):+.4f} ({se / max(m.sum(), 1):.4f})")
    for lab, m0 in (("d >= 1", dd >= 1), ("d <= 0", dd <= 0)):
        m = m0
        g = d["gap"][m]
        Xg = np.column_stack([np.ones(m.sum()), g])
        b = np.linalg.lstsq(Xg, T[m], rcond=None)[0]
        resid = T[m] - Xg @ b
        # game-clustered sandwich SE of the slope
        XtX_inv = np.linalg.inv(Xg.T @ Xg)
        gid = d["gid"][m]
        meat = np.zeros((2, 2))
        sg = np.zeros((gid.max() + 1, 2))
        np.add.at(sg, gid, Xg * resid[:, None])
        meat = sg.T @ sg
        V = XtX_inv @ meat @ XtX_inv
        print(f"  {lab}: label = {b[0]:+.4f} + {b[1]:+.4f} G (slope SE {math.sqrt(V[1, 1]):.4f}), n {m.sum()}")

    print("\n7. the count slope within unit identity: label on the 43 unit columns (ridge 5,000) and d (ridge 1),")
    print("   game-clustered sandwich SE; and on d alone with an intercept")
    for lab, M, pen in (("units + d", XD, np.r_[np.full(k, RIDGE), 1.0]),
                        ("d alone", D1, np.array([1.0, 1.0]))):
        A_inv = np.linalg.inv(M.T @ M + np.diag(pen))
        b = A_inv @ (M.T @ T)
        resid = T - M @ b
        sg = np.zeros((d["gid"].max() + 1, M.shape[1]))
        np.add.at(sg, d["gid"], M * resid[:, None])
        V = A_inv @ (sg.T @ sg) @ A_inv
        j = M.shape[1] - 1
        print(f"  {lab:10s} slope on d {b[j]:+.5f} ({math.sqrt(V[j, j]):.5f}), z {b[j] / math.sqrt(V[j, j]):+.2f}")


if __name__ == "__main__":
    main()
