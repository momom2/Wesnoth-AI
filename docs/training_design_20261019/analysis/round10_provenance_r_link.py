"""Round 10, provenance stance: what batch 1's records say about L's sealed prediction before L runs.

L plays the recruit rule R (43 additive unit values, ridge 5,000 labels) on one seat of own1 against
own1, uncoined. Its sealed prediction is F, R's cross-fitted first-order welfare in the coin world. Three
things separate F from what L reads, and two of them can be read on the coin records now:

  (a) R's own switches interact. Under R every R decision is switched; in the coin world an earlier R
      decision of the same side was switched with P 1/4. The coin at the side's first R decision is
      independent of its past, so E[Y | D1 = 1] - E[Y | D1 = 0], Y the sum of the side's later R-switch
      labels, is the exact effect of switching the first one on the later switches' summed advantage
      (count changes included). It gives a first-step correction of F, 0.75 x that difference.
  (b) The coin world perturbs the states R's decisions are reached in: about 19 deviations per side-game
      by each side. Within strata of (turn, prior band decisions), the number of prior deviations is set
      by the coins; the slope of R's labels on it is a dose-response reading, and its linear extrapolation
      to zero deviations is a model-based reading of F at uncoined states (reported, not sealed).
  (c) The continuation after each R decision is coined in the labels and uncoined in L. The derivative
      of R's summed advantage in the continuation's deviation probability p is exact from the later
      coins' scores (D_k - p)/(p(1 - p)) multiplied into the label (section 2b); the step from p = 0.25
      to p = 0 along it is a Taylor step, reported, not sealed.
  Caution on (a): pooling the "earlier switch played" contrast over ranks mixes ranks, since
  P(any earlier played) grows with rank; the within-rank contrasts are the randomized ones.

Also: (d) R's learning curve over training games (does the rule L plays, fitted on all 60,000 games,
differ from the cross-fitted fold rules F describes?), from round9_provenance_cells_*.json's cell sums;
(e) the match wall's fixed cost per match, from walls.txt, and what it does to an overhead read from a
2,000-game dumped run against a 1,000-game undumped one.

Usage: python round10_provenance_r_link.py extract b01   (one match, under a minute each)
       python round10_provenance_r_link.py summary       (reads the per-match outputs)
Labels T = s omega (z - beta.S) with batch 1's beta (readout_b.json), b = 0, as round9_provenance_cells.py
forms them. R for match m is fitted on the other six matches' cell sums (leave one match out), so every
label read is out of sample for the rule that switches it. numpy and the standard library only.
"""
from __future__ import annotations

import gzip
import itertools
import json
import math
import sys
import tarfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REC = HERE.parents[2] / "training" / "metrics" / "near_tie_coins_20261020"
MATCHES = ("b00", "b01", "b02", "b03", "b04", "b05", "b06")
COLUMNS = ("hp", "material", "material_nonterminal", "terminal", "value_50", "value_100", "value_200")
ELO = 800 / math.log(10)
RIDGE = 5_000.0          # labels; the progressive loss's minimum, round9_synthesis_checks.out section 1
OUT = HERE / "round10_provenance_r_link_{}.json"


# ---------------------------------------------------------------- the rule R, from the round-9 cell sums
def units_of(cell):
    fac, pair = cell.split("|")
    u0, u1 = pair.split(">")
    return f"{fac}|{u0}", f"{fac}|{u1}"


def cell_sums():
    data = {m: json.loads((HERE / f"round9_provenance_cells_{m}.json").read_text()) for m in MATCHES}
    cells = sorted({c for d in data.values() for c in d["partitions"]["rr_units"]["cells"]} - {"rest"})
    at = {c: i for i, c in enumerate(cells)}
    per = {}
    for m, d in data.items():
        p = d["partitions"]["rr_units"]
        s, ss, n = np.zeros(len(cells)), np.zeros(len(cells)), np.zeros(len(cells))
        for j, c in enumerate(p["cells"]):
            if c != "rest":
                s[at[c]] += sum(p["s_T"][j])
                ss[at[c]] += sum(p["ss_T"][j])
                n[at[c]] += sum(p["n"][j])
        per[m] = (s, ss, n, d["side_games"], d["games"])
    return cells, per


def design(cells):
    names = sorted({u for c in cells for u in units_of(c)})
    at = {u: i for i, u in enumerate(names)}
    X = np.zeros((len(cells), len(names)))
    for i, c in enumerate(cells):
        u0, u1 = units_of(c)
        X[i, at[u1]] += 1.0
        X[i, at[u0]] -= 1.0
    return X


def fit_pred(X, s, n, lam=RIDGE):
    A = X.T @ (X * n[:, None]) + lam * np.eye(X.shape[1])
    return X @ np.linalg.solve(A, X.T @ s)


def rule_without(match):
    cells, per = cell_sums()
    X = design(cells)
    s = sum(per[m][0] for m in MATCHES if m != match)
    n = sum(per[m][2] for m in MATCHES if m != match)
    pred = fit_pred(X, s, n)
    return {c for c, v in zip(cells, pred) if v > 0}


# ---------------------------------------------------------------- extraction, one match
def records(match):
    with tarfile.open(REC / f"games_{match}.tar.gz", "r:*") as tf:
        for m in tf:
            name = m.name.rsplit("/", 1)[-1]
            if m.isfile() and name.endswith(".game.jsonl.gz"):
                game = name[: -len(".game.jsonl.gz")]
                lines = gzip.decompress(tf.extractfile(m).read()).decode("utf-8").splitlines()
                recs = [json.loads(x) for x in lines if x.strip()]
                yield game, next((r for r in recs if r.get("game_label") == game), recs[0])


def extract(match):
    beta = np.asarray(json.loads((REC / "readout_b.json").read_text())["beta"]["values"], dtype=float)
    rows = {}
    with open(REC / f"rows_{match}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if "game" in r and "S" in r and r.get("decisive"):
                rows[r["game"]] = r
    rule = rule_without(match)
    out_rows = []          # one per R-switch decision
    side_games = 0
    gi = 0
    for game, rec in records(match):
        row = rows.get(game)
        if row is None:
            continue
        side_games += 2
        side_a = int(row["side_a"])
        fac = {side_a: row["faction_a"], 3 - side_a: row["faction_b"]}
        R = int(row["z"]) - float(np.dot(beta, [row["S"][c] for c in COLUMNS]))
        cum, k = [], 0
        for c in rec["commands"]:
            if c[0] == "init_side":
                k += 1
            cum.append(k)
        band = {1: 0, 2: 0}
        dev = {1: 0, 2: 0}
        rank = {1: 0, 2: 0}
        any_r_d1 = {1: 0, 2: 0}
        seq = []                 # every band decision: (side, side-turn index, score of its coin in p)
        rows_here = []           # (index in seq, row) for R-switch decisions
        for e in (rec.get("coins") or {}).get("decisions", []):
            if e.get("refused"):
                continue
            side, p, d = int(e["side"]), float(e["p"]), int(e["d"])
            a0, a1 = e["a0"], e["a1"]
            opp = 3 - side
            st = cum[min(int(e.get("cmd", 0)), len(cum) - 1)]
            is_r = (a0["kind"] == "recruit" and a1["kind"] == "recruit" and a0["unit"] != a1["unit"]
                    and f"{fac[side]}|{a0['unit']}>{a1['unit']}" in rule)
            if is_r:
                s = 1 if side == side_a else -1
                T = s * (d - p) / (p * (1 - p)) * R
                rows_here.append((len(seq), [gi, side, round(T, 6), d, band[side], band[opp], dev[side], dev[opp],
                                             (st + 1) // 2, rank[side], any_r_d1[side]]))
                rank[side] += 1
                any_r_d1[side] = max(any_r_d1[side], d)
            band[side] += 1
            dev[side] += d          # under own1 (pi plays a0) d = 1 is a deviation from argmax
            seq.append((side, st, (d - p) / (p * (1 - p))))   # d log P(D) / dp: the coin's score in p
        # later coins' scores after each R-switch decision: own side, opponent, and within 4 side-turns
        for idx, r in rows_here:
            side, st0 = r[1], seq[idx][1]
            own = opp_s = win = 0.0
            for sd, st, sc in seq[idx + 1:]:
                if sd == side:
                    own += sc
                else:
                    opp_s += sc
                if st <= st0 + 4:
                    win += sc
            out_rows.append(r + [round(own, 4), round(opp_s, 4), round(win, 4)])
        gi += 1
    fields = ["game", "side", "T", "d", "K_own", "K_opp", "dev_own", "dev_opp", "turn", "rank", "prior_r_d1",
              "later_score_own", "later_score_opp", "later_score_window4"]
    OUT.with_name(OUT.name.format(match)).write_text(json.dumps(
        {"match": match, "side_games": side_games, "games": gi, "rule_cells": len(rule),
         "fields": fields, "rows": out_rows}))
    print(f"{match}: {gi} games, {len(out_rows)} R-switch decisions, rule of {len(rule)} cells")


# ---------------------------------------------------------------- summary
def load_rows():
    blocks, sg = [], 0
    for m in MATCHES:
        d = json.loads(OUT.with_name(OUT.name.format(m)).read_text())
        a = np.asarray(d["rows"], dtype=float)
        a[:, 0] += 1e6 * MATCHES.index(m)          # unique game ids across matches
        blocks.append(a)
        sg += d["side_games"]
    return np.vstack(blocks), sg


def section_a(a, sg):
    T, d, game, side, rank, prior = a[:, 2], a[:, 3], a[:, 0], a[:, 1], a[:, 9], a[:, 10]
    F = T.sum() / sg * ELO
    F_se = math.sqrt((T * T).sum()) / sg * ELO
    print(f"1. R, each match's rule fitted on the other six (ridge {RIDGE:.0f}): {len(T)} R-switch decisions, "
          f"{len(T) / sg:.2f} per side-game; first-order welfare F_LOMO {F:+.1f} +- {F_se:.1f} Elo per side-game "
          f"(the design's 20-fold F: +21.3 +- 3.1)")
    later = rank > 0
    print(f"   switches after an earlier R decision of the same side: {later.mean():.3f}; of those, with an "
          f"earlier one played as a1: {prior[later].mean():.3f}")
    for lab, mask in (("first R decision of its side", rank == 0), ("later R decisions", later)):
        print(f"   {lab}: {mask.sum() / sg:.2f} per side-game, welfare {T[mask].sum() / sg * ELO:+.1f} "
              f"+- {math.sqrt((T[mask] ** 2).sum()) / sg * ELO:.1f} Elo")
    # (a1) per switch: later labels by whether any earlier R decision of the side was played as a1
    x1, x0 = T[later & (prior == 1)], T[later & (prior == 0)]
    diff = x1.mean() - x0.mean()
    se = math.sqrt(x1.var() / len(x1) + x0.var() / len(x0))
    print(f"   per switch, later R decisions: earlier one played {x1.mean():+.4f} ({math.sqrt(x1.var() / len(x1)):.4f}, "
          f"n {len(x1)}), none played {x0.mean():+.4f} ({math.sqrt(x0.var() / len(x0)):.4f}, n {len(x0)}); "
          f"difference {diff:+.4f} +- {se:.4f} (z {diff / se:+.2f})  [pooled over ranks: confounded, see next lines]")
    # the pooled contrast mixes ranks: P(any earlier played) = 1 - 0.75^rank grows with rank, and the label's
    # mean falls with rank whatever was played. Within a rank the contrast is randomized.
    num = den = 0.0
    for r in (1, 2, 3, 4):
        m = (rank == r) if r < 4 else (rank >= r)
        y1, y0 = T[m & (prior == 1)], T[m & (prior == 0)]
        dd = y1.mean() - y0.mean()
        v = y1.var() / len(y1) + y0.var() / len(y0)
        num += dd / v
        den += 1 / v
        print(f"     rank {r}{'+' if r == 4 else ''}: mean label {T[m].mean():+.4f}; any earlier played {y1.mean():+.4f} "
              f"(n {len(y1)}), none {y0.mean():+.4f} (n {len(y0)}): {dd:+.4f} +- {math.sqrt(v):.4f}")
    print(f"   within-rank contrast, inverse-variance pooled: {num / den:+.4f} +- {math.sqrt(1 / den):.4f} "
          f"(z {num / den * math.sqrt(den):+.2f})")
    # (a2) per side-game: the first R decision's coin D1, and Y = sum of the side's later R-switch labels
    keys = game * 4 + side
    order = np.lexsort((rank, keys))
    k_sorted, T_s, d_s, r_s = keys[order], T[order], d[order], rank[order]  # noqa: F841 (an analysis script kept as run)
    starts = np.nonzero(np.r_[True, k_sorted[1:] != k_sorted[:-1]])[0]
    ends = np.r_[starts[1:], len(k_sorted)]
    D1 = d_s[starts]
    Y = np.array([T_s[s0 + 1:e0].sum() for s0, e0 in zip(starts, ends)])
    N = ends - starts - 1
    y1, y0 = Y[D1 == 1], Y[D1 == 0]
    dY = y1.mean() - y0.mean()
    dY_se = math.sqrt(y1.var() / len(y1) + y0.var() / len(y0))
    dN = N[D1 == 1].mean() - N[D1 == 0].mean()
    dN_se = math.sqrt(N[D1 == 1].var() / len(y1) + N[D1 == 0].var() / len(y0))
    share = len(starts) / sg
    corr = 0.75 * dY * share * ELO
    corr_se = 0.75 * dY_se * share * ELO
    print(f"   per side-game ({len(starts)} side-games with an R decision, {share:.3f} of side-games; D1 = 1 in "
          f"{D1.mean():.3f}): E[Y | D1 = 1] {y1.mean():+.4f}, E[Y | D1 = 0] {y0.mean():+.4f}, difference "
          f"{dY:+.4f} +- {dY_se:.4f} (z {dY / dY_se:+.2f}); later R decisions {N[D1 == 1].mean():.3f} against "
          f"{N[D1 == 0].mean():.3f} ({dN:+.3f} +- {dN_se:.3f})")
    print(f"   first-step correction of F for R's own interactions: 0.75 x difference x share = {corr:+.1f} +- "
          f"{corr_se:.1f} Elo per side-game")
    return F, F_se, corr, corr_se


def strata_ols(y, X, strata, clusters):
    """OLS of y on X with stratum fixed effects; cluster-robust (by game) covariance."""
    Xd, yd = X.copy(), y.copy()
    for s in np.unique(strata):
        m = strata == s
        Xd[m] -= Xd[m].mean(0)
        yd[m] -= yd[m].mean()
    XtX = Xd.T @ Xd
    b = np.linalg.solve(XtX, Xd.T @ yd)
    e = yd - Xd @ b
    order = np.argsort(clusters, kind="stable")
    cs, sc = clusters[order], Xd[order] * e[order, None]
    starts = np.nonzero(np.r_[True, cs[1:] != cs[:-1]])[0]
    G = np.add.reduceat(sc, starts, axis=0)
    inv = np.linalg.inv(XtX)
    V = inv @ (G.T @ G) @ inv
    return b, V


def section_b(a, sg, F, F_se):
    T, game = a[:, 2], a[:, 0]
    K = a[:, 4] + a[:, 5]
    dev_own, dev_opp, turn = a[:, 6], a[:, 7], a[:, 8]
    print("\n2. the coin world's perturbation of R's states (prior band decisions K and prior deviations, both sides)")
    print(f"   at R-switch decisions: mean K {K.mean():.1f}; prior deviations own {dev_own.mean():.2f}, "
          f"opponent {dev_opp.mean():.2f}; with no prior deviation by either side {np.mean(dev_own + dev_opp == 0):.3f}")
    for lo, hi in ((1, 1), (2, 2), (3, 3), (4, 6), (7, 99)):
        m = (turn >= lo) & (turn <= hi)
        print(f"   turn {lo}-{hi}: {m.sum() / sg:.2f} switches per side-game, welfare {T[m].sum() / sg * ELO:+.1f} "
              f"+- {math.sqrt((T[m] ** 2).sum()) / sg * ELO:.1f} Elo; mean prior deviations {dev_own[m].mean() + dev_opp[m].mean():.1f}")
    t1 = turn == 1
    rk = a[:, 9]
    print(f"   turn-1 switches by rank (earlier R decisions of the side): rank 0 {np.mean(rk[t1] == 0):.3f}, "
          f"rank 1 {np.mean(rk[t1] == 1):.3f}, rank 2+ {np.mean(rk[t1] >= 2):.3f}")
    zero = (dev_own + dev_opp) == 0
    print(f"   no prior deviation: {zero.sum() / sg:.3f} switches per side-game, welfare {T[zero].sum() / sg * ELO:+.1f} "
          f"+- {math.sqrt((T[zero] ** 2).sum()) / sg * ELO:.1f} Elo")
    kb = np.digitize(K, [1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 200])
    tb = np.minimum(turn, 8)
    strata = kb * 10 + tb
    b, V = strata_ols(T, np.column_stack([dev_own, dev_opp]), strata, game)
    se = np.sqrt(np.diag(V))
    print(f"   dose-response within (turn, K) strata, clustered by game: per own prior deviation {b[0]:+.5f} +- {se[0]:.5f}"
          f" (z {b[0] / se[0]:+.2f}); per opponent prior deviation {b[1]:+.5f} +- {se[1]:.5f} (z {b[1] / se[1]:+.2f})")
    m = np.array([dev_own.sum(), dev_opp.sum()])
    shift = float(m @ b) / sg * ELO
    shift_se = math.sqrt(float(m @ V @ m)) / sg * ELO
    print(f"   linear extrapolation to no prior deviation: F would move by {-shift:+.1f} +- {shift_se:.1f} Elo "
          f"(a model-based reading; F_LOMO {F:+.1f} +- {F_se:.1f})")
    bb, VV = strata_ols(T, (dev_own + dev_opp)[:, None].astype(float), strata, game)
    print(f"   one pooled slope: {bb[0]:+.5f} +- {math.sqrt(VV[0, 0]):.5f} per prior deviation")


def section_d():
    print("\n3. R's learning curve: rule fitted on k matches, read on the others (all subsets; mean over subsets)")
    cells, per = cell_sums()
    X = design(cells)
    for k in range(1, len(MATCHES)):
        ws, gs = [], []
        for sub in itertools.combinations(MATCHES, k):
            s = sum(per[m][0] for m in sub)
            n = sum(per[m][2] for m in sub)
            choose = fit_pred(X, s, n) > 0
            rest = [m for m in MATCHES if m not in sub]
            num = sum(per[m][0][choose].sum() for m in rest)
            den = sum(per[m][3] for m in rest)
            ws.append(num / den * ELO)
            gs.append(sum(per[m][4] for m in sub))
        ws, gs = np.asarray(ws), np.asarray(gs)
        print(f"   k {k}: training games {gs.mean():,.0f} (range {gs.min():,}-{gs.max():,}); read welfare "
              f"{ws.mean():+.1f} Elo (sd over {len(ws)} subsets {ws.std():.1f})")
    # slope of welfare on log training games between the two largest k, for the 60,000-game rule
    print("   (each reading's own SE is about 3.1 x sqrt(60,000 / read games); the subsets overlap, so the sd above is not an SE)")


def section_e():
    print("\n4. match wall model, wall = a + b x games, batch 1's band-coin matches (walls.txt, matches.txt)")
    data = [(2000, 382), (10000, 1666), (10000, 1612), (10000, 1618), (10000, 1608), (10000, 1601), (8000, 1296)]
    n = np.array([x for x, _ in data], dtype=float)
    w = np.array([y for _, y in data], dtype=float)
    A = np.column_stack([np.ones_like(n), n])
    coef, res, *_ = np.linalg.lstsq(A, w, rcond=None)
    resid = w - A @ coef
    s2 = float(resid @ resid) / (len(w) - 2)
    cov = s2 * np.linalg.inv(A.T @ A)
    a0, b0 = coef
    print(f"   a {a0:.0f} +- {math.sqrt(cov[0, 0]):.0f} s per match; b {b0 * 1000:.1f} +- {math.sqrt(cov[1, 1]) * 1000:.1f} ms "
          f"per game; residual sd {math.sqrt(s2):.0f} s")
    for o in (0.0, 0.1, 0.3, 0.56):
        dumped = (a0 + 2000 * b0 * (1 + o)) / 2000
        plain = (a0 + 1000 * b0) / 1000
        print(f"   true overhead o {o:.2f}: per-game wall of 2,000 dumped {dumped:.4f} s against 1,000 undumped "
              f"{plain:.4f} s reads o {dumped / plain - 1:+.3f}; with a removed it reads "
              f"{((dumped * 2000 - a0) / 2000) / ((plain * 1000 - a0) / 1000) - 1:+.3f}")
    srv = json.loads((REC / "b01.inference_server_0.json").read_text())
    print(f"   b01's server: GPU busy {srv['gpu_ms'] / 1000:.0f} s of {srv['wall_s']:.0f} s wall "
          f"({srv['gpu_ms'] / 1000 / srv['wall_s']:.2f}); idle {srv['idle_s']:.0f} s; stage seconds "
          + ", ".join(f"{k[2:]} {v:.0f}" for k, v in srv["stage_s"].items()))


def section_c(a, sg):
    """The continuation's coins: dDelta_j/dp = E[T_j x sum_{k>j} (D_k - p)/(p(1 - p))] exactly (score of each
    later coin in p, applied through the label identity); F(p = 0) ~ F(0.25) - 0.25 dF/dp is a Taylor step."""
    if a.shape[1] < 14:
        print("\n2b. (later-coin scores not extracted)")
        return
    T, game = a[:, 2], a[:, 0]
    print("\n2b. the continuation: derivative of R's summed advantage in the continuation's deviation probability p")
    for name, col in (("all later coins", None), ("own side's later coins", 11), ("opponent's later coins", 12),
                      ("later coins within 4 side-turns", 13)):
        sc = a[:, 11] + a[:, 12] if col is None else a[:, col]
        v = T * sc
        order = np.argsort(game, kind="stable")
        g = game[order]
        starts = np.nonzero(np.r_[True, g[1:] != g[:-1]])[0]
        se = math.sqrt(float(np.sum(np.add.reduceat(v[order], starts) ** 2))) / sg
        der = v.sum() / sg
        print(f"   {name:34s}: dF/dp {der:+.4f} +- {se:.4f} z per side-game; Taylor step to p = 0 moves F by "
              f"{-0.25 * der * ELO:+.1f} +- {0.25 * se * ELO:.1f} Elo")


def summary():
    a, sg = load_rows()
    F, F_se, corr, corr_se = section_a(a, sg)
    section_b(a, sg, F, F_se)
    section_c(a, sg)
    section_d()
    section_e()
    miss = 2 * math.sqrt(3.1 ** 2 + (ELO / math.sqrt(37_800)) ** 2)
    print(f"\n5. L's line: sealed F +21.3 +- 3.1, match SE {ELO / math.sqrt(37_800):.2f}, miss half-width {miss:.1f}; "
          f"R's own-interaction correction {corr:+.1f} +- {corr_se:.1f} (|correction| / half-width {abs(corr) / miss:.2f})")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "extract":
        extract(sys.argv[2])
    else:
        summary()
