"""Round 10, challenge stance: does the recruit near-tie structure depend on the
matchup (the opponent's faction) and on the game phase, beyond the 43
per-(faction, unit) values of rule R?

Two modes, one process each, numpy and the standard library only:

    python round10_challenge_recruit_context.py extract MATCH
    python round10_challenge_recruit_context.py combine

extract reads games_MATCH.tar.gz and rows_MATCH.jsonl as
round9_provenance_cells.py does (T = s * omega * (z - beta.S), beta from
readout_b.json, capped games dropped, 20 folds by SHA-256 of the game name)
and writes n, sum T, sum T^2 per (own faction, u0 > u1, enemy faction, turn
bin, fold) for recruit pairs of two different units.

combine, every rule fitted on 19 folds and read on the 20th (cross-fitted,
first order, per side-game, Elo = z x 800/ln 10):
  A. additive values per (own faction, unit): 43 values (rule R's form);
  B. additive values per (own faction, enemy faction, unit), shrunk toward A
     (ridge on the deviation from A, A refitted within the 19 folds);
  C. additive values per (own faction, unit, turn bin), shrunk toward A;
for ridge in labels {2,000, 5,000, 20,000}; and B - A, C - A paired on the
same folds (SE from the per-fold differences).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REC = HERE.parents[2] / "training" / "metrics" / "near_tie_coins_20261020"
MATCHES = ("b00", "b01", "b02", "b03", "b04", "b05", "b06")
FOLDS = 20
ELO = 800 / math.log(10)
COLUMNS = ("hp", "material", "material_nonterminal", "terminal", "value_50", "value_100", "value_200")
TURN_EDGES = (1, 2, 3, 5, 9, 10**6)  # turn 1, turn 2, turns 3-4, turns 5-8, turns 9+


def fold_of(game: str) -> int:
    return hashlib.sha256(game.encode("utf-8")).digest()[0] % FOLDS


def turn_bin(t: int) -> int:
    for k in range(len(TURN_EDGES) - 1):
        if TURN_EDGES[k] <= t < TURN_EDGES[k + 1]:
            return k
    return len(TURN_EDGES) - 2


def records(match: str):
    with tarfile.open(REC / f"games_{match}.tar.gz", "r:*") as tf:
        for m in tf:
            name = m.name.rsplit("/", 1)[-1]
            if m.isfile() and name.endswith(".game.jsonl.gz"):
                game = name[: -len(".game.jsonl.gz")]
                lines = gzip.decompress(tf.extractfile(m).read()).decode("utf-8").splitlines()
                recs = [json.loads(x) for x in lines if x.strip()]
                yield game, next((r for r in recs if r.get("game_label") == game), recs[0])


def extract(match: str) -> None:
    beta = np.asarray(json.loads((REC / "readout_b.json").read_text())["beta"]["values"], dtype=float)
    rows = {}
    with open(REC / f"rows_{match}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if "game" in r and "S" in r:
                rows[r["game"]] = r
    acc = defaultdict(lambda: [0, 0.0, 0.0])
    games = side_games = 0
    for game, rec in records(match):
        row = rows.get(game)
        if row is None or not row.get("decisive"):
            continue
        z = int(row["z"])
        side_a = int(row["side_a"])
        R = z - float(np.dot(beta, [row["S"][c] for c in COLUMNS]))
        fac = {side_a: row["faction_a"], 3 - side_a: row["faction_b"]}
        f = fold_of(game)
        games += 1
        side_games += 2
        cum, n = [], 0
        for c in rec["commands"]:
            if c[0] == "init_side":
                n += 1
            cum.append(n)
        for e in (rec.get("coins") or {}).get("decisions", []):
            if e.get("refused"):
                continue
            a0, a1 = e["a0"], e["a1"]
            if a0["kind"] != "recruit" or a1["kind"] != "recruit" or a0["unit"] == a1["unit"]:
                continue
            side = int(e["side"])
            p, d = float(e["p"]), int(e["d"])
            t = (1 if side == side_a else -1) * (d - p) / (p * (1 - p)) * R
            k = int(e.get("cmd", 0))
            turn = (cum[min(k, len(cum) - 1)] + 1) // 2
            key = f"{fac[side]}|{a0['unit']}|{a1['unit']}|{fac[3 - side]}|{turn_bin(turn)}|{f}"
            a = acc[key]
            a[0] += 1
            a[1] += t
            a[2] += t * t
    out = {"match": match, "games": games, "side_games": side_games, "cells": dict(acc)}
    (HERE / f"round10_challenge_recruit_context_{match}.json").write_text(json.dumps(out))
    print(f"{match}: {games} games, {sum(v[0] for v in acc.values())} different-unit recruit pairs")


def load():
    side_games = 0
    rows = []
    for m in MATCHES:
        d = json.loads((HERE / f"round10_challenge_recruit_context_{m}.json").read_text())
        side_games += d["side_games"]
        for key, (n, s, ss) in d["cells"].items():
            fac, u0, u1, enemy, tb, f = key.split("|")
            rows.append((fac, u0, u1, enemy, int(tb), int(f), n, s, ss))
    return side_games, rows


def design(rows, key_fn):
    """Index the value columns: for each row, the column of u1 and of u0 under key_fn."""
    names = {}
    c0, c1 = [], []
    for r in rows:
        k1, k0 = key_fn(r, r[2]), key_fn(r, r[1])
        for k in (k0, k1):
            if k not in names:
                names[k] = len(names)
        c1.append(names[k1])
        c0.append(names[k0])
    return np.asarray(c0), np.asarray(c1), len(names)


def fit(c0, c1, m, n, s, mask, lam, prior=None):
    """Label-weighted ridge of cell means on e(u1) - e(u0); prior is a per-row
    predicted mean the fit is shrunk toward (the deviation is penalised)."""
    i = np.flatnonzero(mask)
    j1, j0, w_ = c1[i], c0[i], n[i]
    resid = s[i] - (0.0 if prior is None else n[i] * prior[i])
    A = np.zeros((m, m))
    np.add.at(A, (j1, j1), w_)
    np.add.at(A, (j0, j0), w_)
    np.add.at(A, (j1, j0), -w_)
    np.add.at(A, (j0, j1), -w_)
    b = np.zeros(m)
    np.add.at(b, j1, resid)
    np.add.at(b, j0, -resid)
    w = np.linalg.solve(A + lam * np.eye(m), b)
    pred = w[c1] - w[c0]
    return pred if prior is None else prior + pred


def welfare(pred, s, ss, test):
    """Switch iff the predicted mean is above 0; sum of T and of T^2 over switched rows."""
    sw = test & (pred > 0)
    return float(np.sum(s[sw])), float(np.sum(ss[sw]))


def combine() -> None:
    sg, rows = load()
    n = np.asarray([r[6] for r in rows], dtype=float)
    s = np.asarray([r[7] for r in rows])
    ss = np.asarray([r[8] for r in rows])
    folds = np.asarray([r[5] for r in rows])
    cA = design(rows, lambda r, u: f"{r[0]}|{u}")
    cB = design(rows, lambda r, u: f"{r[0]}|{u}|{r[3]}")
    cC = design(rows, lambda r, u: f"{r[0]}|{u}|t{r[4]}")
    print(f"{len(rows)} cells, {int(n.sum())} different-unit recruit pairs, {sg} side-games")
    print(f"columns: A {cA[2]} (faction, unit); B {cB[2]} (faction, unit, enemy faction); C {cC[2]} (faction, unit, turn bin)")
    for lam in (2000.0, 5000.0, 20000.0):
        per_fold = {k: np.zeros((FOLDS, 2)) for k in ("A", "B", "C")}
        for f in range(FOLDS):
            train, test = folds != f, folds == f
            pA = fit(cA[0], cA[1], cA[2], n, s, train, lam)
            pB = fit(cB[0], cB[1], cB[2], n, s, train, lam, prior=pA)
            pC = fit(cC[0], cC[1], cC[2], n, s, train, lam, prior=pA)
            for k, p in (("A", pA), ("B", pB), ("C", pC)):
                per_fold[k][f] = welfare(p, s, ss, test)
        print(f"\nridge {lam:,.0f} labels (B and C shrunk toward A at the same ridge)")
        for k in ("A", "B", "C"):
            tot, tot2 = per_fold[k].sum(0)
            print(f"  {k}: {tot / sg:+.5f} ({math.sqrt(tot2) / sg:.5f}) z = {tot / sg * ELO:+.1f} ({math.sqrt(tot2) / sg * ELO:.1f}) Elo per side-game")
        for k in ("B", "C"):
            d = (per_fold[k][:, 0] - per_fold["A"][:, 0]) / sg * ELO
            print(f"  {k} - A: {d.sum():+.2f} Elo, SE from the {FOLDS} per-fold differences {math.sqrt(FOLDS) * d.std(ddof=1):.2f}")
    print("\nby turn bin (turn 1, turn 2, turns 3-4, turns 5-8, turns 9+): pairs per side-game and A's cross-fitted welfare, ridge 5,000")
    tb = np.asarray([r[4] for r in rows])
    acc = np.zeros((len(TURN_EDGES) - 1, 3))
    for f in range(FOLDS):
        train, test = folds != f, folds == f
        pA = fit(cA[0], cA[1], cA[2], n, s, train, 5000.0)
        for b_ in range(len(TURN_EDGES) - 1):
            t = test & (tb == b_)
            w, w2 = welfare(pA, s, ss, t)
            acc[b_] += (w, w2, n[t].sum())
    for b_, name in enumerate(("turn 1", "turn 2", "turns 3-4", "turns 5-8", "turns 9+")):
        w, w2, cnt = acc[b_]
        print(f"  {name:10s} {cnt / sg:.2f} pairs per side-game; welfare {w / sg * ELO:+.1f} ({math.sqrt(w2) / sg * ELO:.1f}) Elo")


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(sys.argv[2])
    else:
        combine()
