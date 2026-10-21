"""Round 10, challenge stance: the first-order in-band curve of a recruit value
shift, the sealed in-band part of the line search's predictions.

The shift rule: when the network's argmax is a recruit, the unit is re-chosen
among the recruit candidates by log-prior + eta * v(faction, unit, enemy
faction); the network still decides whether to recruit. At a band decision
whose two candidates recruit different units u0 (top) and u1 (runner-up) with
joint gap G, the shift plays u1 iff eta * (v(u1) - v(u0)) > G. Batch 1 logs
G only inside the band (G < 0.35), so this curve is the first-order gain of
the flips inside the band; flips of decisions outside the band are not
labelled and are what the play line search reads.

    python round10_challenge_shift_curve.py extract MATCH
    python round10_challenge_shift_curve.py combine

extract: per different-unit recruit band pair, n, sum T, sum T^2 per (own
faction, u0, u1, enemy faction, fold, gap bin of 0.025). Labels as in
round9_provenance_cells.py. combine: A (43 values, ridge 5,000) and B
(matchup values shrunk toward A, ridge 5,000, the progressive loss's choice in
round10_challenge_recruit_context_ridge.out), each fitted on 19 folds and read
on the 20th; for each eta, the cross-fitted welfare of the in-band flips and
their count per side-game; and the same for the reversed direction (-v). numpy and the standard library only.
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

from round10_challenge_recruit_context import COLUMNS, ELO, FOLDS, MATCHES, REC, design, fit

HERE = Path(__file__).resolve().parent
BIN = 0.025
ETAS = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 40.0, 1e6)


def fold_of(game: str) -> int:
    return hashlib.sha256(game.encode("utf-8")).digest()[0] % FOLDS


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
    side_games = 0
    for game, rec in records(match):
        row = rows.get(game)
        if row is None or not row.get("decisive"):
            continue
        side_a = int(row["side_a"])
        R = int(row["z"]) - float(np.dot(beta, [row["S"][c] for c in COLUMNS]))
        fac = {side_a: row["faction_a"], 3 - side_a: row["faction_b"]}
        f = fold_of(game)
        side_games += 2
        for e in (rec.get("coins") or {}).get("decisions", []):
            if e.get("refused"):
                continue
            a0, a1 = e["a0"], e["a1"]
            if a0["kind"] != "recruit" or a1["kind"] != "recruit" or a0["unit"] == a1["unit"]:
                continue
            side = int(e["side"])
            p, d = float(e["p"]), int(e["d"])
            t = (1 if side == side_a else -1) * (d - p) / (p * (1 - p)) * R
            gb = min(int(float(e["gap"]) / BIN), 13)
            a = acc[f"{fac[side]}|{a0['unit']}|{a1['unit']}|{fac[3 - side]}|{gb}|{f}"]
            a[0] += 1
            a[1] += t
            a[2] += t * t
    (HERE / f"round10_challenge_shift_curve_{match}.json").write_text(
        json.dumps({"match": match, "side_games": side_games, "cells": dict(acc)}))
    print(f"{match}: {side_games} side-games, {sum(v[0] for v in acc.values())} pairs")


def combine() -> None:
    rows, sg = [], 0
    for m in MATCHES:
        d = json.loads((HERE / f"round10_challenge_shift_curve_{m}.json").read_text())
        sg += d["side_games"]
        for key, (n, s, ss) in d["cells"].items():
            fac, u0, u1, enemy, gb, f = key.split("|")
            rows.append((fac, u0, u1, enemy, int(gb), int(f), n, s, ss))
    n = np.asarray([r[6] for r in rows], dtype=float)
    s = np.asarray([r[7] for r in rows])
    ss = np.asarray([r[8] for r in rows])
    folds = np.asarray([r[5] for r in rows])
    gmid = (np.asarray([r[4] for r in rows]) + 0.5) * BIN
    cA = design(rows, lambda r, u: f"{r[0]}|{u}")
    cB = design(rows, lambda r, u: f"{r[0]}|{u}|{r[3]}")
    preds = {"A": np.zeros(len(rows)), "B": np.zeros(len(rows))}
    for f in range(FOLDS):
        train, test = folds != f, folds == f
        pA = fit(cA[0], cA[1], cA[2], n, s, train, 5000.0)
        pB = fit(cB[0], cB[1], cB[2], n, s, train, 5000.0, prior=pA)
        preds["A"][test] = pA[test]
        preds["B"][test] = pB[test]
    print(f"{int(n.sum())} different-unit recruit band pairs, {sg} side-games; out-of-fold predictions p = v(u1) - v(u0)")
    for k in ("A", "B"):
        p = preds[k]
        print(f"\n{k}: sd of p over pairs {math.sqrt(np.average(p * p, weights=n) - np.average(p, weights=n) ** 2):.4f};"
              f" share with p > 0 {n[p > 0].sum() / n.sum():.3f}")
        print("  eta     flips/side-game   in-band first-order welfare, Elo per side-game (SE)")
        for eta in ETAS:
            sw = eta * p > gmid
            w, w2 = s[sw].sum(), ss[sw].sum()
            name = "inf" if eta >= 1e6 else f"{eta:g}"
            print(f"  {name:>5s}   {n[sw].sum() / sg:6.2f}           {w / sg * ELO:+6.1f} ({math.sqrt(w2) / sg * ELO:.1f})")
        print("  reversed direction (v -> -v), the shift path's known answer: eta, flips/side-game, in-band welfare")
        for eta in ETAS[:-1]:
            sw = -eta * p > gmid
            w, w2 = s[sw].sum(), ss[sw].sum()
            print(f"  {eta:>5g}   {n[sw].sum() / sg:6.2f}           {w / sg * ELO:+6.1f} ({math.sqrt(w2) / sg * ELO:.1f})")


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(sys.argv[2])
    else:
        combine()
