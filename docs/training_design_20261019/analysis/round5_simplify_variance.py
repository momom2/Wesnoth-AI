"""Round 5, simplify stance: the result's variance in recorded argmax matches,
and how much of it the setup (map, the two factions, who moves first) explains.

Input: training/metrics/match_luck_20261019/rows.jsonl (one row per walked
game of 23 argmax matches: result z for side A, map, factions, side numbers,
whether the game was decisive).

Output (stdout and round5_simplify_variance.json):
- per match and for the own1 self-play matches: games, decisive share,
  mean z and Var(z) over decisive games;
- the share of Var(z) a setup baseline explains, cross-fitted over two
  folds: the first mover's result minus the match's side-A mean (signed),
  regressed on cell means of (map, first-mover faction, second-mover
  faction), each cell mean shrunk by n/(n + k) with k the within/between
  variance ratio fitted on the training fold. Also map-only and
  first-mover-only baselines.

Standard library and numpy only.
"""
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC = ROOT / "training/metrics/match_luck_20261019/rows.jsonl"
OUT = pathlib.Path(__file__).with_suffix(".json")
OWN1_SELFPLAY = ("gen_own1_selfplay", "own64_selfplay")


def load():
    rows = [json.loads(line) for line in SRC.open()]
    return rows


def match_summary(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["match"]].append(r)
    out = {}
    for m, rs in sorted(by.items()):
        dec = [r for r in rs if r["decisive"]]
        z = np.array([r["z"] for r in dec], dtype=float)
        out[m] = {"games": len(rs), "decisive": len(dec), "decisive_share": len(dec) / len(rs),
                  "mean_z": float(z.mean()), "var_z": float(z.var(ddof=1))}
    return out


def first_mover_view(r):
    """(z of the side that moves first, its faction, the other's faction, sign
    that maps side A to the first mover)."""
    if r["side_a"] == 1:
        return r["z"], r["faction_a"], r["faction_b"], 1.0
    return -r["z"], r["faction_b"], r["faction_a"], -1.0


def setup_share(rows, key_fn, seed=20261020, with_luck=False):
    dec = [r for r in rows if r["decisive"]]
    match_mean = defaultdict(list)
    for r in dec:
        match_mean[r["match"]].append(r["z"])
    match_mean = {m: float(np.mean(v)) for m, v in match_mean.items()}
    y, keys = [], []
    for r in dec:
        z1, f1, f2, sign = first_mover_view(r)
        # remove the strength difference of the two players of the match
        y.append(z1 - sign * match_mean[r["match"]])
        keys.append(repr(key_fn(r, f1, f2)))
    y = np.array(y)
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, 2, len(y))
    resid = np.empty_like(y)
    for f in (0, 1):
        tr, te = fold != f, fold == f
        groups = defaultdict(list)
        for yi, k in zip(y[tr], [k for k, t in zip(keys, tr) if t]):
            groups[k].append(yi)
        grand = float(y[tr].mean())
        means = {k: (np.mean(v), len(v)) for k, v in groups.items()}
        within = np.mean([np.var(v, ddof=1) for v in groups.values() if len(v) > 1])
        cell_means = np.array([m for m, _ in means.values()])
        cell_ns = np.array([n for _, n in means.values()])
        between = max(np.var(cell_means - grand) - np.mean(within / cell_ns), 1e-9)
        k_shrink = within / between
        for i in np.flatnonzero(te):
            m, n = means.get(keys[i], (grand, 0))
            pred = grand + (n / (n + k_shrink)) * (m - grand) if n else grand
            resid[i] = y[i] - pred
    summary = {"games": len(y), "cells": len(set(keys)),
               "var_y": float(y.var()), "var_resid": float(resid.var()),
               "explained_share": float(1 - resid.var() / y.var())}
    if with_luck:
        # the seven luck columns of tools/match_luck.py, signed to the first
        # mover, regressed on the setup residual (cross-fitted, same folds)
        cols = sorted(dec[0]["S"])
        X = np.array([[first_mover_view(r)[3] * r["S"][c] for c in cols] for r in dec])
        X = np.column_stack([np.ones(len(y)), X])
        resid2 = np.empty_like(y)
        for f in (0, 1):
            tr, te = fold != f, fold == f
            beta, *_ = np.linalg.lstsq(X[tr], resid[tr], rcond=None)
            resid2[te] = resid[te] - X[te] @ beta
        summary["with_luck_columns"] = {"var_resid": float(resid2.var()),
                                        "explained_share": float(1 - resid2.var() / y.var())}
        resid3 = np.empty_like(y)
        for f in (0, 1):
            tr, te = fold != f, fold == f
            beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
            resid3[te] = y[te] - X[te] @ beta
        summary["luck_columns_only"] = {"var_resid": float(resid3.var()),
                                        "explained_share": float(1 - resid3.var() / y.var())}
    return summary


def main():
    rows = load()
    out = {"matches": match_summary(rows)}
    own = [r for r in rows if r["match"] in OWN1_SELFPLAY]
    dec = [r for r in own if r["decisive"]]
    z = np.array([r["z"] for r in dec], dtype=float)
    out["own1_selfplay"] = {"games": len(own), "decisive": len(dec),
                           "mean_z": float(z.mean()), "var_z": float(z.var(ddof=1)),
                           "mean_turns": float(np.mean([r["turns"] for r in own]))}
    out["setup_baselines_all_matches"] = {
        "first_mover_only": setup_share(rows, lambda r, f1, f2: 0),
        "map": setup_share(rows, lambda r, f1, f2: r["scenario_id"]),
        "factions": setup_share(rows, lambda r, f1, f2: (f1, f2)),
        "map_x_factions": setup_share(rows, lambda r, f1, f2: (r["scenario_id"], f1, f2)),
        "factions_and_luck_columns": setup_share(rows, lambda r, f1, f2: (f1, f2), with_luck=True),
    }
    json.dump(out, OUT.open("w"), indent=1)
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
