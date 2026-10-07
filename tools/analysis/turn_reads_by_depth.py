#!/usr/bin/env python3
"""How well a read taken after rollouts ranks candidate turns, by depth.

On the turn-value validation records (tier-b/turn_value_20260925/
validation.json on the model host: 200 holdout positions, 5 candidate turns
each, 28 playouts per candidate by the reference), for the value head and
for the HP margin read at each of the first 8 player turn starts after the
candidate turn (read 0 is right after it; the mover's own turn starts k turns
later are reads 2k - 1): the noise-corrected within-position correlation of
the mean read over the first 8 playouts with the other 20 playouts' raw
outcomes. A read past the end of a game takes the outcome for the value and
nothing for the margin, as the verdict's arrays do, so the margin averages
only the playouts still running.

The measure is the verdict's own, copied from tools/turn_value_fit.py on
branch exp/turn-value (`corrected_correlation`), and the value column
reproduces the verdict's raw-outcome figures (read 1 0.311, read 3 0.449,
read 7 0.588).

    python tools/analysis/turn_reads_by_depth.py --records validation.json [--verdict verdict.json]
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np

GRADE_PLAYOUTS = 8          # the rollout grader's playouts, as in the verdict
HORIZON_READS = 8
BOOTSTRAP_RESAMPLES = 1000


def _position_sums(grade, truth, positions, clusters):
    n = np.isfinite(truth).sum(axis=1)
    usable = np.isfinite(grade) & (n >= 2)
    rows_by_position = {}
    for row in np.nonzero(usable)[0]:
        rows_by_position.setdefault(int(positions[row]), []).append(int(row))
    sums, owners = [], []
    for rows in rows_by_position.values():
        if len(rows) < 2:
            continue
        g = grade[rows]
        y = np.nanmean(truth[rows], axis=1)
        noise = np.nanvar(truth[rows], axis=1, ddof=1) / n[rows]
        gc, yc = g - g.mean(), y - y.mean()
        sums.append((gc @ yc, gc @ gc, yc @ yc, (1 - 1 / len(rows)) * noise.sum()))
        owners.append(clusters[rows[0]])
    return np.asarray(owners), np.asarray(sums, dtype=float).reshape(-1, 4)


def _ratios(s):
    sgy, sgg, syy, noise = s
    if sgg <= 0 or syy <= 0:
        return math.nan, math.nan, math.nan
    reliability = 1.0 - noise / syy
    observed = sgy / math.sqrt(sgg * syy)
    corrected = observed / math.sqrt(reliability) if reliability > 0 else math.nan
    return observed, reliability, corrected


def _percentile_se(draws):
    finite = draws[np.isfinite(draws)]
    if len(finite) < 10:
        return math.nan
    low, high = np.percentile(finite, [15.87, 84.13])
    return float(high - low) / 2.0


def corrected_correlation(grade, truth, positions, clusters, seed: int = 0) -> dict:
    owners, sums = _position_sums(grade, truth, positions, clusters)
    _, cluster_of = np.unique(owners, return_inverse=True)
    per_cluster = np.zeros((cluster_of.max() + 1, 4))
    np.add.at(per_cluster, cluster_of, sums)
    _, reliability, corrected = _ratios(per_cluster.sum(axis=0))
    rng = np.random.default_rng(seed)
    n = len(per_cluster)
    boots = np.array([_ratios(per_cluster[rng.integers(0, n, n)].sum(axis=0))
                      for _ in range(BOOTSTRAP_RESAMPLES)])
    return {"positions": len(sums), "corrected": corrected,
            "corrected_se": _percentile_se(boots[:, 2]), "reliability": reliability}


def load(records: Path):
    """Per candidate: its position, its source game, its outcomes [P] and
    its reads [P, 8, (value, margin)]."""
    data = json.loads(records.read_text(encoding="utf-8"))
    rows = []
    for pos in data["positions"]:
        for cand in [pos["base"]] + list(pos.get("alternatives", [])):
            if cand.get("terminal_in_turn") or not cand.get("reads"):
                continue
            rows.append((pos["index"], pos["meta"]["file"], cand["outcomes"],
                         [r["horizon"] for r in cand["reads"]]))
    playouts = max(len(r[2]) for r in rows)
    outcomes = np.full((len(rows), playouts), np.nan)
    reads = np.full((len(rows), playouts, HORIZON_READS, 2), np.nan)
    for i, (_, _, outs, horizons) in enumerate(rows):
        outcomes[i, :len(outs)] = outs
        for p, horizon in enumerate(horizons):
            for k in range(HORIZON_READS):
                if k >= len(horizon):
                    reads[i, p, k, 0] = outs[p]
                    continue
                value, margin = horizon[k]
                reads[i, p, k, 0] = np.nan if value is None else value
                reads[i, p, k, 1] = np.nan if margin is None else margin
    return (np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), outcomes, reads)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--records", type=Path, required=True, help="the turn-value validation.json")
    ap.add_argument("--verdict", type=Path, default=None, help="its verdict.json, to print the recorded figures beside")
    args = ap.parse_args(argv)
    positions, clusters, outcomes, reads = load(args.records)
    truth = outcomes[:, GRADE_PLAYOUTS:]
    recorded = {}
    if args.verdict:
        v = json.loads(args.verdict.read_text(encoding="utf-8"))["validation"]
        recorded = {1: v["rollout_r8_h1"]["raw"]["corrected"], 3: v["rollout"]["raw"]["corrected"],
                    7: v["rollout_r8_h7"]["raw"]["corrected"]}
    print(f"{len(positions)} candidates at {len(set(positions.tolist()))} positions, "
          f"{outcomes.shape[1]} playouts each; grade over the first {GRADE_PLAYOUTS}, truth the rest (raw outcomes)")
    print("read  finished  value head          (recorded)  HP margin")
    for k in range(HORIZON_READS):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)     # a candidate with no read: NaN
            value = np.nanmean(reads[:, :GRADE_PLAYOUTS, k, 0], axis=1)
            margin = np.nanmean(reads[:, :GRADE_PLAYOUTS, k, 1], axis=1)
        finished = float((np.isnan(reads[:, :GRADE_PLAYOUTS, k, 1])
                          & ~np.isnan(reads[:, :GRADE_PLAYOUTS, k, 0])).mean())
        v = corrected_correlation(value, truth, positions, clusters)
        m = corrected_correlation(margin, truth, positions, clusters)
        rec = f"{recorded[k]:.3f}" if k in recorded else "-"
        print(f"{k:>4}  {finished:8.3f}  {v['corrected']:.3f} +- {v['corrected_se']:.3f}   {rec:>9}   "
              f"{m['corrected']:.3f} +- {m['corrected_se']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
