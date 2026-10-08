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

The second table is what a planner would use: at each position, the
candidate the read ranks first, and the gain of its truth over the base
turn's (the player's own), averaged over positions, with a bootstrap over
positions; beside it, the same read's gain minus the static read's (read 0),
paired. With `--verdict`, both tables are also given against outcomes with
each playout's own luck removed (the verdict's luck coefficients for the
playout's HP and kill luck). The candidate turn's own dice stay in the
truth on both bases: every playout starts from the turn's realized
outcomes.

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
GAIN_RESAMPLES = 4000


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


def load(records: Path) -> dict:
    """Per candidate: its position, its source game, its slot (0 for the
    base turn), its outcomes [P], its playouts' HP and kill luck [P, 2] and
    its reads [P, 8, (value, margin)]."""
    data = json.loads(records.read_text(encoding="utf-8"))
    rows = []
    for pos in data["positions"]:
        for slot, cand in enumerate([pos["base"]] + list(pos.get("alternatives", []))):
            if cand.get("terminal_in_turn") or not cand.get("reads"):
                continue
            rows.append((pos["index"], pos["meta"]["file"], slot, cand["outcomes"], cand["reads"]))
    playouts = max(len(r[3]) for r in rows)
    outcomes = np.full((len(rows), playouts), np.nan)
    luck = np.full((len(rows), playouts, 2), np.nan)
    reads = np.full((len(rows), playouts, HORIZON_READS, 2), np.nan)
    for i, (_, _, _, outs, playout_reads) in enumerate(rows):
        outcomes[i, :len(outs)] = outs
        for p, read in enumerate(playout_reads):
            if read.get("luck"):
                luck[i, p] = (read["luck"]["hp"], read["luck"]["kills"])
            horizon = read["horizon"]
            for k in range(HORIZON_READS):
                if k >= len(horizon):
                    reads[i, p, k, 0] = outs[p]
                    continue
                value, margin = horizon[k]
                reads[i, p, k, 0] = np.nan if value is None else value
                reads[i, p, k, 1] = np.nan if margin is None else margin
    return {"positions": np.array([r[0] for r in rows]), "clusters": np.array([r[1] for r in rows]),
            "slots": np.array([r[2] for r in rows]), "outcomes": outcomes, "luck": luck, "reads": reads}


def without_playout_luck(outcomes: np.ndarray, luck: np.ndarray, beta) -> np.ndarray:
    """The outcomes minus each playout's HP and kill luck, centred, by the
    verdict's coefficients for them (its luck beta[0] and beta[1])."""
    centred = luck - np.nanmean(luck, axis=(0, 1))
    return outcomes - beta[0] * np.nan_to_num(centred[..., 0]) - beta[1] * np.nan_to_num(centred[..., 1])


def selection_gains(grade: np.ndarray, truth: np.ndarray, positions: np.ndarray,
                    slots: np.ndarray) -> dict:
    """Per position: the truth of the candidate `grade` ranks first minus
    the base turn's (slot 0)."""
    gains = {}
    for position in np.unique(positions):
        rows = np.nonzero(positions == position)[0]
        base = rows[slots[rows] == 0]
        if len(rows) < 2 or len(base) == 0 or not np.isfinite(grade[rows]).any():
            continue
        best = rows[np.nanargmax(grade[rows])]
        gains[int(position)] = truth[best] - truth[base[0]]
    return gains


def mean_and_se(values: np.ndarray, seed: int = 3):
    rng = np.random.default_rng(seed)
    boots = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(GAIN_RESAMPLES)]
    return float(values.mean()), float(np.std(boots))


def mean_read(reads: np.ndarray, k: int, column: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # a candidate with no read: NaN
        return np.nanmean(reads[:, :GRADE_PLAYOUTS, k, column], axis=1)


def gain_table(d: dict, truth: np.ndarray, label: str) -> None:
    truth_mean = np.nanmean(truth, axis=1)
    static = selection_gains(mean_read(d["reads"], 0, 1), truth_mean, d["positions"], d["slots"])
    print(f"\nselection gain over the base turn, truth: {label}")
    print("read  HP margin          minus static read   value head")
    for k in range(HORIZON_READS):
        margin = selection_gains(mean_read(d["reads"], k, 1), truth_mean, d["positions"], d["slots"])
        value = selection_gains(mean_read(d["reads"], k, 0), truth_mean, d["positions"], d["slots"])
        common = sorted(set(margin) & set(static))
        m, m_se = mean_and_se(np.array([margin[p] for p in common]))
        dm, dm_se = mean_and_se(np.array([margin[p] - static[p] for p in common]))
        v, v_se = mean_and_se(np.array(list(value.values())))
        print(f"{k:>4}  {m:+.3f} +- {m_se:.3f}   {dm:+.3f} +- {dm_se:.3f}     {v:+.3f} +- {v_se:.3f}"
              f"   positions {len(common)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--records", type=Path, required=True, help="the turn-value validation.json")
    ap.add_argument("--verdict", type=Path, default=None, help="its verdict.json, to print the recorded figures beside")
    args = ap.parse_args(argv)
    d = load(args.records)
    positions, clusters, outcomes, reads = d["positions"], d["clusters"], d["outcomes"], d["reads"]
    truth = outcomes[:, GRADE_PLAYOUTS:]
    recorded, beta = {}, None
    if args.verdict:
        verdict = json.loads(args.verdict.read_text(encoding="utf-8"))
        v = verdict["validation"]
        recorded = {1: v["rollout_r8_h1"]["raw"]["corrected"], 3: v["rollout"]["raw"]["corrected"],
                    7: v["rollout_r8_h7"]["raw"]["corrected"]}
        beta = verdict["luck"]["beta"]
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
    gain_table(d, truth, "raw outcomes")
    if beta is not None:
        gain_table(d, without_playout_luck(outcomes, d["luck"], beta)[:, GRADE_PLAYOUTS:],
                   "outcomes without the playouts' own luck")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
