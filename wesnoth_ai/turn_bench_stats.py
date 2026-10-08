"""Statistics on the turn-value benchmark (docs/turn_value_prereg_20260925.md
"Validation" and "Measure", on branch exp/turn-value), and the readings of
step 1 (docs/selfplay_program_20261008.md, "Step 1").

The estimators are the experiment's own, ported from its
tools/turn_value_fit.py (`corrected_correlation`, `adjusted_outcomes`) and
from tools/analysis/turn_reads_by_depth.py (`selection_gains`,
`mean_and_se`); the bootstraps draw the same resamples, vectorized. Added
here: the paired versions step 1 reads, each grader against another over
the same resamples of positions.

The benchmark, per candidate turn (a row): its position, its source game
(the bootstrap's cluster), its slot (0 for the base turn, `obs8`'s own),
its playouts' outcomes [P], each playout's fight luck (HP, kills) [P, 2]
and the candidate turn's own [2]. The truth of a candidate is the mean of
its truth playouts; step 1's are playouts 9 to 28 (TRUTH_FROM), luck
adjusted by the verdict's coefficients (`adjusted_outcomes`), with the raw
outcomes beside.
"""
from __future__ import annotations

import functools
import math
import warnings
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

TRUTH_FROM = 8                    # step 1's truth: the last 20 of 28 playouts
CORRELATION_RESAMPLES = 1000      # tools/turn_value_fit.BOOTSTRAP_RESAMPLES
GAIN_RESAMPLES = 4000             # tools/analysis/turn_reads_by_depth.GAIN_RESAMPLES
BAR_SE = 2.0                      # "by 2 paired standard errors" (Data-limited)
PASS_BAR_SE = 2.0                 # Pass: both differences with the HP margin, in paired SE
READS = ("read0", "pre")          # right after the end_turn, and before it


# ---------------------------------------------------------------------
# Truth
# ---------------------------------------------------------------------

def adjusted_outcomes(outcomes: np.ndarray, luck: np.ndarray, turn_luck: np.ndarray,
                      beta: Sequence[float]) -> np.ndarray:
    """[N, P]: each playout's outcome minus beta . (its HP and kill luck,
    its candidate turn's HP and kill luck); a missing luck term counts 0
    (the verdict's `adjusted_outcomes`)."""
    turn = np.broadcast_to(np.asarray(turn_luck, dtype=float)[:, None, :], np.shape(luck))
    columns = np.concatenate([np.asarray(luck, dtype=float), turn], axis=2)
    return np.asarray(outcomes, dtype=float) - np.nan_to_num(columns * np.asarray(beta, dtype=float)).sum(axis=2)


def truth_mean(truth: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(truth, axis=1)


# ---------------------------------------------------------------------
# The corrected within-position correlation
# ---------------------------------------------------------------------

def _segments(positions: np.ndarray, rows: np.ndarray):
    """`rows` grouped by position in row order: (the rows sorted by their
    position, stably; each segment's start; each segment's length)."""
    rows = rows[np.argsort(np.asarray(positions)[rows], kind="stable")]
    keys = np.asarray(positions)[rows]
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]]) if len(rows) else np.zeros(0, dtype=int)
    lengths = np.diff(np.r_[starts, len(rows)])
    return rows, starts, lengths


def position_sums(grade: np.ndarray, truth: np.ndarray, positions: np.ndarray,
                  clusters: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per position with two or more usable candidates (a finite grade, two
    or more truth playouts): its cluster, and the sums over its candidates'
    deviations from the position's means of grade x truth, grade^2 and
    truth^2, with the truth's expected noise in the last."""
    grade, truth = np.asarray(grade, dtype=float), np.asarray(truth, dtype=float)
    n = np.isfinite(truth).sum(axis=1)
    rows, starts, lengths = _segments(positions, np.nonzero(np.isfinite(grade) & (n >= 2))[0])
    keep = lengths >= 2
    if not keep.any():
        return np.asarray([]), np.zeros((0, 4))
    g = grade[rows]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        y = np.nanmean(truth[rows], axis=1)
        noise = np.nanvar(truth[rows], axis=1, ddof=1) / n[rows]
    count = lengths.astype(float)
    sg, sy = np.add.reduceat(g, starts), np.add.reduceat(y, starts)
    sums = np.stack([np.add.reduceat(g * y, starts) - sg * sy / count,
                     np.add.reduceat(g * g, starts) - sg * sg / count,
                     np.add.reduceat(y * y, starts) - sy * sy / count,
                     (1 - 1 / count) * np.add.reduceat(noise, starts)], axis=1)
    return np.asarray(clusters)[rows[starts]][keep], sums[keep]


def ratios(s: np.ndarray) -> np.ndarray:
    """(observed correlation, reliability, corrected correlation) of summed
    position sums [..., 4], NaN where undefined."""
    s = np.asarray(s, dtype=float)
    sgy, sgg, syy, noise = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = (sgg > 0) & (syy > 0)
        reliability = np.where(ok, 1.0 - noise / np.where(syy > 0, syy, 1.0), np.nan)
        observed = np.where(ok, sgy / np.sqrt(np.where(ok, sgg * syy, 1.0)), np.nan)
        corrected = np.where(ok & (reliability > 0), observed / np.sqrt(np.where(reliability > 0, reliability, 1.0)),
                             np.nan)
    return np.stack([observed, reliability, corrected], axis=-1)


def percentile_se(draws: np.ndarray) -> float:
    """Half the central 68% interval of the bootstrap draws (robust to the
    long tail of resamples whose reliability falls near zero)."""
    finite = np.asarray(draws)[np.isfinite(draws)]
    if len(finite) < 10:
        return math.nan
    low, high = np.percentile(finite, [15.87, 84.13])
    return float(high - low) / 2.0


@functools.lru_cache(maxsize=16)
def resample_index(n: int, resamples: int, seed: int) -> np.ndarray:
    """[resamples, n]: the bootstraps' draws, the same as `resamples`
    successive `rng.integers(0, n, n)` calls (read-only: shared by every
    call with the same arguments)."""
    idx = np.random.default_rng(seed).integers(0, n, (resamples, n))
    idx.setflags(write=False)
    return idx


def cluster_sums(grade, truth, positions, clusters) -> Tuple[np.ndarray, np.ndarray]:
    """(cluster names, per-cluster summed position sums [C, 4])."""
    owners, sums = position_sums(np.asarray(grade, dtype=float), np.asarray(truth, dtype=float),
                                 np.asarray(positions), np.asarray(clusters))
    if len(sums) == 0:
        return np.asarray([]), np.zeros((0, 4))
    names, cluster_of = np.unique(owners, return_inverse=True)
    per_cluster = np.zeros((len(names), 4))
    np.add.at(per_cluster, cluster_of, sums)
    return names, per_cluster


def corrected_correlation(grade, truth, positions, clusters, seed: int = 0,
                          resamples: int = CORRELATION_RESAMPLES) -> Dict:
    """The verdict's measure, with bootstrap standard errors over clusters."""
    names, per_cluster = cluster_sums(grade, truth, positions, clusters)
    if len(names) == 0:
        return {"positions": 0, "clusters": 0, "observed": math.nan, "reliability": math.nan,
                "corrected": math.nan, "observed_se": math.nan, "corrected_se": math.nan}
    observed, reliability, corrected = ratios(per_cluster.sum(axis=0))
    idx = resample_index(len(names), resamples, seed)
    boots = ratios(per_cluster[idx].sum(axis=1))
    n_positions = len(position_sums(np.asarray(grade, dtype=float), np.asarray(truth, dtype=float),
                                    np.asarray(positions), np.asarray(clusters))[1])
    return {"positions": n_positions, "clusters": len(names), "observed": float(observed),
            "reliability": float(reliability), "corrected": float(corrected),
            "observed_se": percentile_se(boots[:, 0]), "corrected_se": percentile_se(boots[:, 2])}


def paired_correlations(grades: Mapping[str, np.ndarray], truth: np.ndarray, positions, clusters,
                        pairs: Sequence[Tuple[str, str]], seed: int = 0,
                        resamples: int = CORRELATION_RESAMPLES) -> Dict:
    """Each grader's corrected correlation and each pair's difference
    (first minus second), over the rows where every grader is finite and
    over the same resamples of clusters, so a difference's standard error
    sees what the two graders share."""
    common = np.all([np.isfinite(np.asarray(g, dtype=float)) for g in grades.values()], axis=0)
    names_ref, sums = None, {}
    for name, g in grades.items():
        masked = np.where(common, np.asarray(g, dtype=float), np.nan)
        names, per_cluster = cluster_sums(masked, truth, positions, clusters)
        if names_ref is not None and not np.array_equal(names, names_ref):
            raise ValueError("graders cover different clusters")
        names_ref, sums[name] = names, per_cluster
    idx = resample_index(len(names_ref), resamples, seed)
    point = {k: ratios(v.sum(axis=0)) for k, v in sums.items()}
    boots = {k: ratios(v[idx].sum(axis=1)) for k, v in sums.items()}
    out = {"clusters": int(len(names_ref)), "rows": int(common.sum()), "graders": {}, "differences": {}}
    for k in grades:
        out["graders"][k] = {"corrected": float(point[k][2]), "corrected_se": percentile_se(boots[k][:, 2]),
                             "reliability": float(point[k][1])}
    for a, b in pairs:
        diff = boots[a][:, 2] - boots[b][:, 2]
        out["differences"][f"{a}-{b}"] = {"mean": float(point[a][2] - point[b][2]), "se": percentile_se(diff)}
    return out


# ---------------------------------------------------------------------
# Selection gains
# ---------------------------------------------------------------------

def selection_gains(grade: np.ndarray, truth: np.ndarray, positions: np.ndarray,
                    slots: np.ndarray) -> Dict[int, float]:
    """Per position: the truth of the candidate `grade` ranks first (the
    first of a tie, in row order) minus the base turn's (slot 0)."""
    grade, truth = np.asarray(grade, dtype=float), np.asarray(truth, dtype=float)
    rows, starts, lengths = _segments(positions, np.arange(len(grade)))
    if len(rows) == 0:
        return {}
    g = np.where(np.isfinite(grade[rows]), grade[rows], -np.inf)
    top = np.maximum.reduceat(g, starts)
    order = np.arange(len(rows))
    never = len(rows)
    best = np.minimum.reduceat(np.where(g == np.repeat(top, lengths), order, never), starts)
    base = np.minimum.reduceat(np.where(np.asarray(slots)[rows] == 0, order, never), starts)
    ok = (lengths >= 2) & (base < never) & np.isfinite(top)
    keys = np.asarray(positions)[rows[starts]]
    gain = truth[rows[np.where(ok, best, 0)]] - truth[rows[np.where(ok, base, 0)]]
    return {int(k): float(v) for k, v, good in zip(keys, gain, ok) if good}


def mean_and_se(values: np.ndarray, seed: int = 3, resamples: int = GAIN_RESAMPLES) -> Tuple[float, float]:
    """The mean and its bootstrap standard deviation over the values."""
    values = np.asarray(values, dtype=float)
    idx = resample_index(len(values), resamples, seed)
    return float(values.mean()), float(np.std(values[idx].mean(axis=1)))


def paired_gains(grades: Mapping[str, np.ndarray], truth_means: np.ndarray, positions, slots,
                 pairs: Sequence[Tuple[str, str]], seed: int = 3, resamples: int = GAIN_RESAMPLES) -> Dict:
    """Each grader's mean selection gain over the base turn and each pair's
    difference, over the positions every grader ranks."""
    gains = {k: selection_gains(np.asarray(g, dtype=float), truth_means, np.asarray(positions), np.asarray(slots))
             for k, g in grades.items()}
    common = sorted(set.intersection(*(set(g) for g in gains.values())))
    out = {"positions": len(common), "graders": {}, "differences": {}}
    for k, g in gains.items():
        m, se = mean_and_se(np.array([g[p] for p in common]), seed, resamples)
        out["graders"][k] = {"mean": m, "se": se}
    for a, b in pairs:
        m, se = mean_and_se(np.array([gains[a][p] - gains[b][p] for p in common]), seed, resamples)
        out["differences"][f"{a}-{b}"] = {"mean": m, "se": se}
    return out


# ---------------------------------------------------------------------
# The readings of step 1
# ---------------------------------------------------------------------

def beats(diff: Optional[Dict], bar: float = BAR_SE) -> bool:
    """A paired difference above `bar` of its standard errors."""
    return bool(diff) and math.isfinite(diff["mean"]) and math.isfinite(diff["se"]) \
        and diff["mean"] > bar * diff["se"]


def z_score(diff: Optional[Dict]) -> float:
    """A paired difference in its standard errors (+inf for a positive
    difference with none, -inf where undefined): `beats(diff, bar)` is
    `z_score(diff) > bar` for every bar above 0."""
    if not diff or not (math.isfinite(diff["mean"]) and math.isfinite(diff["se"])):
        return -math.inf
    if diff["se"] > 0:
        return diff["mean"] / diff["se"]
    return math.inf if diff["mean"] > 0 else -math.inf


def pass_candidates(by_read: Mapping[str, Dict], critics: Sequence[str], margin: str = "hp_margin"
                    ) -> List[Dict]:
    """Every (critic, read) with its selection-gain and corrected-correlation
    differences with the HP margin, and the smaller of their z-scores."""
    out = []
    for r in by_read:
        for c in critics:
            gain = by_read[r]["gains"]["differences"].get(f"{c}-{margin}")
            corr = by_read[r]["correlations"]["differences"].get(f"{c}-{margin}")
            out.append({"critic": c, "read": r, "gain": gain, "correlation": corr,
                        "z": min(z_score(gain), z_score(corr))})
    return out


def pass_score(by_read: Mapping[str, Dict], critics: Sequence[str], margin: str = "hp_margin") -> float:
    """The largest bar Pass clears: the reading is Pass at a bar below it."""
    return max((p["z"] for p in pass_candidates(by_read, critics, margin)), default=-math.inf)


def readings(by_read: Mapping[str, Dict], critics: Sequence[str], margin: str = "hp_margin",
             large: str = "T100", small: str = "T25", pass_bar: float = PASS_BAR_SE) -> Dict:
    """The pre-registered reading, applied mechanically to the luck-adjusted
    statistics of each read (`by_read[read]` holds "gains" and
    "correlations" as `paired_gains` and `paired_correlations` give them,
    each critic paired with `margin`, and `large` with `small`).

    Pass: for some critic at some read, both its selection-gain difference
    and its corrected-correlation difference with the HP margin exceed
    `pass_bar` paired standard errors. Data-limited: otherwise, when `large`
    beats `small` in correlation by 2 paired standard errors at either
    read. Kill: otherwise."""
    candidates = pass_candidates(by_read, critics, margin)
    passing = [p for p in candidates if beats(p["gain"], pass_bar) and beats(p["correlation"], pass_bar)]
    if passing:
        why = "; ".join(f"{p['critic']} at {p['read']}: over the HP margin, selection gain "
                        f"{p['gain']['mean']:+.3f} +- {p['gain']['se']:.3f}, correlation "
                        f"{p['correlation']['mean']:+.3f} +- {p['correlation']['se']:.3f}" for p in passing)
        return {"reading": "Pass", "why": why, "passing": passing, "pass_bar": pass_bar}
    size = [{"read": r, **by_read[r]["correlations"]["differences"][f"{large}-{small}"]}
            for r in by_read if f"{large}-{small}" in by_read[r]["correlations"]["differences"]]
    growing = [s for s in size if beats(s)]
    if growing:
        why = "; ".join(f"{large} over {small} in correlation at {s['read']}: {s['mean']:+.3f} +- {s['se']:.3f}"
                        for s in growing)
        return {"reading": "Data-limited", "why": "no critic passes; " + why, "size": size, "pass_bar": pass_bar}
    why = (f"no critic beats the HP margin by {pass_bar:g} paired SE in both selection gain and correlation")
    best = max(candidates, key=lambda p: p["z"], default=None)
    if best is not None and best["gain"] and best["correlation"]:
        why += (f" (closest: {best['critic']} at {best['read']}, gain {best['gain']['mean']:+.3f} +- "
                f"{best['gain']['se']:.3f}, correlation {best['correlation']['mean']:+.3f} +- "
                f"{best['correlation']['se']:.3f})")
    why += f", and {large} does not beat {small} in correlation by 2 paired SE"
    if size:
        why += " (" + ", ".join(f"{s['read']} {s['mean']:+.3f} +- {s['se']:.3f}" for s in size) + ")"
    return {"reading": "Kill", "why": why, "size": size, "pass_bar": pass_bar}
