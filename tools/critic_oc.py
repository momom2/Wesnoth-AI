#!/usr/bin/env python3
"""How often each reading of step 1 fires under assumed true effects
(docs/selfplay_program_20261008.md, "Step 1", "Operating characteristics").

    python tools/critic_oc.py [--sims 80] [--out training/metrics/critic_step1_20261008/oc.json]

A synthetic benchmark with the real one's shape (its positions, their
candidate counts and slots: training/metrics/turn_value_20260925/bench_truth.npz)
and its noise, calibrated on its records:
  - each candidate's true value v = s w_p z (z standard normal, independent
    within a position), plus the base turn's advantage d w_p / mean(w) for
    slot 0: s^2 is the within-position variance the luck-adjusted truth of
    playouts 9 to 28 implies (its spread of candidate means less their
    playout noise), d the base turn's mean lead over the alternatives in
    that truth, and w_p a position's spread, lognormal with SPREAD_SIGMA,
    scaled to a mean square of 1;
  - 20 truth playouts per candidate: v plus noise whose variance is a
    candidate's empirical playout variance, resampled;
  - each grader g = w_p (r v / (s w_p) + sqrt(1 - r^2) e): r is the grader's
    true within-position correlation (the measure the benchmark estimates),
    e its error, correlated across graders as calibrated below.
SPREAD_SIGMA makes a few positions carry most of the within-position
variance, as the records' do: without it the HP margin's simulated
corrected correlation has a standard error of 0.047 where the record reads
0.098; at 0.9, 0.094 (40 draws), with a selection-gain standard error of
0.020 against the record's 0.017.
Error correlations, from the records where they exist: the HP margin and a
learned head (`obs8`'s, `value_reference`) from their within-position
correlation; a head's two reads (before and after the end_turn) from
`obs8`'s `value_reference` and `value_post`; the HP margin's two reads
0.9 (they differ by one turn start's healing); two critics, which no record
measures, KAPPA_CRITICS (0.5; 0.2 and 0.8 reported as a sensitivity).

Each simulated benchmark is read by the readout's own statistics
(wesnoth_ai/turn_bench_stats: paired gains and correlations with their
bootstrap standard errors, at both reads) and its own readings function.
The bootstraps run at a quarter of the readout's resamples, which adds
noise to each standard error, not bias.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wesnoth_ai import turn_bench_stats as S  # noqa: E402

CRITICS = ("T25", "T50", "T100", "O100", "TH", "Tsmall")
MARGIN = "hp_margin"
TRUTH_PLAYOUTS = 20
SPREAD_SIGMA = 0.9
KAPPA_CRITICS = 0.5
KAPPA_MARGIN_READS = 0.9
# The scenarios: each critic's true correlation; the HP margin's is the
# measured one (`calibrate`).
SCENARIOS = {
    "null: every critic equals the HP margin": {c: 0.0 for c in CRITICS},
    "+0.05 for every critic, no size effect": {c: 0.05 for c in CRITICS},
    "the lead's predictions (T100 +0.10, T25 +0.05)": {"T25": 0.05, "T50": 0.075, "T100": 0.10, "O100": 0.04,
                                                      "TH": 0.085, "Tsmall": 0.07},
    "+0.20 for T100, T25 +0.12": {"T25": 0.12, "T50": 0.16, "T100": 0.20, "O100": 0.15, "TH": 0.18,
                                  "Tsmall": 0.17},
    "below the margin, size +0.10 (T100 -0.05, T25 -0.15)": {"T25": -0.15, "T50": -0.10, "T100": -0.05,
                                                             "O100": -0.10, "TH": -0.08, "Tsmall": -0.10},
    "below the margin by 0.13, no size effect": {c: -0.13 for c in CRITICS},
}


def within(x: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """x minus its position's mean (NaN rows left out of the mean)."""
    out = np.full(len(x), np.nan)
    for p in np.unique(positions):
        rows = np.nonzero((positions == p) & np.isfinite(x))[0]
        if len(rows):
            out[rows] = x[rows] - x[rows].mean()
    return out


def within_corr(a: np.ndarray, b: np.ndarray, positions: np.ndarray) -> float:
    both = np.isfinite(a) & np.isfinite(b)
    x = within(np.where(both, a, np.nan), positions)
    y = within(np.where(both, b, np.nan), positions)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def error_corr(observed: float, r1: float, r2: float) -> float:
    """The error correlation two graders of true correlations r1 and r2 need
    for their own within-position correlation to be `observed`."""
    return (observed - r1 * r2) / math.sqrt((1 - r1 * r1) * (1 - r2 * r2))


def calibrate(arrays: Dict) -> Dict:
    """The noise model's figures from the benchmark's records."""
    positions = np.unique(arrays["index"], return_inverse=True)[1]
    truth = S.adjusted_outcomes(arrays["outcomes"], arrays["luck"], arrays["turn_luck"],
                                arrays["beta"])[:, S.TRUTH_FROM:]
    _, sums = S.cluster_sums(arrays["hp_margin_post"], truth, positions, arrays["cluster"])
    total = sums.sum(axis=0)
    counts = np.bincount(positions)
    var_v = (total[2] - total[3]) / float((counts[counts >= 2] - 1).sum())
    r = {k: S.corrected_correlation(arrays[k], truth, positions, arrays["cluster"])["corrected"]
         for k in ("hp_margin_post", "value_reference", "value_post")}
    head_margin = error_corr(within_corr(arrays["value_reference"], arrays["hp_margin_post"], positions),
                             r["value_reference"], r["hp_margin_post"])
    head_reads = error_corr(within_corr(arrays["value_reference"], arrays["value_post"], positions),
                            r["value_reference"], r["value_post"])
    noise = np.nanvar(truth, axis=1, ddof=1)
    means = S.truth_mean(truth)
    leads = []
    for p in np.unique(positions):
        rows = np.nonzero(positions == p)[0]
        base, others = rows[arrays["slot"][rows] == 0], rows[arrays["slot"][rows] != 0]
        if len(base) and len(others):
            leads.append(means[base[0]] - means[others].mean())
    return {"var_v": float(var_v), "base_advantage": float(np.mean(leads)),
            "noise_var": noise[np.isfinite(noise)].tolist(),
            "r_margin": float(r["hp_margin_post"]), "r_head": float(r["value_reference"]),
            "kappa_head_margin": float(head_margin), "kappa_head_reads": float(head_reads)}


def grader_names() -> List[Tuple[str, str]]:
    """(grader, read) of every simulated grade."""
    return [(MARGIN, read) for read in S.READS] + [(c, read) for c in CRITICS for read in S.READS]


def error_covariance(cal: Dict, kappa_critics: float) -> np.ndarray:
    names = grader_names()
    n = len(names)
    cov = np.eye(n)
    for i, (gi, ri) in enumerate(names):
        for j, (gj, rj) in enumerate(names):
            if i == j:
                continue
            if gi == gj:
                k = KAPPA_MARGIN_READS if gi == MARGIN else cal["kappa_head_reads"]
            elif MARGIN in (gi, gj):
                k = cal["kappa_head_margin"]
            else:
                k = kappa_critics * (1.0 if ri == rj else cal["kappa_head_reads"])
            cov[i, j] = k
    if np.linalg.eigvalsh(cov).min() <= 0:
        raise ValueError("the assumed error correlations are not a covariance")
    return cov


def simulate(arrays: Dict, cal: Dict, effects: Dict[str, float], kappa_critics: float, rng: np.random.Generator,
             correlation_resamples: int, gain_resamples: int) -> Dict:
    """One synthetic benchmark read by the readout: its statistics at both
    reads and its reading."""
    positions = np.unique(arrays["index"], return_inverse=True)[1]
    n = len(positions)
    sd_v = math.sqrt(cal["var_v"])
    spread = np.exp(SPREAD_SIGMA * rng.normal(size=positions.max() + 1))
    spread /= math.sqrt(float(np.mean(spread ** 2)))
    w = spread[positions]
    v = sd_v * w * rng.normal(size=n) + (arrays["slot"] == 0) * cal["base_advantage"] * w / spread.mean()
    standardized = v / (sd_v * w)
    noise_sd = np.sqrt(rng.choice(np.asarray(cal["noise_var"]), n))
    truth = v[:, None] + noise_sd[:, None] * rng.normal(size=(n, TRUTH_PLAYOUTS))
    names = grader_names()
    errors = rng.multivariate_normal(np.zeros(len(names)), error_covariance(cal, kappa_critics), size=n)
    by_read: Dict[str, Dict[str, np.ndarray]] = {read: {} for read in S.READS}
    for k, (grader, read) in enumerate(names):
        r = cal["r_margin"] + (0.0 if grader == MARGIN else effects[grader])
        by_read[read][grader] = w * (r * standardized + math.sqrt(1 - r * r) * errors[:, k])
    pairs = [(c, MARGIN) for c in CRITICS] + [("T100", "T25")]
    stats = {read: {"correlations": S.paired_correlations(g, truth, positions, arrays["cluster"], pairs,
                                                          resamples=correlation_resamples),
                    "gains": S.paired_gains(g, truth.mean(axis=1), positions, arrays["slot"], pairs,
                                            resamples=gain_resamples)}
             for read, g in by_read.items()}
    return {"reading": S.readings(stats, CRITICS)["reading"],
            "t100_gain": stats["read0"]["gains"]["differences"]["T100-hp_margin"],
            "t100_corr": stats["read0"]["correlations"]["differences"]["T100-hp_margin"],
            "size_corr": stats["read0"]["correlations"]["differences"]["T100-T25"],
            "margin_corr_se": stats["read0"]["correlations"]["graders"][MARGIN]["corrected_se"],
            "margin_gain_se": stats["read0"]["gains"]["graders"][MARGIN]["se"]}


def run(arrays: Dict, sims: int, seed: int, kappas: Sequence[float], correlation_resamples: int,
        gain_resamples: int) -> Dict:
    cal = calibrate(arrays)
    rng = np.random.default_rng(seed)
    rows = []
    for name, effects in SCENARIOS.items():
        for kappa in (kappas if name.startswith("null") else kappas[:1]):
            results = [simulate(arrays, cal, effects, kappa, rng, correlation_resamples, gain_resamples)
                       for _ in range(sims)]
            readings = [r["reading"] for r in results]
            rows.append({"scenario": name, "kappa_critics": kappa, "effects": effects, "sims": sims,
                         **{f"p_{k}": readings.count(k) / sims for k in ("Pass", "Data-limited", "Kill")},
                         "t100_gain_minus_margin": float(np.mean([r["t100_gain"]["mean"] for r in results])),
                         "t100_gain_minus_margin_se": float(np.mean([r["t100_gain"]["se"] for r in results])),
                         "t100_corr_minus_margin_se": float(np.nanmean([r["t100_corr"]["se"] for r in results])),
                         "t100_minus_t25_corr_se": float(np.nanmean([r["size_corr"]["se"] for r in results])),
                         "margin_corr_se": float(np.nanmean([r["margin_corr_se"] for r in results])),
                         "margin_gain_se": float(np.mean([r["margin_gain_se"] for r in results]))})
    return {"calibration": {k: v for k, v in cal.items() if k != "noise_var"}, "rows": rows, "seed": seed,
            "resamples": {"correlation": correlation_resamples, "gain": gain_resamples}}


def markdown(out: Dict) -> str:
    cal = out["calibration"]
    lines = [f"calibration: within-position sd of true values {math.sqrt(cal['var_v']):.3f}, base lead "
             f"{cal['base_advantage']:.3f}, spread sigma {SPREAD_SIGMA}, HP margin r {cal['r_margin']:.3f}, "
             f"obs8's head r {cal['r_head']:.3f}, error correlations head-margin {cal['kappa_head_margin']:.2f}, "
             f"head's two reads {cal['kappa_head_reads']:.2f}", "",
             "| scenario (critics' r minus the margin's) | kappa | Pass | Data-limited | Kill | "
             "T100 gain - margin (SE) | SE: T100-T25 r, margin r, margin gain |", "|---|---|---|---|---|---|---|"]
    for r in out["rows"]:
        lines.append(f"| {r['scenario']} | {r['kappa_critics']:.1f} | {r['p_Pass']:.2f} | {r['p_Data-limited']:.2f} | "
                     f"{r['p_Kill']:.2f} | {r['t100_gain_minus_margin']:+.3f} ({r['t100_gain_minus_margin_se']:.3f}) | "
                     f"{r['t100_minus_t25_corr_se']:.3f}, {r['margin_corr_se']:.3f}, {r['margin_gain_se']:.3f} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arrays", type=Path, default=None, help="the benchmark's truth (tools/critic_bench.py extract)")
    ap.add_argument("--sims", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--kappas", type=float, nargs="+", default=[KAPPA_CRITICS, 0.2, 0.8])
    ap.add_argument("--correlation-resamples", type=int, default=S.CORRELATION_RESAMPLES // 4)
    ap.add_argument("--gain-resamples", type=int, default=S.GAIN_RESAMPLES // 4)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    from tools.critic_bench import BENCH_ARRAYS, load_arrays
    t0 = time.time()
    out = run(load_arrays(args.arrays or BENCH_ARRAYS), args.sims, args.seed, args.kappas,
              args.correlation_resamples, args.gain_resamples)
    out["seconds"] = round(time.time() - t0, 1)
    print(markdown(out))
    print(f"{out['seconds']} s")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
