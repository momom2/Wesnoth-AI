#!/usr/bin/env python3
"""The readout of the rollout re-grade (docs/rollout_regrade_prereg_20261012.md,
"Statistics" and "Readings"; the rollouts are tools/rollout_regrade.py's).

    python tools/rollout_readout.py --rollouts OUT/rollouts.jsonl \\
        --static training/metrics/critic_step1_20261009/rebuild.jsonl --out OUT

Graders, per benchmark candidate: the mean over its rollouts of the mover's
HP margin and of the value head at read 2 and read 4 (margin_r2, margin_r4,
value_r2, value_r4), and the static HP margin right after the turn
(hp_margin, step 1's rebuild: its `hp_margin_post`). A candidate is graded
only when every one of its rollouts was played.

Statistics: step 1's (tools/critic_bench.statistics over
wesnoth_ai/turn_bench_stats.py) on the luck-adjusted truth (primary) and the
raw truth: per grader the corrected within-position correlation and the
selection gain over the base turn, each paired against the static HP margin
over the same resamples. Beside them, per grader: the share of the positions
it ranks where its pick is the "continue" edit, and its selection gain over
the positions where its pick is not (bootstrapped over those positions as
step 1's gains are); and the share of rollouts that had ended by each read.

The reading, on the luck-adjusted truth, of the primary grader (margin_r2):
Pass when its selection gain over the base turn is above 0 by two paired
standard errors; Pass, tempo only when it passes and its gain over the
positions where it does not pick the continue edit is not above 0 by two
standard errors; Fail otherwise. readout.json and readout.md; the reading is
printed with its reason. It is refused (exit 4, READING_REFUSED) when a
benchmark candidate has no complete record or its state is not step 1's,
when fewer than MIN_POSITIONS positions enter a statistic, or when no record
names a continue edit.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools import critic_bench as cb  # noqa: E402
from tools import rollout_regrade as rr  # noqa: E402
from wesnoth_ai import turn_bench_stats as S  # noqa: E402

STATIC = cb.MARGIN                    # "hp_margin", step 1's static grader
PRIMARY = "margin_r2"
GRADERS = ("margin_r2", "margin_r4", "value_r2", "value_r4")
READ = "rollouts"                     # the one "read" of critic_bench.statistics: every grader sits after the turn
EXIT_REFUSED = 4


# ---------------------------------------------------------------------
# Grades
# ---------------------------------------------------------------------

def row_index(arrays: Dict) -> Dict[Tuple[int, int], int]:
    return {(int(i), int(s)): k for k, (i, s) in enumerate(zip(arrays["index"], arrays["slot"]))}


def static_grade(arrays: Dict, step1: Mapping[Tuple[int, int], Dict]) -> np.ndarray:
    """Step 1's HP margin right after the turn, per benchmark row: NaN where
    step 1 did not rebuild the candidate, and over a position whose base
    turn it did not rebuild (critic_bench.grades_at's rule)."""
    out = np.full(len(arrays["index"]), np.nan)
    for (index, slot), k in row_index(arrays).items():
        row, base = step1.get((index, slot)), step1.get((index, 0))
        base_ok = base is None or "skipped" in base or (base.get("digest_ok") and "error" not in base)
        if row is not None and row.get("digest_ok") and "error" not in row and base_ok:
            out[k] = row["hp_margin_post"]
    return out


def rollout_grades(arrays: Dict, records: Mapping[Tuple[int, int], Dict], cfg: rr.RolloutConfig
                   ) -> Dict[str, np.ndarray]:
    """Per grader, the mean over a candidate's rollouts, per benchmark row
    (NaN where the candidate's record is missing or not complete)."""
    out = {g: np.full(len(arrays["index"]), np.nan) for g in GRADERS}
    for key, k in row_index(arrays).items():
        rec = records.get(key)
        if rec is None or "skipped" in rec or not rr.is_complete(rec, cfg):
            continue
        for g in GRADERS:
            field, read = g.split("_r")
            out[g][k] = float(np.mean([ro["reads"][read][field] for ro in rec["rollouts"]]))
    return out


def continue_rows(arrays: Dict, records: Mapping[Tuple[int, int], Dict]) -> np.ndarray:
    """Per benchmark row: whether the candidate is the continue edit."""
    out = np.zeros(len(arrays["index"]), dtype=bool)
    for key, k in row_index(arrays).items():
        out[k] = (records.get(key) or {}).get("proposer") == "continue"
    return out


def ended_shares(records: Mapping[Tuple[int, int], Dict], cfg: rr.RolloutConfig) -> Dict[str, float]:
    """Per read, the share of the played rollouts whose game had ended."""
    played = [ro for rec in records.values() for ro in rec.get("rollouts") or () if "error" not in ro]
    return {str(k): (float(np.mean([ro["reads"][str(k)]["ended"] for ro in played])) if played else None)
            for k in cfg.reads}


# ---------------------------------------------------------------------
# The continue split
# ---------------------------------------------------------------------

def tempo_split(grades: Mapping[str, np.ndarray], truth_means: np.ndarray, positions: np.ndarray,
                slots: np.ndarray, is_continue: np.ndarray, seed: int = 3,
                resamples: int = S.GAIN_RESAMPLES) -> Dict:
    """Per grader, over the positions every grader ranks (those of
    `paired_gains`): how many it ranks, how many of its picks are the
    continue edit and their share, and its selection gain over the base turn
    over the positions where its pick is not (mean and bootstrap SE)."""
    picks = {g: S.selections(np.asarray(v, dtype=float), positions, slots) for g, v in grades.items()}
    common = sorted(set.intersection(*(set(p) for p in picks.values()))) if picks else []
    out = {}
    for g, by_position in picks.items():
        chosen = [by_position[p] for p in common]
        kept = [truth_means[best] - truth_means[base] for best, base in chosen if not is_continue[best]]
        n_continue = len(chosen) - len(kept)
        if kept:
            m, se = S.mean_and_se(np.asarray(kept), seed, resamples)
        else:
            m, se = float("nan"), float("nan")
        out[g] = {"positions": len(chosen), "continue_picks": n_continue,
                  "continue_share": n_continue / len(chosen) if chosen else float("nan"),
                  "gain_without_continue": {"mean": m, "se": se, "positions": len(kept)}}
    return out


def truths(arrays: Dict) -> Dict[str, np.ndarray]:
    """Per basis, each candidate's truth playouts (step 1's)."""
    return {"adjusted": S.adjusted_outcomes(arrays["outcomes"], arrays["luck"], arrays["turn_luck"],
                                            arrays["beta"])[:, S.TRUTH_FROM:],
            "raw": arrays["outcomes"][:, S.TRUTH_FROM:]}


# ---------------------------------------------------------------------
# Statistics and the reading
# ---------------------------------------------------------------------

def readout_statistics(arrays: Dict, grades: Mapping[str, np.ndarray], is_continue: np.ndarray,
                       correlation_resamples: int = S.CORRELATION_RESAMPLES,
                       gain_resamples: int = S.GAIN_RESAMPLES) -> Dict:
    """Per basis: step 1's paired statistics of every grader against the
    static margin ("correlations", "gains") and the continue split
    ("tempo")."""
    graders = [g for g in grades if g != STATIC]
    stats = cb.statistics(arrays, {READ: dict(grades)}, graders, correlation_resamples, gain_resamples)
    positions = np.unique(arrays["index"], return_inverse=True)[1]
    common = np.all([np.isfinite(np.asarray(g, dtype=float)) for g in grades.values()], axis=0)
    masked = {g: np.where(common, np.asarray(v, dtype=float), np.nan) for g, v in grades.items()}
    out = {}
    for basis, truth in truths(arrays).items():
        out[basis] = dict(stats[basis][READ])
        out[basis]["tempo"] = tempo_split(masked, S.truth_mean(truth), positions, arrays["slot"], is_continue,
                                          resamples=gain_resamples)
    return out


def reading(adjusted: Dict, primary: str = PRIMARY, bar: float = S.PASS_BAR_SE) -> Dict:
    """The pre-registered reading of the primary grader on the
    luck-adjusted statistics (module docstring), with its reason."""
    gain = adjusted["gains"]["graders"][primary]
    tempo = adjusted["tempo"][primary]
    without = tempo["gain_without_continue"]
    said = (f"{primary}: selection gain {gain['mean']:+.3f} +- {gain['se']:.3f} over {adjusted['gains']['positions']} "
            f"positions; {tempo['continue_picks']} of {tempo['positions']} picks the continue edit, gain without "
            f"them {without['mean']:+.3f} +- {without['se']:.3f} over {without['positions']} positions")
    if not S.beats(gain, bar):
        return {"reading": "Fail", "why": f"the gain is not above 0 by {bar:g} SE; {said}", "pass_bar": bar}
    if not S.beats(without, bar):
        return {"reading": "Pass, tempo only",
                "why": f"the gain passes and is not above 0 by {bar:g} SE without the continue picks; {said}",
                "pass_bar": bar}
    return {"reading": "Pass", "why": f"the gain passes with and without the continue picks; {said}",
            "pass_bar": bar}


def refusal_reasons(arrays: Dict, records: Mapping[Tuple[int, int], Dict], grades: Mapping[str, np.ndarray],
                    is_continue: np.ndarray, stats: Dict) -> List[str]:
    """Why the statistics cannot be read (none when they can)."""
    out = []
    keys = list(row_index(arrays))
    missing = [k for k in keys if k not in records]
    if missing:
        out.append(f"{len(missing)} benchmark candidates have no record")
    for g, v in grades.items():
        holes = int((~np.isfinite(np.asarray(v, dtype=float))).sum())
        if holes:
            out.append(f"{g} has no grade for {holes} of {len(keys)} benchmark candidates")
    if not is_continue.any():
        out.append("no record names a continue edit")
    for basis, s in stats.items():
        n = min(s["correlations"]["clusters"], s["gains"]["positions"])
        if n < cb.MIN_POSITIONS:
            out.append(f"{s['correlations']['clusters']} positions in the correlations and "
                       f"{s['gains']['positions']} in the gains ({basis})")
    return out


def markdown(stats: Dict, verdict: Dict, counts: Dict, ended: Dict) -> str:
    lines = [f"# Rollout re-grade readout: {verdict['reading']}", "", verdict["why"], "",
             f"records: {json.dumps(counts)}", f"rollouts ended by read: {json.dumps(ended)}", ""]
    for basis in ("adjusted", "raw"):
        s = stats[basis]
        lines += [f"## {basis} truth", "",
                  "| grader | corrected r | minus HP margin | selection gain | minus HP margin "
                  "| continue picks | gain without them |",
                  "|---|---|---|---|---|---|---|"]
        for name, c in s["correlations"]["graders"].items():
            g = s["gains"]["graders"][name]
            dc = s["correlations"]["differences"].get(f"{name}-{STATIC}")
            dg = s["gains"]["differences"].get(f"{name}-{STATIC}")
            t = s["tempo"][name]
            w = t["gain_without_continue"]
            lines.append(f"| {name} | {c['corrected']:.3f} +- {c['corrected_se']:.3f} | {_fmt(dc)} | "
                         f"{g['mean']:+.3f} +- {g['se']:.3f} | {_fmt(dg)} | "
                         f"{t['continue_picks']}/{t['positions']} | {w['mean']:+.3f} +- {w['se']:.3f} |")
        lines += ["", f"positions: correlation {s['correlations']['clusters']}, gains {s['gains']['positions']}", ""]
    return "\n".join(lines)


def _fmt(d: Optional[Dict]) -> str:
    return "-" if not d else f"{d['mean']:+.3f} +- {d['se']:.3f}"


def record_counts(arrays: Dict, records: Mapping[Tuple[int, int], Dict], cfg: rr.RolloutConfig) -> Dict:
    keys = set(row_index(arrays))
    recs = [records[k] for k in keys if k in records]
    return {"benchmark_candidates": len(keys), "recorded": len(recs),
            "complete": sum(rr.is_complete(r, cfg) for r in recs),
            "failed": sum("error" in r for r in recs),
            "continue_edits": int(sum(r.get("proposer") == "continue" for r in recs))}


def run(rollouts: Path, static: Path, out: Path, arrays_path: Path = cb.BENCH_ARRAYS,
        cfg: rr.RolloutConfig = rr.RolloutConfig()) -> int:
    from wesnoth_ai import __version__
    arrays = cb.load_arrays(arrays_path)
    records = rr.last_records(rr.read_rollouts(rollouts))
    grades = {STATIC: static_grade(arrays, rr.load_step1(static)), **rollout_grades(arrays, records, cfg)}
    is_continue = continue_rows(arrays, records)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stats = readout_statistics(arrays, grades, is_continue)
    counts = record_counts(arrays, records, cfg)
    ended = ended_shares(records, cfg)
    out.mkdir(parents=True, exist_ok=True)
    refusal = refusal_reasons(arrays, records, grades, is_continue, stats)
    if refusal:
        (out / "readout_refused.json").write_text(
            json.dumps({"refused": refusal, "records": counts, "statistics": stats}, indent=1, default=float),
            encoding="utf-8")
        print(f"READING_REFUSED: {'; '.join(refusal)}")
        return EXIT_REFUSED
    verdict = reading(stats["adjusted"])
    payload = {"reading": verdict, "statistics": stats, "records": counts, "rollouts_ended_by_read": ended,
               "primary": PRIMARY, "static": STATIC, "code_version": __version__,
               "truth": {"playouts": f"{S.TRUTH_FROM + 1}-28", "basis": "luck-adjusted (verdict beta)",
                         "beta": arrays["beta"].tolist()},
               "sources": {"rollouts": str(rollouts), "static": str(static), "arrays": str(arrays_path)}}
    (out / "readout.json").write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
    (out / "readout.md").write_text(markdown(stats, verdict, counts, ended), encoding="utf-8")
    print(f"READING {verdict['reading']}: {verdict['why']}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rollouts", type=Path, required=True, help="tools/rollout_regrade.py's rollouts.jsonl")
    ap.add_argument("--static", type=Path, required=True, help="step 1's rebuild.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arrays", type=Path, default=cb.BENCH_ARRAYS, help="the benchmark's truth")
    args = ap.parse_args(argv)
    return run(args.rollouts, args.static, args.out, args.arrays)


if __name__ == "__main__":
    raise SystemExit(main())
