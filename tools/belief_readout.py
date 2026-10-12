#!/usr/bin/env python3
"""The readout of the rollout re-grade in belief worlds
(docs/belief_worlds_prereg_20261012.md, "Statistics" and "Readings"; the
rollouts are tools/rollout_regrade.py's, one run per world).

    python tools/belief_readout.py --dir RUN_OUT --out OUT

Each arm's records are read from its file in --dir
(tools/rollout_worlds.out_files): T `true`, TU `true_units`, RB
`roster_belief`, RU `roster_uniform`, LS `last_seen`, O2 `none`. The graders
per arm are the re-grade's (tools/rollout_readout.py): the mean over a
candidate's rollouts of the mover's HP margin and of the value head at reads
2 and 4.

Statistics, per truth basis (luck-adjusted primary, raw beside) and per
grader: each arm's selection gain over the base turn and the paired
differences TU-O2, T-TU, RB-RU, RU-O2 and LS-O2, over the positions every
arm ranks, every arm and difference bootstrapped over the same resamples of
positions (wesnoth_ai/turn_bench_stats.paired_gains). Beside them: each
arm's agreement with TU's pick (the primary grader's), and per arm the
placement counts summed over its rollouts, with the rollouts and positions
whose opponent leader is hidden.

The readings, on the luck-adjusted primary grader (margin_r2), a difference
"above 0" by two of its paired standard errors (turn_bench_stats.beats):
  - Unresolved ceiling: TU - O2 is not above 0; nothing else is read;
  - otherwise each that holds: Last seen suffices (LS - O2 above 0); The
    belief head knows where (RB - RU above 0); What exists, not where (RU -
    O2 above 0 and RB - RU within two standard errors of 0); None of the
    three when none of these holds.
belief_readout.json and belief_readout.md; the readings are printed with
their reasons. The readout is refused (exit 4, READING_REFUSED) when an
arm's file is missing, holds a record of another world, lacks a complete
record for a benchmark candidate, or when fewer than MIN_POSITIONS positions
enter the statistics.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tools import critic_bench as cb  # noqa: E402
from tools import rollout_readout as ro  # noqa: E402
from tools import rollout_regrade as rr  # noqa: E402
from tools.rollout_worlds import out_files, record_world  # noqa: E402
from wesnoth_ai import turn_bench_stats as S  # noqa: E402

ARMS = {"T": "true", "TU": "true_units", "RB": "roster_belief", "RU": "roster_uniform", "LS": "last_seen",
        "O2": "none"}
PAIRS = (("TU", "O2"), ("T", "TU"), ("RB", "RU"), ("RU", "O2"), ("LS", "O2"))
PRIMARY = ro.PRIMARY
GRADERS = ro.GRADERS
EXIT_REFUSED = ro.EXIT_REFUSED
PLACEMENT_COUNTS = ("hidden", "kept", "placed", "fallbacks", "occupied", "left_out", "removed", "hp_clamped")


# ---------------------------------------------------------------------
# Records and grades
# ---------------------------------------------------------------------

def arm_records(run_dir: Path) -> Dict[str, Dict[Tuple[int, int], Dict]]:
    """Per arm label, the last record of each candidate in its file (an
    absent file gives no records)."""
    return {label: rr.last_records(rr.read_rollouts(out_files(run_dir, world)[0]))
            for label, world in ARMS.items()}


def arm_grades(arrays: Dict, records: Mapping[str, Mapping], cfg: rr.RolloutConfig
               ) -> Dict[str, Dict[str, np.ndarray]]:
    """Per arm, per grader, a grade per benchmark row (NaN where the
    candidate's record is missing or not complete)."""
    return {label: ro.rollout_grades(arrays, recs, cfg) for label, recs in records.items()}


def positions_of(arrays: Dict) -> np.ndarray:
    return np.unique(arrays["index"], return_inverse=True)[1]


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------

def arm_statistics(arrays: Dict, grades: Mapping[str, Mapping[str, np.ndarray]],
                   resamples: int = S.GAIN_RESAMPLES) -> Dict:
    """Per basis, per grader: every arm's selection gain over the base turn
    and the pre-registered paired differences (module docstring)."""
    positions = positions_of(arrays)
    out: Dict = {}
    for basis, truth in ro.truths(arrays).items():
        means = S.truth_mean(truth)
        out[basis] = {g: S.paired_gains({label: by_grader[g] for label, by_grader in grades.items()}, means,
                                        positions, arrays["slot"], PAIRS, resamples=resamples)
                      for g in GRADERS}
    return out


def agreement_with(arrays: Dict, grades: Mapping[str, Mapping[str, np.ndarray]], reference: str = "TU",
                   grader: str = PRIMARY) -> Dict[str, Dict]:
    """Per arm, over the positions both it and `reference` rank: how many
    of its picks are the reference's, and their share."""
    positions, slots = positions_of(arrays), arrays["slot"]
    picks = {label: S.selections(np.asarray(g[grader], dtype=float), positions, slots)
             for label, g in grades.items()}
    ref = picks[reference]
    out = {}
    for label, by_position in picks.items():
        common = sorted(set(by_position) & set(ref))
        same = sum(by_position[p][0] == ref[p][0] for p in common)
        out[label] = {"positions": len(common), "same_pick": int(same),
                      "share": same / len(common) if common else float("nan")}
    return out


def placement_counts(records: Mapping[str, Mapping[Tuple[int, int], Dict]]) -> Dict[str, Dict]:
    """Per arm: the placement counts summed over its played rollouts, the
    rollouts and the positions whose opponent leader is hidden, and the
    rollouts whose hidden leader went to its starting position."""
    out = {}
    for label, recs in records.items():
        sums, rollouts, leader_rollouts, at_start, leader_positions = Counter(), 0, 0, 0, set()
        for (index, _slot), rec in recs.items():
            for ro_ in rec.get("rollouts") or ():
                p = ro_.get("placement")
                if "error" in ro_ or not p:
                    continue
                rollouts += 1
                sums.update({k: int(p.get(k, 0)) for k in PLACEMENT_COUNTS})
                at_start += bool(p.get("leader_at_start"))
                if p.get("leader_hidden"):
                    leader_rollouts += 1
                    leader_positions.add(index)
        out[label] = {"rollouts": rollouts, **{k: sums.get(k, 0) for k in PLACEMENT_COUNTS},
                      "leader_hidden_rollouts": leader_rollouts, "leader_hidden_positions": len(leader_positions),
                      "leader_at_start_rollouts": at_start}
    return out


# ---------------------------------------------------------------------
# The readings
# ---------------------------------------------------------------------

def within(diff: Optional[Dict], bar: float) -> bool:
    """A paired difference within `bar` of its standard errors of 0."""
    return bool(diff) and math.isfinite(diff["mean"]) and math.isfinite(diff["se"]) \
        and abs(diff["mean"]) <= bar * diff["se"]


def _said(diffs: Mapping[str, Dict], a: str, b: str) -> str:
    d = diffs[f"{a}-{b}"]
    return f"{a} - {b} {d['mean']:+.3f} +- {d['se']:.3f}"


def readings(primary: Dict, bar: float = S.PASS_BAR_SE) -> Dict:
    """The pre-registered readings of the primary grader's gains on the
    luck-adjusted truth (module docstring), each with its reason."""
    diffs = primary["differences"]
    d = {f"{a}-{b}": diffs[f"{a}-{b}"] for a, b in PAIRS}
    if not S.beats(d["TU-O2"], bar):
        return {"readings": ["Unresolved ceiling"], "pass_bar": bar,
                "why": [f"{_said(d, 'TU', 'O2')} is not above 0 by {bar:g} SE"]}
    held, why = [], [f"{_said(d, 'TU', 'O2')} is above 0 by {bar:g} SE"]
    if S.beats(d["LS-O2"], bar):
        held.append("Last seen suffices")
        why.append(f"{_said(d, 'LS', 'O2')} is above 0 by {bar:g} SE")
    if S.beats(d["RB-RU"], bar):
        held.append("The belief head knows where")
        why.append(f"{_said(d, 'RB', 'RU')} is above 0 by {bar:g} SE")
    if S.beats(d["RU-O2"], bar) and within(d["RB-RU"], bar):
        held.append("What exists, not where")
        why.append(f"{_said(d, 'RU', 'O2')} is above 0 and {_said(d, 'RB', 'RU')} within {bar:g} SE")
    if not held:
        held.append("None of the three")
        why.append(f"none holds: {_said(d, 'LS', 'O2')}, {_said(d, 'RB', 'RU')}, {_said(d, 'RU', 'O2')}")
    return {"readings": held, "why": why, "pass_bar": bar}


def refusal_reasons(arrays: Dict, records: Mapping[str, Mapping], grades: Mapping[str, Mapping[str, np.ndarray]],
                    stats: Dict, run_dir: Path) -> List[str]:
    """Why the statistics cannot be read (none when they can)."""
    out = []
    keys = list(ro.row_index(arrays))
    for label, world in ARMS.items():
        path = out_files(run_dir, world)[0]
        if not path.exists():
            out.append(f"{label}: no {path.name}")
            continue
        recs = records[label]
        foreign = Counter(record_world(r) for r in recs.values() if record_world(r) != world)
        if foreign:
            out.append(f"{label}: {sum(foreign.values())} records of another world {dict(foreign)}")
        missing = sum(k not in recs for k in keys)
        if missing:
            out.append(f"{label}: {missing} benchmark candidates have no record")
        holes = int((~np.isfinite(np.asarray(grades[label][PRIMARY], dtype=float))).sum())
        if holes:
            out.append(f"{label}: no complete record for {holes} of {len(keys)} benchmark candidates")
    for basis, by_grader in stats.items():
        n = by_grader[PRIMARY]["positions"]
        if n < cb.MIN_POSITIONS:
            out.append(f"{n} positions enter the statistics ({basis})")
    return out


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def _fmt(d: Optional[Dict]) -> str:
    return "-" if not d else f"{d['mean']:+.3f} +- {d['se']:.3f}"


def markdown(stats: Dict, verdict: Dict, agreement: Dict, placement: Dict) -> str:
    lines = [f"# Belief worlds readout: {'; '.join(verdict['readings'])}", ""]
    lines += [f"- {w}" for w in verdict["why"]] + [""]
    for basis in ("adjusted", "raw"):
        lines += [f"## {basis} truth", "",
                  "| grader | " + " | ".join(ARMS) + " | " + " | ".join(f"{a}-{b}" for a, b in PAIRS) + " |",
                  "|---" * (1 + len(ARMS) + len(PAIRS)) + "|"]
        for g in GRADERS:
            s = stats[basis][g]
            cells = [_fmt(s["graders"][label]) for label in ARMS]
            cells += [_fmt(s["differences"][f"{a}-{b}"]) for a, b in PAIRS]
            lines.append(f"| {g} | " + " | ".join(cells) + " |")
        lines += ["", f"positions: {stats[basis][PRIMARY]['positions']}", ""]
    lines += ["## Agreement with TU's pick (margin_r2)", "", "| arm | same pick | positions | share |",
              "|---|---|---|---|"]
    lines += [f"| {label} | {a['same_pick']} | {a['positions']} | {a['share']:.3f} |" for label, a in agreement.items()]
    lines += ["", "## Placements (summed over rollouts)", "",
              "| arm | rollouts | " + " | ".join(PLACEMENT_COUNTS)
              + " | leader hidden: rollouts | positions | leader at its start: rollouts |",
              "|---" * (5 + len(PLACEMENT_COUNTS)) + "|"]
    for label, p in placement.items():
        lines.append(f"| {label} | {p['rollouts']} | " + " | ".join(str(p[k]) for k in PLACEMENT_COUNTS)
                     + f" | {p['leader_hidden_rollouts']} | {p['leader_hidden_positions']}"
                     + f" | {p['leader_at_start_rollouts']} |")
    return "\n".join(lines) + "\n"


def run(run_dir: Path, out: Path, arrays_path: Path = cb.BENCH_ARRAYS,
        cfg: rr.RolloutConfig = rr.RolloutConfig(), resamples: int = S.GAIN_RESAMPLES) -> int:
    from wesnoth_ai import __version__
    arrays = cb.load_arrays(arrays_path)
    records = arm_records(run_dir)
    grades = arm_grades(arrays, records, cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stats = arm_statistics(arrays, grades, resamples)
    agreement = agreement_with(arrays, grades)
    placement = placement_counts(records)
    out.mkdir(parents=True, exist_ok=True)
    refusal = refusal_reasons(arrays, records, grades, stats, run_dir)
    if refusal:
        (out / "belief_readout_refused.json").write_text(
            json.dumps({"refused": refusal, "placement": placement, "statistics": stats}, indent=1, default=float),
            encoding="utf-8")
        print(f"READING_REFUSED: {'; '.join(refusal)}")
        return EXIT_REFUSED
    verdict = readings(stats["adjusted"][PRIMARY])
    payload = {"readings": verdict, "statistics": stats, "agreement_with_TU": agreement, "placement": placement,
               "arms": ARMS, "pairs": [f"{a}-{b}" for a, b in PAIRS], "primary": PRIMARY,
               "code_version": __version__,
               "truth": {"playouts": f"{S.TRUTH_FROM + 1}-28", "basis": "luck-adjusted (verdict beta)",
                         "beta": arrays["beta"].tolist()},
               "sources": {"dir": str(run_dir), "arrays": str(arrays_path)}}
    (out / "belief_readout.json").write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
    (out / "belief_readout.md").write_text(markdown(stats, verdict, agreement, placement), encoding="utf-8")
    print(f"READING {'; '.join(verdict['readings'])}: {' | '.join(verdict['why'])}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", type=Path, required=True, help="the out dir of the six arms' re-grade runs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arrays", type=Path, default=cb.BENCH_ARRAYS, help="the benchmark's truth")
    args = ap.parse_args(argv)
    return run(args.dir, args.out, args.arrays)


if __name__ == "__main__":
    raise SystemExit(main())
