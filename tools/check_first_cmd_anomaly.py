"""Survey first-command divergences across the full vanilla replay
dataset to determine whether the "mid-game extractor cmd-order
anomaly" (BACKLOG.md, 2026-05-03) survives the mod purge.

Hypothesis under test
---------------------
The BACKLOG flagged a Hornshark Turn-13 case (`1643fdc46e00.json.gz`)
where extracted cmd[0] was a move from a hex that didn't hold a
side-1 unit at turn 1. The raw .bz2 for that case was purged; the
question is whether the bug is still present in the SURVIVING
replays (vanilla, mod-free corpus, ~13.9k files), or whether the
anomaly only manifested on now-deleted mod-affected replays.

Methodology
-----------
For each .json.gz in the dataset:
  1. Run diff_replay.diff_replay() with stop_on_first=True.
  2. Classify the FIRST divergence by:
     - cmd_index == 0 -> "starting-state mismatch" (the BACKLOG
       signature: extractor handed cmd[0] a state where the source
       unit doesn't exist).
     - cmd_index <= 2 -> "near-start mismatch" (likely the same
       failure class, just hitting an end_turn / init_side first).
     - kind starts with "move:src_missing" or "move:src_wrong_side"
       -> the BACKLOG-described anomaly even when cmd_index is
       larger (some other path drift first).
     - else -> mid-game divergence (different bug class, not what
       we're hunting here).
  3. Report counts overall and broken down by scenario.

Output
------
Top-line counts plus a per-scenario breakdown so we can see
whether the cmd-order anomaly is concentrated in one map (Hornshark
was the named offender) or scattered.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root on sys.path before importing diff_replay.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.diff_replay import diff_replay, Divergence

log = logging.getLogger("check_first_cmd_anomaly")


def _scenario_of(path: Path) -> str:
    """Pull scenario_id from the dataset file (one open per replay)."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("scenario_id", "unknown")
    except (OSError, json.JSONDecodeError):
        return "unknown"


def _classify(div: Divergence) -> str:
    """Bucket a divergence into the BACKLOG-relevant categories."""
    kind = div.kind
    # The exact BACKLOG signature: extractor's cmd[0] references a
    # source the starting state can't satisfy.
    if div.cmd_index == 0 and kind.startswith(
            ("move:src_missing", "move:src_wrong_side",
             "attack:attacker_missing", "attack:attacker_wrong_side",
             "recruit:")):
        return "first_cmd_anomaly"
    # Same bug class hitting cmd[1] or cmd[2] because cmd[0] was an
    # init_side / end_turn that doesn't reference units.
    if div.cmd_index <= 2 and kind.startswith(
            ("move:src_missing", "move:src_wrong_side",
             "attack:attacker_missing", "attack:attacker_wrong_side")):
        return "near_start_anomaly"
    # Mid-replay src_missing — could be cascade from earlier
    # divergence, but worth tracking separately.
    if kind.startswith(("move:src_missing", "move:src_wrong_side",
                        "attack:attacker_missing")):
        return "mid_replay_src_missing"
    return "other"


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", type=Path,
                    help="Directory of replay .json.gz files (e.g. "
                         "replays_dataset/).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap replays processed (default: all).")
    ap.add_argument("--show-examples", type=int, default=3,
                    help="Per-bucket example replays to dump.")
    ap.add_argument("--filter-competitive-2p", action="store_true",
                    help="Filter to scenarios in tools.scenarios."
                         "COMPETITIVE_2P_SCENARIOS — the Ladder Era "
                         "whitelist used by training. Only counts "
                         "anomalies that would actually affect self-play.")
    args = ap.parse_args(argv[1:])

    competitive: set = set()
    if args.filter_competitive_2p:
        from tools.scenarios import COMPETITIVE_2P_SCENARIOS
        competitive = set(COMPETITIVE_2P_SCENARIOS)
        log.warning(f"Filtering to {len(competitive)} competitive 2p scenarios")

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    files = sorted(args.dataset_dir.glob("*.json.gz"))
    if competitive:
        # Pre-filter via index.jsonl if present, else per-file scenario
        # peek (slower). Avoids running diff_replay on replays that
        # the training pipeline rejects anyway.
        idx = args.dataset_dir / "index.jsonl"
        if idx.exists():
            keep_names: set = set()
            with idx.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        m = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if m.get("scenario_id") in competitive:
                        keep_names.add(m.get("file"))
            files = [p for p in files if p.name in keep_names]
            log.warning(f"After whitelist filter: {len(files)} replays")
        else:
            log.warning("no index.jsonl; per-file scenario peek (slower)")
            files = [p for p in files
                     if _scenario_of(p) in competitive]
            log.warning(f"After whitelist filter: {len(files)} replays")
    if args.limit:
        files = files[: args.limit]
    if not files:
        log.error(f"no .json.gz files found in {args.dataset_dir}")
        return 2

    n_files = len(files)
    log.warning(f"Surveying {n_files} replays...")

    bucket_counts: Counter = Counter()
    bucket_examples: Dict[str, List[Tuple[str, int, str, str]]] = defaultdict(list)
    by_scenario: Dict[str, Counter] = defaultdict(Counter)
    n_clean = 0

    for i, p in enumerate(files):
        try:
            divs = diff_replay(p, stop_on_first=True)
        except Exception as e:
            bucket_counts["loader_exception"] += 1
            if len(bucket_examples["loader_exception"]) < args.show_examples:
                bucket_examples["loader_exception"].append(
                    (p.name, -1, type(e).__name__, str(e)[:120]))
            continue
        if not divs:
            n_clean += 1
            continue
        d = divs[0]
        bucket = _classify(d)
        bucket_counts[bucket] += 1
        scenario = _scenario_of(p)
        by_scenario[scenario][bucket] += 1
        if len(bucket_examples[bucket]) < args.show_examples:
            bucket_examples[bucket].append(
                (p.name, d.cmd_index, d.kind, d.detail[:120]))

        if (i + 1) % 500 == 0:
            log.warning(
                f"  {i + 1}/{n_files}: clean={n_clean}  "
                f"first_cmd={bucket_counts['first_cmd_anomaly']}  "
                f"near_start={bucket_counts['near_start_anomaly']}  "
                f"mid={bucket_counts['mid_replay_src_missing']}  "
                f"other={bucket_counts['other']}")

    print()
    print("=" * 72)
    print(f"check_first_cmd_anomaly: {n_files} replays, {n_clean} clean")
    print("=" * 72)
    print(f"  {n_clean} clean ({100.0 * n_clean / n_files:.2f}%)")
    for bucket in ("first_cmd_anomaly", "near_start_anomaly",
                   "mid_replay_src_missing", "other", "loader_exception"):
        n = bucket_counts[bucket]
        if not n:
            continue
        print(f"  {n:5d} {bucket} ({100.0 * n / n_files:.2f}%)")
        for ex in bucket_examples[bucket]:
            fname, idx, kind, detail = ex
            print(f"        e.g. {fname}#{idx}  {kind}: {detail}")
    print()

    # Per-scenario breakdown for the BACKLOG-targeted anomaly buckets.
    print("Per-scenario breakdown (first_cmd + near_start anomalies):")
    rows: List[Tuple[str, int, int, int]] = []
    for scen, c in by_scenario.items():
        first = c["first_cmd_anomaly"]
        near = c["near_start_anomaly"]
        total = sum(c.values())
        if first or near:
            rows.append((scen, first, near, total))
    rows.sort(key=lambda r: r[1] + r[2], reverse=True)
    for scen, first, near, total in rows[:20]:
        print(f"  {first:4d} first_cmd  {near:4d} near_start  "
              f"{total:4d} any-divergence  {scen}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
