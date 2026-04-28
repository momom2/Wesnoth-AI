"""Compute inverse-frequency action-type loss weights from the
supervised corpus.

Output: one normalized weight per leaf-action-type, scaled so the
mean weight is 1.0. The intent is "rare action types get more
gradient signal per occurrence so the model doesn't ignore them."
With raw frequencies (e.g. 50% end_turn / 30% move / 15% recruit
/ 5% attack) the inverse is (2, 3.33, 6.67, 20). Mean is 8;
normalizing gives (~0.25, ~0.42, ~0.83, ~2.5). The 5x ratio
between attack and end_turn matches the user's intuition while
being grounded in the actual distribution.

Usage:

    python tools/compute_action_type_weights.py
        [--replays-dir replays_dataset]
        [--out cluster/configs/action_type_weights.json]

Output format (JSON):

    {
      "_about": "inverse-frequency weights scaled to mean=1; ...",
      "_corpus_size": 53450,
      "_counts": {"end_turn": ..., "move": ..., ...},
      "_total_actions": ...,
      "weights": {
        "recruit": ...,
        "attack":  ...,
        "move":    ...,
        "end_turn":...,
        "recall":  ...
      }
    }

The supervised trainer reads `weights` and applies them to the
per-action-type loss. `recall` is included for completeness (PvP
shouldn't have it; surfaces in ~30% of mixed corpus, see
flag_replays_with_recalls.py).

Cost: O(N) over the corpus, ~30s for 50k replays.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

# Project root importable when run as a script.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))


log = logging.getLogger("compute_action_type_weights")


def _scan_replay_action_types(path: Path) -> Counter:
    """Count action types in one replay's commands."""
    out: Counter = Counter()
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, gzip.BadGzipFile) as e:
        log.warning(f"could not read {path.name}: {e}")
        return out
    for cmd in data.get("commands", []):
        if not isinstance(cmd, list) or not cmd:
            continue
        kind = cmd[0]
        # Skip control flow (init_side); count only action kinds.
        if kind in ("move", "attack", "recruit", "recall", "end_turn"):
            out[kind] += 1
    return out


def _inverse_frequency_weights(counts: Counter,
                               ignore_recall: bool = True) -> Dict[str, float]:
    """Convert raw action-type counts -> mean-1-normalized
    inverse-frequency weights.

    `ignore_recall=True` (default): exclude recall events from BOTH
    the total denominator AND the per-type weights. Recall slot
    still appears in the output with weight 0 (for schema
    completeness; the supervised loss won't fire on it anyway).
    Rationale: PvP corpus shouldn't have recalls; including them
    in normalization inflates the mean and squeezes the other
    types' weights closer together than they should be.

    Action types with zero observations get weight 0 (no
    upweighting of unseen types since their loss term never fires)."""
    if ignore_recall:
        considered = {k: counts.get(k, 0)
                      for k in ("move", "attack", "recruit", "end_turn")}
    else:
        considered = {k: counts.get(k, 0)
                      for k in ("move", "attack", "recruit", "recall",
                                "end_turn")}
    total = sum(considered.values())
    if total == 0:
        return {}
    raw: Dict[str, float] = {}
    for k in ("move", "attack", "recruit", "recall", "end_turn"):
        n = considered.get(k, 0)
        if n == 0:
            raw[k] = 0.0
        else:
            raw[k] = total / n
    nonzero = [v for v in raw.values() if v > 0]
    if not nonzero:
        return raw
    mean = sum(nonzero) / len(nonzero)
    return {k: (v / mean if v > 0 else 0.0) for k, v in raw.items()}


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replays-dir", type=Path,
                    default=Path("replays_dataset"),
                    help="Directory of .json.gz replays. Default: replays_dataset/")
    ap.add_argument("--out", type=Path,
                    default=Path("cluster/configs/action_type_weights.json"),
                    help="Output JSON path. Default: cluster/configs/action_type_weights.json")
    ap.add_argument("--max-files", type=int, default=0,
                    help="Cap on files scanned (0 = no cap; for quick dev iterations).")
    ap.add_argument("--ignore-recall", dest="ignore_recall",
                    action="store_true", default=True,
                    help="Drop recall events from the count + normalization "
                         "(default: True, since PvP shouldn't include recalls; "
                         "see flag_replays_with_recalls.py). Recall events still "
                         "get weight=0 in the output for completeness.")
    ap.add_argument("--keep-recall", dest="ignore_recall",
                    action="store_false",
                    help="Include recall events in the normalization. Useful "
                         "when training on a mixed campaign+PvP corpus.")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING"])
    args = ap.parse_args(argv[1:])

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.replays_dir.is_dir():
        print(f"[compute_action_type_weights] no such directory: "
              f"{args.replays_dir}", file=sys.stderr)
        return 2

    files = sorted(args.replays_dir.glob("*.json.gz"))
    if args.max_files > 0:
        files = files[:args.max_files]
    if not files:
        print(f"[compute_action_type_weights] no .json.gz under "
              f"{args.replays_dir}", file=sys.stderr)
        return 2

    counts: Counter = Counter()
    for i, path in enumerate(files, 1):
        counts.update(_scan_replay_action_types(path))
        if i % 5000 == 0:
            log.info(f"scanned {i}/{len(files)}")

    weights = _inverse_frequency_weights(counts, ignore_recall=args.ignore_recall)
    total = sum(counts.values())

    out_payload = {
        "_about": (
            "Inverse-frequency action-type loss weights scaled so the "
            "mean of nonzero weights is 1.0. Generated by "
            "tools/compute_action_type_weights.py from "
            f"{args.replays_dir}. Re-run after corpus changes."
        ),
        "_corpus_size": len(files),
        "_total_actions": total,
        "_counts": dict(counts),
        "weights": weights,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, sort_keys=False)
    print(
        f"[compute_action_type_weights] scanned {len(files)} replays, "
        f"{total} actions; wrote {args.out}",
        file=sys.stderr,
    )
    print(f"  weights: {json.dumps(weights, indent=2)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
