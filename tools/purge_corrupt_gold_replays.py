"""Detect and quarantine-for-review replays with impossible recruit-gold
sequences.

When a replay records a recruit whose cost exceeds the player's
gold (computed from starting_gold - prior recruits + income), the
replay MAY be corrupt -- a player edited the file manually, played
on a non-vanilla patch, or hit a save/load engine bug. The user
verified two specific cases on 2026-05-03 (Hornshark Turn 2 and
Den of Onis Turn 2) by replaying them in the Wesnoth GUI: both
desync at the illegal recruit. But not all `recruit:insufficient_gold`
flags are necessarily corrupt -- some might be legitimate
state-drift cascades from earlier sim bugs. So this tool MOVES
flagged replays to a separate folder for human review rather than
deleting them.

Wesnoth's UI gates recruits on `gold >= cost`
(menu_events.cpp:327), so a legitimate replay should not record
such a sequence. Negative gold only arises from upkeep > income,
not from recruits.

This tool walks each replay through the simulator, flags any
replay where a `recruit` command is issued with `gold < cost`,
and (with --apply) MOVES those raw .bz2 files to a review folder
for the user to inspect by hand. Default is dry-run (just count).

Dependencies: tools.diff_replay, tools.replay_dataset.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import List

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

from tools.diff_replay import diff_replay


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", type=Path,
                    help="Extracted .json.gz dataset (e.g. replays_dataset_full)")
    ap.add_argument("--raw-dir", type=Path, default=Path("replays_raw"),
                    help="Raw .bz2 corpus to quarantine from (default: replays_raw)")
    ap.add_argument("--quarantine", type=Path,
                    default=Path("replays_raw_review_gold"),
                    help="Where to MOVE flagged replays for user review")
    ap.add_argument("--apply", action="store_true",
                    help="Actually move files (default: dry-run)")
    ap.add_argument("--max-turn", type=int, default=None,
                    help="Only purge if first recruit:insufficient_gold "
                         "happens at turn <= N (default: any turn)")
    args = ap.parse_args(argv[1:])

    files = sorted(args.dataset_dir.glob("*.json.gz"))
    print(f"Scanning {len(files)} extracted replays...")

    corrupt_ids: List[str] = []
    by_turn = Counter()
    n_processed = 0
    for p in files:
        n_processed += 1
        try:
            divs = diff_replay(p, stop_on_first=True)
        except Exception:
            continue
        if not divs:
            continue
        d = divs[0]
        if d.kind != "recruit:insufficient_gold":
            continue
        if args.max_turn is not None and d.turn > args.max_turn:
            continue
        # Pull game_id to find the raw .bz2
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                data = json.load(f)
            game_id = data.get("game_id", "")
        except Exception:
            continue
        if not game_id:
            continue
        corrupt_ids.append(game_id)
        by_turn[d.turn] += 1
        if n_processed % 1000 == 0:
            print(f"  scanned {n_processed}: {len(corrupt_ids)} corrupt found")

    print()
    print(f"Found {len(corrupt_ids)} replays with recruit:insufficient_gold "
          f"(flag for review, not necessarily all corrupt).")
    print("By turn of first divergence:")
    for t, n in sorted(by_turn.items()):
        print(f"  turn {t}: {n}")
    if not corrupt_ids:
        return 0

    if not args.apply:
        print("\n(dry-run; pass --apply to actually move files)")
        print("First 5 corrupt game_ids:")
        for gid in corrupt_ids[:5]:
            print(f"  {gid}")
        return 0

    args.quarantine.mkdir(parents=True, exist_ok=True)
    n_moved = 0
    for gid in corrupt_ids:
        # Locate the raw .bz2
        raws = list(args.raw_dir.rglob(f"{gid}.bz2"))
        if not raws:
            continue
        for r in raws:
            dst = args.quarantine / r.name
            shutil.move(str(r), str(dst))
            n_moved += 1
    print(f"Moved {n_moved} raw .bz2 files to {args.quarantine}/")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
