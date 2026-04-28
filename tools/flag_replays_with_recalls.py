"""Scan the replay corpus for PvP replays containing recall actions.

PvP shouldn't have recalls -- recalls require a recall list, which
only persists across scenarios in a campaign. Replays in
`replays_dataset/*.json.gz` that contain a `recall` action are
either misclassified (campaign games leaking into the PvP corpus)
or use a non-default scenario that allows recalls. Either way they
shouldn't train the policy.

Output: writes `logs/replays_with_recalls.txt`, one tab-separated
line per recall instance (multiple lines per game if multiple
recalls):

    <game_id>\t<turn>\t<side>\t<unit_id>\t<x>\t<y>

Plus a one-line summary on stderr:

    [flag_replays_with_recalls] scanned 53451 replays, 17 hit a
    recall (0.03%); see logs/replays_with_recalls.txt

Usage:

    python tools/flag_replays_with_recalls.py
        [--replays-dir replays_dataset]
        [--out logs/replays_with_recalls.txt]

Cost: O(N) over the corpus. ~1-2 min for 50k replays on this
machine.

Once a replay is flagged, manual inspection decides whether to:
  - delete from the corpus (it's a campaign / bad data),
  - keep but skip during training (filter out via the index), or
  - re-export with the recall stripped (if the rest of the game
    is informative).
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path
from typing import List

# Project root importable when run as a script.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))


log = logging.getLogger("flag_replays_with_recalls")


def _scan_replay(path: Path) -> List[dict]:
    """Return a list of recall events found in `path`. Each event is
    a dict with keys: game_id, turn, side, unit_id, x, y. Empty
    list if no recalls are present (the common case).

    We work directly on the JSON structure (replays_dataset format)
    rather than running the full recon pipeline -- a corpus scan
    needs to be cheap. Format (from tools/replay_extract.py):

        data["commands"] = [
            ["init_side", side],            # marks turn boundary
            ["move", xs, ys, side],
            ["attack", ax, ay, dx, dy, ...],
            ["recruit", unit_type, x, y, seed],
            ["recall", unit_id, x, y],      # <-- what we look for
            ["end_turn"],
            ...
        ]

    `turn` and `side` aren't on each command; we track them by
    walking the stream and watching `init_side` (which alternates
    sides; turn increments when side wraps to 1).
    """
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, gzip.BadGzipFile) as e:
        log.warning(f"could not read {path.name}: {e}")
        return []
    game_id = data.get("game_id", path.stem)
    out: List[dict] = []
    commands = data.get("commands") or []
    cur_side = 0
    cur_turn = 0
    prev_side = 0
    for cmd in commands:
        if not isinstance(cmd, list) or not cmd:
            continue
        kind = cmd[0]
        if kind == "init_side":
            cur_side = int(cmd[1]) if len(cmd) > 1 else 0
            # Turn increments when side wraps from N back to 1
            # (Wesnoth's convention; n_sides=2 in PvP).
            if cur_side <= prev_side:
                cur_turn += 1
            prev_side = cur_side
            continue
        if kind != "recall":
            continue
        unit_id = cmd[1] if len(cmd) > 1 else "<unknown>"
        x = cmd[2] if len(cmd) > 2 else -1
        y = cmd[3] if len(cmd) > 3 else -1
        out.append({
            "game_id":  game_id,
            "turn":     cur_turn,
            "side":     cur_side,
            "unit_id":  unit_id,
            "x":        x,
            "y":        y,
        })
    return out


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replays-dir", type=Path,
                    default=Path("replays_dataset"),
                    help="Directory of .json.gz replays to scan. Default: replays_dataset/")
    ap.add_argument("--out", type=Path,
                    default=Path("logs/replays_with_recalls.txt"),
                    help="Output file. Default: logs/replays_with_recalls.txt")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING"])
    args = ap.parse_args(argv[1:])

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.replays_dir.is_dir():
        print(f"[flag_replays_with_recalls] no such directory: "
              f"{args.replays_dir}", file=sys.stderr)
        return 2

    files = sorted(args.replays_dir.glob("*.json.gz"))
    if not files:
        print(f"[flag_replays_with_recalls] no .json.gz under "
              f"{args.replays_dir}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total_files = 0
    files_with_recalls = 0
    total_recalls = 0
    with args.out.open("w", encoding="utf-8") as f_out:
        f_out.write("# Replays containing recall actions. Generated by\n")
        f_out.write("# tools/flag_replays_with_recalls.py.\n")
        f_out.write("# Format: <game_id>\\t<turn>\\t<side>\\t<unit_id>\\t<x>\\t<y>\n")
        for path in files:
            total_files += 1
            events = _scan_replay(path)
            if events:
                files_with_recalls += 1
                total_recalls += len(events)
                for e in events:
                    f_out.write(
                        f"{e['game_id']}\t{e['turn']}\t{e['side']}\t"
                        f"{e['unit_id']}\t{e['x']}\t{e['y']}\n"
                    )
            if total_files % 5000 == 0:
                log.info(f"scanned {total_files}/{len(files)} "
                         f"({files_with_recalls} hit so far)")

    pct = (100.0 * files_with_recalls / total_files) if total_files else 0.0
    print(
        f"[flag_replays_with_recalls] scanned {total_files} replays, "
        f"{files_with_recalls} hit a recall ({pct:.3f}%); "
        f"{total_recalls} total recall events; "
        f"see {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
