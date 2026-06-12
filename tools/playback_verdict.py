"""Scan Wesnoth logs for replay-verification failures.

Companion to the verifiable-checkup exports (sim_to_replay): after
you watch an exported replay in Wesnoth (open the save with
"Show replay" and press Play -- there is no non-interactive mode on
the stock binary, see BACKLOG "Wesnoth-playback verification
harness"), run this to turn the session's log into a verdict:

    python tools/playback_verdict.py            # newest log
    python tools/playback_verdict.py --all      # every log file
    python tools/playback_verdict.py --log <path-to-wesnoth-*.log>

Exit codes: 0 = no divergence markers found, 1 = found, 2 = no log.

What it looks for (1.18.4 sources):
  - "SYNC:"            attack.cpp's errbuf_ lines when a [checkup]
                       [result] disagrees with playback calculations
                       (the per-strike chance/hits/damage/dies we
                       emit) -- the primary combat-parity signal.
  - "error replay"     the replay log domain: structural command
                       errors (illegal moves, gold shortfalls,
                       orphaned dependent commands).
  - "out of sync"      the generic OOS dialog text / log echo.
  - "Checksum mismatch" unit-state checksum failures.

A clean run prints CLEAN with the number of lines scanned. Note the
limits: empty [checkup] blocks (moves, recruits) verify nothing, so
CLEAN means "no divergence DETECTED", strongest for combat-heavy
replays.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

_DEFAULT_LOG_DIR = (Path.home() / "Documents" / "My Games"
                    / "Wesnoth1.18" / "logs")

# (marker regex, human label) -- case-insensitive.
_MARKERS: List[Tuple[str, str]] = [
    (r"SYNC:", "checkup result mismatch (combat divergence)"),
    (r"error replay", "replay-engine error"),
    (r"out of sync", "out-of-sync report"),
    (r"checksum mismatch", "unit checksum mismatch"),
]


def scan_log(path: Path) -> Tuple[int, List[str]]:
    """Return (lines_scanned, offending_lines) for one log file."""
    hits: List[str] = []
    n = 0
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        n += 1
        for pattern, label in _MARKERS:
            if re.search(pattern, line, re.IGNORECASE):
                hits.append(f"[{label}] {line.strip()}")
                break
    return n, hits


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, default=None,
                    help="Specific log file (default: newest "
                         "wesnoth-*.log under the userdata logs dir).")
    ap.add_argument("--log-dir", type=Path, default=_DEFAULT_LOG_DIR,
                    help="Wesnoth userdata logs directory.")
    ap.add_argument("--all", action="store_true",
                    help="Scan every log in the directory, newest "
                         "first, instead of just the newest.")
    args = ap.parse_args(argv[1:])

    if args.log is not None:
        logs = [args.log]
    else:
        if not args.log_dir.is_dir():
            print(f"no log dir at {args.log_dir}", file=sys.stderr)
            return 2
        logs = sorted(args.log_dir.glob("wesnoth-*.log"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if not args.all:
            logs = logs[:1]
    if not logs:
        print("no wesnoth logs found", file=sys.stderr)
        return 2

    any_hits = False
    for log_path in logs:
        n, hits = scan_log(log_path)
        verdict = "DIVERGED" if hits else "CLEAN"
        print(f"{verdict}  {log_path.name}  ({n} lines)")
        for h in hits:
            print(f"    {h}")
        any_hits = any_hits or bool(hits)
    return 1 if any_hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
