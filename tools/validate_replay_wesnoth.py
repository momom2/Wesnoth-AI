"""Layer 2 replay validation: play a sim-exported replay back in REAL
Wesnoth and fail on out-of-sync markers in the engine log.

This is the ground-truth check layer 1 approximates: Wesnoth itself
re-simulates the recorded commands during replay playback and logs
desyncs. We launch `wesnoth --load <replay> --with-replay --nodelay`
(the man page's replay-playback path), pin the window to the
background (no focus stealing — the project rule), tail the newest
`wesnoth-*.out.log`, and scan for OOS signatures until the replay
ends or a timeout.

Requires a real Wesnoth install; used by the opt-in pytest
(WESNOTH_E2E=1) and manually:

    python tools/validate_replay_wesnoth.py logs/validate_mini.bz2
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from constants import WESNOTH_PATH, WESNOTH_LOGS_PATH

log = logging.getLogger("validate_replay_wesnoth")

SAVES_DIR = (Path.home() / "Documents" / "My Games" / "Wesnoth1.18"
             / "saves")

# Engine log signatures of replay desync, calibrated 2026-07-06 by
# playing a KNOWN-BAD replay with --log-debug=replay and reading what
# the engine actually wrote (don't guess — the first marker list
# false-PASSED the bad replay):
#   error replay: Recruiting leader not found at 6,5.
#   error replay: unfound location for source of movement: 5,5 -> 4,4
#   error replay: unfound location for source of attack
#   error replay: received a synced [command] from side 1. Expacted...
#   error replay: found dependent command in replay while is_synced=false
# " error replay:" prefixes them all; keep a couple of generic OOS
# phrases as belt-and-suspenders.
_OOS_MARKERS = (" error replay:", "out of sync", "desync",
                "checksum mismatch", "verification failed")
# Playback-activity signature: with --log-debug=replay the replay
# domain chats constantly during playback. A run with fewer than
# _MIN_ACTIVITY such lines never actually played the replay — report
# that as a FAILURE (an inconclusive run must not pass).
_ACTIVITY_MARKER = "replay:"
_MIN_ACTIVITY = 20


def validate_in_wesnoth(replay: Path, timeout: float = 180.0,
                        settle: float = 20.0) -> list:
    """Returns a list of offending log lines (empty = no OOS seen).

    `settle`: extra seconds to keep scanning after the last log
    growth, so slow playback isn't cut short. A replay that plays to
    the end OR goes quiet without OOS lines counts as clean."""
    if not WESNOTH_PATH.exists():
        raise RuntimeError(f"Wesnoth not found at {WESNOTH_PATH}")
    SAVES_DIR.mkdir(parents=True, exist_ok=True)
    target = SAVES_DIR / replay.name
    if replay.resolve() != target.resolve():
        shutil.copy2(replay, target)

    # Engine logs go to `wesnoth-*.log`; the `.out.log` twins carry
    # only Lua std_print output (learned 2026-07-06 — tailing the
    # wrong one made every verdict vacuous).
    def _engine_logs():
        if not WESNOTH_LOGS_PATH.exists():
            return set()
        return {p for p in WESNOTH_LOGS_PATH.glob("wesnoth-*.log")
                if not p.name.endswith(".out.log")}

    pre_logs = _engine_logs()
    cmd = [str(WESNOTH_PATH), "--load", replay.name, "--with-replay",
           "--nodelay", "--log-debug=replay", "--log-info=engine"]
    proc = subprocess.Popen(cmd)
    # Reuse the bridge's minimize-without-focus machinery.
    try:
        from wesnoth_interface import _pin_to_background
        _pin_to_background(proc.pid, log)
    except Exception:                                   # noqa: BLE001
        pass

    offending: list = []
    activity = 0
    log_file = None
    pos = 0
    t0 = time.time()
    last_growth = t0
    try:
        while time.time() - t0 < timeout:
            if log_file is None:
                cand = [p for p in _engine_logs() if p not in pre_logs]
                if cand:
                    log_file = max(cand, key=lambda p: p.stat().st_mtime)
                else:
                    time.sleep(1.0)
                    continue
            try:
                with log_file.open("r", encoding="utf-8",
                                   errors="replace") as f:
                    f.seek(pos)
                    new = f.read()
                    pos = f.tell()
            except OSError:
                time.sleep(1.0)
                continue
            if new:
                last_growth = time.time()
                for line in new.splitlines():
                    low = line.lower()
                    if _ACTIVITY_MARKER in low:
                        activity += 1
                    if any(m in low for m in _OOS_MARKERS):
                        offending.append(line.strip())
            if offending:
                return offending          # fail fast on first OOS
            if proc.poll() is not None:
                break                     # Wesnoth exited
            # The quiet-settle exit only applies once playback has
            # visibly STARTED — boot + asset load takes ~40s with
            # sparse logging, and bailing during it produced
            # inconclusive runs (2026-07-06 calibration).
            if (activity >= _MIN_ACTIVITY
                    and time.time() - last_growth > settle):
                break                     # playback done / stalled
            time.sleep(1.0)
        else:
            offending.append(f"TIMEOUT after {timeout}s")
        if activity < _MIN_ACTIVITY:
            offending.append(
                f"INCONCLUSIVE: only {activity} replay-domain log "
                f"lines — playback never ran (bad --load path?); "
                f"refusing to report success")
        return offending
    finally:
        if proc.poll() is None:
            proc.terminate()


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("replay", type=Path)
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=logging.INFO)
    bad = validate_in_wesnoth(args.replay, timeout=args.timeout)
    if bad:
        print(f"OOS/errors in Wesnoth playback of {args.replay.name}:")
        for line in bad[:20]:
            print(f"  {line}")
        return 1
    print(f"clean playback: {args.replay.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
