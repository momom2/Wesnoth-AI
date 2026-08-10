"""Box-side training stall watchdog (BACKLOG item 1, 2026-08-10).

The 2026-08-08 imitation run HUNG at ~0 CPU with no traceback and
billed ~2 idle days before discovery: the A5 tripwires only fire on
COMPLETED iterations, and the onstart relaunch loop only catches
exit codes -- a hang produces neither. This watchdog is the missing
leg: it watches the cumulative CPU time of every `tools/
sim_self_play.py` process (the hang symptom IS ~0 CPU; /proc/loadavg
is host-wide on Vast containers and must not be used), and when the
whole set burns less than STALL_CPU_SECONDS over STALL_WINDOW
seconds, it

  1. writes $WORKDIR/WATCHDOG_STALL (timestamped evidence), then
  2. SIGKILLs the training processes.

The onstart relaunch loop treats a signal exit WITH that marker as a
crash (consume marker, relaunch) instead of an operator stop (stand
down) -- so a stalled leg loses minutes, not days, and 20 repeated
stalls still exhaust the relaunch cap and stop the box loudly.

Linux-only (/proc); launched by vast_onstart.sh next to the HF
uploader. Env knobs: STALL_WINDOW (s, default 1800), STALL_CPU_SECONDS
(default 30), WATCH_EVERY (s, default 300), WORKDIR.
"""
from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

WORKDIR = Path(os.environ.get("WORKDIR", "/workspace"))
WATCH_EVERY = int(os.environ.get("WATCH_EVERY", "300"))
STALL_WINDOW = int(os.environ.get("STALL_WINDOW", "1800"))
STALL_CPU_SECONDS = float(os.environ.get("STALL_CPU_SECONDS", "30"))
MARKER = WORKDIR / "WATCHDOG_STALL"
PATTERN = "tools/sim_self_play.py"


def _training_procs():
    """{pid: cpu_seconds} for every process running the training
    entry point. /proc scan; a pid vanishing mid-read is skipped."""
    out = {}
    hertz = os.sysconf("SC_CLK_TCK")
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmdline = (p / "cmdline").read_bytes().replace(b"\0", b" ")
            if PATTERN.encode() not in cmdline:
                continue
            stat = (p / "stat").read_text().rsplit(") ", 1)[1].split()
            # fields 12/13 (utime/stime) counted from AFTER comm --
            # here indices 11 and 12 of the post-comm tail.
            out[int(p.name)] = (int(stat[11]) + int(stat[12])) / hertz
        except (OSError, IndexError, ValueError):
            continue
    return out


def main() -> int:
    print(f"stall_watchdog: window={STALL_WINDOW}s "
          f"threshold={STALL_CPU_SECONDS} cpu-s every {WATCH_EVERY}s",
          flush=True)
    history: list = []          # (timestamp, {pid: cpu_s})
    while True:
        time.sleep(WATCH_EVERY)
        try:
            now = time.time()
            procs = _training_procs()
            history.append((now, procs))
            history = [(t, s) for t, s in history
                       if now - t <= STALL_WINDOW + WATCH_EVERY]
            if not procs:
                continue        # nothing running: the relaunch loop's job
            if now - history[0][0] < STALL_WINDOW:
                continue        # window not full yet (also post-kill reset)
            old = history[0][1]
            # CPU burned by pids alive across the whole window. New
            # pids (relaunch churn) don't count toward EITHER side --
            # a fresh healthy process resets the window below.
            shared = set(old) & set(procs)
            if not shared:
                history = [(now, procs)]
                continue
            burned = sum(procs[p] - old[p] for p in shared)
            if burned >= STALL_CPU_SECONDS:
                continue
            MARKER.write_text(
                f"{time.strftime('%FT%TZ', time.gmtime())} pids="
                f"{sorted(shared)} cpu_delta={burned:.1f}s over "
                f"{STALL_WINDOW}s (threshold {STALL_CPU_SECONDS}s)\n",
                encoding="utf-8")
            print(f"stall_watchdog: STALL -- {MARKER.read_text().strip()} "
                  f"-- killing", flush=True)
            for pid in shared:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
            history = []        # fresh window for the relaunched run
        except Exception as e:                      # noqa: BLE001
            print(f"stall_watchdog: cycle failed: {e!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
