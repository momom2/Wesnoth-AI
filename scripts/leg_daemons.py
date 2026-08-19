"""Leg daemon reconciler (user ruling 2026-08-19: architectural,
not guarded).

Replaces the launcher's nohup-spawn + pkill-by-pattern blocks with
convergence to a declared state:

  ensure NAME -- CMD...   spawn NAME in its OWN process group iff
                          not already healthy (idempotent: N calls
                          converge; no lock discipline needed by
                          callers -- the status file is the single
                          synchronized truth).
  stop NAME | stop-all    SIGTERM then SIGKILL the recorded process
                          GROUP -- teardown by pgid, never by
                          pattern, so self-match is inexpressible.
  status                  the structured state (also what watchers
                          should read instead of grepping logs).

State: $WORKDIR/leg_status.json -- {name: {pid, pgid, cmd, started}}
under an fcntl lock (the lock is the file's consistency primitive,
not an incident guard). Daemon stdout/stderr append to
$WORKDIR/<name>.log.

POSIX-only by nature (setsid/killpg); the spawn/kill decision logic
is pure and unit-tested cross-platform (tests/test_leg_daemons.py).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

WORKDIR = Path(os.environ.get("WORKDIR", "/workspace"))
STATUS = WORKDIR / "leg_status.json"


# ---------------------------------------------------------------------
# Pure decision logic (unit-tested; no syscalls)
# ---------------------------------------------------------------------

def decide_ensure(entry: Optional[Dict], alive: bool) -> str:
    """-> 'keep' | 'spawn'. `alive` is the liveness of entry's pid
    as established by the caller; a missing/dead entry spawns."""
    if entry and alive:
        return "keep"
    return "spawn"


def merge_status(status: Dict, name: str, rec: Optional[Dict]) -> Dict:
    out = dict(status)
    if rec is None:
        out.pop(name, None)
    else:
        out[name] = rec
    return out


# ---------------------------------------------------------------------
# POSIX shell
# ---------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _load_locked(f):
    try:
        return json.loads(f.read() or "{}")
    except json.JSONDecodeError:
        return {}


def _with_status(mutate):
    """Read-modify-write the status file under an exclusive lock."""
    import fcntl
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        status = _load_locked(f)
        status, ret = mutate(status)
        f.seek(0)
        f.truncate()
        json.dump(status, f, indent=1)
        f.write("\n")
        return ret


def ensure(name: str, cmd: List[str]) -> int:
    def mutate(status):
        entry = status.get(name)
        alive = bool(entry) and _pid_alive(int(entry["pid"]))
        if decide_ensure(entry, alive) == "keep":
            print(f"{name}: already running (pid {entry['pid']})")
            return status, 0
        log = open(WORKDIR / f"{name}.log", "ab")
        proc = subprocess.Popen(
            cmd, stdout=log, stderr=log,
            start_new_session=True,   # own process group == own pgid
            cwd=os.getcwd())
        rec = {"pid": proc.pid, "pgid": os.getpgid(proc.pid),
               "cmd": cmd,
               "started": time.strftime("%FT%TZ", time.gmtime())}
        print(f"{name}: spawned pid {proc.pid} pgid {rec['pgid']}")
        return merge_status(status, name, rec), 0
    return _with_status(mutate)


def _kill_group(pgid: int) -> None:
    import signal
    for sig, wait in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 0.0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        t0 = time.time()
        while time.time() - t0 < wait:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.2)


def stop(names: Optional[List[str]] = None) -> int:
    def mutate(status):
        targets = names if names else list(status)
        for n in targets:
            e = status.get(n)
            if e:
                _kill_group(int(e["pgid"]))
                print(f"{n}: stopped (pgid {e['pgid']})")
                status = merge_status(status, n, None)
            else:
                print(f"{n}: not recorded")
        return status, 0
    return _with_status(mutate)


def show() -> int:
    def mutate(status):
        for n, e in sorted(status.items()):
            alive = _pid_alive(int(e["pid"]))
            print(f"{n:<12} pid={e['pid']:<8} pgid={e['pgid']:<8} "
                  f"{'ALIVE' if alive else 'DEAD'}  since {e['started']}")
        if not status:
            print("(no daemons recorded)")
        return status, 0
    return _with_status(mutate)


def main(argv) -> int:
    if len(argv) >= 4 and argv[1] == "ensure" and argv[3] == "--":
        return ensure(argv[2], argv[4:])
    if len(argv) >= 3 and argv[1] == "stop":
        return stop(argv[2:])
    if len(argv) == 2 and argv[1] == "stop-all":
        return stop(None)
    if len(argv) == 2 and argv[1] == "status":
        return show()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
