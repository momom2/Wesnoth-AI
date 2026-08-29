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

def decide_ensure(entry: Optional[Dict], alive: bool,
                  orphans: bool = False) -> str:
    """-> 'keep' | 'spawn' | 'orphan'. `alive` is the identity-
    verified liveness of entry's pid; `orphans` = the leader is
    dead but same-namespace group members survive (round-37 C1:
    spawning a SECOND trainer over a live orphaned sim would
    interleave two runs' weights into one rolling checkpoint)."""
    if entry and alive:
        return "keep"
    if entry and orphans:
        return "orphan"
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

def _ns_identity():
    """(pid-namespace inode, pid 1 starttime) -- the identity of
    THIS container's pid space; None off-Linux. boot_id was the
    wrong key (round-36 C0: Docker does not namespace the host's
    boot id, so a Vast stop/start -- fresh pid namespace, pids
    from 1 -- looked 'same boot' while every recorded number was
    recycled)."""
    try:
        ns = os.stat("/proc/self/ns/pid").st_ino
        with open("/proc/1/stat", "rb") as f:
            st = f.read().decode("ascii", "replace")
        p1 = int(st.rsplit(")", 1)[1].split()[19])
        return ns, p1
    except (OSError, IndexError, ValueError):
        return None


def _group_members(pgid: int):
    """Live pids whose process group is `pgid` (Linux /proc scan;
    [] off-Linux or on error)."""
    out = []
    try:
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                with open(f"/proc/{d}/stat", "rb") as f:
                    st = f.read().decode("ascii", "replace")
                if int(st.rsplit(")", 1)[1].split()[2]) == pgid:
                    out.append(int(d))
            except (OSError, IndexError, ValueError):
                continue
    except OSError:
        pass
    return out


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _proc_identity(pid: int):
    """(starttime_ticks, argv0) of a live pid, or None. starttime is
    /proc/<pid>/stat field 22 (ticks since host boot) -- the same
    parse shape stall_watchdog uses; together with argv0 it makes a
    recorded pid distinguishable from an UNRELATED process that
    reused the number after an instance restart (round-33 C2:
    leg_status.json persists on /workspace across stop/restart, so
    a bare kill(pid, 0) could 'keep' a stranger and never respawn
    the daemon -- or later kill that stranger's process group)."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            stat = f.read().decode("ascii", "replace")
        # comm can contain spaces/parens: split after the LAST ')'.
        tail = stat.rsplit(")", 1)[1].split()
        starttime = int(tail[19])
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            argv0 = f.read().split(b"\0", 1)[0].decode(
                "utf-8", "replace")
        return starttime, argv0
    except (OSError, IndexError, ValueError):
        return None


def _cmdline_matches(pid: int, cmd) -> bool:
    """Exact argv match of a live pid against a recorded cmd list;
    False on any error/off-Linux."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            parts = f.read().split(b"\0")
        if parts and parts[-1] == b"":
            parts = parts[:-1]
        return [p.decode("utf-8", "replace")
                for p in parts] == list(cmd or [])
    except OSError:
        return False


def _entry_alive(entry) -> bool:
    """Liveness = pid exists AND its identity matches the record.
    A record without identity fields (the committed pre-round-33
    shape) is accepted ONLY on an exact argv match -- the True
    fallback bypassed every identity guard for exactly the records
    that predate them, killpg'ing recycled numbers and refusing to
    respawn dead daemons (round-37 C0)."""
    if not entry:
        return False
    pid = int(entry["pid"])
    if not _pid_alive(pid):
        return False
    if "starttime" not in entry:
        return _cmdline_matches(pid, entry.get("cmd"))
    ident = _proc_identity(pid)
    return (ident is not None
            and ident[0] == entry["starttime"]
            and ident[1] == entry.get("argv0"))


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
        alive = _entry_alive(entry)
        _orph = []
        if (entry and not alive
                and not _pid_alive(int(entry["pid"]))):
            _ns0 = _ns_identity()
            if (_ns0 is not None
                    and entry.get("ns_id") == _ns0[0]
                    and entry.get("pid1_start") == _ns0[1]):
                _orph = _group_members(int(entry["pgid"]))
        _decision = decide_ensure(entry, alive,
                                  orphans=bool(_orph))
        if _decision == "orphan":
            print(f"{name}: NOT spawned -- leader {entry['pid']} "
                  f"is dead but pids {_orph} still run in pgid "
                  f"{entry['pgid']} (round-37 C1: a second "
                  f"{name} would fight the orphan over the same "
                  f"files). Verify with `ps -o pid,pgid,cmd` and "
                  f"reap with `kill -- -{entry['pgid']}`, then "
                  f"re-run ensure.")
            return status, 1
        if _decision == "keep":
            if "starttime" not in entry:
                # One-boot migration: stamp identity onto a
                # verified legacy record (round-37 C0).
                _id2 = _proc_identity(int(entry["pid"]))
                if _id2 is not None:
                    entry = dict(entry)
                    entry["starttime"], entry["argv0"] = _id2
                    _ns2 = _ns_identity()
                    if _ns2 is not None:
                        entry["ns_id"], entry["pid1_start"] = _ns2
                    status = merge_status(status, name, entry)
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
        ident = _proc_identity(proc.pid)
        if ident is not None:
            rec["starttime"], rec["argv0"] = ident
        _ns = _ns_identity()
        if _ns is not None:
            rec["ns_id"], rec["pid1_start"] = _ns
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
        rc = 0
        for n in targets:
            e = status.get(n)
            if e and _entry_alive(e):
                _kill_group(int(e["pgid"]))
                print(f"{n}: stopped (pgid {e['pgid']})")
                status = merge_status(status, n, None)
            elif e:
                # NEVER killpg on an inferred number: the round-35
                # auto-reap fired on recycled pids/pgids in two ways
                # (round-36 C0: container restarts keep the HOST
                # boot id while the pid namespace resets; round-36
                # C1: an alive-but-different pid PROVES recycling,
                # and the stranger may be a group leader). Signals
                # go only to identity-VERIFIED records (the branch
                # above); here we detect the one genuine orphan
                # case and tell the operator instead of guessing.
                _ns = _ns_identity()
                _same_ns = (_ns is not None
                            and e.get("ns_id") == _ns[0]
                            and e.get("pid1_start") == _ns[1])
                _members = ([] if not _same_ns
                            else _group_members(int(e["pgid"])))
                if _members and not _pid_alive(int(e["pid"])):
                    print(f"{n}: leader {e['pid']} is dead but "
                          f"pids {_members} still run in pgid "
                          f"{e['pgid']} (same pid namespace). "
                          f"NOT killed automatically -- verify "
                          f"with `ps -o pid,pgid,cmd` and reap "
                          f"with `kill -- -{e['pgid']}` if they "
                          f"are the orphaned daemon.")
                    # The record SURVIVES (round-38 C0: dropping
                    # it disarmed ensure()'s orphan gate, so the
                    # very next ensure spawned a second daemon
                    # over the live orphan). The stale path below
                    # drops it once the group is actually empty.
                    rc = 1
                else:
                    print(f"{n}: stale record dropped (pid "
                          f"{e['pid']} is dead or a different "
                          f"process); nothing killed")
                    status = merge_status(status, n, None)
            else:
                print(f"{n}: not recorded")
        return status, rc
    return _with_status(mutate)


def show() -> int:
    def mutate(status):
        for n, e in sorted(status.items()):
            alive = _entry_alive(e)
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
