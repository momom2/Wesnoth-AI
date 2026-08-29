"""leg_daemons stop/ensure decision logic (rounds 34-36): signals
go ONLY to identity-verified records; stale records are dropped or
reported, never killed on an inferred number. All /proc and status
I/O is stubbed, so this runs on any platform."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_SPEC = importlib.util.spec_from_file_location(
    "leg_daemons",
    Path(__file__).resolve().parent.parent / "scripts"
    / "leg_daemons.py")
ld = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ld)


def _run_stop(monkeypatch, entry, *, pid_alive, identity_ok,
              same_ns, members):
    """Drive stop('d') against one stubbed record; returns the list
    of pgids killed, the printed lines, and the surviving status."""
    killed = []
    printed = []
    survived = {}
    monkeypatch.setattr(ld, "_pid_alive", lambda pid: pid_alive)
    monkeypatch.setattr(
        ld, "_entry_alive",
        lambda e: bool(e) and pid_alive and identity_ok)
    monkeypatch.setattr(
        ld, "_ns_identity",
        lambda: (7, 100) if same_ns else (8, 999))
    monkeypatch.setattr(ld, "_group_members",
                        lambda pgid: list(members))
    monkeypatch.setattr(ld, "_kill_group",
                        lambda pgid: killed.append(pgid))
    def _fake_with_status(mutate):
        st, ret = mutate({"d": dict(entry)})
        survived.clear()
        survived.update(st)
        return ret
    monkeypatch.setattr(ld, "_with_status", _fake_with_status)
    import builtins
    real_print = builtins.print
    monkeypatch.setattr(builtins, "print",
                        lambda *a, **k: printed.append(" ".join(
                            str(x) for x in a)))
    try:
        ld.stop(["d"])
    finally:
        monkeypatch.setattr(builtins, "print", real_print)
    return killed, printed, survived


_ENTRY = {"pid": 4242, "pgid": 4242, "cmd": ["python", "x.py"],
          "started": "t", "starttime": 5, "argv0": "python",
          "ns_id": 7, "pid1_start": 100}


def test_verified_record_is_killed(monkeypatch):
    killed, _, survived = _run_stop(monkeypatch, _ENTRY,
                                    pid_alive=True,
                                    identity_ok=True, same_ns=True,
                                    members=[4242])
    assert killed == [4242]
    assert survived == {}


def test_alive_but_different_pid_never_killed(monkeypatch):
    """Round-36 C1: an alive-but-mismatched pid PROVES the number
    was recycled -- the group at that pgid belongs to a stranger."""
    killed, printed, survived = _run_stop(
        monkeypatch, _ENTRY, pid_alive=True, identity_ok=False,
        same_ns=True, members=[4242])
    assert killed == []
    assert any("nothing killed" in ln for ln in printed)
    assert survived == {}


def test_dead_leader_with_orphans_reports_not_kills(monkeypatch):
    """Round-36 C0/C1: the genuine orphan case is REPORTED with the
    manual reap command, never auto-killed."""
    killed, printed, survived = _run_stop(
        monkeypatch, _ENTRY, pid_alive=False, identity_ok=False,
        same_ns=True, members=[4300, 4301])
    assert killed == []
    assert any("NOT killed automatically" in ln for ln in printed)
    # The record SURVIVES so ensure()'s orphan gate stays armed
    # (round-38 C0: dropping it here let the next ensure spawn a
    # second daemon over the live orphan).
    assert "d" in survived


def test_cross_namespace_record_dropped_silently(monkeypatch):
    """Round-36 C0: a container restart resets the pid namespace
    (host boot id unchanged!); every recorded number is meaningless
    and nothing may be signalled."""
    killed, printed, survived = _run_stop(
        monkeypatch, _ENTRY, pid_alive=False, identity_ok=False,
        same_ns=False, members=[4242])
    assert killed == []
    assert any("nothing killed" in ln for ln in printed)
    assert survived == {}


def test_legacy_record_of_stranger_is_not_alive(monkeypatch):
    """Round-37 C0: a pre-identity record (committed shape) whose
    pid is alive but running a DIFFERENT command must count dead --
    the True fallback let stop killpg recycled numbers and let
    ensure refuse to respawn."""
    monkeypatch.setattr(ld, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(ld, "_cmdline_matches",
                        lambda pid, cmd: False)
    legacy = {"pid": 512, "pgid": 512,
              "cmd": ["python", "trainer.py"], "started": "t"}
    assert ld._entry_alive(legacy) is False


def test_legacy_record_matching_cmdline_is_alive(monkeypatch):
    monkeypatch.setattr(ld, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(ld, "_cmdline_matches",
                        lambda pid, cmd: cmd == ["python", "t.py"])
    legacy = {"pid": 512, "pgid": 512, "cmd": ["python", "t.py"],
              "started": "t"}
    assert ld._entry_alive(legacy) is True


def test_ensure_refuses_to_spawn_over_live_orphans():
    """Round-37 C1: decide_ensure returns 'orphan' when the leader
    is dead but same-namespace group members survive -- a second
    daemon would fight the orphan over the same files."""
    e = {"pid": 1, "pgid": 1}
    assert ld.decide_ensure(e, alive=False, orphans=True) == "orphan"
    assert ld.decide_ensure(e, alive=False, orphans=False) == "spawn"
    assert ld.decide_ensure(e, alive=True, orphans=False) == "keep"
    assert ld.decide_ensure(None, alive=False,
                            orphans=False) == "spawn"
