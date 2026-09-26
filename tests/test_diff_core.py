"""tools/diff_core: one replay's failure, a Rust panic included, is that
replay's result and the sweep goes on.

pyo3 raises a Rust panic as `PanicException`, a BaseException, so the
sweep's `except Exception` let it through and one panicking replay ended
a corpus sweep. Stand-ins replace the replay loader and the core here,
so these run without a wheel.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import diff_core  # noqa: E402


class PanicException(BaseException):
    """The shape of pyo3's panic: named so, and not an Exception."""


def test_a_panicking_replay_is_listed_and_the_sweep_goes_on(monkeypatch, capsys):
    def fake(gz, **_kw):
        if gz.name == "a.json.gz":
            raise PanicException("index out of bounds")
        return []

    monkeypatch.setattr(diff_core, "diff_core", fake)
    rc = diff_core.main(["a.json.gz", "b.json.gz"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "2 replays, 1 clean, 1 with divergences" in out
    assert "a.json.gz: harness panicked PanicException('index out of bounds')" in out


def test_a_panic_inside_a_command_is_a_divergence(monkeypatch, tmp_path):
    import tools.replay_dataset as rd
    import wesnoth_ai.game_core as gc

    class Core:
        @classmethod
        def from_state(cls, _gs):
            return cls()

        def apply_command(self, cmd):
            if cmd[0] == "attack":
                raise PanicException("attempt to subtract with overflow")
            return "rust"

        def to_state(self):
            return object()

    monkeypatch.setattr(rd, "_build_initial_gamestate", lambda data: object())
    monkeypatch.setattr(rd, "_setup_scenario_events", lambda gs, sid: None)
    monkeypatch.setattr(rd, "_apply_command", lambda gs, cmd: None)
    monkeypatch.setattr(gc, "CoreState", Core)
    monkeypatch.setattr(gc, "state_differences", lambda a, b, stash=True: [])
    replay = tmp_path / "x.json.gz"
    with gzip.open(replay, "wt", encoding="utf-8") as f:
        json.dump({"commands": [["init_side", 1], ["attack"], ["end_turn"]]}, f)
    assert diff_core.diff_core(replay) == [
        "x.json.gz#1 attack: core panicked PanicException('attempt to subtract with overflow')"]


def test_an_interrupt_still_stops_the_sweep(monkeypatch):
    def fake(gz, **_kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(diff_core, "diff_core", fake)
    with pytest.raises(KeyboardInterrupt):
        diff_core.main(["a.json.gz"])
