"""The wheel report must answer CAPABILITY, not importability.

A wheel that imports is not a wheel that serves the source: every rule
runs in the Rust core, so a wheel older than the adapter's phase stops
the simulator, and one behind the source tests old Rust (measured
2026-09-13: a phase-3 wheel imported while the source declared 9).

Dependencies: tools.kernel_status
Dependents:   pytest only
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import kernel_status as ks  # noqa: E402

EXPECTED = {"GameCore", "wheel_current"}


def test_every_check_is_reported_and_the_answer_is_a_bool():
    st = ks.kernel_status()
    assert set(st) == EXPECTED, st
    assert all(isinstance(v, bool) for v in st.values()), st


def test_a_check_that_raises_counts_as_failed():
    """One broken check must not hide the others."""
    def _boom():
        raise RuntimeError("no wheel here")

    saved = dict(ks._GATES)
    try:
        ks._GATES["GameCore"] = _boom
        st = ks.kernel_status()
        assert st["GameCore"] is False and set(st) == EXPECTED
    finally:
        ks._GATES.clear()
        ks._GATES.update(saved)


def test_the_banner_says_rebuild_for_a_wheel_behind_the_source(monkeypatch):
    monkeypatch.setattr(ks, "wheel_phase", lambda: 5)
    monkeypatch.setattr(ks, "source_phase", lambda: 9)
    saved = dict(ks._GATES)
    try:
        ks._GATES["GameCore"] = lambda: False
        line = ks.banner()
    finally:
        ks._GATES.clear()
        ks._GATES.update(saved)
    assert "phase 5" in line and "source 9" in line and "REBUILD" in line, line
    monkeypatch.setattr(ks, "wheel_phase", lambda: None)
    assert "NOT INSTALLED" in ks.banner()


def test_the_source_phase_is_readable():
    """The banner's phase comparison goes quiet if this returns None,
    so it is worth pinning that the parse works against the real file."""
    want = ks.source_phase()
    assert want is not None and want >= 1, want


def test_a_wheel_of_the_sources_phase_built_from_another_source_is_not_current(monkeypatch):
    """Two changes have landed within one phase number: the phase alone let
    a stale wheel pass. The wheel is current only when it carries the hash
    of the source in front of it."""
    monkeypatch.setattr(ks, "wheel_phase", lambda: 35)
    monkeypatch.setattr(ks, "source_phase", lambda: 35)
    monkeypatch.setattr(ks, "source_hash", lambda: "a" * 64)
    saved = dict(ks._GATES)
    try:
        ks._GATES["GameCore"] = lambda: True
        for wheel, current in (("b" * 64, False), (None, False), ("a" * 64, True)):
            monkeypatch.setattr(ks, "wheel_source_hash", lambda w=wheel: w)
            assert ks.kernel_status()["wheel_current"] is current, wheel
            line = ks.banner()
            assert ("REBUILD" not in line) == current and line.endswith("current") == current, line
    finally:
        ks._GATES.clear()
        ks._GATES.update(saved)
