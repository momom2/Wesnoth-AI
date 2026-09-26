"""The kernel report must answer CAPABILITY, not importability.

A wheel that imports is not a wheel that serves every kernel: each
kernel needs the wheel phase that gave it its current contract, so a
wheel several phases behind the source imports cleanly while the
kernels it is too old for fall back to Python (measured 2026-09-13: a
phase-3 wheel imported and four of the five kernels then gated were
Python).

Dependencies: tools.kernel_status
Dependents:   pytest only
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import kernel_status as ks  # noqa: E402

EXPECTED = {"reach", "enumeration", "observation", "rows_from_reach",
            "combat", "encode_streams", "GameCore"}


def test_every_kernel_is_reported_and_the_answer_is_a_bool():
    st = ks.kernel_status()
    assert set(st) == EXPECTED, st
    assert all(isinstance(v, bool) for v in st.values()), st


def test_a_gate_that_raises_counts_as_python():
    """Production sees a missing kernel as a fallback, not a crash, so
    the report must too -- otherwise one broken gate hides the rest."""
    def _boom():
        raise RuntimeError("no kernel here")

    saved = dict(ks._GATES)
    try:
        ks._GATES["combat"] = _boom
        assert ks.kernel_status()["combat"] is False
    finally:
        ks._GATES.clear()
        ks._GATES.update(saved)


def test_a_wheel_too_old_for_one_kernel_still_serves_the_others(monkeypatch):
    """The reach and enumeration kernels are gated on their own phases
    and the report follows the gates: a wheel one phase short of the
    enumeration's contract is refused for it and still serves the reach
    Dijkstra."""
    from tools import pathfind_sim as pf

    class _Wheel:
        __phase__ = pf.ENUMERATE_KERNEL_PHASE - 1

        @staticmethod
        def unit_reach_arrays(*a, **k):
            raise AssertionError("not called here")

        @staticmethod
        def enumerate_moves(*a, **k):
            raise AssertionError("a wheel too old for the kernel must not serve it")

    monkeypatch.setattr(pf, "_RUST", _Wheel)
    assert pf.reach_kernel() is not None and pf.enumerate_kernel() is None
    st = ks.kernel_status()
    assert st["reach"] is True and st["enumeration"] is False
    monkeypatch.setattr(_Wheel, "__phase__", pf.ENUMERATE_KERNEL_PHASE)
    assert pf.enumerate_kernel() is not None


def test_the_banner_names_both_sides_and_the_phase_gap():
    saved = dict(ks._GATES)
    try:
        ks._GATES["reach"] = lambda: True
        for k in EXPECTED - {"reach"}:
            ks._GATES[k] = lambda: False
        line = ks.banner()
    finally:
        ks._GATES.clear()
        ks._GATES.update(saved)

    assert "RUST: reach |" in line, line
    for k in EXPECTED - {"reach"}:
        assert k in line.split("PYTHON:")[1], f"{k} missing from {line}"
    # It must never read as a clean "RUST" when six kernels are not.
    assert "PYTHON:" in line and "none" not in line.split("PYTHON:")[1]


def test_the_source_phase_is_readable():
    """The banner's phase comparison goes quiet if this returns None,
    so it is worth pinning that the parse works against the real file."""
    want = ks.source_phase()
    assert want is not None and want >= 1, want
