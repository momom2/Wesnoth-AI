"""The kernel report must answer CAPABILITY, not importability.

`pathfind_sim.rust_active()` is "did `import wesnoth_core` succeed",
and a launcher bannered "RUST (wesnoth_core)" off it. The wheel exposes
its kernels by PHASE, so a wheel several phases behind the source
imports cleanly while observe, combat and GameCore fall back to Python:
measured 2026-09-13, `rust_active()` was True and four of the five
kernels were Python.

Dependencies: tools.kernel_status
Dependents:   pytest only
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import kernel_status as ks  # noqa: E402

EXPECTED = {"reach/enumeration", "observation", "rows_from_reach",
            "combat", "GameCore"}


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


def test_the_banner_names_both_sides_and_the_phase_gap():
    saved = dict(ks._GATES)
    try:
        ks._GATES["reach/enumeration"] = lambda: True
        for k in EXPECTED - {"reach/enumeration"}:
            ks._GATES[k] = lambda: False
        line = ks.banner()
    finally:
        ks._GATES.clear()
        ks._GATES.update(saved)

    assert "RUST: reach/enumeration" in line, line
    for k in EXPECTED - {"reach/enumeration"}:
        assert k in line.split("PYTHON:")[1], f"{k} missing from {line}"
    # It must never read as a clean "RUST" when four kernels are not.
    assert "PYTHON:" in line and "none" not in line.split("PYTHON:")[1]


def test_the_source_phase_is_readable():
    """The banner's phase comparison goes quiet if this returns None,
    so it is worth pinning that the parse works against the real file."""
    want = ks.source_phase()
    assert want is not None and want >= 1, want
