"""Rust reachability kernel certification (docs/rust_port_plan.md
phase 1): the wesnoth_core Dijkstra must be BIT-EXACT against the
Python implementation — mp/prev/landable identical, cost floats
equal by ==. Skips when the wheel isn't built (maturin develop
--release in rust/wesnoth_core); the flip of the production default
is gated on this file passing.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

wesnoth_core = pytest.importorskip("wesnoth_core")

from sim_test_helpers import fresh_scenario_sim  # noqa: E402
from tools import pathfind_sim as pf  # noqa: E402


def _both(unit, gs, ctx, budget=None):
    saved = pf._RUST
    try:
        pf._RUST = None
        py = pf.unit_reach(unit, gs, ctx, budget=budget)
        pf._RUST = wesnoth_core
        rs = pf.unit_reach(unit, gs, ctx, budget=budget)
    finally:
        pf._RUST = saved
    return py, rs


def _assert_identical(py, rs, tag):
    assert rs.start == py.start, tag
    assert rs.mp == py.mp, f"{tag}: mp differs"
    assert rs.prev == py.prev, f"{tag}: prev differs"
    assert rs.landable == py.landable, f"{tag}: landable differs"
    assert set(rs.cost) == set(py.cost), tag
    for p, c in py.cost.items():
        assert rs.cost[p] == c, (
            f"{tag}: cost at {p} differs: {rs.cost[p]!r} vs {c!r}")


def test_rust_matches_python_on_scenario_units():
    sim = fresh_scenario_sim()
    gs = sim.gs
    for u in list(gs.map.units):
        ctx = pf.ReachContext.for_side(gs, u.side, god_view=True)
        py, rs = _both(u, gs, ctx)
        _assert_identical(py, rs, f"unit {u.id}")


def test_rust_matches_python_under_fuzzed_contexts():
    """Random ZoC/enemy/ally flag patterns + budgets + skirmisher:
    the flag interactions (ZoC drain, ally subcost, enemy walls)
    are where an off-by-one would hide."""
    sim = fresh_scenario_sim()
    gs = sim.gs
    units = list(gs.map.units)
    hexes = [(h.position.x, h.position.y) for h in gs.map.hexes]
    rng = random.Random(1234)
    for case in range(60):
        u = rng.choice(units)
        ctx = pf.ReachContext.for_side(gs, u.side, god_view=True)
        for _ in range(rng.randrange(0, 12)):
            ctx.zoc_hexes.add(rng.choice(hexes))
        for _ in range(rng.randrange(0, 6)):
            ctx.enemy_hexes.add(rng.choice(hexes))
        for _ in range(rng.randrange(0, 6)):
            ctx.ally_hexes.add(rng.choice(hexes))
        budget = rng.randrange(0, 14)
        skirm = rng.random() < 0.3
        saved_ab = u.abilities
        try:
            if skirm:
                u.abilities = (set(saved_ab) if saved_ab
                               else set()) | {"skirmisher"}
            py, rs = _both(u, gs, ctx, budget=budget)
        finally:
            u.abilities = saved_ab
        _assert_identical(py, rs, f"fuzz case {case}")


def test_rust_kernel_rejects_bad_shapes():
    import numpy as np
    with pytest.raises(ValueError):
        wesnoth_core.unit_reach_arrays(
            np.zeros(5, dtype=np.int64),      # not H*6
            np.zeros(2, dtype=np.int64),
            np.zeros(2, dtype=np.int64),
            np.zeros(2, dtype=np.uint8),
            np.zeros(2, dtype=np.uint8),
            np.zeros(2, dtype=np.uint8),
            0, 5, False)
