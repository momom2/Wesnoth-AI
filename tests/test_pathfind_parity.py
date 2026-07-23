#!/usr/bin/env python3
"""Bit-exact parity of the array-index `unit_reach` (2026-07-22)
against the reference tuple-dict implementation it replaced.

The route planner is the shared waist between the legality mask and
the sim's order execution (user contract 2026-07-17), and its float
tie-break costs decide PREFERRED ROUTES that flow into exported
replays -- so the port must reproduce mp/cost/prev/landable exactly
(floats compared with ==, not approx) or not ship at all.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from tools.pathfind_sim import (ReachContext, unit_reach,
                                _unit_reach_reference)
from wesnoth_ai.visibility import is_scenery_unit
from sim_test_helpers import fresh_scenario_sim

CORPUS = Path("replays_dataset")


def _assert_reach_equal(a, b, label):
    assert a.start == b.start, label
    assert a.mp == b.mp, f"{label}: mp differs"
    assert a.cost == b.cost, f"{label}: cost differs (float-exact)"
    assert a.prev == b.prev, f"{label}: preferred routes differ"
    assert a.landable == b.landable, f"{label}: landable differs"


def _check_state(gs, label):
    n = 0
    for side in (1, 2):
        ctx = ReachContext.for_side(gs, side)
        for u in list(gs.map.units):
            if u.side != side or is_scenery_unit(u):
                continue
            ref = _unit_reach_reference(u, gs, ctx)
            got = unit_reach(u, gs, ctx)
            _assert_reach_equal(got, ref, f"{label}/{u.id}")
            # Budget override path too (used by opener tooling).
            ref2 = _unit_reach_reference(u, gs, ctx, budget=3)
            got2 = unit_reach(u, gs, ctx, budget=3)
            _assert_reach_equal(got2, ref2, f"{label}/{u.id}/b3")
            n += 1
    return n


def test_parity_on_fresh_scenarios():
    total = 0
    for seed in range(4):
        sim = fresh_scenario_sim(seed=30 + seed, max_turns=12,
                                 mini=(seed % 2 == 0))
        total += _check_state(sim.gs, f"fresh{seed}")
    assert total > 0


def test_parity_with_skirmisher():
    """Exercise the skirmisher branch (ZoC pass-through) explicitly:
    grant the ability to a copy of a real unit."""
    sim = fresh_scenario_sim(seed=77, max_turns=12, mini=True)
    gs = sim.gs
    u = next(x for x in gs.map.units if x.side == 1
             and not is_scenery_unit(x))
    u.abilities = set(u.abilities or set()) | {"skirmisher"}
    ctx = ReachContext.for_side(gs, 1)
    _assert_reach_equal(unit_reach(u, gs, ctx),
                        _unit_reach_reference(u, gs, ctx), "skirm")


@pytest.mark.skipif(not CORPUS.exists(),
                    reason="replays_dataset not present")
def test_parity_on_midgame_corpus_states():
    from tools.midgame_starts import sample_midgame_start
    from tools.wesnoth_sim import WesnothSim
    rng = random.Random(20260722)
    total = 0
    for k in range(3):
        mg = sample_midgame_start(rng, CORPUS)
        if mg is None:
            continue
        gs, scen_id = mg[0], mg[1]
        sim = WesnothSim(gs, scenario_id=scen_id, max_turns=40)
        total += _check_state(sim.gs, f"midgame{k}")
    assert total > 0, "no corpus states sampled"
