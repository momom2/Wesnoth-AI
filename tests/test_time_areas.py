#!/usr/bin/env python3
"""[time_area] zones on the FRESH-BUILD path (2026-07-15).

Tombs of Kesorak and Elensefar Courtyard define per-hex ToD
overrides (dark tombs / underground keeps). Reconstruction has
long applied them (`setup_static_time_areas`, verified via replay
parity); this pins that a fresh self-play sim gets the same zones
-- they attach at `WesnothSim.__init__` via
`_setup_scenario_events`, NOT at bare `build_scenario_gamestate`
(probing the wrong layer briefly looked like a missing-zones bug).

Also pins that the global start-slot scan (`_scenario_tod_info`)
ignoring [time_area] blocks does not disturb the zones themselves.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from tools.scenario_pool import ScenarioSetup, build_scenario_gamestate
from wesnoth_sim import WesnothSim


def _fresh_sim(sid):
    setup = ScenarioSetup(
        scenario_id=sid,
        faction1="Knalgan Alliance", leader1="Dwarvish Steelclad",
        faction2="Rebels", leader2="Elvish Captain")
    return WesnothSim(build_scenario_gamestate(setup),
                      scenario_id=sid, max_turns=10)


def test_tombs_of_kesorak_fresh_build_has_three_zones():
    sim = _fresh_sim("multiplayer_Tombs_of_Kesorak")
    areas = getattr(sim.gs.global_info, "_time_areas", None) or {}
    assert len(areas) == 9, f"expected 9 zone hexes, got {len(areas)}"
    assert len({tuple(c) for c in areas.values()}) == 3, \
        "expected 3 distinct zone cycles"


def test_elensefar_courtyard_fresh_build_has_underground_area():
    sim = _fresh_sim("multiplayer_elensefar_courtyard")
    areas = getattr(sim.gs.global_info, "_time_areas", None) or {}
    assert len(areas) == 220, f"expected 220 zone hexes, got {len(areas)}"


def test_global_tod_scan_leaves_zones_intact():
    """The start-slot reader excludes [time_area] blocks (engine
    reads current_time/random_start_time as top-level attrs only)
    -- but the zones must still land on the sim."""
    from tools.scenario_pool import _scenario_tod_info
    ct, rand, n = _scenario_tod_info("multiplayer_Tombs_of_Kesorak")
    assert ct is None and rand is False and n == 6
    sim = _fresh_sim("multiplayer_Tombs_of_Kesorak")
    assert getattr(sim.gs.global_info, "_time_areas", None), \
        "zones must attach regardless of the global scan"
