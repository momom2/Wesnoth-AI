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

from wesnoth_ai.rules.scenario_pool import ScenarioSetup, build_scenario_gamestate
from tools.wesnoth_sim import WesnothSim


def _fresh_sim(sid, tod_start=None):
    setup = ScenarioSetup(
        scenario_id=sid,
        faction1="Knalgan Alliance", leader1="Dwarvish Steelclad",
        faction2="Rebels", leader2="Elvish Captain", tod_start=tod_start)
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


def test_an_area_keeps_its_own_slot_when_the_board_starts_elsewhere():
    """The engine starts each [time_area] at its own current_time (default
    0), and a random start moves only the board's slot (`resolve_random`,
    src/tod_manager.cpp, 1.18.4). Tombs of Kesorak's areas therefore read
    the same on every turn whether the game starts at dawn or at
    afternoon, while the board itself shifts by two slots."""
    from tools.replay_dataset import _lawful_bonus_at, _lawful_bonus_for_turn
    dawn = _fresh_sim("multiplayer_Tombs_of_Kesorak", tod_start=0).gs
    afternoon = _fresh_sim("multiplayer_Tombs_of_Kesorak", tod_start=2).gs
    areas = dawn.global_info._time_areas
    turns = range(1, 7)
    varying = 0
    for x, y in areas:
        at_dawn = [_lawful_bonus_at(dawn, x, y, t) for t in turns]
        assert [_lawful_bonus_at(afternoon, x, y, t) for t in turns] == at_dawn, (x, y)
        varying += len(set(at_dawn)) > 1
    assert varying >= 4, "the areas must change with the turn for this to test anything"
    plain = next((h.position.x, h.position.y) for h in dawn.map.hexes
                 if (h.position.x, h.position.y) not in areas
                 and [_lawful_bonus_at(dawn, h.position.x, h.position.y, t) for t in turns]
                 == [_lawful_bonus_for_turn(t, 0) for t in turns])
    assert [_lawful_bonus_at(afternoon, *plain, t) for t in turns] == \
        [_lawful_bonus_for_turn(t, 2) for t in turns]


def test_an_area_placed_later_starts_from_its_own_current_time():
    """`add_time_area` sets the area's slot to its current_time on the
    turn it is placed, so at turn t it reads slot
    (current_time + t - placed) mod len, whatever the board's slot."""
    from tools.replay_dataset import _lawful_bonus_at, _lawful_bonus_for_turn
    from tools.replay_extract import parse_wml
    from tools.scenario_events import _time_area_action
    gs = _fresh_sim("multiplayer_Tombs_of_Kesorak", tod_start=3).gs
    areas = gs.global_info._time_areas
    x, y = next((h.position.x, h.position.y) for h in gs.map.hexes
                if (h.position.x, h.position.y) not in areas
                and [_lawful_bonus_at(gs, h.position.x, h.position.y, t) for t in range(1, 7)]
                == [_lawful_bonus_for_turn(t, 3) for t in range(1, 7)])
    declared = [-20, -10, 0, 10, 20, 5]
    times = "".join(f"[time]\nlawful_bonus={v}\n[/time]\n" for v in declared)
    node = parse_wml(f"[time_area]\nx={x + 1}\ny={y + 1}\ncurrent_time=2\n"
                     f"{times}[/time_area]\n").first("time_area")
    gs.global_info.turn_number = 4
    _time_area_action(gs, node)
    assert [_lawful_bonus_at(gs, x, y, t) for t in range(4, 13)] == \
        [declared[(2 + t - 4) % len(declared)] for t in range(4, 13)]


def test_global_tod_scan_leaves_zones_intact():
    """The start-slot reader excludes [time_area] blocks (engine
    reads current_time/random_start_time as top-level attrs only)
    -- but the zones must still land on the sim."""
    from wesnoth_ai.rules.scenario_pool import _scenario_tod_info
    ct, rand, n = _scenario_tod_info("multiplayer_Tombs_of_Kesorak")
    assert ct is None and rand is False and n == 6
    sim = _fresh_sim("multiplayer_Tombs_of_Kesorak")
    assert getattr(sim.gs.global_info, "_time_areas", None), \
        "zones must attach regardless of the global scan"
