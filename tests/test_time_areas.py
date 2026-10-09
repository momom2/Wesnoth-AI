#!/usr/bin/env python3
"""[time_area] zones on the FRESH-BUILD path (2026-07-15).

Tombs of Kesorak and Elensefar Courtyard define per-hex ToD
overrides (dark tombs / underground keeps). Reconstruction applies
them in the core's scenario setup (`CoreState.setup_scenario`,
verified via replay parity); this pins that a fresh self-play sim
gets the same zones -- they attach at `WesnothSim.__init__` through
the same setup, NOT at bare `build_scenario_gamestate` (probing the
wrong layer briefly looked like a missing-zones bug).

Also pins that the global start-slot scan (`_scenario_tod_info`)
ignoring [time_area] blocks does not disturb the zones themselves.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from wesnoth_ai.combat import TOD_DEFAULT_CYCLE
from wesnoth_ai.rules.scenario_pool import ScenarioSetup, build_scenario_gamestate
from tools.wesnoth_sim import WesnothSim


def _board_bonus(turn: int, start_slot: int) -> int:
    """The default cycle's lawful bonus on `turn` when turn 1 is slot
    `start_slot` (the engine's modulo)."""
    return TOD_DEFAULT_CYCLE[(max(1, turn) - 1 + start_slot) % len(TOD_DEFAULT_CYCLE)][1]


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
    dawn_sim = _fresh_sim("multiplayer_Tombs_of_Kesorak", tod_start=0)
    dawn, afternoon = dawn_sim.core.core, _fresh_sim("multiplayer_Tombs_of_Kesorak", tod_start=2).core.core
    areas = dawn_sim.gs.global_info._time_areas
    turns = range(1, 7)
    varying = 0
    for x, y in areas:
        at_dawn = [dawn.lawful_bonus(x, y, t) for t in turns]
        assert [afternoon.lawful_bonus(x, y, t) for t in turns] == at_dawn, (x, y)
        varying += len(set(at_dawn)) > 1
    assert varying >= 4, "the areas must change with the turn for this to test anything"
    plain = next((h.position.x, h.position.y) for h in dawn_sim.gs.map.hexes
                 if (h.position.x, h.position.y) not in areas
                 and [dawn.lawful_bonus(h.position.x, h.position.y, t) for t in turns]
                 == [_board_bonus(t, 0) for t in turns])
    assert [afternoon.lawful_bonus(*plain, t) for t in turns] == [_board_bonus(t, 2) for t in turns]


def test_an_area_placed_later_starts_from_its_own_current_time():
    """`add_time_area` sets the area's slot to its current_time on the
    turn it is placed, so at turn t it reads slot
    (current_time + t - placed) mod len, whatever the board's slot."""
    from tools.replay_extract import parse_wml
    from tools.scenario_events import collect_events
    from wesnoth_ai.game_core import CoreState, _event_actions
    gs = _fresh_sim("multiplayer_Tombs_of_Kesorak", tod_start=3).gs
    gs.global_info.turn_number = 4
    core = CoreState.from_state(gs).core
    areas = {(x, y) for x, y, _cycle in core.time_areas_export()}
    x, y = next((h.position.x, h.position.y) for h in sorted(gs.map.hexes, key=lambda h: (h.position.y, h.position.x))
                if (h.position.x, h.position.y) not in areas
                and [core.lawful_bonus(h.position.x, h.position.y, t) for t in range(1, 7)]
                == [_board_bonus(t, 3) for t in range(1, 7)])
    declared = [-20, -10, 0, 10, 20, 5]
    times = "".join(f"[time]\nlawful_bonus={v}\n[/time]\n" for v in declared)
    events = collect_events(parse_wml(
        f"[multiplayer]\n[event]\nname=probe\n[time_area]\nx={x + 1}\ny={y + 1}\ncurrent_time=2\n"
        f"{times}[/time_area]\n[/event]\n[/multiplayer]\n"), "probe")
    _fired, wml, stored = core.events_export()
    core.load_events([(ev.name, bool(ev.first_time_only), _event_actions(ev), ev.scenario_id, False)
                      for ev in events], wml, stored)
    core.fire_events(["probe"])
    assert [core.lawful_bonus(x, y, t) for t in range(4, 13)] == \
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
