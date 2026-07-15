#!/usr/bin/env python3
"""Mini_Maps_Collection tentacle spawns (2026-07-14).

Three parser bugs silently emptied the enclave maps' side 3 (the
game trained on emptier boards than real Wesnoth): add-on utility
macros never loaded, parenthesized preprocessor args were split on
whitespace, and parallel-assign expansion ran before macro
substitution. Pins the exact spawns per map (WML 1-indexed cfg ->
0-indexed sim).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

from tools.scenario_pool import ScenarioSetup, build_scenario_gamestate
from wesnoth_sim import WesnothSim

# scenario id -> expected 0-indexed side-3 tentacle positions
EXPECTED_SIDE3 = {
    "enclave_micro_isar": {(1, 1), (3, 2), (5, 3)},
    "enclave_mini_fallenstar_1v1": {(5, 2), (5, 6), (5, 9)},
    "enclave_small_fallenstar_1v1": {(9, 2), (9, 6), (9, 9)},
    "2p_mini": {(3, 1), (2, 2)},
}


def _sim(sid):
    setup = ScenarioSetup(
        scenario_id=sid,
        faction1="Knalgan Alliance", leader1="Dwarvish Steelclad",
        faction2="Rebels", leader2="Elvish Captain")
    gs = build_scenario_gamestate(setup)
    return WesnothSim(gs, scenario_id=sid, max_turns=10)


def test_side3_tentacles_spawn_at_exact_positions():
    for sid, expected in EXPECTED_SIDE3.items():
        sim = _sim(sid)
        s3 = {(u.position.x, u.position.y)
              for u in sim.gs.map.units if u.side == 3}
        names = {u.name for u in sim.gs.map.units if u.side == 3}
        assert s3 == expected, f"{sid}: side-3 at {s3} != {expected}"
        assert names == {"Tentacle of the Deep"}, f"{sid}: {names}"


def test_enclave_tentacles_carry_loyal():
    sim = _sim("enclave_micro_isar")
    for u in sim.gs.map.units:
        if u.side == 3:
            assert "loyal" in u.traits


def test_around_mini_tentacle_is_a_side2_player_unit():
    sim = _sim("around_mini")
    assert not any(u.side == 3 for u in sim.gs.map.units)
    s2 = [u for u in sim.gs.map.units
          if u.side == 2 and u.name == "Tentacle of the Deep"]
    assert len(s2) == 1
    assert (s2[0].position.x, s2[0].position.y) == (4, 5)
