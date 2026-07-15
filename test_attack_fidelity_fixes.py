#!/usr/bin/env python3
"""Decisive-game fidelity fixes (validation pipeline catches,
2026-07-15).

The first live training run's strict-sync exports failed on
decisive games in two clusters:

1. "found corrupt movement in replay" -- our sim let units MOVE
   after attacking. Engine rule: attack.cpp:1371-1372 subtracts
   attack_type::movement_used() (default: everything) from the
   attacker after combat; no default-era weapon overrides it.
   Human-replay parity never caught this because the UI forbids
   the attempt, so no human replay contains one.

2. "expecting a user choice" -- the export's advancement [choose]
   detection keyed on unit NAME CHANGE, which misses AMLA (name
   unchanged) and undercounts double-advance chains. Now every
   advancement step is recorded at the source
   (replay_dataset._record_advance_event side-channel) with its
   actual choice index.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

from classes import Position
from tools.abilities import hex_neighbors
from tools.scenario_pool import ScenarioSetup, build_scenario_gamestate
from wesnoth_sim import WesnothSim


def _sim():
    setup = ScenarioSetup(
        scenario_id="enclave_micro_isar",
        faction1="Knalgan Alliance", leader1="Dwarvish Steelclad",
        faction2="Rebels", leader2="Elvish Captain",
        fogless=True)     # parked enemies must be visible to attack
    return WesnothSim(build_scenario_gamestate(setup),
                      scenario_id="enclave_micro_isar", max_turns=10)


def _park_enemies_adjacent(sim):
    """Move side-2's leader adjacent to side-1's leader; strip its
    attacks so combat outcome is deterministic-ish and the defender
    survives."""
    l1 = next(u for u in sim.gs.map.units if u.side == 1 and u.is_leader)
    l2 = next(u for u in sim.gs.map.units if u.side == 2 and u.is_leader)
    occupied = {(u.position.x, u.position.y) for u in sim.gs.map.units}
    nb = next(p for p in hex_neighbors(l1.position.x, l1.position.y)
              if p not in occupied)
    l2.position = Position(*nb)
    l2.attacks.clear()
    l2.current_hp = l2.max_hp
    return l1, l2


def test_attack_zeroes_attacker_movement():
    sim = _sim()
    l1, l2 = _park_enemies_adjacent(sim)
    assert l1.current_moves > 0
    sim.step({"type": "attack",
              "start_hex": l1.position,
              "target_hex": l2.position,
              "attack_index": 0})     # returns game-over, not success
    att = next(u for u in sim.gs.map.units
               if u.side == 1 and u.is_leader)
    assert att.has_attacked, "attack was not applied"
    assert att.current_moves == 0, \
        f"attacker must have 0 MP after attacking " \
        f"(got {att.current_moves})"


def test_amla_advancement_emits_choose_event():
    """A max-level attacker crossing its XP threshold AMLAs
    (+3 max_hp, XP carryover) AND the recorded attack must carry an
    advance_choices event so the export emits the dependent
    [choose] Wesnoth expects."""
    sim = _sim()
    l1, l2 = _park_enemies_adjacent(sim)
    # Rebuild side-1's leader as a max-level Dwarvish Lord one XP
    # short of AMLA. Fighting a level-2 defender yields >= 2 XP
    # even without a kill.
    sim.gs.map.units.discard(l1)
    lord = replace(l1, name="Dwarvish Lord")
    setattr(lord, "_defense_table", getattr(l1, "_defense_table", None))
    sim.gs.map.units.add(lord)
    lord.current_exp = lord.max_exp - 1
    pre_max_hp = lord.max_hp
    sim.step({"type": "attack",
              "start_hex": lord.position,
              "target_hex": l2.position,
              "attack_index": 0})     # returns game-over, not success
    att = next(u for u in sim.gs.map.units
               if u.side == 1 and u.is_leader)
    assert att.name == "Dwarvish Lord"          # AMLA: no type change
    assert att.max_hp == pre_max_hp + 3, "AMLA grants +3 max_hp"
    rc = next(r for r in reversed(sim.command_history)
              if r.kind == "attack")
    assert rc.extras.get("advance_choices") == [(1, 0)], \
        f"AMLA must record a [choose] event " \
        f"(got {rc.extras.get('advance_choices')})"
    # And the WML emitter turns it into a dependent [choose] block.
    from tools.sim_to_replay import _build_replay_wml
    wml = _build_replay_wml([rc])
    assert "[choose]" in wml and 'dependent="yes"' in wml
