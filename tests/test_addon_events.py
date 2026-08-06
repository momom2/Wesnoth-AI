#!/usr/bin/env python3
"""Vendored add-on scenario events: [capture_village] + the Marshy
Fill turn-1 leader-MP chain ([store_unit]/[if]/[set_variable
sub=]/[modify_unit]) -- the 2026-08-06 whitelist-audit DISCUSS maps,
included on user order after sim support landed.

Production path throughout: WesnothSim(build_scenario_gamestate(...))
fires prestart+start through tools/scenario_events, exactly as replay
reconstruction does.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from tools.scenario_pool import ScenarioSetup, build_scenario_gamestate  # noqa: E402
from tools.wesnoth_sim import WesnothSim  # noqa: E402
from tools.scenario_events import (  # noqa: E402
    fire_event, load_events_for_scenario,
)


def _fresh_sim(sid, leader1="Elvish Captain", leader2="Elvish Captain"):
    setup = ScenarioSetup(
        scenario_id=sid,
        faction1="Rebels", leader1=leader1,
        faction2="Rebels", leader2=leader2)
    return WesnothSim(build_scenario_gamestate(setup),
                      scenario_id=sid, max_turns=10)


def test_cold_war_prestart_capture_village_ownership():
    """WL Cold War: prestart [capture_village] gives side 1 the
    village at WML (33,18) and side 2 those at (48,19) and (3,4).
    Without the handler these were silently dropped -> 1g/turn income
    drift (audit 2026-08-06)."""
    sim = _fresh_sim("WL_Cold_War")
    owner = getattr(sim.gs.global_info, "_village_owner", None) or {}
    assert owner.get((32, 17)) == 1, owner
    assert owner.get((47, 18)) == 2, owner
    assert owner.get((2, 3)) == 2, owner


def test_summer_frosts_prestart_capture_village():
    """WL Summer Frosts: the single [capture_village] side=2 (40,7)."""
    sim = _fresh_sim("WL_Summer_Frosts")
    owner = getattr(sim.gs.global_info, "_village_owner", None) or {}
    assert owner.get((39, 6)) == 2, owner


def test_marshy_fill_leader_mp_shave_6mp():
    """WL Marshy Fill start event: side-1 leader at WML (18,1) with
    5 <= max_moves <= 8 gets current moves = 9 - moves on turn 1.
    Elvish Captain (5 MP) -> 4. max_moves untouched (the event writes
    the CURRENT-moves attribute only; modify_unit.lua:14-17,41)."""
    sim = _fresh_sim("WL_Marshy_Fill")
    leader = next(u for u in sim.gs.map.units
                  if u.side == 1 and u.position.x == 17
                  and u.position.y == 0)
    assert leader.max_moves == 5
    assert leader.current_moves == 4, leader.current_moves
    # side 2's leader is NOT filtered by the event: untouched.
    l2 = next(u for u in sim.gs.map.units if u.side == 2)
    assert l2.current_moves == l2.max_moves


def test_marshy_fill_leader_mp_branches():
    """The [if] branches on a re-fired fresh event list: moves >= 9
    -> 0; moves <= 4 -> untouched (condition greater_than=4 fails)."""
    sim = _fresh_sim("WL_Marshy_Fill")
    leader = next(u for u in sim.gs.map.units
                  if u.side == 1 and u.position.x == 17
                  and u.position.y == 0)
    # >= 9 branch: else-arm of the inner [if] sets 0.
    leader.current_moves = 9
    fire_event(sim.gs, load_events_for_scenario("WL_Marshy_Fill"),
               "start")
    assert leader.current_moves == 0, leader.current_moves
    # <= 4 branch: outer [if] condition false, no [else] -> untouched.
    leader.current_moves = 4
    fire_event(sim.gs, load_events_for_scenario("WL_Marshy_Fill"),
               "start")
    assert leader.current_moves == 4, leader.current_moves


def test_vendored_seamless_variant_shares_event_logic():
    """The Seamless Marshy Fill-(R) vendored copy carries the same id
    (WL_Marshy_Fill) and identical event logic; by-id event loading
    resolves to ONE cfg -- the sim behavior must be the same shave."""
    events = load_events_for_scenario("WL_Marshy_Fill")
    assert any(ev.name == "start" for ev in events)
    names = {a.tag for ev in events for a in ev.actions}
    assert "store_unit" in names and "if" in names
