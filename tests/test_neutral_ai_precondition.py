"""The neutral AI's substitution rests on a precondition; here it is,
checked (2026-09-22).

Wesnoth drives neutral sides with the full RCA AI. We substitute a
combat-only algorithm (`tools/neutral_ai.py`, user-approved scope
2026-07-14), and that is EXACT only for a unit the real AI would not
move: for a stationary attacker the default AI reduces to its combat
candidate action, and the exposure term is zero by construction.

A unit qualifies for exactly three reasons, all now read rather than
assumed:

  * `ai_special=guardian`. It sets STATE_GUARDIAN (1.18.4
    `src/units/unit.cpp:659`), and the move phase then hands the unit
    a move from its own hex to its own hex -- "is guardian, staying
    still" (`src/ai/default/ca_move_to_targets.cpp:269-277`).
  * No movement left, because the scenario pins it every
    `turn refresh` ({MODIFY_UNIT (role=monster) moves 0}).
  * No landable hex: terrain-locked.

Until 2026-09-22 `ai_special` was read by nothing, and `neutral_ai`'s
docstring justified the substitution by the other two reasons only --
which do not cover `Modified_Tiny_Close_Relation`, whose Tentacle has
full MP and two adjacent water hexes it can enter. The substitution
was right there for a reason nobody had written down.

The enclave pin is also the precondition the scenario-build plan flags
as at risk: it comes from a macro our expander reduces to
`[modify_unit]`. Read the engine's own expansion instead and it
becomes `[store_unit kill=yes]` / `[foreach]` / `[unstore_unit]`,
none of which we run, so those units would silently become mobile.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import tools.neutral_ai as neutral_ai  # noqa: E402
from tools.pathfind_sim import ReachContext, unit_reach  # noqa: E402
from wesnoth_ai.rules.scenario_pool import (LADDER_SCENARIO_IDS,  # noqa: E402
                                            MINI_MAP_SCENARIO_IDS, ScenarioSetup,
                                            build_scenario_gamestate)
from tools.wesnoth_sim import WesnothSim  # noqa: E402

POOL = list(LADDER_SCENARIO_IDS) + list(MINI_MAP_SCENARIO_IDS)
# Measured 2026-09-22: the pool scenarios with a neutral side the
# engine actually gives a turn to (controller != null), and why each
# one's units stay put.
EXPECTED_ACTORS = {
    "2p_mini": "guardian",
    "2p_mini_edited": "guardian",
    "Modified_Tiny_Close_Relation": "guardian",
    "enclave_micro_isar": "pinned",
    "enclave_mini_fallenstar_1v1": "pinned",
    "enclave_small_fallenstar_1v1": "pinned",
}


def _sim(scenario_id: str) -> WesnothSim:
    gs = build_scenario_gamestate(ScenarioSetup(
        scenario_id=scenario_id, faction1="Rebels",
        leader1="Elvish Captain", faction2="Loyalists",
        leader2="Lieutenant", fogless=False, tod_start=None))
    return WesnothSim(gs, scenario_id=scenario_id)


@pytest.fixture(scope="module")
def actor_scenarios():
    return {scenario_id: sim for scenario_id, sim in
            ((s, _sim(s)) for s in POOL)
            if getattr(sim.gs.global_info, "_neutral_actor_sides", None)}


def test_the_set_of_acting_neutral_scenarios_is_the_known_one(actor_scenarios):
    """A new one means a neutral unit this AI has never been scoped
    to."""
    assert set(actor_scenarios) == set(EXPECTED_ACTORS)


def test_every_acting_neutral_unit_would_stay_put(actor_scenarios):
    """The precondition, through the production check itself, so the
    test and the runtime agree by construction."""
    for scenario_id, sim in sorted(actor_scenarios.items()):
        assert neutral_ai._check_units_are_stationary(
            sim.gs, 3, scenario_id), scenario_id


def test_each_scenario_stays_put_for_the_recorded_reason(actor_scenarios):
    """Not just THAT they stay put but WHY, because the reasons have
    different failure modes: a guardian flag is lost by an unread
    attribute, a pin by an unrun event, a terrain lock by a map edit."""
    for scenario_id, reason in sorted(EXPECTED_ACTORS.items()):
        gs = actor_scenarios[scenario_id].gs
        units = [u for u in gs.map.units if u.side >= 3]
        assert units, scenario_id
        if reason == "guardian":
            assert all(getattr(u, "_ai_guardian", False) for u in units), (
                f"{scenario_id}: ai_special=guardian is no longer read, so "
                f"the reason these units stay put is gone")
        elif reason == "pinned":
            assert all(u.current_moves == 0 for u in units), (
                f"{scenario_id}: the turn-refresh MP pin is not applied")
        else:                                       # terrain
            ctx = ReachContext.for_side(gs, 3)
            assert all(not unit_reach(u, gs, ctx).landable for u in units)


def test_a_mobile_neutral_unit_is_refused_under_strict(actor_scenarios,
                                                       monkeypatch):
    """The detector has to fire, or it is decoration. Strip the
    guardian flag from a map that depends on it and the check must
    fail."""
    sim = _sim("Modified_Tiny_Close_Relation")
    for unit in (u for u in sim.gs.map.units if u.side >= 3):
        assert getattr(unit, "_ai_guardian", False)
        setattr(unit, "_ai_guardian", False)
    assert not neutral_ai._check_units_are_stationary(sim.gs, 3, "x")
    monkeypatch.setenv("WESNOTH_STRICT_WML", "1")
    with pytest.raises(neutral_ai.MobileNeutralUnit):
        neutral_ai._check_units_are_stationary(sim.gs, 3, "x")


def test_the_enclave_pin_survives_a_turn_refresh(actor_scenarios):
    """The pin is re-applied every `turn refresh`, so it has to hold
    after the event fires, not just at build."""
    import tools.scenario_events as se

    for scenario_id, reason in sorted(EXPECTED_ACTORS.items()):
        if reason != "pinned":
            continue
        gs = actor_scenarios[scenario_id].gs
        for unit in (u for u in gs.map.units if u.side >= 3):
            unit.current_moves = unit.max_moves     # as a new turn would
        events = getattr(gs.global_info, "_scenario_events", []) or []
        assert se.fire_event(gs, events, "turn refresh") >= 1, scenario_id
        moves = [u.current_moves for u in gs.map.units if u.side >= 3]
        assert moves and set(moves) == {0}, f"{scenario_id}: {moves}"
