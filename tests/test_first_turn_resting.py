"""Every unit of a side rests from each of its turn starts, the game's
first included, petrified units too (play_controller.cpp:509-514, 1.18.4;
docs/wesnoth_rules.md "Resting lifecycle"), in both appliers; records of
format 3 and earlier rebuild under the rule they were played with."""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from sim_test_helpers import Brawler, scenario_setup  # noqa: E402
from tools import game_record  # noqa: E402
from tools.wesnoth_sim import WesnothSim  # noqa: E402
from wesnoth_ai.game_core import game_core_class  # noqa: E402
from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate  # noqa: E402

CORE = pytest.param(True, marks=pytest.mark.skipif(game_core_class() is None,
                                                   reason="needs the wesnoth_core wheel of this source's phase"))


def _sim(*, use_core, petrify_leader=False, old_rule=False, max_turns=6):
    setup = scenario_setup(5)
    gs = build_scenario_gamestate(setup)
    if petrify_leader:
        leader = next(u for u in gs.map.units if u.side == 1 and u.is_leader)
        gs.map.units = (gs.map.units - {leader}) | {
            dataclasses.replace(leader, statuses=set(leader.statuses) | {"petrified"})}
    if old_rule:
        gs.global_info._skip_first_turn_resting = True
    return WesnothSim(gs, scenario_id=setup.scenario_id, max_turns=max_turns, use_core=use_core), setup


@pytest.mark.parametrize("use_core", [False, CORE])
def test_side_1_rests_from_the_games_first_side_turn(use_core):
    sim, _ = _sim(use_core=use_core)
    side1 = [u for u in sim.gs.map.units if u.side == 1]
    assert side1 and all("resting" in u.statuses for u in side1)
    old, _ = _sim(use_core=use_core, old_rule=True)
    assert not any("resting" in u.statuses for u in old.gs.map.units if u.side == 1)


@pytest.mark.parametrize("use_core", [False, CORE])
def test_a_petrified_unit_rests_with_its_side(use_core):
    sim, _ = _sim(use_core=use_core, petrify_leader=True)
    leader = next(u for u in sim.gs.map.units if u.side == 1 and u.is_leader)
    assert {"petrified", "resting"} <= set(leader.statuses)


@pytest.mark.parametrize("use_core", [False, CORE])
def test_a_leader_that_stood_still_on_turn_1_rest_heals_on_turn_2(use_core):
    """Hurt during side 2's first turn, side 1's leader, which has not
    moved since the game began, heals the rest heal at its second turn;
    under the earlier rule it did not rest, so it did not heal."""
    healed = {}
    for old_rule in (False, True):
        sim, _ = _sim(use_core=use_core, old_rule=old_rule)
        sim.step({"type": "end_turn"})                         # side 1 ends turn 1 without acting
        gs = sim.gs
        leader = next(u for u in gs.map.units if u.side == 1 and u.is_leader)
        gs.map.units = (gs.map.units - {leader}) | {dataclasses.replace(leader, current_hp=leader.max_hp - 6)}
        sim.gs = gs
        sim.step({"type": "end_turn"})                         # side 2 ends turn 1; side 1's turn 2 starts
        assert (sim.turn_number, sim.current_side) == (2, 1)
        leader = next(u for u in sim.gs.map.units if u.side == 1 and u.is_leader)
        healed[old_rule] = leader.current_hp - (leader.max_hp - 6)
    assert healed == {False: 2, True: 0}


def _played_record(old_rule: bool) -> dict:
    sim, setup = _sim(use_core=True, old_rule=old_rule, max_turns=4)
    policy = Brawler()
    while not sim.done:
        sim.step(policy.select_action(sim.gs, game_label="rest"))
    return json.loads(json.dumps(game_record.game_record(sim, setup, game_label="rest", build={})))


@pytest.mark.skipif(game_core_class() is None, reason="needs the wesnoth_core wheel of this source's phase")
def test_records_rebuild_under_the_rule_they_were_played_with():
    """A game played under the earlier rule and recorded as format 3
    rebuilds; the same record read as format 4 does not, from its first
    fingerprint. A game played now is format 4 and rebuilds."""
    old = _played_record(old_rule=True)
    old["format"] = 3
    game_record.rebuild(old)
    old["format"] = 4
    with pytest.raises(game_record.RecordMismatch, match="after command 0"):
        game_record.rebuild(old)
    new = _played_record(old_rule=False)
    assert new["format"] == game_record.FORMAT == 4
    game_record.rebuild(new)
