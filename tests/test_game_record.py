"""Game records (tools/game_record.py): a game the simulator played,
recorded and rebuilt from its record, reaches the same position after
every command; the record carries the outcome data the simulator and
the search computed."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

from tools import game_record  # noqa: E402
from wesnoth_ai.game_core import states_equal  # noqa: E402

# What the simulator keeps on the state for its own bookkeeping and the
# applier does not: the seed counter, the advancement salt and counter,
# and the last command's side channels.
SIM_ONLY = ("_rng_request_counter", "_advance_counter", "_advance_salt",
            "_last_move_walk", "_last_checkup_strikes", "_last_advance_events")


def _differences(a, b):
    return [d for d in states_equal(a, b, stash=False)
            if not any(k in d for k in SIM_ONLY)]


def _played_game(max_turns=8):
    """A mini game fought to its cap by the deterministic Brawler, with
    a recruit rejection on the way, and the position after each step
    keyed by the index of the step's last command (a step that ends a
    turn records end_turn and the next init_side)."""
    import wesnoth_ai.dummy_policy as dummy_policy
    from sim_test_helpers import Brawler, scenario_setup
    from tools.scenario_pool import build_scenario_gamestate
    from tools.wesnoth_sim import WesnothSim
    setup = scenario_setup(3, mini=True)
    sim = WesnothSim(build_scenario_gamestate(setup), scenario_id=setup.scenario_id,
                     max_turns=max_turns, use_core=False)
    sim._seed_salt = "record-test"
    sim.enable_uniform_advancement()
    after = {len(sim.command_history) - 1: copy.deepcopy(sim.gs)}
    cap = dummy_policy._BOOTSTRAP_UNITS
    dummy_policy._BOOTSTRAP_UNITS = 8
    try:
        policy = Brawler()
        while not sim.done:
            if len(sim.command_history) == 12:
                sim.reject_recruit_hex(0, 0)
                after[11] = copy.deepcopy(sim.gs)
            sim.step(policy.select_action(sim.gs, game_label="rec"))
            after[len(sim.command_history) - 1] = copy.deepcopy(sim.gs)
    finally:
        dummy_policy._BOOTSTRAP_UNITS = cap
    return sim, setup, after


def test_a_recorded_game_rebuilds_position_by_position(tmp_path):
    sim, setup, _after = _played_game()
    game_record.configure(tmp_path, "t")
    try:
        game_record.record_game(sim, setup, game_label="g", build={})
    finally:
        game_record.configure(None, "")
    [rec] = list(game_record.read_records(tmp_path / "t.jsonl.gz"))
    kinds = [c[0] for c in rec["commands"]]
    assert kinds.count("attack") >= 5 and len(kinds) >= 40, kinds
    assert rec["rejections"] == [[12, 0, 0]]
    assert (rec["winner"], rec["ended_by"]) == (sim.winner, sim.ended_by)
    assert _differences(game_record.rebuild(rec), sim.gs) == []


def test_the_walk_passes_through_every_position_the_game_did():
    sim, setup, after = _played_game(max_turns=4)
    rec = game_record.game_record(sim, setup, game_label="g", build={})
    rec = game_record.json.loads(game_record.json.dumps(rec))     # through JSON, as stored
    walked = [copy.deepcopy(gs) for _k, gs, _cmd in game_record.walk(rec)]
    assert len(walked) == len(rec["commands"])
    # the state before command k + 1 is the simulator's after command k
    compared = 0
    for k, gs in sorted(after.items()):
        if k + 1 < len(walked):
            assert _differences(walked[k + 1], gs) == [], k
            compared += 1
    assert compared >= 20


def test_an_attack_records_the_counter_weapon_tables_the_simulator_computed():
    """A Dwarvish Fighter answers a Spearman's spear with its axe or its
    hammer (both melee): the simulator runs both strike tables to choose,
    and the recorded attack carries them."""
    from tools.replay_dataset import _build_initial_gamestate
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.classes import Position
    gs = _build_initial_gamestate({
        "game_id": "rec", "map_data": "Xv, Xv, Xv, Xv, Xv\nXv, Gg, Gg, Gg, Xv\nXv, Xv, Xv, Xv, Xv",
        "starting_units": [
            {"uid": 1, "type": "Spearman", "side": 1, "x": 0, "y": 0, "is_leader": True},
            {"uid": 2, "type": "Dwarvish Fighter", "side": 2, "x": 1, "y": 0, "is_leader": True}],
        "starting_sides": [{"side": k, "gold": 0, "recruit": []} for k in (1, 2)]})
    sim = WesnothSim(gs, scenario_id="rec", max_turns=2, use_core=False)
    sim.step({"type": "attack", "start_hex": Position(0, 0), "target_hex": Position(1, 0),
              "attack_index": 0})
    rc = sim.command_history[-1]
    assert rc.kind == "attack"
    data = rc.extras["outcomes"]["counter_weapon"]
    assert set(data["tables"]) == {"0", "1"} and data["chosen"] == rc.cmd[6]
    for table in data["tables"].values():
        assert abs(sum(row[-1] for row in table) - 1.0) < 1e-9


def test_the_search_distribution_of_a_played_attack_reaches_its_command():
    """`note_search_outcomes` moves the distribution a policy reports for
    the attack it just played into that command's outcome data."""
    from tools.combat_outcomes import enumerate_attack_outcomes
    from tools.replay_dataset import _build_initial_gamestate
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.classes import Position
    gs = _build_initial_gamestate({
        "game_id": "rec", "map_data": "Xv, Xv, Xv, Xv\nXv, Gg, Gg, Xv\nXv, Xv, Xv, Xv",
        "starting_units": [
            {"uid": 1, "type": "Spearman", "side": 1, "x": 0, "y": 0, "is_leader": True},
            {"uid": 2, "type": "Spearman", "side": 2, "x": 1, "y": 0, "is_leader": True}],
        "starting_sides": [{"side": k, "gold": 0, "recruit": []} for k in (1, 2)]})
    sim = WesnothSim(gs, scenario_id="rec", max_turns=2, use_core=False)
    action = {"type": "attack", "start_hex": Position(0, 0), "target_hex": Position(1, 0),
              "attack_index": 0}
    dist = enumerate_attack_outcomes(sim.gs, action)
    assert dist is not None

    class _Searched:
        def pop_played_outcomes(self, _label):
            return dist

    sim.step(action)
    game_record.note_search_outcomes(sim, _Searched(), "g")
    data = sim.command_history[-1].extras["outcomes"]["search"]
    assert (data["attacker"], data["defender"]) == ("u1", "u2")
    assert abs(sum(row[-1] for row in data["outcomes"]) - 1.0) < 1e-9
