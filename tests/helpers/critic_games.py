"""A whole-game record for the step-1 tools' tests: a mini game fought by
the deterministic Brawler to its turn cap, recorded as tools/game_record.py
records a match game."""
from __future__ import annotations


def played_record(max_turns: int = 4, label: str = "game_x_s1_5") -> dict:
    import wesnoth_ai.dummy_policy as dummy_policy
    from sim_test_helpers import Brawler, scenario_setup
    from tools import game_record
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate
    setup = scenario_setup(3, mini=True)
    sim = WesnothSim(build_scenario_gamestate(setup), scenario_id=setup.scenario_id, max_turns=max_turns)
    sim._seed_salt = "critic-test"
    cap = dummy_policy._BOOTSTRAP_UNITS
    dummy_policy._BOOTSTRAP_UNITS = 4
    try:
        policy = Brawler()
        while not sim.done:
            sim.step(policy.select_action(sim.gs, game_label="critic"))
    finally:
        dummy_policy._BOOTSTRAP_UNITS = cap
    return game_record.game_record(sim, setup, game_label=label, build={})
