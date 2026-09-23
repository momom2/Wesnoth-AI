"""What the scenario builder reads that the scenario-init oracle found
unread (tools/scenario_init_oracle.py, 2026-09-23).

Three minis declare `fog=no` on their player sides, and generation played
them under fog because it never read the declaration. The statues of
Caves of the Basilisk, Sullas Ruins and Thousand Stings Garrison carry
modifications that leave them 1 hp and no moves (a custom `remove_hp`
trait, or the same effects in an [object]); units placed in a [side]
block got none of them.
"""
import logging
import random
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import scenario_pool as sp  # noqa: E402


def _build(scenario_id: str, fogless: bool = False):
    base = sp.random_setup(random.Random(1), forced_faction=None,
                           mini_maps=scenario_id in sp.MINI_MAP_SCENARIO_IDS)
    setup = replace(base, scenario_id=scenario_id, fogless=fogless)
    logging.disable(logging.WARNING)
    try:
        return sp.build_scenario_gamestate(setup)
    finally:
        logging.disable(logging.NOTSET)


def test_the_scenario_declares_the_fog_and_fogless_still_turns_it_off():
    assert getattr(_build("2p_mini").global_info, "_fog") is False
    assert getattr(_build("Modified_Tiny_Close_Relation").global_info, "_fog") is False
    assert getattr(_build("multiplayer_Hamlets").global_info, "_fog") is True
    assert getattr(_build("multiplayer_Hamlets", fogless=True).global_info, "_fog") is False


@pytest.mark.parametrize("scenario_id", ["multiplayer_Basilisk", "multiplayer_thousand_stings_garrison"])
def test_statues_placed_in_a_side_block_carry_their_trait_effects(scenario_id):
    statues = [u for u in _build(scenario_id).map.units if "petrified" in u.statuses]
    assert statues
    assert all(u.max_hp == u.current_hp == 1 and u.max_moves == 0 for u in statues)
