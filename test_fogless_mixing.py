#!/usr/bin/env python3
"""Fogless-ratio mixing (2026-07-11).

`random_setup(fogless_ratio=r)` plays a fraction r of LADDER-pool
games with fog of war off: `setup.fogless=True` makes
`build_scenario_gamestate` set `global_info._fog = False`, which
`visibility.units_visible_to` and the encoder's village fog rule
consume. Mini/drill games always keep fog.
"""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

from tools.scenario_pool import (LADDER_SCENARIO_IDS,
                                 build_scenario_gamestate, random_setup)


def test_ratio_one_marks_every_ladder_game_fogless():
    rng = random.Random(7)
    for _ in range(10):
        setup = random_setup(rng, fogless_ratio=1.0)
        assert setup.scenario_id in LADDER_SCENARIO_IDS
        assert setup.fogless


def test_default_is_fogged():
    rng = random.Random(7)
    for _ in range(10):
        assert not random_setup(rng).fogless


def test_mini_and_drill_games_always_keep_fog():
    rng = random.Random(7)
    for _ in range(20):
        setup = random_setup(rng, mini_ratio=0.5, drill_ratio=0.5,
                             fogless_ratio=1.0)
        assert not setup.fogless, \
            "fogless applies to the ladder pool only"


def test_fogless_setup_sets_fog_attr_and_survives_deepcopy():
    rng = random.Random(7)
    setup = random_setup(rng, fogless_ratio=1.0)
    gs = build_scenario_gamestate(setup)
    assert getattr(gs.global_info, "_fog", True) is False
    # MCTS deepcopies states before encoding; the underscore attr
    # must survive GlobalInfo.__deepcopy__.
    gs2 = copy.deepcopy(gs)
    assert getattr(gs2.global_info, "_fog", True) is False


def test_fogged_setup_leaves_fog_attr_unset():
    rng = random.Random(7)
    setup = random_setup(rng)
    gs = build_scenario_gamestate(setup)
    assert getattr(gs.global_info, "_fog", True) is True
