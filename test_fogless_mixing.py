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


def test_outcome_carries_fog_flag_and_village_metrics():
    """The fogless-mixing experiment is only observable if outcomes
    record the condition and capture activity: GameOutcome.fogless
    must mirror the game's _fog state, and the village fields must
    populate (per-turn time-average + end counts). Drives the REAL
    play_one_game path."""
    import numpy as np
    import torch
    from sim_test_helpers import fresh_scenario_sim
    from tools.mcts import MCTSConfig
    from tools.mcts_policy import MCTSPolicy
    from tools.sim_self_play import _recruit_cost_lookup, play_one_game
    from transformer_policy import TransformerPolicy

    pol = TransformerPolicy(device=torch.device("cpu"), d_model=32,
                            num_layers=1, num_heads=4, d_ff=64)
    mp = MCTSPolicy(pol, MCTSConfig(n_simulations=4, batch_size=1,
                                    add_root_noise=False))
    cost = _recruit_cost_lookup()

    mp._rng = np.random.default_rng(3)
    sim = fresh_scenario_sim(seed=3, max_turns=4, mini=True)
    out = play_one_game(sim, mp, lambda d: 0.0, game_label="g",
                        cost_lookup=cost)
    assert out.fogless is False
    assert out.villages_mean_s1 >= 0.0
    assert out.villages_end_s1 >= 0

    mp._rng = np.random.default_rng(3)
    sim2 = fresh_scenario_sim(seed=3, max_turns=4, mini=True)
    setattr(sim2.gs.global_info, "_fog", False)
    out2 = play_one_game(sim2, mp, lambda d: 0.0, game_label="g2",
                         cost_lookup=cost)
    assert out2.fogless is True
