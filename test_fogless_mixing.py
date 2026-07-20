#!/usr/bin/env python3
"""Fogless mixing (2026-07-11; absolute-mix redesign 2026-07-20).

`random_setup(category="fogless")` plays a LADDER-pool game with
fog of war off: `setup.fogless=True` makes
`build_scenario_gamestate` set `global_info._fog = False`, which
`visibility.units_visible_to` and the encoder's village fog rule
consume. Mixing is the caller's job via `roll_mix` (absolute
proportions over all five categories, guarded to sum to 1).
"""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

import pytest

from tools.scenario_pool import (LADDER_SCENARIO_IDS,
                                 build_scenario_gamestate, random_setup,
                                 roll_mix, validate_mix)


def test_fogless_category_marks_every_ladder_game_fogless():
    rng = random.Random(7)
    for _ in range(10):
        setup = random_setup(rng, category="fogless")
        assert setup.scenario_id in LADDER_SCENARIO_IDS
        assert setup.fogless


def test_default_is_fogged():
    rng = random.Random(7)
    for _ in range(10):
        assert not random_setup(rng).fogless


def test_mini_and_drill_games_always_keep_fog():
    rng = random.Random(7)
    for cat in ("mini", "drill"):
        for _ in range(10):
            setup = random_setup(rng, category=cat)
            assert not setup.fogless, \
                "fogless applies to the ladder pool only"


def test_roll_mix_respects_absolute_proportions():
    rng = random.Random(7)
    counts = {}
    n = 4000
    for _ in range(n):
        c = roll_mix(rng, midgame=0.2, mini=0.2, drill=0.0,
                     fogless=0.2, ladder=0.4)
        counts[c] = counts.get(c, 0) + 1
    assert counts.get("drill", 0) == 0
    # 4000 rolls: each category lands well within +-0.05.
    for cat, expect in (("midgame", 0.2), ("mini", 0.2),
                        ("fogless", 0.2), ("ladder", 0.4)):
        assert abs(counts.get(cat, 0) / n - expect) < 0.05, \
            (cat, counts)


def test_mix_guard_rejects_bad_sums():
    with pytest.raises(ValueError):
        validate_mix(midgame=0.2, mini=0.2, drill=0.0,
                     fogless=0.2, ladder=0.5)   # sums to 1.1
    with pytest.raises(ValueError):
        validate_mix(midgame=0.2, ladder=0.4)   # sums to 0.6
    with pytest.raises(ValueError):
        validate_mix(midgame=-0.1, ladder=1.1)  # out of range
    validate_mix(midgame=0.2, mini=0.2, drill=0.0,
                 fogless=0.2, ladder=0.4)       # exact -> OK


def test_fogless_setup_sets_fog_attr_and_survives_deepcopy():
    rng = random.Random(7)
    setup = random_setup(rng, category="fogless")
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
