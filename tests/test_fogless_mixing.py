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
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

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


def test_mini_games_always_keep_fog():
    rng = random.Random(7)
    for cat in ("mini",):
        for _ in range(10):
            setup = random_setup(rng, category=cat)
            assert not setup.fogless, \
                "fogless applies to the ladder pool only"


def test_roll_mix_respects_absolute_proportions():
    rng = random.Random(7)
    counts = {}
    n = 4000
    for _ in range(n):
        c = roll_mix(rng, midgame=0.2, mini=0.2,
                     fogless=0.2, ladder=0.4)
        counts[c] = counts.get(c, 0) + 1
    # 4000 rolls: each category lands well within +-0.05.
    for cat, expect in (("midgame", 0.2), ("mini", 0.2),
                        ("fogless", 0.2), ("ladder", 0.4)):
        assert abs(counts.get(cat, 0) / n - expect) < 0.05, \
            (cat, counts)


def test_mix_guard_rejects_bad_sums():
    with pytest.raises(ValueError):
        validate_mix(midgame=0.2, mini=0.2,
                     fogless=0.2, ladder=0.5)   # sums to 1.1
    with pytest.raises(ValueError):
        validate_mix(midgame=0.2, ladder=0.4)   # sums to 0.6
    with pytest.raises(ValueError):
        validate_mix(midgame=-0.1, ladder=1.1)  # out of range
    validate_mix(midgame=0.2, mini=0.2,
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
    from tools.selfplay_game import _recruit_cost_lookup, play_one_game
    from wesnoth_ai.transformer_policy import TransformerPolicy

    pol = TransformerPolicy(device=torch.device("cpu"), d_model=32,
                            num_layers=1, num_heads=4, d_ff=64)
    mp = MCTSPolicy(pol, MCTSConfig(n_simulations=4, batch_size=1,
                                    add_root_noise=False))
    cost = _recruit_cost_lookup()

    mp._rng = np.random.default_rng(3)
    sim = fresh_scenario_sim(seed=3, max_turns=4, mini=True)
    fog = getattr(sim.gs.global_info, "_fog", True)
    out = play_one_game(sim, mp, lambda d: 0.0, game_label="g",
                        cost_lookup=cost)
    assert out.fogless is (not fog)
    assert out.villages_mean_s1 >= 0.0
    assert out.villages_end_s1 >= 0

    mp._rng = np.random.default_rng(3)
    sim2 = fresh_scenario_sim(seed=3, max_turns=4, mini=True)
    setattr(sim2.gs.global_info, "_fog", False)
    out2 = play_one_game(sim2, mp, lambda d: 0.0, game_label="g2",
                         cost_lookup=cost)
    assert out2.fogless is True


# ---------------------------------------------------------------------
# Scenario-economy fidelity (2026-07-21 bugfixes): scenario [side]
# gold= and income= are ground truth; nothing in the training path
# may override them (user ruling: "scenarios were never meant to
# start at a different value than the one specified").

def test_scenario_gold_is_ground_truth():
    from tools.scenario_pool import (ScenarioSetup,
                                     build_scenario_gamestate)
    s = ScenarioSetup(scenario_id="multiplayer_Arcanclave_Citadel",
                      faction1="Rebels", leader1="Elvish Captain",
                      faction2="Loyalists", leader2="Lieutenant",
                      fogless=False, tod_start=1)
    gs = build_scenario_gamestate(s)
    assert [sd.current_gold for sd in gs.sides[:2]] == [175, 175]


def test_side_income_offset_does_not_leak_across_sides():
    """[side] income= is an OFFSET on game_config::base_income
    (team.hpp 1.18.4: `base_income() { return info_.income +
    game_config::base_income; }`) and is parsed per PLAYER side.
    Thousand Stings Garrison's income=-2 sits on SIDE 3 (the
    garrison) -- player sides must stay at the base 2. No current
    ladder-pool map sets player-side income, so the offset
    mechanism is dormant; this pins that side-3 attrs don't leak
    and the default stays correct."""
    from tools.scenario_pool import (ScenarioSetup,
                                     build_scenario_gamestate)
    s = ScenarioSetup(
        scenario_id="multiplayer_Thousand_Stings_Garrison",
        faction1="Rebels", leader1="Elvish Captain",
        faction2="Loyalists", leader2="Lieutenant",
        fogless=False, tod_start=1)
    gs = build_scenario_gamestate(s)
    assert [sd.base_income for sd in gs.sides[:2]] == [2, 2]


def test_the_training_path_overrides_no_scenario_setting(monkeypatch):
    """The training path passed `PvPDefaults.starting_gold=100` over
    every scenario's own gold until 2026-07-21 (minis designed for
    ~50g trained on 100; the since-deleted gold=0 drills gained
    recruiting), and its village gold and experience modifier over the
    scenario's until 2026-09-21 (five of the seven mini scenarios ask
    for 3 gold per village and got 2). Scenario settings are ground
    truth: every economy knob must reach the builder as None, whatever
    `PvPDefaults` carries.

    Checked at the seam rather than by reading the source: the
    previous version of this test grepped `_play_one_game_safe` for
    the literal `"sg = None"`, which passes or fails on how the line
    is spelled."""
    from tools import scenario_pool, selfplay_game
    from tools.wesnoth_sim import PvPDefaults

    seen = {}

    class _Stop(RuntimeError):
        pass

    def spy(setup, **kwargs):
        seen.update(kwargs)
        raise _Stop("captured")            # the caller logs and returns None

    monkeypatch.setattr(scenario_pool, "build_scenario_gamestate", spy)
    loud = PvPDefaults(starting_gold=100, base_income=2, village_gold=2,
                       village_support=1, experience_modifier=70)
    out = selfplay_game._play_one_game_safe(
        setup=random_setup(random.Random(3), mini_maps=True), max_turns=4,
        pvp_defaults=loud, policy=None, reward_fn=None, cost_lookup=None,
        game_label="t")
    assert out is None and seen, "the builder was never reached"
    assert seen["starting_gold"] is None
    assert seen["village_gold"] is None
    assert seen["village_upkeep"] is None
    assert seen["experience_modifier"] is None
