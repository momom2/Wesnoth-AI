#!/usr/bin/env python3
"""Mid-game self-play starts from the human corpus (2026-07-12).

Pins: uniform cut in [1, end_turn] lands on a player side-turn
boundary with both leaders alive; the sim continues the position
through the production play_one_game with scenario events NOT
re-applied; outcomes carry the midgame flag.

Skips when replays_dataset/ is absent (fresh clones re-download it;
see CLAUDE.md).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

DATASET = Path(__file__).parent.parent / "replays_dataset"

pytestmark = pytest.mark.skipif(
    not (DATASET / "value_corpus_index.jsonl").is_file(),
    reason="value corpus not extracted locally")


def test_sample_midgame_start_valid_state():
    from tools.midgame_starts import sample_midgame_start
    rng = random.Random(3)
    mg = sample_midgame_start(rng, DATASET)
    assert mg is not None
    gs, scen_id, cut, begin_side, prov = mg
    assert scen_id
    assert 1 <= gs.global_info.turn_number
    assert begin_side in (1, 2)
    alive = {u.side for u in gs.map.units if u.is_leader}
    assert {1, 2} <= alive
    # Both sides field entries exist (gold, recruit lists).
    assert len(gs.sides) >= 2


@pytest.mark.slow          # ~19s: see pytest.ini two-tier note
def test_midgame_continuation_through_production_path():
    import numpy as np
    import torch
    from tools.mcts import MCTSConfig
    from tools.mcts_policy import MCTSPolicy
    from tools.midgame_starts import sample_midgame_start
    from tools.sim_self_play import _recruit_cost_lookup, play_one_game
    from wesnoth_ai.transformer_policy import TransformerPolicy
    from wesnoth_sim import WesnothSim

    # Individual samples may legitimately return None (cut at game
    # end, dead leader); production falls back to fresh starts. Retry
    # like production would.
    mg = None
    for seed in range(11, 31):
        mg = sample_midgame_start(random.Random(seed), DATASET)
        if mg is not None:
            break
    assert mg is not None, "no valid midgame sample in 20 tries"
    gs, scen_id, cut, begin_side, prov = mg
    sim = WesnothSim(gs, scenario_id=scen_id,
                     max_turns=gs.global_info.turn_number + 4,
                     apply_scenario_events=False,
                     begin_side=begin_side)
    sim._midgame_start = True
    # C1 regression (review 2026-07-12): the sim resumes the side the
    # cut landed on -- no skipped side-2 turn, no side-1 double turn,
    # no spurious turn bump (init_side(2) does not increment turns).
    assert sim.gs.global_info.current_side == begin_side
    if begin_side == 2:
        assert sim.gs.global_info.turn_number == cut

    pol = TransformerPolicy(device=torch.device("cpu"), d_model=32,
                            num_layers=1, num_heads=4, d_ff=64)
    mp = MCTSPolicy(pol, MCTSConfig(n_simulations=4, batch_size=1,
                                    add_root_noise=False))
    mp._rng = np.random.default_rng(11)
    out = play_one_game(sim, mp, lambda d: 0.0, game_label="mg",
                        cost_lookup=_recruit_cost_lookup())
    assert out.midgame is True
    assert out.engagement is not None
    assert out.engagement["attacks_invalid_wesnoth"] == {1: 0, 2: 0}
