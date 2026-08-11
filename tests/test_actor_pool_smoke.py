"""Actor-pool end-to-end smoke (slow tier).

The 2026-08-10 handoff launch burned three boots on a bug this test
would have caught in seconds: an X4 leftover (`roll_mix(...,
drill=...)`) crashed EVERY actor at spawn, and neither the fast tier
nor the in-process fork-guard smoke exercises the pool path at all.
This drives the real ActorPool -- spawn actors, play games through
the central inference server, ship experiences back -- at the
smallest viable scale.

Windows spawn makes actor boot slow (~10s+ each); marked slow.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.mcts import MCTSConfig  # noqa: E402
from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402


@pytest.mark.slow
def test_actor_pool_plays_games_end_to_end():
    from tools.actor_pool import ActorPool

    policy = TransformerPolicy(device=torch.device("cpu"), d_model=32,
                               num_layers=1, num_heads=2, d_ff=64)
    cfg = MCTSConfig(n_simulations=2, batch_size=1)
    pool = ActorPool(
        policy, 2, cfg,
        scenario_opts=dict(mini_maps=True, mini_ratio=1.0,
                           fogless_ratio=0.0, midgame_ratio=0.0,
                           ladder_ratio=0.0),
        max_turns=4,
        iteration_timeout=600.0,
    )
    pool.start()
    try:
        outcomes, exps = pool.run_iteration(0, 2, base_seed=7)
    finally:
        pool.shutdown()

    # The drill-kwarg bug produced exactly (0 games, 0 experiences)
    # with every actor dead -- the assertion below is the one that
    # would have failed.
    assert len(outcomes) >= 1, "pool produced no completed games"
    assert len(exps) >= 1, "pool shipped no experiences"
    for o in outcomes:
        assert o.turns >= 1
