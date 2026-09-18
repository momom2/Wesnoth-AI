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

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_root_masks_shipped import _packed_on_the_state, assert_same_pack  # noqa: E402
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
        server_priors=False,          # the legacy protocol stays covered
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
    # Legacy protocol: the actor's encoder packs nothing, the trainer
    # rebuilds the masks.
    assert all(e.masks is None for e in exps)


@pytest.mark.slow
def test_actor_pool_server_priors_end_to_end(monkeypatch):
    """Same drive with server-side priors on: actors ship packed masks,
    the server returns compact legal actions (wesnoth_ai/server_priors),
    and every experience comes back with the root's pack -- the one
    pack_masks builds on its state -- which step_mcts stages as it is."""
    from tools.actor_pool import ActorPool
    from tools.bench_train_step import configure_trainer_like_az_loop
    from wesnoth_ai import trainer as trainer_module

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
    pool.server_priors = True
    pool.start()
    try:
        outcomes, exps = pool.run_iteration(0, 2, base_seed=7)
    finally:
        pool.shutdown()
    assert len(outcomes) >= 1, "pool produced no completed games"
    assert len(exps) >= 1, "pool shipped no experiences"
    assert all(e.visit_counts for e in exps)
    for e in exps:
        assert_same_pack(e.masks, _packed_on_the_state(policy, e.game_state))

    def no_build(*a, **k):
        raise AssertionError("step_mcts rebuilt masks an experience already carried")
    monkeypatch.setattr(trainer_module, "_host_packed_masks", no_build)
    configure_trainer_like_az_loop(policy._trainer)
    policy._trainer.config.train_batch_size = 4
    stats = policy._trainer.step_mcts(exps)
    assert math.isfinite(float(stats.policy_loss))


@pytest.mark.slow
def test_actor_pool_streams_games_across_a_publication():
    """Continuous generation (tools/actor_stream.py) on the real pool:
    two windows of games with every actor playing throughout, a weight
    publication between them that the games in flight straddle, and a
    clean drain."""
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
        stream = pool.stream(base_seed=7)
        stream.start()
        first = stream.collect(2, timeout=600.0)
        assert len(first.games) == 2 and len(first.outcomes) == 2
        assert first.experiences and first.decisions > 0
        assert first.straddle_max == 0
        assert pool.last_served_forwards > 0 and stream.leaves_served() > 0
        version_before = policy._inference_model._weights_version
        with torch.no_grad():
            for prm in policy._model.parameters():
                prm.add_(0.1 * torch.randn_like(prm))
        policy._snapshot_inference_weights()       # under the server's gate
        assert policy._inference_model._weights_version == version_before + 1
        stream.publish(value_center=0.1, decision_step=int(policy._decision_step))
        second = stream.collect(2, timeout=600.0)
        assert len(second.games) == 2
        # Both actors were mid-game at the publication.
        assert second.straddle_mean == 1.0 and second.straddled_share == 1.0
        assert [g.index for g in second.games] == [2, 3] or set(
            g.index for g in second.games) == {2, 3}
        tail = stream.stop(grace=120.0)
        assert stream._live == set(), "every actor reported done after the drain"
        assert not pool._streaming and not pool._serving
        assert isinstance(tail.games, list)
    finally:
        pool.shutdown()
