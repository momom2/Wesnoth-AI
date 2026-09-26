"""Serve processes of the actor pool (tools/actor_pool.py, module
docstring "Serve processes"): the copy of the inference pair another
process builds, the weight transfer, the dead-server reply marker, and
-- slow tier, real processes on CPU -- a pool with one serve process
next to the learner's: games complete, the stats merge (each server's
compiled-trunk state with them), a weight publication is refused until
synced and changes the server's outputs once it is, a failure the
serve process reports while serving aborts the iteration at once, and
a stream publishes into the serve process while it serves.
"""
from __future__ import annotations

import queue as _queue
import sys
import time
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_actor_pool_smoke import (  # noqa: E402
    assert_publication_straddled, collect_across_publication,
)
from tools.actor_pool import _IPCInferenceClient, ServeProcessDied  # noqa: E402
from tools.actor_protocol import _S_ERROR, _RID_SERVER_DEAD  # noqa: E402
from tools.inference_seam import (  # noqa: E402
    build_inference_pair, inference_blueprint, load_inference_state,
    pack_inference_state,
)
from tools.mcts import MCTSConfig  # noqa: E402
from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402

CPU = torch.device("cpu")


def _tiny_policy() -> TransformerPolicy:
    return TransformerPolicy(device=CPU, d_model=32, num_layers=1, num_heads=2, d_ff=64)


def _perturb(policy: TransformerPolicy) -> None:
    """A new publication: the trainer-side weights move, the snapshot
    follows (what train_step does after its optimizer step)."""
    with torch.no_grad():
        for module in (policy._model, policy._encoder):
            for p in module.parameters():
                p.add_(0.5 * torch.randn_like(p))
    policy._snapshot_inference_weights()


def test_blueprint_rebuilds_the_pair_and_state_round_trips():
    pol = _tiny_policy()
    model, encoder = build_inference_pair(
        inference_blueprint(pol._inference_model, pol._inference_encoder), CPU)
    assert set(model.state_dict()) == set(pol._inference_model.state_dict())
    assert set(encoder.state_dict()) == set(pol._inference_encoder.state_dict())
    _perturb(pol)
    load_inference_state(pack_inference_state(pol._inference_model, pol._inference_encoder),
                         model, encoder, CPU)
    for mine, theirs in ((model, pol._inference_model), (encoder, pol._inference_encoder)):
        ref = theirs.state_dict()
        for k, v in mine.state_dict().items():
            assert torch.equal(v, ref[k]), k
    assert not model.training and not encoder.training


class _Q:
    def __init__(self, items=None):
        self.items = list(items or [])

    def put(self, x):
        self.items.append(x)

    def get(self, timeout=None):
        if not self.items:
            raise _queue.Empty
        return self.items.pop(0)


def test_client_raises_on_the_dead_server_marker_of_its_own_play():
    """The manager marks the reply queues of a dead server's actors with
    the tag of the iteration it aborts. A marker the actor reads under a
    later PLAY was left behind by that iteration and is dropped (CI
    2026-09-24: one crashed the first game of the next stream)."""
    req, resp = _Q(), _Q([(_RID_SERVER_DEAD, 3)])
    client = _IPCInferenceClient(3, [req], resp)
    client.use_server(0, 3)
    with pytest.raises(RuntimeError, match="died"):
        client.infer_batch([object()])
    assert req.items[0][0] == 3

    client.use_server(0, 4)
    resp.items = [(_RID_SERVER_DEAD, 3), (1, [])]
    assert client.infer_batch([object()]) == []


@pytest.mark.slow
def test_pool_with_a_serve_process_serves_syncs_and_refuses_stale_weights():
    from sim_test_helpers import fresh_scenario_sim
    from tools.actor_pool import ActorPool

    policy = _tiny_policy()
    pool = ActorPool(
        policy, 2, MCTSConfig(n_simulations=2, batch_size=1),
        scenario_opts=dict(mini_maps=True, mini_ratio=1.0, fogless_ratio=0.0,
                           midgame_ratio=0.0, ladder_ratio=0.0),
        max_turns=3, iteration_timeout=600.0, serve_processes=2,
    )
    gs = fresh_scenario_sim(seed=21, max_turns=6, mini=True).gs
    pool.start()
    try:
        before = pool.probe([gs])
        assert len(before) == 2
        assert torch.allclose(before[0][0].value_logits, before[1][0].value_logits, atol=1e-5)

        outcomes, exps = pool.run_iteration(0, 2, base_seed=7)
        assert len(outcomes) >= 1, "pool produced no completed games"
        assert len(exps) >= 1, "pool shipped no experiences"
        # Actor 0 asks the learner process, actor 1 the serve process.
        assert len(pool.last_leaves_per_server) == 2
        assert all(n > 0 for n in pool.last_leaves_per_server), pool.last_leaves_per_server
        assert pool.last_served_forwards == sum(pool.last_leaves_per_server)
        # The serve process's own compiled packed trunk state rides its
        # stats (no compile configured here: not active on either).
        assert [s["active"] for s in pool.last_packed_compile_per_server] == [False, False]

        _perturb(policy)
        with pytest.raises(RuntimeError, match="sync_servers"):
            pool.run_iteration(1, 2, base_seed=8)
        pool.sync_servers()
        after = pool.probe([gs])
        assert not torch.allclose(before[1][0].value_logits, after[1][0].value_logits)
        assert torch.allclose(after[0][0].value_logits, after[1][0].value_logits, atol=1e-5)

        outcomes, _ = pool.run_iteration(1, 2, base_seed=8)
        assert len(outcomes) >= 1

        # An error reply the serve process posts WHILE SERVING (what its
        # command handler sends when a command fails; the process stays
        # alive) surfaces during the iteration as a dead server, not at
        # the timeout.
        pool._server_q.put((_S_ERROR, 1, "injected failure"))
        with pytest.raises(ServeProcessDied, match="injected failure"):
            pool.run_iteration(2, 2, base_seed=9)

        # Continuous generation: a publication lands in the serve
        # process WHILE it serves (its SYNC loads under the gate), the
        # stream refuses nothing, and both servers agree afterwards.
        t_open = time.time()
        stream = pool.stream(base_seed=11, tag=3)
        stream.start()
        first = stream.collect(2, timeout=600.0)
        assert len(first.games) == 2
        # Iteration 2 ended with its games in flight: the reports its
        # actors sent afterwards are not the stream's games, and the
        # dead-server marker it left crashes none of them.
        assert all(g.t_start > t_open and g.outcome is not None for g in first.games), \
            [(g.actor, g.index, round(g.t_start - t_open, 3), g.outcome is None)
             for g in first.games]
        # Both actors play through the first window, each on its own
        # server, so both serve in it.
        assert all(n > 0 for n in pool.last_leaves_per_server), pool.last_leaves_per_server
        versions = []

        def publish():
            _perturb(policy)
            versions.append(stream.publish())

        games, targets, t_pub = collect_across_publication(stream, publish)
        assert versions == [policy._inference_model._weights_version]
        assert_publication_straddled(games, targets, t_pub)
        # The last window runs from the previous collect to this one and
        # can be too short for a batch (CI 2026-09-23 read 0), so only
        # its accounting is checked.
        assert sum(pool.last_leaves_per_server) == pool.last_served_forwards
        stream.stop(grace=120.0)
        synced = pool.probe([gs])
        assert torch.allclose(synced[0][0].value_logits, synced[1][0].value_logits, atol=1e-5)
        assert not torch.allclose(after[1][0].value_logits, synced[1][0].value_logits)
    finally:
        pool.shutdown()
