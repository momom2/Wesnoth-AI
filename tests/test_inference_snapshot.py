#!/usr/bin/env python3
"""Tests for the inference-snapshot design (option (b) in the
train_step concurrency discussion).

Verified:

  1. After train_step, inference weights match trainer weights, on
     the TransformerPolicy and the MCTSPolicy paths.
  2. Mutating the trainer's `_model.parameters()` directly does NOT
     affect inference until `_snapshot_inference_weights()` runs.
  3. load_checkpoint syncs the inference snapshot and keeps the two
     encoders on one vocabulary.

The concurrency stress of rollouts against train_step is
tests/test_parallel_rollouts.py's.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

import torch

from helpers.tiny_state import _gs
from wesnoth_ai.transformer_policy import TransformerPolicy


def _small_policy() -> TransformerPolicy:
    """A small network: these tests are about which weights the
    inference copy holds, not about the network's size."""
    return TransformerPolicy(d_model=32, num_layers=1, num_heads=2, d_ff=64)


# ---------------------------------------------------------------------
# Snapshot semantics
# ---------------------------------------------------------------------

def test_trainer_weight_mutation_does_not_affect_inference():
    """Mutating `_model.parameters()` directly should not change
    `_inference_model`'s output until snapshot fires."""
    policy = TransformerPolicy()
    gs = _gs()
    # First inference call -- record action.
    policy.select_action(gs, game_label="g1")

    # Mutate trainer's weights directly. Use a separate game label
    # for the second select_action so the debug-tripwire (same-state-
    # twice) doesn't fire.
    with torch.no_grad():
        for p in policy._model.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    # Second call (different game label, same state). Should match
    # the first if the inference model wasn't affected by the
    # trainer-side mutation.
    policy.select_action(gs, game_label="g2")
    # Both calls used the SAME _inference_model state. With the
    # trainer mutating in between but no snapshot, the inference
    # model is unchanged. The actions should match (DummyPolicy-
    # style determinism: same state, same model, same RNG seed)...
    # but our sampler uses Gumbel-max with torch.randn each call,
    # so actions can differ from RNG even on identical models.
    # Instead, compare RAW LOGITS deterministically.
    with torch.no_grad():
        encoded1 = policy._inference_encoder.encode(gs)
        out1_logits = policy._inference_model(encoded1).actor_logits.clone()

    # Now snapshot and re-check; logits SHOULD differ from the
    # pre-snapshot ones (we mutated trainer weights significantly).
    policy._snapshot_inference_weights()
    with torch.no_grad():
        encoded2 = policy._inference_encoder.encode(gs)
        out2_logits = policy._inference_model(encoded2).actor_logits.clone()

    assert not torch.allclose(out1_logits, out2_logits, atol=1e-3), (
        "snapshot didn't propagate trainer mutations to inference")


def test_snapshot_after_train_step_syncs_weights():
    """After a train_step, inference weights match trainer weights
    again (i.e. the snapshot fires at the end of train_step)."""
    policy = TransformerPolicy()
    gs = _gs()
    # Burn a few rollouts so there's a queue to train on.
    for i in range(4):
        policy.select_action(gs, game_label=f"g{i}")
        policy.observe(f"g{i}", 1, reward=0.5, done=True)
    # Run train_step (mutates _model).
    policy.train_step()
    # Inference weights should now match trainer weights byte-equal.
    for k, v_t in policy._model.state_dict().items():
        v_i = policy._inference_model.state_dict()[k]
        assert torch.allclose(v_t, v_i), (
            f"post-train_step mismatch at {k}")


def test_mcts_train_step_syncs_inference_weights():
    """Regression for the 2026-06-29 frozen-inference bug.

    `MCTSPolicy.train_step` calls `trainer.step_mcts` directly,
    BYPASSING `TransformerPolicy.train_step`'s snapshot. Without an
    explicit refresh, the self-play/search network (`_inference_model`)
    stays frozen at warm-start weights for the entire `--mcts` run
    while only the saved checkpoint's `_model` drifts -- the AlphaZero
    loop never closes.

    We drive the REAL `MCTSPolicy.train_step` and the REAL
    `_snapshot_inference_weights`; only the gradient compute is stubbed
    (its numerics are covered by the trainer's own tests) and made to
    perturb `_model` so the model<->inference divergence is observable.
    """
    from types import SimpleNamespace
    from tools.mcts_policy import MCTSPolicy, ReplayConfig
    from wesnoth_ai.trainer import MCTSExperience, TrainStats

    base = _small_policy()

    # Stand-in gradient step: perturb _model so it diverges from the
    # inference snapshot, exactly as a real optimizer.step() would.
    def fake_step_mcts(batch):
        with torch.no_grad():
            for p in base._model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        return TrainStats(n_transitions=len(batch), n_trajectories=1)
    base._trainer = SimpleNamespace(step_mcts=fake_step_mcts)

    pol = MCTSPolicy(base, replay_config=ReplayConfig(enabled=False))
    pol._queue = [MCTSExperience(game_state=None, visit_counts=[], z=0.0)]

    pol.train_step()

    for k, v_t in base._model.state_dict().items():
        v_i = base._inference_model.state_dict()[k]
        assert torch.allclose(v_t, v_i), (
            f"MCTS train_step left the inference net stale at {k} "
            f"-- self-play would run on a frozen network")


def test_load_checkpoint_syncs_inference():
    """load_checkpoint must propagate the loaded weights into the
    inference snapshot too -- otherwise select_action keeps using
    initial-random weights until the first train_step."""
    import tempfile
    policy = _small_policy()
    # Save the current state.
    with tempfile.TemporaryDirectory() as td:
        ckpt_path = Path(td) / "ckpt.pt"
        # Mutate trainer weights so the ckpt isn't just init state.
        with torch.no_grad():
            for p in policy._model.parameters():
                p.add_(torch.randn_like(p) * 0.3)
        policy.save_checkpoint(ckpt_path)

        # Build a fresh policy + load. Verify inference matches.
        policy2 = _small_policy()
        policy2.load_checkpoint(ckpt_path)
        for k, v_t in policy2._model.state_dict().items():
            v_i = policy2._inference_model.state_dict()[k]
            assert torch.allclose(v_t, v_i), (
                f"post-load mismatch at {k}")


def test_load_checkpoint_preserves_vocab_sharing():
    """After load_checkpoint, the inference encoder must see the
    LOADED vocab through the construction-time shared dicts.

    Regression (2026-07-02): load_checkpoint used to REBIND
    `_encoder.unit_type_to_id` to a fresh dict, orphaning the
    inference encoder's shared reference. The inference encoder --
    which generates ALL rollout data in MCTS mode -- then kept an
    empty vocab and re-grew its own conflicting ids during play,
    reading trained embedding rows under the wrong unit identities
    (silent warm-start corruption)."""
    import tempfile
    policy = TransformerPolicy()
    # Populate the vocab through the normal encode path ('Spearman'
    # from _gs's units), then save.
    policy.select_action(_gs(), game_label="vocab-seed")
    assert "Spearman" in policy._encoder.unit_type_to_id
    with tempfile.TemporaryDirectory() as td:
        ckpt_path = Path(td) / "ckpt.pt"
        policy.save_checkpoint(ckpt_path)

        policy2 = TransformerPolicy()
        policy2.load_checkpoint(ckpt_path)
        # Sharing invariant survives the load...
        assert (policy2._encoder.unit_type_to_id is
                policy2._inference_encoder.unit_type_to_id), (
            "load_checkpoint broke the trainer/inference vocab "
            "sharing -- inference rollouts would re-grow their own "
            "conflicting unit-type ids")
        assert (policy2._encoder.faction_to_id is
                policy2._inference_encoder.faction_to_id)
        # ...and the loaded content is visible through BOTH handles.
        assert "Spearman" in policy2._inference_encoder.unit_type_to_id
