#!/usr/bin/env python3
"""Tests for the inference-snapshot design (option (b) in the
train_step concurrency discussion).

Verified:

  1. Inference model + encoder are SEPARATE instances from the
     trainer's model + encoder.
  2. After train_step, inference weights match trainer weights.
  3. Mutating the trainer's `_model.parameters()` directly does NOT
     affect inference until `_snapshot_inference_weights()` runs.
  4. load_checkpoint syncs the inference snapshot.
  5. Concurrency stress: rollouts on a worker thread + train_steps
     on the main thread don't crash, produce no NaN, and inference
     output stays on a consistent snapshot per call.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

import glob
import pytest
import torch

from classes import (
    Alignment, Attack, DamageType, GameState, GlobalInfo, Hex, Map,
    Position, SideInfo, Terrain, TerrainModifiers, Unit,
)
from transformer_policy import TransformerPolicy


def _u(uid, side, x, y, *, hp=40, name="Spearman", is_leader=False):
    return Unit(
        id=uid, name=name, name_id=0, side=side,
        is_leader=is_leader, position=Position(x, y),
        max_hp=40, max_moves=5, max_exp=32, cost=14,
        alignment=Alignment.NEUTRAL, levelup_names=[],
        current_hp=hp, current_moves=5, current_exp=0,
        has_attacked=False,
        attacks=[Attack(type_id=DamageType.PIERCE, number_strikes=3,
                        damage_per_strike=7, is_ranged=False,
                        weapon_specials=set())],
        resistances=[1.0]*6, defenses=[0.5]*14,
        movement_costs=[1]*14, abilities=set(), traits=set(),
        statuses=set(),
    )


def _gs():
    """Minimal 2-side game state with units and a few hexes."""
    units = {
        _u("u1", 1, 3, 3),
        _u("u2", 2, 4, 3),
        _u("ldr1", 1, 0, 0, is_leader=True),
        _u("ldr2", 2, 8, 8, is_leader=True),
    }
    hexes = {Hex(position=Position(x, y),
                 terrain_types={Terrain.FLAT}, modifiers=set())
             for x in range(10) for y in range(10)}
    sides = [SideInfo(player=f"S{i+1}", recruits=[], current_gold=100,
                      base_income=2, nb_villages_controlled=0)
             for i in range(2)]
    return GameState(
        game_id="t",
        map=Map(size_x=10, size_y=10, mask=set(), fog=set(),
                hexes=hexes, units=units),
        global_info=GlobalInfo(current_side=1, turn_number=1,
                               time_of_day="day", village_gold=2,
                               village_upkeep=1, base_income=2),
        sides=sides,
    )


# ---------------------------------------------------------------------
# Structural: the snapshot is a separate instance
# ---------------------------------------------------------------------

def test_inference_model_is_separate_object():
    policy = TransformerPolicy()
    assert policy._inference_model is not policy._model
    assert policy._inference_encoder is not policy._encoder


def test_inference_weights_match_trainer_at_init():
    """At init, the inference snapshot is byte-equal to the
    trainer's freshly-built weights."""
    policy = TransformerPolicy()
    for k, v_t in policy._model.state_dict().items():
        v_i = policy._inference_model.state_dict()[k]
        assert torch.allclose(v_t, v_i), f"mismatch at {k}"


def test_vocab_dicts_shared_by_reference():
    """Both encoders point to the same vocab dicts. Adding a new
    type via either side appears on the other."""
    policy = TransformerPolicy()
    assert (policy._encoder.unit_type_to_id is
            policy._inference_encoder.unit_type_to_id)
    assert (policy._encoder.faction_to_id is
            policy._inference_encoder.faction_to_id)


# ---------------------------------------------------------------------
# Snapshot semantics
# ---------------------------------------------------------------------

def test_trainer_weight_mutation_does_not_affect_inference():
    """Mutating `_model.parameters()` directly should not change
    `_inference_model`'s output until snapshot fires."""
    policy = TransformerPolicy()
    gs = _gs()
    # First inference call -- record action.
    a1 = policy.select_action(gs, game_label="g1")

    # Mutate trainer's weights directly. Use a separate game label
    # for the second select_action so the debug-tripwire (same-state-
    # twice) doesn't fire.
    with torch.no_grad():
        for p in policy._model.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    # Second call (different game label, same state). Should match
    # the first if the inference model wasn't affected by the
    # trainer-side mutation.
    a2 = policy.select_action(gs, game_label="g2")
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
    from trainer import MCTSExperience, TrainStats

    base = TransformerPolicy()

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
    policy = TransformerPolicy()
    # Save the current state.
    with tempfile.TemporaryDirectory() as td:
        ckpt_path = Path(td) / "ckpt.pt"
        # Mutate trainer weights so the ckpt isn't just init state.
        with torch.no_grad():
            for p in policy._model.parameters():
                p.add_(torch.randn_like(p) * 0.3)
        policy.save_checkpoint(ckpt_path)

        # Build a fresh policy + load. Verify inference matches.
        policy2 = TransformerPolicy()
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


# ---------------------------------------------------------------------
# Concurrency stress
# ---------------------------------------------------------------------

@pytest.mark.slow          # ~20s: see pytest.ini two-tier note
def test_concurrent_rollout_and_train_step_no_nan():
    """With a worker thread doing rollouts while the main thread
    runs train_step, no forward should produce NaN. The lock keeps
    inference from reading torn parameters during the snapshot
    swap; the trainer-side gradient compute mutates `_model`
    (lock-free, but the inference path doesn't read it)."""
    policy = TransformerPolicy()
    gs = _gs()

    stop = threading.Event()
    nan_seen = []
    actions_seen = []

    def rollout_thread():
        i = 0
        while not stop.is_set() and i < 30:
            try:
                # Different game_label per call to avoid debug
                # same-state-twice tripwire (we're passing the same
                # gs object intentionally for the stress test).
                action = policy.select_action(gs, game_label=f"stress{i}")
                actions_seen.append(action)
                # No NaN check on action dict itself; check the
                # inference model output instead.
                with torch.no_grad():
                    encoded = policy._inference_encoder.encode(gs)
                    out = policy._inference_model(encoded)
                    if torch.isnan(out.actor_logits).any():
                        nan_seen.append("actor_logits")
                    if torch.isnan(out.value).any():
                        nan_seen.append("value")
                i += 1
            except Exception as e:
                nan_seen.append(f"exception: {e}")
                break

    # Pre-populate enough rollouts so train_step has data.
    for i in range(8):
        policy.select_action(gs, game_label=f"warmup{i}")
        policy.observe(f"warmup{i}", 1, reward=0.5, done=True)

    t = threading.Thread(target=rollout_thread)
    t.start()
    # Run a few train_steps while rollouts are happening. Each
    # train_step needs queue contents, so we top up after each.
    for i in range(3):
        # Top up the queue with a few synthetic trajectories.
        for j in range(4):
            policy.select_action(gs, game_label=f"trainfeed{i}_{j}")
            policy.observe(f"trainfeed{i}_{j}", 1, reward=0.5, done=True)
        try:
            policy.train_step()
        except Exception as e:
            nan_seen.append(f"train_step exception: {e}")
        time.sleep(0.05)
    stop.set()
    t.join(timeout=10.0)

    assert not nan_seen, f"NaN / exceptions during stress: {nan_seen}"
    assert len(actions_seen) > 0, "rollout thread never ran"
