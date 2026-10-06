#!/usr/bin/env python3
"""Tests for the multi-worker rollout path in `sim_self_play.run_iteration`.

Builds on the snapshot+lock design in TransformerPolicy: workers
calling `select_action` / `observe` concurrently must not corrupt
the policy's `_pending` / `_queue` state, must not produce NaN
forwards (the inference snapshot is consistent), and must produce
the same number of trajectories as workers Ã— games.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

import random
import threading

import pytest


@pytest.fixture
def small_replay_pool() -> None:
    """Legacy fixture name, now a None pool marker: `run_iteration`
    ignores its pool argument and seeds games from
    wesnoth_ai.rules.scenario_pool (scenarios are always exactly 2 player
    sides, so the "trajectories == 2 * outcomes" parity assertions
    hold by construction). The fixture survives only to guard on
    the vendored scenario data being present."""
    from sim_test_helpers import require_scenario_data
    require_scenario_data()
    return None


def _build_policy_and_reward():
    """Tiny model: these tests exercise THREADING contracts (queue
    parity, pending leaks, concurrent train_step), not model
    quality. Default-size forwards on CPU made the file take
    minutes; d=64/L=2 plus mini maps keeps it in seconds."""
    import torch
    from wesnoth_ai.rewards import WeightedReward
    from wesnoth_ai.transformer_policy import TransformerPolicy
    policy = TransformerPolicy(d_model=64, num_layers=2, num_heads=4,
                               d_ff=128, device=torch.device("cpu"))
    return policy, WeightedReward()


def _cost_lookup():
    from tools.selfplay_game import _recruit_cost_lookup
    return _recruit_cost_lookup()


# ---------------------------------------------------------------------
# Behavioral parity: parallel == serial (modulo RNG order)
# ---------------------------------------------------------------------

def test_parallel_iteration_queue_parity(small_replay_pool):
    """workers=4 + games_per_iter=8: every successfully-completed
    game contributes exactly 2 trajectories (one per side) to the
    queue, and no half-finished trajectory is left in _pending.

    We don't insist on 8 successful outcomes -- the underlying sim
    can legitimately crash on some replays (a separate, pre-existing
    bug); when it does, `_play_one_game_safe` calls
    `policy.drop_pending`, which is itself a thread-safety contract
    we want to exercise. What matters HERE is the parity between
    outcomes and queue size: that's the signal that no trajectory
    was lost or duplicated under concurrent workers."""
    from tools.sim_self_play import run_iteration

    policy, reward_fn = _build_policy_and_reward()
    rng = random.Random(0)
    outcomes = run_iteration(
        policy, small_replay_pool, reward_fn, _cost_lookup(),
        iter_idx=0, games_per_iter=8, max_turns=3, mini_maps=True,
        rng=rng, workers=4, train_at_end=False,
    )
    n_traj = len(policy._queue)
    assert n_traj == 2 * len(outcomes), (
        f"queue/outcome parity broken: {n_traj} trajectories vs "
        f"{len(outcomes)} outcomes (expected {2 * len(outcomes)})")
    # No half-finished trajectories left in _pending: drop_pending
    # on crash and observe(done=True) on success both clear it.
    assert len(policy._pending) == 0, (
        f"_pending leaked after iteration: {list(policy._pending.keys())}")


def test_parallel_iteration_no_pending_leaks(small_replay_pool):
    """After all workers finish, _pending should be empty (every
    started trajectory got a terminal observe OR drop_pending on
    crash)."""
    from tools.sim_self_play import run_iteration

    policy, reward_fn = _build_policy_and_reward()
    rng = random.Random(1)
    run_iteration(
        policy, small_replay_pool, reward_fn, _cost_lookup(),
        iter_idx=0, games_per_iter=4, max_turns=3, mini_maps=True,
        rng=rng, workers=2,
    )
    # No half-finished trajectories left in _pending.
    assert len(policy._pending) == 0, (
        f"_pending leaked after iteration: {list(policy._pending.keys())}")


def test_parallel_iteration_with_train_step(small_replay_pool):
    """Run the full parallel path INCLUDING train_step. Verifies
    that the queue drains cleanly under the lock and that
    train_step doesn't crash on a multi-worker-fed queue."""
    from tools.sim_self_play import run_iteration

    policy, reward_fn = _build_policy_and_reward()
    rng = random.Random(2)
    outcomes = run_iteration(
        policy, small_replay_pool, reward_fn, _cost_lookup(),
        iter_idx=0, games_per_iter=4, max_turns=3, mini_maps=True,
        rng=rng, workers=2, train_at_end=False,
    )
    stats = policy.train_step()
    # n_trajectories drained == 2 per successful game.
    assert stats.n_trajectories == 2 * len(outcomes)
    if outcomes:
        assert stats.n_transitions > 0
    # Queue empty after train.
    assert len(policy._queue) == 0


# ---------------------------------------------------------------------
# Concurrency stress: train_step DURING rollout
# ---------------------------------------------------------------------

@pytest.mark.slow          # ~28s: see pytest.ini two-tier note
def test_concurrent_train_step_during_rollouts(small_replay_pool):
    """Spawn rollout workers AND fire train_step from a separate
    thread mid-rollout. The snapshot lock should keep everything
    safe: no NaN, no exceptions, train_step processes the queue
    contents at the moment it ran."""
    from tools.sim_self_play import run_iteration
    import torch

    policy, reward_fn = _build_policy_and_reward()
    rng = random.Random(3)

    # Pre-warm with a few trajectories so train_step has data.
    # Use enough games that at least some survive sim crashes (a
    # known unrelated bug), so train_step has something to chew on.
    # train_at_end=False so the queue keeps the pre-warm
    # trajectories for the stress thread to consume.
    run_iteration(
        policy, small_replay_pool, reward_fn, _cost_lookup(),
        iter_idx=-1, games_per_iter=8, max_turns=3, mini_maps=True,
        rng=rng, workers=0, train_at_end=False,    # serial pre-warm
    )
    if len(policy._queue) == 0:
        pytest.skip("pre-warm produced no trajectories (every game "
                    "hit the sim invariant bug)")

    # Now: kick off a parallel rollout AND train_step from the main
    # thread while workers are still running.
    train_results = []
    train_errs = []
    nan_seen = []

    def _train_in_a_loop(stop_evt):
        while not stop_evt.is_set():
            try:
                stats = policy.train_step()
                train_results.append(stats)
                # Check inference model isn't NaN.
                with torch.no_grad():
                    for p in policy._inference_model.parameters():
                        if torch.isnan(p).any():
                            nan_seen.append("inference_model NaN")
                            return
            except Exception as e:
                train_errs.append(e)
                return

    stop_evt = threading.Event()
    t = threading.Thread(target=_train_in_a_loop, args=(stop_evt,))
    t.start()

    # Fire a parallel rollout with workers feeding the queue.
    run_iteration(
        policy, small_replay_pool, reward_fn, _cost_lookup(),
        iter_idx=0, games_per_iter=8, max_turns=3, mini_maps=True,
        rng=rng, workers=4,
    )
    stop_evt.set()
    t.join(timeout=20.0)

    assert not train_errs, f"train_step threw: {train_errs}"
    assert not nan_seen, f"NaN detected: {nan_seen}"
    assert len(train_results) >= 1, "train_step never fired during stress"
