#!/usr/bin/env python3
"""Tests for the multi-worker rollout path in `sim_self_play.run_iteration`.

Builds on the snapshot+lock design in TransformerPolicy: workers
calling `select_action` / `observe` concurrently must not corrupt
the policy's `_pending` / `_queue` state, must not produce NaN
forwards (the inference snapshot is consistent), and must produce
the same number of trajectories as workers × games.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

import glob
import random
import threading

import pytest


@pytest.fixture
def small_replay_pool() -> list:
    """A handful of bootstrapping replays, filtered to 2-side
    scenarios. Replays with 3+ declared sides spawn extra
    trajectories (one per acting side), which breaks the
    "trajectories == 2 * outcomes" parity assertion the parallel
    correctness tests rely on. The production code-path handles
    >2 sides correctly; we just want a controlled fixture here."""
    import gzip
    import json as _json
    cands = sorted(glob.glob("replays_dataset/*.json.gz"))
    keep: list = []
    for p in cands:
        try:
            with gzip.open(p, "rt", encoding="utf-8") as f:
                data = _json.load(f)
        except Exception:
            continue
        if len(data.get("starting_sides", [])) == 2:
            keep.append(Path(p))
        if len(keep) >= 10:
            break
    if len(keep) < 4:
        pytest.skip("need >=4 two-side replays_dataset/*.json.gz")
    return keep


def _build_policy_and_reward():
    from rewards import WeightedReward
    from transformer_policy import TransformerPolicy
    return TransformerPolicy(), WeightedReward()


def _cost_lookup():
    from sim_self_play import _recruit_cost_lookup
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
    from sim_self_play import run_iteration

    policy, reward_fn = _build_policy_and_reward()
    rng = random.Random(0)
    outcomes = run_iteration(
        policy, small_replay_pool, reward_fn, _cost_lookup(),
        iter_idx=0, games_per_iter=8, max_turns=3,
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
    from sim_self_play import run_iteration

    policy, reward_fn = _build_policy_and_reward()
    rng = random.Random(1)
    run_iteration(
        policy, small_replay_pool, reward_fn, _cost_lookup(),
        iter_idx=0, games_per_iter=4, max_turns=3,
        rng=rng, workers=2,
    )
    # No half-finished trajectories left in _pending.
    assert len(policy._pending) == 0, (
        f"_pending leaked after iteration: {list(policy._pending.keys())}")


def test_parallel_iteration_with_train_step(small_replay_pool):
    """Run the full parallel path INCLUDING train_step. Verifies
    that the queue drains cleanly under the lock and that
    train_step doesn't crash on a multi-worker-fed queue."""
    from sim_self_play import run_iteration

    policy, reward_fn = _build_policy_and_reward()
    rng = random.Random(2)
    outcomes = run_iteration(
        policy, small_replay_pool, reward_fn, _cost_lookup(),
        iter_idx=0, games_per_iter=4, max_turns=3,
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

def test_concurrent_train_step_during_rollouts(small_replay_pool):
    """Spawn rollout workers AND fire train_step from a separate
    thread mid-rollout. The snapshot lock should keep everything
    safe: no NaN, no exceptions, train_step processes the queue
    contents at the moment it ran."""
    from sim_self_play import run_iteration
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
        iter_idx=-1, games_per_iter=8, max_turns=3,
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
        iter_idx=0, games_per_iter=8, max_turns=3,
        rng=rng, workers=4,
    )
    stop_evt.set()
    t.join(timeout=20.0)

    assert not train_errs, f"train_step threw: {train_errs}"
    assert not nan_seen, f"NaN detected: {nan_seen}"
    assert len(train_results) >= 1, "train_step never fired during stress"


# ---------------------------------------------------------------------
# Lock contracts: observe / drop_pending / register_names
# ---------------------------------------------------------------------

def test_observe_is_thread_safe():
    """Many threads concurrently observing into the same policy
    don't corrupt _pending / _queue. Uses the simplest possible
    sequence: select_action -> observe(done=True) repeated."""
    import torch
    from rewards import WeightedReward
    from transformer_policy import TransformerPolicy
    from classes import (
        Alignment, Attack, DamageType, GameState, GlobalInfo, Hex, Map,
        Position, SideInfo, Terrain, Unit,
    )
    import copy

    policy = TransformerPolicy()
    # Build a minimal valid GameState that select_action can encode.
    units = {
        Unit(id="ldr1", name="Spearman", name_id=0, side=1,
             is_leader=True, position=Position(0, 0),
             max_hp=40, max_moves=5, max_exp=32, cost=14,
             alignment=Alignment.NEUTRAL, levelup_names=[],
             current_hp=40, current_moves=5, current_exp=0,
             has_attacked=False,
             attacks=[Attack(type_id=DamageType.PIERCE,
                             number_strikes=3, damage_per_strike=7,
                             is_ranged=False, weapon_specials=set())],
             resistances=[1.0]*6, defenses=[0.5]*14,
             movement_costs=[1]*14, abilities=set(), traits=set(),
             statuses=set()),
        Unit(id="ldr2", name="Spearman", name_id=0, side=2,
             is_leader=True, position=Position(8, 8),
             max_hp=40, max_moves=5, max_exp=32, cost=14,
             alignment=Alignment.NEUTRAL, levelup_names=[],
             current_hp=40, current_moves=5, current_exp=0,
             has_attacked=False, attacks=[],
             resistances=[1.0]*6, defenses=[0.5]*14,
             movement_costs=[1]*14, abilities=set(), traits=set(),
             statuses=set()),
    }
    hexes = {Hex(position=Position(x, y),
                 terrain_types={Terrain.FLAT}, modifiers=set())
             for x in range(10) for y in range(10)}
    sides = [SideInfo(player=f"S{i+1}", recruits=[], current_gold=100,
                      base_income=2, nb_villages_controlled=0)
             for i in range(2)]
    base_gs = GameState(
        game_id="t",
        map=Map(size_x=10, size_y=10, mask=set(), fog=set(),
                hexes=hexes, units=units),
        global_info=GlobalInfo(current_side=1, turn_number=1,
                               time_of_day="day", village_gold=2,
                               village_upkeep=1, base_income=2),
        sides=sides,
    )

    errs = []

    def worker(worker_id):
        try:
            for i in range(20):
                gs = copy.deepcopy(base_gs)
                label = f"w{worker_id}_g{i}"
                policy.select_action(gs, game_label=label)
                policy.observe(label, 1, reward=0.5, done=True)
        except Exception as e:
            errs.append(e)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errs, f"worker threads errored: {errs}"
    # 4 workers * 20 games = 80 sealed trajectories on the queue.
    assert len(policy._queue) == 80, (
        f"expected 80 trajectories, got {len(policy._queue)}")
    # No leaks in _pending.
    assert len(policy._pending) == 0
