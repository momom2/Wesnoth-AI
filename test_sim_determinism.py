#!/usr/bin/env python3
"""Sim determinism test.

Two runs with the same starting state and the same action sequence
MUST produce identical end states. If they don't, MCTS branching is
broken (different nodes give different outcomes for the same path),
self-play training is non-reproducible, and replay export is unreliable
(the export from one run won't match a re-run).

The interesting determinism sources to verify:

  - **Combat damage rolls** (`combat.MTRng` seeded by
    `request_seed(N)`). The sim's `_rng_requests` counter increments
    once per attack; both runs request the same seed for attack #N.

  - **Trait rolls on recruit** (same plumbing, separate counter slot
    in the recruit cmd).

  - **Unit set iteration order**. `gs.map.units` is a hash-set so its
    iteration order depends on Python hash randomization. The sim's
    logic must NOT depend on iteration order; if it does, two runs
    diverge silently.

This test uses the deterministic `DummyPolicy` (sorts units by id
before iterating, so its action choices ARE reproducible across
runs). Comparing two sim runs end-to-end via `classes.state_key` --
the canonical content hash -- is a strong consistency check: any
unit position / HP / MP / XP / status / village ownership / sides
gold / RNG counter divergence produces a different key.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

import glob
import pytest

from classes import state_key
from dummy_policy import DummyPolicy
from tools.wesnoth_sim import WesnothSim


def _pick_sample_replay() -> Path:
    """A small dataset replay to bootstrap from."""
    candidates = sorted(glob.glob("replays_dataset/*.json.gz"))
    if not candidates:
        pytest.skip("no replays_dataset/ to bootstrap from")
    return Path(candidates[0])


def _run_dummy_game(replay: Path, max_turns: int = 6) -> WesnothSim:
    """Drive a sim from `replay` to completion (or `max_turns`) using
    DummyPolicy for both sides. Returns the finished sim so the caller
    can inspect command_history and gs."""
    sim = WesnothSim.from_replay(replay, max_turns=max_turns)
    pol = DummyPolicy()
    while not sim.done:
        action = pol.select_action(sim.gs, game_label="det")
        sim.step(action)
    return sim


def test_two_runs_same_state_key():
    """Two independent runs with DummyPolicy from the same starting
    state produce the same `state_key` at termination.

    state_key sums over:
      - unit (id, side, position, hp, mp, exp, statuses, name, leader)
      - sides (faction, gold, income, villages, recruits)
      - village ownership
      - global (turn, ToD, gold, upkeep, RNG counter)

    so any divergence in any of these fields makes the keys differ.
    """
    replay = _pick_sample_replay()
    sim_a = _run_dummy_game(replay)
    sim_b = _run_dummy_game(replay)
    key_a = state_key(sim_a.gs)
    key_b = state_key(sim_b.gs)
    assert key_a == key_b, (
        f"state_key differs: {key_a} vs {key_b}. The sim is NOT "
        f"deterministic; MCTS / self-play results will be unreliable.")


def test_command_history_identical():
    """The recorded command_history must match byte-for-byte (modulo
    Python object identity) between two runs. This is finer-grained
    than state_key -- catches divergences that happen mid-game even
    if they cancel out by termination."""
    replay = _pick_sample_replay()
    sim_a = _run_dummy_game(replay)
    sim_b = _run_dummy_game(replay)
    assert len(sim_a.command_history) == len(sim_b.command_history), (
        f"history lengths differ: {len(sim_a.command_history)} vs "
        f"{len(sim_b.command_history)}")
    for i, (a, b) in enumerate(zip(sim_a.command_history,
                                   sim_b.command_history)):
        assert a.kind == b.kind, f"cmd {i}: kind {a.kind!r} vs {b.kind!r}"
        assert a.side == b.side, f"cmd {i}: side {a.side} vs {b.side}"
        assert a.cmd == b.cmd, f"cmd {i}: cmd {a.cmd!r} vs {b.cmd!r}"
        assert a.extras == b.extras, f"cmd {i}: extras differ"


def test_rng_counter_advances_consistently():
    """`_rng_requests` ends at the same value in both runs -- confirms
    the synced-RNG plumbing is in lockstep. Also serves as a regression
    against accidental seed-fetches from non-deterministic sources."""
    replay = _pick_sample_replay()
    sim_a = _run_dummy_game(replay)
    sim_b = _run_dummy_game(replay)
    assert sim_a._rng_requests == sim_b._rng_requests, (
        f"_rng_requests differs: {sim_a._rng_requests} vs "
        f"{sim_b._rng_requests}")


def test_fork_does_not_perturb_parent():
    """`WesnothSim.fork()` produces a clone for MCTS branching. After
    the fork advances independently, the parent sim's state must be
    unchanged. If fork-then-step bleeds state into the parent, MCTS
    rollouts contaminate each other."""
    replay = _pick_sample_replay()
    sim = WesnothSim.from_replay(replay, max_turns=6)
    parent_pre_key = state_key(sim.gs)

    fork = sim.fork()
    pol = DummyPolicy()
    # Run several actions on the fork only.
    for _ in range(8):
        if fork.done:
            break
        a = pol.select_action(fork.gs, game_label="fork")
        fork.step(a)

    parent_post_key = state_key(sim.gs)
    assert parent_pre_key == parent_post_key, (
        f"parent sim state mutated by fork's steps: "
        f"pre={parent_pre_key} post={parent_post_key}")
