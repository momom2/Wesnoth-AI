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

from classes import state_key
from dummy_policy import DummyPolicy
from sim_test_helpers import fresh_scenario_sim, twin_scenario_sims
from tools.wesnoth_sim import WesnothSim


def _drive_to_end(sim: WesnothSim) -> WesnothSim:
    """Drive a sim to completion (or its turn cap) using DummyPolicy
    for both sides; DummyPolicy is deterministic (sorts units by id),
    so identical starting states yield identical action sequences."""
    pol = DummyPolicy()
    while not sim.done:
        action = pol.select_action(sim.gs, game_label="det")
        sim.step(action)
    return sim


def test_twin_runs_identical():
    """ONE twin pair, all the internal-determinism facets asserted
    together: terminal state_key, byte-level command history, and
    the synced-RNG request counter.

    Scope note (user discussion 2026-06-12): sim-vs-sim agreement is
    NECESSARY (MCTS chance nodes / tree reuse assume identical state
    + identical action => identical outcome) but NOT SUFFICIENT for
    correctness — it is structurally blind to implicit-RNG leaks
    where the recorded stream leaves a resolution for Wesnoth to
    roll at playback (the d_weapon=-1 retaliation bug was exactly
    that, and twins agreed perfectly through it). The contract that
    nothing is left to Wesnoth lives in test_rng_accounting.py; the
    end-state oracle (actual Wesnoth playback verification) is in
    BACKLOG. This file keeps exactly one cheap test for the
    internal-determinism property and the fork-isolation test."""
    sim_a, sim_b = twin_scenario_sims(seed=3, max_turns=6, mini=True)
    _drive_to_end(sim_a)
    _drive_to_end(sim_b)

    key_a, key_b = state_key(sim_a.gs), state_key(sim_b.gs)
    assert key_a == key_b, (
        f"state_key differs: {key_a} vs {key_b}. The sim is NOT "
        f"deterministic; MCTS / self-play results will be unreliable.")

    assert len(sim_a.command_history) == len(sim_b.command_history), (
        f"history lengths differ: {len(sim_a.command_history)} vs "
        f"{len(sim_b.command_history)}")
    for i, (a, b) in enumerate(zip(sim_a.command_history,
                                   sim_b.command_history)):
        assert a.kind == b.kind, f"cmd {i}: kind {a.kind!r} vs {b.kind!r}"
        assert a.side == b.side, f"cmd {i}: side {a.side} vs {b.side}"
        assert a.cmd == b.cmd, f"cmd {i}: cmd {a.cmd!r} vs {b.cmd!r}"
        assert a.extras == b.extras, f"cmd {i}: extras differ"

    assert sim_a._rng_requests == sim_b._rng_requests, (
        f"_rng_requests differs: {sim_a._rng_requests} vs "
        f"{sim_b._rng_requests}")


def test_fork_does_not_perturb_parent():
    """`WesnothSim.fork()` produces a clone for MCTS branching. After
    the fork advances independently, the parent sim's state must be
    unchanged. If fork-then-step bleeds state into the parent, MCTS
    rollouts contaminate each other."""
    sim = fresh_scenario_sim(seed=6, max_turns=6, mini=True)
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
