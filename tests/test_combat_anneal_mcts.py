#!/usr/bin/env python3
"""C7 (2026-06-29 review): the combat-oracle bias must anneal in MCTS
mode. Previously decision_step was frozen in the MCTS path (only the
REINFORCE select_action incremented it), and the search + distillation
loss built legality masks at decision_step=0 (full-strength oracle for
the whole run). Now MCTSPolicy.select_action advances the counter per
decision and records it on each training target, so the loss rebuilds
reference logits at the SAME annealed alpha the search used.

Drives the REAL MCTSPolicy.select_action / finalize_game (no mirroring).
"""
from __future__ import annotations

import copy
import random as _random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from wesnoth_ai.action_sampler import combat_alphas_at   # noqa: E402
from wesnoth_ai.constants import COMBAT_ANNEAL_HORIZON    # noqa: E402


def test_anneal_schedule_actually_decays():
    """Sanity: the anneal MACHINERY decays alpha with step. The
    oracle is RETIRED (user 2026-07-16: configured alphas 0.0 make
    every bias zero), so the decay property is pinned against
    synthetic nonzero alphas -- the machinery must stay intact for
    a future un-retirement."""
    t0, y0 = combat_alphas_at(0)
    t1, y1 = combat_alphas_at(COMBAT_ANNEAL_HORIZON)
    if t0 == 0.0 and y0 == 0.0:
        # Retired config: bias is exactly zero at every step.
        assert t1 == y1 == 0.0
        from wesnoth_ai import action_sampler as asamp
        import unittest.mock as mock
        with mock.patch.object(asamp, "COMBAT_TARGET_ALPHA", 0.1), \
                mock.patch.object(asamp, "COMBAT_TYPE_ALPHA", 0.1):
            t0, y0 = combat_alphas_at(0)
            t1, y1 = combat_alphas_at(COMBAT_ANNEAL_HORIZON)
    assert t1 < t0 and y1 < y0, "anneal machinery must decay with step"


def _mcts_policy_and_sim(seed: int):
    from wesnoth_ai.transformer_policy import TransformerPolicy
    from tools.mcts import MCTSConfig
    from tools.mcts_policy import MCTSPolicy, ReplayConfig
    from tools.scenario_pool import (
        random_setup, build_scenario_gamestate, load_factions,
    )
    from tools.wesnoth_sim import WesnothSim
    from sim_test_helpers import require_scenario_data

    require_scenario_data()
    torch.manual_seed(seed)
    base = TransformerPolicy(d_model=48, num_layers=2, num_heads=4,
                             d_ff=96, device=torch.device("cpu"))
    load_factions()
    setup = random_setup(_random.Random(seed), forced_faction=None,
                         mini_maps=True)
    gs = build_scenario_gamestate(setup)
    sim = WesnothSim(gs, scenario_id=setup.scenario_id, max_turns=12)
    cfg = MCTSConfig(n_simulations=8, add_root_noise=False)
    pol = MCTSPolicy(base, cfg, replay_config=ReplayConfig(enabled=False))
    return base, pol, sim


def test_decision_step_advances_and_is_recorded():
    base, pol, sim = _mcts_policy_and_sim(seed=3)
    assert base._decision_step == 0, "fresh policy starts at step 0"

    captured = []
    n_decisions = 5
    for _ in range(n_decisions):
        if sim.done:
            break
        snap = copy.deepcopy(sim.gs)
        before = base._decision_step
        action = pol.select_action(snap, game_label="g", sim=sim)
        # Each select_action advances the counter by exactly one.
        assert base._decision_step == before + 1
        captured.append(before)
        sim.step(action)

    assert len(captured) >= 2, "need a couple of decisions to test"
    # Counter advanced once per decision, contiguous from 0.
    assert captured == list(range(len(captured)))
    assert base._decision_step == len(captured)

    # finalize_game stamps each recorded state's decision_step onto its
    # MCTSExperience (so the loss can match the search's alpha).
    pol.finalize_game("g", winner=0, final_gs=sim.gs)
    steps = [e.decision_step for e in pol._queue]
    assert steps, "expected queued experiences"
    # Every recorded experience's step is one of the captured values, in
    # the same increasing order they were generated.
    assert steps == sorted(steps)
    assert set(steps).issubset(set(captured))
