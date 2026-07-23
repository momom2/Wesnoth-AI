#!/usr/bin/env python3
"""B1 (2026-06-29 review): batched-Gumbel leaf evaluation.

Under the default Gumbel root, `--mcts-batch-size` used to be a silent
no-op -- every sim issued a B=1 model forward. Now each sequential-
halving phase evaluates its sims through one `model.forward_batch`
(virtual-loss parallelization) when `batch_size > 1`, the headline GPU
throughput lever.

The sequential-halving SCHEDULE (num_phases, sims_per, candidate
reduction) is independent of batching -- only the forward grouping
changes -- so the TOTAL sim count is identical at B=1 and B>1. We can't
show a speedup on this CPU laptop (B>1 adds overhead here; the win is
CUDA-only), but we can pin that correctness invariant and that the
batched path produces a valid policy/action.

Drives the REAL mcts_search (no mirroring).
"""
from __future__ import annotations

import random as _random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


def _model_and_sim(seed: int):
    from wesnoth_ai.transformer_policy import TransformerPolicy
    from tools.scenario_pool import (
        random_setup, build_scenario_gamestate, load_factions,
    )
    from tools.wesnoth_sim import WesnothSim
    from sim_test_helpers import require_scenario_data

    require_scenario_data()
    torch.manual_seed(seed)
    pol = TransformerPolicy(d_model=48, num_layers=2, num_heads=4,
                            d_ff=96, device=torch.device("cpu"))
    load_factions()
    setup = random_setup(_random.Random(seed), forced_faction=None)
    gs = build_scenario_gamestate(setup)
    sim = WesnothSim(gs, scenario_id=setup.scenario_id, max_turns=30)
    return pol, sim


def _gumbel_cfg(batch_size: int):
    from tools.mcts import MCTSConfig
    return MCTSConfig(n_simulations=24, gumbel_root=True, gumbel_m=8,
                      batch_size=batch_size, add_root_noise=False)


def test_batched_gumbel_matches_serial_sim_count():
    from tools.mcts import mcts_search

    pol, sim = _model_and_sim(seed=5)

    root_serial = mcts_search(
        sim, pol._inference_model, pol._inference_encoder,
        _gumbel_cfg(1), rng=np.random.default_rng(7))
    root_batched = mcts_search(
        sim, pol._inference_model, pol._inference_encoder,
        _gumbel_cfg(4), rng=np.random.default_rng(7))

    total_serial = sum(e.n_visits for e in root_serial.edges)
    total_batched = sum(e.n_visits for e in root_batched.edges)

    # Same root, same seed -> same candidate set -> same sequential-
    # halving schedule -> identical total sim count regardless of B.
    assert total_serial == total_batched > 0, (
        f"batched total {total_batched} != serial {total_serial}")
    # The batched search still yields a valid improved-policy action.
    assert root_batched.gumbel_action is not None
    assert isinstance(root_batched.gumbel_action, dict)
    assert "type" in root_batched.gumbel_action


def test_batched_gumbel_chooses_a_legal_candidate():
    """The batched Gumbel action must be one of the root's legal edges."""
    from tools.mcts import mcts_search

    pol, sim = _model_and_sim(seed=9)
    root = mcts_search(
        sim, pol._inference_model, pol._inference_encoder,
        _gumbel_cfg(4), rng=np.random.default_rng(3))

    assert root.gumbel_action is not None
    legal = [e.action for e in root.edges]
    assert any(root.gumbel_action is a or root.gumbel_action == a
               for a in legal), "gumbel action must be a legal root edge"
