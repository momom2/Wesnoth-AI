"""Gradient-tree smoke (tiny net, CPU, synthetic experiences).
Deliberately outside the main suite's testpaths — run with
`pytest signal_profiler/tests -q`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

from sim_test_helpers import fresh_scenario_sim  # noqa: E402
from signal_profiler.gradient_tree import build_tree  # noqa: E402
from wesnoth_ai.trainer import MCTSExperience  # noqa: E402


def _factory():
    from tools.mcts import MCTSConfig
    from tools.mcts_policy import MCTSPolicy, ReplayConfig
    from wesnoth_ai.transformer_policy import TransformerPolicy

    def make():
        torch.manual_seed(7)
        base = TransformerPolicy(device=torch.device("cpu"),
                                 d_model=32, num_layers=1,
                                 num_heads=4, d_ff=64, gbc=True,
                                 aux_score=True)
        return MCTSPolicy(base, MCTSConfig(n_simulations=2),
                          ReplayConfig(enabled=False),
                          gbc_labels=True,
                          value_memory_games=8,
                          value_memory_states_per_game=8)
    return make


def _batch(gs, n=6):
    out = []
    for i in range(n):
        out.append(MCTSExperience(
            game_state=gs,
            visit_counts=[(0, None, None, 3.0, None)],
            z=1.0 if i % 2 == 0 else -1.0,
            aux_target=0.2,
            gbc_labels=None,
            game_id=f"g{i % 2}",
        ))
    return out


def test_tree_builds_and_terms_isolate():
    gs = fresh_scenario_sim().gs
    tree = build_tree(_factory(), _batch(gs))
    terms = tree["terms"]
    assert "total" in terms and terms["total"]["norm"] > 0
    # Policy and value terms both carry real amplitude.
    assert terms["policy_distill"]["norm"] > 0
    assert terms["value_inbatch"]["norm"] > 0
    # gbc term must be ~0: the batch carries no gbc labels, so its
    # isolated step has nothing to push (isolation actually works).
    assert terms["gbc"]["norm"] < 1e-6
    # The value-only term must not touch the policy heads.
    v_groups = terms["value_inbatch"]["groups"]
    assert v_groups["actor_head"]["norm"] < 1e-6
    assert v_groups["value_head"]["norm"] > 0
    # Value-memory step ran (memory enabled, decisive games present)
    # and is head-only by construction.
    assert "value_memory" in terms
    vm = terms["value_memory"]["groups"]
    assert vm["value_head"]["norm"] > 0
    assert vm["encoder"]["norm"] < 1e-6
    # proj_frac of the total against itself is 1.
    assert abs(terms["total"]["proj_frac"] - 1.0) < 1e-5


def test_render_smoke():
    from signal_profiler.render import render_tree
    gs = fresh_scenario_sim().gs
    tree = build_tree(_factory(), _batch(gs),
                      include_value_memory=False)
    txt = render_tree(tree)
    assert "TOTAL gradient" in txt and "policy_distill" in txt
