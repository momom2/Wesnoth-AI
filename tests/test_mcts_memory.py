"""MCTS over a model with a memory (tools/mcts.py, tools/mcts_policy.py):
every node's evaluation reads the memory its side holds there, what the
last node of that side on its path wrote or else the side's memory at the
root, and the root's write is the decision's memory, undone when the
decision bounces."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_test_helpers import fresh_scenario_sim  # noqa: E402
from wesnoth_ai import game_core as gc  # noqa: E402
from wesnoth_ai.game_core import snapshot_view  # noqa: E402
from wesnoth_ai.memory import MemoryState  # noqa: E402

needs_core = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")
SLOTS = 8


def _policy():
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(0)
    return TransformerPolicy(relevant_set_hexes=True, observation_parity=True, memory_slots=SLOTS,
                             relevant_set_version=2, device=torch.device("cpu"),
                             d_model=32, num_layers=1, num_heads=2, d_ff=64)


def _side_1_can_only_end_its_turn(sim):
    """Side 1 without gold and its leader without moves: its one action
    is end_turn, so the search crosses into side 2's turn at once."""
    gs = sim.gs
    leader = next(u for u in gs.map.units if u.side == 1 and u.is_leader)
    gs.map.units = (gs.map.units - {leader}) | {dataclasses.replace(leader, current_moves=0)}
    gs.sides[0] = dataclasses.replace(gs.sides[0], current_gold=0)
    sim.gs = gs
    return sim


@needs_core
def test_every_node_reads_the_memory_its_side_holds_on_its_path():
    from tools.mcts import MCTSConfig, MemoryContext, mcts_search
    policy = _policy()
    model, encoder = policy._inference_model, policy._inference_encoder
    sim = _side_1_can_only_end_its_turn(fresh_scenario_sim(seed=5, max_turns=10))
    side_2_root = model.initial_memory(SLOTS) + 0.1
    root = mcts_search(sim, model, encoder, MCTSConfig(n_simulations=24, batch_size=4),
                       rng=np.random.default_rng(0),
                       memory=MemoryContext(SLOTS, {1: None, 2: side_2_root}))
    checked = {1: 0, 2: 0}

    def walk(node, held):
        if node.memory_out is not None:
            with torch.no_grad():
                want = model(encoder.encode(node.sim.gs), memory=MemoryState(SLOTS, held[node.side])).memory
            torch.testing.assert_close(node.memory_out, want)
            checked[node.side] += 1
            held = {**held, node.side: node.memory_out}
        for edge in node.edges:
            for child in edge.children.values():
                walk(child, held)

    walk(root, {1: None, 2: side_2_root})
    assert checked[1] >= 1 and checked[2] >= 2, checked


@needs_core
def test_the_decisions_memory_is_the_roots_write_and_a_bounce_undoes_it():
    from tools.mcts import MCTSConfig
    from tools.mcts_policy import MCTSPolicy
    policy = _policy()
    player = MCTSPolicy(policy, MCTSConfig(n_simulations=4), memory_slots=SLOTS, rng_seed=0)
    sim = fresh_scenario_sim(seed=5, max_turns=10)
    snapshot = snapshot_view(sim.gs)
    player.select_action(snapshot, game_label="g", sim=sim)
    with torch.no_grad():
        want = policy._inference_model(policy._inference_encoder.encode(snapshot),
                                       memory=MemoryState(SLOTS, None)).memory
    torch.testing.assert_close(player._memories[("g", 1)], want)
    player.drop_last_pending("g")
    assert ("g", 1) not in player._memories
    player.select_action(snapshot_view(sim.gs), game_label="g", sim=sim)
    player.drop_pending("g")
    assert not player._memories
