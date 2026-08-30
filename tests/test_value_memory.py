"""Value-memory reservoir (user ruling 2026-08-30): the value head
trains on a per-GAME outcome memory spanning many iterations, because
its noise scales with independent game outcomes per fit (the arm-T
oscillation diagnosis; weight-averaging confirmed the catastrophic
component is noise). These pin the reservoir semantics and that the
step actually trains the value head.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_test_helpers import fresh_scenario_sim  # noqa: E402
from tools.mcts import MCTSConfig  # noqa: E402
from tools.mcts_policy import MCTSPolicy, ReplayConfig  # noqa: E402
from wesnoth_ai.trainer import MCTSExperience  # noqa: E402
from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402


def _policy(value_memory_games=4, cap=3):
    torch.manual_seed(0)
    net = TransformerPolicy(device=torch.device("cpu"), d_model=32,
                            num_layers=1, num_heads=4, d_ff=64)
    return MCTSPolicy(net, MCTSConfig(n_simulations=2),
                      ReplayConfig(enabled=False),
                      value_memory_games=value_memory_games,
                      value_memory_states_per_game=cap)


def _exps(gid: str, n: int, gs, z=1.0, value_weight=1.0):
    return [MCTSExperience(game_state=gs, visit_counts=[],
                           z=z, value_weight=value_weight,
                           game_id=gid)
            for _ in range(n)]


def test_reservoir_caps_per_game_and_evicts_fifo():
    pol = _policy(value_memory_games=2, cap=3)
    gs = fresh_scenario_sim().gs
    pol._value_memory_ingest(_exps("g1", 10, gs))
    assert len(pol._value_memory["g1"]) == 3, "stride-thinned to cap"
    pol._value_memory_ingest(_exps("g2", 2, gs))
    pol._value_memory_ingest(_exps("g3", 2, gs))
    assert "g1" not in pol._value_memory, "oldest game evicted"
    assert set(pol._value_memory) == {"g2", "g3"}


def test_censored_and_legacy_states_never_enter():
    pol = _policy()
    gs = fresh_scenario_sim().gs
    pol._value_memory_ingest(_exps("cens", 4, gs, z=0.0,
                                   value_weight=0.0))
    pol._value_memory_ingest(_exps("", 4, gs))   # legacy: no game_id
    assert not pol._value_memory


def test_value_memory_step_trains_value_head_only():
    pol = _policy()
    gs = fresh_scenario_sim().gs
    pol._value_memory_ingest(_exps("g1", 4, gs, z=1.0))
    pol._value_memory_ingest(_exps("g2", 4, gs, z=-1.0))
    net = pol._base
    v_before = net._model.value_head[0].weight.detach().clone()
    p_before = net._model.actor_head.weight.detach().clone()
    stats = pol.value_memory_step(batch_size=8)
    assert stats["memory_games"] == 2
    assert stats["value_loss"] > 0.0
    assert not torch.equal(
        net._model.value_head[0].weight.detach(), v_before), \
        "value head must receive gradient"
    assert torch.equal(
        net._model.actor_head.weight.detach(), p_before), \
        "policy head must NOT receive gradient from the value step"


def test_off_by_default_is_inert():
    pol = _policy(value_memory_games=0)
    gs = fresh_scenario_sim().gs
    pol._value_memory_ingest(_exps("g1", 4, gs))
    assert not pol._value_memory
    assert pol.value_memory_step() == {}
