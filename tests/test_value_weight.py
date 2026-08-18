"""Truncation ruling, training side (user 2026-08-17: "there are no
draws in real Wesnoth"): winnerless games are CENSORED for the value
head -- sealed with value_weight 0 -- while their policy targets keep
full weight. A silent regression here re-poisons the grader with the
z=0 flood that killed leg 3, so both halves are pinned on production
paths: the finalize_game seal and the trainer's value-weight
arithmetic (via the starvation-watch counter).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_test_helpers import fresh_scenario_sim  # noqa: E402
from tools.mcts import MCTSConfig  # noqa: E402
from tools.mcts_policy import MCTSPolicy, _PendingMCTSState  # noqa: E402
from wesnoth_ai.trainer import MCTSExperience  # noqa: E402
from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402


def _tiny():
    torch.manual_seed(0)
    return TransformerPolicy(device=torch.device("cpu"), d_model=32,
                             num_layers=1, num_heads=4, d_ff=64)


def _seal(policy, label, winner):
    sim = fresh_scenario_sim()
    with policy._lock:
        policy._pending[label] = [
            _PendingMCTSState(gs=sim.gs,
                              visit_counts=[(0, None, None, 1.0, None)],
                              side=1, decision_step=0),
            _PendingMCTSState(gs=sim.gs,
                              visit_counts=[(0, None, None, 1.0, None)],
                              side=2, decision_step=0),
        ]
    policy.finalize_game(label, winner, final_gs=sim.gs)
    with policy._lock:
        exps = [e for e in policy._queue]
        policy._queue = []
    return exps


def test_winnerless_games_seal_value_weight_zero():
    base = _tiny()
    policy = MCTSPolicy(base, MCTSConfig())
    draws = _seal(policy, "g_draw", winner=0)
    wins = _seal(policy, "g_win", winner=1)
    assert draws and all(e.value_weight == 0.0 for e in draws)
    assert wins and all(e.value_weight == 1.0 for e in wins)
    # Policy-side weight untouched: censoring is value-only.
    assert all(e.game_weight > 0 for e in draws)


def test_trainer_value_arithmetic_ignores_censored_states():
    """The starvation watch must count only states that actually
    feed the value head: censored (value_weight 0) states vanish
    from it even though they train the policy."""
    policy = _tiny()
    sim = fresh_scenario_sim()
    vc = [(0, None, None, 1.0, None)]
    exps = (
        [MCTSExperience(game_state=sim.gs, visit_counts=vc, z=1.0)] * 3
        + [MCTSExperience(game_state=sim.gs, visit_counts=vc, z=0.0,
                          value_weight=0.0)] * 5
    )
    stats = policy._trainer.step_mcts(exps)
    assert stats.value_signal_states == 3
