"""MCTSExperience.policy_weight (2026-08-26): the magnitude channel
for scaled policy targets. Scaling visit COUNTS cancels (the loss
divides by total visits -- counts are a distribution); policy_weight
multiplies AFTER that normalization and must scale the update
linearly. A regression here silently defeats any margin-scaled /
conservative-mixture distillation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_test_helpers import fresh_scenario_sim  # noqa: E402
from wesnoth_ai.trainer import MCTSExperience  # noqa: E402
from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402


def _tiny():
    torch.manual_seed(0)
    return TransformerPolicy(device=torch.device("cpu"), d_model=32,
                             num_layers=1, num_heads=4, d_ff=64)


def _step(policy_weight: float, count: float) -> float:
    net = _tiny()
    sim = fresh_scenario_sim()
    exp = MCTSExperience(
        game_state=sim.gs,
        visit_counts=[(0, None, None, count, None)],
        z=1.0, policy_weight=policy_weight)
    return net._trainer.step_mcts([exp]).policy_loss


def test_policy_weight_scales_policy_loss_linearly():
    full = _step(policy_weight=1.0, count=1.0)
    half = _step(policy_weight=0.5, count=1.0)
    assert full != 0.0
    assert abs(half / full - 0.5) < 1e-5


def test_visit_count_scaling_cancels_by_design():
    """Counts are a distribution: uniformly scaling them must NOT
    change the loss. This pins the normalization that makes
    policy_weight the only sound magnitude channel."""
    full = _step(policy_weight=1.0, count=1.0)
    scaled = _step(policy_weight=1.0, count=0.5)
    assert abs(scaled - full) < 1e-6
