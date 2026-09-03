"""TrainerConfig.value_loss_form = "mse_mean" (docs/az_minimal_spec.md).

The squared-error form must charge in proportion to the miss: zero
gradient when the head's mean already equals the label, and a
gradient that scales with the gap -- unlike the categorical loss,
which charges a confident head heavily for a coin-flip label.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from test_inference_snapshot import _gs  # noqa: E402
from wesnoth_ai.trainer import MCTSExperience  # noqa: E402
from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402


def _value_grad_norm(policy, z):
    tr = policy._trainer
    tr.optimizer.zero_grad()
    real_step = tr.optimizer.step
    tr.optimizer.step = lambda *a, **k: None
    try:
        exps = [MCTSExperience(game_state=_gs(), visit_counts=[], z=z,
                               policy_weight=0.0) for _ in range(4)]
        stats = tr.step_mcts(exps)
    finally:
        tr.optimizer.step = real_step
    tot = sum(float(p.grad.pow(2).sum()) for p in tr.model.parameters()
              if p.grad is not None)
    return tot ** 0.5, stats


def test_mse_mean_gradient_scales_with_the_miss():
    policy = TransformerPolicy()
    tr = policy._trainer
    tr.config.value_loss_form = "mse_mean"
    tr.config.grad_clip = 1e9
    with torch.no_grad():
        v0 = float(policy._model(policy._encoder.encode(_gs()))
                   .value.squeeze().item())
    g0, s0 = _value_grad_norm(policy, z=v0)          # no miss
    g1, s1 = _value_grad_norm(policy, z=v0 + 0.2)    # miss 0.2
    g2, s2 = _value_grad_norm(policy, z=v0 + 0.4)    # miss 0.4
    assert s0.value_loss < 1e-6 and g0 < 1e-3 * max(g2, 1e-9)
    assert 1.7 < g2 / g1 < 2.3, f"not linear in the miss: {g2/g1:.2f}"
    assert abs(s2.value_loss / max(s1.value_loss, 1e-12) - 4.0) < 0.3


def test_c51_remains_the_default_form():
    policy = TransformerPolicy()
    assert policy._trainer.config.value_loss_form == "c51"
