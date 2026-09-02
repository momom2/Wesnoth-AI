"""Arm VG2: principled label mixture + trust region.

Behavioral checks through the production trainer:
  - "consist" states leave the categorical value loss entirely;
  - the Gaussian consistency term's gradient scales LINEARLY with
    the (bias-corrected) gap -- the property that replaces the
    categorical blow-up;
  - the trust-region term is zero at the anchor and positive off
    it; lambda's dual ascent moves the right way.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from test_inference_snapshot import _gs  # noqa: E402
from wesnoth_ai.trainer import MCTSExperience  # noqa: E402
from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402


def _grad_norm(policy, exps):
    tr = policy._trainer
    tr.optimizer.zero_grad()
    real_step = tr.optimizer.step
    tr.optimizer.step = lambda *a, **k: None
    try:
        stats = tr.step_mcts(exps)
    finally:
        tr.optimizer.step = real_step
    tot = sum(float(p.grad.pow(2).sum()) for p in tr.model.parameters()
              if p.grad is not None)
    return tot ** 0.5, stats


def _consist(z, z_pair=None, v_anchor=None):
    return MCTSExperience(game_state=_gs(), visit_counts=[], z=z,
                          value_weight=0.25, policy_weight=0.0,
                          label_kind="consist", z_pair=z_pair,
                          v_anchor=v_anchor)


def test_consist_states_skip_categorical_and_gradient_is_linear_in_gap():
    policy = TransformerPolicy()
    tr = policy._trainer
    tr.config.grad_clip = 1e9
    tr.config.consist_bias = 0.0
    tr.config.consist_sigma2 = 1.0
    tr.config.trust_lambda = 0.0
    with torch.no_grad():
        v0 = float(policy._model(policy._encoder.encode(_gs()))
                   .value.squeeze().item())
    # Two batches whose bias-corrected gaps differ by exactly 2x.
    g = 0.3
    n1, s1 = _grad_norm(policy, [_consist(v0 + g) for _ in range(4)])
    n2, s2 = _grad_norm(policy, [_consist(v0 + 2 * g) for _ in range(4)])
    assert s1.value_loss == 0.0 and s2.value_loss == 0.0, \
        "consist states must not feed the categorical value loss"
    assert s1.consist_loss > 0 and s2.consist_loss > s1.consist_loss
    assert 1.6 < n2 / n1 < 2.4, f"gaussian grad not linear: {n2/n1:.2f}"


def test_trust_region_zero_at_anchor_and_lambda_dual_ascent():
    policy = TransformerPolicy()
    tr = policy._trainer
    tr.config.grad_clip = 1e9
    tr.config.trust_lambda = 1.0
    with torch.no_grad():
        v0 = float(policy._model(policy._encoder.encode(_gs()))
                   .value.squeeze().item())
    _, at_anchor = _grad_norm(
        policy, [_consist(v0, v_anchor=v0) for _ in range(3)])
    _, off_anchor = _grad_norm(
        policy, [_consist(v0, v_anchor=v0 + 0.5) for _ in range(3)])
    assert at_anchor.trust_loss < 1e-6
    assert off_anchor.trust_loss > 0.1

    from tools.mcts import MCTSConfig
    from tools.mcts_policy import MCTSPolicy
    from wesnoth_ai.trainer import TrainStats
    mp = MCTSPolicy(policy, MCTSConfig(n_simulations=1))
    mp._trust_lambda = 1.0
    st = TrainStats()
    mp._vg2_finish(st, dv_mean=0.16)         # 2x delta -> x2
    assert abs(mp._trust_lambda - 2.0) < 1e-9
    mp._vg2_finish(st, dv_mean=0.8)          # 10x delta -> capped x4
    assert abs(mp._trust_lambda - 8.0) < 1e-9
    mp._vg2_finish(st, dv_mean=0.01)         # 1/8 delta -> floored /2
    assert abs(mp._trust_lambda - 4.0) < 1e-9
    assert st.trust_lambda == mp._trust_lambda


def test_paired_estimates_reach_trainer_config():
    policy = TransformerPolicy()
    from tools.mcts import MCTSConfig
    from tools.mcts_policy import MCTSPolicy
    mp = MCTSPolicy(policy, MCTSConfig(n_simulations=1))
    # Search estimate systematically +0.4 above the rollout truth.
    batch = [_consist(0.4 + 0.05 * (i % 3), z_pair=0.05 * (i % 3))
             for i in range(8)]
    # Pre-update predictions on those states: head sits 0.7 above
    # the rollout truth -> the head_minus_truth monitor reads +0.7.
    sig_pre = [[e.game_state for e in batch],
               [e.z_pair + 0.7 for e in batch]]
    mp._vg2_prepare(batch, sig_pre)
    cfg = policy._trainer.config
    assert abs(cfg.consist_bias - 0.4) < 1e-6
    assert cfg.consist_sigma2 > 0
    assert mp._vg2_pair_n == 8
    r = mp._vg2_residuals
    assert abs(r["consist_head_minus_truth"] - 0.7) < 1e-6
    assert abs(r["consist_label_minus_truth"]) < 1e-6
    assert all(e.v_anchor == e.z_pair + 0.7 for e in batch)


def test_signal_telemetry_norms_are_opt_in_but_dv_always_logs():
    from tools.mcts import MCTSConfig
    from tools.mcts_policy import MCTSPolicy
    from wesnoth_ai.trainer import TrainStats
    policy = TransformerPolicy()
    batch = [_consist(0.2, z_pair=0.1) for _ in range(4)]
    sig_pre = [[e.game_state for e in batch], [0.5] * 4]
    off = MCTSPolicy(policy, MCTSConfig(n_simulations=1))
    st_off = TrainStats()
    off._attach_signal_telemetry(st_off, batch, sig_pre)
    assert st_off.sig_policy_norm != st_off.sig_policy_norm  # nan
    assert st_off.sig_dv_consult_n == 4                      # logged
    on = MCTSPolicy(policy, MCTSConfig(n_simulations=1),
                    signal_telemetry=True)
    st_on = TrainStats()
    on._attach_signal_telemetry(st_on, batch, sig_pre)
    assert st_on.sig_policy_norm == st_on.sig_policy_norm    # number
