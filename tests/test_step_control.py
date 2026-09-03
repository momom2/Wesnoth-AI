"""Backtracking step control (tools/step_control.py).

Two games' worth of positions with CONTRADICTORY value labels: the
stepped-on game says +1, the held-out game says -1. Any move along
the training gradient raises held-out loss, so the rule must shrink
to nothing and restore the weights. With AGREEING labels, the full
proposal lowers held-out loss and alpha stays 1.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from test_inference_snapshot import _gs  # noqa: E402
from tools.step_control import backtracking_step, split_holdout  # noqa: E402
from wesnoth_ai.trainer import MCTSExperience  # noqa: E402
from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402


def _exps(game_id: str, z: float, n: int = 6):
    return [MCTSExperience(game_state=_gs(), visit_counts=[], z=z,
                           policy_weight=0.0, game_id=game_id)
            for _ in range(n)]


def _policy():
    policy = TransformerPolicy()
    tr = policy._trainer
    tr.config.value_loss_form = "mse_mean"
    tr.config.grad_clip = 1.0
    for g in tr.optimizer.param_groups:
        g["lr"] = 1e-3
    return policy


def _weights(policy):
    return {k: v.detach().clone() for k, v in policy._model.state_dict().items()}


def test_contradictory_holdout_skips_and_restores():
    policy = _policy()
    train, held = _exps("g_train", +1.0), _exps("g_held", -1.0)
    w0 = _weights(policy)
    res = backtracking_step(
        policy, lambda: policy._trainer.step_mcts(train), train, held,
        max_trials=4)
    assert res.skipped and res.alpha == 0.0 and res.trials == 4
    assert res.held_after["total"] >= res.held_before["total"]
    for k, v in _weights(policy).items():
        if torch.is_floating_point(v):
            assert torch.equal(v, w0[k]), f"{k} not restored"
    inf = policy._inference_model.state_dict()
    for k, v in w0.items():
        if torch.is_floating_point(v):
            assert torch.equal(inf[k].cpu(), v.cpu()), f"inference {k} not restored"


def test_agreeing_holdout_accepts_full_step():
    policy = _policy()
    train, held = _exps("g_train", +1.0), _exps("g_held", +1.0)
    w0 = _weights(policy)
    res = backtracking_step(
        policy, lambda: policy._trainer.step_mcts(train), train, held)
    assert not res.skipped and res.alpha == 1.0 and res.trials == 1
    assert res.held_after["total"] < res.held_before["total"]
    moved = any(not torch.equal(v, w0[k])
                for k, v in _weights(policy).items()
                if torch.is_floating_point(v))
    assert moved


def test_level_cap_shrinks_a_step_that_moves_the_value_level():
    policy = _policy()
    train, held = _exps("g_train", +1.0), _exps("g_held", +1.0)
    step = lambda: policy._trainer.step_mcts(train)  # noqa: E731
    free = backtracking_step(policy, step, train, held, held,
                             max_level_shift=None)
    assert free.alpha == 1.0 and abs(free.shift["dv_mean"]) > 1e-6
    policy2 = _policy()
    train2, held2 = _exps("g_train", +1.0), _exps("g_held", +1.0)
    step2 = lambda: policy2._trainer.step_mcts(train2)  # noqa: E731
    capped = backtracking_step(policy2, step2, train2, held2, held2,
                               max_level_shift=1e-7)
    assert capped.alpha < 1.0 and capped.trials > 1
    assert capped.skipped or abs(capped.shift["dv_mean"]) <= 1e-7


def test_not_significantly_worse_is_a_paired_test_over_games():
    from tools.step_control import not_significantly_worse as nsw
    b = {g: {"total": 5.0} for g in "abc"}
    # Mixed: one game worse, two better; mean +0.0 -> accept.
    a = {"a": {"total": 5.3}, "b": {"total": 4.85}, "c": {"total": 4.85}}
    ok, mean, se = nsw(b, a)
    assert ok and abs(mean) < 1e-9
    # Every game worse by the same amount: se = 0, mean > 0 -> reject.
    a = {g: {"total": 5.05} for g in "abc"}
    ok, mean, se = nsw(b, a)
    assert not ok and mean > 0 and se == 0.0
    # Slightly worse on average but within 2 se -> accept.
    a = {"a": {"total": 5.2}, "b": {"total": 5.0}, "c": {"total": 4.9}}
    ok, mean, se = nsw(b, a)
    assert ok and mean > 0 and mean <= 2 * se
    # One game only: strict decrease required.
    assert nsw({"a": {"total": 5.0}}, {"a": {"total": 4.99}})[0]
    assert not nsw({"a": {"total": 5.0}}, {"a": {"total": 5.0}})[0]


def test_split_holdout_is_by_game():
    exps = _exps("a", 1.0) + _exps("b", 1.0) + _exps("c", 1.0) + _exps("d", 1.0)
    train, held = split_holdout(exps, 0.25, random.Random(0))
    assert len(held) == 6 and len(train) == 18
    assert len({e.game_id for e in held}) == 1
    assert not ({e.game_id for e in held} & {e.game_id for e in train})
