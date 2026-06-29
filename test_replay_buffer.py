"""Replay buffer + multi-epoch training (MCTSPolicy.train_step).

Uses a fake trainer that records each step_mcts call's batch size, so
we pin the SCHEDULE (how many gradient steps, over how many samples,
warm-up, capacity) without heavy model forwards. The numerical
equivalence of the gradient step itself is the trainer's concern,
unchanged here.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

from trainer import MCTSExperience, TrainStats  # noqa: E402
from tools.mcts_policy import MCTSPolicy, ReplayConfig  # noqa: E402


class _FakeTrainer:
    def __init__(self):
        self.calls = []   # batch sizes seen by step_mcts

    def step_mcts(self, batch):
        self.calls.append(len(batch))
        # Distinct per-call values so _combine_stats is checkable.
        k = len(self.calls)
        return TrainStats(policy_loss=float(k), value_loss=float(k) * 2,
                          total_loss=float(k) * 3, grad_norm=float(k),
                          n_transitions=len(batch), n_trajectories=1)


def _policy(replay_config):
    # `_snaps` counts _snapshot_inference_weights() calls. The MCTS path
    # MUST refresh the inference net after each gradient iteration, or
    # self-play/search runs on a frozen network (the 2026-06-29 bug).
    base = SimpleNamespace(_trainer=_FakeTrainer(),
                           _inference_model=None, _inference_encoder=None,
                           _snaps=[0])
    base._snapshot_inference_weights = lambda: base._snaps.__setitem__(
        0, base._snaps[0] + 1)
    return MCTSPolicy(base, replay_config=replay_config), base._trainer


def _exps(n):
    return [MCTSExperience(game_state=None, visit_counts=[], z=0.0)
            for _ in range(n)]


def test_disabled_is_legacy_one_pass():
    pol, tr = _policy(ReplayConfig(enabled=False))
    pol._queue = _exps(5)
    pol.train_step()
    assert tr.calls == [5], "disabled must do exactly one step over the fresh batch"
    assert pol._base._snaps[0] == 1, "gradient step must refresh the inference net"


def test_disabled_empty_queue_noop():
    pol, tr = _policy(ReplayConfig(enabled=False))
    pol.train_step()
    assert tr.calls == [], "no experiences -> no gradient step"
    assert pol._base._snaps[0] == 0, "no gradient step -> no inference refresh"


def test_warmup_uses_one_pass_until_min_size():
    pol, tr = _policy(ReplayConfig(enabled=True, min_size=10,
                                   updates_per_iter=4, minibatch=8))
    pol._queue = _exps(6)            # buffer (6) < min_size (10)
    pol.train_step()
    assert tr.calls == [6], "below min_size must fall back to one-pass on fresh batch"
    assert len(pol._replay) == 6
    assert pol._base._snaps[0] == 1, "warm-up gradient step must refresh inference"


def test_multiepoch_after_min_size():
    pol, tr = _policy(ReplayConfig(enabled=True, min_size=4,
                                   updates_per_iter=3, minibatch=8,
                                   capacity=100))
    pol._queue = _exps(10)          # buffer (10) >= min_size (4)
    pol.train_step()
    assert tr.calls == [8, 8, 8], (
        f"expected 3 minibatch steps of size 8, got {tr.calls}")
    # ONE refresh per iteration (after the final update), not per minibatch:
    # the inference net only needs the latest weights once self-play resumes.
    assert pol._base._snaps[0] == 1, "exactly one inference refresh per iteration"


def test_minibatch_capped_to_buffer_size():
    pol, tr = _policy(ReplayConfig(enabled=True, min_size=4,
                                   updates_per_iter=2, minibatch=64,
                                   capacity=100))
    pol._queue = _exps(5)           # buffer 5 >= min 4, but < minibatch 64
    pol.train_step()
    assert tr.calls == [5, 5], "minibatch must clamp to buffer size"


def test_capacity_bounds_buffer():
    pol, tr = _policy(ReplayConfig(enabled=True, min_size=1,
                                   updates_per_iter=1, minibatch=4,
                                   capacity=10))
    for _ in range(5):
        pol._queue = _exps(4)       # 20 experiences total fed
        pol.train_step()
    assert len(pol._replay) == 10, "buffer must be bounded to capacity"


def test_combine_stats_means_and_sums():
    stats = [
        TrainStats(policy_loss=2.0, value_loss=4.0, total_loss=6.0,
                   grad_norm=1.0, n_transitions=8),
        TrainStats(policy_loss=4.0, value_loss=8.0, total_loss=12.0,
                   grad_norm=9.0, n_transitions=8),
    ]
    out = MCTSPolicy._combine_stats(stats, buffer_size=50)
    assert out.policy_loss == 3.0          # mean
    assert out.value_loss == 6.0           # mean
    assert out.grad_norm == 9.0            # last step
    assert out.n_transitions == 16         # sum
    assert out.n_trajectories == 50        # buffer size


def test_reproducible_sampling_given_seed():
    """Two policies with the same seed draw the same minibatches."""
    cfg = dict(enabled=True, min_size=4, updates_per_iter=3, minibatch=4,
               capacity=100)
    pol1, tr1 = _policy(ReplayConfig(**cfg))
    pol2, tr2 = _policy(ReplayConfig(**cfg))
    # Tag experiences so we can compare which got sampled.
    e1 = _exps(20)
    for i, e in enumerate(e1):
        e.z = float(i)
    pol1._queue = list(e1); pol2._queue = list(e1)
    import random as _r
    pol1._replay_rng = _r.Random(123)
    pol2._replay_rng = _r.Random(123)
    pol1.train_step(); pol2.train_step()
    assert tr1.calls == tr2.calls
