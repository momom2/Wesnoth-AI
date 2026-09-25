"""Behavioral tests for tools/value_grounding (arm VG).

Exercises the production build path with a stubbed rollout sim:
perspective flips, weights, budgets, and censoring must match the
contract in the module docstring.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from helpers.tiny_state import _gs  # noqa: E402
from tools.value_grounding import (  # noqa: E402
    GroundCapture, GroundingConfig, build_grounding_experiences,
)


class _StubSim:
    """Terminal sim: fork returns self; already done with a winner."""

    def __init__(self, gs, winner):
        self.gs = gs
        self.done = True
        self.winner = winner

    def fork(self):
        return self


def _capture(side=1, winner=1, projected=0.5, current_side=None):
    gs = _gs()
    if current_side is not None:
        gs.global_info.current_side = current_side
    return GroundCapture(sim=_StubSim(gs, winner), side=side,
                         decision_step=7, projected=projected)


def test_grounding_experiences_weights_perspective_and_budgets():
    cfg = GroundingConfig(enabled=True, max_consist_per_game=2,
                          max_rollout_per_game=1,
                          consist_value_weight=0.25)
    rng = np.random.default_rng(0)
    # Captured state where the OPPONENT is to move: side-perspective
    # values must flip sign when stored as side-to-move z.
    caps = [_capture(side=1, winner=1, projected=0.5, current_side=2)
            for _ in range(4)]
    exps, stats = build_grounding_experiences(None, caps, cfg, rng)
    # 2 consistency + 1 rollout experience, all value-only.
    assert len(exps) == 3
    assert all(e.policy_weight == 0.0 for e in exps)
    assert all(e.visit_counts == [] for e in exps)
    consist = [e for e in exps if e.value_weight == 0.25]
    ground = [e for e in exps if e.value_weight == 1.0]
    assert len(consist) == 2 and len(ground) == 1
    # side=1 wins (+1 side-persp), mover is side 2 -> z stored -1.
    assert ground[0].z == -1.0
    assert all(abs(e.z + 0.5) < 1e-9 for e in consist)
    assert stats["ground_win"] == 1 and stats["ground_censored"] == 0
    # Two identical rollouts (deterministic stub) -> the paired
    # consist state carries z_pair with measured variance 0.
    paired = [e for e in consist if e.z_pair is not None]
    assert len(paired) == 1 and paired[0].z_pair == -1.0
    assert paired[0].z_pair_var == 0.0
    assert stats["ground_rollouts"] == 2


def test_grounding_censored_rollout_produces_no_experience():
    class _NeverEnds(_StubSim):
        def __init__(self, gs):
            super().__init__(gs, 0)
            self.done = False

        def step(self, action):
            return None

    cfg = GroundingConfig(enabled=True, max_consist_per_game=0,
                          max_rollout_per_game=1,
                          rollout_max_halfturns=2,
                          rollout_max_actions=1)
    gs = _gs()
    caps = [GroundCapture(sim=_NeverEnds(gs), side=1,
                          decision_step=0, projected=0.0)]

    class _P:
        pass

    import tools.value_grounding as vg
    orig = vg.rollout_outcome
    try:
        exps, stats = build_grounding_experiences(
            _P(), caps, cfg, np.random.default_rng(0))
    finally:
        vg.rollout_outcome = orig
    assert exps == []
    assert stats["ground_censored"] == 2      # both rollouts censored
