#!/usr/bin/env python3
"""Smoke test: end-to-end reward flow through sim_self_play's
rollout loop.

Verifies that for ONE game (DummyPolicy on both sides):
  - per-step rewards land on the right side via policy.observe;
  - terminal rewards are attached to each side's last transition;
  - per-unit-type bonus weights propagate through the rollout into
    side1_reward;
  - per-game reward components (turn-conditional once-per-game)
    reset between games via the harness's reset_game_state hook.

This is a behavioral test, not a unit test -- the reward function's
arithmetic is locked in by test_rewards.py. Here we check that the
harness wires those components together correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

import glob
import pytest

from rewards import (
    StepDelta, WeightedReward,
    UnitTypeBonus, TurnConditionalBonus, register_predicate,
)


@pytest.fixture
def small_replay() -> Path:
    """Pick the lex-smallest replay in the dataset (deterministic)
    so this test is reproducible and stays fast."""
    cands = sorted(glob.glob("replays_dataset/*.json.gz"))
    if not cands:
        pytest.skip("no replays_dataset/ to bootstrap from")
    return Path(cands[0])


def _wrap_dummy() -> "object":
    """Wrap DummyPolicy with the trainable-protocol no-op stubs that
    sim_self_play's harness calls (observe / drop_pending /
    train_step / save_checkpoint). Returns the wrapped object."""
    from dummy_policy import DummyPolicy
    p = DummyPolicy()
    p.observe = lambda *a, **kw: None
    p.drop_pending = lambda *a, **kw: None
    p.train_step = lambda *a, **kw: _NoStats()
    p.save_checkpoint = lambda *a, **kw: None
    return p


class _NoStats:
    """Minimal stand-in for trainer.TrainStats that run_iteration's
    log line dereferences. All zeros."""
    n_trajectories = 0
    n_transitions = 0
    total_loss = 0.0
    policy_loss = 0.0
    value_loss = 0.0
    entropy = 0.0
    mean_return = 0.0
    grad_norm = 0.0


def test_one_game_emits_observe_per_step(small_replay):
    """play_one_game calls policy.observe at least once per accepted
    action + once per terminal. Verify shape on a counting stub."""
    from sim_self_play import play_one_game, _recruit_cost_lookup
    from wesnoth_sim import WesnothSim

    class _CountingPolicy:
        """Always end-turn so the game terminates fast."""
        def __init__(self):
            self.selects: List[Dict] = []
            self.observes: List[tuple] = []
        def select_action(self, gs, *, game_label="default"):
            self.selects.append({"side": gs.global_info.current_side,
                                 "turn": gs.global_info.turn_number})
            return {"type": "end_turn"}
        def observe(self, game_label, side, reward, done):
            self.observes.append((game_label, side, reward, done))
        def drop_pending(self, game_label):
            pass

    sim = WesnothSim.from_replay(small_replay, max_turns=4)
    policy = _CountingPolicy()
    reward_fn = WeightedReward()
    cost_lookup = _recruit_cost_lookup()

    play_one_game(sim, policy, reward_fn,
                  game_label="g0", cost_lookup=cost_lookup)

    assert len(policy.selects) >= 1, "no decisions made?"
    # Every side that made a decision should get exactly one
    # terminal observe(done=True). play_one_game iterates over
    # `last_acting_side`, which records every side that called
    # select_action -- so this is a stricter invariant than the old
    # "<= 2 terminals" assertion (which assumed 2-side replays).
    sides_with_decision = {s["side"] for s in policy.selects}
    sides_with_terminal = {o[1] for o in policy.observes if o[3] is True}
    assert sides_with_terminal == sides_with_decision, (
        f"terminal sides {sides_with_terminal} != "
        f"deciding sides {sides_with_decision}")
    for _, _, r, _ in policy.observes:
        assert isinstance(r, float)


def test_unit_type_bonus_fires_through_pipeline(small_replay):
    """A WeightedReward with a UnitTypeBonus on a recruitable unit
    type credits the bonus when DummyPolicy recruits one. Compare
    side1_reward across two runs (bonus on/off): the delta must be
    exactly weight × n_recruits_of_target_type."""
    from sim_self_play import play_one_game, _recruit_cost_lookup
    from wesnoth_sim import WesnothSim

    cost_lookup = _recruit_cost_lookup()

    def _run_with_reward_fn(rf):
        sim = WesnothSim.from_replay(small_replay, max_turns=3)
        outcome = play_one_game(sim, _wrap_dummy(), rf,
                                game_label="bonus_test",
                                cost_lookup=cost_lookup)
        return outcome, sim.command_history

    rf_baseline = WeightedReward()
    out_a, hist_a = _run_with_reward_fn(rf_baseline)

    recruited_types_s1 = [
        rc.cmd[1] for rc in hist_a
        if rc.kind == "recruit" and rc.side == 1
    ]
    if not recruited_types_s1:
        pytest.skip("DummyPolicy didn't recruit anything on side 1; "
                    "test premise unmet")
    target_type = recruited_types_s1[0]
    n_recruits = sum(1 for t in recruited_types_s1 if t == target_type)

    rf_bonus = WeightedReward(
        unit_type_bonuses=[UnitTypeBonus(target_type, weight=0.5)],
    )
    out_b, _ = _run_with_reward_fn(rf_bonus)

    expected_delta = 0.5 * n_recruits
    actual_delta = out_b.side1_reward - out_a.side1_reward
    assert actual_delta == pytest.approx(expected_delta, abs=1e-4), (
        f"unit-type bonus didn't propagate: expected ~+{expected_delta}, "
        f"got {actual_delta} (side1_a={out_a.side1_reward}, "
        f"side1_b={out_b.side1_reward}, n_recruits={n_recruits})")


def test_turn_conditional_once_resets_between_games(small_replay):
    """A `once=True` turn-conditional bonus must fire ONCE PER GAME.
    The run_iteration harness calls reward_fn.reset_game_state(label)
    before each game; without that, the second game's fired-set
    would still hold game-1's flag and the bonus wouldn't re-fire,
    so side-1 reward in game 2 would be zero (other terms are
    zero-weighted)."""
    from sim_self_play import _recruit_cost_lookup, run_iteration
    import random

    rf = WeightedReward(
        # Zero everything except the bonus so any nonzero reward must
        # come from the bonus path.
        gold_killed_delta=0.0, village_delta=0.0, damage_dealt=0.0,
        unit_recruited_cost=0.0, leader_move_penalty=0.0,
        invalid_action_penalty=0.0, min_enemy_distance_penalty=0.0,
        terminal_win=0.0, terminal_loss=0.0, terminal_draw=0.0,
        terminal_timeout=0.0,
        turn_conditional_bonuses=[TurnConditionalBonus(
            name="x", turn_range=(1, 100),
            predicate=lambda st, side: True,
            weight=0.7, once=True,
        )],
    )
    cost_lookup = _recruit_cost_lookup()
    rng = random.Random(0)

    outcomes = run_iteration(
        _wrap_dummy(), [small_replay], rf, cost_lookup,
        iter_idx=0, games_per_iter=2, max_turns=3, rng=rng,
    )
    assert len(outcomes) == 2
    # Both games fire the once-bonus -> both have nonzero rewards.
    assert outcomes[0].side1_reward == pytest.approx(0.7, abs=1e-4)
    assert outcomes[1].side1_reward == pytest.approx(0.7, abs=1e-4)
    # If the harness forgot to reset, game 2's reward would be 0.0.
