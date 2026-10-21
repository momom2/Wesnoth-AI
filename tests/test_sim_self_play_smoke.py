#!/usr/bin/env python3
"""Smoke test: the reward flow through the self-play rollout loop.

For one game, every side that decided gets exactly one observe, at the
game's end, carrying its outcome: +1 win, -1 loss, 0 draw.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from sim_test_helpers import fresh_scenario_sim  # noqa: E402


class _CountingPolicy:
    """Always end-turn so the game terminates fast."""
    def __init__(self):
        self.selects: List[Dict] = []
        self.observes: List[tuple] = []

    def select_action(self, gs, *, game_label="default", sim=None):
        self.selects.append({"side": gs.global_info.current_side,
                             "turn": gs.global_info.turn_number})
        return {"type": "end_turn"}

    def observe(self, game_label, side, reward, done):
        self.observes.append((game_label, side, reward, done))

    def drop_pending(self, game_label):
        pass

    def finalize_game(self, game_label, winner, final_gs=None, midgame=False):
        pass


def test_each_side_gets_its_outcome_once_at_the_end():
    from tools.selfplay_game import play_one_game

    sim = fresh_scenario_sim(seed=1, max_turns=4, mini=True)
    policy = _CountingPolicy()
    play_one_game(sim, policy, game_label="g0")

    assert policy.selects, "no decisions made?"
    sides = {s["side"] for s in policy.selects}
    assert sorted(o[1] for o in policy.observes) == sorted(sides), \
        "one observe per deciding side, none per step"
    for _label, side, reward, done in policy.observes:
        assert done is True
        want = 0.0 if sim.winner == 0 else (1.0 if sim.winner == side else -1.0)
        assert reward == want


def test_the_outcome_reward():
    from tools.selfplay_game import outcome_reward
    assert outcome_reward(0, 1) == 0.0
    assert outcome_reward(1, 1) == 1.0
    assert outcome_reward(1, 2) == -1.0
    assert outcome_reward(2, 2) == 1.0
