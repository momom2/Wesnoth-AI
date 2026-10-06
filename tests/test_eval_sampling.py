"""Deployment-sampling ruling (user, 2026-08-26).

Eval plays the SAME decision procedure training uses: TCS is the
production data generator, so elo_eval_game defaults to
TurnCommitPolicy (--no-turn-search restores the pre-2026-08-26
catalog protocol). A verdict measured on a different object than
training optimizes is a measurement artifact candidate -- the leg-5
resume verdict forced the question.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_eval_default_matches_training_sampling():
    from tools.elo_eval_game import _search_policy_cls
    from tools.turn_policy import TurnCommitPolicy
    from tools.mcts_policy import MCTSPolicy
    assert _search_policy_cls(True) is TurnCommitPolicy
    assert _search_policy_cls(False) is MCTSPolicy
