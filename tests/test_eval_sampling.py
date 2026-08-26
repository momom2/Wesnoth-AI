"""Deployment-sampling and anchor-default rulings (user, 2026-08-26).

1. Eval plays the SAME decision procedure training uses: TCS is the
   production data generator, so elo_eval_game defaults to
   TurnCommitPolicy (--no-turn-search restores the pre-2026-08-26
   catalog protocol). A verdict measured on a different object than
   training optimizes is a measurement artifact candidate -- the
   leg-5 resume verdict forced the question.
2. Anchors default OFF everywhere: trainer CLI defaults None, and
   the launcher no longer auto-builds the value anchor.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

REPO = Path(__file__).parent.parent


def test_eval_default_matches_training_sampling():
    from tools.elo_eval_game import _search_policy_cls
    from tools.turn_policy import TurnCommitPolicy
    from tools.mcts_policy import MCTSPolicy
    assert _search_policy_cls(True) is TurnCommitPolicy
    assert _search_policy_cls(False) is MCTSPolicy


def test_batch_runner_forwards_turn_search_optout():
    src = (REPO / "tools/run_elo_batch.py").read_text(encoding="utf-8")
    assert "--no-turn-search" in src
    # Per-checkpoint deployment (user follow-up 2026-08-26): a mixed
    # match gives each side the sampling it was trained for.
    assert "--no-turn-search-a" in src and "--no-turn-search-b" in src
    game = (REPO / "tools/elo_eval_game.py").read_text(encoding="utf-8")
    assert "no_turn_search_a" in game and "no_turn_search_b" in game


def test_trainer_anchor_defaults_are_off():
    # Assert on the source (importing sim_self_play pulls torch):
    # both anchor args must default to None.
    src = (REPO / "tools/sim_self_play.py").read_text(encoding="utf-8")
    for flag in ("--human-anchor-file", "--human-anchor-policy-file"):
        i = src.index(flag)
        assert "default=None" in src[i:i + 200], f"{flag} must default OFF"


def test_launcher_does_not_default_value_anchor_on():
    onstart = (REPO / "scripts/vast_onstart.sh").read_text(encoding="utf-8")
    assert 'HUMAN_ANCHOR_FILE="${HUMAN_ANCHOR_FILE-}"' in onstart, \
        "launcher must not auto-enable the value anchor (ruling 2026-08-26)"
