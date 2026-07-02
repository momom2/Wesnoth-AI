#!/usr/bin/env python3
"""Held-out value probe (--holdout-size) + decisive-rate abort
tripwire (--abort-decisive-rate).

Holdout: MCTSPolicy.finalize_game must divert WHOLE games into the
frozen holdout set until it reaches target size, then send every
later game to training; holdout_metrics() evaluates the current net's
value CE on the frozen set WITHOUT touching weights. Rationale: the
logged train value loss is measured on replay-buffer samples the net
has already fit, so it cannot distinguish learning from memorization
(see BACKLOG §2026-07-02).

Tripwire: sim_self_play.main() must exit with code 4 -- after saving
a checkpoint and flushing the trainer-history CSV -- when the
trailing decisive-game rate drops below --abort-decisive-rate.
Exercised through the real main() on tiny mini-map games (2-turn
games always draw), per the tests-drive-real-code rule.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

import glob
import pytest
import torch

from transformer_policy import TransformerPolicy
from tools.mcts_policy import MCTSPolicy, _PendingMCTSState

# Minimal synthetic GameState builder shared with the snapshot tests.
from test_inference_snapshot import _gs


def _pend(n):
    return [_PendingMCTSState(gs=_gs(), visit_counts=[], side=1)
            for _ in range(n)]


# ---------------------------------------------------------------------
# Holdout diversion + freeze + metrics
# ---------------------------------------------------------------------

def test_holdout_diverts_whole_games_then_freezes():
    policy = TransformerPolicy()
    mp = MCTSPolicy(policy, holdout_size=4)

    # Game 1 (3 states): below target -> whole game diverted.
    mp._pending["g1"] = _pend(3)
    mp.finalize_game("g1", winner=1)
    assert len(mp._holdout) == 3
    assert not mp._queue, "held-out states must not reach training"
    assert mp.holdout_metrics() is None, "still collecting -> no metric"

    # Game 2 (3 states): still below target when the game ends ->
    # diverted as a whole (6 > 4 is fine; granularity is per game).
    mp._pending["g2"] = _pend(3)
    mp.finalize_game("g2", winner=0)
    assert len(mp._holdout) == 6
    assert not mp._queue

    # Game 3: target reached -> trains normally.
    mp._pending["g3"] = _pend(2)
    mp.finalize_game("g3", winner=2)
    assert len(mp._queue) == 2, "post-freeze games must train"
    assert len(mp._holdout) == 6, "holdout is frozen once full"

    # Metrics: finite, sized, and gradient-free.
    before = [p.detach().clone() for p in policy._model.parameters()]
    loss, n = mp.holdout_metrics()
    assert n == 6
    assert math.isfinite(loss) and loss > 0.0
    for a, b in zip(before, policy._model.parameters()):
        assert torch.equal(a, b), "holdout eval must not update weights"


def test_holdout_off_by_default():
    policy = TransformerPolicy()
    mp = MCTSPolicy(policy)  # holdout_size defaults to 0
    mp._pending["g1"] = _pend(2)
    mp.finalize_game("g1", winner=1)
    assert len(mp._queue) == 2, "holdout off -> everything trains"
    assert mp.holdout_metrics() is None


def test_eval_value_loss_matches_step_mcts_scale():
    """Same experiences, same net: eval_value_loss must land in the
    same ballpark as step_mcts's value term (identical normalization:
    mean CE per experience). Guards against a silent /N drift that
    would make holdout and train curves incomparable."""
    policy = TransformerPolicy()
    mp = MCTSPolicy(policy, holdout_size=2)
    mp._pending["g"] = _pend(2)
    mp.finalize_game("g", winner=1)
    loss, n = mp.holdout_metrics()
    assert n == 2
    # C51 CE against a projected point mass is bounded by log(K);
    # a normalization bug (e.g. missing /N) would double it.
    k_atoms = policy._model._value_atoms.numel()
    assert 0.0 < loss < math.log(k_atoms) + 1.0


# ---------------------------------------------------------------------
# Abort tripwire, through the real main()
# ---------------------------------------------------------------------

def test_abort_tripwire_exits_4_and_saves(tmp_path):
    """2-turn mini-map games always draw -> decisive rate 0 -> the
    tripwire must fire once the window fills, after saving the
    checkpoint and the (line-buffered) history CSV."""
    from sim_self_play import main

    ckpt = tmp_path / "trip.pt"
    csv = tmp_path / "history.csv"
    rc = main([
        "sim_self_play.py",
        "--iterations", "6", "--games-per-iter", "1",
        "--max-turns", "2", "--mini-ratio", "1.0",
        "--d-model", "32", "--num-layers", "1",
        "--num-heads", "2", "--d-ff", "64",
        "--abort-decisive-rate", "0.5", "--abort-window", "2",
        "--checkpoint-out", str(ckpt),
        "--trainer-history-csv", str(csv),
        "--save-every", "100",  # only the tripwire save should fire
        "--seed", "7", "--log-level", "WARNING",
    ])
    assert rc == 4, f"expected tripwire exit code 4, got {rc}"
    assert ckpt.exists(), "tripwire must save a final checkpoint"
    rows = csv.read_text(encoding="utf-8").strip().splitlines()
    # Header + one row per completed iteration (window=2 -> 2 iters).
    assert len(rows) >= 3, f"CSV must hold the aborted run's rows: {rows}"
