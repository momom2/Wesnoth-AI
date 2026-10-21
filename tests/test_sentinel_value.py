"""A refused action is worth what the parent state is worth.

The search parks two outcomes under sentinel children: a step that
raised (`step_error`) and a step the sim refused or that changed
nothing (`noop_resample`, e.g. a recruit onto a fog-occupied hex).
Both are marked terminal so the descent stops, and the value that
backs up is the PARENT's own estimate: the state did not change, so
the action is neither a draw nor a win. Never 0 at the node level
either (0 is above a losing side's own estimate, which would have
steered the search into refused actions when behind).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_test_helpers import fresh_scenario_sim  # noqa: E402
from tools.mcts import MCTSNode, _node_terminal_value, _terminal_value  # noqa: E402


def _drawn_sim(ended_by: str):
    """A sim whose game ended without a winner, tagged `ended_by`."""
    sim = fresh_scenario_sim(0, max_turns=6)
    sim.done, sim.winner, sim.ended_by = True, 0, ended_by
    return sim


def test_a_sentinel_node_backs_up_its_parent_value():
    """What the search actually reads: the sentinel child carries the
    parent's own estimate; a real ending its outcome."""
    node = MCTSNode(_drawn_sim("step_error"))
    node.value = 0.37
    node.is_sentinel = True
    assert _node_terminal_value(node) == 0.37
    real = MCTSNode(_drawn_sim("max_turns"))
    real.value = 0.37
    assert _node_terminal_value(real) == _terminal_value(real.sim, real.side) == 0.0


@pytest.mark.parametrize("ending", ["max_turns", "step_error", "noop_resample"])
def test_every_draw_is_zero(ending):
    sim = _drawn_sim(ending)
    assert _terminal_value(sim, 1) == 0.0
    assert _terminal_value(sim, 2) == 0.0


def test_a_win_is_unaffected_by_the_ending_tag():
    sim = _drawn_sim("step_error")
    sim.winner = 1
    assert _terminal_value(sim, 1) == 1.0
    assert _terminal_value(sim, 2) == -1.0
