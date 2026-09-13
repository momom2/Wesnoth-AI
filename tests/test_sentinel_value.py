"""A refused action is worth 0, not the material draw score.

The search parks two outcomes under sentinel children: a step that
raised (`step_error`) and a step the sim refused or that changed
nothing (`noop_resample`, e.g. a recruit onto a fog-occupied hex).
Both are marked terminal so the descent stops and, in the words of the
code that creates them, "a neutral value backs up".

With `draw_tiebreak` configured -- which `tools/selfplay_worker.py`
does -- `_terminal_value` used to price every winner==0 state by the
material differential, so a side that was ahead saw a REFUSED action as
a favourable draw (+cap * material) and the search steered into
rejected actions exactly when it was winning. A real draw at the turn
cap must still get the tiebreak.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_test_helpers import fresh_scenario_sim  # noqa: E402
from tools.draw_tiebreak import DrawTiebreakConfig  # noqa: E402
from tools.mcts import _terminal_value  # noqa: E402


def _drawn_sim(ended_by: str):
    """A sim whose game ended without a winner, tagged `ended_by`, with
    a material edge for side 1 so the tiebreak is non-zero."""
    from tools.replay_dataset import _replace_unit
    sim = fresh_scenario_sim(0, max_turns=6, use_core=False)
    # Hurt side 2 so the material differential is non-zero (a scenario
    # starts with leaders only, and two healthy leaders are symmetric).
    hurt = next(u for u in sorted(sim.gs.map.units, key=lambda u: u.id) if u.side == 2)
    _replace_unit(sim.gs, hurt, current_hp=max(1, hurt.max_hp // 4))
    sim.done, sim.winner, sim.ended_by = True, 0, ended_by
    return sim


def test_a_real_draw_still_gets_the_material_tiebreak():
    sim = _drawn_sim("max_turns")
    cfg = DrawTiebreakConfig(cap=0.25)
    v1 = _terminal_value(sim, 1, cfg)
    v2 = _terminal_value(sim, 2, cfg)
    assert v1 != 0.0, "a genuine cap draw must carry the tiebreak"
    assert v1 == pytest.approx(-v2), "the tiebreak must be antisymmetric in the side"
    assert -0.25 <= v1 <= 0.25


@pytest.mark.parametrize("ending", ["step_error", "noop_resample"])
def test_a_refused_action_is_neutral_even_with_a_tiebreak(ending):
    sim = _drawn_sim(ending)
    cfg = DrawTiebreakConfig(cap=0.25)
    assert _terminal_value(sim, 1, cfg) == 0.0
    assert _terminal_value(sim, 2, cfg) == 0.0


def test_without_a_tiebreak_every_draw_is_zero():
    for ending in ("max_turns", "step_error", "noop_resample"):
        assert _terminal_value(_drawn_sim(ending), 1, None) == 0.0


def test_a_win_is_unaffected_by_the_ending_tag():
    sim = _drawn_sim("step_error")
    sim.winner = 1
    assert _terminal_value(sim, 1, DrawTiebreakConfig(cap=0.25)) == 1.0
    assert _terminal_value(sim, 2, DrawTiebreakConfig(cap=0.25)) == -1.0
