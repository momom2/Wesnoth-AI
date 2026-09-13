"""A bounced recruit stays bounced for the rest of the turn.

The per-turn recruit-rejection history is observable state (CLAUDE.md's
legality contract): the mask consults it and the encoder shows the
model a per-hex bit for it. It must therefore live in the STATE OF
RECORD. With the Rust-owned state that is the core, and `sim.gs` is a
view the core rebuilds after every command -- a rejection written to
the view is erased by the next command, so the mask offers the same
bounced hex again inside one turn and the encoder's bit flips back.

The Python state of record is tested here too, as the control: both
must behave the same.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_test_helpers import scenario_setup  # noqa: E402
from tools.scenario_pool import build_scenario_gamestate  # noqa: E402
from tools.wesnoth_sim import WesnothSim  # noqa: E402
from wesnoth_ai.dummy_policy import DummyPolicy  # noqa: E402
from wesnoth_ai.game_core import game_core_class  # noqa: E402


def _sim(seed: int, use_core: bool) -> WesnothSim:
    setup = scenario_setup(seed, mini=True)
    return WesnothSim(build_scenario_gamestate(setup), scenario_id=setup.scenario_id,
                      max_turns=6, use_core=use_core)


@pytest.mark.parametrize("use_core", [
    False,
    pytest.param(True, marks=pytest.mark.skipif(
        game_core_class() is None, reason="wesnoth_core.GameCore not available")),
])
def test_a_recruit_bounce_survives_a_mid_turn_command(use_core):
    sim = _sim(3, use_core)
    action = DummyPolicy().select_action(sim.gs, game_label="bounce")
    if action.get("type") == "end_turn":
        pytest.skip("the driver opened with a turn-ending action")
    hexes = sorted(sim.gs.map.hexes, key=lambda h: (h.position.y, h.position.x))
    spot = (hexes[4].position.x, hexes[4].position.y)

    sim.reject_recruit_hex(*spot)
    assert spot in (getattr(sim.gs.global_info, "_recruit_rejected_hexes", None) or set())

    turn_before = sim.turn_number
    sim.step(action)
    assert sim.turn_number == turn_before, "the action ended the turn; pick another"
    assert spot in (getattr(sim.gs.global_info, "_recruit_rejected_hexes", None) or set()), \
        "the bounce was lost when the state of record was re-read"


@pytest.mark.parametrize("use_core", [
    False,
    pytest.param(True, marks=pytest.mark.skipif(
        game_core_class() is None, reason="wesnoth_core.GameCore not available")),
])
def test_the_bounce_clears_at_the_next_init_side(use_core):
    """Per TURN, not forever: the enemy may have moved away."""
    sim = _sim(3, use_core)
    hexes = sorted(sim.gs.map.hexes, key=lambda h: (h.position.y, h.position.x))
    spot = (hexes[4].position.x, hexes[4].position.y)
    sim.reject_recruit_hex(*spot)
    sim.step({"type": "end_turn"})          # end_turn runs the next side's init_side
    assert spot not in (getattr(sim.gs.global_info, "_recruit_rejected_hexes", None) or set())
