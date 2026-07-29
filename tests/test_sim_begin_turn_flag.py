"""`WesnothSim(begin_turn=False)` must not fire init_side.

Why this exists: the ctor unconditionally fired `_begin_side_turn`,
which pays income, applies healing/poison, refreshes movement, and
advances the turn counter. Any tool that rebuilds a state captured
MID-turn through the ctor therefore got a free turn's worth of gold
and HP -- and the corruption is invisible, because the resulting
state is still structurally valid. It bit the 2026-07-29 hoarding
probe, which had to work around it with a sacrificial-copy swap.

These tests pin BOTH directions: the default still begins a turn
(so no game-playing caller changed behaviour), and the opt-out
leaves the passed gamestate alone.

They exercise the production ctor directly -- no mirror of its
logic -- per CLAUDE.md ("tests must call the production path").
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_test_helpers import scenario_setup            # noqa: E402
from tools.scenario_pool import build_scenario_gamestate  # noqa: E402
from tools.wesnoth_sim import WesnothSim               # noqa: E402


def _gold(gs):
    return [s.current_gold for s in gs.sides]


def _hp_mp(gs):
    """(hp, moves) per unit, keyed by (id, side), order-independent.

    Units live on `map.units`, NOT on `hex.unit` -- reading them off
    hexes yields an empty dict and turns the comparison below into a
    vacuous {} == {}. The callers assert non-emptiness for that
    reason."""
    return {(u.id, u.side): (u.current_hp, u.current_moves)
            for u in gs.map.units}


def _build():
    setup = scenario_setup(seed=7)
    return setup, build_scenario_gamestate(setup)


def test_default_ctor_still_begins_the_turn():
    """Guard the DEFAULT path: everything that plays games relies on
    the ctor firing init_side, so this must not have changed."""
    setup, gs = _build()
    before_turn = gs.global_info.turn_number
    sim = WesnothSim(copy.deepcopy(gs), scenario_id=setup.scenario_id,
                     max_turns=6)
    # init_side(1) bumps turn 0 -> 1 and records itself.
    assert sim.gs.global_info.turn_number == before_turn + 1
    assert any(c.kind == "init_side" for c in sim.command_history)


def test_begin_turn_false_leaves_the_state_untouched():
    """The opt-out must not pay income, heal, refresh movement, or
    advance the turn -- that is the whole point of it."""
    setup, gs = _build()
    snapshot = copy.deepcopy(gs)
    sim = WesnothSim(gs, scenario_id=setup.scenario_id, max_turns=6,
                     begin_turn=False)

    assert sim.gs.global_info.turn_number == \
        snapshot.global_info.turn_number
    assert sim.gs.global_info.current_side == \
        snapshot.global_info.current_side
    before = _hp_mp(snapshot)
    assert before, "no units found -- the HP/MP check would be vacuous"
    assert _gold(sim.gs) == _gold(snapshot)
    assert _hp_mp(sim.gs) == before


def test_begin_turn_false_records_no_init_side_command():
    """A stray init_side in command_history would desync a replay
    export built from a mid-turn reconstruction."""
    setup, gs = _build()
    sim = WesnothSim(gs, scenario_id=setup.scenario_id, max_turns=6,
                     begin_turn=False)
    assert not any(c.kind == "init_side" for c in sim.command_history)


def test_begin_turn_flag_changes_gold_relative_to_default():
    """The two paths must actually DIFFER on the same input, or the
    flag is a no-op and the other assertions prove nothing. Income is
    the cheapest observable difference."""
    setup, gs = _build()
    began = WesnothSim(copy.deepcopy(gs), scenario_id=setup.scenario_id,
                       max_turns=6)
    skipped = WesnothSim(copy.deepcopy(gs), scenario_id=setup.scenario_id,
                         max_turns=6, begin_turn=False)
    assert (began.gs.global_info.turn_number
            != skipped.gs.global_info.turn_number)
