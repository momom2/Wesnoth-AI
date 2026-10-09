"""A command the core cannot apply is skipped with a warning, never
silently (the 2026-09-29 audit, C4 and C5): a move or an attack whose hex
holds no unit, and an advancement choice outside the unit's options. The
Rust core reports it through its warning channel
(`wesnoth_core.drain_warnings`)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from wesnoth_ai import game_core as gc  # noqa: E402


def _state():
    from helpers.parity_games import record, state_of
    return state_of(record([("Lieutenant", 1, 1, 3, True), ("Spearman", 2, 6, 3, False),
                            ("Lieutenant", 2, 18, 3, True)], fog=False))


EMPTY_MOVE = ["move", [9, 9], [2, 3], 1]
EMPTY_ATTACK = ["attack", 9, 2, 9, 3, 0, 0, "00c0ffee"]


@pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")
def test_the_core_warns_when_a_command_finds_no_unit():
    import wesnoth_core
    cs = gc.CoreState.from_state(_state())
    cs.apply_command(["init_side", 1])
    wesnoth_core.drain_warnings()
    cs.core.apply_move(list(EMPTY_MOVE[1]), list(EMPTY_MOVE[2]), 1)
    cs.core.apply_attack(9, 2, 9, 3, 0, 0, 1, True, [])
    texts = wesnoth_core.drain_warnings()
    assert any("move from (9, 2) finds no unit" in t for t in texts)
    assert any("attack from (9, 2) on (9, 3) misses a unit" in t for t in texts)
