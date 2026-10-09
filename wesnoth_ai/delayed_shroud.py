"""Delayed shroud updates (docs/wesnoth_rules.md "Delayed shroud updates"):
the state a view of the Rust core carries.

A side whose player turned "delay shroud updates" on (the
`[auto_shroud] active=no` command) clears no fog when its units move or
are recruited during its turn. Each such action waits on the engine's
undo stack with the hexes the unit occupied and the unit's vision then,
and the fog is cleared from all of them at the next commit:
`[update_shroud]`, `[auto_shroud] active=yes` while delayed, or an
action that cannot be undone (an attack, a recruit that drew random
numbers, a move that was ambushed or blocked, the end of the turn). An
advancement on the delaying side's own turn clears nothing. In a game
with the Plan Unit Advance modification its handlers make two more
actions final: the first move of each side turn (its `moveto` handler
calls `wesnoth.allow_undo(false)`, data/modifications/pick_advance/main.lua)
and each Plan Advancement menu event.

The Rust core keeps and applies the rule
(rust/wesnoth_core/src/core_shroud.rs). A view carries it on
`global_info`: `_shroud_delayed`, the frozenset of sides that delay, and
`_pending_vision`, a tuple of (side, route, unit id, vision points,
slowed) with the route a tuple of (x, y); a replay record starts a game
with them (`replay_dataset._build_initial_gamestate`).
"""
from __future__ import annotations

from wesnoth_ai.classes import GameState

SHROUD_DELAYED = "_shroud_delayed"
PENDING_VISION = "_pending_vision"
PLAN_UNIT_ADVANCE = "_plan_unit_advance"       # the modification is on


def delaying_sides(state: GameState) -> frozenset:
    return frozenset(getattr(state.global_info, SHROUD_DELAYED, None) or ())


def pending_vision(state: GameState) -> tuple:
    return tuple(getattr(state.global_info, PENDING_VISION, None) or ())


def vision_delayed(state: GameState, side: int) -> bool:
    """Whether `side`'s fog clearing waits for a commit: fog is on, it is
    the side's turn and the side delays its shroud updates
    (`current_uses_fog_`, src/actions/move.cpp:371 at 1.18.4; the
    recruit, create.cpp:697; an advancement, vision.cpp:467-469)."""
    return (bool(getattr(state.global_info, "_fog", True))
            and side == state.global_info.current_side
            and side in delaying_sides(state))
