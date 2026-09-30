"""Delayed shroud updates (docs/wesnoth_rules.md "Delayed shroud updates").

A side whose player turned "delay shroud updates" on (the
`[auto_shroud] active=no` command) clears no fog when its units move or
are recruited during its turn. Each such action waits on the engine's
undo stack with the hexes the unit occupied and the unit's vision then,
and the fog is cleared from all of them at the next commit:

  clear_undo_stack(state)   an action that cannot be undone: an attack,
                            a recruit that drew random numbers, a move
                            that was ambushed or blocked, the end of
                            the turn (`undo_list::clear`);
  commit_vision(state)      `[update_shroud]`, and `[auto_shroud]
                            active=yes` while delayed
                            (`undo_list::commit_vision`).

An advancement on the delaying side's own turn clears nothing.

State, on `global_info` and replaced rather than mutated (search forks
share it): `_shroud_delayed`, the frozenset of sides that delay, and
`_pending_vision`, a tuple of (side, route, unit id, vision points,
slowed) with the route a tuple of (x, y). The Rust core keeps the same
(rust/wesnoth_core/src/core_shroud.rs); the applier's hooks are in
`tools.replay_dataset._apply_command`.
"""
from __future__ import annotations

import copy
import logging
from typing import Iterable, Tuple

from wesnoth_ai.classes import GameState, Unit
from wesnoth_ai.visibility import _fog_on, _set_cleared, unit_vision, visible_hexes_for

log = logging.getLogger("delayed_shroud")

SHROUD_DELAYED = "_shroud_delayed"
PENDING_VISION = "_pending_vision"

Hex = Tuple[int, int]


def delaying_sides(state: GameState) -> frozenset:
    return frozenset(getattr(state.global_info, SHROUD_DELAYED, None) or ())


def pending_vision(state: GameState) -> tuple:
    return tuple(getattr(state.global_info, PENDING_VISION, None) or ())


def vision_delayed(state: GameState, side: int) -> bool:
    """Whether `side`'s fog clearing waits for a commit: fog is on, it is
    the side's turn and the side delays its shroud updates
    (`current_uses_fog_`, src/actions/move.cpp:371 at 1.18.4; the
    recruit, create.cpp:697; an advancement, vision.cpp:467-469)."""
    return (_fog_on(state) and side == state.global_info.current_side
            and side in delaying_sides(state))


def defer_vision(state: GameState, unit: Unit, route: Iterable[Hex]) -> None:
    """`unit`'s action over `route` waits for the commit
    (`undo_list::add_move` / `add_recruit`, with the unit's
    `clearer_info`, src/actions/vision.cpp:100-106)."""
    entry = (int(unit.side), tuple((int(x), int(y)) for x, y in route), str(unit.id),
             int(unit.max_moves), "slowed" in (unit.statuses or ()))
    setattr(state.global_info, PENDING_VISION, pending_vision(state) + (entry,))


def _viewer(unit: Unit, vision: int, slowed: bool) -> Unit:
    """`unit` with the vision points and slowed status an entry recorded."""
    if int(unit.max_moves) == vision and ("slowed" in (unit.statuses or ())) == slowed:
        return unit
    view = copy.copy(unit)
    view.max_moves = vision
    statuses = set(unit.statuses or ())
    if slowed:
        statuses.add("slowed")
    else:
        statuses.discard("slowed")
    view.statuses = statuses
    return view


def _apply_pending_vision(state: GameState) -> bool:
    """`undo_list::apply_shroud_changes` (src/actions/undo.cpp:431-470):
    the fog cleared from every hex of every pending route with the
    vision recorded, when the current side still delays its updates.
    Returns whether a hex was cleared."""
    side = state.global_info.current_side
    pending = pending_vision(state)
    if not _fog_on(state) or side not in delaying_sides(state) or not pending:
        return False
    units = {u.id: u for u in state.map.units}
    cleared = False
    for entry_side, route, unit_id, vision, slowed in pending:
        unit = units.get(unit_id)
        if unit is None:
            log.warning("delayed shroud: unit %s left the board before its vision was "
                        "committed; its pending vision is dropped", unit_id)
            continue
        viewer = _viewer(unit, vision, slowed)
        before = frozenset(visible_hexes_for(state, entry_side))
        after = before.union(*(unit_vision(state, viewer, at=h) for h in route))
        if after != before:
            cleared = True
            _set_cleared(state, entry_side, after)
    return cleared


def clear_undo_stack(state: GameState) -> None:
    """`undo_list::clear` (undo.cpp:201-215): an action that cannot be
    undone commits the pending vision and empties the stack."""
    _apply_pending_vision(state)
    setattr(state.global_info, PENDING_VISION, ())


def commit_vision(state: GameState) -> bool:
    """`undo_list::commit_vision` (undo.cpp:222-236): the pending vision
    committed; the stack empties when something was cleared."""
    cleared = _apply_pending_vision(state)
    if cleared:
        setattr(state.global_info, PENDING_VISION, ())
    return cleared


def apply_auto_shroud(state: GameState, active: bool) -> None:
    """The `[auto_shroud]` synced command (src/synced_commands.cpp:367-381):
    turning the updates back on commits the pending vision first."""
    side = state.global_info.current_side
    delayed = delaying_sides(state)
    if active and side in delayed:
        commit_vision(state)
    setattr(state.global_info, SHROUD_DELAYED, delayed - {side} if active else delayed | {side})


def apply_update_shroud(state: GameState) -> None:
    """The `[update_shroud]` synced command (synced_commands.cpp:383-398)."""
    commit_vision(state)


def reset_pending(state: GameState) -> None:
    """A side's turn starts with an empty undo stack
    (`undo_list::new_side_turn`, undo.cpp:243-262)."""
    if pending_vision(state):
        setattr(state.global_info, PENDING_VISION, ())
