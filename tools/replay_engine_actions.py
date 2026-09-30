"""Commands a replay records under a player's side that the player did not
decide at that position: the engine made them. They are applied to the state
like any command, and never paired as decisions.

- **Standing orders.** A move that stops short of its route's end without
  being interrupted leaves the unit a goto order for the route's end
  (`unit_mover::~unit_mover`, src/actions/move.cpp:401-414, 1.18.4). At the
  start of its side's next turn, before any input from the player, the engine
  moves every unit toward its goto (`playmp_controller::play_human_turn`
  calls `execute_gotos`, src/playmp_controller.cpp:149-151;
  `menu_handler::execute_gotos`, src/menu_events.cpp:903-980), and records
  each move as it records a player's (`move_unit_and_record` with the default
  arguments, so nothing in the command marks it).
- **Timeouts.** When a side's turn timer runs out the engine ends the turn
  (src/playmp_controller.cpp:177-182) and records the side's new time,
  `1000 * min(left_s + turn_bonus + action_bonus * n, reservoir)`
  (`after_human_turn`, :290-305), with `left_s` 0 after a timeout; `n` counts
  the turn's recruits and village captures (src/menu_events.cpp:358,
  src/actions/move.cpp:172). A timeout is certain only when the recorded time
  is the turn bonus exactly, with no action bonus and a bonus below the
  reservoir: under a cap the time a player left and a timeout read the same.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

Hex = Tuple[int, int]


@dataclass(frozen=True)
class TurnTimer:
    """A game's turn timer, in seconds, as [multiplayer] declares it."""
    init_s: int
    turn_bonus_s: int
    action_bonus_s: int
    reservoir_s: int

    def is_timeout(self, recorded_ms: int) -> bool:
        """The recorded new time can only follow a turn that ran out."""
        return (self.action_bonus_s == 0 and self.turn_bonus_s < self.reservoir_s
                and recorded_ms == 1000 * self.turn_bonus_s)


def turn_timer(mp_attrs: dict) -> Optional[TurnTimer]:
    """The timer of a replay's [multiplayer] attributes, or None when off."""
    if str(mp_attrs.get("mp_countdown", "no")).strip('"').lower() not in ("yes", "true", "1"):
        return None

    def seconds(key: str) -> int:
        try:
            return int(str(mp_attrs.get(key, "0")).strip('"') or 0)
        except ValueError:
            return 0

    return TurnTimer(seconds("mp_countdown_init_time"), seconds("mp_countdown_turn_bonus"),
                     seconds("mp_countdown_action_bonus"), seconds("mp_countdown_reservoir_time"))


def _route_end(cmd: list) -> Hex:
    """The hex a compact move was ordered to: the clicked hex when the engine
    stopped the unit short of it, else the path's end."""
    order = cmd[4] if len(cmd) > 4 and isinstance(cmd[4], dict) else None
    if order is not None and "clicked" in order:
        return int(order["clicked"][0]), int(order["clicked"][1])
    return int(cmd[1][-1]), int(cmd[2][-1])


def engine_goto_moves(commands: List[list]) -> List[int]:
    """Indices of the compact moves the engine made for standing orders at a
    side's turn start. A unit holds a goto after a move with an order whose
    `stopped_early` is False (it stopped where the turn's movement ended, not
    interrupted); at its side's next turn start, before the side's first
    decision, a move of the unit toward that hex is the engine's. Units are
    followed by the hex they stand on."""
    goto: dict = {}
    marked: List[int] = []
    opening_side: Optional[int] = None     # side at turn start, before its first decision
    for i, cmd in enumerate(commands):
        if not cmd:
            continue
        kind = cmd[0]
        if kind == "init_side":
            opening_side = int(cmd[1])
            continue
        if kind != "move":
            opening_side = None              # any other command is the player's: the turn start is over
            continue
        side = int(cmd[3]) if len(cmd) > 3 else 0
        start = (int(cmd[1][0]), int(cmd[2][0]))
        stop = (int(cmd[1][-1]), int(cmd[2][-1]))
        if opening_side is not None and side == opening_side and goto.get(start) == _route_end(cmd):
            marked.append(i)
        elif side == opening_side:
            opening_side = None
        goto.pop(start, None)
        order = cmd[4] if len(cmd) > 4 and isinstance(cmd[4], dict) else None
        if order is not None and order.get("stopped_early") is False and stop != _route_end(cmd):
            goto[stop] = _route_end(cmd)
    return marked
