"""What each player side saw of the other sides' units, computed from the
Python applier's state: the oracle `tools/diff_core.py` holds the Rust
core's sighting records against (rust/wesnoth_core/src/core_sight.rs;
docs/parity_memory_design_20260929.md "The watched turn"). The Python
applier keeps no sighting record; this one follows the replay beside it,
from the rules:

- after a command the core notes sightings after (a move, an attack, a
  recruit, a turn start or end, the shroud commands), each player side
  records every unit of another side it sees
  (`visibility.units_visible_to_python`), scenery excluded: its type, hit
  points, maximum hit points and hex; and adds (side, type) to the types it
  has seen in the game;
- during a move, each player side other than the mover's records the
  mover on the last hex of the walked route (its start, then every hex it
  entered) that the side sees and where the mover is not hidden by its
  hide ability;
- a unit that leaves the board leaves a side's record when the side saw
  its hex at that moment, and otherwise stays, as gone, until the side's
  end of turn;
- a side's end of turn empties its record; the seen types stay for the
  game.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, FrozenSet, List, Set, Tuple

from wesnoth_ai.classes import GameState, Position, Unit

PLAYER_SIDES = (1, 2)
# The commands after which the core notes what each side sees.
NOTED_KINDS = frozenset({"move", "attack", "recruit", "init_side", "end_turn", "auto_shroud",
                         "update_shroud", "menu_item"})

Row = Tuple[str, str, int, int, int, int]      # id, type, hp, max hp, x, y


class SightingOracle:
    """Follows one replay: `before(gs, cmd)`, apply the command to `gs`,
    then `after(gs, cmd)`; `records()` answers in the core's export form."""

    def __init__(self) -> None:
        self.sightings: Dict[int, Dict[str, Row]] = {s: {} for s in PLAYER_SIDES}
        self.gone: Dict[int, Set[str]] = {s: set() for s in PLAYER_SIDES}
        self.seen_types: Dict[int, Set[Tuple[int, str]]] = {s: set() for s in PLAYER_SIDES}
        self._board: Dict[str, Tuple[int, int]] = {}
        self._seen_before: Dict[int, FrozenSet[Tuple[int, int]]] = {}
        self._mover = None
        self._side = 0

    # ---- around a command
    def before(self, gs: GameState, cmd: list) -> None:
        """Read what the command's rules need from the state before it."""
        from wesnoth_ai.visibility import visible_hexes_for
        kind = cmd[0] if cmd else ""
        self._side = int(gs.global_info.current_side)
        self._board, self._seen_before, self._mover = {}, {}, None
        if kind == "attack":             # the one command that removes units
            self._board = {u.id: (u.position.x, u.position.y) for u in gs.map.units}
            if getattr(gs.global_info, "_fog", True):
                self._seen_before = {s: frozenset(visible_hexes_for(gs, s)) for s in PLAYER_SIDES}
        if kind == "move":
            sx, sy = int(cmd[1][0]), int(cmd[2][0])
            from_side = int(cmd[3]) if len(cmd) > 3 else 0
            self._mover = next((u.id for u in gs.map.units if (u.position.x, u.position.y) == (sx, sy)
                                and (from_side == 0 or u.side == from_side)), None)

    def after(self, gs: GameState, cmd: list) -> None:
        kind = cmd[0] if cmd else ""
        if kind not in NOTED_KINDS:
            return
        self._note_departures(gs)
        if kind == "move" and self._mover is not None:
            self._note_path(gs, cmd)
        if kind == "end_turn" and self._side in self.sightings:
            self.sightings[self._side].clear()
            self.gone[self._side].clear()
        self._note_visible(gs)

    # ---- the rules
    def _record(self, side: int, u: Unit, x: int, y: int) -> None:
        self.sightings[side][u.id] = (u.id, u.name, int(u.current_hp), int(u.max_hp), int(x), int(y))
        self.seen_types[side].add((int(u.side), u.name))

    def _note_visible(self, gs: GameState) -> None:
        from wesnoth_ai.visibility import is_scenery_unit, units_visible_to_python
        for side in PLAYER_SIDES:
            for u in units_visible_to_python(gs, side):
                if u.side != side and not is_scenery_unit(u):
                    self._record(side, u, u.position.x, u.position.y)
        on_board = {u.id for u in gs.map.units}
        for side in PLAYER_SIDES:
            record, gone = self.sightings[side], self.gone[side]
            for uid in [uid for uid in record if uid not in on_board and uid not in gone]:
                del record[uid]

    def _note_departures(self, gs: GameState) -> None:
        on_board = {u.id for u in gs.map.units}
        for uid, hex_ in self._board.items():
            if uid in on_board:
                continue
            for side in PLAYER_SIDES:
                if uid in self.sightings[side] and self._seen_before and hex_ not in self._seen_before[side]:
                    self.gone[side].add(uid)

    def _note_path(self, gs: GameState, cmd: list) -> None:
        from wesnoth_ai.visibility import is_scenery_unit, visible_hexes_for
        mover = next((u for u in gs.map.units if u.id == self._mover), None)
        walk = getattr(gs.global_info, "_last_move_walk", None)
        if mover is None or walk is None or is_scenery_unit(mover):
            return
        xs, ys = [int(v) for v in cmd[1]], [int(v) for v in cmd[2]]
        landed = tuple(walk["landed"])
        final = next((j for j in range(len(xs)) if (xs[j], ys[j]) == landed), 0)
        if final < 1:                    # the unit did not leave its hex: what it shows is noted after
            return
        walked = list(zip(xs[:final + 1], ys[:final + 1]))
        fog_on = getattr(gs.global_info, "_fog", True)
        for side in PLAYER_SIDES:
            if side == mover.side:
                continue
            seen = frozenset(visible_hexes_for(gs, side)) if fog_on else None
            last = next((h for h in reversed(walked)
                         if (seen is None or h in seen) and not _hidden_at(gs, mover, h)), None)
            if last is not None:
                self._record(side, mover, *last)

    # ---- the core's export form
    def records(self, side: int) -> Tuple[List[Row], List[Tuple[int, str]], List[str]]:
        return (sorted(self.sightings[side].values()), sorted(self.seen_types[side]),
                sorted(self.gone[side]))


def _hidden_at(gs: GameState, mover: Unit, hex_: Tuple[int, int]) -> bool:
    """Whether `mover`, standing on `hex_`, is hidden there by its hide
    ability: not uncovered, its cover active on the hex, and no enemy of
    it that is not scenery adjacent (`visibility._discovered_by_adjacency`)."""
    from wesnoth_ai.visibility import _discovered_by_adjacency, _hide_cover_active
    uncovered = getattr(gs.global_info, "_uncovered_units", None) or set()
    if mover.id in uncovered:
        return False
    there = dataclasses.replace(mover, position=Position(x=hex_[0], y=hex_[1]))
    return _hide_cover_active(gs, there) and not _discovered_by_adjacency(gs, there, 0)


def record_differences(oracle: SightingOracle, core) -> List[str]:
    """The core's sighting records against the oracle's, per player side."""
    out = []
    for side in PLAYER_SIDES:
        rows, types, gone = oracle.records(side)
        core_rows = sorted(tuple(r) for r in core.sightings_export(side))
        core_types = sorted(tuple(t) for t in core.seen_types_export(side))
        core_gone = sorted(core.sightings_gone_export(side))
        if core_rows != rows:
            only_core = sorted(set(core_rows) - set(rows))[:3]
            only_oracle = sorted(set(rows) - set(core_rows))[:3]
            out.append(f"side {side} sightings: core {only_core} oracle {only_oracle}")
        if core_types != types:
            out.append(f"side {side} seen types: core-only {sorted(set(core_types) - set(types))[:3]} "
                       f"oracle-only {sorted(set(types) - set(core_types))[:3]}")
        if core_gone != gone:
            out.append(f"side {side} gone: core {core_gone} oracle {gone}")
    return out
