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
  has seen in the game, unless the scenario placed the unit (the applier's
  `_scenario_unit_ids`);
- nothing along a move's route: the record follows the display with move
  animations off (user ruling 2026-10-01), so a unit that walks out of
  view is remembered where it stood before the move;
- a fight that refogs the defender's side (the defender died, or was newly
  slowed or petrified) was shown to that side first: it records what it
  sees with the fog it had before the fight, the attacker as the fight
  left it (the engine advances it after the refog);
- a unit that leaves the board leaves a side's record when the side saw
  its hex at that moment, and otherwise stays, as gone, until the side's
  end of turn;
- a side's end of turn empties its record; the seen types stay for the
  game.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Set, Tuple

from wesnoth_ai.classes import GameState, Unit

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
        self._placed: FrozenSet[str] = frozenset()
        self._side = 0

    # ---- around a command
    def before(self, gs: GameState, cmd: list) -> None:
        """Read what the command's rules need from the state before it."""
        from wesnoth_ai.visibility import visible_hexes_for
        kind = cmd[0] if cmd else ""
        self._side = int(gs.global_info.current_side)
        self._placed = frozenset(getattr(gs.global_info, "_scenario_unit_ids", None) or ())
        self._board, self._seen_before = {}, {}
        if kind == "attack":             # the one command that removes units, or refogs mid-command
            self._board = {u.id: (u.position.x, u.position.y) for u in gs.map.units}
            if getattr(gs.global_info, "_fog", True):
                self._seen_before = {s: frozenset(visible_hexes_for(gs, s)) for s in PLAYER_SIDES}

    def after(self, gs: GameState, cmd: list) -> None:
        kind = cmd[0] if cmd else ""
        if kind not in NOTED_KINDS:
            return
        self._note_departures(gs)
        if kind == "attack":
            self._note_fight(gs)
        if kind == "end_turn" and self._side in self.sightings:
            self.sightings[self._side].clear()
            self.gone[self._side].clear()
        self._note_visible(gs)

    # ---- the rules
    def _record(self, side: int, u: Unit, x: int, y: int) -> None:
        self.sightings[side][u.id] = (u.id, u.name, int(u.current_hp), int(u.max_hp), int(x), int(y))
        if u.id not in self._placed:
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

    def _note_fight(self, gs: GameState) -> None:
        """The applier's `_last_fight` says whether the fight refogged the
        defender's side and holds the attacker as the fight left it."""
        from wesnoth_ai.visibility import is_scenery_unit, units_visible_to_python
        fight = getattr(gs.global_info, "_last_fight", None)
        if not fight or not fight["refog"] or fight["defender_side"] not in self.sightings:
            return
        side, shown = int(fight["defender_side"]), fight["attacker"]
        fog = self._seen_before.get(side) if self._seen_before else None
        units = [u for u in gs.map.units if shown is None or u.id != shown.id]
        for u in units_visible_to_python(gs, side, vis_set=fog, units=units + ([shown] if shown else [])):
            if u.side != side and not is_scenery_unit(u):
                self._record(side, u, u.position.x, u.position.y)

    def _note_departures(self, gs: GameState) -> None:
        on_board = {u.id for u in gs.map.units}
        for uid, hex_ in self._board.items():
            if uid in on_board:
                continue
            for side in PLAYER_SIDES:
                if uid in self.sightings[side] and self._seen_before and hex_ not in self._seen_before[side]:
                    self.gone[side].add(uid)

    # ---- the core's export form
    def records(self, side: int) -> Tuple[List[Row], List[Tuple[int, str]], List[str]]:
        return (sorted(self.sightings[side].values()), sorted(self.seen_types[side]),
                sorted(self.gone[side]))


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
