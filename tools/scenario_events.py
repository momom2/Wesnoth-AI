"""A scenario's [event] blocks as data, and the map a [terrain] event leaves.

Many maps trigger gameplay-affecting WML `[event]` blocks during play
(Aethermaw morphs impassable terrain into water at turns 4-6;
 Caves of the Basilisk spawns petrified statue units at prestart;
 etc.). The replay file does NOT carry the result of those events --
 the engine re-fires them by re-loading the scenario .cfg.

The Rust core runs the events (rust/wesnoth_core/src/events.rs, which
`game_core.CoreState.setup_scenario` hands the scenario's WML). This
module keeps the Python side of them: the events as `ScenarioEvent`s
(`collect_events`), which a view of the core carries with each one's
fired latch, and the hex set and terrain codes a view builds from the
core's terrain writes (`terrain_writes_applied`).

Dependencies: wesnoth_ai.rules.scenario_cfg (load_scenario_wml),
              tools.replay_extract (WMLNode), classes
Dependents: wesnoth_ai.game_core
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wesnoth_ai.classes import Hex, Position
from wesnoth_ai.rules.scenario_cfg import load_scenario_wml
from tools.replay_extract import WMLNode


def standard_event_name(name: str) -> str:
    """The engine's `event_handlers::standardize_name`
    (src/game_events/manager_impl.cpp:65-76, 1.18.4): trimmed, every
    internal space an underscore, case kept. `side 1 turn` and
    `side_1_turn` are one name; `Prestart` is not `prestart`."""
    return name.strip().replace(" ", "_")


def event_names(raw: str) -> List[str]:
    """The names an [event] answers to: its `name=` is a comma-separated
    list, split with empty pieces dropped, each piece standardized
    (`event_handler::names`, src/game_events/handlers.cpp:64-88)."""
    return [standard_event_name(piece) for piece in raw.split(",") if piece.strip()]


@dataclass
class ScenarioEvent:
    """One [event] block extracted from a scenario .cfg."""
    name: str                         # "prestart", "side 1 turn 4", etc.
    first_time_only: bool = True
    actions: List[WMLNode] = field(default_factory=list)
    fired: bool = False               # latched by the interpreter
    scenario_id: str = ""             # so an unmodelled tag names its map

    @property
    def names(self) -> List[str]:
        return event_names(self.name)

    def can_fire(self) -> bool:
        return not (self.first_time_only and self.fired)


def collect_events(root: WMLNode, scenario_id: str = "") -> List[ScenarioEvent]:
    """Find every [event] block under [multiplayer] / [scenario] and
    return them in WML-order so the caller can fire them sequentially."""
    out: List[ScenarioEvent] = []
    container = root.first("multiplayer") or root.first("scenario")
    if container is None:
        return out
    for ev in container.all("event"):
        name = ev.attrs.get("name", "").strip().strip('"')
        first_time = ev.attrs.get("first_time_only", "yes").strip().lower() in (
            "yes", "true", "1",
        )
        # The "actions" of an event are its inner WML children except
        # nested [filter] (those are predicates, not actions).
        actions = [ch for ch in ev.children if ch.tag != "filter"]
        out.append(ScenarioEvent(
            name=name, first_time_only=first_time, actions=actions,
            scenario_id=scenario_id,
        ))
    return out


def terrain_writes_applied(hexes, codes, raw: str, writes):
    """The hex set, terrain codes and raw map data after `writes`, each
    a (WML x, WML y, code) the way a [terrain] event writes it; new
    containers, the given ones untouched. A view of the Rust core
    (`game_core`, from the core's terrain log) builds its Python map
    through here.

    `codes` keeps the FULL code, overlay included: an overlay can
    dominate its base (^Xo is the impassable overlay, mvt_alias=Xt,
    wesnoth_src/data/core/terrain.cfg:1743-1751), and storing
    Aethermaw's turn-6 'Chw^Xo' as 'Chw' made its whirlpool walls
    walkable ("found corrupt movement in replay", engine-verified
    2026-07-29).

    COPY-ON-WRITE (adversarial-review HIGH finding, 2026-07-18): search
    forks alias `map.hexes` and `_terrain_codes`, so mutating them in
    place morphed the live game's terrain from inside a fork. A new hex
    set also changes `id(gs.map.hexes)`, which invalidates the caches
    keyed on it (`observe.map_geometry`).

    The raw map data includes the 1-hex border, so WML (X, Y) is
    raw_cells[Y][X]. A [terrain] event replaces the terrain, never the
    hex's starting-position label, and the engine writes a cell back as
    label + " " + code (wesnoth_src/src/terrain/translation.cpp:775-782,
    number_to_string_)."""
    from tools.replay_dataset import _parse_hex_code
    from wesnoth_ai.rules.terrain_resolver import split_start_position, terrain_mask
    raw_cells: List[List[str]] = [[c.strip() for c in row.split(",")]
                                  for row in (raw.splitlines() if raw else [])]
    new_hexes = set(hexes)
    by_pos = {(h.position.x, h.position.y): h for h in new_hexes}
    new_codes = dict(codes) if codes is not None else None
    for wml_x, wml_y, new_code in writes:
        py_x, py_y = wml_x - 1, wml_y - 1
        new_terr, new_mods = _parse_hex_code(new_code)
        old_hex = by_pos.get((py_x, py_y))
        if old_hex is not None:
            new_hexes.discard(old_hex)
        fresh = Hex(position=Position(x=py_x, y=py_y), terrain_types=set(new_terr),
                    modifiers=set(new_mods), terrain_mask=terrain_mask(new_code))
        new_hexes.add(fresh)
        by_pos[(py_x, py_y)] = fresh
        if new_codes is not None:
            new_codes[(py_x, py_y)] = new_code
        if 0 <= wml_y < len(raw_cells) and 0 <= wml_x < len(raw_cells[wml_y]):
            label, _old_code = split_start_position(raw_cells[wml_y][wml_x])
            raw_cells[wml_y][wml_x] = f"{label} {new_code}" if label else new_code
    new_raw = "\n".join(", ".join(row) for row in raw_cells) if raw_cells else raw
    return new_hexes, new_codes, new_raw


def load_events_for_scenario(scenario_id: str) -> List[ScenarioEvent]:
    """Convenience wrapper: load the .cfg and return its events. Returns
    an empty list if the scenario isn't found in the source tree."""
    root = load_scenario_wml(scenario_id)
    if root is None:
        return []
    return collect_events(root, scenario_id)


__all__ = [
    "ScenarioEvent", "load_events_for_scenario", "collect_events",
    "standard_event_name", "event_names", "terrain_writes_applied",
]
