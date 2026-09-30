"""Hand-built games for the parity observation's tests: a grass board
inside an impassable border, units as a replay record lists them, the
Rust core of it, and the parity encoding of the side to move."""
from __future__ import annotations

import json
from typing import Dict, Iterable, Optional, Sequence, Tuple

from wesnoth_ai.constants import DEFAULT_FACTIONS
from wesnoth_ai.paths import UNIT_STATS_PATH

FACTION_IDS = {f: i for i, f in enumerate(DEFAULT_FACTIONS)}


def board(width: int, height: int, special: Optional[Dict[Tuple[int, int], str]] = None) -> str:
    """The map data of a width x height board of grass inside an
    impassable border; `special` gives other codes by 0-indexed hex."""
    special = special or {}
    rows = [", ".join(special.get((x, y), "Gg") for x in range(width)) for y in range(height)]
    border = ", ".join(["Xv"] * (width + 2))
    return "\n".join([border] + [f"Xv, {r}, Xv" for r in rows] + [border])


def record(units: Iterable[tuple], *, width: int = 20, height: int = 7,
           special: Optional[Dict[Tuple[int, int], str]] = None, fog: bool = True,
           factions: Sequence[str] = ("Loyalists", "Rebels"),
           recruits: Optional[Dict[int, list]] = None,
           villages: Optional[Dict[int, list]] = None, gold: Sequence[int] = (100, 100),
           village_gold: int = 2, village_support: int = 1, experience_modifier: int = 100,
           tod_start_index: int = 0) -> dict:
    """A replay record of a two-sided game with no commands; `units` are
    (type, side, x, y, is_leader) with 0-indexed hexes."""
    recruits = recruits or {}
    villages = villages or {}
    return {
        "game_id": "parity", "scenario_id": "", "map_data": board(width, height, special),
        "experience_modifier": experience_modifier, "tod_start_index": tod_start_index,
        "starting_units": [{"uid": k + 1, "type": t, "side": s, "x": x, "y": y, "is_leader": leader}
                           for k, (t, s, x, y, leader) in enumerate(units)],
        "starting_sides": [{"side": s, "faction": factions[s - 1], "gold": gold[s - 1],
                            "recruit": list(recruits.get(s, [])), "fog": fog, "shroud": False,
                            "village_income": village_gold, "village_support": village_support}
                           for s in (1, 2)],
        "starting_villages": [{"x": x, "y": y, "side": s}
                              for s, owned in villages.items() for x, y in owned],
        "commands": [],
    }


def state_of(data: dict, *, time_areas: Optional[Dict[Tuple[int, int], list]] = None,
             chose_random: Sequence[bool] = (False, False), era_factions: Optional[tuple] = None):
    """The record's starting GameState, with time areas (cycles phased
    to turn 1), the sides' Random choice and the era's factions."""
    from tools.replay_dataset import _build_initial_gamestate
    gs = _build_initial_gamestate(data)
    if time_areas:
        gs.global_info._time_areas = dict(time_areas)
    for side, random_choice in zip(gs.sides, chose_random):
        side.chose_random = bool(random_choice)
    if era_factions is not None:
        gs.era_factions = tuple(era_factions)
    return gs


def core_of(data: dict, **kw):
    """The Rust core (`game_core.CoreState`) of the record's start."""
    from wesnoth_ai.game_core import CoreState
    return CoreState.from_state(state_of(data, **kw))


def unit_types() -> dict:
    return json.loads(UNIT_STATS_PATH.read_text(encoding="utf-8"))["units"]


def vocab_of(names: Iterable[str]) -> Dict[str, int]:
    return {n: i for i, n in enumerate(sorted(set(names)))}


def parity_raw(cs, type_to_id: Dict[str, int], *, relevant_set: bool = True, parity: bool = True):
    """The side to move's encoding under obs8's flags, with the parity
    recipe (its observation and the relevant set's version 2) when
    `parity`."""
    return cs.encode_raw(type_to_id=type_to_id, faction_to_id=FACTION_IDS, relevant_set=relevant_set,
                         fog_hides_enemy_villages=True, terrain_multi_hot=True,
                         observation_parity=parity,
                         relevant_set_version=2 if parity and relevant_set else 1)


def seed_rolling(unit_type: str, wanted: str) -> str:
    """A recruit seed whose trait roll for `unit_type` includes `wanted`."""
    import wesnoth_core
    for k in range(1, 4000):
        seed = f"{k:08x}"
        if wanted in wesnoth_core.roll_type_traits(unit_type, seed):
            return seed
    raise AssertionError(f"no seed below 4000 rolls {wanted} for {unit_type}")


def unit_row(raw, unit_id: str):
    """The unit stream's row of `unit_id`."""
    return raw.unit_feats[raw.unit_ids.index(unit_id)]


def unit_id_at(cs, x: int, y: int) -> str:
    uid = cs.core.unit_id_at(x, y)
    assert uid is not None, (x, y)
    return uid
