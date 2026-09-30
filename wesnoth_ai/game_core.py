"""The Rust-owned game state (docs/rust_port_plan.md phase 4,
docs/rust_core_port_20260928.md): the adapter between
`wesnoth_ai.classes.GameState` and `wesnoth_core.GameCore`.

`CoreState.from_state(gs)` builds a core from a Python state: the map's
geometry, one-class view and terrain codes once per hex set
(`map_static`; the core resolves every terrain fact and movement class
from the codes itself), then the units, sides, globals, the stash the
simulator keeps on `global_info` and any scenario events a Python setup
left. `setup_scenario` runs a scenario's setup on the core itself. The
unit and terrain databases are loaded into the extension once per
process (`load_databases`). `to_state()` builds a GameState view whose
modeled content equals the original (tests/test_game_core.py).

What the core does not model -- the mask and fog, the raw map data and
every other stash key -- stays Python in `statics`, handed to the views
by reference. The hex set, the terrain codes and the time areas are the
core's and are brought into `statics` when an event changed them
(`_sync_map`). A search fork (`fork()`) copies `statics` by
`GlobalInfo.__deepcopy__`'s rules (`_fork_statics`): the hex set, the
mask, the fog and the terrain codes stay aliased, dict, set and list
values are copied shallowly. A unit's underscore attributes are fields
of its core record (`unit_fields`, `unit_from_fields`).
"""
from __future__ import annotations

import copy
import logging
import os
import weakref
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from wesnoth_ai.classes import (Attack, GameState, GlobalInfo, Map, Position, SideInfo,
                                TerrainModifiers, Unit, opponent_of)
from wesnoth_ai.constants import DEFAULT_ERA_FACTIONS

log = logging.getLogger("game_core")

_KERNEL_CHECKED = False
_GAME_CORE = None

# The GlobalInfo stash the core models; every other underscore attribute
# is shared by reference through `CoreState.statics`.
MODELED_GLOBALS = (
    "_fog", "_village_owner", "_uncovered_units", "_recruit_rejected_hexes",
    "_tod_start_offset", "_experience_modifier",
    "_next_uid_counter", "_rng_request_counter", "_advance_choices", "_pickadvance_game",
    "_did_first_init_side", "_last_move_walk", "_last_checkup_strikes", "_last_advance_events",
    "_advance_uniform", "_advance_salt", "_advance_counter", "_fog_cleared",
    "_shroud_delayed", "_pending_vision", "_plan_unit_advance", "_pa_fresh_turn",
)
# The unit underscore attributes the state comparison checks; the core
# also keeps `_object_effects` (WML nodes) and `_ai_guardian`.
UNIT_STASH_KEYS = ("_defense_table", "_pickadvance", "_trait_order", "_feeding_count", "_wml_role")
_DROPPED_GLOBALS = ("_hex_lookup_cache_id", "_hex_lookup_by_xy", "_hex_lookup_by_wml")
# What each player side saw of the other sides' units, which the core keeps
# (rust/wesnoth_core/src/core_sight.rs) and a view carries for
# `classes.state_digest`: `_sightings` {side: ((id, type, hp, max hp, x,
# y), ...)} and `_seen_types` {side: ((other side, type), ...)}. The Python
# applier keeps neither, so the state comparisons leave them out.
SIGHT_RECORDS = ("_sightings", "_seen_types", "_sightings_gone")
# The players' sides, the ones that keep a sighting record.
_RECORD_SIDES = (1, 2)
# The wheel phase this adapter reads: 23 keeps the sighting records and
# builds the parity observation, 24 delays a side's shroud updates, 25 adds
# the Plan Unit Advance modification's undo blocks, 26 keeps a unit that
# left the board unseen in the sighting record.
_CORE_PHASE = 26

# Scenario WML in the core's tuple form, per scenario id (the WML a
# process reads for a scenario never changes).
_WML_TUPLES: Dict[str, tuple] = {}

# The core behind each bound view: id(view) -> (weak reference to the
# view, CoreState, the view's fingerprint when WESNOTH_CHECK_VIEWS is set).
_VIEW_CORES: Dict[int, tuple] = {}


def game_core_class():
    """`wesnoth_core.GameCore`, or None (wheel absent or older). The
    unit and terrain databases are loaded into the extension on the
    first call."""
    global _KERNEL_CHECKED, _GAME_CORE
    if not _KERNEL_CHECKED:
        _KERNEL_CHECKED = True
        try:
            import wesnoth_core
        except ImportError:
            wesnoth_core = None
        # Phase 22: the core reads the unit and terrain databases, resolves
        # every terrain fact and movement class from the terrain codes, keeps
        # each unit's underscore attributes in its record, builds units
        # itself (recruits, plague corpses, advancement), runs the
        # scenario's events, encodes the terrain set, and answers the
        # defender's weapon choice and an attack's exact outcomes. Phase
        # 23: it keeps each side's sighting record and builds the parity
        # observation.
        if wesnoth_core is not None and getattr(wesnoth_core, "__phase__", 0) >= _CORE_PHASE:
            load_databases(wesnoth_core)
            _GAME_CORE = wesnoth_core.GameCore
    return _GAME_CORE


def core_enabled() -> bool:
    """The Rust-owned state as the state of record of the simulator and
    of replay reconstruction: the wheel carries GameCore and
    WESNOTH_RUST_CORE is not 0 (on by default since 2026-09-28; 0 keeps
    the Python applier, the port's oracle, until its retirement)."""
    if os.environ.get("WESNOTH_RUST_CORE", "1") == "0":
        return False
    return game_core_class() is not None


def _view_fingerprint(gs: GameState) -> tuple:
    from wesnoth_ai.classes import state_key
    return (state_key(gs), bool(getattr(gs.global_info, "_fog", True)), id(gs.map.hexes))


def bind_view(gs: GameState, cs: "CoreState") -> None:
    """Record that `gs` is a view of `cs`, so that what is computed from
    the view (its encoding, `encoder.encode_raw`) comes from the core. A
    bound view is read-only: an edit made in it is not in the core.
    With WESNOTH_CHECK_VIEWS set (the test suite sets it) `core_of`
    refuses a view edited after its binding."""
    key = id(gs)

    def _drop(ref, key=key):
        hit = _VIEW_CORES.get(key)
        if hit is not None and hit[0] is ref:
            _VIEW_CORES.pop(key, None)

    fp = _view_fingerprint(gs) if os.environ.get("WESNOTH_CHECK_VIEWS") else None
    _VIEW_CORES[key] = (weakref.ref(gs, _drop), cs, fp)


def core_of(gs: GameState) -> Optional["CoreState"]:
    """The core a view is bound to (`bind_view`), or None for a state
    built another way."""
    hit = _VIEW_CORES.get(id(gs))
    if hit is None or hit[0]() is not gs:
        return None
    if hit[2] is not None and _view_fingerprint(gs) != hit[2]:
        raise AssertionError(
            "a view of the Rust core was edited in place after it was bound: the core does not "
            "see the edit. Hand the edited view back (`sim.gs = view`) or edit a copy.")
    return hit[1]


def snapshot_view(gs: GameState) -> GameState:
    """A copy of `gs` to keep while the game goes on: for a view bound to
    a core, a view of a fork of that core (bound to it, so it is encoded
    by the core, the parity observation included); for any other state,
    `copy.deepcopy(gs)`, which is unbound."""
    cs = core_of(gs)
    if cs is None:
        return copy.deepcopy(gs)
    fork = cs.fork()
    view = fork.to_state()
    bind_view(view, fork)
    return view


def unit_db_fallbacks() -> Dict[str, int]:
    """The lookups of unit types the core's database lacks, per name,
    since the process started (each took the fallback statistics)."""
    import wesnoth_core
    return dict(wesnoth_core.fallback_type_counts())


def load_databases(wesnoth_core) -> None:
    """The committed `unit_stats.json` and `terrain_db.json` into the
    extension, for every core this process builds."""
    import json
    from wesnoth_ai.paths import TERRAIN_DB_PATH, UNIT_STATS_PATH
    with UNIT_STATS_PATH.open(encoding="utf-8") as f:
        units = json.load(f)
    with TERRAIN_DB_PATH.open(encoding="utf-8") as f:
        terrain = json.load(f)
    wesnoth_core.load_databases(units, terrain)


def _strict_wml() -> bool:
    """`WESNOTH_STRICT_WML`: an unmodelled event action or value raises."""
    return bool(os.environ.get("WESNOTH_STRICT_WML"))


# ---------------------------------------------------------------------
# Static tables
# ---------------------------------------------------------------------

def map_static(gs: GameState) -> dict:
    """The core's static map arrays from the state: geometry from
    `wesnoth_ai.observe.map_geometry`, the hex set's one-class view and
    terrain classes, the terrain codes (the core resolves heal, light,
    hide cover and the movement classes from them) and the time areas the
    scenario set up."""
    from wesnoth_ai.encoder import _first_terrain_id
    from wesnoth_ai.observe import map_geometry
    geom = map_geometry(gs)
    H = len(geom.keys)
    codes = getattr(gs.global_info, "_terrain_codes", {}) or {}
    areas = getattr(gs.global_info, "_time_areas", None) or {}
    by_pos = {(h.position.x, h.position.y): h for h in gs.map.hexes}
    village_mod = np.zeros(H, dtype=np.uint8)
    castle_mod = np.zeros(H, dtype=np.uint8)
    terrain_type_id = np.zeros(H, dtype=np.int64)
    terrain_mask = np.zeros(H, dtype=np.int64)
    for i, key in enumerate(geom.keys):
        h = by_pos[key]
        village_mod[i] = 1 if TerrainModifiers.VILLAGE in h.modifiers else 0
        castle_mod[i] = 1 if TerrainModifiers.CASTLE in h.modifiers else 0
        terrain_type_id[i] = _first_terrain_id(h.terrain_types)
        terrain_mask[i] = int(getattr(h, "terrain_mask", 0) or 0)
    area_cycle = np.full(H, -1, dtype=np.int64)
    cycles: List[List[int]] = []
    cycle_index: Dict[tuple, int] = {}
    for i, (x, y) in enumerate(geom.keys):
        cyc = areas.get((x, y))
        if cyc:
            key = tuple(int(v) for v in cyc)
            if key not in cycle_index:
                cycle_index[key] = len(cycles)
                cycles.append(list(key))
            area_cycle[i] = cycle_index[key]
    return {
        "hx": geom.hx.tolist(), "hy": geom.hy.tolist(), "nbrs": geom.nbrs.tolist(),
        "castle_or_keep": geom.castle_or_keep.tolist(), "keep": geom.keep.tolist(),
        "village_terrain": geom.village.tolist(), "village_mod": village_mod.tolist(),
        "terrain_type_id": terrain_type_id.tolist(), "terrain_mask": terrain_mask.tolist(),
        "codes": [codes.get(key) or "" for key in geom.keys],
        "area_cycle": area_cycle.tolist(), "cycles": cycles,
        "full_slot": geom.full_slot.tolist(), "castle_mod": castle_mod.tolist(),
    }


def wml_tuple(node) -> tuple:
    """A `tools.replay_extract.WMLNode` as the core's nested
    `(tag, [(key, value)], [children])` tuple."""
    return (node.tag, [(str(k), str(v)) for k, v in node.attrs.items()],
            [wml_tuple(c) for c in node.children])


def wml_node(t):
    """The WMLNode back from the core's tuple form."""
    from tools.replay_extract import WMLNode
    tag, attrs, children = t
    node = WMLNode(tag)
    node.attrs = dict(attrs)
    node.children = [wml_node(c) for c in children]
    return node


def _event_actions(ev) -> list:
    """A Python ScenarioEvent's actions in tuple form, converted once per
    event object (copies share the conversion)."""
    cached = getattr(ev, "_core_actions", None)
    if cached is None:
        cached = [wml_tuple(a) for a in ev.actions]
        setattr(ev, "_core_actions", cached)
    return cached


def unit_fields(u: Unit) -> dict:
    """A unit's dataclass fields and underscore attributes for
    `GameCore.add_unit`."""
    table = getattr(u, "_defense_table", None)
    pick = getattr(u, "_pickadvance", None)
    order = getattr(u, "_trait_order", None)
    feeding = getattr(u, "_feeding_count", None)
    role = getattr(u, "_wml_role", None)
    return {
        "id": u.id, "name": u.name, "name_id": int(u.name_id), "side": int(u.side),
        "is_leader": bool(u.is_leader), "x": int(u.position.x), "y": int(u.position.y),
        "max_hp": int(u.max_hp), "max_moves": int(u.max_moves), "max_exp": int(u.max_exp),
        "cost": int(u.cost), "alignment": int(u.alignment),
        "levelup_names": [str(n) for n in u.levelup_names],
        "current_hp": int(u.current_hp), "current_moves": int(u.current_moves),
        "current_exp": int(u.current_exp), "has_attacked": bool(u.has_attacked),
        "attacks": [(int(a.type_id), int(a.number_strikes), int(a.damage_per_strike),
                     bool(a.is_ranged), [str(s) for s in (a.weapon_specials or ())])
                    for a in u.attacks],
        "resistances": [float(v) for v in u.resistances],
        "defenses": [float(v) for v in u.defenses],
        "movement_costs": [int(v) for v in u.movement_costs],
        "abilities": [str(a) for a in (u.abilities or ())],
        "traits": [str(t) for t in (u.traits or ())],
        "statuses": [str(s) for s in (u.statuses or ())],
        "defense_table": None if table is None else [(str(k), int(v)) for k, v in table.items()],
        "pickadvance": None if pick is None else [str(t) for t in pick],
        "trait_order": None if order is None else [str(t) for t in order],
        "feeding_count": None if feeding is None else int(feeding),
        "wml_role": None if role is None else str(role),
        "object_effects": [wml_tuple(n) for n in (getattr(u, "_object_effects", None) or ())],
        "ai_guardian": bool(getattr(u, "_ai_guardian", False)),
    }


def unit_from_fields(d: dict) -> Unit:
    """The dataclass back from the core's export (`GameCore.unit_export`),
    its record fields set as the underscore attributes they stand for."""
    from wesnoth_ai.classes import Alignment, DamageType
    u = Unit(
        id=d["id"], name=d["name"], name_id=int(d["name_id"]), side=int(d["side"]),
        is_leader=bool(d["is_leader"]), position=Position(x=int(d["x"]), y=int(d["y"])),
        max_hp=int(d["max_hp"]), max_moves=int(d["max_moves"]), max_exp=int(d["max_exp"]),
        cost=int(d["cost"]), alignment=Alignment(int(d["alignment"])),
        levelup_names=list(d["levelup_names"]),
        current_hp=int(d["current_hp"]), current_moves=int(d["current_moves"]),
        current_exp=int(d["current_exp"]), has_attacked=bool(d["has_attacked"]),
        attacks=[Attack(type_id=DamageType(int(t)), number_strikes=int(n), damage_per_strike=int(dm),
                        is_ranged=bool(r), weapon_specials=set(sp))
                 for (t, n, dm, r, sp) in d["attacks"]],
        resistances=list(d["resistances"]), defenses=list(d["defenses"]),
        movement_costs=list(d["movement_costs"]),
        abilities=set(d["abilities"]), traits=set(d["traits"]), statuses=set(d["statuses"]),
    )
    if d["defense_table"] is not None:
        u._defense_table = dict(d["defense_table"])
    for attr, key in (("_pickadvance", "pickadvance"), ("_trait_order", "trait_order")):
        if d[key] is not None:
            setattr(u, attr, list(d[key]))
    if d["feeding_count"] is not None:
        u._feeding_count = int(d["feeding_count"])
    if d["wml_role"] is not None:
        u._wml_role = d["wml_role"]
    if d["object_effects"]:
        u._object_effects = [wml_node(t) for t in d["object_effects"]]
    if d["ai_guardian"]:
        u._ai_guardian = True
    return u


# ---------------------------------------------------------------------
# The state wrapper
# ---------------------------------------------------------------------

@dataclass
class CoreState:
    """A GameCore plus what stays Python: the aliased statics, and how far
    they follow the core's map."""
    core: object
    game_id: str
    statics: Dict[str, object]                  # aliased GlobalInfo stash and map fields
    hexes_holder: object = None                 # the Python hex set (identity keeps the cache)
    _view_cache: Optional[GameState] = None     # the statics as a unit-less GameState
    caches: Dict[tuple, object] = field(default_factory=dict)   # vocab and recruit rows, shared by forks
    map_synced: int = 0                         # the core's map version `statics` reflects
    terrain_synced: int = 0                     # the terrain writes `statics` reflects
    _geometry: Optional[tuple] = None           # (map version, MapGeometry in the core's hex order)

    @classmethod
    def from_state(cls, gs: GameState) -> "CoreState":
        core_cls = game_core_class()
        if core_cls is None:
            raise RuntimeError(f"wesnoth_core.GameCore is not available (phase {_CORE_PHASE} wheel)")
        core = core_cls(map_static(gs), gs.game_id, int(gs.map.size_x), int(gs.map.size_y))
        gi = gs.global_info
        statics: Dict[str, object] = {"hexes": gs.map.hexes, "mask": gs.map.mask, "fog": gs.map.fog}
        for k, v in gi.__dict__.items():
            if (k.startswith("_") and k not in MODELED_GLOBALS and k not in _DROPPED_GLOBALS
                    and k not in SIGHT_RECORDS):
                statics[k] = v
        statics["size_x"], statics["size_y"] = int(gs.map.size_x), int(gs.map.size_y)
        # What the core does not keep and the faction posterior reads: who
        # chose Random, the era's factions and the random faction mode.
        statics["chose_random"] = tuple(bool(getattr(s, "chose_random", False)) for s in gs.sides)
        statics["era_factions"] = tuple(getattr(gs, "era_factions", None) or DEFAULT_ERA_FACTIONS)
        statics["random_faction_mode"] = str(getattr(gs, "random_faction_mode", None) or "Independent")
        cs = cls(core=core, game_id=gs.game_id, statics=statics, hexes_holder=gs.map.hexes)
        for u in gs.map.units:
            cs._add_unit(u)
        cs._load_scalars(gs)
        cs._load_events(gi)
        cs._load_sight_records(gi)
        _log_core_warnings()
        return cs

    def _load_sight_records(self, gi) -> None:
        """A view's sighting records (`SIGHT_RECORDS`) into the core."""
        for side, rows in (getattr(gi, "_sightings", None) or {}).items():
            self.core.set_sightings(int(side), [tuple(r) for r in rows])
        for side, rows in (getattr(gi, "_seen_types", None) or {}).items():
            self.core.set_seen_types(int(side), [(int(s), str(t)) for s, t in rows])
        for side, ids in (getattr(gi, "_sightings_gone", None) or {}).items():
            self.core.set_sightings_gone(int(side), [str(i) for i in ids])

    def _add_unit(self, u: Unit) -> None:
        self.core.add_unit(unit_fields(u))

    def _load_events(self, gi) -> None:
        """The scenario events a Python setup left on `gi`, with their
        latches and variables, into the core."""
        events = getattr(gi, "_scenario_events", None) or []
        wml = getattr(gi, "_wml_variables", None) or {}
        stored = getattr(gi, "_scenario_vars", None) or {}
        if not (events or wml or stored):
            return
        self.core.load_events(
            [(ev.name, bool(ev.first_time_only), _event_actions(ev), ev.scenario_id, bool(ev.fired))
             for ev in events],
            [(str(k), str(v)) for k, v in wml.items()],
            [(str(k), sorted((int(x), int(y)) for x, y in v)) for k, v in stored.items()],
            _strict_wml())

    def setup_scenario(self, scenario_id: str) -> None:
        """`_setup_scenario_events` on the core: the scenario's WML read
        here, its time areas, [side] modifications, events and prestart
        and start run in the core. A scenario without WML is warned about
        once and runs without events."""
        from tools.scenario_events import collect_events
        from wesnoth_ai.rules.scenario_cfg import load_scenario_wml
        root = load_scenario_wml(scenario_id) if scenario_id else None
        if scenario_id and root is None:
            from tools.replay_dataset import _warn_scenario_without_wml
            _warn_scenario_without_wml(scenario_id)
        tup = None
        if root is not None:
            tup = _WML_TUPLES.get(scenario_id)
            if tup is None:
                tup = _WML_TUPLES[scenario_id] = wml_tuple(root)
        self.core.setup_scenario(tup, scenario_id, _strict_wml())
        self.statics["_scenario_events"] = collect_events(root, scenario_id) if root is not None else []
        _log_core_warnings()

    def to_state(self) -> GameState:
        """A GameState view with the core's content; statics by reference."""
        self._sync_map()
        core = self.core
        g = core.globals_export()
        gi = GlobalInfo(current_side=g["current_side"], turn_number=g["turn_number"],
                        time_of_day=g["time_of_day"], village_gold=g["village_gold"],
                        village_upkeep=g["village_upkeep"], base_income=g["base_income"])
        for k, v in self.statics.items():
            if k.startswith("_"):
                setattr(gi, k, v)
        gi._fog = g["fog_on"]
        gi._did_first_init_side = g["did_first_init_side"]
        gi._tod_start_offset = g["tod_start_offset"]
        gi._experience_modifier = g["experience_modifier"]
        gi._next_uid_counter = g["next_uid_counter"]
        gi._rng_request_counter = g["rng_request_counter"]
        if g["advance_uniform"]:
            gi._advance_uniform = True
        if g["advance_salt"]:
            gi._advance_salt = g["advance_salt"]
        gi._advance_counter = g["advance_counter"]
        gi._village_owner = {(x, y): s for (x, y, s) in core.village_owner_export()}
        gi._uncovered_units = set(core.uncovered_export())
        gi._recruit_rejected_hexes = set(core.recruit_rejected_hexes())
        choices, pick, last_events = core.advance_state_export()
        gi._advance_choices = list(choices)
        gi._pickadvance_game = {(side, t): list(lst) for (side, t, lst) in pick}
        gi._last_advance_events = [tuple(e) for e in last_events]
        walk = core.last_move_walk_export()
        if walk is not None:
            gi._last_move_walk = {"ordered": (walk[0], walk[1]), "landed": (walk[2], walk[3]),
                                  "stop_reason": walk[4]}
        flat = core.last_checkup_strikes_export()
        strikes = []
        for k in range(0, len(flat), 4):
            strikes.append({"chance": flat[k], "hits": bool(flat[k + 1]), "damage": flat[k + 2]})
            strikes.append({"dies": bool(flat[k + 3])})
        gi._last_checkup_strikes = strikes or None
        gi._fog_cleared = {side: frozenset(map(tuple, hexes)) for side, hexes in core.fog_cleared_export()}
        delayed, pending, plan_unit_advance, fresh_turn = core.shroud_state_export()
        gi._shroud_delayed = frozenset(delayed)
        gi._pending_vision = tuple(_pending_row(r) for r in pending)
        gi._plan_unit_advance = bool(plan_unit_advance)
        gi._pa_fresh_turn = bool(fresh_turn)
        gi._sightings = {side: tuple(core.sightings_export(side)) for side in _RECORD_SIDES}
        gi._seen_types = {side: tuple(core.seen_types_export(side)) for side in _RECORD_SIDES}
        gi._sightings_gone = {side: tuple(core.sightings_gone_export(side)) for side in _RECORD_SIDES}
        self._events_into(gi)
        units = {unit_from_fields(d) for d in core.units_export()}
        chose_random = self.statics.get("chose_random") or ()
        sides = [SideInfo(player=p, recruits=list(r), current_gold=gold, base_income=b,
                          nb_villages_controlled=v, faction=f,
                          chose_random=bool(chose_random[k]) if k < len(chose_random) else False)
                 for k, (p, r, gold, b, v, f) in enumerate(core.sides_export())]
        m = Map(size_x=int(self.statics["size_x"]), size_y=int(self.statics["size_y"]),
                mask=self.statics["mask"], fog=self.statics["fog"],
                hexes=self.statics["hexes"], units=units)
        return GameState(game_id=self.game_id, map=m, global_info=gi, sides=sides,
                         game_over=bool(g["game_over"]),
                         winner=None if g["winner"] < 0 else int(g["winner"]),
                         era_factions=tuple(self.statics.get("era_factions") or DEFAULT_ERA_FACTIONS),
                         random_faction_mode=str(self.statics.get("random_faction_mode") or "Independent"))

    def _events_into(self, gi) -> None:
        """The event latches and variables of the core on a view: each
        Python event a copy carrying the core's latch."""
        fired, wml, stored = self.core.events_export()
        events = self.statics.get("_scenario_events")
        if events is not None and len(events) == len(fired):
            copies = []
            for ev, f in zip(events, fired):
                c = copy.copy(ev)
                c.fired = bool(f)
                copies.append(c)
            gi._scenario_events = copies
        if wml or getattr(gi, "_wml_variables", None) is not None:
            gi._wml_variables = dict(wml)
        if stored or getattr(gi, "_scenario_vars", None) is not None:
            gi._scenario_vars = {k: set(map(tuple, v)) for k, v in stored}

    def _sync_map(self) -> None:
        """Bring the core's map changes (terrain writes, time areas) into
        `statics`: new containers, never the aliased ones."""
        version = self.core.map_version
        if version == self.map_synced:
            return
        terrain_log = self.core.terrain_log()
        writes = terrain_log[self.terrain_synced:]
        st = self.statics
        if writes:
            from tools.pathfind_sim import next_terrain_epoch
            from tools.scenario_events import terrain_writes_applied
            hexes, codes, raw = terrain_writes_applied(
                st["hexes"], st.get("_terrain_codes"), st.get("_raw_map_data", "") or "", writes)
            st["hexes"] = hexes
            if codes is not None:
                st["_terrain_codes"] = codes
                st["_terrain_epoch"] = next_terrain_epoch()
            if raw:
                st["_raw_map_data"] = raw
            self.hexes_holder = hexes
            self._view_cache = None
        st["_time_areas"] = {(x, y): list(c) for x, y, c in self.core.time_areas_export()}
        self.terrain_synced = len(terrain_log)
        self.map_synced = version

    def fork(self) -> "CoreState":
        """A search fork: the core cloned, the statics copied as
        `GlobalInfo.__deepcopy__` copies them (dicts, sets and lists
        shallow; the terrain codes and the hex set aliased)."""
        return CoreState(core=self.core.fork(), game_id=self.game_id, statics=_fork_statics(self.statics),
                         hexes_holder=self.hexes_holder, caches=self.caches, map_synced=self.map_synced,
                         terrain_synced=self.terrain_synced, _geometry=self._geometry)

    def state_key(self) -> int:
        return int(self.core.state_key())

    # ---- commands ----------------------------------------------------

    def apply_command(self, cmd: list) -> str:
        """One replay or simulator command (`_apply_command`'s
        vocabulary). Returns "rust" when the core applied it, "python" for
        the recall bookkeeping and the kinds the applier ignores."""
        path = self._apply(cmd)
        _log_core_warnings()
        return path

    def _apply(self, cmd: list) -> str:
        kind = cmd[0] if cmd else ""
        if kind == "init_side":
            self.core.apply_init_side(int(cmd[1]))
            self._emit_heal_events()
            return "rust"
        if kind == "end_turn":
            self.core.apply_end_turn()
            return "rust"
        if kind == "move":
            from_side = int(cmd[3]) if len(cmd) > 3 else 0
            order = cmd[4] if len(cmd) > 4 and isinstance(cmd[4], dict) else {}
            nxt = order.get("next")
            self.core.apply_move([int(v) for v in cmd[1]], [int(v) for v in cmd[2]], from_side,
                                 next=None if nxt is None else (int(nxt[0]), int(nxt[1])))
            return "rust"
        if kind == "attack":
            self._apply_attack(cmd)
            return "rust"
        if kind == "recruit":
            seed = cmd[4] if len(cmd) > 4 else ""
            self.core.apply_recruit(str(cmd[1]), int(cmd[2]), int(cmd[3]), str(seed or ""))
            return "rust"
        if kind == "auto_shroud":
            self.core.apply_auto_shroud(bool(cmd[1]))
            return "rust"
        if kind == "update_shroud":
            self.core.apply_update_shroud()
            return "rust"
        if kind == "menu_item":
            self.core.apply_menu_item(str(cmd[1]) if len(cmd) > 1 else "")
            return "rust"
        if kind == "pickadvance":
            self.core.apply_pickadvance(int(cmd[1]), int(cmd[2]), str(cmd[3] or ""), str(cmd[4] or ""),
                                        bool(cmd[5]), bool(cmd[6]))
            return "rust"
        if kind == "recall":
            self._recall(cmd)
        return "python"

    def _apply_attack(self, cmd: list) -> None:
        """The attack on the core, which finishes its fed kills,
        advancements and plague corpses; the fight goes to the
        engagement telemetry."""
        from tools.engagement_stats import emit_event
        from wesnoth_ai.combat import seed_int_of
        ax, ay, dx, dy, a_weapon = (int(v) for v in cmd[1:6])
        d_weapon = int(cmd[6]) if len(cmd) > 6 else -1
        seed_hex = cmd[7] if len(cmd) > 7 else ""
        choices = [int(c) if isinstance(c, int) else -1 for c in (cmd[8] if len(cmd) > 8 else [])]
        out = self.core.apply_attack(ax, ay, dx, dy, a_weapon, d_weapon, seed_int_of(seed_hex),
                                     bool(seed_hex), choices)
        if out is None:
            return
        emit_event("combat", a_side=out["att_side"], d_side=out["dfd_side"],
                   dmg_to_defender=out["dmg_to_defender"], dmg_to_attacker=out["dmg_to_attacker"],
                   defender_died=not out["dfd_alive"], attacker_died=not out["att_alive"],
                   attacker_name=out["att_name"], defender_name=out["dfd_name"],
                   attacker_cost=out["att_cost"], defender_cost=out["dfd_cost"])

    def _emit_heal_events(self) -> None:
        """The init_side's heal and poison telemetry (no-op without a sink)."""
        from tools.engagement_stats import emit_event
        for kind, side, a, b, c in self.core.heal_events():
            if kind == 0:
                emit_event("heal", side=side, village=a, ability=b, rest=c)
            elif kind == 1:
                emit_event("poison", side=side, cured=True, damage=0)
            else:
                emit_event("poison", side=side, cured=False, damage=a)

    def _recall(self, cmd: list) -> None:
        """A recall in a PvP replay: logged and noted for
        tools/flag_replays_with_recalls.py, the state unchanged (the
        Python applier's recall branch)."""
        unit_id = cmd[1] if len(cmd) > 1 else "<unknown>"
        tx = cmd[2] if len(cmd) > 2 else -1
        ty = cmd[3] if len(cmd) > 3 else -1
        side, turn = int(self.core.current_side), int(self.core.turn_number)
        log.error(f"recall in PvP replay {self.game_id!r} (turn={turn}, side={side}, "
                  f"unit_id={unit_id!r}, hex=({tx},{ty})). Run tools/flag_replays_with_recalls.py "
                  f"to surface for inspection / removal from the supervised corpus.")
        recall_log = list(self.statics.get("_recall_log") or [])
        recall_log.append({"turn": turn, "side": side, "unit_id": unit_id, "x": tx, "y": ty})
        self.statics["_recall_log"] = recall_log
        self.statics["_has_recall"] = True

    # ---- the observation and the encoding over the core ----------------

    def geometry(self):
        """The map's geometry in the core's hex order (the order of its
        map-space arrays), rebuilt when an event changed the terrain."""
        version = self.core.map_version
        if self._geometry is not None and self._geometry[0] == version:
            return self._geometry[1]
        self._sync_map()
        from wesnoth_ai.observe import MapGeometry
        hx, hy, nbrs, castle_or_keep, keep, village, full_slot = self.core.geometry_export()
        keys = list(zip(hx.tolist(), hy.tolist()))
        geom = MapGeometry(self.statics["hexes"], keys, {p: i for i, p in enumerate(keys)}, hx, hy, nbrs,
                           castle_or_keep, keep, village, full_slot)
        self._geometry = (version, geom)
        return geom

    def observe(self, side: int, reach: bool = False):
        """`observe.observe(state, side, reach)` over the core: the same
        Observation record, without Unit references (the unit arrays
        follow the core's unit order)."""
        return _observation_from_dict(self.core.observe(int(side), bool(reach)), self.geometry())

    def encode_raw(self, *, type_to_id: Dict[str, int], faction_to_id: Dict[str, int],
                   relevant_set: bool = False, fog_hides_enemy_villages: bool = False,
                   terrain_multi_hot: bool = False, observation_parity: bool = False,
                   relevant_set_version: int = 1):
        """`encoder.encode_raw` over the core for the side to move: the
        same RawEncoded, byte for byte (tests/test_game_core.py). With
        `observation_parity` the parity observation (encoder.py's layout
        "The parity observation"), which only the core builds, with the
        faction posterior and the sighting stream. `relevant_set_version`
        2 widens the relevant set (the parity recipe's; needs
        `relevant_set`)."""
        from wesnoth_ai import encoder as enc
        core = self.core
        side = int(core.current_side)
        sides = core.sides_export()
        us = side - 1
        them = opponent_of(side) - 1
        our_fac = sides[us][5] if 0 <= us < len(sides) else ""
        them_fac = sides[them][5] if 0 <= them < len(sides) else ""
        own_recruits = list(sides[us][1]) if 0 <= us < len(sides) else []
        r_ids, r_stats = self._recruit_rows(own_recruits, type_to_id)
        parity_norms = None
        if observation_parity:
            _check_parity_layout()
            parity_norms = (enc.WEAPON_DAMAGE_NORM, enc.WEAPON_STRIKES_NORM, enc.LAWFUL_BONUS_NORM,
                            enc.LEADERSHIP_NORM, enc.VILLAGE_GOLD_NORM, enc.VILLAGE_SUPPORT_NORM)
            r_stats = []           # the core reads the recruits' types itself
        d = core.encode_streams(
            side, bool(relevant_set), self._type_vocab(type_to_id), r_ids, r_stats,
            bool(fog_hides_enemy_villages),
            (enc.HP_NORM, enc.MOVES_NORM, enc.EXP_NORM, enc.COST_NORM, enc.GOLD_NORM,
             enc.INCOME_NORM, enc.VILLAGES_NORM, enc.TURN_NORM),
            enc.MAX_MAP_SIZE - 1, enc.NUM_ALIGNMENTS, bool(terrain_multi_hot),
            observation_parity=bool(observation_parity), parity_norms=parity_norms,
            relevant_set_version=int(relevant_set_version))
        _require_widths(d, bool(observation_parity))
        static = enc._static_hex_arrays(self._view())
        if relevant_set:
            hex_positions = [static.positions[t] for t in d["full_slots"].tolist()]
        else:
            hex_positions = static.positions
        raw = enc.RawEncoded(
            hex_subset=bool(relevant_set), hex_positions=hex_positions,
            hex_xs=d["hex_xs"], hex_ys=d["hex_ys"], hex_terrain_ids=d["hex_terrain_ids"],
            hex_modifier_flags=d["hex_modifier_flags"], hex_dynamic_flags=d["hex_dynamic_flags"],
            unit_positions=[Position(x=x, y=y) for x, y in zip(d["unit_raw_xs"], d["unit_raw_ys"])],
            unit_ids=list(d["unit_ids"]), unit_is_ours=d["unit_is_ours"], unit_type_ids=d["unit_type_ids"],
            unit_side_ids=d["unit_side_ids"], unit_xs=d["unit_xs"], unit_ys=d["unit_ys"],
            unit_feats=d["unit_feats"], recruit_types=own_recruits, recruit_is_ours=d["recruit_is_ours"],
            recruit_type_ids=d["recruit_type_ids"], recruit_side_ids=d["recruit_side_ids"],
            recruit_xs=d["recruit_xs"], recruit_ys=d["recruit_ys"], recruit_feats=d["recruit_feats"],
            global_feats=d["global_feats"],
            our_faction_id=enc._lookup_id(our_fac, faction_to_id, enc.MAX_FACTIONS),
            their_faction_id=enc._lookup_id(them_fac, faction_to_id, enc.MAX_FACTIONS),
            material=float(d["material"]),
            observation=_observation_from_dict(d["observation"], self.geometry()))
        if observation_parity:
            raw.their_faction_probs = self._faction_probs(side, our_fac, them_fac, faction_to_id)
            raw.sight_type_ids = d["sight_type_ids"]
            raw.sight_xs = d["sight_xs"]
            raw.sight_ys = d["sight_ys"]
            raw.sight_feats = d["sight_feats"]
        return raw

    def _faction_probs(self, side: int, our_fac: str, them_fac: str, faction_to_id: Dict[str, int]):
        """The posterior over the opponent's faction (faction_posterior)."""
        from wesnoth_ai.faction_posterior import faction_posterior
        them = opponent_of(side)
        chose_random = self.statics.get("chose_random") or ()
        return faction_posterior(
            them_fac, bool(chose_random[them - 1]) if them - 1 < len(chose_random) else False,
            self.statics.get("era_factions") or DEFAULT_ERA_FACTIONS,
            self.core.seen_types(side, them), faction_to_id, own_faction=our_fac,
            random_faction_mode=str(self.statics.get("random_faction_mode") or "Independent"))

    def _type_vocab(self, type_to_id: Dict[str, int]) -> List[int]:
        """The vocab row of every registered type (`encoder.type_row`: the
        overflow row for a name the vocabulary lacks, counted); cached per
        vocab and type count."""
        from wesnoth_ai.encoder import type_row
        names = tuple(self.core.type_names())
        key = ("vocab", id(type_to_id), len(type_to_id))
        hit = self.caches.get(key)
        if hit is None or hit[0] != names:
            hit = (names, [type_row(n, type_to_id) for n in names])
            self.caches[key] = hit
        return hit[1]

    def _recruit_rows(self, own_recruits: List[str], type_to_id: Dict[str, int]):
        from wesnoth_ai.encoder import _recruit_rows
        key = ("recruits", tuple(own_recruits), id(type_to_id), len(type_to_id))
        hit = self.caches.get(key)
        if hit is None:
            ids, stats = _recruit_rows(own_recruits, type_to_id)
            hit = (list(ids), list(stats))
            self.caches[key] = hit
        return hit

    def _view(self) -> GameState:
        """The statics as a GameState without units: the hex set, terrain
        codes and time areas the encoder's static arrays read."""
        self._sync_map()
        if self._view_cache is None:
            gi = GlobalInfo(current_side=0, turn_number=0, time_of_day="", village_gold=0,
                            village_upkeep=0, base_income=0)
            for k, v in self.statics.items():
                if k.startswith("_"):
                    setattr(gi, k, v)
            m = Map(size_x=int(self.statics["size_x"]), size_y=int(self.statics["size_y"]),
                    mask=self.statics["mask"], fog=self.statics["fog"],
                    hexes=self.statics["hexes"], units=set())
            self._view_cache = GameState(game_id=self.game_id, map=m, global_info=gi, sides=[],
                                         game_over=False, winner=None)
        return self._view_cache

    def _load_scalars(self, gs: GameState) -> None:
        core, gi = self.core, gs.global_info
        core.set_sides([(s.player, list(s.recruits), int(s.current_gold), int(s.base_income),
                         int(s.nb_villages_controlled), s.faction or "") for s in gs.sides])
        core.set_globals({
            "current_side": int(gi.current_side), "turn_number": int(gi.turn_number),
            "time_of_day": str(gi.time_of_day), "village_gold": int(gi.village_gold),
            "village_upkeep": int(gi.village_upkeep), "base_income": int(gi.base_income),
            "fog_on": bool(getattr(gi, "_fog", True)),
            "did_first_init_side": bool(getattr(gi, "_did_first_init_side", False)),
            "tod_start_offset": int(getattr(gi, "_tod_start_offset", 0) or 0),
            "experience_modifier": int(getattr(gi, "_experience_modifier", 100) or 100),
            "next_uid_counter": int(getattr(gi, "_next_uid_counter", 1) or 1),
            "rng_request_counter": int(getattr(gi, "_rng_request_counter", 0) or 0),
            "advance_uniform": bool(getattr(gi, "_advance_uniform", False)),
            "advance_salt": str(getattr(gi, "_advance_salt", "") or ""),
            "advance_counter": int(getattr(gi, "_advance_counter", 0) or 0),
            "game_over": bool(gs.game_over), "winner": -1 if gs.winner is None else int(gs.winner),
        })
        owner = getattr(gi, "_village_owner", None) or {}
        core.set_village_owner([(int(x), int(y), int(s)) for (x, y), s in owner.items() if s])
        core.set_uncovered([str(i) for i in (getattr(gi, "_uncovered_units", None) or ())])
        core.set_recruit_rejected([tuple(p) for p in (getattr(gi, "_recruit_rejected_hexes", None) or ())])
        pick = getattr(gi, "_pickadvance_game", None) or {}
        core.set_advance_state(
            [int(c) if isinstance(c, int) else -1 for c in (getattr(gi, "_advance_choices", None) or [])],
            [(int(side), str(t), [str(x) for x in lst]) for (side, t), lst in pick.items()],
            [(int(a), int(b)) for a, b in (getattr(gi, "_last_advance_events", None) or [])])
        walk = getattr(gi, "_last_move_walk", None)
        core.set_last_move_walk((int(walk["ordered"][0]), int(walk["ordered"][1]),
                                 int(walk["landed"][0]), int(walk["landed"][1]),
                                 str(walk["stop_reason"])) if walk else None)
        strikes = getattr(gi, "_last_checkup_strikes", None) or []
        flat: List[int] = []
        for k in range(0, len(strikes) - 1, 2):
            s, d = strikes[k], strikes[k + 1]
            flat += [int(s["chance"]), int(bool(s["hits"])), int(s["damage"]), int(bool(d["dies"]))]
        core.set_last_checkup_strikes(flat)
        core.set_fog_cleared(_fog_cleared_rows(gi))
        core.set_shroud_state(sorted(int(s) for s in (getattr(gi, "_shroud_delayed", None) or ())),
                              [_pending_row(r) for r in (getattr(gi, "_pending_vision", None) or ())],
                              bool(getattr(gi, "_plan_unit_advance", False)),
                              bool(getattr(gi, "_pa_fresh_turn", False)))


def _log_core_warnings() -> None:
    """The warnings the extension recorded (an unmodelled [effect]
    apply_to or event action), each once per process."""
    import wesnoth_core
    for text in wesnoth_core.drain_warnings():
        log.warning("%s", text)


def _pending_row(row) -> tuple:
    """A pending vision entry (`wesnoth_ai.delayed_shroud`) in plain values:
    (side, ((x, y), ...), unit id, vision points, slowed)."""
    side, route, unit_id, vision, slowed = row
    return (int(side), tuple((int(x), int(y)) for x, y in route), str(unit_id), int(vision), bool(slowed))


def _fog_cleared_rows(gi) -> List[Tuple[int, List[Tuple[int, int]]]]:
    """`global_info._fog_cleared` as the core's (side, [(x, y)]) rows."""
    cleared = getattr(gi, "_fog_cleared", None) or {}
    return [(int(side), [(int(x), int(y)) for x, y in hexes]) for side, hexes in cleared.items()]


def _fork_statics(statics: Dict[str, object]) -> Dict[str, object]:
    """The per-fork copy of the aliased stash (the rules of
    `classes.GlobalInfo.__deepcopy__`)."""
    out: Dict[str, object] = {}
    for k, v in statics.items():
        if k == "_terrain_codes" or not k.startswith("_"):
            out[k] = v
        elif isinstance(v, dict):
            out[k] = dict(v)
        elif isinstance(v, set):
            out[k] = set(v)
        elif isinstance(v, list):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _require_global_width(global_feats) -> None:
    """Refuse a core whose encoder emits a different number of global
    features than this one (a wheel built from an older or newer
    rust/wesnoth_core); the failure would otherwise surface as a shape
    error inside the first forward pass, nowhere near its cause. The
    Python kernel path has the same guard in
    `encoder._rust_encode_kernel`."""
    from wesnoth_ai import encoder as enc
    _require_width("global_feats", global_feats, enc.GLOBAL_FEAT_DIM)


def _require_widths(d: dict, parity: bool) -> None:
    """`_require_global_width` for an encoding, and under the parity flag
    every row the parity observation widens."""
    from wesnoth_ai import encoder as enc
    if not parity:
        _require_global_width(d["global_feats"])
        return
    for key, width in (("global_feats", enc.GLOBAL_FEAT_DIM_PARITY), ("unit_feats", enc.UNIT_FEAT_DIM_PARITY),
                       ("recruit_feats", enc.UNIT_FEAT_DIM_PARITY),
                       ("hex_dynamic_flags", enc.NUM_HEX_DYNAMIC_FLAGS_PARITY),
                       ("sight_feats", enc.SIGHT_FEAT_DIM)):
        _require_width(key, d[key], width)


def _require_width(key: str, array, width: int) -> None:
    got = int(getattr(array, "shape", (len(array),))[-1])
    if got != width:
        try:
            import wesnoth_core
        except ImportError:
            wesnoth_core = None
        raise RuntimeError(
            f"wesnoth_core.GameCore encoded {got} {key} columns where the encoder expects "
            f"{width}: the installed wheel is phase {getattr(wesnoth_core, '__phase__', '?')}. "
            f"Rebuild the wheel from rust/wesnoth_core.")


_PARITY_LAYOUT_CHECKED = False


def _check_parity_layout() -> None:
    """Once per process: the core's parity layout (vocabularies, widths,
    terrain classes) is the encoder's, or RuntimeError."""
    global _PARITY_LAYOUT_CHECKED
    if _PARITY_LAYOUT_CHECKED:
        return
    import wesnoth_core
    from wesnoth_ai import encoder as enc
    got = wesnoth_core.parity_layout()
    want = {
        "damage_types": list(enc.PARITY_DAMAGE_TYPES), "specials": list(enc.PARITY_SPECIALS),
        "traits": list(enc.PARITY_TRAITS), "abilities": list(enc.PARITY_ABILITIES),
        "weapon_slots": enc.PARITY_WEAPON_SLOTS, "weapon_cols": enc.PARITY_WEAPON_COLS,
        "unit_extra": enc.PARITY_UNIT_EXTRA,
        "hex_extra": enc.NUM_HEX_DYNAMIC_FLAGS_PARITY - enc.NUM_HEX_DYNAMIC_FLAGS,
        "global_extra": enc.GLOBAL_FEAT_DIM_PARITY - enc.GLOBAL_FEAT_DIM,
        "sight_feat_dim": enc.SIGHT_FEAT_DIM, "terrain_classes": enc.NUM_TERRAINS_PARITY,
    }
    off = {k: (got.get(k), v) for k, v in want.items() if got.get(k) != v}
    if off:
        raise RuntimeError(f"the core's parity layout differs from the encoder's (core, encoder): {off}")
    _PARITY_LAYOUT_CHECKED = True


def _observation_from_dict(d: dict, geometry):
    """`observe.Observation` from the core's dict of arrays."""
    from wesnoth_ai.observe import Observation
    return Observation(int(d["side"]), bool(d["fog_on"]), geometry, list(d["unit_ids"]), d["unit_hex"],
                       d["seen"], d["visible"], d["zoc"], d["enemy"], d["ally"], d["occupied"], d["inert"],
                       d["recruit_row"], d["network"], bool(d["leader_on_keep"]), d.get("acting"),
                       d.get("unit_can_move"), d.get("unit_can_attack"), d.get("landable"),
                       d.get("relevant"), d.get("tok_of_hex"))


__all__ = ["CoreState", "map_static", "unit_fields", "unit_from_fields", "wml_tuple", "wml_node",
           "game_core_class", "core_enabled", "load_databases", "bind_view", "core_of", "snapshot_view",
           "unit_db_fallbacks", "MODELED_GLOBALS", "UNIT_STASH_KEYS", "SIGHT_RECORDS"]
