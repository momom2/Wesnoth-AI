"""The Rust-owned game state (docs/rust_port_plan.md phase 4): the
adapter between `wesnoth_ai.classes.GameState` and `wesnoth_core.GameCore`.

`CoreState.from_state(gs)` builds a core from a Python state: the map's
static arrays once per hex set (`map_static`), the unit types it needs
(`type_fields`), the movement classes (the pathfinder's cost arrays
and the terrain resolver's defense percentages per hex, per unit type
and slowed status), then the units, sides, globals and the stash the
simulator keeps on `global_info`. `to_state()` rebuilds a GameState
whose modeled content equals the original (tests/test_game_core.py);
the hex set, the terrain codes, the time areas, the scenario events
and every stash key the core does not model are shared by reference,
as `GlobalInfo.__deepcopy__` shares them across forks. Per-unit stash
attributes (`_defense_table`, `_pickadvance`, `_trait_order`,
`_feeding_count`) live in `unit_stash`, replaced never mutated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np

from wesnoth_ai.classes import (Attack, GameState, GlobalInfo, Map, Position, SideInfo,
                                TerrainModifiers, Unit)

_KERNEL_CHECKED = False
_GAME_CORE = None

# The GlobalInfo stash the core models; every other underscore attribute
# is shared by reference through `CoreState.statics`.
MODELED_GLOBALS = (
    "_fog", "_village_owner", "_uncovered_units", "_recruit_rejected_hexes",
    "_move_rejected_hexes", "_tod_start_offset", "_experience_modifier",
    "_next_uid_counter", "_rng_request_counter", "_advance_choices", "_pickadvance_game",
    "_did_first_init_side", "_last_move_walk", "_last_checkup_strikes", "_last_advance_events",
    "_advance_uniform", "_advance_salt", "_advance_counter",
)
# Per-unit stash keys the state comparison checks (every underscore
# attribute of a unit travels beside the core, shared across forks).
UNIT_STASH_KEYS = ("_defense_table", "_pickadvance", "_trait_order", "_feeding_count", "_wml_role")
_DROPPED_GLOBALS = ("_hex_lookup_cache_id", "_hex_lookup_by_xy", "_hex_lookup_by_wml")


def game_core_class():
    """`wesnoth_core.GameCore`, or None (wheel absent or older)."""
    global _KERNEL_CHECKED, _GAME_CORE
    if not _KERNEL_CHECKED:
        _KERNEL_CHECKED = True
        try:
            import wesnoth_core
        except ImportError:
            wesnoth_core = None
        if wesnoth_core is not None and getattr(wesnoth_core, "__phase__", 0) >= 7:
            _GAME_CORE = wesnoth_core.GameCore
    return _GAME_CORE


# ---------------------------------------------------------------------
# Static tables
# ---------------------------------------------------------------------

def _light_params(code: str) -> Tuple[int, int, int, bool]:
    """(light, max_light, min_light, any) of a terrain code, the
    composite rule of `terrain_resolver.terrain_light_bonus`."""
    from tools.terrain_resolver import load_terrain_db
    db = load_terrain_db()
    base_str, _, overlay = code.partition("^")
    b = db.get(base_str) or {}
    o = db.get("^" + overlay) if overlay else None
    light = int(b.get("light", 0) or 0)
    max_l = int(b.get("max_light", 0) or 0)
    min_l = int(b.get("min_light", 0) or 0)
    if o:
        light += int(o.get("light", 0) or 0)
        max_l = max(max_l, int(o.get("max_light", 0) or 0))
        min_l = min(min_l, int(o.get("min_light", 0) or 0))
    return light, max_l, min_l, not (light == 0 and max_l == 0 and min_l == 0)


def map_static(gs: GameState) -> dict:
    """The core's static map arrays from the state: geometry from
    `wesnoth_ai.observe.map_geometry`, terrain facts from the terrain
    codes and the time areas the scenario set up."""
    from tools.terrain_resolver import hides_cover, strip_start_position, terrain_heals
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
    for i, key in enumerate(geom.keys):
        h = by_pos[key]
        village_mod[i] = 1 if TerrainModifiers.VILLAGE in h.modifiers else 0
        castle_mod[i] = 1 if TerrainModifiers.CASTLE in h.modifiers else 0
        terrain_type_id[i] = _first_terrain_id(h.terrain_types)
    heal = np.zeros(H, dtype=np.int64)
    light_mod = np.zeros(H, dtype=np.int64)
    light_max = np.zeros(H, dtype=np.int64)
    light_min = np.zeros(H, dtype=np.int64)
    has_light = np.zeros(H, dtype=np.uint8)
    is_forest = np.zeros(H, dtype=np.uint8)
    is_village_key = np.zeros(H, dtype=np.uint8)
    is_deep_water = np.zeros(H, dtype=np.uint8)
    area_cycle = np.full(H, -1, dtype=np.int64)
    cycles: List[List[int]] = []
    cycle_index: Dict[tuple, int] = {}
    for i, (x, y) in enumerate(geom.keys):
        raw = codes.get((x, y))
        code = strip_start_position(raw)
        if code:
            heal[i] = terrain_heals(code)
            lm, lx, ln, any_light = _light_params(code)
            light_mod[i], light_max[i], light_min[i], has_light[i] = lm, lx, ln, int(any_light)
        # The hide-ability cover flags the core's `hide_cover_active`
        # reads are the ENGINE's terrain filters, not defense keys
        # (see `terrain_resolver.hides_cover`).
        is_forest[i] = hides_cover(raw or "", "ambush")
        is_village_key[i] = hides_cover(raw or "", "concealment")
        is_deep_water[i] = hides_cover(raw or "", "submerge")
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
        "terrain_type_id": terrain_type_id.tolist(), "heal": heal.tolist(),
        "light_mod": light_mod.tolist(), "light_max": light_max.tolist(),
        "light_min": light_min.tolist(), "has_light": has_light.tolist(),
        "area_cycle": area_cycle.tolist(), "cycles": cycles,
        "is_forest": is_forest.tolist(), "is_village_key": is_village_key.tolist(),
        "is_deep_water": is_deep_water.tolist(), "full_slot": geom.full_slot.tolist(),
        "castle_mod": castle_mod.tolist(),
    }


def type_fields(name: str) -> dict:
    """A unit type's fields for `GameCore.register_type`."""
    from tools.replay_dataset import _stats_for
    from wesnoth_ai import combat as cb
    stats = _stats_for(name)
    res = stats.get("resistance", {})
    attacks = []
    for a in stats.get("attacks", []):
        attacks.append((str(a.get("type", "blade")), a.get("range") == "ranged",
                        [str(s) for s in a.get("specials", [])],
                        int(a.get("accuracy", 0) or 0), int(a.get("parry", 0) or 0)))
    return {
        "name": name,
        "level": int(stats.get("level", 1)),
        "alignment": int(cb.alignment_from_str(stats.get("alignment", "neutral"))),
        "resist": [int(res.get(dt, 100)) for dt in cb.DAMAGE_TYPES],
        "abilities": [str(a) for a in stats.get("abilities", [])],
        "attacks": attacks,
        "cost": int(stats.get("cost", 14)),
        "race": str(stats.get("race", "") or ""),
        "undead_variation": str(stats.get("undead_variation", "") or ""),
        "advances_to": [str(t) for t in stats.get("advances_to", [])],
    }


def unit_fields(u: Unit) -> dict:
    """A unit's dataclass fields for `GameCore.add_unit`."""
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
    }


def unit_from_fields(d: dict, stash: Optional[dict]) -> Unit:
    """The dataclass back from the core's export (`GameCore.unit_export`)."""
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
    for k, v in (stash or {}).items():
        setattr(u, k, v)
    return u


# ---------------------------------------------------------------------
# The state wrapper
# ---------------------------------------------------------------------

@dataclass
class CoreState:
    """A GameCore plus what stays Python: the aliased statics, the
    per-unit stash and the movement-class registry keys."""
    core: object
    game_id: str
    statics: Dict[str, object]                  # aliased GlobalInfo stash and map fields
    unit_stash: Dict[str, dict] = field(default_factory=dict)
    class_ids: Dict[tuple, int] = field(default_factory=dict)   # (name, slowed, defense id) -> class
    hexes_holder: object = None                 # the Python hex set (identity keeps the cache)
    _view_cache: Optional[GameState] = None     # the statics as a unit-less GameState
    caches: Dict[tuple, object] = field(default_factory=dict)   # vocab and recruit rows, shared by forks

    @classmethod
    def from_state(cls, gs: GameState) -> "CoreState":
        core_cls = game_core_class()
        if core_cls is None:
            raise RuntimeError("wesnoth_core.GameCore is not available (phase 7 wheel)")
        core = core_cls(map_static(gs), gs.game_id, int(gs.map.size_x), int(gs.map.size_y))
        gi = gs.global_info
        statics: Dict[str, object] = {"hexes": gs.map.hexes, "mask": gs.map.mask, "fog": gs.map.fog}
        for k, v in gi.__dict__.items():
            if k.startswith("_") and k not in MODELED_GLOBALS and k not in _DROPPED_GLOBALS:
                statics[k] = v
        statics["size_x"], statics["size_y"] = int(gs.map.size_x), int(gs.map.size_y)
        cs = cls(core=core, game_id=gs.game_id, statics=statics, hexes_holder=gs.map.hexes)
        for u in gs.map.units:
            cs._add_unit(u, gs)
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
        core.set_rejected([tuple(p) for p in (getattr(gi, "_recruit_rejected_hexes", None) or ())],
                          [tuple(p) for p in (getattr(gi, "_move_rejected_hexes", None) or ())])
        pick = getattr(gi, "_pickadvance_game", None) or {}
        core.set_advance_state(
            [int(c) if isinstance(c, int) else -1 for c in (getattr(gi, "_advance_choices", None) or [])],
            [(int(side), str(t), [str(x) for x in lst]) for (side, t), lst in pick.items()],
            [(int(a), int(b)) for a, b in (getattr(gi, "_last_advance_events", None) or [])])
        walk = getattr(gi, "_last_move_walk", None)
        if walk:
            core.set_last_move_walk((int(walk["ordered"][0]), int(walk["ordered"][1]),
                                     int(walk["landed"][0]), int(walk["landed"][1]),
                                     str(walk["stop_reason"])))
        strikes = getattr(gi, "_last_checkup_strikes", None) or []
        flat: List[int] = []
        for k in range(0, len(strikes) - 1, 2):
            s, d = strikes[k], strikes[k + 1]
            flat += [int(s["chance"]), int(bool(s["hits"])), int(s["damage"]), int(bool(d["dies"]))]
        core.set_last_checkup_strikes(flat)
        return cs

    def _register_type(self, name: str) -> None:
        if self.core.type_index_of(name) < 0:
            self.core.register_type(type_fields(name))

    def _class_id(self, u: Unit, gs: GameState, slowed: bool) -> int:
        """The movement class of a unit: the pathfinder's arrays and
        the resolver's defense per hex, registered once per (type,
        slowed, defense table)."""
        from tools.pathfind_sim import _terrain_arrays_for
        from tools.replay_dataset import _rebuild_unit, _stats_for, _terrain_def_pct
        from wesnoth_ai.observe import map_geometry
        def_table = getattr(u, "_defense_table", None) or _stats_for(u.name).get("defense", {})
        # Keyed on the table's CONTENT, not its address: a freed
        # table's id can be recycled by a different table, which would
        # hand a unit another unit's defense percentages (the same
        # hazard `encoder._static_hex_arrays` and
        # `pathfind_sim._terrain_arrays_for` guard against). Content
        # keying also collapses every recruit of a type onto one
        # registered class -- each recruit builds a fresh defense dict,
        # and the class registry is shared by every fork and never
        # freed, so address keying grew it without bound.
        key = (u.name, slowed, hash(frozenset(def_table.items())))
        hit = self.class_ids.get(key)
        if hit is not None:
            return hit
        probe = u
        if slowed != ("slowed" in (u.statuses or set())):
            st = set(u.statuses or set())
            (st.add if slowed else st.discard)("slowed")
            probe = _rebuild_unit(u, statuses=st)
        _pos_to_idx, positions, _nbrs, mcost, dsub = _terrain_arrays_for(probe, gs)
        geom = map_geometry(gs)
        if list(positions) != list(geom.keys):
            raise ValueError("pathfinder map and geometry disagree")
        defense = [int(_terrain_def_pct(gs, x, y, def_table)) for (x, y) in geom.keys]
        cid = self.core.register_class([int(c) for c in mcost], [int(c) for c in dsub], defense)
        self.class_ids[key] = cid
        return cid

    def _add_unit(self, u: Unit, gs: GameState) -> None:
        self._register_type(u.name)
        fields = unit_fields(u)
        fields["class_id"] = self._class_id(u, gs, False)
        fields["class_slowed_id"] = self._class_id(u, gs, True)
        self.core.add_unit(fields)
        stash = {k: v for k, v in u.__dict__.items() if k.startswith("_")}
        if stash:
            self.unit_stash[u.id] = stash

    def to_state(self) -> GameState:
        """A GameState with the core's content; statics by reference."""
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
        rec, mov = core.rejected_export()
        gi._recruit_rejected_hexes = set(rec)
        gi._move_rejected_hexes = set(mov)
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
        units = {unit_from_fields(d, self.unit_stash.get(d["id"])) for d in core.units_export()}
        sides = [SideInfo(player=p, recruits=list(r), current_gold=gold, base_income=b,
                          nb_villages_controlled=v, faction=f)
                 for (p, r, gold, b, v, f) in core.sides_export()]
        m = Map(size_x=int(self.statics["size_x"]), size_y=int(self.statics["size_y"]),
                mask=self.statics["mask"], fog=self.statics["fog"],
                hexes=self.statics["hexes"], units=units)
        return GameState(game_id=self.game_id, map=m, global_info=gi, sides=sides,
                         game_over=bool(g["game_over"]),
                         winner=None if g["winner"] < 0 else int(g["winner"]))

    def fork(self) -> "CoreState":
        """A search fork: the core cloned, the statics copied as
        `GlobalInfo.__deepcopy__` copies them (the unfired scenario
        events per fork, since their `fired` latch is mutable state;
        dicts, sets and lists shallow; the terrain codes and the
        hex set aliased)."""
        return CoreState(core=self.core.fork(), game_id=self.game_id, statics=_fork_statics(self.statics),
                         unit_stash=dict(self.unit_stash), class_ids=self.class_ids,
                         hexes_holder=self.hexes_holder, caches=self.caches)

    def state_key(self) -> int:
        return int(self.core.state_key())

    # ---- commands ----------------------------------------------------

    RUST_KINDS = ("init_side", "end_turn", "move", "attack", "recruit")

    def apply_command(self, cmd: list) -> str:
        """One replay or simulator command (`_apply_command`'s
        vocabulary). Returns "rust" when the core applied it, "python"
        when it went through a Python state (a kind the core does not
        apply yet, or an init_side while the scenario still has an
        event that can fire)."""
        kind = cmd[0] if cmd else ""
        if kind == "init_side" and not self._events_pending():
            self.core.apply_init_side(int(cmd[1]))
            return "rust"
        if kind == "end_turn":
            self.core.apply_end_turn()
            return "rust"
        if kind == "move":
            from_side = int(cmd[3]) if len(cmd) > 3 else 0
            self.core.apply_move([int(v) for v in cmd[1]], [int(v) for v in cmd[2]], from_side)
            return "rust"
        if kind == "attack":
            self._apply_attack(cmd)
            return "rust"
        if kind == "recruit":
            self._apply_recruit(cmd)
            return "rust"
        self._python_path(cmd)
        return "python"

    def _apply_recruit(self, cmd: list) -> None:
        """The recruit branch of the applier: the unit from the Python
        builder (`_build_recruit_unit`, the trait roll from the seed),
        placed unable to act this turn, the game's pick-advance
        override on it, the uid counter and the side's gold."""
        from tools.replay_dataset import _build_recruit_unit, _rebuild_unit, _stats_for
        unit_type, tx, ty = str(cmd[1]), int(cmd[2]), int(cmd[3])
        trait_seed = cmd[4] if len(cmd) > 4 else ""
        core = self.core
        g = core.globals_export()
        side = int(g["current_side"])
        ids = core.unit_ids()
        next_uid = max((int(i[1:]) for i in ids if i.startswith("u") and i[1:].isdigit()), default=0) + 1
        new_unit = _build_recruit_unit(unit_type, side, tx, ty, next_uid, game_id=self.game_id,
                                       trait_seed_hex=trait_seed, exp_modifier=int(g["experience_modifier"]))
        spawned = _rebuild_unit(new_unit, current_moves=0, has_attacked=True)
        _choices, pick, _events = core.advance_state_export()
        for (pside, ptype, lst) in pick:
            if pside == side and ptype == unit_type and lst:
                setattr(spawned, "_pickadvance", list(lst))
        self._add_unit(spawned, self._view())
        core.set_global_int("next_uid_counter", int(g["next_uid_counter"]) + 1)
        core.spend_gold(side, int(_stats_for(unit_type).get("cost", 14)))

    def _apply_attack(self, cmd: list) -> None:
        """The attack on the core, then what the kernel leaves to the
        Python builders in the applier's order: the attacker's feeding
        and advancement, the corpse of an attacker killed by a plague
        counter, the defender's feeding and advancement, the corpse of
        a defender killed by plague."""
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
        if out["att_alive"]:
            if out["att_feed"]:
                self._feed(out["att_id"])
            if out["att_advances"]:
                self._advance(out["att_id"])
        elif out["plague_reverse"]:
            self._spawn_corpse(out["att_x"], out["att_y"], out["dfd_side"], out["att_name"])
        if out["dfd_alive"]:
            if out["dfd_feed"]:
                self._feed(out["dfd_id"])
            if out["dfd_advances"]:
                self._advance(out["dfd_id"])
        elif out["plague_forward"]:
            self._spawn_corpse(out["dfd_x"], out["dfd_y"], out["att_side"], out["dfd_name"])

    def _feed(self, uid: str) -> None:
        """One more fed kill on the unit's stash (a new dict: the stash
        is shared across forks)."""
        st = self.unit_stash.get(uid) or {}
        self.unit_stash[uid] = {**st, "_feeding_count": int(st.get("_feeding_count", 0) or 0) + 1}

    def _advance(self, uid: str) -> None:
        """`_maybe_advance_unit` on a carrier holding the unit and the
        advancement globals; the result replaces the unit, the queue,
        the events and the counter go back to the core."""
        from tools.replay_dataset import _maybe_advance_unit
        core = self.core
        g = core.globals_export()
        choices, pick, events = core.advance_state_export()
        unit = unit_from_fields(core.unit_export(uid), self.unit_stash.get(uid))
        gi = SimpleNamespace(
            _experience_modifier=int(g["experience_modifier"]), _advance_choices=list(choices),
            _last_advance_events=[tuple(e) for e in events], _advance_uniform=bool(g["advance_uniform"]),
            _advance_salt=g["advance_salt"], _advance_counter=int(g["advance_counter"]),
            _pickadvance_game={(side, t): list(lst) for (side, t, lst) in pick})
        carrier = SimpleNamespace(game_id=self.game_id, map=SimpleNamespace(units={unit}), global_info=gi)
        advanced = _maybe_advance_unit(carrier, unit)
        core.remove_unit(uid)
        self.unit_stash.pop(uid, None)
        if advanced is not None:
            self._add_unit(advanced, self._view())
        core.set_advance_state([int(c) if isinstance(c, int) else -1 for c in gi._advance_choices],
                               list(pick), [(int(a), int(b)) for a, b in gi._last_advance_events])
        core.set_global_int("advance_counter", int(gi._advance_counter))

    def _spawn_corpse(self, x: int, y: int, side: int, dead_name: str) -> None:
        """The plague corpse (`_build_plague_corpse`) with the next unit
        id of the units left, and the uid counter bumped."""
        from tools.replay_dataset import _build_plague_corpse
        ids = self.core.unit_ids()
        next_uid = max((int(i[1:]) for i in ids if i.startswith("u") and i[1:].isdigit()), default=0) + 1
        g = self.core.globals_export()
        corpse = _build_plague_corpse(dead_name, int(side), int(x), int(y), next_uid, self.game_id,
                                      int(g["experience_modifier"]))
        self._add_unit(corpse, self._view())
        self.core.set_global_int("next_uid_counter", int(g["next_uid_counter"]) + 1)

    # ---- the observation and the encoding over the core ----------------

    def geometry(self):
        from wesnoth_ai.observe import map_geometry
        return map_geometry(self._view())

    def observe(self, side: int, reach: bool = False):
        """`observe.observe(state, side, reach)` over the core: the same
        Observation record, without Unit references (the unit arrays
        follow the core's unit order)."""
        return _observation_from_dict(self.core.observe(int(side), bool(reach)), self.geometry())

    def encode_raw(self, *, type_to_id: Dict[str, int], faction_to_id: Dict[str, int],
                   relevant_set: bool = False, fog_hides_enemy_villages: bool = False):
        """`encoder.encode_raw` over the core for the side to move: the
        same RawEncoded, byte for byte (tests/test_game_core.py)."""
        from wesnoth_ai import encoder as enc
        from wesnoth_ai.classes import Position
        core = self.core
        side = int(core.current_side)
        sides = core.sides_export()
        us = side - 1
        them = 1 - us if len(sides) == 2 else us
        our_fac = sides[us][5] if 0 <= us < len(sides) else ""
        them_fac = sides[them][5] if 0 <= them < len(sides) else ""
        own_recruits = list(sides[us][1]) if 0 <= us < len(sides) else []
        r_ids, r_stats = self._recruit_rows(own_recruits, type_to_id)
        d = core.encode_streams(
            side, bool(relevant_set), self._type_vocab(type_to_id), r_ids, r_stats,
            bool(fog_hides_enemy_villages),
            (enc.HP_NORM, enc.MOVES_NORM, enc.EXP_NORM, enc.COST_NORM, enc.GOLD_NORM,
             enc.INCOME_NORM, enc.VILLAGES_NORM, enc.TURN_NORM),
            enc.MAX_MAP_SIZE - 1, enc.NUM_ALIGNMENTS)
        static = enc._static_hex_arrays(self._view())
        if relevant_set:
            hex_positions = [static.positions[t] for t in d["full_slots"].tolist()]
        else:
            hex_positions = static.positions
        return enc.RawEncoded(
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

    def _type_vocab(self, type_to_id: Dict[str, int]) -> List[int]:
        """The vocab id of every registered type, the overflow bucket
        for the unknown (`encoder._lookup_id`); cached per vocab and
        type count."""
        from wesnoth_ai.encoder import MAX_UNIT_TYPES
        names = tuple(self.core.type_names())
        key = ("vocab", id(type_to_id), len(type_to_id))
        hit = self.caches.get(key)
        if hit is None or hit[0] != names:
            overflow = MAX_UNIT_TYPES - 1
            hit = (names, [min(type_to_id.get(n, overflow), overflow) for n in names])
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
        """The statics as a GameState without units: what the movement
        class registration reads (hexes, terrain codes, time areas)."""
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

    def _events_pending(self) -> bool:
        events = self.statics.get("_scenario_events") or []
        return any(not (getattr(ev, "first_time_only", True) and getattr(ev, "fired", False))
                   for ev in events)

    def _python_path(self, cmd: list) -> None:
        from tools.replay_dataset import _apply_command
        gs = self.to_state()
        _apply_command(gs, list(cmd))
        self.reload(gs)

    def reload(self, gs: GameState) -> None:
        """Take the content of `gs` (a state derived from `to_state`)
        back into the core: units, sides, globals and the stash."""
        if gs.map.hexes is not self.hexes_holder:
            # A terrain morph replaced the hex set: a new core, new
            # statics, new movement classes, and the unit-less view
            # the builders read is rebuilt on its next use.
            fresh = CoreState.from_state(gs)
            self.core, self.statics, self.unit_stash = fresh.core, fresh.statics, fresh.unit_stash
            self.class_ids, self.hexes_holder = fresh.class_ids, fresh.hexes_holder
            self._view_cache = None
            return
        core = self.core
        core.clear_units()
        self.unit_stash = {}
        for u in gs.map.units:
            self._add_unit(u, gs)
        gi = gs.global_info
        for k, v in gi.__dict__.items():
            if k.startswith("_") and k not in MODELED_GLOBALS and k not in _DROPPED_GLOBALS:
                self.statics[k] = v
        self._load_scalars(gs)

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
        core.set_rejected([tuple(p) for p in (getattr(gi, "_recruit_rejected_hexes", None) or ())],
                          [tuple(p) for p in (getattr(gi, "_move_rejected_hexes", None) or ())])
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


def _fork_statics(statics: Dict[str, object]) -> Dict[str, object]:
    """The per-fork copy of the aliased stash (the rules of
    `classes.GlobalInfo.__deepcopy__`)."""
    import copy as _copy
    out: Dict[str, object] = {}
    for k, v in statics.items():
        if k == "_scenario_events":
            out[k] = [ev if getattr(ev, "fired", False) else _copy.copy(ev) for ev in v]
        elif k == "_terrain_codes" or not k.startswith("_"):
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


def _observation_from_dict(d: dict, geometry):
    """`observe.Observation` from the core's dict of arrays."""
    from wesnoth_ai.observe import Observation
    return Observation(int(d["side"]), bool(d["fog_on"]), geometry, list(d["unit_ids"]), d["unit_hex"],
                       d["disc"], d["visible"], d["zoc"], d["enemy"], d["ally"], d["occupied"], d["inert"],
                       d["recruit_row"], d["network"], bool(d["leader_on_keep"]), d.get("acting"),
                       d.get("unit_can_move"), d.get("unit_can_attack"), d.get("landable"),
                       d.get("relevant"), d.get("tok_of_hex"))


def units_equal(a: Unit, b: Unit) -> bool:
    """Field-level equality of two units (the dataclass compares ids only)."""
    if a.__dict__.keys() - {k for k in a.__dict__ if k.startswith("_")} != \
            b.__dict__.keys() - {k for k in b.__dict__ if k.startswith("_")}:
        return False
    for k, v in a.__dict__.items():
        if k.startswith("_"):
            continue
        w = getattr(b, k)
        if k == "attacks":
            if len(v) != len(w) or any(
                    (p.type_id, p.number_strikes, p.damage_per_strike, p.is_ranged, set(p.weapon_specials or ()))
                    != (q.type_id, q.number_strikes, q.damage_per_strike, q.is_ranged, set(q.weapon_specials or ()))
                    for p, q in zip(v, w)):
                return False
        elif isinstance(v, (set, frozenset)):
            if set(map(str, v)) != set(map(str, w)):
                return False
        elif v != w:
            return False
    return True


def states_equal(a: GameState, b: GameState, *, stash: bool = True) -> List[str]:
    """The differences between two states over the modeled content, as
    strings (empty when equal)."""
    diffs: List[str] = []
    ua = {u.id: u for u in a.map.units}
    ub = {u.id: u for u in b.map.units}
    if ua.keys() != ub.keys():
        diffs.append(f"units: {sorted(ua.keys() ^ ub.keys())}")
    for uid in ua.keys() & ub.keys():
        if not units_equal(ua[uid], ub[uid]):
            diffs.append(f"unit {uid}: {ua[uid]} != {ub[uid]}")
        if stash:
            for k in UNIT_STASH_KEYS:
                if getattr(ua[uid], k, None) != getattr(ub[uid], k, None):
                    diffs.append(f"unit {uid} stash {k}")
    if len(a.sides) != len(b.sides):
        diffs.append("sides count")
    for i, (s, t) in enumerate(zip(a.sides, b.sides)):
        if (s.player, list(s.recruits), s.current_gold, s.base_income, s.nb_villages_controlled, s.faction) != \
                (t.player, list(t.recruits), t.current_gold, t.base_income, t.nb_villages_controlled, t.faction):
            diffs.append(f"side {i + 1}: {s} != {t}")
    ga, gb = a.global_info, b.global_info
    for k in ("current_side", "turn_number", "time_of_day", "village_gold", "village_upkeep", "base_income"):
        if getattr(ga, k) != getattr(gb, k):
            diffs.append(f"global {k}: {getattr(ga, k)} != {getattr(gb, k)}")
    for k in MODELED_GLOBALS:
        va, vb = getattr(ga, k, None), getattr(gb, k, None)
        if k in ("_advance_uniform", "_did_first_init_side", "_fog"):
            va, vb = bool(va), bool(vb)
        if k in ("_tod_start_offset", "_rng_request_counter", "_advance_counter"):
            va, vb = int(va or 0), int(vb or 0)
        if k == "_advance_salt":
            va, vb = str(va or ""), str(vb or "")
        if k == "_experience_modifier":
            va, vb = int(va or 100), int(vb or 100)
        if k == "_next_uid_counter":
            va, vb = int(va or 1), int(vb or 1)
        if k in ("_uncovered_units", "_recruit_rejected_hexes", "_move_rejected_hexes"):
            va, vb = set(va or ()), set(vb or ())
        if k in ("_village_owner", "_pickadvance_game"):
            va = {kk: v for kk, v in (va or {}).items() if v}
            vb = {kk: v for kk, v in (vb or {}).items() if v}
        if k in ("_advance_choices", "_last_advance_events"):
            va, vb = [tuple(x) if isinstance(x, (list, tuple)) else x for x in (va or [])], \
                [tuple(x) if isinstance(x, (list, tuple)) else x for x in (vb or [])]
        if k == "_last_checkup_strikes":
            va, vb = va or None, vb or None
        if va != vb:
            diffs.append(f"global {k}: {va!r} != {vb!r}")
    if (a.game_over, a.winner) != (b.game_over, b.winner):
        diffs.append("game over / winner")
    return diffs


__all__ = ["CoreState", "map_static", "type_fields", "unit_fields", "unit_from_fields",
           "units_equal", "states_equal", "game_core_class"]
