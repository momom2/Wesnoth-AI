"""Replay-based supervised-training dataset for Wesnoth AI.

Loads the compact per-replay .json.gz files produced by
`replay_extract.py` and yields `(GameState, action_indices)` pairs by
replaying each game forward.

`GameState` matches the exact shape our encoder expects (see
classes.py / encoder.py), so the encoder we use for self-play can
also train on this data without modification.

`action_indices` is a small dict of slot indices the model's heads
should predict:
  {"actor_idx":  int, "target_idx": int|None, "weapon_idx": int|None,
   "action_type": "move"|"attack"|"recruit"|"recall"|"end_turn"}

The indices correspond to the slot ordering the encoder produces for
the given state — computed by re-running the encoder's ordering rules
here at dataset-load time. If we ever change the encoder's sort, the
same rule change has to land here too.

CLI: `python tools/replay_dataset.py replays_dataset`
prints a summary of the first N replays.
"""

from __future__ import annotations

import gzip
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

# Re-use existing game-state dataclasses.
from classes import (
    Alignment as AlignmentEnum, Attack, DamageType, GameState, GlobalInfo,
    Hex, Map, Position, SideInfo, Terrain, TerrainModifiers, Unit,
)
import combat as cb


log = logging.getLogger("replay_dataset")


# ---------------------------------------------------------------------
# Unit-type stats database (scraped from Wesnoth source)
# ---------------------------------------------------------------------

_UNIT_STATS_PATH = Path(__file__).resolve().parent.parent / "unit_stats.json"
_UNIT_DB: Dict[str, dict] = {}
_MOVETYPE_DB: Dict[str, dict] = {}


def _load_unit_db() -> None:
    """Load the scraped Wesnoth unit data on first access."""
    global _UNIT_DB, _MOVETYPE_DB
    if _UNIT_DB:
        return
    try:
        with _UNIT_STATS_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        _UNIT_DB     = data.get("units", {})
        _MOVETYPE_DB = data.get("movement_types", {})
        log.info(f"Loaded {len(_UNIT_DB)} unit types from unit_stats.json")
    except FileNotFoundError:
        log.warning(f"{_UNIT_STATS_PATH} not found; using fallback stats. "
                    f"Run `python tools/scrape_unit_stats.py wesnoth_src "
                    f"unit_stats.json` to fix.")


# Map our DamageType enum ↔ Wesnoth WML names so we can look up
# resistances stored in unit_stats.json.
_DT_NAME_TO_ENUM = {
    "blade":  DamageType.SLASH,
    "pierce": DamageType.PIERCE,
    "impact": DamageType.IMPACT,
    "fire":   DamageType.FIRE,
    "cold":   DamageType.COLD,
    "arcane": DamageType.ARCANE,
}
_DT_ENUM_TO_NAME = {v: k for k, v in _DT_NAME_TO_ENUM.items()}
# Order matches the resistances list-of-floats in Unit (see classes.py).
_DT_ORDER = [DamageType.SLASH, DamageType.PIERCE, DamageType.IMPACT,
             DamageType.FIRE,  DamageType.COLD,   DamageType.ARCANE]
_DT_NAMES_ORDERED = ["blade", "pierce", "impact", "fire", "cold", "arcane"]


_FALLBACK_STATS = {
    "hitpoints": 33, "moves": 5, "experience": 50, "cost": 14,
    "alignment": "neutral", "level": 1, "advances_to": [],
    "attacks": [{"name": "blade", "type": "blade", "range": "melee",
                 "damage": 5, "number": 2, "specials": []}],
    "defense":    {t: 50 for t in cb.DAMAGE_TYPES + ["flat", "forest", "hills"]},
    "resistance": {t: 100 for t in cb.DAMAGE_TYPES},
    "abilities":  [],
}


def _stats_for(unit_type: str) -> dict:
    """Look up unit-type stats; fall back to defaults for unknown types
    (we'd rather train on approximate stats than crash on a custom unit
    name)."""
    _load_unit_db()
    return _UNIT_DB.get(unit_type, _FALLBACK_STATS)


# ---------------------------------------------------------------------
# Map/hex-code parsing
# ---------------------------------------------------------------------

# Same mapping as state_converter.TERRAIN_BASE_MAP — keep in sync.
_TERRAIN_BASE = {
    "Aa": Terrain.FROZEN, "Gg": Terrain.FLAT, "Gs": Terrain.FLAT,
    "Gd": Terrain.FLAT, "Hh": Terrain.HILLS, "Ha": Terrain.HILLS,
    "Mm": Terrain.MOUNTAINS, "Ms": Terrain.MOUNTAINS, "Md": Terrain.MOUNTAINS,
    # Water variants — all share the SHALLOWWATER defense table for our
    # purposes (Wwf=ford, Wwt=tropical, Wwr=river, Wwg=algae, etc.).
    "Ww": Terrain.SHALLOWWATER, "Wwf": Terrain.SHALLOWWATER,
    "Wwt": Terrain.SHALLOWWATER, "Wwr": Terrain.SHALLOWWATER,
    "Wwg": Terrain.SHALLOWWATER, "Wo": Terrain.DEEPWATER,
    "Ss": Terrain.SWAMP, "Ds": Terrain.SAND,
    "Rr": Terrain.FLAT, "Re": Terrain.FLAT,
    "Ql": Terrain.CAVE, "Xu": Terrain.IMPASSABLE,
    "Uu": Terrain.UNWALKABLE,
    # Castle variants — Chr (river), Chw (water), Cha (snow), Chs (sand)
    # all behave as castles for combat (defense_pct from the unit's
    # `castle` defense entry). Caves of the Basilisk uses Cha; Aethermaw
    # uses Chw / Chw^Xo when the barrier morphs into walkable castles.
    "Ch": Terrain.CASTLE, "Cha": Terrain.CASTLE,
    "Chr": Terrain.CASTLE, "Chs": Terrain.CASTLE, "Chw": Terrain.CASTLE,
    # Off-board placeholder. Aethermaw initially fences the central
    # passageway with `_off^_usr`; T4-T6 [terrain] events convert those
    # cells to Wwf/Chw and they become playable. We include them in the
    # hex set so the events have something to update — without this,
    # units later traversing those hexes get the default `flat` terrain
    # and combat defense math goes wrong (a Merman Hunter on Wwf has
    # 60% defense; on `flat` only 30%).
    "_off": Terrain.IMPASSABLE,
}


# Alias terrains: Wesnoth's `aliasof=Gt, Wst` etc. tells the engine
# "treat this terrain as both grass AND shallow_water for defense /
# movement, picking whichever is best for the unit on it." We mirror
# that by mapping each WML terrain code to the LIST of canonical WML
# defense keys it should be evaluated against. Defense is then the
# minimum (best) over the list. From wesnoth_src/data/core/terrain.cfg.
_DEFENSE_KEYS_FOR_CODE: Dict[str, List[str]] = {
    "Aa":   ["frozen"],
    "Gg":   ["flat"], "Gs": ["flat"], "Gd": ["flat"],
    "Hh":   ["hills"], "Ha": ["hills"],
    "Mm":   ["mountains"], "Ms": ["mountains"], "Md": ["mountains"],
    "Ww":   ["shallow_water"],
    "Wwf":  ["shallow_water", "flat"],   # Ford = Wst | Gt
    "Wwt":  ["shallow_water"],
    "Wwr":  ["shallow_water"],
    "Wwg":  ["shallow_water"],
    "Wo":   ["deep_water"],
    "Ss":   ["swamp_water"],
    "Ds":   ["sand"],
    "Rr":   ["flat"], "Re": ["flat"],
    "Ql":   ["cave"],
    "Xu":   ["impassable"], "Xv": ["impassable"], "Xm": ["impassable"],
    "Uu":   ["unwalkable"],
    "Ch":   ["castle"],
    "Cha":  ["castle", "frozen"],        # snowy castle = Ct | At
    "Chr":  ["castle"],                  # ruined castle = Ct
    "Chs":  ["castle"],
    "Chw":  ["castle", "shallow_water"], # sunken castle = Ct | Wst
    "Cd":   ["castle"], "Kd": ["castle"], "Kh": ["castle"], "Ko": ["castle"],
    "Hhd":  ["hills"], "Mv": ["mountains"],
    "Tb":   ["flat"], "Iwr": ["flat"],
    "Rb":   ["flat"],
    "Wwt": ["shallow_water"],
    "Qxua": ["unwalkable"], "Qxu":  ["unwalkable"], "Wog": ["deep_water"],
    "Qlf":  ["cave"], "Md":   ["mountains"],
    "_off": ["impassable"],
}


# Overlay codes (the part after `^` in 'Re^Fmf') and their defense
# keys. An overlay typically REPLACES the base for defense purposes
# (forest overlay on grass → use forest defense, not grass) but the
# rule isn't universal: village overlays add the village defense
# alongside the base, not replace it. We model the conservative
# interpretation: overlay keys EXTEND the base's defense-key list,
# and defense_pct is the min (best) over the union.
_OVERLAY_DEFENSE_KEYS: Dict[str, List[str]] = {
    # Forest overlays — Fp/Fpa/Ftr/Fma/Fda/Fmf/Fdf/Fet etc. all read
    # as "forest" for defense.
    "Fp":  ["forest"], "Fpa": ["forest"], "Ftr": ["forest"],
    "Fma": ["forest"], "Fda": ["forest"], "Fmf": ["forest"],
    "Fdf": ["forest"], "Fet": ["forest"], "Ft":  ["forest"],
    "Fds": ["forest"],
    # Village overlays — defender uses VILLAGE defense.
    "Vh":  ["village"], "Vhc": ["village"], "Vhh": ["village"],
    "Vhs": ["village"], "Vct": ["village"], "Vc":  ["village"],
    "Vd":  ["village"], "Vda": ["village"], "Vdt": ["village"],
    "Vm":  ["village"], "Vmd": ["village"], "Vmw": ["village"],
    "Vo":  ["village"], "Vot": ["village"], "Vov": ["village"],
    "Vu":  ["village"], "Vud": ["village"], "Vu_a": ["village"],
    "Vwm": ["village"], "Vwh": ["village"], "Vws": ["village"],
    "Vd1": ["village"], "Vh1": ["village"], "Vc1": ["village"],
    "Gvs": ["village"],
    # Castle overlays.
    "Xo":  ["castle"],
    # Misc structural overlays — leave the base alone.
    "Em":  [], "Edp": [], "Eh":  [], "Es":  [],
    "Bsb\\": [], "Bsb/": [], "Bs\\": [], "Bs/": [], "Bs|": [],
    "Tf":  [], "Tu":  [], "Th":  [],
    "Xm":  ["impassable"], "Xv":  ["impassable"],
}


def _defense_keys_for_code(code: str) -> List[str]:
    """Return the canonical WML defense-table keys for a Wesnoth
    terrain code (handles `^` overlays and `aliasof=` aliases).

    Examples:
      'Wwf'        → ['shallow_water', 'flat']  (Ford = Wst | Gt)
      'Re^Fmf'     → ['forest']                 (forest overlay on road)
      'Aa^Vha'     → ['frozen', 'village']      (snow + village)
      'Hh^Vhh'     → ['hills', 'village']       (hills + village)

    Falls back to ['flat'] for unknown codes."""
    base = code
    overlay = ""
    if "^" in code:
        base, overlay = code.split("^", 1)
    base_keys = _DEFENSE_KEYS_FOR_CODE.get(base)
    if base_keys is None:
        # Single-letter-prefix fallback for unknown codes.
        base_keys = ["flat"]
    keys = list(base_keys)
    if overlay:
        ov_keys = _OVERLAY_DEFENSE_KEYS.get(overlay, [])
        # Forest/village overlays REPLACE the base for the purpose of
        # defense — a forest on grass is forest, a village on hills is
        # village. We append their keys so callers can min() over the
        # union; for a unit with better forest defense than grass
        # defense, forest wins.
        for k in ov_keys:
            if k not in keys:
                keys.append(k)
    return keys


def _parse_hex_code(code: str) -> Tuple[set, set]:
    """Return (terrain_types, static_modifiers) for one hex code like
    'Hh^Fms' or 'Gg^Vh'."""
    terrains: set = set()
    modifiers: set = set()
    if "^" in code:
        base, overlay = code.split("^", 1)
    else:
        base, overlay = code, ""
    terrains.add(_TERRAIN_BASE.get(base, Terrain.FLAT))
    if "V" in overlay:
        terrains.add(Terrain.VILLAGE)
    if "F" in overlay:
        terrains.add(Terrain.FOREST)
    # Keep/castle — track as modifiers (static property of the tile).
    if "K" in overlay or "K" in base:
        modifiers.add(TerrainModifiers.KEEP)
        terrains.add(Terrain.CASTLE)
    if "C" in base or "C" in overlay:
        modifiers.add(TerrainModifiers.CASTLE)
        terrains.add(Terrain.CASTLE)
    return terrains, modifiers


def parse_terrain_codes(map_data: str) -> Dict[Tuple[int, int], str]:
    """Return a dict from playable (x, y) → full WML terrain code
    INCLUDING overlay (e.g., 'Re^Fmf'). The overlay matters for
    defense — `Re^Fmf` is "road overlaid with mixed forest" which a
    unit treats as forest, not flat. `_defense_keys_for_code` resolves
    the full code to the list of defense keys (handling Wesnoth's
    `aliasof=` semantics; the unit's defense_pct is the BEST/min over
    those keys). Coords are 0-indexed (border-stripped)."""
    out: Dict[Tuple[int, int], str] = {}
    rows = [r for r in map_data.splitlines() if r.strip()]
    if not rows:
        return out
    border = 1
    for y_with_border, row in enumerate(rows):
        if y_with_border < border or y_with_border >= len(rows) - border:
            continue
        cells = [c.strip() for c in row.split(",")]
        for x_with_border, cell in enumerate(cells):
            if x_with_border < border or x_with_border >= len(cells) - border:
                continue
            if not cell:
                continue
            if cell[:1].isdigit() and cell[1:2] == " ":
                cell = cell[2:]
            out[(x_with_border - border, y_with_border - border)] = cell
    return out


def parse_map_data(map_data: str) -> List[Hex]:
    """Split the row-major `map_data` string into a list of Hex.

    The WML map_data format is comma-separated rows of hex codes,
    one line per row, Y-major. Wesnoth maps include a 1-hex border
    around the playable area (see `wesnoth_src/src/map/map.hpp`:
    `border_size = 1`). The .map / map_data string therefore has its
    first/last row and first/last column as border padding.

    We strip the border and produce Hex objects at 0-indexed coordinates
    that align with Wesnoth's INTERNAL coordinates: hex at
    Position(0,0) corresponds to WML (1,1) (the first playable hex,
    file row 1 col 1). The dumper then emits WML by adding 1 again.
    """
    out: List[Hex] = []
    rows = [r for r in map_data.splitlines() if r.strip()]
    if not rows:
        return out
    border = 1
    for y_with_border, row in enumerate(rows):
        # Skip the first and last border row.
        if y_with_border < border or y_with_border >= len(rows) - border:
            continue
        cells = [c.strip() for c in row.split(",")]
        for x_with_border, cell in enumerate(cells):
            # Skip border columns.
            if x_with_border < border or x_with_border >= len(cells) - border:
                continue
            if not cell:
                continue
            # Strip leading "1 " or "2 " starting-position markers.
            if cell[:1].isdigit() and cell[1:2] == " ":
                cell = cell[2:]
            # `_off^_usr` (and other `_off*` codes) mark off-board cells
            # that scenarios may convert to playable terrain via [terrain]
            # events. Include them with an IMPASSABLE marker so events
            # have a Hex to update; otherwise post-event combat on those
            # hexes uses the default `flat` and gets defense wrong.
            terr, mods = _parse_hex_code(cell)
            # Subtract the border offset so internal coords are 0-indexed
            # from the first playable hex.
            out.append(Hex(
                position=Position(x=x_with_border - border,
                                  y=y_with_border - border),
                terrain_types=terr,
                modifiers=mods,
            ))
    return out


# ---------------------------------------------------------------------
# Replay → GameState reconstruction
# ---------------------------------------------------------------------

@dataclass
class ActionIndices:
    """Observed-action → model-head target indices.

    actor_idx is an index into (unit_slots + recruit_slots + end_turn),
    the same ordering the encoder produces. target_idx is an index
    into hex_positions. weapon_idx is an attack-slot index (0..3).
    None where the head is irrelevant for this action type.
    """
    action_type: str
    actor_idx:   int
    target_idx:  Optional[int]  = None
    weapon_idx:  Optional[int]  = None


def _alignment_from_str(s: str) -> AlignmentEnum:
    return {
        "lawful":  AlignmentEnum.LAWFUL,
        "neutral": AlignmentEnum.NEUTRAL,
        "chaotic": AlignmentEnum.CHAOTIC,
        "liminal": AlignmentEnum.LIMINAL,
    }.get((s or "").lower(), AlignmentEnum.NEUTRAL)


def _attacks_from_stats(stats: dict) -> List[Attack]:
    out: List[Attack] = []
    for a in stats.get("attacks", []):
        dt = _DT_NAME_TO_ENUM.get(a.get("type", "blade"), DamageType.SLASH)
        out.append(Attack(
            type_id=dt,
            number_strikes=int(a.get("number", 1)),
            damage_per_strike=int(a.get("damage", 1)),
            is_ranged=(a.get("range") == "ranged"),
            weapon_specials=set(),  # specials carried in stats["attacks"][i]["specials"]
                                    # — combat.py handles them by name look-up
        ))
    if not out:
        out.append(Attack(DamageType.SLASH, 1, 1, False, set()))
    return out


def _scaled_max_exp(base_exp: int, exp_modifier: int) -> int:
    """Port of `unit_type::experience_needed(true)` from
    src/units/types.cpp:
        int exp = (experience_needed_ * experience_modifier + 50) / 100;
        if (exp < 1) exp = 1;
    `experience_modifier` is a per-game setting (default 100, common
    values 30/50/70). Replays carry it in [multiplayer]
    experience_modifier=. Affects EVERY unit's xp-to-advance.
    """
    return max(1, (int(base_exp) * int(exp_modifier) + 50) // 100)


def _build_unit(u: dict, apply_leader_traits: bool = False,
                game_id: str = "", exp_modifier: int = 100) -> Unit:
    """Reconstruct a Unit dataclass from the extractor's starting_units
    entry, supplemented with stats from `unit_stats.json` for the
    fields we know are needed by the combat pipeline (resistances,
    alignment, level, max_experience, attacks).

    Replay's `[unit]` block in `[replay_start]` carries hp/max_hp/etc
    for the leader; for everyone else we lean on the stats DB.

    When `apply_leader_traits=True` and the unit is a leader, applies
    the user-stated leader-trait rule (no traits except race-musthaves
    and "quick" for ≤4-mp leaders).
    """
    stats = _stats_for(u["type"])
    max_hp     = u.get("max_hp",     int(stats.get("hitpoints", 33)))
    max_moves  = u.get("max_moves",  int(stats.get("moves",     5)))
    base_max_exp = int(stats.get("experience", 50))
    max_exp    = u.get("max_exp",    _scaled_max_exp(base_max_exp, exp_modifier))
    cost       = u.get("cost",       int(stats.get("cost",      14)))

    # Resistances list aligned with DamageType enum order; values are
    # fractions (0.0 = no special resistance, 0.5 = 50% damage taken,
    # i.e. WML resistance=50). Wesnoth's WML stores percentages where
    # 100 = "takes full damage"; we convert to "fraction reduction":
    #   fraction = 1 - (100 - resist) / 100 = resist/100 - ?? — actually
    # we store the raw percent so combat.py's resistance logic can use
    # it directly. (combat.py expects Dict[str,int] of percent.)
    res = stats.get("resistance", {})
    resistances = [float(res.get(name, 100)) / 100.0 for name in _DT_NAMES_ORDERED]

    # Defense per terrain — needed for combat. Stored as a list aligned
    # with the terrain order in classes.py (we keep the same convention
    # as state_converter for cross-compat).
    terrain_keys = [
        "castle", "cave", "deep_water", "flat", "forest", "frozen",
        "fungus", "hills", "mountains", "reef", "sand", "shallow_water",
        "swamp_water", "village",
    ]
    def_table = stats.get("defense", {})
    defenses = [float(def_table.get(t, 50)) / 100.0 for t in terrain_keys]

    # Petrified statues (Thousand Stings Garrison) keep their unit
    # type for visualization but lose attacks/movement/HP-display per
    # the UNIT_PETRIFY scenario macro. We mirror that by tagging them
    # `petrified` in statuses; the combat handler then forces no
    # counter-attack when defending. We also zero out moves so the
    # encoder doesn't think they can act.
    petrified = bool(u.get("petrified", False))
    initial_statuses: set = {"petrified"} if petrified else set()
    base = Unit(
        id=f"u{u['uid']}",
        name=u["type"],
        name_id=0,
        side=u["side"],
        is_leader=u.get("is_leader", False),
        position=Position(x=u["x"], y=u["y"]),
        max_hp=max_hp,
        max_moves=0 if petrified else max_moves,
        max_exp=max_exp,
        cost=cost,
        alignment=_alignment_from_str(stats.get("alignment", "neutral")),
        levelup_names=list(stats.get("advances_to", [])),
        current_hp=u.get("hp", max_hp),
        current_moves=0 if petrified else max_moves,
        current_exp=0,
        has_attacked=petrified,         # statues never act
        attacks=[] if petrified else _attacks_from_stats(stats),
        resistances=resistances,
        defenses=defenses,
        movement_costs=[1] * 14,
        abilities=set() if petrified else set(stats.get("abilities", [])),
        traits=set(),
        statuses=initial_statuses,
    )
    # Build a per-unit defense table starting from the unit-type's
    # cap-collapsed values (scraper already merged movetype + unit
    # [defense] entries). Traits like `feral` mutate this in-place
    # via apply_traits_to_unit's defense_overrides hook. Stashed on
    # the Unit via setattr so combat can pick it up without us
    # changing classes.py.
    defense_table = dict(stats.get("defense", {}))
    setattr(base, "_defense_table", defense_table)
    if apply_leader_traits and base.is_leader:
        from tools.traits import roll_traits, apply_traits_to_unit
        race = stats.get("race", "")
        trait_ids = roll_traits(
            u["type"], race,
            seed_token=f"{game_id}:leader{u['side']}:{u['type']}",
            is_leader=True,
            base_movement=int(stats.get("moves", 5)),
            trait_info=stats.get("traits"),
        )
        if trait_ids:
            base = apply_traits_to_unit(
                base, trait_ids,
                level=int(stats.get("level", 1)),
                defense_table=defense_table,
            )
            setattr(base, "_defense_table", defense_table)
    return base


def _build_recruit_unit(unit_type: str, side: int, x: int, y: int,
                        next_uid: int, game_id: str = "",
                        trait_seed_hex: str = "",
                        exp_modifier: int = 100) -> Unit:
    """Spawn a fresh recruit with the trait roll Wesnoth used.

    `trait_seed_hex` is the per-recruit `[random_seed]` from the
    replay (back-filled by replay_extract). For undead/mechanical/
    elemental recruits this is the empty string — those races have
    only musthave traits and don't consume a seed.
    """
    base = _build_unit({
        "uid": next_uid, "type": unit_type, "side": side,
        "x": x, "y": y, "is_leader": False,
    }, exp_modifier=exp_modifier)
    from tools.traits import roll_traits, apply_traits_to_unit
    stats = _stats_for(unit_type)
    race = stats.get("race", "")
    trait_ids = roll_traits(
        unit_type, race,
        seed_hex=trait_seed_hex,
        seed_token=f"{game_id}:u{next_uid}:{unit_type}",
        is_leader=False,
        base_movement=int(stats.get("moves", 5)),
        trait_info=stats.get("traits"),
        n_genders=int(stats.get("n_genders", 1)),
    )
    defense_table = dict(getattr(base, "_defense_table", None) or
                         stats.get("defense", {}))
    out = apply_traits_to_unit(base, trait_ids,
                               level=int(stats.get("level", 1)),
                               defense_table=defense_table)
    setattr(out, "_defense_table", defense_table)
    return out


def _build_initial_gamestate(data: dict) -> GameState:
    raw_map = data.get("map_data", "")
    hexes = set(parse_map_data(raw_map))
    terrain_codes = parse_terrain_codes(raw_map)
    game_id = data.get("game_id", "")
    exp_mod = int(data.get("experience_modifier", 100) or 100)
    units = {
        _build_unit(u, apply_leader_traits=True, game_id=game_id,
                    exp_modifier=exp_mod)
        for u in data.get("starting_units", [])
    }
    sides = [
        SideInfo(
            player=f"Side {s['side']}",
            recruits=list(s.get("recruit", [])),
            current_gold=s.get("gold", 100),
            base_income=s.get("base_income", 2),
            nb_villages_controlled=0,
            # Faction name for encoder conditioning. Persisted in the
            # per-replay json.gz by replay_extract.extract_replay.
            faction=s.get("faction", ""),
        )
        for s in data.get("starting_sides", [])
    ]
    current_side = 1
    size_x = max((h.position.x for h in hexes), default=0) + 1
    size_y = max((h.position.y for h in hexes), default=0) + 1

    gs = GameState(
        game_id=data.get("game_id", "?"),
        map=Map(size_x=size_x, size_y=size_y,
                mask=set(), fog=set(),
                hexes=hexes, units=units),
        global_info=GlobalInfo(
            current_side=current_side, turn_number=0,
            time_of_day=_tod_for_turn(1), village_gold=2,
            village_upkeep=1, base_income=2,
        ),
        sides=sides,
    )
    # Stash the raw replay metadata for tools that need pixel-exact
    # round-tripping (the save-state dumper uses these to avoid
    # re-rendering the map and to recover side leader-types). These
    # attributes are not part of the encoder contract — purely a
    # pass-through for tools that have the original .json.gz on hand.
    setattr(gs.global_info, "_raw_map_data", data.get("map_data", ""))
    setattr(gs.global_info, "_terrain_codes", terrain_codes)
    # ToD start offset for random_start_time scenarios. 0 means turn-1
    # is dawn (the default 2p case). Other values shift the cycle so
    # that turn-1 reads as e.g. afternoon (offset=2) — matching the
    # server-side `tod_manager::resolve_random` decision recorded in
    # the replay's [scenario] / [replay_start] `current_time` attr.
    setattr(gs.global_info, "_tod_start_offset",
            int(data.get("tod_start_index", 0) or 0))
    setattr(gs.global_info, "_raw_starting_sides",
            list(data.get("starting_sides", [])))
    setattr(gs.global_info, "_scenario_id", data.get("scenario_id", ""))
    setattr(gs.global_info, "_experience_modifier", exp_mod)
    return gs


def _find_unit_at(gs: GameState, x: int, y: int) -> Optional[Unit]:
    for u in gs.map.units:
        if u.position.x == x and u.position.y == y:
            return u
    return None


def _terrain_keys_at(gs: GameState, x: int, y: int) -> List[str]:
    """Return the WML defense-table keys to evaluate for the hex at
    (x,y). Honors Wesnoth's `aliasof=` semantics: a Ford (Wwf) returns
    ['shallow_water', 'flat'] so callers can pick whichever defense is
    best for the unit standing on it."""
    codes_dict = getattr(gs.global_info, "_terrain_codes", {}) or {}
    code = codes_dict.get((x, y))
    if not code:
        return ["flat"]
    return _defense_keys_for_code(code)


def _terrain_at(gs: GameState, x: int, y: int) -> str:
    """Return a Wesnoth-WML terrain key (e.g. 'forest', 'flat',
    'village') for the hex at (x, y), defaulting to 'flat' if missing
    from our parsed hex set or carrying an unmapped Terrain enum."""
    hex_obj = next(
        (h for h in gs.map.hexes if h.position.x == x and h.position.y == y),
        None,
    )
    if hex_obj is None:
        return "flat"
    # Pick the most informative terrain: village > castle > forest > base.
    tts = hex_obj.terrain_types
    name_map = {
        Terrain.VILLAGE: "village",
        Terrain.CASTLE:  "castle",
        Terrain.FOREST:  "forest",
        Terrain.HILLS:   "hills",
        Terrain.MOUNTAINS: "mountains",
        Terrain.SWAMP:   "swamp_water",
        Terrain.SAND:    "sand",
        Terrain.SHALLOWWATER: "shallow_water",
        Terrain.DEEPWATER: "deep_water",
        Terrain.FROZEN:  "frozen",
        Terrain.CAVE:    "cave",
        Terrain.UNWALKABLE: "unwalkable",
        Terrain.IMPASSABLE: "impassable",
        Terrain.FLAT:    "flat",
    }
    for pref in (Terrain.VILLAGE, Terrain.CASTLE, Terrain.FOREST,
                 Terrain.HILLS, Terrain.MOUNTAINS, Terrain.SWAMP,
                 Terrain.SAND, Terrain.SHALLOWWATER, Terrain.DEEPWATER,
                 Terrain.FROZEN, Terrain.CAVE, Terrain.UNWALKABLE,
                 Terrain.IMPASSABLE, Terrain.FLAT):
        if pref in tts:
            return name_map[pref]
    return "flat"


def _tod_cycle_index(turn_number: int, start_offset: int = 0) -> int:
    """Compute the cycle index (0..5) for `turn_number` given a starting
    offset. `start_offset` defaults to 0 (turn 1 = dawn). For replays
    with `random_start_time=yes` resolved server-side, the offset
    encodes which ToD the server picked."""
    return (max(1, turn_number) - 1 + max(0, start_offset)) % 6


def _lawful_bonus_for_turn(turn_number: int, start_offset: int = 0) -> int:
    """Default 6-step ToD cycle: dawn(0), morning(+25), afternoon(+25),
    dusk(0), first_watch(-25), second_watch(-25). `start_offset`
    handles random-start-time scenarios where turn-1 is not dawn."""
    return cb.TOD_DEFAULT_CYCLE[_tod_cycle_index(turn_number, start_offset)][1]


def _tod_for_turn(turn_number: int, start_offset: int = 0) -> str:
    """Return the Wesnoth ToD name (dawn / morning / afternoon / dusk /
    first_watch / second_watch) for the given 1-indexed turn. Honors
    `start_offset` for random_start_time scenarios."""
    return cb.TOD_DEFAULT_CYCLE[_tod_cycle_index(turn_number, start_offset)][0]


def _lawful_bonus_at(gs: GameState, x: int, y: int, turn_number: int) -> int:
    """Per-hex lawful_bonus. Honors scenario-defined [time_area] zones
    (Tombs of Kesorak's dark/illuminated regions, Elensefar Courtyard's
    underground keeps, etc.) — those override the global ToD cycle on
    their hexes, and the override has its own per-position cycle.
    Falls back to the default 6-step cycle when no [time_area] applies.
    """
    start_offset = int(getattr(gs.global_info, "_tod_start_offset", 0) or 0)
    areas = getattr(gs.global_info, "_time_areas", None)
    if areas:
        cycle = areas.get((x, y))
        if cycle:
            # Area cycles index from turn 1 too — apply the same offset
            # so a random-start-time scenario sees the area cycle in
            # the right phase relative to the global one.
            idx = (max(1, turn_number) - 1 + start_offset) % len(cycle)
            return int(cycle[idx])
    return _lawful_bonus_for_turn(turn_number, start_offset)


def _to_combat_unit(u: Unit, terrain_key) -> cb.CombatUnit:
    """Convert our Unit dataclass + current terrain → CombatUnit
    snapshot consumable by combat.resolve_attack.

    `terrain_key` may be a single string OR a list of strings — pass a
    list for aliased terrains (e.g., Wwf=Ford resolves to
    ['shallow_water', 'flat']). The unit's defense_pct is then the
    BEST (minimum) over the listed keys, matching Wesnoth's `aliasof=`
    semantics where a Drake on a Ford takes the better of its grass
    (70) or shallow-water (80) defense.

    Weapons are taken from `u.attacks`, NOT base stats — so traits
    that modify damage (strong: +1 melee, dextrous: +1 ranged, weak:
    -1 melee) are reflected. We pull the name / specials list from
    base stats by ALIGNMENT to u.attacks (same index, since
    `_attacks_from_stats` and trait application both preserve order).
    """
    stats = _stats_for(u.name)
    base_attacks = stats.get("attacks", [])
    weapons = []
    for i, ua in enumerate(u.attacks):
        # Match name / specials / range / type from base stats by index.
        if i < len(base_attacks):
            atk_stat = base_attacks[i]
            name = atk_stat.get("name", "?")
            wtype = atk_stat.get("type", "blade")
            wrange = atk_stat.get("range", "melee")
            specials = list(atk_stat.get("specials", []))
        else:
            name = "?"
            wtype = _DT_ENUM_TO_NAME.get(ua.type_id, "blade")
            wrange = "ranged" if ua.is_ranged else "melee"
            specials = list(ua.weapon_specials) if ua.weapon_specials else []
        weapons.append(cb.Weapon(
            name=name,
            damage=int(ua.damage_per_strike),  # trait-adjusted
            number=int(ua.number_strikes),
            range=wrange,
            type=wtype,
            specials=specials,
        ))
    if not weapons:
        weapons.append(cb.Weapon("none", 1, 1, "melee", "blade", []))
    res = stats.get("resistance", {})
    # Per-unit defense table if the unit has trait-applied overrides
    # (feral village=50). Falls back to the unit-type's static table.
    def_table = getattr(u, "_defense_table", None) or stats.get("defense", {})
    keys = [terrain_key] if isinstance(terrain_key, str) else list(terrain_key)
    if not keys:
        keys = ["flat"]
    defense_pct = min(int(def_table.get(k, 50)) for k in keys)
    return cb.CombatUnit(
        side=u.side,
        hp=int(u.current_hp),
        max_hp=int(u.max_hp),
        level=int(stats.get("level", 1)),
        experience=int(u.current_exp),
        max_experience=int(u.max_exp),
        alignment=cb.alignment_from_str(stats.get("alignment", "neutral")),
        weapons=weapons,
        resistance={dt: int(res.get(dt, 100)) for dt in cb.DAMAGE_TYPES},
        defense_pct=defense_pct,
        is_slowed="slowed" in u.statuses,
        is_poisoned="poisoned" in u.statuses,
    )


def _replace_unit(gs: GameState, old: Unit, **changes) -> Unit:
    """Replace `old` in gs.map.units with a new Unit carrying changed
    fields. Returns the replacement (or the original on no-op).

    Any non-dataclass attributes stashed via setattr (e.g.,
    `_defense_table` for trait-modified defense tables) are copied
    forward. Without this, every combat round would silently drop
    feral's village=50 override.
    """
    base_fields = {
        k: v for k, v in old.__dict__.items() if not k.startswith("_")
    }
    new = Unit(**{**base_fields, **changes})
    for k, v in old.__dict__.items():
        if k.startswith("_"):
            setattr(new, k, v)
    gs.map.units.discard(old)
    gs.map.units.add(new)
    return new


def _maybe_advance_unit(gs: GameState, u: Unit) -> Unit:
    """Port of Wesnoth's advance_unit_at: when a unit's XP reaches its
    max_experience, advance it to the next type. The replay's [choose]
    commands carry the player's pick when there are multiple
    advancement options; we apply them via _apply_advancement_choice
    when those fire. Otherwise, default to advances_to[0].

    Wesnoth rules:
      - HP heals to the new max_hp on advancement.
      - XP carries over the excess (current_exp - max_exp).
      - The unit gains all musthave traits of the new type (we keep
        existing traits + apply new musthaves).
      - If the unit is at the highest level (no `advances_to`), no
        advance happens.
    """
    if u.current_exp < u.max_exp:
        return u
    stats = _stats_for(u.name)
    targets = list(stats.get("advances_to", []))
    if not targets:
        # Max level — Wesnoth fires AMLA (after-max-level advancement)
        # which restores HP and grants +3 max_hp. We approximate as
        # +3 max_hp + heal to full + reset XP.
        new_max_hp = u.max_hp + 3
        return _replace_unit(
            gs, u,
            max_hp=new_max_hp,
            current_hp=new_max_hp,
            current_exp=max(0, u.current_exp - u.max_exp),
        )
    # Pick the choice — pop from the per-attack queue if available
    # (replay [choose] commands push integer indices here, in
    # attacker-first / defender-second order per Wesnoth's
    # attack_unit_and_advance).
    pending = list(getattr(gs.global_info, "_advance_choices", []) or [])
    if pending:
        choice = pending.pop(0)
        if isinstance(choice, int) and 0 <= choice < len(targets):
            new_type = targets[choice]
        elif isinstance(choice, str) and choice in targets:
            new_type = choice
        else:
            new_type = targets[0]
    else:
        new_type = targets[0]
    # Variation persistence: a "Walking Corpse:mounted" advances to
    # "Soulless:mounted" if Soulless defines the same variation. This
    # mirrors Wesnoth's `unit::advance_to` which copies the source's
    # variation across when the destination has it. Falls back to the
    # base advances_to entry if no matching variation exists.
    if ":" in u.name:
        _, src_var = u.name.split(":", 1)
        if src_var:
            candidate = f"{new_type}:{src_var}"
            if candidate in _UNIT_DB:
                new_type = candidate
    setattr(gs.global_info, "_advance_choices", pending)

    new_stats = _stats_for(new_type)
    new_max_hp_base = int(new_stats.get("hitpoints", u.max_hp))
    exp_mod = int(getattr(gs.global_info, "_experience_modifier", 100) or 100)
    new_max_xp_base = _scaled_max_exp(int(new_stats.get("experience", 50)), exp_mod)
    new_attacks_base = _attacks_from_stats(new_stats)
    new_max_moves_base = int(new_stats.get("moves", u.max_moves))
    new_level = int(new_stats.get("level", 1))
    res = new_stats.get("resistance", {})
    resistances = [float(res.get(name, 100)) / 100.0 for name in _DT_NAMES_ORDERED]
    terrain_keys = [
        "castle", "cave", "deep_water", "flat", "forest", "frozen",
        "fungus", "hills", "mountains", "reef", "sand", "shallow_water",
        "swamp_water", "village",
    ]
    def_table = new_stats.get("defense", {})
    defenses = [float(def_table.get(t, 50)) / 100.0 for t in terrain_keys]

    # Build a "fresh base" unit at the new type — traits are re-applied
    # below so percentage modifiers (quick -5% HP, intelligent -20% XP)
    # use the new type's base stats, not the old type's. Wesnoth recomputes
    # stats on advancement and re-applies trait modifications; clearing
    # u.traits here forces apply_traits_to_unit to re-walk the deltas
    # cleanly.
    fresh = _replace_unit(
        gs, u,
        name=new_type,
        max_hp=new_max_hp_base,
        current_hp=new_max_hp_base,      # full heal on advancement
        max_exp=new_max_xp_base,
        current_exp=max(0, u.current_exp - u.max_exp),
        max_moves=new_max_moves_base,
        current_moves=min(u.current_moves, new_max_moves_base),
        cost=int(new_stats.get("cost", u.cost or 14)),
        alignment=_alignment_from_str(new_stats.get("alignment", "neutral")),
        levelup_names=list(new_stats.get("advances_to", [])),
        attacks=new_attacks_base,
        resistances=resistances,
        defenses=defenses,
        abilities=set(new_stats.get("abilities", [])),
        traits=set(),
    )
    # Build the new defense table from the advanced unit-type, then
    # let trait re-application stamp `feral` village=50 etc. back on
    # top — same hook used at recruit time. Stash on the unit so
    # combat post-advancement reads from the right table.
    advanced_def_table = dict(new_stats.get("defense", {}))
    setattr(fresh, "_defense_table", advanced_def_table)
    trait_ids = list(u.traits)
    if trait_ids:
        from tools.traits import apply_traits_to_unit
        advanced = apply_traits_to_unit(
            fresh, trait_ids, level=new_level,
            defense_table=advanced_def_table,
        )
        setattr(advanced, "_defense_table", advanced_def_table)
        gs.map.units.discard(fresh)
        gs.map.units.add(advanced)
        return advanced
    return fresh


def _apply_command(gs: GameState, cmd: list) -> None:
    """Mutate GameState by applying one replay command (compact format
    from replay_extract.extract_replay)."""
    if not cmd:
        return
    kind = cmd[0]

    if kind == "init_side":
        side = cmd[1]
        gs.global_info.current_side = side
        # Turn counter: increment on every init_side(1). Initial state
        # has turn_number=0 (pre-game), so the very first init_side(1)
        # bumps to 1 (Turn 1), matching Wesnoth's UI numbering.
        if side == 1:
            gs.global_info.turn_number += 1
            # ToD advances with the turn (default 6-step cycle). Update
            # so encoded states / save dumps / combat reads see the
            # correct lawful_bonus and ToD name. Side-2 init_side keeps
            # the current ToD — Wesnoth changes ToD only between turns.
            tod_offset = int(
                getattr(gs.global_info, "_tod_start_offset", 0) or 0
            )
            gs.global_info.time_of_day = _tod_for_turn(
                gs.global_info.turn_number, tod_offset
            )
        # Fire scenario-script side/turn events for whichever scenario
        # we're in. For Aethermaw this morphs impassable terrain into
        # water at side N turn 4/5/6.
        _fire_turn_events(gs, side, gs.global_info.turn_number)

        # Apply per-turn healing for `side`'s units. Direct port of
        # Wesnoth's wesnoth_src/src/actions/heal.cpp::calculate_healing.
        #
        # Algorithm:
        #   1. Rest heal: +2 if patient.resting OR patient.is_healthy
        #      (resting is True iff the unit didn't move or fight on
        #      its previous turn — set via the `resting` status by us
        #      below).
        #   2. Main healing branch:
        #      - If NOT poisoned: healing += max(village_heal,
        #        regen_value, max_adjacent_healer_value).
        #        (`update_healing` in heal.cpp keeps the MAX across
        #        sources, so multiple +4 healers don't stack and a
        #        +4 healer adjacent to a unit on a village still gives
        #        only the +8 village heal.)
        #      - If poisoned: poison_progress() classifies the cure
        #        situation:
        #          POISON_CURE  → poison cleared, NO main healing
        #          POISON_SLOW  → no poison damage, NO main healing
        #          POISON_NORMAL → -8 HP (only on patient's own turn)
        #   3. Cap healing to [-current_hp+1, max_hp - current_hp].
        #   4. After healing, set resting=True for all side-N units
        #      (Wesnoth does this in play_controller.cpp:548 right
        #      after calculate_healing).
        from tools.abilities import healer_heal_amount, adjacent_curer
        new_units = set()
        for u in gs.map.units:
            if u.side != side:
                new_units.add(u); continue

            stats = _stats_for(u.name)
            abilities = set(u.abilities) | set(stats.get("abilities", []))
            terrain = _terrain_at(gs, u.position.x, u.position.y)
            on_village = (terrain == "village")
            has_regen = "regenerate" in abilities
            healer_amt = healer_heal_amount(u, gs.map.units)
            has_curer = adjacent_curer(u, gs.map.units)
            poisoned = "poisoned" in u.statuses
            new_statuses = set(u.statuses)

            # --- step 1: rest heal --------------------------------------
            rest_eligible = (
                "resting" in u.statuses or "healthy" in u.traits
            )
            healing = cb.REST_HEAL_AMOUNT if rest_eligible else 0

            # --- step 2: main healing / poison interaction --------------
            cure_poison = False
            if not poisoned:
                # MAX of available healing sources, NOT sum.
                main_heal = 0
                if on_village:
                    main_heal = max(main_heal, cb.VILLAGE_HEAL)
                if has_regen:
                    main_heal = max(main_heal, cb.REGENERATE_AMOUNT)
                main_heal = max(main_heal, healer_amt)
                healing += main_heal
            else:
                # poison_progress: village/regen=cure → CURE; adjacent
                # healer (heals+4/+8 without cures) → SLOW; otherwise
                # NORMAL.
                if on_village or has_regen or has_curer:
                    curing = "cure"
                elif healer_amt > 0:
                    curing = "slow"
                else:
                    curing = "normal"

                if curing == "cure":
                    cure_poison = True
                    # No main healing, no poison damage. Rest heal
                    # stays (added before).
                elif curing == "slow":
                    # Heal and poison cancel — neither effect on HP.
                    # Rest heal stays.
                    pass
                else:  # normal
                    # Poison damage only applies on patient's own turn,
                    # which is exactly init_side(side) for side=patient.side
                    # — already guaranteed by our outer `if u.side != side` check.
                    healing -= cb.POISON_AMOUNT

            # --- step 3: cap to [-current_hp+1, max_hp - current_hp] ----
            max_heal = max(0, u.max_hp - u.current_hp)
            min_heal = min(0, 1 - u.current_hp)
            if healing < min_heal:
                healing = min_heal
            elif healing > max_heal:
                healing = max_heal

            new_hp = u.current_hp + healing
            if cure_poison:
                new_statuses.discard("poisoned")

            # NOTE: slowed clears at end_turn (NOT here). Wesnoth's
            # `unit::end_turn()` (units/unit.cpp:1284) does
            # `set_state(STATE_SLOWED, false)`, called once per unit by
            # `game_board::end_turn(side)` when that side's turn ends.
            # That means a unit slowed during the enemy's turn keeps
            # the slow active throughout its OWN next turn, and only
            # drops it at end_turn — exactly the window where the
            # halved damage matters.

            # --- step 4: post-healing, set resting=True -----------------
            # (will be cleared again by move/attack during the upcoming
            # turn). Subsequent moves and attacks discard "resting".
            new_statuses.add("resting")
            base_fields = {
                k: v for k, v in u.__dict__.items() if not k.startswith("_")
            }
            healed = Unit(**{
                **base_fields,
                "current_moves": u.max_moves,
                "has_attacked": False,
                "current_hp": new_hp,
                "statuses": new_statuses,
            })
            for k, v in u.__dict__.items():
                if k.startswith("_"):
                    setattr(healed, k, v)
            new_units.add(healed)
        gs.map.units = new_units

        # Income & upkeep — direct port of play_controller.cpp:524-534
        # (only fires when turn > 1, matching Wesnoth's "no income on
        # the first side turn" rule).
        #   income  = base_income + villages_owned * village_gold
        #   upkeep  = sum(unit.level for non-loyal units of side)
        #   support = villages_owned * village_support
        #   net     = +income − max(0, upkeep − support)
        if (1 <= side <= len(gs.sides)
                and gs.global_info.turn_number > 1):
            s = gs.sides[side - 1]
            owned = s.nb_villages_controlled
            village_gold = gs.global_info.village_gold or 2
            village_support = gs.global_info.village_upkeep or 1
            income = s.base_income + owned * village_gold
            upkeep = 0
            for u in gs.map.units:
                if u.side != side:
                    continue
                # Leaders never contribute to upkeep, regardless of
                # whether they have the `loyal` trait. Verified in
                # wesnoth_src/src/units/unit.cpp:1746-1751
                # (`unit::upkeep` short-circuits on `can_recruit()`).
                # Without this, our income is short by leader.level
                # gold per turn -- compounding to ~30g over a 30-turn
                # game and biasing the policy toward smaller armies.
                if u.is_leader:
                    continue
                if "loyal" in u.traits:
                    continue
                u_stats = _stats_for(u.name)
                upkeep += int(u_stats.get("level", 1))
            support = owned * village_support
            net_upkeep = max(0, upkeep - support)
            new_gold = s.current_gold + income - net_upkeep
            gs.sides[side - 1] = SideInfo(
                player=s.player, recruits=s.recruits,
                current_gold=new_gold, base_income=s.base_income,
                nb_villages_controlled=owned, faction=s.faction,
            )
        return

    if kind == "end_turn":
        # Port of game_board::end_turn(side) → unit::end_turn() per
        # unit on the ending side. Currently we only need it for
        # clearing the SLOWED state — Wesnoth keeps slow active
        # throughout the slowed unit's own turn and drops it only at
        # the very end. (resting/has_attacked are managed elsewhere.)
        ending_side = gs.global_info.current_side
        new_units = set()
        for u in gs.map.units:
            if u.side != ending_side or "slowed" not in u.statuses:
                new_units.add(u); continue
            new_statuses = set(u.statuses)
            new_statuses.discard("slowed")
            base_fields = {
                k: v for k, v in u.__dict__.items() if not k.startswith("_")
            }
            unslowed = Unit(**{**base_fields, "statuses": new_statuses})
            for k, v in u.__dict__.items():
                if k.startswith("_"):
                    setattr(unslowed, k, v)
            new_units.add(unslowed)
        gs.map.units = new_units
        return

    if kind == "move":
        xs, ys = cmd[1], cmd[2]
        # 4th slot is the from_side (added in newer extracts). Older
        # extracts omit it; treat as 0 (no filter) for backward compat.
        from_side = cmd[3] if len(cmd) > 3 else 0
        sx, sy, tx, ty = xs[0], ys[0], xs[-1], ys[-1]
        # Match the unit at the source hex of the SAME side as the
        # player who issued the move. Without this filter, a stale
        # unit from a previous bug can get moved instead of the real
        # moving unit.
        unit = None
        for u in gs.map.units:
            if u.position.x == sx and u.position.y == sy:
                if from_side == 0 or u.side == from_side:
                    unit = u
                    break
        if unit is None:
            return
        new_statuses = set(unit.statuses)
        new_statuses.discard("resting")
        moved = _replace_unit(
            gs, unit,
            position=Position(x=tx, y=ty),
            current_moves=max(0, unit.current_moves - (len(xs) - 1)),
            statuses=new_statuses,
        )
        if _terrain_at(gs, tx, ty) == "village":
            _capture_village(gs, tx, ty, moved.side)
        return

    if kind == "attack":
        # Compact format: ["attack", ax, ay, dx, dy, a_weapon,
        #                  d_weapon, seed_hex, [choose_indices]?]
        ax, ay, dx, dy, a_weapon = cmd[1], cmd[2], cmd[3], cmd[4], cmd[5]
        d_weapon = cmd[6] if len(cmd) > 6 else -1
        seed_hex = cmd[7] if len(cmd) > 7 else ""
        choices  = list(cmd[8]) if len(cmd) > 8 else []
        # Push the choices onto the advance-choice queue. The unit-type
        # name from advances_to[choice_idx] will be resolved by
        # _maybe_advance_unit.
        if choices:
            existing = list(getattr(gs.global_info, "_advance_choices", []) or [])
            # Convert each choice index → unit-type name. We don't know
            # which advance_to list applies (attacker's vs defender's)
            # at queue time; resolution happens lazily via the index.
            existing.extend(choices)
            setattr(gs.global_info, "_advance_choices", existing)

        att = _find_unit_at(gs, ax, ay)
        dfd = _find_unit_at(gs, dx, dy)
        if att is None or dfd is None:
            return

        # Build CombatUnit snapshots using each unit's CURRENT terrain
        # for defense. (Attacker on its hex can be counter-attacked,
        # so we pass its own terrain for its defense_pct.)
        att_cu = _to_combat_unit(att, _terrain_keys_at(gs, att.position.x, att.position.y))
        dfd_cu = _to_combat_unit(dfd, _terrain_keys_at(gs, dx, dy))
        # Defensive clamp: if our DB lacks weapons for this unit type
        # (e.g., a custom-era unit), fall back to weapon 0 to keep the
        # replay reconstruction running rather than crashing the loader.
        if a_weapon >= len(att_cu.weapons):
            a_weapon = 0
        if d_weapon >= 0 and d_weapon >= len(dfd_cu.weapons):
            d_weapon = 0
        # Petrified defenders can't counter-attack — the UNIT_PETRIFY
        # macro strips their attacks. Force d_weapon = -1 so combat
        # treats this as a one-sided strike. (Wesnoth's attack code
        # checks `defender->attacks().empty()` before counter-attack.)
        if "petrified" in dfd.statuses:
            d_weapon = -1

        # Adjacency-based effects: leadership, illuminate, backstab.
        from tools.abilities import (
            leadership_bonus, illuminate_step, is_backstab_active,
        )
        a_stats_db = _stats_for(att.name)
        d_stats_db = _stats_for(dfd.name)
        a_level = int(a_stats_db.get("level", 1))
        d_level = int(d_stats_db.get("level", 1))

        a_leadership = leadership_bonus(att, gs.map.units, opponent_level=d_level)
        d_leadership = leadership_bonus(dfd, gs.map.units, opponent_level=a_level)

        # Illuminate shifts ToD by +25 (bumps lawful_bonus). We compute
        # per-side using each combatant's own hex — both for the
        # adjacent-illuminate ability AND for scenario [time_area]
        # overrides (Tombs of Kesorak's permanent dark/illuminated
        # zones, Elensefar Courtyard's underground keeps).
        turn = gs.global_info.turn_number
        a_base = _lawful_bonus_at(gs, att.position.x, att.position.y, turn)
        d_base = _lawful_bonus_at(gs, dx, dy, turn)
        a_illum = 25 * illuminate_step(att, gs.map.units)
        d_illum = 25 * illuminate_step(dfd, gs.map.units)
        a_lawful = max(-25, min(25, a_base + a_illum))
        d_lawful = max(-25, min(25, d_base + d_illum))

        # Backstab: active when there's an enemy of the defender on
        # the hex opposite the attacker. Symmetric for the counter-
        # attack (a backstab-flagged defender weapon hitting the
        # attacker also needs an enemy of the attacker opposite from
        # the defender — usually irrelevant, but supported).
        a_backstab = is_backstab_active(att, dfd, gs.map.units)
        d_backstab = is_backstab_active(dfd, att, gs.map.units)

        rng = cb.MTRng(seed_hex) if seed_hex else cb.MTRng("00000000")
        result = cb.resolve_attack(
            att_cu, dfd_cu,
            a_weapon_idx=a_weapon,
            d_weapon_idx=d_weapon if d_weapon >= 0 else None,
            a_lawful_bonus=a_lawful,
            d_lawful_bonus=d_lawful,
            a_leadership_bonus=a_leadership,
            d_leadership_bonus=d_leadership,
            a_backstab_active=a_backstab,
            d_backstab_active=d_backstab,
            rng=rng,
        )

        # Write outcomes back to gs. has_attacked True for attacker;
        # HP/XP from the result; remove dead units. Both combatants
        # also lose the `resting` status (Wesnoth's attack.cpp:1279-80
        # calls `set_resting(false)` on both unit_info refs).
        # Slow/poison flags from the result must be PROPAGATED back to
        # the Unit's statuses set — combat resolved them on a CombatUnit
        # snapshot but we own the persistent state. Without this the
        # next turn's combat sees the unit as un-slowed and damage math
        # diverges (e.g., a slowed Drake firing at a Wose deals 4 dmg
        # per hit instead of 9, letting the Wose live one extra turn).
        att_statuses = set(att.statuses); att_statuses.discard("resting")
        dfd_statuses = set(dfd.statuses); dfd_statuses.discard("resting")
        if result.attacker_slowed:
            att_statuses.add("slowed")
        if result.attacker_poisoned:
            att_statuses.add("poisoned")
        if result.defender_slowed:
            dfd_statuses.add("slowed")
        if result.defender_poisoned:
            dfd_statuses.add("poisoned")
        if result.attacker_alive:
            new_att = _replace_unit(gs, att,
                          has_attacked=True,
                          current_hp=result.attacker_hp_after,
                          current_exp=result.attacker_xp_after,
                          statuses=att_statuses)
            _maybe_advance_unit(gs, new_att)
        else:
            gs.map.units.discard(att)
        if result.defender_alive:
            new_dfd = _replace_unit(gs, dfd,
                          current_hp=result.defender_hp_after,
                          current_exp=result.defender_xp_after,
                          statuses=dfd_statuses)
            _maybe_advance_unit(gs, new_dfd)
        else:
            gs.map.units.discard(dfd)
            # Plague: a kill by a [plague] weapon raises a Walking
            # Corpse (the default plague_type) on the dead unit's hex,
            # side=attacker. Wesnoth restricts it to "plague-able"
            # races: not undead, not mechanical, not elemental — those
            # set status `unplagueable` indirectly via [resistance] or
            # via the unit's `not_living` flag. We check race only
            # (sufficient for default-era 2p).
            if result.plague_spawned and att in gs.map.units:
                _spawn_plague_corpse(gs, dfd,
                                     attacker_side=att.side,
                                     attacker_name=att.name)
        return

    if kind == "recruit":
        unit_type = cmd[1]
        tx, ty = cmd[2], cmd[3]
        # Slot 4 holds the per-recruit trait seed when extracted from
        # the replay's [random_seed] command (empty for undead/etc.).
        trait_seed = cmd[4] if len(cmd) > 4 else ""
        side = gs.global_info.current_side
        next_uid = (max(
            (int(u.id[1:]) for u in gs.map.units if u.id.startswith("u") and u.id[1:].isdigit()),
            default=0,
        ) + 1)
        exp_mod = int(getattr(gs.global_info, "_experience_modifier", 100) or 100)
        new_unit = _build_recruit_unit(unit_type, side, tx, ty, next_uid,
                                       game_id=gs.game_id,
                                       trait_seed_hex=trait_seed,
                                       exp_modifier=exp_mod)
        # On spawn turn, recruits have 0 moves. Preserve any non-field
        # stash (`_defense_table`) — Unit() reconstruction drops it.
        base_fields = {
            k: v for k, v in new_unit.__dict__.items() if not k.startswith("_")
        }
        spawned = Unit(**{**base_fields, "current_moves": 0})
        for k, v in new_unit.__dict__.items():
            if k.startswith("_"):
                setattr(spawned, k, v)
        gs.map.units.add(spawned)

        # Deduct cost from side gold (use unit_db cost; fall back to 14).
        cost = int(_stats_for(unit_type).get("cost", 14))
        if 1 <= side <= len(gs.sides):
            s = gs.sides[side - 1]
            gs.sides[side - 1] = SideInfo(
                player=s.player, recruits=s.recruits,
                current_gold=max(0, s.current_gold - cost),
                base_income=s.base_income,
                nb_villages_controlled=s.nb_villages_controlled,
                faction=s.faction,
            )
        return

    if kind == "recall":
        # Unit identity unknown without the recall list; skip creating one.
        return


# Traits that grant the `unplagueable` status (also `unpoisonable`,
# `undrainable`). From wesnoth_src/data/core/macros/traits.cfg —
# {TRAIT_UNDEAD}, {TRAIT_MECHANICAL}, {TRAIT_ELEMENTAL} all add these
# statuses via [effect] apply_to=status.
_UNPLAGUEABLE_TRAITS = {"undead", "mechanical", "elemental"}


def _is_unplagueable(unit: Unit) -> bool:
    """Mirror Wesnoth's `opp.get_state("unplagueable")` check. A unit
    is unplagueable if (a) it carries the `unplagueable` status (set
    explicitly on some statue/scenery [unit]s) or (b) it has the
    undead / mechanical / elemental musthave trait — those traits
    apply_to=status add=unplagueable in core/macros/traits.cfg."""
    if "unplagueable" in unit.statuses:
        return True
    return bool(set(unit.traits) & _UNPLAGUEABLE_TRAITS)


def _spawn_plague_corpse(gs: GameState, dead: Unit,
                         attacker_side: int,
                         attacker_name: str) -> None:
    """Spawn a Walking Corpse for `attacker_side` on `dead.position`.
    Direct port of Wesnoth's plague-reanimation logic in
    src/actions/attack.cpp:1287. Eligibility:

      1. Killed unit's hex is NOT a village (Wesnoth: `!is_village`).
      2. Killed unit isn't `unplagueable` (undead / mechanical /
         elemental trait, or explicit status).
      3. Killed unit's `undead_variation != "null"` — Mudcrawlers and
         Jinns explicitly opt out of being raised.
      4. Attacker's plague special is present (caller already filtered
         on `result.plague_spawned`).

    Type and variation:
      - The plague special's `type=` attribute (rare; most weapons
        leave it empty) overrides the spawn type.
      - Otherwise the spawn uses the ATTACKER's `parent_id` — for
        a base "Walking Corpse" attacker, parent_id is "Walking
        Corpse" (variations carry that as parent).
      - The killed unit's `undead_variation` is then applied as a
        [variation] modification on the new corpse, picking the
        right Walking Corpse flavor (Mounted, Drake, Wose, etc.).
        Different variations have different stats. We approximate
        this by spawning the base "Walking Corpse" and noting the
        variation in `unit.statuses` so encoding/dumping can carry
        it. A future pass could materialize per-variation stats.

    Fresh corpse: full HP, 0 XP, 0 moves, has_attacked=True
    (Wesnoth sets `attacks=0`/`movement=0` so it can't act this turn).
    """
    # Eligibility checks.
    if _is_unplagueable(dead):
        return
    dead_stats = _stats_for(dead.name)
    if str(dead_stats.get("undead_variation", "")).lower() == "null":
        return
    if _terrain_at(gs, dead.position.x, dead.position.y) == "village":
        return

    # Pick the spawn unit-type: attacker's PARENT_ID (`Walking Corpse`
    # / `Soulless` for default-era plague users), NOT the attacker's
    # own name. If the attacker is itself a variation -- e.g. a
    # `Walking Corpse:mounted` (a previously-raised cavalry corpse)
    # killing a Wose -- Wesnoth's plague spawns a fresh `Walking
    # Corpse` (the parent) with the dead unit's `undead_variation`
    # applied, NOT a `Walking Corpse:mounted:wose` chained-variation
    # which doesn't exist in the unit DB. The parent_id is stored
    # under "id" on every entry (variations inherit their parent's
    # id; see scrape_unit_stats.extract_variations). Fall back to
    # attacker.name if the lookup misses (custom era, missing scrape).
    attacker_stats = _stats_for(attacker_name)
    base_type = str(attacker_stats.get("id", "") or "").strip() or attacker_name
    variation = str(dead_stats.get("undead_variation", "")).strip()
    spawn_type = f"{base_type}:{variation}" if variation else base_type
    if variation and spawn_type not in _UNIT_DB:
        # Variation table didn't include this id — fall back to base.
        # Logged once per missing variation so we can tighten up later.
        log.debug(f"plague: no variation '{variation}' for {base_type}; using base")
        spawn_type = base_type

    next_uid = (max(
        (int(u.id[1:]) for u in gs.map.units if u.id.startswith("u") and u.id[1:].isdigit()),
        default=0,
    ) + 1)
    exp_mod = int(getattr(gs.global_info, "_experience_modifier", 100) or 100)
    corpse = _build_recruit_unit(
        spawn_type,
        side=attacker_side,
        x=dead.position.x, y=dead.position.y,
        next_uid=next_uid,
        game_id=gs.game_id,
        trait_seed_hex="",
        exp_modifier=exp_mod,
    )
    new_statuses = set(corpse.statuses)
    base_fields = {
        k: v for k, v in corpse.__dict__.items() if not k.startswith("_")
    }
    spawned = Unit(**{**base_fields,
                     "current_moves": 0,
                     "has_attacked": True,
                     "statuses": new_statuses})
    for k, v in corpse.__dict__.items():
        if k.startswith("_"):
            setattr(spawned, k, v)
    if variation:
        spawned.statuses.add(f"variation:{variation}")
    gs.map.units.add(spawned)


def _capture_village(gs: GameState, x: int, y: int, capturing_side: int) -> None:
    """Mark the village at (x,y) as belonging to `capturing_side` and
    update side village counts. We track ownership via a per-game
    `_village_owner: Dict[(x,y) -> side]` we stash on gs.global_info
    (lightweight; survives within iter_replay_pairs)."""
    hex_obj = next(
        (h for h in gs.map.hexes if h.position.x == x and h.position.y == y),
        None,
    )
    if hex_obj is None:
        return

    # Per-replay village-owner map. Lazy-initialize on first call.
    owner_map: Dict[Tuple[int, int], int] = getattr(
        gs.global_info, "_village_owner", None
    ) or {}
    prev_owner = owner_map.get((x, y), 0)
    owner_map[(x, y)] = capturing_side
    setattr(gs.global_info, "_village_owner", owner_map)

    # Mark static modifier so encoder sees a "owned" village.
    hex_obj.modifiers.add(TerrainModifiers.VILLAGE)

    # Decrement old owner's count (if any), increment new.
    if prev_owner and 1 <= prev_owner <= len(gs.sides):
        s = gs.sides[prev_owner - 1]
        gs.sides[prev_owner - 1] = SideInfo(
            player=s.player, recruits=s.recruits,
            current_gold=s.current_gold,
            base_income=s.base_income,
            nb_villages_controlled=max(0, s.nb_villages_controlled - 1),
            faction=s.faction,
        )
    if 1 <= capturing_side <= len(gs.sides) and prev_owner != capturing_side:
        s = gs.sides[capturing_side - 1]
        gs.sides[capturing_side - 1] = SideInfo(
            player=s.player, recruits=s.recruits,
            current_gold=s.current_gold,
            base_income=s.base_income,
            nb_villages_controlled=s.nb_villages_controlled + 1,
            faction=s.faction,
        )


def _action_indices(gs: GameState, cmd: list) -> Optional[ActionIndices]:
    """Convert a compact replay command into slot indices the model's
    heads should predict.

    Returns None for commands that aren't player policy actions
    (init_side, etc.). Actor/target ordering MATCHES the encoder's
    sort: units sorted by (y, x, id), then recruits.
    """
    if not cmd:
        return None
    kind = cmd[0]

    # Helper: produce the encoder-aligned unit list for the current side.
    current_side = gs.global_info.current_side
    units_sorted = sorted(
        gs.map.units,
        key=lambda u: (u.position.y, u.position.x, u.id),
    )

    # Hex ordering: encoder sorts hexes by (y, x) row-major. Same here.
    hex_positions = sorted(
        (h.position for h in gs.map.hexes),
        key=lambda p: (p.y, p.x),
    )
    pos_to_hex_idx = {(p.x, p.y): i for i, p in enumerate(hex_positions)}

    if kind == "end_turn":
        # Last actor slot = end_turn sentinel.
        end_idx = len(units_sorted) + sum(
            len(s.recruits) for s in gs.sides
        )
        return ActionIndices("end_turn", actor_idx=end_idx)

    if kind == "move":
        xs, ys = cmd[1], cmd[2]
        sx, sy = xs[0], ys[0]
        tx, ty = xs[-1], ys[-1]
        # Find actor = the unit at (sx, sy).
        actor = None
        for i, u in enumerate(units_sorted):
            if u.position.x == sx and u.position.y == sy and u.side == current_side:
                actor = i; break
        if actor is None:
            return None
        target = pos_to_hex_idx.get((tx, ty))
        if target is None:
            return None
        return ActionIndices("move", actor_idx=actor, target_idx=target)

    if kind == "attack":
        ax, ay, dx, dy, weapon = cmd[1], cmd[2], cmd[3], cmd[4], cmd[5]
        actor = None
        for i, u in enumerate(units_sorted):
            if u.position.x == ax and u.position.y == ay and u.side == current_side:
                actor = i; break
        if actor is None:
            return None
        target = pos_to_hex_idx.get((dx, dy))
        if target is None:
            return None
        return ActionIndices("attack", actor_idx=actor,
                             target_idx=target, weapon_idx=weapon)

    if kind == "recruit":
        unit_type = cmd[1]
        tx, ty = cmd[2], cmd[3]
        # Recruit actor slots come after all units, ordered by encoder
        # rules (iterate sides in order, then their recruit list).
        offset = len(units_sorted)
        actor = None
        for side_idx, side_info in enumerate(gs.sides, start=1):
            for r_name in side_info.recruits:
                if side_idx == current_side and r_name == unit_type:
                    actor = offset
                    break
                offset += 1
            if actor is not None:
                break
        if actor is None:
            return None
        target = pos_to_hex_idx.get((tx, ty))
        if target is None:
            return None
        return ActionIndices("recruit", actor_idx=actor, target_idx=target)

    # recall / init_side / unknown → skip.
    return None


# ---------------------------------------------------------------------
# Public Iterator
# ---------------------------------------------------------------------

def _setup_scenario_events(gs: GameState, scenario_id: str):
    """Load scenario WML for `scenario_id` (if present in our wesnoth_src
    tree) and stash the event list on `gs.global_info` for use during
    command application. Fires `prestart` and `start` events immediately
    (these run before turn 1 in Wesnoth). Also processes any top-level
    [time_area] blocks (e.g., Tombs of Kesorak's dark/illuminated zones)
    so per-hex lawful_bonus overrides are in place before turn 1.
    """
    try:
        from tools.scenario_events import (
            load_events_for_scenario, load_scenario_wml,
            fire_event, setup_static_time_areas,
        )
    except ImportError:
        # If scenario_events isn't importable for some reason, silently
        # skip — the reconstruction still runs, just without events.
        return
    if not scenario_id:
        setattr(gs.global_info, "_scenario_events", [])
        return
    root = load_scenario_wml(scenario_id)
    if root is None:
        setattr(gs.global_info, "_scenario_events", [])
        return
    # Top-level [time_area]s (declared outside any event) apply from
    # game start. Must run BEFORE prestart events — Elensefar's prestart
    # event itself contains a [time_area] (see scenario_events.py
    # _time_area_action), and we don't want order-of-events to flip.
    setup_static_time_areas(gs, root)
    from tools.scenario_events import collect_events
    events = collect_events(root)
    setattr(gs.global_info, "_scenario_events", events)
    if events:
        fire_event(gs, events, "prestart")
        fire_event(gs, events, "start")


def _fire_turn_events(gs: GameState, side: int, turn: int) -> None:
    """Fire the side/turn events Wesnoth would dispatch at this moment.
    Triggers we recognize: 'side N turn M', 'turn M', 'new turn'.
    """
    events = getattr(gs.global_info, "_scenario_events", None)
    if not events:
        return
    from tools.scenario_events import fire_event
    fire_event(gs, events, f"side {side} turn {turn}")
    fire_event(gs, events, f"turn {turn}")
    fire_event(gs, events, "new turn")
    fire_event(gs, events, "side turn")


def iter_replay_pairs(gz_path: Path) -> Iterator[Tuple[GameState, ActionIndices]]:
    """Yield (state_before, action_indices) for each player command
    in one .json.gz replay."""
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    gs = _build_initial_gamestate(data)
    _setup_scenario_events(gs, data.get("scenario_id", ""))
    for cmd in data.get("commands", []):
        ai = _action_indices(gs, cmd)
        if ai is not None:
            yield gs, ai
        _apply_command(gs, cmd)


def iter_replay_pairs_with_state(gz_path: Path
                                 ) -> Iterator[Tuple[GameState, Optional[ActionIndices]]]:
    """Like iter_replay_pairs but yields the running state for EVERY
    command (including init_side / recall) and yields the FINAL state
    after the last command. Useful for tools that want to inspect or
    dump the state at any point in the replay (e.g. save-state dumper)."""
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    gs = _build_initial_gamestate(data)
    _setup_scenario_events(gs, data.get("scenario_id", ""))
    for cmd in data.get("commands", []):
        ai = _action_indices(gs, cmd)
        yield gs, ai
        _apply_command(gs, cmd)
    # One more yield with the post-final state for "give me the end".
    yield gs, None


def iter_dataset(dataset_dir: Path) -> Iterator[Tuple[GameState, ActionIndices]]:
    """Walk every replay in `dataset_dir` and yield all pairs.

    Order: file-sorted (stable). Caller can shuffle at file-granularity
    by shuffling the glob result.
    """
    for gz in sorted(dataset_dir.glob("*.json.gz")):
        try:
            yield from iter_replay_pairs(gz)
        except Exception as e:
            log.warning(f"{gz.name}: {e}")


def filter_competitive_2p(dataset_dir: Path) -> List[Path]:
    """Return only replay files whose (scenario_id, factions) pass the
    competitive-2p filter. Reads each replay's index.jsonl line first
    (cheap) — avoids decompressing the full .json.gz unless it passes
    the cheap checks.
    """
    # Import here to keep replay_dataset importable even if tools/ isn't
    # on sys.path (the main training entry point does the insert).
    from scenarios import is_competitive_2p

    PLAYER_FACTIONS = {"Drakes", "Knalgan Alliance", "Rebels",
                       "Loyalists", "Northerners", "Undead"}
    index_path = dataset_dir / "index.jsonl"
    if not index_path.exists():
        log.warning(f"{index_path} not found; scanning all .json.gz instead")
        return sorted(dataset_dir.glob("*.json.gz"))

    import json as _json
    kept: List[Path] = []
    for line in index_path.open(encoding="utf-8"):
        meta = _json.loads(line)
        if not is_competitive_2p(meta.get("scenario_id", "")):
            continue
        factions = meta.get("factions", [])
        players = [f for f in factions if f in PLAYER_FACTIONS]
        non_players = [f for f in factions if f not in PLAYER_FACTIONS]
        if len(players) != 2 or len(non_players) > 1:
            continue
        kept.append(dataset_dir / meta["file"])
    return kept


# ---------------------------------------------------------------------
# CLI — sanity-check summary
# ---------------------------------------------------------------------

def main(argv: List[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(argv) != 2:
        print("usage: replay_dataset.py DATASET_DIR"); return 2
    d = Path(argv[1])
    n_files = 0; n_pairs = 0
    type_counts: Dict[str, int] = {}
    for gz in sorted(d.glob("*.json.gz")):
        n_files += 1
        try:
            for _state, ai in iter_replay_pairs(gz):
                n_pairs += 1
                type_counts[ai.action_type] = type_counts.get(ai.action_type, 0) + 1
        except Exception as e:
            log.warning(f"  skip {gz.name}: {e}")
        if n_files % 100 == 0:
            print(f"  {n_files} files  {n_pairs} pairs  types={type_counts}")
        if n_files >= 500:
            break
    print(f"\nDone. {n_files} files, {n_pairs} pairs")
    print(f"Action-type distribution: {type_counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
