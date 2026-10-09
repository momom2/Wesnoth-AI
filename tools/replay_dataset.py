"""Replay-based supervised-training dataset for Wesnoth AI.

Loads the compact per-replay .json.gz files produced by
`replay_extract.py` and yields `(GameState, action_indices)` pairs by
replaying each game forward on the Rust core (`record_core`: the
record's initial state, its scenario set up, then its commands through
`CoreState.apply_command`); each pair's state is a view of the core.

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

import dataclasses
import gzip
import hashlib
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

# Re-use existing game-state dataclasses.
from wesnoth_ai.classes import (
    PLAYER_SIDES, GameState, GlobalInfo, Hex, Map, Position, SideInfo, Terrain,
    TerrainModifiers, Unit,
)
from wesnoth_ai import combat as cb
from wesnoth_ai.paths import UNIT_STATS_PATH
# The sides that delay their shroud updates (docs/wesnoth_rules.md
# "Delayed shroud updates").
from wesnoth_ai import delayed_shroud
# The one place that knows how a map cell's starting-position prefix is
# stripped (the engine's string_to_number_); never re-implement it here.
from wesnoth_ai.rules.terrain_resolver import strip_start_position, terrain_mask
from wesnoth_ai.rules.wml_state import split_map_grid          # noqa: F401 (re-export)
from wesnoth_ai.rules.wml_state import fix_time_index, village_economy


log = logging.getLogger("replay_dataset")


# ---------------------------------------------------------------------
# Unit-type stats database (scraped from Wesnoth source)
# ---------------------------------------------------------------------

_UNIT_DB: Dict[str, dict] = {}


def _load_unit_db() -> None:
    """Load the scraped Wesnoth unit data on first access."""
    global _UNIT_DB
    if _UNIT_DB:
        return
    try:
        with UNIT_STATS_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        _UNIT_DB = data.get("units", {})
        log.info(f"Loaded {len(_UNIT_DB)} unit types from unit_stats.json")
    except FileNotFoundError:
        log.warning(f"{UNIT_STATS_PATH} not found; using fallback stats. "
                    f"Run `python tools/scrape_unit_stats.py wesnoth_src "
                    f"unit_stats.json` to fix.")


_FALLBACK_STATS = {
    "hitpoints": 33, "moves": 5, "experience": 50, "cost": 14,
    "alignment": "neutral", "level": 1, "advances_to": [],
    "attacks": [{"name": "blade", "type": "blade", "range": "melee",
                 "damage": 5, "number": 2, "specials": []}],
    "defense":    {t: 50 for t in cb.DAMAGE_TYPES + ["flat", "forest", "hills"]},
    "resistance": {t: 100 for t in cb.DAMAGE_TYPES},
    "abilities":  [],
}


# Unit types that fell back to _FALLBACK_STATS, so the warning fires
# once per type instead of once per lookup (this is a hot path).
_UNKNOWN_TYPES: set = set()


def unknown_unit_types() -> frozenset:
    """Every unit type that has fallen back to the generic stats in
    this process. Empty on a healthy run: the 1.18.4 scrape covers
    every unit the default era and the shipped scenarios field."""
    return frozenset(_UNKNOWN_TYPES)


def _stats_for(unit_type: str) -> dict:
    """Look up unit-type stats; fall back to defaults for unknown types
    (we'd rather train on approximate stats than crash on a custom unit
    name).

    The fallback is a 33 HP level-1 with 50% defense everywhere and one
    5x2 blade attack, so it is WRONG for anything real: combat math,
    the value head's material reading and the combat oracle all read
    these numbers. It used to be silent, which is how a stats mismatch
    would hide (see the 2026-09-11 out-of-memory batches: a quiet
    fallback path costs data nobody counts). It now warns once per
    type, and `unknown_unit_types()` reports the set.
    """
    _load_unit_db()
    stats = _UNIT_DB.get(unit_type)
    if stats is None:
        if unit_type not in _UNKNOWN_TYPES:
            _UNKNOWN_TYPES.add(unit_type)
            log.warning(
                "unit type %r is not in unit_stats.json; using generic "
                "fallback stats (33 HP, 50%% defense, 5x2 blade). Combat "
                "and value readings for it are wrong. If this is a real "
                "1.18.4 unit the scrape is incomplete.", unit_type)
        return _FALLBACK_STATS
    return stats


# ---------------------------------------------------------------------
# Map/hex-code parsing
# ---------------------------------------------------------------------

# The ONE terrain-base table. Add codes HERE only.
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
    # `Uu` is "Cave Floor" (aliasof=Ut, the cave abstract). NOT
    # unwalkable -- units walk into caves all the time. Mapping it to
    # UNWALKABLE made every cave-floor hex impassable for our
    # movement validator and broke recon on Caves of the Basilisk
    # (multiple replays' turn-2 leader moves into the underground
    # village `Uu^Vud` rejected as "impassable"). Same fix for
    # `Uue` (earthy cave floor).
    "Uu": Terrain.CAVE, "Uue": Terrain.CAVE,
    # Castle variants — Chr (river), Chw (water), Cha (snow), Chs (sand)
    # all behave as castles for combat (defense_pct from the unit's
    # `castle` defense entry). Caves of the Basilisk uses Cha; Aethermaw
    # morphs its central barrier into walkable Chw castles at turns 4-6.
    # (Its two Chw^Xo whirlpool hexes are NOT walkable: ^Xo is the
    # Impassable Overlay, mvt_alias=Xt — the movement resolver reads
    # the full code from `_terrain_codes` and prices them 99. This
    # base-code table only feeds Hex.terrain_types/defense-key
    # fallbacks, which never legalize standing there.)
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
    # Cave floor (Uu / Uue) -- aliasof=Ut, defends/moves as cave.
    # NOT unwalkable: cf. correction note above on the
    # `_TERRAIN_FOR_CODE` mapping.
    "Uu":   ["cave"], "Uue": ["cave"],
    "Ch":   ["castle"],
    "Cha":  ["castle", "frozen"],        # snowy castle = Ct | At
    "Chr":  ["castle"],                  # ruined castle = Ct
    "Chs":  ["castle"],
    "Chw":  ["castle", "shallow_water"], # sunken castle = Ct | Wst
    "Cd":   ["castle"], "Kd": ["castle"], "Kh": ["castle"], "Ko": ["castle"],
    "Hhd":  ["hills"], "Mv": ["mountains"],
    "Tb":   ["flat"], "Iwr": ["flat"],
    "Rb":   ["flat"],
    "Qxua": ["unwalkable"], "Qxu":  ["unwalkable"], "Wog": ["deep_water"],
    "Qlf":  ["cave"],
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
    rows, border = split_map_grid(map_data)
    if not rows:
        return out
    for y_with_border, row in enumerate(rows):
        if y_with_border < border or y_with_border >= len(rows) - border:
            continue
        cells = [c.strip() for c in row.split(",")]
        for x_with_border, cell in enumerate(cells):
            if x_with_border < border or x_with_border >= len(cells) - border:
                continue
            if not cell:
                continue
            out[(x_with_border - border, y_with_border - border)] = \
                strip_start_position(cell)
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
    rows, border = split_map_grid(map_data)
    if not rows:
        return out
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
            cell = strip_start_position(cell)
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
                terrain_mask=terrain_mask(cell),
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
    type_idx is the UnitActionType (ATTACK/MOVE) for unit actors,
    None for recruit / end_turn / legacy. Derived at extract time
    from action_type:
      "attack" -> 0 (UnitActionType.ATTACK)
      "move"   -> 1 (UnitActionType.MOVE)
      else     -> None
    """
    action_type: str
    actor_idx:   int
    target_idx:  Optional[int]  = None
    weapon_idx:  Optional[int]  = None
    type_idx:    Optional[int]  = None
    # Relevant-set label basis only: the human's target hex is on the
    # board but has no slot in the relevant subset. The pair is kept
    # (actor / type / weapon heads still train) with target_idx None,
    # and the trainer counts these -- by construction of the subset
    # they should never occur (docs/model_cost_study_20260905.md 2.3).
    target_off_subset: bool = False
    # What the slots must point at: the acting unit's hex, the label's
    # target hex and the recruited type, from the command itself.
    # `encode_worker.label_slot_mismatch` checks them against the
    # encoded tokens. None on end_turn, and on records pickled before
    # the fields existed (the class defaults serve those).
    source_hex: Optional[Tuple[int, int]] = None
    target_hex: Optional[Tuple[int, int]] = None
    recruit_type: Optional[str] = None


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


def _player_sides(starting_sides) -> list:
    """Sides 1 and 2, the players of a 2p game. Maps with a scenery
    side 3 (1,648 of the first 3,000 corpus games) carry no fog
    attribute for it, and it must not decide the game's setting."""
    return [s for s in (starting_sides or []) if int(s.get("side", 0) or 0) in PLAYER_SIDES]


def extra_side_turns(data: dict) -> Tuple[frozenset, frozenset]:
    """(acting, silent): the sides a replay declares beyond the players,
    split by whether they take turns. The engine gives a turn every
    round to each side whose controller is not null and never to a null
    one (docs/wesnoth_rules.md "Side order within a turn"). An extracted
    record keeps no controller, but its commands hold an init_side for
    every side turn the engine played, so the split is complete once
    the record reaches turn 2. Over the imitation corpus it matches the
    scenarios' own [side] controllers in all 7,118 games that declare a
    third side (2026-09-25)."""
    declared = {int(s.get("side", 0) or 0) for s in data.get("starting_sides") or []}
    extra = declared - set(PLAYER_SIDES) - {0}
    acting = {int(c[1]) for c in data.get("commands") or []
              if c and c[0] == "init_side" and len(c) > 1}
    return frozenset(extra & acting), frozenset(extra - acting)


def fog_on_for(starting_sides) -> bool:
    """The encoder's fog switch for a replay: on when either player has
    fog, off when both have it off (games with shroud never reach it:
    `quarantine_reason`). Files from before the flags were recorded (no
    `fog` key) read as fog on, which is what the encoder assumed for
    them all along."""
    sides = _player_sides(starting_sides)
    if not sides or not any("fog" in s for s in sides):
        return True
    return any(bool(s.get("fog", True)) for s in sides)


def corpus_version_of(dataset_dir: Path) -> int:
    """The version of the rules a corpus was built under, from its
    manifest's rows (tools/build_imitation_dataset.CORPUS_VERSION);
    1 for a corpus whose rows do not say, or without a manifest. A
    corpus mixing versions raises ValueError: its labels follow two
    sets of rules."""
    man = Path(dataset_dir) / "manifest.jsonl"
    if not man.exists():
        return 1
    versions = {int(json.loads(line).get("corpus_version", 1))
                for line in man.read_text(encoding="utf-8").splitlines() if line.strip()}
    if len(versions) > 1:
        raise ValueError(f"{man} mixes corpus versions {sorted(versions)}")
    return versions.pop() if versions else 1


def manifest_holdout_split(rows, dataset_dir: Path):
    """(train_rows, holdout_rows) of index rows (dicts with "file") by
    the dataset manifest's holdout flag, or None without a manifest.
    Rows absent from the manifest (quarantined, stale) are dropped.
    Every tool that trains on a corpus splits through here: the
    shuffled first-N splits of the value tools had 98% of the
    manifest holdout in their training rows (2026-09-08 review)."""
    man = Path(dataset_dir) / "manifest.jsonl"
    if not man.exists():
        return None
    flags = {r["file"]: bool(r.get("holdout"))
             for r in (json.loads(line) for line in man.read_text(encoding="utf-8").splitlines()
                       if line.strip())}
    train = [r for r in rows if r["file"] in flags and not flags[r["file"]]]
    hold = [r for r in rows if flags.get(r["file"])]
    return train, hold


# Commands a match key reads at least. A game reloaded from a save and
# played on differently shares its history up to the reload: 9 clusters
# (19 games) of the 2026-09 corpus share 30 or more commands and differ
# within 200, each between the same two players (2026-09-26 crawl).
MATCH_KEY_PREFIX = 30
# ... and at most: the key's length before 2026-09-26, reached only by a
# game with no seed in its first 30 commands (5 of 2,000 sampled).
MATCH_KEY_PREFIX_MAX = 200


def _seeded(cmd) -> bool:
    """Whether a command carries an engine random seed (a recruit's
    trait roll, an attack's strikes): no two games share one."""
    if not isinstance(cmd, list) or not cmd:
        return False
    return ((cmd[0] == "recruit" and len(cmd) > 4 and bool(cmd[4]))
            or (cmd[0] == "attack" and len(cmd) > 7 and bool(cmd[7])))


def match_key(data: dict, prefix: int = MATCH_KEY_PREFIX) -> str:
    """One string per MATCH: the map, the starting units and the first
    `prefix` commands, read on to the first command carrying an engine
    random seed when those have none (up to MATCH_KEY_PREFIX_MAX).
    Copies of the same game saved at different turns share it, a
    re-upload under another date shares it, and so does a game reloaded
    and played on differently; two different games never do."""
    commands = data.get("commands", [])
    first_seed = next((i for i, c in enumerate(commands) if _seeded(c)), len(commands))
    head = commands[:min(max(prefix, first_seed + 1), MATCH_KEY_PREFIX_MAX)]
    h = hashlib.sha1()
    h.update(json.dumps(head, separators=(",", ":")).encode("utf-8"))
    h.update(b"|")
    h.update(str(data.get("map_data", "")).encode("utf-8"))
    h.update(b"|")
    h.update(json.dumps(data.get("starting_units", []), sort_keys=True,
                        separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()


def command_hash(data: dict) -> str:
    """sha1 of the whole command stream (identity across corpora)."""
    return hashlib.sha1(json.dumps(data.get("commands", []),
                                   separators=(",", ":")).encode("utf-8")).hexdigest()


def quarantine_reason(starting_sides) -> Optional[str]:
    """Why a replay is kept out of the imitation corpus: a player side
    with shroud. The simulator does not model shroud (unexplored terrain
    hidden until seen), and none of our games use it (user ruling
    2026-09-30: 0.5% of the corpus is not worth the edge case)."""
    if any(bool(s.get("shroud", False)) for s in _player_sides(starting_sides)):
        return "shroud"
    return None


def era_factions_of(era_id: Optional[str]) -> Tuple[str, ...]:
    """The factions a Random choice can draw in the record's era; a record
    without an era is the default era's. An era the table lacks keeps the
    default era's factions, with a warning: the corpus plays only the
    eras it lists."""
    from wesnoth_ai.constants import DEFAULT_ERA_FACTIONS, ERA_FACTIONS
    if not era_id:
        return DEFAULT_ERA_FACTIONS
    factions = ERA_FACTIONS.get(era_id)
    if factions is None:
        log.warning("era %r is not in constants.ERA_FACTIONS; a Random side's prior uses the "
                    "default era's factions", era_id)
        return DEFAULT_ERA_FACTIONS
    return factions


def _build_initial_gamestate(data: dict) -> GameState:
    """The record's starting state: its map, its sides, and its starting
    units as the core builds them (`game_core.build_unit`, a leader with
    its traits)."""
    from wesnoth_ai.game_core import build_unit
    raw_map = data.get("map_data", "")
    hexes = set(parse_map_data(raw_map))
    terrain_codes = parse_terrain_codes(raw_map)
    game_id = data.get("game_id", "")
    exp_mod = int(data.get("experience_modifier", 100) or 100)
    units = {
        build_unit(u, apply_leader_traits=True, game_id=game_id, exp_modifier=exp_mod)
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
            chose_random=bool(s.get("chose_random", False)),
        )
        for s in data.get("starting_sides", [])
    ]
    current_side = 1
    size_x = max((h.position.x for h in hexes), default=0) + 1
    size_y = max((h.position.y for h in hexes), default=0) + 1

    # Wesnoth stores `village_gold` (gold per village per turn) and
    # `village_support` (free upkeep per village) per [side]. In MP
    # they're always equal across sides (they're set by the host's
    # game options, not per-player). Read from side[0]'s extracted
    # values; fall back to vanilla defaults (1 / 1) only if the
    # replay header didn't have them at all (very old extractions).
    starting_sides_data = data.get("starting_sides", [])
    if starting_sides_data:
        first_side = starting_sides_data[0]
        village_gold = first_side.get("village_income", 2)
        village_support = first_side.get("village_support", 1)
    else:
        village_gold = 2
        village_support = 1
    # The turn-1 slot, wrapped the engine's way (`fix_time_index`) into
    # the default schedule, the only one the simulator models (a
    # scenario declaring another is flagged by
    # `wml_state.check_board_cycle`): the Rust core indexes its cycle
    # with this value and panics on a negative one.
    tod_start = fix_time_index(len(cb.TOD_DEFAULT_CYCLE),
                               int(data.get("tod_start_index", 0) or 0))
    gs = GameState(
        game_id=data.get("game_id", "?"),
        map=Map(size_x=size_x, size_y=size_y,
                mask=set(), fog=set(),
                hexes=hexes, units=units),
        global_info=GlobalInfo(
            current_side=current_side, turn_number=0,
            time_of_day=_tod_for_turn(1, tod_start),
            village_gold=village_gold,
            village_upkeep=village_support, base_income=2,
        ),
        sides=sides,
        era_factions=era_factions_of(data.get("era_id")),
        random_faction_mode=str(data.get("random_faction_mode") or "Independent"),
    )
    # Stash the raw replay metadata for tools that need pixel-exact
    # round-tripping (the save-state dumper uses these to avoid
    # re-rendering the map and to recover side leader-types). These
    # attributes are not part of the encoder contract — purely a
    # pass-through for tools that have the original .json.gz on hand.
    setattr(gs.global_info, "_raw_map_data", data.get("map_data", ""))
    setattr(gs.global_info, "_terrain_codes", terrain_codes)
    # Terrain epoch: reach-planner cache key that survives deepcopy
    # (MCTS forks share entries) and is BUMPED by terrain-morph
    # events (see pathfind_sim._terrain_maps_for).
    from tools.pathfind_sim import next_terrain_epoch
    setattr(gs.global_info, "_terrain_epoch", next_terrain_epoch())
    # ToD start offset for random_start_time scenarios. 0 means turn-1
    # is dawn (the default 2p case). Other values shift the cycle so
    # that turn-1 reads as e.g. afternoon (offset=2) — matching the
    # server-side `tod_manager::resolve_random` decision recorded in
    # the replay's [scenario] / [replay_start] `current_time` attr.
    setattr(gs.global_info, "_tod_start_offset", tod_start)
    setattr(gs.global_info, "_raw_starting_sides",
            list(data.get("starting_sides", [])))
    # wesnoth_ai.visibility reads it: the encoder hides enemy units
    # outside the mover's sight only in fog games (18.9% of the corpus
    # was played fog-off, tabulated 2026-09-06).
    setattr(gs.global_info, "_fog", fog_on_for(data.get("starting_sides", [])))
    delaying = frozenset(int(s["side"]) for s in data.get("starting_sides", [])
                         if not s.get("auto_shroud", True))
    if delaying:
        setattr(gs.global_info, delayed_shroud.SHROUD_DELAYED, delaying)
    if data.get("plan_unit_advance"):
        setattr(gs.global_info, delayed_shroud.PLAN_UNIT_ADVANCE, True)
    setattr(gs.global_info, "_scenario_id", data.get("scenario_id", ""))
    setattr(gs.global_info, "_experience_modifier", exp_mod)
    # The sides beyond the players that take turns and those that never
    # do, the two sets the scenario builder reads from its [side]
    # controllers: the simulator's side order reads them when it
    # continues this game.
    acting, silent = extra_side_turns(data)
    if silent:
        setattr(gs.global_info, "_null_controller_sides", silent)
    if acting:
        setattr(gs.global_info, "_neutral_actor_sides", acting)
    # Wesnoth's monotonic next_unit_id counter: increments on every
    # unit creation (recruit, plague spawn, scenario [unit] event;
    # advancement keeps the unit's uid) and never decrements on death.
    # Initialized to (max starting uid) + 1, Wesnoth's post-prestart
    # state.
    initial_max_uid = 0
    for u in units:
        if u.id.startswith("u") and u.id[1:].isdigit():
            initial_max_uid = max(initial_max_uid, int(u.id[1:]))
    setattr(gs.global_info, "_next_uid_counter", initial_max_uid + 1)

    # Pre-owned villages from the replay's [side]/[village] children
    # (or scenario-pool's _village_owner). Apply BEFORE turn-1 income
    # would be computed so the income/upkeep math at the first
    # init_side(turn>1) sees the right village count. Wesnoth source:
    # team::team(const config&) at wesnoth_src/src/team.cpp:208-217.
    starting_villages = data.get("starting_villages", []) or []
    owner_map: Dict[Tuple[int, int], int] = {}
    side_increments: Dict[int, int] = {}
    for v in starting_villages:
        try:
            vx = int(v["x"])
            vy = int(v["y"])
            vs = int(v["side"])
        except (KeyError, ValueError, TypeError):
            continue
        if vs <= 0 or vs > len(gs.sides):
            continue
        owner_map[(vx, vy)] = vs
        side_increments[vs] = side_increments.get(vs, 0) + 1
    if owner_map:
        # Ownership is carried ONLY by `_village_owner` (below); the
        # encoder derives its owned-village bit from it. We deliberately
        # do NOT stamp TerrainModifiers.VILLAGE on the hexes -- Hex
        # objects are aliased across MCTS forks (Map.__deepcopy__), and
        # the modifier-as-ownership-cache pattern is what caused the
        # 2026-07-29 fork-isolation leak.
        # Bump nb_villages_controlled per side.
        for sn, n in side_increments.items():
            old = gs.sides[sn - 1]
            gs.sides[sn - 1] = dataclasses.replace(
                old, nb_villages_controlled=old.nb_villages_controlled + n)
        # The owner map: a later capture of these hexes moves
        # ownership instead of crediting it twice.
        setattr(gs.global_info, "_village_owner", owner_map)
    return gs


def _terrain_keys_at(gs: GameState, x: int, y: int) -> List[str]:
    """Return the WML defense-table keys to evaluate for the hex at
    (x,y). Honors Wesnoth's `aliasof=` semantics: a Ford (Wwf) returns
    ['shallow_water', 'flat'] so callers can pick whichever defense is
    best for the unit standing on it.

    NOTE: callers that need authoritative defense_pct should use
    `_terrain_def_pct(gs, x, y, def_table)` directly -- it walks the
    terrain alias graph via terrain_resolver.def_pct, which handles
    the FULL set of overlay codes (^Fms, ^Fp, etc.). This function
    survives because some callers want a list-of-string keys for
    encoder features and trait overrides."""
    codes_dict = getattr(gs.global_info, "_terrain_codes", {}) or {}
    code = codes_dict.get((x, y))
    if not code:
        return ["flat"]
    return _defense_keys_for_code(code)


def _tod_cycle_index(turn_number: int, start_offset: int = 0) -> int:
    """Compute the cycle index (0..5) for `turn_number` given a starting
    offset. `start_offset` defaults to 0 (turn 1 = dawn). For replays
    with `random_start_time=yes` resolved server-side, the offset
    encodes which ToD the server picked. The readers hand over an
    offset already in range (`wml_state.read_tod`,
    `_build_initial_gamestate`); the wrap is the engine's modulo
    (`tod_manager::calculate_time_index_at_turn`), as the time-area
    path in `_lawful_bonus_at` wraps, never a clamp to dawn."""
    return (max(1, turn_number) - 1 + start_offset) % len(cb.TOD_DEFAULT_CYCLE)


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


# The [illuminates] ability's value and max_value, both 25 in
# `{ABILITY_ILLUMINATES}` (data/core/macros/abilities.cfg:232-236), the
# only definition of it in the default era. The Rust core keeps it as
# `ILLUMINATION` (core_attack.rs); tests/test_rust_constants.py compares.
ILLUMINATES_VALUE = 25


def apply_unit_illumination(base: int, illuminated: bool) -> int:
    """`bounded_add(base, 25, max_sum=25, min_sum=0)`'s positive branch
    (tod_manager.cpp:265-281): the [illuminates] ability on top of the
    terrain-lit time of day, `min(base + 25, max(base, 25))`."""
    if not illuminated:
        return base
    return min(base + ILLUMINATES_VALUE, max(base, ILLUMINATES_VALUE))


def illuminated_lawful_bonus_at(gs: GameState, unit: Unit, turn: int) -> int:
    """The lawful bonus the engine's `get_illuminated_time_of_day`
    gives a unit's own hex: the time area or default cycle, the
    terrain light (`_lawful_bonus_at`) and an [illuminates] unit on
    the hex or next to it (`abilities.illuminate_step`). What combat
    reads for both combatants and what a [hides] filter reads for
    nightstalk (abilities.cpp:447-450 evaluates it with
    use_flat_tod=false, filter.cpp:268-273)."""
    from tools.abilities import illuminate_step
    base = _lawful_bonus_at(gs, unit.position.x, unit.position.y, turn)
    return apply_unit_illumination(base, illuminate_step(unit, gs.map.units) > 0)


def side_income(gs: GameState, side: int) -> Tuple[int, int]:
    """(income, net upkeep) that `side` is paid at a turn start, a
    direct port of play_controller.cpp:524-534:
      income  = base_income + villages_owned * village_gold
      upkeep  = sum(unit.level for the side's non-loyal units)
      support = villages_owned * village_support
      net upkeep = max(0, upkeep - support)
    The engine reports the first as a side's `total_income`."""
    s = gs.sides[side - 1]
    owned = s.nb_villages_controlled
    village_gold, village_support = village_economy(gs.global_info)
    income = s.base_income + owned * village_gold
    upkeep = 0
    for u in gs.map.units:
        if u.side != side:
            continue
        # Leaders never contribute to upkeep, regardless of whether they
        # have the `loyal` trait (src/units/unit.cpp:1746-1751,
        # `unit::upkeep` short-circuits on `can_recruit()`).
        if u.is_leader or "loyal" in u.traits:
            continue
        upkeep += int(_stats_for(u.name).get("level", 1))
    return income, max(0, upkeep - owned * village_support)


def _lawful_bonus_at(gs: GameState, x: int, y: int, turn_number: int) -> int:
    """Per-hex lawful_bonus. Honors scenario-defined [time_area] zones
    (Tombs of Kesorak's dark/illuminated regions, Elensefar Courtyard's
    underground keeps, etc.) — those override the global ToD cycle on
    their hexes with a cycle of their own, whose slot does not follow a
    random start of the board's. Falls back to the default 6-step cycle,
    shifted by the board's start slot, when no [time_area] applies.

    On top of the base ToD, applies terrain-level light bonus per
    `terrain.hpp:132`: the campfire overlay (^Ecf), wallfire (^Efs),
    icicle (^Ii), eldritch fire (^Ebn) and similar "lit" overlays
    add +25 to lawful_bonus and CLAMP via max_light / min_light --
    e.g. ^Ecf with max=min=25 fixes the hex's lawful_bonus to exactly
    25 regardless of base ToD. Without this, a Poacher on Rrc^Ecf
    at Tombs of Kesorak's illuminated zone takes +25% chaotic-night
    damage instead of -25% chaotic-day, killing units that should
    survive (witnessed in 2p__Tombs_of_Kesorak_Turn_*_(208025) at
    cmd[138]: Poacher retal-bow dmg should be 3 (4*0.75) but our
    sim computed 4 (4*1.0), the cumulative drift over later attacks
    killed the Dark Adept which Wesnoth keeps alive).
    """
    start_offset = int(getattr(gs.global_info, "_tod_start_offset", 0) or 0)
    areas = getattr(gs.global_info, "_time_areas", None)
    cycle = areas.get((x, y)) if areas else None
    if cycle:
        # An area keeps its own slot, stored phased to turn 1 (the
        # core's [time_area] action); the board's start slot
        # moves the board's cycle only (`tod_manager::resolve_random`).
        base = int(cycle[(max(1, turn_number) - 1) % len(cycle)])
    else:
        base = _lawful_bonus_for_turn(turn_number, start_offset)
    # Apply terrain light_bonus: bounded_add(base, light, max_light,
    # min_light). The result is the unit's effective lawful_bonus.
    codes = getattr(gs.global_info, "_terrain_codes", {}) or {}
    code = codes.get((x, y))
    if code:
        from wesnoth_ai.rules.terrain_resolver import terrain_light_bonus
        return terrain_light_bonus(strip_start_position(code), base)
    return base


def _rebuild_unit(unit: Unit, **changes) -> Unit:
    """Return a NEW Unit copy of `unit` with `**changes` applied to
    the dataclass fields, preserving any `_`-prefixed setattr stash
    (e.g. `_defense_table`). Pure -- doesn't touch any GameState.

    Why a separate helper from `_replace_unit`: spawn paths build a
    fresh Unit (recruit, plague corpse) and then need to apply
    final-state overrides (e.g. `current_moves=0`, `has_attacked=True`)
    while keeping the FRESHLY-built stash. _replace_unit is for the
    in-place modify-existing case (`old` is already on the map and
    its stash should carry forward). Both go through this helper to
    consolidate the "preserve _-stash" pattern that several open-
    coded sites used to repeat verbatim.
    """
    base_fields = {
        k: v for k, v in unit.__dict__.items() if not k.startswith("_")
    }
    new = Unit(**{**base_fields, **changes})
    for k, v in unit.__dict__.items():
        if k.startswith("_"):
            setattr(new, k, v)
    return new


def move_order_of(cmd: list) -> Optional[dict]:
    """The order a compact move carries beside its path when the engine
    stopped the unit short of the hex the player clicked:
    {"clicked": [x, y] (0-indexed), "stopped_early": bool or None, and
    "next": [x, y], the route's hex after the stop, when there is one}
    (`replay_extract.extract_replay` writes it). None for a move that
    went where it was ordered, and for every move of a record extracted
    before the field existed."""
    if len(cmd) > 4 and isinstance(cmd[4], dict):
        return cmd[4]
    return None


def move_label_hex(gs: GameState, unit: Unit, cmd: list) -> Tuple[Tuple[int, int], str]:
    """The hex a move's label names, and why: (hex, source).

    The label is the player's choice. The path's end is where the unit
    stopped, which differs from the hex clicked when the engine cut the
    move short (`move_order_of`); the simulator does not stop a move on
    sighting an enemy, so in our games the unit heads for the clicked
    hex, and the label names it:
      "path_end"   the move went where it was ordered (or the record
                   predates orders): the path's end;
      "turn_end"   the engine says the unit reached the end of this
                   turn's part of the order (`stopped_early` no: a
                   multi-turn order, or an end hex another unit held):
                   the path's end, which is the turn's target;
      "clicked"    stopped early, and the clicked hex is one this unit
                   can end a move on in the pre-move observable state
                   (the legality mask's move targets): the clicked hex;
      "clicked_unreachable"  stopped early, but the clicked hex is not
                   such a target (an order longer than a turn): the path's
                   end, where the unit stopped.
    """
    stop = (cmd[1][-1], cmd[2][-1])
    order = move_order_of(cmd)
    if order is None:
        return stop, "path_end"
    if order.get("stopped_early") is False:
        return stop, "turn_end"
    clicked = (int(order["clicked"][0]), int(order["clicked"][1]))
    from tools.pathfind_sim import ReachContext, unit_reach
    reach = unit_reach(unit, gs, ReachContext.for_side(gs, unit.side))
    if clicked in reach.landable:
        return clicked, "clicked"
    return stop, "clicked_unreachable"


def _action_indices(gs: GameState, cmd: list, *,
                    relevant_set: bool = False,
                    stats: Optional[Counter] = None) -> Optional[ActionIndices]:
    """Convert a compact replay command into slot indices the model's
    heads should predict.

    Returns None for commands that aren't player policy actions
    (init_side, etc.). Actor/target ordering MATCHES the encoder's
    sort: units sorted by (y, x, id), then recruits.

    `relevant_set`: target_idx indexes `relevant_hexes_in_slot_order`
    (the basis `encode_raw(relevant_set=True)` emits) instead of the
    full board. An on-board target with no subset slot keeps the
    pair and flags it (`target_off_subset`); an off-board target
    drops the pair exactly as in the full-board basis, so both bases
    yield the same pair stream.

    `stats`, when given, counts each move label's source
    (`move_label_hex`).
    """
    if not cmd or cmd[0] not in PAIRED_KINDS:
        return None
    kind = cmd[0]

    # Slot contract: units / recruits / hexes all come from the
    # SHARED enumeration in visibility.py -- the same functions the
    # encoder builds its tokens from. (The previous hand-mirrored
    # copy silently rotted when the encoder became fog-filtered,
    # mislabeling 19%+ of behavior-cloning pairs; root-caused and
    # de-mirrored 2026-07-16.)
    from wesnoth_ai.visibility import (hexes_in_slot_order, own_recruit_types,
                            relevant_hexes_in_slot_order,
                            visible_units_in_slot_order)
    current_side = gs.global_info.current_side
    units_sorted = visible_units_in_slot_order(gs, current_side)
    hex_positions = [h.position for h in hexes_in_slot_order(gs)]
    pos_to_hex_idx = {(p.x, p.y): i for i, p in enumerate(hex_positions)}
    if relevant_set:
        subset_idx = {(h.position.x, h.position.y): i for i, h
                      in enumerate(relevant_hexes_in_slot_order(gs))}

    def _target(x: int, y: int) -> Tuple[Optional[int], bool, bool]:
        """(target_idx, on_board, off_subset) for a target hex."""
        if (x, y) not in pos_to_hex_idx:
            return None, False, False
        if not relevant_set:
            return pos_to_hex_idx[(x, y)], True, False
        j = subset_idx.get((x, y))
        return j, True, j is None

    if kind == "end_turn":
        # Last actor slot = end_turn sentinel.
        end_idx = len(units_sorted) + len(
            own_recruit_types(gs, current_side))
        return ActionIndices("end_turn", actor_idx=end_idx)

    if kind == "move":
        xs, ys = cmd[1], cmd[2]
        sx, sy = xs[0], ys[0]
        # Find actor = the unit at (sx, sy).
        actor = None
        for i, u in enumerate(units_sorted):
            if u.position.x == sx and u.position.y == sy and u.side == current_side:
                actor = i
                break
        if actor is None:
            return None
        (tx, ty), source = move_label_hex(gs, units_sorted[actor], cmd)
        target, on_board, off_subset = _target(tx, ty)
        if not on_board:
            return None
        if stats is not None:
            stats[f"move_label_{source}"] += 1
        # type_idx=1 (MOVE) for the action-type head; lazy import
        # of model.UnitActionType to avoid a hard dep cycle (model
        # already imports replay_dataset transitively via the
        # encoder).
        from wesnoth_ai.model import UnitActionType
        return ActionIndices("move", actor_idx=actor, target_idx=target,
                             type_idx=UnitActionType.MOVE,
                             target_off_subset=off_subset,
                             source_hex=(sx, sy), target_hex=(tx, ty))

    if kind == "attack":
        ax, ay, dx, dy, weapon = cmd[1], cmd[2], cmd[3], cmd[4], cmd[5]
        actor = None
        for i, u in enumerate(units_sorted):
            if u.position.x == ax and u.position.y == ay and u.side == current_side:
                actor = i
                break
        if actor is None:
            return None
        target, on_board, off_subset = _target(dx, dy)
        if not on_board:
            return None
        from wesnoth_ai.model import UnitActionType
        return ActionIndices("attack", actor_idx=actor,
                             target_idx=target, weapon_idx=weapon,
                             type_idx=UnitActionType.ATTACK,
                             target_off_subset=off_subset,
                             source_hex=(ax, ay), target_hex=(dx, dy))

    if kind == "recruit":
        unit_type = cmd[1]
        tx, ty = cmd[2], cmd[3]
        # Recruit actor slots follow the visible units (slot
        # contract: own_recruit_types).
        actor = None
        for j, r_name in enumerate(own_recruit_types(gs, current_side)):
            if r_name == unit_type:
                actor = len(units_sorted) + j
                break
        if actor is None:
            return None
        target, on_board, off_subset = _target(tx, ty)
        if not on_board:
            return None
        return ActionIndices("recruit", actor_idx=actor, target_idx=target,
                             target_off_subset=off_subset,
                             target_hex=(tx, ty), recruit_type=unit_type)

    # recall / init_side / unknown → skip.
    return None


# ---------------------------------------------------------------------
# Public Iterator
# ---------------------------------------------------------------------

# Scenario ids whose WML was not found, each warned about once per process.
_SCENARIOS_WITHOUT_WML: set = set()


def _warn_scenario_without_wml(scenario_id: str) -> None:
    """Say once per id that a scenario's WML is missing: the game then
    runs without its events, time areas and side modifications."""
    if scenario_id in _SCENARIOS_WITHOUT_WML:
        return
    _SCENARIOS_WITHOUT_WML.add(scenario_id)
    log.warning(f"no scenario WML found for {scenario_id!r}: the game runs "
                f"without its events, time areas and side modifications")


# The command kinds that are a player's decision and must each yield a
# pair; a recall (not in the action space) and the mod's pick-advance
# input yield none by design.
PAIRED_KINDS = frozenset({"move", "attack", "recruit", "end_turn"})


def iter_replay_pairs(gz_path: Path, *, relevant_set: bool = False,
                      stats: Optional[Counter] = None, timeouts: bool = False
                      ) -> Iterator[Tuple[GameState, ActionIndices]]:
    """Yield (state_before, action_indices) for each command a player
    side (1 or 2) made in one .json.gz replay; the neutral side's are
    its AI's, not a player's to imitate. `relevant_set` selects the
    label's hex basis (see `_action_indices`); it must match the
    encoder's.

    `stats`, when given, receives this file's counts: each move label's
    source (`move_label_hex`) and `unpaired`, the player commands of a
    `PAIRED_KINDS` kind that yielded no pair -- the record and its
    reconstruction disagree on an actor or a target, so the game lost a
    decision. A file with any is logged as a warning.

    `timeouts`: also yield, where a turn ran out of time, the position the
    player was deciding in with the `TIMEOUT` label (`timeout_label`),
    which names no action: the policy gets no target there, the rest of
    the network does."""
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    counts: Counter = Counter()
    yield from iter_record_pairs(data, relevant_set=relevant_set, stats=counts, timeouts=timeouts)
    if counts["unpaired"]:
        log.warning(f"{Path(gz_path).name}: {counts['unpaired']} player commands "
                    f"yielded no pair")
    if stats is not None:
        stats.update(counts)


def iter_record_pairs(data: dict, *, relevant_set: bool = False,
                      stats: Optional[Counter] = None, timeouts: bool = False
                      ) -> Iterator[Tuple[GameState, ActionIndices]]:
    """`iter_replay_pairs` over an extracted record already in memory,
    replayed on the Rust core (`record_core`): each pair's state is a
    view of its own fork of the core."""
    from wesnoth_ai.game_core import view_of
    cs = record_core(data)
    engine = engine_issued_of(data)
    for i, cmd in enumerate(data.get("commands", [])):
        if i in engine:
            _count_engine_issued(stats, engine[i])
            if timeouts and engine[i] == TIMEOUT:
                yield view_of(cs.fork()), timeout_label()
        elif int(cs.core.current_side) in PLAYER_SIDES:
            gs = view_of(cs.fork())
            ai = _action_indices(gs, cmd, relevant_set=relevant_set, stats=stats)
            if ai is not None:
                yield gs, ai
            elif stats is not None and cmd and cmd[0] in PAIRED_KINDS:
                stats["unpaired"] += 1
        cs.apply_command(list(cmd))


def engine_issued_of(data: dict) -> Dict[int, str]:
    """The commands of a record that the engine made under a player side
    (index -> "goto" or "timeout", tools/replay_engine_actions.py): applied
    to the state, never paired as the player's decision (a timeout's
    position can be, with the TIMEOUT label: `iter_record_pairs`). Empty
    for records extracted before the field existed (extraction version 3)."""
    marks = data.get("engine_issued") or {}
    return {int(i): kind for kind, idx in marks.items() for i in idx}


def _count_engine_issued(stats: Optional[Counter], kind: str) -> None:
    if stats is not None:
        stats[f"engine_{kind}"] += 1


# The label of a position whose turn ran out of time while its player was
# still deciding: no action (the policy has no target there and can never
# pick it); the position still counts for everything else.
TIMEOUT = "timeout"


def timeout_label() -> ActionIndices:
    return ActionIndices(action_type=TIMEOUT, actor_idx=-1)


def record_core(data: dict):
    """The Rust core of an extracted record's initial state, its
    scenario set up (`CoreState.setup_scenario`)."""
    from wesnoth_ai.game_core import CoreState
    cs = CoreState.from_state(_build_initial_gamestate(data))
    cs.setup_scenario(data.get("scenario_id", ""))
    return cs


def iter_replay_pairs_with_state(gz_path: Path
                                 ) -> Iterator[Tuple[GameState, Optional[ActionIndices]]]:
    """Like iter_replay_pairs but yields the state before EVERY command
    (including init_side / recall), each a view of its own fork of the
    core, and the FINAL state after the last command. Useful for tools
    that want to inspect or dump the state at any point in the replay
    (e.g. save-state dumper)."""
    from wesnoth_ai.game_core import view_of
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    cs = record_core(data)
    for cmd in data.get("commands", []):
        gs = view_of(cs.fork())
        yield gs, _action_indices(gs, cmd)
        cs.apply_command(list(cmd))
    yield view_of(cs), None


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
    from wesnoth_ai.rules.scenarios import is_competitive_2p

    PLAYER_FACTIONS = {"Drakes", "Knalgan Alliance", "Rebels",
                       "Loyalists", "Northerners", "Undead"}
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"replay dataset {dataset_dir} does not exist")
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
        print("usage: replay_dataset.py DATASET_DIR")
        return 2
    d = Path(argv[1])
    n_files = 0
    n_pairs = 0
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
