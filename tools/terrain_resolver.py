"""Runtime resolver for Wesnoth terrain movement / defense costs.

Reads `terrain_db.json` (built by `tools/scrape_terrain.py`) and
provides:

  - `mvt_cost(code, costs)` — movement cost for a unit with the
    given `costs` dict (keyed by canonical terrain ids: castle,
    cave, deep_water, flat, forest, frozen, fungus, hills,
    impassable, mountains, reef, sand, shallow_water, swamp_water,
    unwalkable, village).
  - `def_pct(code, defenses)` — defense percentage (chance to be hit)
    for a unit with the given `defenses` dict, same key set.

Both functions accept composite codes like `Gs^Vhs` or `Wwt^Bw/`.

Algorithm
---------
Mirrors `wesnoth_src/src/movetype.cpp` `terrain_info::data::calc_value`
(lines 276-369 in 1.18.4) and the composite construction in
`wesnoth_src/src/terrain/terrain.cpp:208-244` + `merge_alias_lists`
(lines 334-377). See `docs/wesnoth_rules.md` for the full explanation.

  1. Get the alias list for `code`:
     - For terminal (`Mm`, `Ut`, etc.): just `[code]`.
     - For aliased non-overlay (`Wwf`): the entry's `mvt_type`.
     - For composite (`Gs^Vhs`): merge_alias_lists(overlay.mvt_type,
       base.mvt_type). The merge replaces the `_bas` token with the
       base's list, with PLUS/MINUS markers around it depending on
       the revert state from preceding markers.

  2. If the resulting list is `[code]` (Wesnoth's `is_indivisible`
     check), look up the cost / defense via the entry's `id` in
     the unit's costs / defenses table.

  3. Otherwise iterate the list with PLUS/MINUS markers controlling
     `prefer_high`. Recursively resolve each terrain code in the list,
     aggregating MIN (default) or MAX (after MINUS marker, until next
     PLUS) over the per-code resolved values.

Defaults
--------
Movement: `min_value=1, max_value=UNREACHABLE`, `high_is_good=false`,
`default_value=max_value`. So with no aliases (terminal not in costs)
the cost is UNREACHABLE.

Defense: `min_value=0, max_value=100`, `high_is_good=false` for the
"prefer-MIN" variant our combat path uses. Default 0.

Dependencies: stdlib (json, pathlib)
Dependents: tools.wesnoth_sim (replaces _move_cost_at_hex internals)
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Mapping, Optional

from wesnoth_ai.paths import TERRAIN_DB_PATH


log = logging.getLogger("terrain_resolver")


# Marker tokens (must match scrape_terrain.MARKER_*).
MARKER_PLUS  = "+"
MARKER_MINUS = "-"
MARKER_BASE  = "_bas"

# Sentinel values matching Wesnoth's mvj_params_ defaults.
# wesnoth_src/src/movetype.cpp:81: mvj_params_{1, movetype::UNREACHABLE}.
# UNREACHABLE is defined as a large int (~256 in Wesnoth source).
# We use 99 which our existing code already treats as "impassable".
UNREACHABLE_COST = 99


# ---------------------------------------------------------------------
# DB loading
# ---------------------------------------------------------------------

_TERRAIN_DB: Optional[Dict[str, dict]] = None


def load_terrain_db() -> Dict[str, dict]:
    """Lazy-load `terrain_db.json` from the project root. Raises on
    missing file -- the file is checked into the repo and rebuilt
    via `tools/scrape_terrain.py` whenever terrain.cfg changes."""
    global _TERRAIN_DB
    if _TERRAIN_DB is not None:
        return _TERRAIN_DB
    if not TERRAIN_DB_PATH.exists():
        raise FileNotFoundError(
            f"{TERRAIN_DB_PATH.name} not found at project root. "
            f"Run `python tools/scrape_terrain.py` to (re)build it.")
    _TERRAIN_DB = json.loads(TERRAIN_DB_PATH.read_text(encoding="utf-8"))
    return _TERRAIN_DB


# ---------------------------------------------------------------------
# The terrain classes a hex belongs to, from the engine's own aliases
# ---------------------------------------------------------------------

# Every abstract terrain the pinned 1.18.4 scrape aliases a code to
# (the closed set over terrain_db.json's aliasof / mvt_alias /
# def_alias lists, 17 letters), mapped onto the encoder's Terrain
# classes. Wesnoth defines a hex's movement and defense by these
# aliases (a ford is Gt AND Wst, a forested hill Ht AND Ft), so they
# are the hex's terrain SET, and what the encoder should carry. Three
# abstracts have no class of their own in the 14-member enum and take
# the nearest: reef (Wrt) shallow water, rails (Rt) flat, fungus (Tt)
# cave. `_bas` names the base under an overlay, `+`/`-` are the
# best-of / worst-of markers; neither is a class.
ALIAS_TO_TERRAIN_NAME = {
    "Gt": "FLAT", "Rt": "FLAT", "Ht": "HILLS", "Mt": "MOUNTAINS", "Ft": "FOREST",
    "Wst": "SHALLOWWATER", "Wrt": "SHALLOWWATER", "Wdt": "DEEPWATER", "St": "SWAMP",
    "Dt": "SAND", "At": "FROZEN", "Ut": "CAVE", "Tt": "CAVE", "Xt": "IMPASSABLE",
    "Qt": "UNWALKABLE", "Vt": "VILLAGE", "Ct": "CASTLE",
}
_ALIAS_MARKERS = {MARKER_PLUS, MARKER_MINUS, MARKER_BASE}
# Terminal terrains no movetype prices (`_off^_usr` off-map and `^_fme`
# the fake map edge, data/core/terrain.cfg): the engine's cost lookup
# misses and falls back to UNREACHABLE for every unit, so their class
# is impassable although no alias says so.
_IMPASSABLE_TERMINAL_IDS = {"off_map", "off_map2"}
_WARNED_TERRAIN_CODES: set = set()


def _alias_lists(code: str, db: Dict[str, dict]) -> List[List[str]]:
    """The movement and defense alias lists the engine gives `code`:
    a code terrain.cfg defines outright (`Mm^Xm`, aliasof=-,Mt,Xt) is
    found before any base/overlay merge, as the engine's terrain map
    finds it; anything else merges as `_get_underlying` does."""
    entry = db.get(code)
    if entry is not None:
        return [list(entry["mvt_type"]), list(entry["def_type"])]
    return [_get_underlying(code, kind, db) for kind in ("mvt_type", "def_type")]


def terrain_members(code: str):
    """The Terrain classes of one hex code (a set), from the movement
    and defense aliases the terrain database resolves it to, overlays
    merged the way the engine merges them. A code the database does
    not know (or an alias outside ALIAS_TO_TERRAIN_NAME) is logged
    once and contributes nothing: a hex with no member is visible to
    the encoder as a fallback, never as a silent default."""
    from wesnoth_ai.classes import Terrain
    code = strip_start_position(code or "")
    if not code:
        return set()
    db = load_terrain_db()
    aliases = set()
    for lst in _alias_lists(code, db):
        for a in lst:
            if a not in _ALIAS_MARKERS:
                aliases.add(a)
    members = set()
    for a in aliases:
        name = ALIAS_TO_TERRAIN_NAME.get(a)
        if name is None and (db.get(a) or {}).get("id") in _IMPASSABLE_TERMINAL_IDS:
            name = "IMPASSABLE"
        if name is None:
            if code not in _WARNED_TERRAIN_CODES:
                _WARNED_TERRAIN_CODES.add(code)
                log.warning("terrain code %r resolves to %r, outside the terrain classes; "
                            "the hex carries no class for it", code, a)
            continue
        members.add(Terrain[name])
    return members


def terrain_mask(code: str) -> int:
    """`terrain_members` as a bitmask over Terrain values (bit v set
    for member v), the form the encoder's hex stream carries."""
    mask = 0
    for t in terrain_members(code):
        mask |= 1 << int(t.value)
    return mask


# ---------------------------------------------------------------------
# Composite alias merge
# ---------------------------------------------------------------------

def merge_alias_lists(first: List[str], second: List[str]) -> List[str]:
    """Mirror of `wesnoth_src/src/terrain/terrain.cpp:334-377`.

    Walks `first` looking for the BASE marker, removes it, and splices
    `second` in at that position. Adds a PLUS or MINUS marker after
    the splice depending on the revert state -- this preserves the
    "prefer-low until MINUS, prefer-high after" semantics of the
    composite alias list.

    Used for composite terrain (e.g. `Gs^Vhs`): start with the
    overlay's mvt_type, merge in the base's mvt_type at the
    overlay's `_bas` placeholder.
    """
    result = list(first)
    revert = bool(result) and result[0] == MARKER_MINUS
    i = 0
    while i < len(result):
        tok = result[i]
        if tok == MARKER_PLUS:
            revert = False
            i += 1
            continue
        if tok == MARKER_MINUS:
            revert = True
            i += 1
            continue
        if tok == MARKER_BASE:
            # Erase BASE.
            result.pop(i)
            # Insert PLUS or MINUS at the now-vacant position to
            # preserve the marker state for tokens AFTER the splice.
            marker = MARKER_MINUS if revert else MARKER_PLUS
            result.insert(i, marker)
            # Insert second's tokens BEFORE the marker we just placed.
            for j, s_tok in enumerate(second):
                result.insert(i + j, s_tok)
            break
        i += 1
    return result


# ---------------------------------------------------------------------
# Underlying mvt / def list extraction
# ---------------------------------------------------------------------

def _get_underlying(code: str, kind: str, db: Dict[str, dict]) -> List[str]:
    """Return the mvt_type or def_type alias list for `code`,
    handling composite codes via merge_alias_lists.

    `kind` is "mvt_type" or "def_type" -- selects which alias list
    to use. Wesnoth's `terrain_type` ctor is symmetric for the two:
    same merge logic, just different list values per terrain entry.
    """
    if "^" in code:
        base_str, overlay_str = code.split("^", 1)
        overlay_full = "^" + overlay_str
        base_entry = db.get(base_str)
        overlay_entry = db.get(overlay_full)
        if not base_entry or not overlay_entry:
            # Unknown composite -- treat as terminal; runtime cost
            # will fall back to default (UNREACHABLE for movement).
            log.debug(f"unknown composite terrain {code!r}: "
                      f"base_known={base_entry is not None}, "
                      f"overlay_known={overlay_entry is not None}")
            return [code]
        return merge_alias_lists(
            list(overlay_entry[kind]),
            list(base_entry[kind]),
        )
    entry = db.get(code)
    if not entry:
        return [code]
    return list(entry[kind])


def _is_indivisible(code: str, underlying: List[str]) -> bool:
    """Wesnoth's `is_indivisible` (terrain.hpp:100): list is just
    [code] => terminal terrain, look up cost directly via id."""
    return len(underlying) == 1 and underlying[0] == code


def _terminal_cost(code: str, costs: Mapping[str, int],
                   db: Dict[str, dict], default: int) -> int:
    """Resolve a terminal terrain to its cost via id lookup.
    Returns abs() of stored value -- negative values mean "min cap"
    semantics, handled by `_collect_neg_caps` separately."""
    entry = db.get(code)
    if not entry:
        return default
    terrain_id = entry["id"]
    val = costs.get(terrain_id, default)
    return abs(int(val))


def _collect_neg_caps(code: str, costs: Mapping[str, int],
                      db: Dict[str, dict],
                      kind: str, recurse: int = 0) -> int:
    """Walk the same alias list as `_calc_value` and return the
    largest abs() value among NEGATIVE entries in `costs`. These
    are 'min cap' floors per Wesnoth's [defense] negative-value
    convention (movetype.cpp::config_to_min). Returns 0 if no
    negative entries are matched.

    Concrete: feral Vampire Bat on Gg^Vc. Aliases: Gt(flat=40),
    Vt(village=-50). flat is positive, village is -50 (cap).
    abs(-50)=50 is the floor. After computing alias-min over
    positives (40), we max with 50 -> def_pct = 50."""
    if recurse > 100:
        return 0
    underlying = _get_underlying(code, kind, db)
    if _is_indivisible(code, underlying):
        entry = db.get(code)
        if not entry:
            return 0
        terrain_id = entry["id"]
        val = costs.get(terrain_id, 0)
        return abs(int(val)) if int(val) < 0 else 0
    best = 0
    for tok in underlying:
        if tok in (MARKER_PLUS, MARKER_MINUS):
            continue
        cap = _collect_neg_caps(tok, costs, db, kind, recurse + 1)
        if cap > best:
            best = cap
    return best


def _calc_value(code: str, costs: Mapping[str, int],
                db: Dict[str, dict], *,
                kind: str,
                default: int, max_value: int, min_value: int,
                high_is_good: bool, recurse: int = 0) -> int:
    """Generic Wesnoth `terrain_info::data::calc_value` port.

    `kind`: "mvt_type" or "def_type" -- which alias list to walk.
        For `^Fms` they differ: mvt_type=['-','_bas','Ft'] (PREFER
        MAX cost over base+forest), def_type=['_bas','Ft'] (PREFER
        MIN, best defense). Hardcoding either one was the bug -- a
        Spearman on Gg^Fms was getting flat-defense=60 instead of
        forest-defense=50 because the mvt_type list flipped the
        prefer-direction.
    `default`: starting result when no aliases match (= max_value
        for movement, min_value for defense-MIN computation).
    `max_value` / `min_value`: clamps applied at the end and used
        when flipping default after a leading MINUS marker.
    `high_is_good`: false for movement (low cost is good) and for
        the prefer-LOW defense path. Markers flip the local
        `prefer_high` state.

    Recursion guard at depth 100 mirrors the source.
    """
    if recurse > 100:
        log.warning(f"infinite alias recursion at {code!r}")
        return default

    underlying = _get_underlying(code, kind, db)
    if _is_indivisible(code, underlying):
        return _terminal_cost(code, costs, db, default)

    # Aliased.
    prefer_high = high_is_good
    result = default
    if underlying and underlying[0] == MARKER_MINUS:
        # Per wesnoth_src/src/movetype.cpp:341-344: if list starts with
        # MINUS, the default flips (so MAX-of-aliases starts from 1
        # instead of UNREACHABLE, etc.).
        result = min_value if result == max_value else max_value

    for tok in underlying:
        if tok == MARKER_PLUS:
            prefer_high = high_is_good
            continue
        if tok == MARKER_MINUS:
            prefer_high = not high_is_good
            continue
        num = _calc_value(
            tok, costs, db,
            kind=kind,
            default=default, max_value=max_value, min_value=min_value,
            high_is_good=high_is_good, recurse=recurse + 1,
        )
        if (prefer_high and num > result) or (not prefer_high and num < result):
            result = num

    # Clamp.
    if result < min_value:
        result = min_value
    if result > max_value:
        result = max_value
    return result


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def mvt_cost(code: str, costs: Mapping[str, int]) -> int:
    """Movement cost for entering hex with terrain `code`.

    `costs`: dict keyed by terminal terrain ids (castle, cave,
    deep_water, flat, forest, frozen, fungus, hills, impassable,
    mountains, reef, sand, shallow_water, swamp_water, unwalkable,
    village). Missing keys are treated as UNREACHABLE_COST.
    Typically supplied as `unit_stats.json`'s
    `movement_types[movetype]['movement_costs']`.
    """
    db = load_terrain_db()
    return _calc_value(
        code, costs, db,
        kind="mvt_type",
        default=UNREACHABLE_COST,
        max_value=UNREACHABLE_COST,
        min_value=1,
        high_is_good=False,
    )


def def_pct(code: str, defenses: Mapping[str, int]) -> int:
    """Defense percentage (chance-to-be-hit) for a unit on `code`.

    `defenses`: dict keyed by terminal terrain ids. Missing keys
    default to 0 (worst possible defense for the prefer-LOW path).

    Note: defense in Wesnoth is "chance to be hit" -- LOWER is
    better. So `high_is_good=False`. The default-min variant we
    use here corresponds to `terrain_defense::params_max_` in
    movetype.cpp:86, which is the BEST-defense computation (lowest
    hit chance picked across aliases).
    """
    db = load_terrain_db()
    base = _calc_value(
        code, defenses, db,
        kind="def_type",
        default=100,             # worst possible (always hit) before any alias resolves
        max_value=100,
        min_value=0,
        high_is_good=False,
    )
    # Honor negative-value caps (min floors on def_pct). For each
    # negative entry in `defenses` whose terrain id appears in this
    # code's alias list, the resulting def_pct must be at least that
    # absolute value. See _collect_neg_caps for the rule.
    neg_cap = _collect_neg_caps(code, defenses, db, kind="def_type")
    if neg_cap > base:
        return neg_cap
    return base


def terrain_heals(code: str) -> int:
    """Per-turn healing the hex at `code` provides (+8 for villages
    and oasis, 0 elsewhere).

    Mirrors `wesnoth_src/src/terrain/terrain.cpp:230` for composite
    terrain: `heals_(std::max<int>(base.heals_, overlay.heals_))`.
    For non-composite terrain it's the entry's own `heals` value.

    The `^Do` (oasis) overlay is the non-village case: heals=8 but
    NOT a village (no capture, no income, no team ownership).
    Wesnoth's `heal.cpp` calls `map().gives_healing(loc)` which
    returns this exact int -- the same path is used for the +HP
    branch AND the poison-cure branch, so an oasis cures poison
    just like a village.
    """
    db = load_terrain_db()
    base, _, overlay = code.partition("^")
    base_heals = int((db.get(base) or {}).get("heals", 0) or 0)
    if overlay:
        overlay_key = "^" + overlay
        overlay_heals = int((db.get(overlay_key) or {}).get("heals", 0) or 0)
    else:
        overlay_heals = 0
    return max(base_heals, overlay_heals)


def terrain_light_bonus(code: str, base: int) -> int:
    """Apply the hex's terrain-level illumination to `base` lawful_bonus.

    Mirrors `wesnoth_src/src/terrain/terrain.hpp:132`:
        light_bonus(base) = bounded_add(base, light_modification_,
                                        max_light_, min_light_)

    Composite (base+overlay) per terrain.cpp:230:
        light_modification = base.light + overlay.light
        max_light = max(base.max_light, overlay.max_light)
        min_light = min(base.min_light, overlay.min_light)

    Wesnoth's bounded_add (utils/math.hpp:38) is ASYMMETRIC:

        if (increment >= 0)  return min(base + increment,
                                        max(base, max_sum));
        else                 return max(base + increment,
                                        min(base, min_sum));

    Positive light (lava chasm Ql, campfire ^Ecf, etc.): ADDS to base
    capped UPWARD at max_sum but never reduces base below itself. So
    Ql (light=25, max_light=35) at base=-25 first_watch yields
    bounded_add(-25, 25, 35, …) = min(0, max(-25, 35)) = min(0, 35)
    = 0 (NOT 25 -- the lava brightens by one step, doesn't fix to 25).
    Whereas ^Ecf (light=25, max_light=25) at base=0 dawn yields
    bounded_add(0, 25, 25, …) = min(25, max(0, 25)) = 25 (the
    max_light=25 cap forces the result to exactly 25 here, but only
    because the base+light=25 is at the cap).

    Negative light (^Edp dim/poison?, etc.): SUBTRACTS from base
    floored DOWNWARD at min_sum but never raises base above itself.

    Composite default: when min_light or max_light isn't explicitly
    set in the cfg, Wesnoth defaults each to `light` (so no min/max
    cfg means light==max==min, fully fixing to base+light). Our scrape
    captures `has_max_light` / `has_min_light` flags but currently
    only the magnitude needs differ -- the bounded_add itself encodes
    the cap correctly.

    Returns `base` unchanged when the terrain has no light/max/min.
    """
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
    if light == 0 and max_l == 0 and min_l == 0:
        return base
    # bounded_add: asymmetric per Wesnoth's utils/math.hpp:38.
    if light >= 0:
        return min(base + light, max(base, max_l))
    else:
        return max(base + light, min(base, min_l))


# ---------------------------------------------------------------------
# Starting positions: the prefix a map cell may carry
# ---------------------------------------------------------------------

def split_start_position(cell: str) -> tuple:
    """(label, code) of a map cell: "1 Gg^Fp" -> ("1", "Gg^Fp"),
    "Gg^Fp" -> ("", "Gg^Fp").

    The engine's rule lives in `string_to_number_`
    (wesnoth_src/src/terrain/translation.cpp:743-756, 1.18.4 tag):

        // Strip the spaces around us
        // unlike the old implementation this also trims newlines.
        utils::trim(str);
        ...
        // Split if we have spaces inside
        std::size_t offset = str.find(' ', 0);
        while(offset != std::string::npos) {
            start_positions.push_back(std::string(str.substr(0, offset)));
            str.remove_prefix(offset + 1);
            offset = str.find(' ', 0);
        }

    Trim first, then cut everything up to the LAST space. A label is
    any text before a space, not one digit: the engine keeps labels as
    STRINGS (`starting_positions` is a string-keyed bimap, filled by
    read_game_map at translation.cpp:317-327). So "10 Wo" is side 10's
    hex, and the named labels mainline ships -- "lake Gs^Vc" in
    data/test/maps/simple_find_path.map, "book_start Isc^Ii" in
    campaigns/Descent_Into_Darkness/maps/07c_A_Small_Favor3.map -- are
    labels too. Trimming before the split is what makes " 1 Gg" and
    "1 " come out as the engine has them ("Gg" and "1").

    A cell may carry several labels (the engine pushes one per space);
    `label` keeps the prefix as written and `label.split()` recovers
    that list. `f"{label} {code}"` rebuilds the cell, which is how
    `number_to_string_` (translation.cpp:775-782) writes it back.
    """
    label, sep, code = (cell or "").strip().rpartition(" ")
    return (label if sep else ""), code


def strip_start_position(code: str) -> str:
    """The terrain code of a map cell, its starting-position label
    dropped: "1 Gg^Fp" -> "Gg^Fp". See `split_start_position`."""
    return split_start_position(code)[1]


def start_position_side(label: str) -> Optional[int]:
    """The side whose start `label` marks, or None when the label names
    a location rather than a side.

    `gamemap_base::starting_position` looks the label up by the side's
    own spelling (wesnoth_src/src/map/map.cpp:324-327, 1.18.4 tag):

        map_location gamemap_base::starting_position(int n) const
        {
            return special_location(std::to_string(n));
        }

    So only the canonical decimal of a side matches: "10" is side 10,
    while "lake", "P1_Burner" and "01" are special locations that no
    side ever asks for.
    """
    label = label or ""
    return int(label) if label.isdecimal() and str(int(label)) == label else None


# Hide-ability cover, straight from the engine's own filters
# (wesnoth_src/data/core/macros/abilities.cfg:280-382). Wesnoth does
# NOT ask what a hex defends like; it matches the hex's terrain CODE
# against a glob list:
#
#   ambush       terrain=*^F*,*^Qhhf,*^Qhuf   (any forest overlay, plus
#                                              the bluff/glutch fungus)
#   concealment  terrain=*^V*                 (any village overlay)
#   submerge     terrain=Wo*^*                (any deep-water base)
#   nightstalk   time_of_day=chaotic          (no terrain at all)
#
# Deciding cover from the DEFENSE keys instead was wrong on every code
# a hand-rolled overlay table did not list: measured over the shipped
# maps -- Wesnoth's 66 core multiplayer maps, every cell including
# the 1-hex border -- 18.9% of forest-overlay hexes (Gs^Fms, Hh^Fms,
# Gs^Ftd, ...) and 25.3% of village-overlay hexes (Gg^Ve, Gs^Vht,
# Aa^Vha, ...) silently
# provided no cover at all.
_AMBUSH_OVERLAY_EXACT = ("Qhhf", "Qhuf")


def _split_code(code: str) -> tuple:
    """(base, overlay) of a terrain code, starting-position prefix
    stripped. Overlay is "" when the code has none."""
    base, sep, overlay = strip_start_position(code).partition("^")
    return base, (overlay if sep else "")


def hides_cover(code: str, ability: str) -> bool:
    """Does the hex `code` give `ability` its cover?

    A transcription of the `[hides]` `[filter_location]` terrain globs
    in abilities.cfg (see the table above). `nightstalk` has no terrain
    condition and is not answered here -- its cover is the time of day.
    """
    base, overlay = _split_code(code)
    if ability == "ambush":                      # *^F*,*^Qhhf,*^Qhuf
        return overlay.startswith("F") or overlay in _AMBUSH_OVERLAY_EXACT
    if ability == "concealment":                 # *^V*
        return overlay.startswith("V")
    if ability == "submerge":                    # Wo*^*
        return base.startswith("Wo")
    return False


__all__ = [
    "MARKER_PLUS", "MARKER_MINUS", "MARKER_BASE", "UNREACHABLE_COST",
    "load_terrain_db", "merge_alias_lists",
    "mvt_cost", "def_pct", "terrain_heals", "terrain_light_bonus",
    "hides_cover",
    "split_start_position", "strip_start_position", "start_position_side",
]
