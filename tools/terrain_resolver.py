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
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Mapping, Optional


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


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TERRAIN_DB_PATH = _PROJECT_ROOT / "terrain_db.json"


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
    if not _TERRAIN_DB_PATH.exists():
        raise FileNotFoundError(
            f"{_TERRAIN_DB_PATH.name} not found at project root. "
            f"Run `python tools/scrape_terrain.py` to (re)build it.")
    _TERRAIN_DB = json.loads(_TERRAIN_DB_PATH.read_text(encoding="utf-8"))
    return _TERRAIN_DB


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


__all__ = [
    "MARKER_PLUS", "MARKER_MINUS", "MARKER_BASE", "UNREACHABLE_COST",
    "load_terrain_db", "merge_alias_lists",
    "mvt_cost", "def_pct", "terrain_heals", "terrain_light_bonus",
]
