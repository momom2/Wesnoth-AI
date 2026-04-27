"""Fog of war + invisibility for replay-derived training data.

Default-era 2p maps all run with `fog=yes` per side. The reconstructor
builds an omniscient state (units of every side fully visible) — that's
what we want for combat verification, but it leaks information when we
train: the model would learn to plan around enemy units the player
couldn't actually see.

This module computes per-side visibility and invisibility on a
reconstructed `GameState`:

    visible_hexes(gs, side)       → Set[(x, y)]  — every hex side N can see
    visible_units_for_side(gs, n) → Set[Unit]    — enemies that survive the
                                                    fog + invisibility checks

The encoder calls these at the moment it builds a training tensor for
side N's POV. For omniscient debugging / save-state dumps, just don't
call them.

Visibility model (mirrors Wesnoth 1.18 src/pathfind/teleport.cpp +
display::clear_hex_overlay):

  - A hex is visible to side N if it's within `vision` steps of any
    unit on side N (allies via team_name not modeled; PvP 2p has each
    side on its own team).
  - "Vision steps" are hex-graph BFS, no terrain cost (Wesnoth
    technically uses `vision_costs` but most movetypes have cost 1
    everywhere off-map; default-era unit-types don't tune these).
  - The unit's own hex is visible. Vision radius `R` includes hexes at
    BFS distance ≤ R.
  - Vision does NOT change with ToD in default Wesnoth (a 1.16
    addition gated to specific unit types).

Invisibility (mirrors src/units/abilities.cpp::invisible):

  - A unit with `submerge` ability is invisible while on `deep_water`
    AND no enemy is adjacent.
  - `concealment` (Cuttlefish-like) is invisible on `village` AND no
    enemy adjacent.
  - `ambush` (Wose, Elvish Ranger…) is invisible on `forest` AND no
    enemy adjacent.
  - `nightstalk` (Nightgaunt…) is invisible at NIGHT (lawful_bonus<0)
    AND no enemy adjacent.

  An "enemy adjacent" reveals the hidden unit to ALL enemies (Wesnoth
  fires `unit_uncovered` once and the unit becomes visible to everyone
  on the opposing side until the start of its next turn). For the
  encoder we collapse that to a per-state check.

Dependencies: classes, tools.abilities, tools.replay_dataset (for
              terrain queries — passed in by caller)
Dependents:   training-time encoder (not yet wired up here; plain API)
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Set, Tuple

from classes import GameState, Unit
from tools.abilities import hex_neighbors


# Used by the encoder to filter out the acting side's enemies that
# they wouldn't see. Imported lazily by callers that don't need fog
# (e.g. combat verification).


def _unit_vision(u: Unit, unit_db: Dict[str, dict]) -> int:
    """Vision radius for `u`. Reads the scraped `vision` field, which
    defaults to the unit's `movement` per Wesnoth's unit_type ctor.
    Falls back to 5 if neither is in the DB (custom-era safety)."""
    stats = unit_db.get(u.name, {})
    return int(stats.get("vision", stats.get("moves", 5)) or 5)


def visible_hexes(gs: GameState, side: int,
                  unit_db: Dict[str, dict]) -> Set[Tuple[int, int]]:
    """Hexes within sight range of any unit on `side`. Empty set means
    the entire map is fogged (e.g., side has no units left).

    BFS from each unit's position, expanding `vision` steps. Off-map
    hexes are pruned via the gs.map.size_x / size_y bounds.
    """
    seen: Set[Tuple[int, int]] = set()
    max_x, max_y = gs.map.size_x, gs.map.size_y
    for u in gs.map.units:
        if u.side != side:
            continue
        radius = _unit_vision(u, unit_db)
        # BFS up to depth `radius` (inclusive). Standard hex BFS.
        start = (u.position.x, u.position.y)
        if start in seen:
            # Optimization: already covered by another unit's wider
            # vision is rare; still cheap to BFS again.
            pass
        frontier: deque = deque([(start, 0)])
        local_seen = {start}
        while frontier:
            (x, y), d = frontier.popleft()
            if d >= radius:
                continue
            for nx, ny in hex_neighbors(x, y):
                if not (0 <= nx < max_x and 0 <= ny < max_y):
                    continue
                if (nx, ny) in local_seen:
                    continue
                local_seen.add((nx, ny))
                frontier.append(((nx, ny), d + 1))
        seen |= local_seen
    return seen


# Terrain-based invisibility predicates. Each takes the raw terrain
# string from `_terrain_at` (e.g. "forest", "village") and returns
# True if that terrain triggers the ability.
_INVISIBILITY_TERRAIN_FOR_ABILITY: Dict[str, Tuple[str, ...]] = {
    "submerge":    ("deep_water",),
    "concealment": ("village",),
    "ambush":      ("forest",),
    # `nightstalk` is time-based, not terrain — handled separately.
}


def _has_enemy_adjacent(u: Unit, all_units: Iterable[Unit]) -> bool:
    """True iff any unit of a different side stands on a hex adjacent
    to `u`. Wesnoth defines "enemy" by team_name, but PvP 2p has every
    player on their own team so side != side suffices."""
    nbrs = set(hex_neighbors(u.position.x, u.position.y))
    for o in all_units:
        if o.side == u.side or o is u:
            continue
        if (o.position.x, o.position.y) in nbrs:
            return True
    return False


def is_invisible(u: Unit, gs: GameState, terrain: str,
                 lawful_bonus_at: int) -> bool:
    """True iff `u` is invisible to enemies right now.

    `terrain` is the WML defense key for u's hex (e.g., 'forest').
    `lawful_bonus_at` is the lawful_bonus on u's hex this turn —
    nightstalk activates whenever this is < 0 (i.e. first/second
    watch under default cycle, or any time_area override that sums
    to night). Adjacency to ANY enemy reveals the unit.
    """
    abilities = u.abilities
    if not abilities:
        return False
    # Adjacency reveal: short-circuit if an enemy is adjacent.
    enemy_near = _has_enemy_adjacent(u, gs.map.units)

    for ab, terrains in _INVISIBILITY_TERRAIN_FOR_ABILITY.items():
        if ab in abilities and terrain in terrains:
            return not enemy_near

    # Nightstalk: invisible at night regardless of terrain.
    if "nightstalk" in abilities and lawful_bonus_at < 0:
        return not enemy_near

    return False


def visible_units_for_side(
    gs: GameState,
    side: int,
    unit_db: Dict[str, dict],
    *,
    terrain_keys_at,
    lawful_bonus_at,
) -> Set[Unit]:
    """Return the units `side` can see right now.

    Always-visible: own units (side N), regardless of terrain.
    Conditionally visible: enemy units whose hex is within vision AND
    that aren't currently invisible (submerge / concealment / ambush /
    nightstalk + no enemy adjacent).

    `terrain_keys_at(gs, x, y)` and `lawful_bonus_at(gs, x, y, turn)`
    are passed in (rather than imported) to avoid a hard dependency
    on tools.replay_dataset, keeping this module testable in
    isolation.
    """
    visible_pos = visible_hexes(gs, side, unit_db)
    out: Set[Unit] = set()
    turn = gs.global_info.turn_number
    for u in gs.map.units:
        if u.side == side:
            out.add(u)
            continue
        # Off-map / never-seen hex.
        pos = (u.position.x, u.position.y)
        if pos not in visible_pos:
            continue
        # Invisibility check: pick the most informative terrain key
        # (in alias order, FOREST > VILLAGE > DEEP_WATER) since the
        # ability triggers on the FIRST matching key.
        keys = terrain_keys_at(gs, u.position.x, u.position.y)
        # Try invisibility against each terrain alias; if any matches
        # and reveals invisibility, the unit is hidden.
        hidden = False
        lb = lawful_bonus_at(gs, u.position.x, u.position.y, turn)
        for k in keys:
            if is_invisible(u, gs, k, lb):
                hidden = True
                break
        # Nightstalk doesn't depend on terrain, so check it once if
        # we haven't matched a terrain-based ability above.
        if not hidden and "nightstalk" in u.abilities:
            if is_invisible(u, gs, "", lb):
                hidden = True
        if not hidden:
            out.add(u)
    return out


__all__ = [
    "visible_hexes",
    "visible_units_for_side",
    "is_invisible",
]
