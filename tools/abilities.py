"""Adjacency-based ability evaluators for replay reconstruction.

Combat outcomes and turn-start healing both depend on which units are
adjacent to whom. Wesnoth uses pointy-top offset hexes — adjacency is
direction-dependent on column parity. This module centralizes that
geometry plus the bookkeeping for the four most gameplay-significant
adjacency abilities:

  - Leadership : adjacent same-side ally with leadership[N] gives the
                 attacker +25%·(N − defender_level) damage when N >
                 defender's level. Caps at +25% × Lmax.
  - Illuminates: adjacent allies (any side same team) treat ToD as one
                 step lighter (lawful_bonus shifted up by +25 capped at
                 +25; chaotic effect inverted).
  - Healers    : `heals=N` adjacent same-side ally heals N HP/turn at
                 init_side, capped at 8 total per healed unit. `cures`
                 also clears poison.
  - Backstab   : melee weapon special; defender is flanked when an
                 enemy of the defender stands on the opposite hex from
                 the attacker (six-hex check).

Dependencies: classes
Dependents:   tools.replay_dataset, combat (via flag passing)
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Set, Tuple

from classes import Unit


# ----------------------------------------------------------------------
# Hex geometry — pointy-top, "odd-q" offset (matching Wesnoth 1.18)
# ----------------------------------------------------------------------

# In Wesnoth, even-x columns are LOW (slightly higher rendered), odd-x
# are HIGH. The neighbor pattern depends on column parity. Reference:
# https://wiki.wesnoth.org/Coordinates_in_Wesnoth
def hex_neighbors(x: int, y: int) -> List[Tuple[int, int]]:
    """Return the 6 neighbors of (x, y) in pointy-top odd-q layout."""
    if x % 2 == 0:
        # even column
        return [
            (x,     y - 1),  # N
            (x + 1, y - 1),  # NE
            (x + 1, y    ),  # SE
            (x,     y + 1),  # S
            (x - 1, y    ),  # SW
            (x - 1, y - 1),  # NW
        ]
    else:
        # odd column — neighbors shift down
        return [
            (x,     y - 1),  # N
            (x + 1, y    ),  # NE
            (x + 1, y + 1),  # SE
            (x,     y + 1),  # S
            (x - 1, y + 1),  # SW
            (x - 1, y    ),  # NW
        ]


def opposite_hex(center: Tuple[int, int],
                 neighbor: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """Given the defender's hex `center` and an attacker on `neighbor`
    (which must be one of the 6 neighbors), return the hex on the
    opposite side of `center` from `neighbor` — i.e. the hex a flanker
    would stand on for backstab."""
    cx, cy = center
    nx, ny = neighbor
    neighbors = hex_neighbors(cx, cy)
    try:
        idx = neighbors.index((nx, ny))
    except ValueError:
        return None
    # Opposite is the +3 index in the 6-hex ring.
    return neighbors[(idx + 3) % 6]


# ----------------------------------------------------------------------
# Ability scanners
# ----------------------------------------------------------------------

def _units_at(units: Iterable[Unit], x: int, y: int) -> List[Unit]:
    return [u for u in units if u.position.x == x and u.position.y == y]


def _adjacent_units(units: Iterable[Unit], x: int, y: int) -> List[Unit]:
    pos = set(hex_neighbors(x, y))
    return [u for u in units if (u.position.x, u.position.y) in pos]


def is_backstab_active(attacker: Unit, defender: Unit,
                       all_units: Iterable[Unit]) -> bool:
    """Backstab is active if the hex opposite the attacker (relative to
    the defender) is occupied by a unit that's an enemy of the defender
    (not necessarily the same side as the attacker; shared-team flank
    qualifies). Excludes incapacitated/petrified flankers."""
    opp = opposite_hex(
        (defender.position.x, defender.position.y),
        (attacker.position.x, attacker.position.y),
    )
    if opp is None:
        return False
    flanker = next(
        (u for u in all_units
         if (u.position.x, u.position.y) == opp
         and u.side != defender.side),
        None,
    )
    return flanker is not None


def leadership_bonus(unit: Unit, all_units: Iterable[Unit],
                     opponent_level: int) -> int:
    """Return the leadership-based damage bonus % for `unit`'s attacks.

    Wesnoth's ABILITY_LEADERSHIP (data/core/macros/abilities.cfg:192):
        [leadership]
            value="(25 * (level - other.level))"
            cumulative=no
            affect_self=no
            [affect_adjacent]
                [filter] formula="level < other.level" [/filter]
            [/affect_adjacent]
        [/leadership]

    Rules:
      - Bonus = 25 × (leader.level − opponent.level), where opponent
        is the unit being attacked.
      - Buffed unit must be ADJACENT to the leader (not self), same
        side, and STRICTLY LOWER level than the leader. The
        lower-level filter is the `[filter] formula="level <
        other.level"` part: same-level units are NOT buffed.
      - `cumulative=no`: multiple adjacent leaders DON'T stack;
        only the highest-level leader's bonus applies. Since the
        bonus is monotonically increasing in leader.level, taking
        the MAX over candidates is equivalent (lvl3 General gives
        +50% vs lvl1 opp, lvl2 Lt only +25%, max = 50%).
      - We don't track per-ability `level=` overrides; assume the
        leadership ability's level equals the unit's level. True for
        all default-era leaders (Lieutenant, Drake Flare, etc.).
    """
    # Unit doesn't carry `level` directly; look it up from unit_stats.
    # Lazy import to avoid circular deps.
    from tools.replay_dataset import _stats_for
    unit_level = int(_stats_for(unit.name).get("level", 1) or 1)
    best = 0
    for ally in _adjacent_units(all_units, unit.position.x, unit.position.y):
        if ally.side != unit.side or ally.id == unit.id:
            continue
        if "leadership" not in ally.abilities:
            continue
        ally_level = int(_stats_for(ally.name).get("level", 1) or 1)
        # Lower-level filter: the buffed unit (`unit`) must be
        # STRICTLY lower-level than the leader.
        if unit_level >= ally_level:
            continue
        # Bonus formula. Skip if it would be <= 0 (leader same-or-
        # lower level than the opponent gives no bonus).
        if ally_level <= opponent_level:
            continue
        best = max(best, 25 * (ally_level - opponent_level))
    return best


def illuminate_step(unit: Unit, all_units: Iterable[Unit]) -> int:
    """Return +1 if `unit` is illuminated (self or adjacent ally has
    `illuminates`), else 0. Used to bump lawful_bonus by 25 at dusk
    or similar — caller multiplies."""
    if "illuminates" in unit.abilities:
        return 1
    for ally in _adjacent_units(all_units, unit.position.x, unit.position.y):
        if ally.side == unit.side and "illuminates" in ally.abilities:
            return 1
    return 0


def healer_heal_amount(unit: Unit, all_units: Iterable[Unit]) -> int:
    """Return total heal-per-turn this unit gets from adjacent healers
    (capped at 8 per Wesnoth's rules — multiple healers don't stack
    above 8; cures heals up to 8 too but additionally cures poison)."""
    if "regenerate" in unit.abilities:
        return 0  # regenerate is self-healing, handled separately
    best = 0
    for ally in _adjacent_units(all_units, unit.position.x, unit.position.y):
        if ally.side != unit.side:
            continue
        if "cures" in ally.abilities or "heals_8" in ally.abilities or "heals+8" in ally.abilities:
            best = max(best, 8)
        elif "heals_4" in ally.abilities or "heals+4" in ally.abilities:
            best = max(best, 4)
    return best


def adjacent_curer(unit: Unit, all_units: Iterable[Unit]) -> bool:
    """Return True if any adjacent same-side unit has `cures` (clears
    poison at init_side)."""
    for ally in _adjacent_units(all_units, unit.position.x, unit.position.y):
        if ally.side != unit.side:
            continue
        if "cures" in ally.abilities:
            return True
    return False


__all__ = [
    "hex_neighbors", "opposite_hex",
    "is_backstab_active", "leadership_bonus", "illuminate_step",
    "healer_heal_amount", "adjacent_curer",
]
