"""Hex adjacency, and the leadership ability's bonus.

Wesnoth's hexes are flat-topped and laid out in columns, every odd
column (0-indexed) half a hex lower than its neighbours, so which hexes
touch depends on the column's parity. `hex_neighbors` gives the six
neighbours in the order N, NE, SE, S, SW, NW, and `opposite_hex` the
hex across a unit from one of them. On that geometry:

  - Leadership (`leadership_bonus`): an adjacent same-side unit with
    `leadership` and a HIGHER level adds 25% x (its level - the unit's
    level) to the unit's damage. Several leaders do not stack, the best
    one counts, and the opponent's level plays no part; a petrified
    unit projects none. The swap detector's leadership screen reads it.

The Rust core applies every ability in play
(rust/wesnoth_core/src/core_attack.rs, core_step.rs).

Dependencies: classes
Dependents:   tools.replay_dataset, wesnoth_sim, neutral_ai,
              swap_detector; wesnoth_ai.observe, rewards (the geometry)
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from wesnoth_ai.classes import Unit


# ----------------------------------------------------------------------
# Hex geometry — flat-top, "odd-q" offset (matching Wesnoth 1.18)
# ----------------------------------------------------------------------

# Odd columns (0-indexed, as the engine's own map_location) sit half a
# hex lower than even ones, so the neighbour pattern depends on the
# column's parity: map_location::get_direction's NORTH_EAST is
# `map_location(x + n, y - (n+is_even(x))/2 )` (src/map/location.cpp:391,
# 1.18.4). Reference:
# https://wiki.wesnoth.org/Coordinates_in_Wesnoth
def hex_neighbors(x: int, y: int) -> List[Tuple[int, int]]:
    """Return the 6 neighbors of (x, y) in flat-top odd-q layout."""
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

def _adjacent_units(units: Iterable[Unit], x: int, y: int) -> List[Unit]:
    """The units of `units` on a hex next to (x, y)."""
    pos = set(hex_neighbors(x, y))
    return [u for u in units if (u.position.x, u.position.y) in pos]


def leadership_bonus(unit: Unit, all_units: Iterable[Unit],
                     opponent_level: int = 0) -> int:
    """Return the leadership-based damage bonus % for `unit`'s attacks.

    Wesnoth's ABILITY_LEADERSHIP (data/core/macros/abilities.cfg):
        [leadership]
            value="(25 * (level - other.level))"
            cumulative=no
            affect_self=no
            [affect_adjacent]
                [filter] formula="level < other.level" [/filter]
            [/affect_adjacent]
        [/leadership]

    The English description on the same macro:
      "All adjacent lower-level units from the same side deal 25%
       more damage for each difference in level."

    The "difference in level" is between the LEADER and the BUFFED
    UNIT — NOT between the leader and the opponent. Verified
    2026-05-03 by user GUI replay of
    2p__Hornshark_Island_Turn_12_(112807).bz2 cmd[96]: a Mage
    (lvl 1) adjacent to a Lieutenant (lvl 2) attacking a Vampire
    Bat (lvl 0) gets +25% (= 25 × (2-1)), NOT +50% (= 25 × (2-0)).

    Rules:
      - Bonus = 25 × (leader.level − buffed_unit.level).
      - Buffed unit must be ADJACENT to the leader (not self),
        same side, and STRICTLY LOWER level than the leader (per
        the [filter] formula).
      - `cumulative=no`: multiple adjacent leaders DON'T stack;
        only the highest-bonus leader applies (= MAX over
        candidates).
      - The opponent's level is irrelevant to the bonus.
        `opponent_level` parameter retained for backward-compat
        with existing callers but unused.
    """
    from tools.replay_dataset import _stats_for
    # `int(stats.get("level", 1) or 1)` was coercing level-0 units
    # to 1 because 0 is falsy in Python. That broke leadership for
    # level-0 buffed units: a Woodsman (level 0) adjacent to a
    # Lieutenant (level 2) should get +50% damage (= 25 * (2 - 0)),
    # but the bug computed 25 * (2 - 1) = +25%. Caused Hornshark
    # cmd[147] Woodsman retaliation against a Thief to deal 5 dmg
    # instead of 6 -- saving the Thief at hp 1 and blocking
    # downstream side-1 moves.
    def _lvl(s):
        try:
            return int(s.get("level", 1))
        except (TypeError, ValueError):
            return 1
    unit_level = _lvl(_stats_for(unit.name))
    best = 0
    for ally in _adjacent_units(all_units, unit.position.x, unit.position.y):
        if ally.side != unit.side or ally.id == unit.id:
            continue
        # Petrified/incapacitated leaders project nothing: get_abilities
        # skips adjacent units where `it->incapacitated()`
        # (abilities.cpp:~150; incapacitated() includes petrified). See
        # docs/wesnoth_rules.md.
        if "petrified" in ally.statuses:
            continue
        if "leadership" not in ally.abilities:
            continue
        ally_level = _lvl(_stats_for(ally.name))
        # Lower-level filter: the buffed unit (`unit`) must be
        # STRICTLY lower-level than the leader.
        if unit_level >= ally_level:
            continue
        # Bonus = 25 × (leader.level − buffed_unit.level).
        bonus = 25 * (ally_level - unit_level)
        if bonus > best:
            best = bonus
    return best


__all__ = ["hex_neighbors", "opposite_hex", "leadership_bonus"]
