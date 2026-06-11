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

def _units_at(units: Iterable[Unit], x: int, y: int,
              pos_index: Optional[dict] = None) -> List[Unit]:
    if pos_index is not None:
        return list(pos_index.get((x, y), ()))
    return [u for u in units if u.position.x == x and u.position.y == y]


def _adjacent_units(units: Iterable[Unit], x: int, y: int,
                    pos_index: Optional[dict] = None) -> List[Unit]:
    """If `pos_index` (a `(x, y) → list[Unit]` dict over `units`) is
    provided, use O(6) hex-neighbor lookups against it. Otherwise fall
    back to an O(N_units) listcomp.

    Callers that issue multiple adjacency queries against the same
    units snapshot — `init_side`'s healing loop is the prominent
    example, with 2 queries × N units per side per turn — should
    build the index once with `build_pos_index(units)` and pass it
    in. The index is unsafe to memoize across `_apply_command`
    boundaries because `_replace_unit` mutates `gs.map.units`
    in-place (discard + add), so an id()-keyed memo would silently
    serve stale data after any move/attack/recruit. Passing it
    explicitly keeps the lifetime narrow and verifiable.
    """
    if pos_index is not None:
        out: List[Unit] = []
        for nx, ny in hex_neighbors(x, y):
            out.extend(pos_index.get((nx, ny), ()))
        return out
    pos = set(hex_neighbors(x, y))
    return [u for u in units if (u.position.x, u.position.y) in pos]


def build_pos_index(units: Iterable[Unit]) -> dict:
    """Build `(x, y) → list[Unit]` over `units`. O(N) one-time;
    caller is responsible for not reusing the index after the unit
    set mutates."""
    out: dict = {}
    for u in units:
        out.setdefault((u.position.x, u.position.y), []).append(u)
    return out


def is_backstab_active(attacker: Unit, defender: Unit,
                       all_units: Iterable[Unit]) -> bool:
    """Backstab is active if the hex opposite the attacker (relative to
    the defender) is occupied by a unit that's an enemy of the defender
    (not necessarily the same side as the attacker; shared-team flank
    qualifies). Excludes incapacitated/petrified flankers.

    Concretely, Wesnoth's `[backstab]` special filters petrified
    flankers (`wesnoth_src/data/core/macros/weapon_specials.cfg`).
    Witnessed in 2p__Sullas_Ruins_Turn_37_(214794).bz2 cmd[420]:
    Thief side 2 attacks Dwarvish Thunderer side 1 at (18,10), with
    a petrified Yeti statue side 3 at (19,10) standing on the
    opposite hex. Without the petrified filter, our sim activated
    backstab (doubling dagger 5->10 dmg, 3 hits = 24 dmg, dropping
    Thunderer 36->12). Wesnoth keeps backstab INACTIVE because the
    statue is petrified, so 3 hits at 5 dmg = 15 dmg, Thunderer at
    21 — surviving subsequent attacks where our sim killed it."""
    opp = opposite_hex(
        (defender.position.x, defender.position.y),
        (attacker.position.x, attacker.position.y),
    )
    if opp is None:
        return False
    flanker = next(
        (u for u in all_units
         if (u.position.x, u.position.y) == opp
         and u.side != defender.side
         and "petrified" not in u.statuses),
        None,
    )
    return flanker is not None


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


def illuminate_step(unit: Unit, all_units: Iterable[Unit]) -> int:
    """Return +1 if `unit`'s hex is illuminated (self or any adjacent
    unit has `illuminates`), else 0. Used to bump lawful_bonus by 25
    at dusk or similar — caller multiplies.

    NB: illumination is a TERRAIN-LIGHT modifier, not an ally-only
    aura. Per `tod_manager::get_illuminated_time_of_day`
    (tod_manager.cpp:237-262), Wesnoth scans all 7 hexes (loc + 6
    adjacent) and contributes light from ANY unit with the
    `illuminates` ability, regardless of side. Filtering by
    `ally.side == unit.side` (the previous behavior) makes our sim
    skip the enemy-cast illumination that boosts the attacker's
    lawful_bonus when they strike INTO the illuminated hex.
    Witnessed 2026-05-08 in 2p__Hamlets_Turn_20_(41655) cmd[731]:
    Mage of Light (side 2) at (15,21) illuminates the surrounding
    area; the side-1 Merman Netcaster striking from (16,21) is
    lawful and should fight at first_watch+illumination = dusk
    (0% modifier) instead of first_watch (-25%). Without this fix
    Netcaster's club at 7×3 stays at 5 dmg/hit (lawful -25%) → 3
    hits = 15 dmg max which DOES kill u46 only if all 3 land; with
    the +25 lawful boost from u46's own illumination, 7×3 stays at
    7 dmg/hit and 2 hits = 14 dmg already kills u46 outright. The
    cascade was cmd[760] attack:attacker_missing because u59's
    cmd[759] move couldn't pass through u46 (alive at hp=4 in our
    sim).
    """
    if "illuminates" in unit.abilities:
        return 1
    for other in _adjacent_units(all_units, unit.position.x, unit.position.y):
        if "illuminates" in other.abilities:
            return 1
    return 0


def healer_heal_amount(unit: Unit, all_units: Iterable[Unit],
                       pos_index: Optional[dict] = None) -> int:
    """Return total heal-per-turn this unit gets from adjacent healers
    (capped at 8 per Wesnoth's rules — multiple healers don't stack
    above 8; cures heals up to 8 too but additionally cures poison).

    Callers issuing many queries against a stable units snapshot
    (init_side healing) can pass `pos_index` to avoid O(N) scans;
    see `_adjacent_units`.
    """
    if "regenerate" in unit.abilities:
        return 0  # regenerate is self-healing, handled separately
    best = 0
    for ally in _adjacent_units(all_units, unit.position.x, unit.position.y,
                                pos_index=pos_index):
        if ally.side != unit.side:
            continue
        if "cures" in ally.abilities or "heals_8" in ally.abilities or "heals+8" in ally.abilities:
            best = max(best, 8)
        elif "heals_4" in ally.abilities or "heals+4" in ally.abilities:
            best = max(best, 4)
    return best


def adjacent_curer(unit: Unit, all_units: Iterable[Unit],
                   pos_index: Optional[dict] = None) -> bool:
    """Return True if any adjacent same-side unit has `cures` (clears
    poison at init_side)."""
    for ally in _adjacent_units(all_units, unit.position.x, unit.position.y,
                                pos_index=pos_index):
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
