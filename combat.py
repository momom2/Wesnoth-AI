"""Bit-exact port of Wesnoth 1.18's combat resolution.

References (all paths under https://github.com/wesnoth/wesnoth/tree/1.18):
  - PRNG seeding:           src/mt_rng.cpp `mt_rng::seed_random`
  - PRNG draw:              src/mt_rng.cpp `mt_rng::get_next_random`
  - 0..N int conversion:    src/random.cpp `rng::get_random_int_in_range_zero_to`
  - round_damage:           src/utils/math.hpp `round_damage`
  - swarm_blows:            src/actions/attack.hpp `swarm_blows`
  - battle_context_unit_stats:  src/actions/attack.cpp ctor lines 73-224
  - perform_hit:            src/actions/attack.cpp lines 954-1213
  - perform / fight loop:   src/actions/attack.cpp lines 1325-1490
  - generic_combat_modifier: src/actions/attack.cpp lines 1602-1626
  - kill_xp / combat_xp:    src/game_config.hpp lines 36-48

The C++ uses `std::mt19937` (32-bit Mersenne Twister) seeded by a single
`uint32_t`. We can't use numpy's `np.random.MT19937(seed)` for this:
numpy seeds via `SeedSequence`, which expands the int through additional
mixing before filling the state, producing a DIFFERENT output stream from
`std::mt19937(uint32)`. Verified empirically:
  std::mt19937(0x9260e745) first output = 3761859111
  numpy MT19937(0x9260e745) first output = 499511960
We therefore implement std::mt19937 directly here (the algorithm is small
and well-defined, and matches the C++ standard library bit-exactly).

NOTE — this module covers the canonical default-era combat path. We do
NOT yet handle every weapon-special edge case; the most common ones
(berserk, charge, drain, firststrike, magical, marksman, swarm, slow,
poison, plague, petrifies, backstab) are in. Less common ones (heal_on_hit,
absolute_silence, etc.) fall through to the no-op baseline. Add as needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# =====================================================================
# PRNG
# =====================================================================

class MTRng:
    """Bit-exact std::mt19937 — Wesnoth's combat RNG.

    Per the Wesnoth source, one synced command (e.g. one [attack])
    creates a fresh `mt_rng`, calls `seed_random(seed_str, call_count)`,
    and from then on all draws come from that same MT state. Each draw
    is a raw `uint32_t` from `mt_()`; combat hit-rolls take it mod 100.

    We implement std::mt19937 directly with the canonical Knuth seeding
    (https://en.wikipedia.org/wiki/Mersenne_Twister), which matches what
    `std::mt19937 mt(seed)` produces in C++. We can NOT use numpy's
    `MT19937(seed)`: numpy seeds via `SeedSequence`, producing a
    different stream.
    """

    # std::mt19937 constants
    _N         = 624
    _M         = 397
    _MATRIX_A  = 0x9908B0DF
    _UPPER     = 0x80000000
    _LOWER     = 0x7FFFFFFF
    _MULT_INIT = 1812433253

    def __init__(self, seed_hex: str, call_count: int = 0):
        # Parse hex to uint32, defaulting to 42 on parse failure
        # (matches `if (!(s >> std::hex >> new_seed)) { new_seed = 42; }`).
        try:
            seed_int = int(seed_hex, 16) & 0xFFFFFFFF
        except (ValueError, TypeError):
            seed_int = 42
        self._mt = [0] * self._N
        self._idx = self._N           # forces twist on first draw
        self._seed(seed_int)
        # mt_.discard(call_count): pull and drop `call_count` outputs.
        for _ in range(call_count):
            self._next_uint32()
        self.calls = call_count

    def _seed(self, seed: int) -> None:
        """Knuth init: state[0]=seed, state[i]=(C * (state[i-1] xor
        (state[i-1]>>30)) + i) & 0xFFFFFFFF for i in 1..N-1."""
        self._mt[0] = seed & 0xFFFFFFFF
        for i in range(1, self._N):
            prev = self._mt[i - 1]
            self._mt[i] = (self._MULT_INIT * (prev ^ (prev >> 30)) + i) & 0xFFFFFFFF
        self._idx = self._N

    def _twist(self) -> None:
        for i in range(self._N):
            y = (self._mt[i] & self._UPPER) | (self._mt[(i + 1) % self._N] & self._LOWER)
            self._mt[i] = self._mt[(i + self._M) % self._N] ^ (y >> 1)
            if y & 1:
                self._mt[i] ^= self._MATRIX_A
        self._idx = 0

    def _next_uint32(self) -> int:
        if self._idx >= self._N:
            self._twist()
        y = self._mt[self._idx]
        self._idx += 1
        # Tempering
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= (y >> 18)
        return y & 0xFFFFFFFF

    def get_next_random(self) -> int:
        """One raw uint32 draw, mirroring `mt_rng::get_next_random()`."""
        v = self._next_uint32()
        self.calls += 1
        return v

    def get_random_int(self, low: int, high: int) -> int:
        """Inclusive [low, high]. Wesnoth uses
        `next_random() % (max+1)` for the [0, max] form (`rng.cpp`
        `get_random_int_in_range_zero_to`), which we mirror.
        """
        span = high - low + 1
        return low + (self.get_next_random() % span)


# =====================================================================
# Constants & enums
# =====================================================================

# Damage type indexing. Order matches Wesnoth's WML keys order in our
# scraped resistance tables.
DAMAGE_TYPES = ["blade", "pierce", "impact", "fire", "cold", "arcane"]

# Time-of-day cycle (default 6-step). lawful_bonus is +25 in day,
# -25 at night, 0 at twilight. Used in `generic_combat_modifier`.
TOD_DEFAULT_CYCLE = [
    ("dawn",          0),
    ("morning",      25),
    ("afternoon",    25),
    ("dusk",          0),
    ("first_watch", -25),
    ("second_watch",-25),
]
MAX_LIMINAL_BONUS = 25


class Alignment(IntEnum):
    LAWFUL   = 0
    NEUTRAL  = 1
    CHAOTIC  = 2
    LIMINAL  = 3


def alignment_from_str(s: str) -> Alignment:
    return {
        "lawful":  Alignment.LAWFUL,
        "neutral": Alignment.NEUTRAL,
        "chaotic": Alignment.CHAOTIC,
        "liminal": Alignment.LIMINAL,
    }.get((s or "").lower(), Alignment.NEUTRAL)


# Game config constants (game_config.hpp lines 36-48).
COMBAT_EXPERIENCE = 1
KILL_EXPERIENCE   = 8
POISON_AMOUNT     = 8
REST_HEAL_AMOUNT  = 2
VILLAGE_HEAL      = 8
REGENERATE_AMOUNT = 8


# =====================================================================
# round_damage — bit-exact port of `src/utils/math.hpp`
# =====================================================================

def round_damage(base_damage: int, bonus: int, divisor: int) -> int:
    """Match the C++:
        int rounding = divisor / 2 - (bonus < divisor || divisor == 1 ? 0 : 1);
        return std::max<int>(1, (base_damage * bonus + rounding) / divisor);

    With divisor=10000 and bonus<=10000: rounding = 5000.
    With divisor=10000 and bonus>10000:  rounding = 4999.
    Floor: 1 if base_damage > 0 (zero base stays zero).
    """
    if base_damage == 0:
        return 0
    if bonus < divisor or divisor == 1:
        rounding = divisor // 2
    else:
        rounding = divisor // 2 - 1
    return max(1, (base_damage * bonus + rounding) // divisor)


# =====================================================================
# Time-of-day damage modifier — bit-exact port of `generic_combat_modifier`
# =====================================================================

def combat_modifier(alignment: Alignment, lawful_bonus: int,
                    is_fearless: bool = False,
                    max_liminal_bonus: int = MAX_LIMINAL_BONUS) -> int:
    """Return the percent-points to ADD to damage_multiplier (base 100)
    for a unit with the given alignment, given the lawful_bonus of the
    hex's effective ToD.

    fearless: floor at 0 — never penalized at night.
    """
    if alignment == Alignment.LAWFUL:
        bonus = lawful_bonus
    elif alignment == Alignment.NEUTRAL:
        bonus = 0
    elif alignment == Alignment.CHAOTIC:
        bonus = -lawful_bonus
    elif alignment == Alignment.LIMINAL:
        bonus = max_liminal_bonus - abs(lawful_bonus)
    else:
        bonus = 0
    if is_fearless:
        bonus = max(bonus, 0)
    return bonus


# =====================================================================
# swarm_blows — bit-exact port of `src/actions/attack.hpp`
# =====================================================================

def swarm_blows(swarm_min: int, swarm_max: int, hp: int, max_hp: int) -> int:
    if hp >= max_hp:
        return swarm_max
    if swarm_max < swarm_min:
        return swarm_min - (swarm_min - swarm_max) * hp // max_hp
    return swarm_min + (swarm_max - swarm_min) * hp // max_hp


# =====================================================================
# Unit + Weapon snapshots
# =====================================================================

@dataclass
class Weapon:
    name: str
    damage: int
    number: int                 # base attack count
    range: str                  # "melee" / "ranged"
    type: str                   # "blade" / "pierce" / ...
    specials: List[str] = field(default_factory=list)


@dataclass
class CombatUnit:
    """Per-combat snapshot of a unit. The replay engine copies these
    out of the live unit at fight-start so mid-fight HP changes don't
    feed back into hit-chance/damage calc (Wesnoth fixes those at
    `battle_context_unit_stats` construction)."""
    side:           int
    hp:             int
    max_hp:         int
    level:          int
    experience:     int
    max_experience: int
    alignment:      Alignment
    weapons:        List[Weapon]
    resistance:     Dict[str, int]   # 0..100, our_resist_pct
    defense_pct:    int               # this unit's WML [defense] value
                                      # for its CURRENT terrain — i.e.
                                      # the % chance an attacker hits it
                                      # (Wesnoth convention; UI displays
                                      # 100 − this as "defense %")
    is_slowed:      bool = False
    is_poisoned:    bool = False
    is_petrified:   bool = False
    is_invulnerable:bool = False
    is_fearless:    bool = False
    has_firststrike:bool = False      # convenience copy from chosen weapon
    abilities:      List[str] = field(default_factory=list)


# =====================================================================
# battle_context_unit_stats — port of attack.cpp ctor (lines 73-224)
# =====================================================================

@dataclass
class BattleStats:
    """Pre-combat, per-side stats fixed at fight start (matches the
    C++ `battle_context_unit_stats` struct)."""
    cth:            int             # 0..100 chance-to-hit
    damage:         int             # damage per HIT
    slow_damage:    int             # halved damage when slowed
    n_attacks:      int             # remaining strikes
    orig_attacks:   int
    rounds:         int             # berserk extension; default 1
    firststrike:    bool
    drains:         bool
    drain_constant: int
    drain_percent:  int
    plague:         bool
    plague_type:    str
    poisons:        bool
    slows:          bool
    petrifies:      bool
    backstab:       bool
    swarm:          bool
    is_attacker:    bool


def _compute_battle_stats(
    self_unit:    CombatUnit,
    opp_unit:     CombatUnit,
    weapon_index: int,
    opp_weapon_index: Optional[int],
    self_lawful_bonus: int,
    opp_lawful_bonus:  int,
    leadership_bonus:  int = 0,    # caller-supplied, % points
    is_attacker:  bool = True,
    backstab_active: bool = False,
) -> BattleStats:
    """Compute the per-side fight stats once, before strikes begin.

    `leadership_bonus` is % points to add to damage_multiplier (the
    caller resolves leadership presence from the surrounding state).
    `backstab_active` is the caller's resolved truth value for the
    `backstab` special (true if behind defender; caller computes from
    the map).
    """
    weapon = self_unit.weapons[weapon_index]
    opp_weapon: Optional[Weapon] = (
        opp_unit.weapons[opp_weapon_index]
        if opp_weapon_index is not None and 0 <= opp_weapon_index < len(opp_unit.weapons)
        else None
    )

    # ---- chance to hit -------------------------------------------------
    if "magical" in weapon.specials:
        cth = 70
    else:
        # opp_unit.defense_pct IS the CTH (Wesnoth's `defense_modifier`
        # returns the % chance an attacker hits the defender on its
        # current terrain — same number as the WML [defense] block).
        cth = opp_unit.defense_pct
        if "marksman" in weapon.specials and is_attacker:
            cth = max(cth, 60)
    cth = max(0, min(100, cth))
    if opp_unit.is_invulnerable:
        cth = 0

    # ---- damage --------------------------------------------------------
    damage_multiplier = 100
    damage_multiplier += combat_modifier(
        self_unit.alignment, self_lawful_bonus,
        is_fearless=self_unit.is_fearless,
    )
    damage_multiplier += leadership_bonus

    # Resistance: opponent's WML resistance value IS already the
    # damage-taken percentage (100 = full damage, 80 = takes 80%
    # damage / has 20% resistance, 120 = takes 120% / weak to type,
    # 0 = immune). attack.cpp does `damage_multiplier *= resistance_modifier`
    # where resistance_modifier = unit::resistance_value() = the raw
    # WML resistance number. We mirror that directly. Default 100
    # (full damage) when the unit's resistance table is missing the
    # key — matches Wesnoth's "no resistance entry" convention.
    resist = opp_unit.resistance.get(weapon.type, 100)
    resist_mult = max(0, resist)

    base_damage = weapon.damage
    # backstab and charge are damage-doublers via specials.
    if backstab_active and "backstab" in weapon.specials and is_attacker:
        base_damage *= 2
    if "charge" in weapon.specials:
        base_damage *= 2

    damage_multiplier *= resist_mult
    damage      = round_damage(base_damage, damage_multiplier, 10000)
    slow_damage = round_damage(base_damage, damage_multiplier, 20000)

    # ---- strike count --------------------------------------------------
    swarm_min = weapon.number
    swarm_max = weapon.number
    if "swarm" in weapon.specials:
        swarm_min = 0
        swarm_max = weapon.number
    n_attacks = swarm_blows(swarm_min, swarm_max, self_unit.hp, self_unit.max_hp)

    # ---- specials presence flags --------------------------------------
    drains = "drains" in weapon.specials
    drain_pct = 50 if drains else 0
    drain_const = 0
    plague = "plague" in weapon.specials
    plague_type = "Walking Corpse"  # default; rare overrides not modeled
    poisons = "poison" in weapon.specials
    slows   = "slow"   in weapon.specials
    petrifies = "petrifies" in weapon.specials
    backstab = "backstab" in weapon.specials  # may be False even if active
    firststrike = "firststrike" in weapon.specials
    swarm = "swarm" in weapon.specials

    # Berserk extends rounds.
    rounds = 30 if "berserk" in weapon.specials else 1

    return BattleStats(
        cth=cth, damage=damage, slow_damage=slow_damage,
        n_attacks=n_attacks, orig_attacks=n_attacks, rounds=rounds,
        firststrike=firststrike, drains=drains,
        drain_constant=drain_const, drain_percent=drain_pct,
        plague=plague, plague_type=plague_type,
        poisons=poisons, slows=slows, petrifies=petrifies,
        backstab=(backstab and backstab_active),
        swarm=swarm, is_attacker=is_attacker,
    )


# =====================================================================
# Combat resolution — port of attack::perform_hit + attack::perform
# =====================================================================

@dataclass
class CombatResult:
    attacker_hp_after: int
    defender_hp_after: int
    attacker_alive:    bool
    defender_alive:    bool
    attacker_xp_after: int
    defender_xp_after: int
    attacker_advanced: bool          # crossed max_experience
    defender_advanced: bool
    defender_poisoned: bool
    defender_slowed:   bool
    defender_petrified:bool
    attacker_poisoned: bool
    attacker_slowed:   bool
    plague_spawned:    bool
    rng_calls_used:    int


def resolve_attack(
    attacker:      CombatUnit,
    defender:      CombatUnit,
    a_weapon_idx:  int,
    d_weapon_idx:  Optional[int],
    a_lawful_bonus:int,
    d_lawful_bonus:int,
    rng:           MTRng,
    a_leadership_bonus: int = 0,
    d_leadership_bonus: int = 0,
    a_backstab_active:  bool = False,
    d_backstab_active:  bool = False,
) -> CombatResult:
    """Run one full Wesnoth attack-vs-defender combat.

    Mutates input snapshots (HP, XP, status flags). Caller is expected
    to copy unit state into a CombatUnit beforehand and write back from
    the result — keeping us decoupled from the live state representation.
    """
    a_stats = _compute_battle_stats(
        attacker, defender, a_weapon_idx, d_weapon_idx,
        a_lawful_bonus, d_lawful_bonus,
        leadership_bonus=a_leadership_bonus,
        is_attacker=True,
        backstab_active=a_backstab_active,
    )
    d_stats = (
        _compute_battle_stats(
            defender, attacker, d_weapon_idx, a_weapon_idx,
            d_lawful_bonus, a_lawful_bonus,
            leadership_bonus=d_leadership_bonus,
            is_attacker=False,
            backstab_active=d_backstab_active,
        )
        if d_weapon_idx is not None and d_weapon_idx >= 0
        else None
    )

    # Per attack.cpp:1414 — defender strikes first only if defender has
    # firststrike AND attacker doesn't.
    defender_first = (
        d_stats is not None
        and d_stats.firststrike
        and not a_stats.firststrike
    )
    rounds_left = max(
        a_stats.rounds, d_stats.rounds if d_stats else 1,
    ) - 1

    plague_spawned = False
    starting_calls = rng.calls

    while True:
        # --- attacker's strike ------------------------------------------
        if not defender_first and a_stats.n_attacks > 0:
            if not _perform_hit(
                attacker, defender, a_stats, d_stats,
                attacker_is_striker=True, rng=rng,
            ):
                if a_stats.petrifies and defender.is_petrified:
                    pass
                # break: someone died (or petrify ended fight)
                break
        defender_first = False

        # --- defender's strike ------------------------------------------
        if d_stats is not None and d_stats.n_attacks > 0:
            if not _perform_hit(
                defender, attacker, d_stats, a_stats,
                attacker_is_striker=False, rng=rng,
            ):
                break

        # --- berserk round restart --------------------------------------
        if (rounds_left > 0
                and a_stats.n_attacks == 0
                and (d_stats is None or d_stats.n_attacks == 0)):
            a_stats.n_attacks = a_stats.orig_attacks
            if d_stats is not None:
                d_stats.n_attacks = d_stats.orig_attacks
            rounds_left -= 1
            defender_first = (
                d_stats is not None
                and d_stats.firststrike
                and not a_stats.firststrike
            )
            continue

        # --- normal end -------------------------------------------------
        if a_stats.n_attacks <= 0 and (d_stats is None or d_stats.n_attacks <= 0):
            break

    # XP awarding (attack.cpp:1474). Set even if combat ended with kill;
    # `unit_killed` overrides the killer's XP to kill_xp.
    a_xp_gain = COMBAT_EXPERIENCE * defender.level
    d_xp_gain = COMBAT_EXPERIENCE * attacker.level if d_stats else 0
    if defender.hp <= 0:
        defender.hp = 0
        a_xp_gain = (
            KILL_EXPERIENCE * defender.level
            if defender.level
            else KILL_EXPERIENCE // 2
        )
        # Plague: spawn a Walking Corpse for attacker's side.
        if a_stats.plague:
            plague_spawned = True
    if attacker.hp <= 0:
        attacker.hp = 0
        d_xp_gain = (
            KILL_EXPERIENCE * attacker.level
            if attacker.level
            else KILL_EXPERIENCE // 2
        ) if d_stats else 0

    if attacker.hp > 0:
        attacker.experience += a_xp_gain
    if defender.hp > 0 and d_stats is not None:
        defender.experience += d_xp_gain

    return CombatResult(
        attacker_hp_after=attacker.hp,
        defender_hp_after=defender.hp,
        attacker_alive=attacker.hp > 0,
        defender_alive=defender.hp > 0,
        attacker_xp_after=attacker.experience,
        defender_xp_after=defender.experience,
        attacker_advanced=attacker.experience >= attacker.max_experience and attacker.hp > 0,
        defender_advanced=defender.experience >= defender.max_experience and defender.hp > 0,
        defender_poisoned=defender.is_poisoned,
        defender_slowed=defender.is_slowed,
        defender_petrified=defender.is_petrified,
        attacker_poisoned=attacker.is_poisoned,
        attacker_slowed=attacker.is_slowed,
        plague_spawned=plague_spawned,
        rng_calls_used=rng.calls - starting_calls,
    )


def _perform_hit(
    striker:      CombatUnit,
    target:       CombatUnit,
    striker_stats:BattleStats,
    target_stats: Optional[BattleStats],
    attacker_is_striker: bool,
    rng:          MTRng,
) -> bool:
    """One strike attempt. Returns False if combat should END after
    this strike (someone died / petrified)."""
    striker_stats.n_attacks -= 1

    # Per-strike RNG: hit if r < cth.
    r = rng.get_next_random() % 100
    hits = r < striker_stats.cth

    if not hits:
        return True  # combat continues; struck nothing

    # Damage application — slow_damage if striker is currently slowed.
    dmg = striker_stats.slow_damage if striker.is_slowed else striker_stats.damage
    if dmg <= 0:
        return True

    # Apply damage to target.
    target.hp -= dmg
    if target.hp < 0:
        target.hp = 0

    # Drains: striker heals up to max_hp.
    if striker_stats.drains and striker.hp > 0:
        heal = max(1 - striker.hp,
                   striker_stats.drain_constant + dmg * striker_stats.drain_percent // 100)
        striker.hp = min(striker.max_hp, striker.hp + heal)

    # Status effects (only when target survives).
    if target.hp > 0:
        if striker_stats.poisons and not target.is_poisoned:
            target.is_poisoned = True
        if striker_stats.slows and not target.is_slowed:
            target.is_slowed = True
        if striker_stats.petrifies:
            target.is_petrified = True
            # End combat: striker keeps no further attacks, target takes none.
            striker_stats.n_attacks = 0
            if target_stats is not None:
                target_stats.n_attacks = -1
            return False

    # Death ends the fight.
    if target.hp <= 0:
        return False
    return True


# =====================================================================
# Public surface
# =====================================================================

__all__ = [
    "MTRng", "CombatUnit", "Weapon", "CombatResult",
    "Alignment", "alignment_from_str",
    "round_damage", "combat_modifier", "swarm_blows",
    "resolve_attack",
    "TOD_DEFAULT_CYCLE",
    "COMBAT_EXPERIENCE", "KILL_EXPERIENCE", "POISON_AMOUNT",
    "REST_HEAL_AMOUNT", "VILLAGE_HEAL", "REGENERATE_AMOUNT",
]
