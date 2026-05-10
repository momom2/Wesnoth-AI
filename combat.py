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
    # `accuracy` / `parry` are signed CTH adjustments per
    # attack.cpp:168-169:
    #     cth = defender.defense_modifier(terrain)
    #         + attacker.weapon.accuracy
    #         - defender.weapon.parry  (only when defender retaliates)
    # Then clamped to [0, 100]. Distinct from the `marksman` /
    # `magical` SPECIALS, which apply floors of 60 / 70 to cth.
    # Default era only uses these on a few weapons (Elvish Champion
    # sword: accuracy=10).
    accuracy: int = 0
    parry: int = 0


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
    # Drain/poison/plague/feeding all share the same target-eligibility
    # filter: undead, mechanical, and elemental units block these
    # life-draining effects. Wesnoth gates each via the target's
    # "undrainable" / "unpoisonable" / "unplagueable" status which the
    # TRAIT_{UNDEAD,MECHANICAL,ELEMENTAL} macros apply via
    # `[effect] apply_to=status add=...`. The three are always set
    # together by the trait macros (verified against
    # wesnoth_src/data/core/macros/traits.cfg) but Wesnoth's engine
    # checks each status separately at the call site, so we keep them
    # separate here too — a future scenario [unit] override could set
    # one without the others.
    is_undrainable: bool = False
    is_unpoisonable: bool = False


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
        # `accuracy` / `parry` numeric attrs on [attack] adjust cth
        # before the marksman/magical floors apply. Per
        # attack.cpp:168-169:
        #     cth = defender.defense_modifier(terrain)
        #         + attacker.weapon.accuracy
        #         - defender.weapon.parry  (only when defender weapon)
        # then clamped to [0, 100]. Default era's only user is the
        # Elvish Champion sword (accuracy=10). Without this term, the
        # Champion's 5-strike sword effectively misses 10% more often
        # than reality. Witnessed 2026-05-08 in
        # 2p__Den_of_Onis_Turn_65_(135596) cmd[1161]: 5-strike sword
        # vs Troll Whelp on grass should land 4 hits (24 dmg, kill)
        # but our sim landed 3 (18 dmg, leaves the Whelp alive,
        # cascading into cmd[1164] move:final_occupied).
        cth += weapon.accuracy
        if opp_weapon is not None:
            cth -= opp_weapon.parry
        if "marksman" in weapon.specials and is_attacker:
            cth = max(cth, 60)
    # DEFLECT (data/core/macros/weapon_specials.cfg:95-104):
    #   [chance_to_hit] id=deflect sub=10 cumulative=yes
    #     active_on=defense apply_to=opponent
    # When the OPPONENT's weapon is deflect-able and they're DEFENDING
    # against our hit, our cth drops by 10. Cumulative across multiple
    # deflect-tagged specials on the opp weapon (we treat it as a
    # single -10 since no default-era weapon stacks). is_attacker
    # here means OUR side is attacking; the opp weapon is the
    # defender's. opp.deflect on the defender's weapon → -10 to our
    # cth iff is_attacker.
    if (is_attacker and opp_weapon is not None
            and "deflect" in opp_weapon.specials):
        cth -= 10
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

    # STEADFAST ability: when the OPPONENT (the one being attacked,
    # i.e., we're computing how much WE damage them) has steadfast and
    # is DEFENDING (active_on=defense), their resistance bonus doubles,
    # capped so resulting resistance >= 50 (i.e., damage taken >= 50%).
    # WML at wesnoth_src/data/core/macros/abilities.cfg:
    #   [resistance] id=steadfast multiply=2 max_value=50
    #     [filter_base_value] greater_than=0 less_than=50
    #     active_on=defense
    # Filter: applies only if base bonus is in (0, 50) -- i.e., the
    # unit has SOME resistance but not full immunity, and isn't weak.
    # is_attacker here means SELF is attacking and we're computing
    # damage SELF deals to opp. opp is the defender iff is_attacker
    # is True. Steadfast on opp applies only when opp is defending,
    # i.e., when is_attacker is True.
    # Without this, a Dwarvish Guardsman (impact resist=80) takes
    # 80% impact damage from a Heavy Infantryman's mace; with steadfast
    # it should take 60%. Witnessed in replay 292008e8eac1 cmd 140
    # turn 7: HI mace 12*80%=10 dmg/hit kills Guardsman at 20 hp; with
    # steadfast 12*60%=7 dmg/hit, Guardsman ends at 6 hp and survives
    # for cmd 141 -- which Wesnoth's recorded sequence confirms.
    if "steadfast" in opp_unit.abilities and is_attacker:
        base_bonus = 100 - resist
        if 0 < base_bonus < 50:
            new_bonus = min(50, base_bonus * 2)
            resist = 100 - new_bonus
    resist_mult = max(0, resist)

    base_damage = weapon.damage
    # backstab and charge are damage-doublers via specials.
    if backstab_active and "backstab" in weapon.specials and is_attacker:
        base_damage *= 2
    # Charge: Wesnoth's [charge] special doubles damage on BOTH SIDES
    # of the attack, but ONLY when the unit with charge is the
    # ATTACKER (initiating). On a counter-attack the charge unit
    # gets no bonus, and the opponent's lack of charge doesn't matter.
    # Per `wesnoth_src/data/core/macros/special-notes.cfg` and the
    # combat.cpp `[specials]` walker: the active set of specials is
    # filtered by `attack_under_attack` — meaning charge applies only
    # in the attack-side context.
    #
    # Concretely, for `Horseman attacks Dark Sorcerer`:
    #   - Horseman's spear (charge) doubles → both Horseman and DS deal 2x.
    #   - DS counter (no charge weapon) -- both still 2x via the
    #     attacker's charge bringing the bonus into the round.
    # And for `Skeleton attacks Horseman` (Horseman is defender):
    #   - Horseman's spear has charge but Horseman is DEFENDING, so
    #     charge doesn't fire. Neither side doubles.
    #
    # Our resolution: double if EITHER (a) self has charge AND self is
    # attacker, OR (b) opp has charge AND opp is attacker (i.e., we're
    # the defender and the attacker has charge).
    self_charges = "charge" in weapon.specials
    opp_charges = (opp_weapon is not None
                   and "charge" in opp_weapon.specials)
    charge_doubled = ((self_charges and is_attacker)
                      or (opp_charges and not is_attacker))
    if charge_doubled:
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
    # When True, the dying side is the ATTACKER (defender's plague
    # kill); a Walking Corpse should spawn on the DEFENDER's side
    # at the ATTACKER's hex. When False (and plague_spawned), the
    # standard direction applies (attacker killed defender; corpse
    # on attacker's side at defender's hex). Both flags can be set
    # if both sides have plague AND both die in the same combat —
    # see replay_dataset's plague handler for resolution.
    plague_spawned_attacker_died: bool = False
    rng_calls_used:    int = 0


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

    # XP awarding (attack.cpp:1395-1400 unconditionally sets combat_xp
    # for BOTH sides, regardless of whether the defender can counter):
    #   a_.xp_ = combat_xp(d_.level())
    #   d_.xp_ = combat_xp(a_.level())
    # Then if a side dies, the SURVIVING side's xp gets overridden to
    # kill_xp via unit_killed (attack.cpp:1474). For level-0 units,
    # kill_xp is half (game_config::combat_xp returns level for combat
    # and 8 * level for kills, with level-0 special-cased to 4 -- not
    # the level/2 we had).
    a_xp_gain = COMBAT_EXPERIENCE * defender.level
    d_xp_gain = COMBAT_EXPERIENCE * attacker.level
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
    plague_spawned_attacker_died = False
    if attacker.hp <= 0:
        attacker.hp = 0
        d_xp_gain = (
            KILL_EXPERIENCE * attacker.level
            if attacker.level
            else KILL_EXPERIENCE // 2
        )
        # Defender's plague counter killed the attacker: spawn a
        # Walking Corpse for DEFENDER's side at the ATTACKER's hex.
        # Wesnoth's plague isn't direction-specific — it triggers on
        # ANY kill by a plague-flagged weapon (attack.cpp:1287 doesn't
        # care which side launched the strike). Only the attacker
        # case was being handled here, dropping the WC on
        # counter-kills. Concrete repro:
        # 2p__Sablestone_Delta_Turn_16_(121180).bz2 cmd[220]:
        # Heavy Infantryman (HP=6) attacks Walking Corpse (HP=4 with
        # plague-impact); WC counter strikes hit and kill the
        # attacker; Wesnoth spawns a side-2 WC at the attacker's
        # hex (15,12), our sim was leaving (15,12) empty. Detected
        # via tools/diff_unit_counter.py 2026-05-03.
        if d_stats is not None and d_stats.plague:
            plague_spawned_attacker_died = True

    if attacker.hp > 0:
        attacker.experience += a_xp_gain
    if defender.hp > 0:
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
        plague_spawned_attacker_died=plague_spawned_attacker_died,
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

    # Apply damage to target. Track damage_done = the actual amount of
    # HP removed (= min(target.hp_pre, dmg)). Drains uses damage_done,
    # NOT the raw weapon damage -- killing a 5-hp defender with a 13-dmg
    # hit drains 5*0.5=2 hp, not 13*0.5=6 hp. Mirrors attack.cpp:1020:
    #   damage_done = min(defender.hitpoints(), attacker.damage_)
    target_hp_pre = target.hp
    target.hp -= dmg
    if target.hp < 0:
        target.hp = 0
    damage_done = target_hp_pre - target.hp

    # Drains: striker heals damage_done * drain_percent / 100 + drain_const.
    # Cap at max_hp; floor at 1 hp (negative drain can't kill).
    # attack.cpp:1031-1040 (apply at 1145-1146 right after damage).
    #
    # Integer-math consequence: with default drain_percent=50, a hit
    # for 1 dmg gives 1*50//100 = 0 -- drain does NOT heal on 1-dmg
    # strikes (Wesnoth's drains_damage is int, same floor). The
    # `if drains_damage > 0` guard at attack.cpp:1145 then skips the
    # heal call entirely. We mirror this implicitly via `striker.hp +=
    # heal` being a no-op when heal is 0.
    # Drain only heals from drainable targets. Undead/mechanical/
    # elemental targets carry the `undrainable` status (set by the
    # corresponding musthave trait at scrape time); the heal step
    # is skipped entirely for those. Mirrors Wesnoth's
    # `opp.get_state("undrainable")` gate at attack.cpp:1023-1031.
    # User-reported repro 2026-05-03:
    # 2p__Caves_of_the_Basilisk_Turn_11_(93641).bz2 cmd[132]:
    # Ghost (drain) attacks Walking Corpse (undead/undrainable).
    # Wesnoth: Ghost takes counter damage but doesn't heal from the
    # drain. Our sim was healing the Ghost back to full, leaving it
    # alive through subsequent combats Wesnoth had killed it in.
    if (striker_stats.drains and damage_done > 0
            and not target.is_undrainable):
        heal = (damage_done * striker_stats.drain_percent // 100
                + striker_stats.drain_constant)
        if heal != 0:
            # Cap first (matches Wesnoth's order):
            heal = min(heal, striker.max_hp - striker.hp)
            # Then floor: negative drain can't kill the striker.
            heal = max(heal, 1 - striker.hp)
            striker.hp += heal

    # Status effects (only when target survives).
    if target.hp > 0:
        # Poison only applies if the target isn't already poisoned AND
        # isn't immune. Wesnoth gates via `opp.get_state("unpoisonable")`
        # before adding the poison status (attack.cpp:1057). Without
        # this check our sim was poisoning undead/mechanical/elemental
        # targets, which then took 8 HP/turn from the per-turn poison
        # damage tick — concrete repro at
        # 2p__Caves_of_the_Basilisk_Turn_11_(93641).bz2 cmd[155]:
        # a side-2 Vampire Bat poison-attack against side-1 Ghoul
        # (undead/unpoisonable) marked the Ghoul poisoned, and over
        # the next several init_side firings our sim drained 16 HP
        # from a unit Wesnoth left at full strength. (Bats have a
        # poison melee attack — discovered via user's GUI HP trace
        # comparison after Stage 6 drain fix.)
        if (striker_stats.poisons and not target.is_poisoned
                and not target.is_unpoisonable):
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
