"""Trait roller and applier for replay reconstruction.

Wesnoth's actual trait selection is rolled via the synced random
generator at recruit time. Reproducing it bit-exact requires modeling
the engine's random-state cumulatively across every game event — a
fragile dependency we don't want to take on. Instead we pick traits
deterministically from a per-recruit hash so the supervised pipeline
sees consistent (if not bit-exact) trait effects.

Trait effects we care about for gameplay reconstruction:
  - max_hp     (strong +1, resilient +4 +1/lvl, healthy +1 +1/lvl,
                weak -1, slow +5%, quick -5%, aged -8)
  - attacks    (strong +1 melee dmg, dextrous +1 ranged dmg,
                weak -1 melee dmg)
  - movement   (quick +1, slow -1, aged -1)
  - max_xp     (intelligent -20%, dim +20%)
  - statuses   (undead/mechanical/elemental → unpoisonable etc.)

The race-trait pool follows Wesnoth's vanilla rules:
  - Default: strong, quick, intelligent, resilient
  - Elves:   + dextrous
  - Dwarves: + healthy (replaces the role of 4-mp = quick auto-trait)
  - Goblins: weak, slow, dim, fearless (always the same fixed pool)
  - Undead:  musthave undead; no random traits
  - Mechanical / Elemental: musthave only
  - Ogres / Trolls: have a different distinct distribution; we use
                    default + plus race-specifics where documented

This isn't bit-exact reproduction. It IS consistent across re-runs and
captures the gameplay impact (HP swings, mp swings, attack-dmg swings)
that the supervised model needs to learn from.

Dependencies: classes
Dependents:   tools.replay_dataset
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

from classes import Attack, DamageType, Unit


# ----------------------------------------------------------------------
# Trait effect tables
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class TraitEffect:
    hp_delta:           int   = 0       # flat +N HP
    hp_per_level:       int   = 0       # +N HP × level
    hp_pct:             float = 0.0     # ±X% of max_hp
    melee_dmg_delta:    int   = 0
    ranged_dmg_delta:   int   = 0
    movement_delta:     int   = 0
    xp_pct:             float = 0.0     # ±X% of max_xp
    rest_heal:          int   = 0       # passive heal/turn (healthy)
    statuses:           FrozenSet[str] = frozenset()
    # Per-terrain CTH overrides applied with `replace=yes` semantics
    # (Wesnoth [effect] apply_to=defense replace=yes [defense] …).
    # Keyed by canonical WML terrain name. Higher number = easier for
    # attackers to hit (lower defense). Feral applies village=50.
    defense_overrides:  FrozenSet[Tuple[str, int]] = frozenset()
    # Whether this trait counts toward the "any"-availability random
    # pool. False for musthaves (undead/mechanical/elemental) and for
    # availability="none" (loyal — reserved for events).
    is_random_eligible: bool = True


TRAITS: Dict[str, TraitEffect] = {
    "strong":      TraitEffect(hp_delta=1, melee_dmg_delta=1),
    "dextrous":    TraitEffect(ranged_dmg_delta=1),
    "quick":       TraitEffect(movement_delta=1, hp_pct=-5.0),
    "intelligent": TraitEffect(xp_pct=-20.0),
    "resilient":   TraitEffect(hp_delta=4, hp_per_level=1),
    "healthy":     TraitEffect(hp_delta=1, hp_per_level=1, rest_heal=2),
    "fearless":    TraitEffect(),       # ToD penalty avoidance — handled in combat
    # Feral musthave (Bats, some monsters): "Receives only 50% defense
    # in land-based villages regardless of base terrain." Applied via
    # `[effect] apply_to=defense replace=yes [defense] village=-50`,
    # which collapses to village CTH = 50 (Wesnoth's max(min,max) rule
    # with both halves = 50). We carry that as a defense_overrides
    # entry so apply_traits_to_unit can stamp it on the unit's
    # defense table.
    # Feral applies `village=-50 replace=yes`. The negative value
    # means a min-CAP floor on def_pct (terrain_resolver honors
    # the sign). Storing -50 (not 50) so the apply-traits stamp
    # propagates the cap semantic to the unit's defense table.
    "feral":       TraitEffect(
        defense_overrides=frozenset({("village", -50)}),
    ),
    # Negative traits (Walking Corpse etc.)
    "weak":        TraitEffect(hp_delta=-1, melee_dmg_delta=-1),
    "slow":        TraitEffect(movement_delta=-1, hp_pct=5.0),
    "dim":         TraitEffect(xp_pct=20.0),
    "aged":        TraitEffect(movement_delta=-1, hp_delta=-8),
    # Musthaves (status-only, no HP/dmg)
    "undead":      TraitEffect(statuses=frozenset({"unpoisonable",
                                                   "undrainable",
                                                   "unplagueable"}),
                                is_random_eligible=False),
    "mechanical":  TraitEffect(statuses=frozenset({"unpoisonable",
                                                   "undrainable",
                                                   "unplagueable"}),
                                is_random_eligible=False),
    "elemental":   TraitEffect(statuses=frozenset({"unpoisonable",
                                                   "undrainable",
                                                   "unplagueable"}),
                                is_random_eligible=False),
    "loyal":       TraitEffect(is_random_eligible=False),
}


# ----------------------------------------------------------------------
# Race trait pools
# ----------------------------------------------------------------------

# Default trait pool: every standard living race draws from these unless
# the race overrides.
_DEFAULT_POOL = ("strong", "quick", "intelligent", "resilient")

RACE_POOLS: Dict[str, Tuple[str, ...]] = {
    # Most living races use the default 4-trait pool.
    "human":     _DEFAULT_POOL,
    "elf":       _DEFAULT_POOL + ("dextrous",),
    "dwarf":     _DEFAULT_POOL + ("healthy",),
    "drake":     _DEFAULT_POOL,
    "saurian":   _DEFAULT_POOL,
    "lizard":    _DEFAULT_POOL,     # alternate name for saurian
    "merman":    _DEFAULT_POOL,
    "naga":      _DEFAULT_POOL,
    "orc":       _DEFAULT_POOL,
    "troll":     _DEFAULT_POOL,
    "ogre":      _DEFAULT_POOL,
    "wose":      _DEFAULT_POOL,
    "dunefolk":  _DEFAULT_POOL,
    "gryphon":   _DEFAULT_POOL,
    "horse":     _DEFAULT_POOL,
    "wolf":      _DEFAULT_POOL,
    "bat":       _DEFAULT_POOL,
    "bats":      _DEFAULT_POOL,
    "cats":      _DEFAULT_POOL,
    "monster":   _DEFAULT_POOL,
    # Goblin trait pool: weak, slow, dim, fearless — fixed (Wesnoth's
    # PEASANT_VARIANT macro)
    "goblin":    ("weak", "slow", "dim", "fearless"),
    # Musthave-only races
    "undead":     (),
    "mechanical": (),
    "elemental":  (),
    "ship":       (),
    "fake":       (),
}


# Musthave traits forced regardless of random rolls.
RACE_MUSTHAVE: Dict[str, Tuple[str, ...]] = {
    "undead":     ("undead",),
    "mechanical": ("mechanical",),
    "elemental":  ("elemental",),
}


# Necrophage is the singular living-undead exception that gets
# fearless+undead as musthaves (per the user's clarification).
UNIT_MUSTHAVE: Dict[str, Tuple[str, ...]] = {
    "Necrophage":  ("fearless", "undead"),
    # Walking Corpse / Soulless / Ancient Lich / Ghoul / etc are race=undead
    # so they automatically get the undead trait via RACE_MUSTHAVE.
}


# Number of random traits to roll per unit, per race. From
# `wesnoth_src/data/core/units.cfg` [race] num_traits= attribute.
# Woses and monsters get 0 traits (so a Wose has its base max_hp
# with no resilient bonus); goblins / mechanical / undead get 1
# (with most of their entries forced as musthaves anyway).
RACE_NUM_TRAITS: Dict[str, int] = {
    "human":     2,
    "elf":       2,
    "dwarf":     2,
    "drake":     2,
    "saurian":   2,
    "lizard":    2,
    "merman":    2,
    "naga":      2,
    "orc":       2,
    "troll":     2,
    "ogre":      2,
    "dunefolk":  2,
    "gryphon":   2,
    "horse":     2,
    "wolf":      2,
    "bat":       2,
    "bats":      2,
    "cats":      2,
    "falcon":    2,
    "raven":     2,
    "wose":      0,
    "monster":   0,
    "goblin":    1,
    "mechanical":1,
    "undead":    1,
    "elemental": 1,
    "ship":      0,
    "fake":      0,
}

# Backward-compat alias (was used as the only knob before per-race rolls).
DEFAULT_NUM_TRAITS = 2


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def roll_traits(unit_type: str, race: str, *,
                seed_hex: str = "",
                seed_token: str = "",
                is_leader: bool = False,
                num_traits: int = DEFAULT_NUM_TRAITS,
                base_movement: int = 5,
                trait_info: Optional[dict] = None,
                n_genders: int = 1,
                ) -> List[str]:
    """Pick the trait set for a unit.

    `trait_info` (preferred): per-unit trait info from unit_stats.json
    (`{"num_traits": N, "musthave": [...], "pool": [...]}`). When
    given, uses Wesnoth's exact selection rule:

      1. Apply ALL musthave traits (no random calls).
      2. Run `max(0, num_traits - len(musthave))` random picks from the
         pool, in order: idx = mt() % len(pool); pop and repeat.

    Falls back to the (race, unit_type) hardcoded tables when
    `trait_info` is missing — that's only safe for vanilla races and
    won't catch per-unit exceptions like Walking Corpse (fearless +
    undead musthaves) or Heavy Infantryman (fearless in the random
    pool). Always pass `trait_info` from unit_stats.json when
    available.

    Leaders: only musthaves, plus the default-era `quick_4mp_leaders`
    auto-quick rule. See `docs/wesnoth_rules.md` for full source
    citations:

      - Random-trait skip: `wesnoth_src/src/units/unit.cpp:880-883`.
        The candidate-trait pool is filtered by `!can_recruit() ||
        avl == "any"`. For leaders (`can_recruit()=true`), only
        traits with `availability="any"` are eligible — and NO
        default-era trait has that attribute, so the pool is empty.

      - Auto-quick: `wesnoth_src/data/multiplayer/eras.lua` +
        `data/core/macros/multiplayer.cfg` `QUICK_4MP_LEADERS`
        macro, included in `ERA_DEFAULT`. At prestart, every
        `canrecruit=true` unit with `max_moves==4` gets the quick
        trait added (and `moves`/`hitpoints` reset to max). Filter
        is exact equality (==4), not <=4. We apply this in
        `roll_traits` rather than as a post-pass so the trait flows
        through `apply_traits_to_unit`'s standard +1 MP / -5% HP
        application; functionally equivalent for leaders since the
        trait is added before HP/MP get used in any decision.
    """
    if trait_info is not None:
        must = list(trait_info.get("musthave", []))
        pool = list(trait_info.get("pool", []))
        target_total = int(trait_info.get("num_traits", num_traits))
    else:
        must = list(UNIT_MUSTHAVE.get(unit_type, RACE_MUSTHAVE.get(race, ())))
        pool = list(RACE_POOLS.get(race, _DEFAULT_POOL))
        target_total = num_traits
    out = list(must)

    if is_leader:
        # Auto-quick rule for base-4-MP leaders (eras.lua
        # quick_4mp_leaders, always-on in default era). Filter is
        # `max_moves == 4` AFTER must-haves are applied; in 1.18.4
        # default era no must-have trait alters max_moves, so
        # checking base_movement is equivalent. If a future era
        # introduces an MP-altering musthave on a 4-MP leader, this
        # check needs to move post-musthave-application.
        if base_movement == 4 and "quick" not in out:
            out.append("quick")
        return out

    n_random = max(0, target_total - len(out))
    if n_random == 0 or not pool:
        return out

    # Wesnoth's `generate_traits` (unit.cpp:813-893) rebuilds
    # candidate_traits each iteration by walking u_type.possible_traits()
    # in order and skipping any trait already applied to the unit.
    # Crucially, possible_traits CAN have DUPLICATES (e.g. trolls' race
    # additional_traits include strong/quick/resilient which are also
    # in the global pool — config::add_child appends without dedup,
    # so each appears twice). The duplicates affect the random pick's
    # probability distribution AND the seed-driven order: a roll of
    # idx=3 over a deduped pool picks a different trait than over the
    # WML-correct pool with dups.
    #
    # Match Wesnoth: keep `pool` with dups, on each iteration build
    # `candidates` = [t for t in pool if t not in out] (dups preserved
    # for traits not yet applied; ALL copies dropped once a trait
    # makes it into out).

    if seed_hex:
        from combat import MTRng
        rng = MTRng(seed_hex)
        # Wesnoth's `unit::init` consumes synced random calls in this
        # order, BEFORE we get to the trait rolls:
        #   - gender selection: 1 call iff the unit has multiple genders
        #     (single-gender unit-types like Saurian Skirmisher or Wose
        #     short-circuit and consume zero).
        # The 12 markov-name calls happen AFTER trait selection, so
        # they don't shift the trait outputs and we can ignore them
        # here.
        if n_genders > 1:
            rng.get_next_random()
        for _ in range(n_random):
            candidates = [t for t in pool if t not in out]
            if not candidates:
                break
            idx = rng.get_random_int(0, len(candidates) - 1)
            out.append(candidates[idx])
    else:
        # Legacy fallback: hash-based deterministic pick. Won't match
        # what Wesnoth would have rolled, but keeps the supervised
        # pipeline running for replays extracted before per-recruit
        # seeds were captured. Re-extract the corpus to use the
        # bit-exact path.
        h = int(hashlib.sha256(seed_token.encode()).hexdigest()[:16], 16)
        for _ in range(n_random):
            candidates = [t for t in pool if t not in out]
            if not candidates:
                break
            idx = h % len(candidates)
            out.append(candidates[idx])
            h //= max(1, len(candidates) + 1)
    return out


def apply_traits_to_unit(u: Unit, trait_ids: List[str], level: int = 1,
                         defense_table: Optional[Dict[str, int]] = None
                         ) -> Unit:
    """Return a copy of `u` with trait effects applied. Mutates a fresh
    Unit (we never mutate inputs because Unit is hashable in our set
    map.units).

    If `defense_table` is provided, traits with `defense_overrides`
    (currently just `feral`) mutate it in-place. Caller stashes the
    final dict on the Unit (via `_defense_table` attr) so combat can
    consult it instead of the unit-type's static defense table.
    """
    fields = {k: v for k, v in u.__dict__.items() if not k.startswith("_")}
    stash = {k: v for k, v in u.__dict__.items() if k.startswith("_")}
    max_hp = u.max_hp
    max_moves = u.max_moves
    max_xp = u.max_exp
    attacks = list(u.attacks)
    statuses = set(u.statuses)
    traits_applied = set(u.traits)

    for tid in trait_ids:
        eff = TRAITS.get(tid)
        if eff is None:
            continue
        traits_applied.add(tid)
        # HP. Wesnoth's `unit::add_modification` applies each
        # [effect] in declaration order via `apply_modifier(max_hp,
        # increase_total)`, where `max_hp` is the CURRENT max
        # (not the base). And `apply_modifier` for percent values
        # uses `div100rounded` (round-half-away-from-zero with
        # +50 bias), NOT Python's int() (truncate toward zero).
        # See wesnoth_src/src/units/unit.cpp:2124-2153 +
        # serialization/string_utils.cpp:395-408 +
        # utils/math.hpp:38-41.
        #
        # Order-dependence: a Resilient-then-Quick Dark Adept gets
        # 28 +5 = 33 → 33 -5% = 33 - div100rounded(165) = 33 - 2 = 31.
        # A Quick-then-Resilient one gets 28 -5% = 28 - 1 = 27,
        # then 27 + 5 = 32. Our trait_ids list is in pick order
        # (matching Wesnoth's RNG-driven application order), so
        # iterating it in order produces the correct result.
        max_hp += eff.hp_delta
        max_hp += eff.hp_per_level * max(1, level)
        if eff.hp_pct:
            pct = int(eff.hp_pct)
            raw = max_hp * pct
            # div100rounded: round half away from zero, +50 bias.
            if raw < 0:
                delta = -(((-raw) + 50) // 100)
            else:
                delta = (raw + 50) // 100
            max_hp += delta
        # Movement
        max_moves += eff.movement_delta
        # XP. Same `apply_modifier` semantics as HP (Wesnoth's
        # `apply_to=max_experience` handler, unit.cpp:2246-2248,
        # calls `apply_modifier(max_experience_, increase, 1)`).
        # Use running max_xp + div100rounded.
        if eff.xp_pct:
            pct = int(eff.xp_pct)
            raw = max_xp * pct
            if raw < 0:
                delta = -(((-raw) + 50) // 100)
            else:
                delta = (raw + 50) // 100
            max_xp += delta
            max_xp = max(1, max_xp)   # apply_modifier minimum=1 here
        # Attack damage
        if eff.melee_dmg_delta or eff.ranged_dmg_delta:
            new_atks: List[Attack] = []
            for atk in attacks:
                bump = (eff.ranged_dmg_delta if atk.is_ranged
                        else eff.melee_dmg_delta)
                if bump:
                    new_atks.append(Attack(
                        type_id=atk.type_id,
                        number_strikes=atk.number_strikes,
                        damage_per_strike=max(1, atk.damage_per_strike + bump),
                        is_ranged=atk.is_ranged,
                        weapon_specials=atk.weapon_specials,
                    ))
                else:
                    new_atks.append(atk)
            attacks = new_atks
        # Statuses
        statuses |= eff.statuses
        # Defense overrides (replace=yes semantics).
        if eff.defense_overrides and defense_table is not None:
            for terrain, cth in eff.defense_overrides:
                defense_table[terrain] = int(cth)

    # Cap HP at the new max if traits boosted it; raise current_hp
    # accordingly when the unit is at full HP (typical for a fresh
    # recruit). Don't unilaterally cap-up damaged units — they keep
    # their relative damage.
    new_current_hp = u.current_hp
    if u.current_hp == u.max_hp:
        new_current_hp = max_hp

    fields.update({
        "max_hp": max_hp,
        "current_hp": new_current_hp,
        "max_moves": max_moves,
        "current_moves": min(u.current_moves, max_moves),
        "max_exp": max_xp,
        "attacks": attacks,
        "statuses": statuses,
        "traits": traits_applied,
    })
    out = Unit(**fields)
    for k, v in stash.items():
        setattr(out, k, v)
    return out


__all__ = [
    "TRAITS", "TraitEffect", "RACE_POOLS", "RACE_MUSTHAVE", "UNIT_MUSTHAVE",
    "DEFAULT_NUM_TRAITS", "roll_traits", "apply_traits_to_unit",
]
