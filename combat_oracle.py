"""Cheap analytical combat-outcome predictor for Wesnoth attacks.

Wesnoth's full combat resolution handles dozens of specials (backstab,
charge, drain, firststrike, marksman, poison, slow, ...), time-of-day
alignment bonuses, and terrain-conditional defense / resistance. We
don't reimplement all of that here — the purpose of this module is to
give the action sampler a cheap "is this attack good?" prior to bias
its target logits toward sensible targets. A trained policy with this
prior will converge faster on "attack what you can kill, avoid what
kills you back" than an unbiased policy.

The formula is intentionally first-order:
  expected_damage = hit_chance × strikes × damage_per_strike × (1 − resist)

Extensions we deliberately punted:
  - Time-of-day alignment multipliers (needs current ToD from state).
  - Weapon specials (backstab, charge, drain, magical, marksman, ...).
  - Unit abilities (leadership, steadfast, ...).
  - Chain-kill implications (e.g., kill one, next turn take their hex).

Calibration expectation: scores range roughly −40 (bad attack, we die)
to +40 (good attack, they die). Sampler scales by `alpha` (default
0.1) to bring into logit-scale.
"""

from __future__ import annotations

from typing import Optional

from classes import DamageType, Unit


# Placeholder defense chance when we can't look up the defender's
# terrain-specific defense. Wesnoth's typical terrain defense ranges
# 20% (mountains, forests for elves) to 70% (flat for non-quick).
# 50% is a neutral midpoint that keeps the oracle from confidently
# over- or under-estimating.
_DEFAULT_HIT_CHANCE_DEFENDER = 0.5
_DEFAULT_HIT_CHANCE_ATTACKER = 0.5  # attacker on defender's terrain — we don't model here


def _resistance(unit: Unit, damage_type: DamageType) -> float:
    """Return unit's resistance to `damage_type` as a fraction in
    [-1.0, 1.0]. Positive = takes less damage; negative = takes more.

    Our Unit.resistances is indexed by DamageType enum order, stored
    as percent values (0 to 100). Convert to fractional multiplier.
    """
    idx = int(damage_type)
    if 0 <= idx < len(unit.resistances):
        return unit.resistances[idx] / 100.0
    return 0.0


def _expected_weapon_damage(
    strikes: int,
    damage_per_strike: int,
    defender_resistance: float,
    hit_chance: float,
) -> float:
    """Expected total damage from one combat exchange for a single
    weapon. `defender_resistance` is fractional (0.3 = 30% less damage;
    -0.2 = 20% more — undead vs arcane, for example)."""
    multiplier = max(0.0, 1.0 - defender_resistance)  # clamp: <0 resist = bonus
    if defender_resistance < 0:
        multiplier = 1.0 - defender_resistance  # actually BOOST damage
    per_hit = damage_per_strike * multiplier
    return max(0.0, hit_chance) * strikes * per_hit


def expected_attack_net_damage(
    attacker: Unit,
    defender: Unit,
    weapon_index: int = 0,
) -> float:
    """Estimate `(damage_to_defender − damage_to_attacker)`.

    Positive = favorable for attacker. Uses the attacker's weapon at
    `weapon_index` and the defender's best-matching counter (same
    ranged-ness). Ignores ToD, specials, leadership.

    Returns 0.0 if either unit has no attacks.
    """
    if not attacker.attacks or not defender.attacks:
        return 0.0
    if weapon_index >= len(attacker.attacks):
        weapon_index = 0

    att_w = attacker.attacks[weapon_index]
    # Defender's counter must match ranged-ness. Pick the highest-
    # expected-damage matching weapon, or None.
    counter: Optional = None
    best_counter_score = -1.0
    for w in defender.attacks:
        if w.is_ranged != att_w.is_ranged:
            continue
        score = w.number_strikes * w.damage_per_strike
        if score > best_counter_score:
            best_counter_score = score
            counter = w

    damage_to_defender = _expected_weapon_damage(
        strikes=att_w.number_strikes,
        damage_per_strike=att_w.damage_per_strike,
        defender_resistance=_resistance(defender, att_w.type_id),
        hit_chance=_DEFAULT_HIT_CHANCE_DEFENDER,
    )
    # Cap by defender's current HP — overkill isn't extra benefit.
    damage_to_defender = min(damage_to_defender, float(defender.current_hp))

    if counter is not None:
        damage_to_attacker = _expected_weapon_damage(
            strikes=counter.number_strikes,
            damage_per_strike=counter.damage_per_strike,
            defender_resistance=_resistance(attacker, counter.type_id),
            hit_chance=_DEFAULT_HIT_CHANCE_ATTACKER,
        )
        damage_to_attacker = min(damage_to_attacker, float(attacker.current_hp))
    else:
        damage_to_attacker = 0.0

    # Death bonus: if defender HP <= expected damage, we kill them.
    # That's worth more than the raw HP number because we removed a
    # piece from the enemy's army. Weight by unit cost as a proxy for
    # strategic value.
    net = damage_to_defender - damage_to_attacker
    if damage_to_defender >= defender.current_hp:
        net += max(8.0, defender.cost * 0.3)  # "killed a unit" bonus
    # Symmetric penalty if WE'D die.
    if damage_to_attacker >= attacker.current_hp:
        net -= max(12.0, attacker.cost * 0.4)  # losing own unit is worse

    return net


__all__ = ["expected_attack_net_damage"]
