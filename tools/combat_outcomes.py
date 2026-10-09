"""Exact combat-outcome enumeration for MCTS chance nodes (Tier 1)
and exact counter-weapon selection (`choose_counter_weapon`).

The Rust core answers both (rust/wesnoth_core/src/outcomes.rs) for the
core behind a state (`game_core.core_for`: the core a view is bound
to, or one built from a state no core stands behind).

The enumeration mirrors Wesnoth's own attack-prediction approach (see
docs/wesnoth_rules.md "Combat-outcome prediction": a sparse DP over
(attacker_hp, defender_hp) with slow-state planes) over the core's
combat semantics: the per-strike transition is a probability-space
transcription of the fight the core plays, with the same fight
parameters, so parameter drift is impossible by construction.
test_combat_outcomes.py cross-checks the DP against empirical
distributions from salted sim sampling.

The counter-weapon chooser is a faithful port of
`battle_context::choose_defender_weapon` (1.18.4 attack.cpp), reusing
the same DP to stand in for the engine's combatant simulation; it
decides which weapon a defender retaliates with for every
sim-originated attack.

Where the engine truncates (berserk rounds at 99% dead mass) or
switches to Monte-Carlo (fight_complexity > 50,000), the DP instead
returns None and lets the chance-node machinery keep sampling through
the real sim -- the caller's fallback IS Monte-Carlo, so no second
implementation is needed.

Outcome key: (a_hp, d_hp, a_slowed, d_slowed, a_poisoned, d_poisoned,
a_petrified, d_petrified, a_type, d_type) with a dead unit's flags
canonicalized to False. Everything else the fight determines (XP,
plague corpse spawn, death) is a deterministic function of the key
given the pre-fight state, so the key uniquely identifies the
successor game state. Without an advancement choice, fights that could
trigger an ADVANCEMENT are refused (return None).
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

from wesnoth_ai.classes import GameState, Unit

log = logging.getLogger("combat_outcomes")

# (a_hp, d_hp, a_slowed, d_slowed, a_poisoned, d_poisoned,
#  a_petrified, d_petrified, a_type, d_type)
# a_type/d_type are the units' type-names (u.name -- the same "cheap
# proxy for unit type" state_key uses). They change when a unit
# ADVANCES, disambiguating advancement targets even when post-advance
# HP collides; "" for a dead (absent) unit. Constant for non-advancement
# fights, so they don't split those outcomes.
OutcomeKey = Tuple[int, int, bool, bool, bool, bool, bool, bool, str, str]


@dataclass
class OutcomeDistribution:
    """Exact probability over combat outcomes, plus the combatant
    ids needed to extract a matching key from a sampled child
    state."""
    probs:       Dict[OutcomeKey, float]
    attacker_id: object
    defender_id: object


def _canonical(key: OutcomeKey) -> OutcomeKey:
    """Zero a dead unit's status flags: the unit is gone, so its
    slow/poison/petrified state is meaningless and must not split
    outcomes. (A killing blow with the petrifies special is a death,
    never a petrify, so a_hp<=0 with a_petrified never arises;
    canonicalizing is a belt-and-braces.)"""
    a_hp, d_hp, a_sl, d_sl, a_po, d_po, a_pe, d_pe, a_ty, d_ty = key
    if a_hp <= 0:
        a_sl = a_po = a_pe = False
        a_ty = ""            # dead unit: type is meaningless / absent
    if d_hp <= 0:
        d_sl = d_po = d_pe = False
        d_ty = ""
    return (max(0, a_hp), max(0, d_hp), a_sl, d_sl, a_po, d_po,
            a_pe, d_pe, a_ty, d_ty)


def _core(gs: GameState):
    """The `wesnoth_core.GameCore` that answers for `gs`."""
    from wesnoth_ai.game_core import core_for
    return core_for(gs).core


def enumerate_attack_outcomes(
    gs:     GameState,
    action: dict,
    *,
    advancement_choice=None,
) -> Optional[OutcomeDistribution]:
    """Exact outcome distribution for an attack action dict
    ({"type": "attack", "start_hex", "target_hex", "attack_index"}),
    or None when enumeration is unsound/too expensive and the caller
    should sample instead (complexity caps, missing units, or -- when
    `advancement_choice` is None -- a fight where a unit could advance).

    `advancement_choice`: None, or "uniform" (matches self-play): an
    outcome crossing an XP threshold is then RESOLVED into the
    advancement chain (new type + full HP, each target equally likely),
    so both the swap detector and MCTS's exact path see advancement
    rather than bailing. Petrify is always modeled exactly."""
    if advancement_choice not in (None, "uniform"):
        raise ValueError(f"advancement_choice is None or 'uniform', not {advancement_choice!r}")
    start = action.get("start_hex")
    target = action.get("target_hex")
    if start is None or target is None:
        return None
    res = _core(gs).attack_outcomes(start.x, start.y, target.x, target.y,
                                    int(action.get("attack_index", 0)),
                                    advancement_choice == "uniform")
    if res is None:
        return None
    probs, attacker_id, defender_id = res
    return OutcomeDistribution(probs=probs, attacker_id=attacker_id,
                               defender_id=defender_id)


def outcome_key_for_child(
    child_gs:    GameState,
    attacker_id: object,
    defender_id: object,
) -> OutcomeKey:
    """Extract the outcome key realized by a sampled successor
    state. A dead unit is simply absent from the child's unit set
    (hp 0, flags canonicalized)."""
    def _of(uid) -> Tuple[int, bool, bool, bool, str]:
        u = next((x for x in child_gs.map.units if x.id == uid), None)
        if u is None:
            return 0, False, False, False, ""
        # u.name is the current type -- already the ADVANCED type if the
        # unit levelled up in this child, which is exactly what the DP's
        # advancement branch keys on.
        return (u.current_hp,
                "slowed" in u.statuses,
                "poisoned" in u.statuses,
                "petrified" in u.statuses,
                u.name)
    a_hp, a_sl, a_po, a_pe, a_ty = _of(attacker_id)
    d_hp, d_sl, d_po, d_pe, d_ty = _of(defender_id)
    return _canonical((a_hp, d_hp, a_sl, d_sl, a_po, d_po,
                       a_pe, d_pe, a_ty, d_ty))


# ---------------------------------------------------------------------
# Counter-weapon selection
# ---------------------------------------------------------------------
# The core's port of battle_context::choose_defender_weapon +
# better_defense / better_combat + calculate_probability_of_debuff, all
# 1.18.4 (src/actions/attack.cpp, src/attack_prediction.cpp). The
# engine's `combatant` simulation is replaced by the exact strike DP,
# which yields the same marginals (death probability, average_hp,
# touched probability).

# Counter-weapon choices that fell back to the v1 heuristic (a DP that
# overflowed). The fallback picks a DIFFERENT weapon than the engine's
# `choose_defender_weapon` would, so a fight resolved through it is not
# the fight Wesnoth would resolve -- and until 2026-09-13 it left no
# trace at all: no log line, no counter, no test. Silence is exactly how
# the hide-cover defect survived.
_FALLBACK_COUNTER_WEAPONS = 0


def fallback_counter_weapon_count() -> int:
    """How many counter-weapon choices used the v1 heuristic instead of
    the engine's rule in this process. Non-zero means some fights were
    resolved with a weapon Wesnoth would not have picked."""
    return _FALLBACK_COUNTER_WEAPONS


def _count_fallback() -> None:
    """Count a counter-weapon choice made by the fallback heuristic, a
    KNOWN divergence from `choose_defender_weapon`; warn the first time
    (see `fallback_counter_weapon_count`)."""
    global _FALLBACK_COUNTER_WEAPONS
    _FALLBACK_COUNTER_WEAPONS += 1
    if _FALLBACK_COUNTER_WEAPONS == 1:
        log.warning(
            "counter-weapon DP overflowed; falling back to the v1 "
            "damage x strikes heuristic, which picks a DIFFERENT weapon "
            "than Wesnoth's choose_defender_weapon. Fights resolved this "
            "way diverge from the engine. Further occurrences are counted "
            "silently (combat_outcomes.fallback_counter_weapon_count).")


def choose_counter_weapon(gs: GameState, att: Unit, dfd: Unit,
                          a_weapon_idx: int) -> int:
    """`counter_weapon_choice` without its strike tables."""
    return counter_weapon_choice(gs, att, dfd, a_weapon_idx)[0]


def counter_weapon_choice(gs: GameState, att: Unit, dfd: Unit,
                          a_weapon_idx: int) -> Tuple[int, Dict[int, dict]]:
    """(the defender's counter-attack weapon, the strike tables
    simulated to choose it: {defender weapon: the DP's final states},
    empty when one weapon or none could answer).

    Defender's counter-attack weapon for a sim-originated attack: the
    core's faithful port of battle_context::choose_defender_weapon
    (1.18.4 attack.cpp). Returns -1 when the defender cannot retaliate.

    History: v1 (2026-06-12, after the retaliation bug) approximated
    with max damage x strikes over matching-range weapons; the port
    makes the sim retaliate with the same weapon live Wesnoth would
    pick. Exported replays record the result either way (playback uses
    the recorded index, not the engine chooser).

    Engine quirk kept verbatim: the min_rating pass assigns
    max_weight BEFORE testing `weight > max_weight`, so that test is
    always false and min_rating never leaves 0 -- the eligibility
    filter is dead code in 1.18.4, and defense_weight has no effect
    beyond its `> 0` candidate filter. Consequently defense_weight is
    1.0 everywhere: the pinned 1.18.4 scrape doesn't carry the
    attribute, and the only mainline setters (Giant Scorpion /
    Scorpling sting, defense_weight=4.0,
    wesnoth_src/data/core/units/monsters/) cannot influence the choice
    through the dead filter anyway.

    Other documented deviations, outside the training pools:
    [disable] specials are unmodeled (no default-era weapon has
    one), and DP-overflow fights fall back to the v1 heuristic
    where the engine would switch its combatant sim to Monte-Carlo.
    Inside them: the strike DP runs every berserk round where the
    prediction stops at 99% dead mass, and counters with exactly
    equal predicted outcomes are decided by floating-point residue
    the DP does not reproduce bit for bit (4 of 737 recorded choices,
    training/metrics/fidelity/counter_weapon_census_20260925.json).
    """
    res = _core(gs).counter_weapon_choice(att.position.x, att.position.y,
                                          dfd.position.x, dfd.position.y,
                                          a_weapon_idx)
    if res is None:
        return -1, {}
    weapon, tables, fallback = res
    if fallback:
        _count_fallback()
    return weapon, tables


def defender_chance_to_hit(gs: GameState, att: Unit, dfd: Unit,
                           a_weapon_idx: int) -> Optional[int]:
    """The defender's chance to hit the attacker, in percent, with the
    weapon it answers with (`choose_counter_weapon`); None when it does
    not answer."""
    d_weapon = choose_counter_weapon(gs, att, dfd, a_weapon_idx)
    res = _core(gs).fight_stats(att.position.x, att.position.y,
                                dfd.position.x, dfd.position.y,
                                a_weapon_idx, d_weapon)
    if res is None or res[1] is None:
        return None
    return int(res[1]["cth"])
