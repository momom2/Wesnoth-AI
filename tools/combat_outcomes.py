"""Exact combat-outcome enumeration for MCTS chance nodes (Tier 1)
and exact counter-weapon selection (`choose_counter_weapon`).

Mirrors Wesnoth's own attack-prediction approach (see
docs/wesnoth_rules.md "Combat-outcome prediction": a sparse DP over
(attacker_hp, defender_hp) with slow-state planes), but implemented
over OUR combat semantics: the per-strike transition function below
is a probability-space transcription of `combat._perform_hit` /
`combat.resolve_attack`, and all fight parameters come from the SAME
`replay_dataset.build_attack_context` + `combat._compute_battle_stats`
the bit-exact resolver uses -- parameter drift is impossible by
construction. test_combat_outcomes.py additionally cross-checks the
DP against empirical distributions from salted sim sampling.

The counter-weapon chooser at the bottom is a faithful port of
`battle_context::choose_defender_weapon` (1.18.4 attack.cpp),
reusing the same DP to stand in for the engine's combatant
simulation; it decides which weapon a defender retaliates with for
every sim-originated attack.

Where the engine truncates (berserk rounds at 99% dead mass) or
switches to Monte-Carlo (fight_complexity > 50,000), we instead
return None and let the chance-node machinery keep sampling through
the real sim -- the caller's fallback IS Monte-Carlo, so no second
implementation is needed.

Outcome key: (a_hp, d_hp, a_slowed, d_slowed, a_poisoned, d_poisoned)
with a dead unit's flags canonicalized to False. Everything else the
fight determines (XP, plague corpse spawn, death) is a deterministic
function of the key given the pre-fight state, so the key uniquely
identifies the successor game state. Fights that could trigger an
ADVANCEMENT are refused (return None): the advanced unit's HP would
diverge from the DP's accounting, so those fall back to sampling.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

from wesnoth_ai import combat as cb
from wesnoth_ai.classes import GameState, Unit

log = logging.getLogger("combat_outcomes")

# (a_hp, d_hp, a_slowed, d_slowed, a_poisoned, d_poisoned,
#  a_petrified, d_petrified)
OutcomeKey = Tuple[int, int, bool, bool, bool, bool, bool, bool]

# Bail-out caps. The engine's threshold is on a different quantity
# (hp_a * hp_b * slow planes > 50,000); ours bound the actual DP
# work: live cells per step and total strike-events (berserk rounds
# multiply the schedule). Typical fights: <= ~40 cells, <= 10 events.
MAX_DP_STATES = 4096
MAX_SCHEDULE  = 512

# Probabilities below this are dropped and the rest renormalized --
# the engine's round_prob_if_close_to_sure analog (it snaps at 1e-9;
# we are slightly more lenient because our consumers renormalize).
PROB_EPSILON = 1e-12


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
    never a petrify -- see combat._perform_hit_body -- so a_hp<=0 with
    a_petrified never arises; canonicalizing is a belt-and-braces.)"""
    a_hp, d_hp, a_sl, d_sl, a_po, d_po, a_pe, d_pe = key
    if a_hp <= 0:
        a_sl = a_po = a_pe = False
    if d_hp <= 0:
        d_sl = d_po = d_pe = False
    return (max(0, a_hp), max(0, d_hp), a_sl, d_sl, a_po, d_po, a_pe, d_pe)


def _build_schedule(a_stats, d_stats) -> Optional[List[bool]]:
    """Ordered strike events (True = attacker strikes), mirroring
    `resolve_attack`'s loop exactly: alternate attacker/defender
    while either has strikes left, defender first only on
    firststrike, berserk refills both sides `rounds`-1 times."""
    defender_first = (
        d_stats is not None
        and d_stats.firststrike
        and not a_stats.firststrike
    )
    rounds_left = max(a_stats.rounds,
                      d_stats.rounds if d_stats else 1) - 1
    a_n = a_stats.n_attacks
    d_n = d_stats.n_attacks if d_stats else 0
    schedule: List[bool] = []
    while True:
        if len(schedule) > MAX_SCHEDULE:
            return None
        if not defender_first and a_n > 0:
            schedule.append(True)
            a_n -= 1
        defender_first = False
        if d_n > 0:
            schedule.append(False)
            d_n -= 1
        if rounds_left > 0 and a_n == 0 and d_n == 0:
            a_n = a_stats.n_attacks
            d_n = d_stats.n_attacks if d_stats else 0
            rounds_left -= 1
            defender_first = (
                d_stats is not None
                and d_stats.firststrike
                and not a_stats.firststrike
            )
            continue
        if a_n <= 0 and d_n <= 0:
            break
    return schedule


def _max_xp_gain(unit_level: int, opp_level: int) -> int:
    """The largest XP the unit could gain from this fight: the kill
    award (combat XP is always smaller). Mirrors resolve_attack's
    awarding (KILL_EXPERIENCE * level, level-0 special-cased)."""
    kill = (cb.KILL_EXPERIENCE * opp_level
            if opp_level else cb.KILL_EXPERIENCE // 2)
    combat = cb.COMBAT_EXPERIENCE * opp_level
    return max(kill, combat)


# ---------------------------------------------------------------------
# Shared strike DP
# ---------------------------------------------------------------------
# Extended state: OutcomeKey + (a_touched, d_touched). The touched
# flags exist for the counter-weapon chooser, which needs P(hit at
# least once) for the engine's poison-probability formula; with
# track_touched=False they stay False, so outcome enumeration pays
# no extra states for them.
# (a_hp, d_hp, a_sl, d_sl, a_po, d_po, a_petrified, d_petrified, a_t, d_t)
_ExtKey = Tuple[int, int, bool, bool, bool, bool, bool, bool, bool, bool]


def _canonical_ext(key: _ExtKey) -> _ExtKey:
    """Zero a dead unit's status flags (see `_canonical`). Touched
    flags are kept: death implies the unit was hit, and the chooser's
    marginals must count that mass."""
    a_hp, d_hp, a_sl, d_sl, a_po, d_po, a_pe, d_pe, a_t, d_t = key
    if a_hp <= 0:
        a_sl = a_po = a_pe = False
    if d_hp <= 0:
        d_sl = d_po = d_pe = False
    return (max(0, a_hp), max(0, d_hp), a_sl, d_sl, a_po, d_po,
            a_pe, d_pe, a_t, d_t)


def _strike_dp(
    a_stats, d_stats, a_cu, d_cu,
    track_touched: bool = False,
) -> Optional[Dict[_ExtKey, float]]:
    """Run the per-strike probability DP over the fight schedule.
    Returns the final extended-state distribution, or None past the
    complexity caps (caller falls back to sampling / heuristic)."""
    schedule = _build_schedule(a_stats, d_stats)
    if schedule is None:
        return None

    init: _ExtKey = (
        a_cu.hp, d_cu.hp,
        a_cu.is_slowed, d_cu.is_slowed,
        a_cu.is_poisoned, d_cu.is_poisoned,
        a_cu.is_petrified, d_cu.is_petrified,
        False, False,
    )
    states: Dict[_ExtKey, float] = {init: 1.0}

    for attacker_strikes in schedule:
        new: Dict[_ExtKey, float] = {}

        def _add(key: _ExtKey, p: float) -> None:
            new[key] = new.get(key, 0.0) + p

        live_mass = 0.0
        for key, p in states.items():
            a_hp, d_hp, a_sl, d_sl, a_po, d_po, a_pe, d_pe, a_t, d_t = key
            if a_hp <= 0 or d_hp <= 0 or a_pe or d_pe:
                _add(key, p)          # fight over (death or petrify): absorb
                continue
            live_mass += p
            if attacker_strikes:
                st, st_hp, st_sl = a_stats, a_hp, a_sl
                tg, tg_hp = d_stats, d_hp
                tg_cu, st_cu = d_cu, a_cu
            else:
                st, st_hp, st_sl = d_stats, d_hp, d_sl
                tg, tg_hp = a_stats, a_hp
                tg_cu, st_cu = a_cu, d_cu

            cth = max(0, min(100, st.cth)) / 100.0
            # Miss branch.
            if cth < 1.0:
                _add(key, p * (1.0 - cth))
            if cth <= 0.0:
                continue
            # Hit branch -- transcription of combat._perform_hit.
            # Any landed strike marks the target "touched" (the
            # engine counts hits, not damage, for poison odds).
            a_t2, d_t2 = a_t, d_t
            if track_touched:
                if attacker_strikes:
                    d_t2 = True
                else:
                    a_t2 = True
            dmg = st.slow_damage if st_sl else st.damage
            if dmg <= 0:
                # A hit that deals no damage applies no statuses
                # either (mirrors the early `return True`).
                _add((a_hp, d_hp, a_sl, d_sl, a_po, d_po, a_pe, d_pe,
                      a_t2, d_t2),
                     p * cth)
                continue
            tg_hp_new = max(0, tg_hp - dmg)
            damage_done = tg_hp - tg_hp_new
            st_hp_new = st_hp
            if (st.drains and damage_done > 0
                    and not tg_cu.is_undrainable):
                heal = (damage_done * st.drain_percent // 100
                        + st.drain_constant)
                if heal != 0:
                    heal = min(heal, st_cu.max_hp - st_hp)
                    heal = max(heal, 1 - st_hp)
                    st_hp_new = st_hp + heal
            # Status application only when the target survives.
            a_sl2, d_sl2, a_po2, d_po2 = a_sl, d_sl, a_po, d_po
            a_pe2, d_pe2 = a_pe, d_pe
            if tg_hp_new > 0:
                if st.poisons and not tg_cu.is_unpoisonable:
                    if attacker_strikes:
                        d_po2 = True
                    else:
                        a_po2 = True
                if st.slows:
                    if attacker_strikes:
                        d_sl2 = True
                    else:
                        a_sl2 = True
                # Petrify: a surviving petrifying hit turns the target
                # to stone and ENDS the fight -- both sides' remaining
                # strikes forfeit (combat._perform_hit_body sets
                # is_petrified, n_attacks 0/-1; attack.cpp sets
                # STATE_PETRIFIED). We only flag it here; the resulting
                # state is terminal and gets frozen against the rest of
                # the schedule by the absorb check at the top of the
                # loop. A KILLING petrify blow is a death, not a petrify
                # (this block is survive-only) -- matches the engine.
                if st.petrifies:
                    if attacker_strikes:
                        d_pe2 = True
                    else:
                        a_pe2 = True
            if attacker_strikes:
                nkey = (st_hp_new, tg_hp_new,
                        a_sl2, d_sl2, a_po2, d_po2, a_pe2, d_pe2, a_t2, d_t2)
            else:
                nkey = (tg_hp_new, st_hp_new,
                        a_sl2, d_sl2, a_po2, d_po2, a_pe2, d_pe2, a_t2, d_t2)
            _add(_canonical_ext(nkey) if (tg_hp_new <= 0) else nkey,
                 p * cth)

        states = new
        if len(states) > MAX_DP_STATES:
            return None
        if live_mass <= PROB_EPSILON:
            break   # everything absorbed; later strikes are no-ops

    return states


def enumerate_attack_outcomes(
    gs:     GameState,
    action: dict,
) -> Optional[OutcomeDistribution]:
    """Exact outcome distribution for an attack action dict
    ({"type": "attack", "start_hex", "target_hex", "attack_index"}),
    or None when enumeration is unsound/too expensive and the caller
    should sample instead (petrify, possible advancement, complexity
    caps, missing units)."""
    from tools.replay_dataset import build_attack_context

    start = action.get("start_hex")
    target = action.get("target_hex")
    if start is None or target is None:
        return None
    att = next((u for u in gs.map.units
                if u.position.x == start.x and u.position.y == start.y),
               None)
    dfd = next((u for u in gs.map.units
                if u.position.x == target.x and u.position.y == target.y),
               None)
    if att is None or dfd is None:
        return None

    # Same counter-weapon resolution the sim applies at
    # command-build time.
    a_weapon = int(action.get("attack_index", 0))
    d_weapon = choose_counter_weapon(gs, att, dfd, a_weapon)

    ctx = build_attack_context(gs, att, dfd, a_weapon, d_weapon)
    a_stats, d_stats = _stats_pair(ctx)

    # --- soundness guards: fall back to sampling -----------------
    # Petrify is now modeled exactly in _strike_dp (a surviving
    # petrifying hit ends the fight with the target stoned), so it no
    # longer bails. Possible ADVANCEMENT still returns None here --
    # MCTS samples those fights, and the swap detector opts in
    # separately (docs/swap_detector_design.md); threading the
    # advancement-choice distribution through the DP is a later step.
    a_cu, d_cu = ctx.att_cu, ctx.dfd_cu
    if (a_cu.experience + _max_xp_gain(a_cu.level, d_cu.level)
            >= a_cu.max_experience):
        return None
    if (d_cu.experience + _max_xp_gain(d_cu.level, a_cu.level)
            >= d_cu.max_experience):
        return None

    states = _strike_dp(a_stats, d_stats, a_cu, d_cu)
    if states is None:
        return None

    # Fold away the (constant-False) touched flags, canonicalize,
    # drop dust, renormalize.
    probs: Dict[OutcomeKey, float] = {}
    for key, p in states.items():
        if p < PROB_EPSILON:
            continue
        ck = _canonical(key[:8])
        probs[ck] = probs.get(ck, 0.0) + p
    total = sum(probs.values())
    if not probs or total <= 0:
        return None
    probs = {k: p / total for k, p in probs.items()}
    return OutcomeDistribution(
        probs=probs,
        attacker_id=att.id,
        defender_id=dfd.id,
    )


def _stats_pair(ctx) -> Tuple["cb.BattleStats", Optional["cb.BattleStats"]]:
    """Both sides' BattleStats for an AttackContext (defender None
    when not retaliating) -- the exact stats resolve_attack uses."""
    a_stats = cb._compute_battle_stats(
        ctx.att_cu, ctx.dfd_cu, ctx.a_weapon,
        ctx.d_weapon if ctx.d_weapon >= 0 else None,
        ctx.a_lawful, ctx.d_lawful,
        leadership_bonus=ctx.a_leadership,
        is_attacker=True,
        backstab_active=ctx.a_backstab,
    )
    d_stats = (
        cb._compute_battle_stats(
            ctx.dfd_cu, ctx.att_cu, ctx.d_weapon, ctx.a_weapon,
            ctx.d_lawful, ctx.a_lawful,
            leadership_bonus=ctx.d_leadership,
            is_attacker=False,
            backstab_active=ctx.d_backstab,
        )
        if ctx.d_weapon is not None and ctx.d_weapon >= 0
        else None
    )
    return a_stats, d_stats


def outcome_key_for_child(
    child_gs:    GameState,
    attacker_id: object,
    defender_id: object,
) -> OutcomeKey:
    """Extract the outcome key realized by a sampled successor
    state. A dead unit is simply absent from the child's unit set
    (hp 0, flags canonicalized)."""
    def _of(uid) -> Tuple[int, bool, bool, bool]:
        u = next((x for x in child_gs.map.units if x.id == uid), None)
        if u is None:
            return 0, False, False, False
        return (u.current_hp,
                "slowed" in u.statuses,
                "poisoned" in u.statuses,
                "petrified" in u.statuses)
    a_hp, a_sl, a_po, a_pe = _of(attacker_id)
    d_hp, d_sl, d_po, d_pe = _of(defender_id)
    return _canonical((a_hp, d_hp, a_sl, d_sl, a_po, d_po, a_pe, d_pe))


# ---------------------------------------------------------------------
# Counter-weapon selection
# ---------------------------------------------------------------------
# Port of battle_context::choose_defender_weapon + better_defense /
# better_combat + calculate_probability_of_debuff, all 1.18.4
# (src/actions/attack.cpp, src/attack_prediction.cpp; fetched
# verbatim 2026-06-12). The engine's `combatant` simulation is
# replaced by our exact strike DP, which yields the same marginals
# (death probability, average_hp, touched probability).

@dataclass
class _CombatantMarginals:
    """The combatant-simulation outputs better_combat consumes."""
    death:    float    # hp_dist[0]
    avg_hp:   float    # average_hp(0): sum p*hp over alive states
    poisoned: float    # engine debuff formula, level-up cure applied


def _probability_of_debuff(
    initial_prob:    float,
    enemy_gives:     bool,
    prob_touched:    float,
    prob_stay_alive: float,
    kill_heals:      bool,
    prob_kill:       float,
) -> float:
    """Verbatim port of `calculate_probability_of_debuff`
    (1.18.4 src/attack_prediction.cpp): post-fight probability of
    carrying a debuff, where leveling up on a kill cures it."""
    prob_touched = max(prob_touched, 0.0)
    prob_stay_alive = max(prob_stay_alive, 0.0)
    prob_kill = min(max(prob_kill, 0.0), 1.0)

    prob_already_debuffed_not_touched = initial_prob * (1.0 - prob_touched)
    prob_already_debuffed_touched = initial_prob * prob_touched
    prob_initially_healthy_touched = (1.0 - initial_prob) * prob_touched

    prob_survive_if_not_hit = 1.0
    prob_survive_if_hit = (
        (prob_stay_alive - (1.0 - prob_touched)) / prob_touched
        if prob_touched > 0.0 else 1.0)
    prob_kill_if_survive = (
        prob_kill / prob_stay_alive if prob_stay_alive > 0.0 else 0.0)

    prob_debuff = 0.0
    if not kill_heals:
        prob_debuff += prob_already_debuffed_not_touched
    else:
        prob_debuff += (prob_already_debuffed_not_touched
                        * (1.0 - prob_survive_if_not_hit
                           * prob_kill_if_survive))
    if not kill_heals:
        prob_debuff += prob_already_debuffed_touched
    else:
        prob_debuff += (prob_already_debuffed_touched
                        * (1.0 - prob_survive_if_hit
                           * prob_kill_if_survive))
    # "Originally not debuffed, not hit" never debuffs us.
    if not enemy_gives:
        pass
    elif not kill_heals:
        prob_debuff += prob_initially_healthy_touched
    else:
        prob_debuff += (prob_initially_healthy_touched
                        * (1.0 - prob_survive_if_hit
                           * prob_kill_if_survive))
    return prob_debuff


def _kill_xp(opp_level: int) -> int:
    """game_config::kill_xp -- mirrors resolve_attack's award."""
    return (cb.KILL_EXPERIENCE * opp_level
            if opp_level else cb.KILL_EXPERIENCE // 2)


def _engine_marginals(
    states, a_stats, d_stats, a_cu, d_cu,
) -> Tuple[_CombatantMarginals, _CombatantMarginals]:
    """(attacker, defender) marginals from a touched-tracked DP,
    matching what `combatant::fight` computes: exact death/avg_hp,
    and `poisoned` via the engine's own approximation formula fed
    with our exact touched probability. The engine approximates
    P(hit at least once) incrementally; ours is exact -- a documented
    (and strictly smaller-error) deviation."""
    a_death = d_death = a_avg = d_avg = a_touch = d_touch = 0.0
    for (a_hp, d_hp, _asl, _dsl, _apo, _dpo, _ape, _dpe,
         a_t, d_t), p in states.items():
        if a_hp <= 0:
            a_death += p
        else:
            a_avg += p * a_hp
        if d_hp <= 0:
            d_death += p
        else:
            d_avg += p * d_hp
        if a_t:
            a_touch += p
        if d_t:
            d_touch += p

    a_pois = _probability_of_debuff(
        1.0 if a_cu.is_poisoned else 0.0,
        bool(d_stats is not None and d_stats.poisons
             and not a_cu.is_unpoisonable),
        a_touch, 1.0 - a_death,
        a_cu.experience + _kill_xp(d_cu.level) >= a_cu.max_experience,
        d_death)
    d_pois = _probability_of_debuff(
        1.0 if d_cu.is_poisoned else 0.0,
        bool(a_stats.poisons and not d_cu.is_unpoisonable),
        d_touch, 1.0 - d_death,
        d_cu.experience + _kill_xp(a_cu.level) >= d_cu.max_experience,
        a_death)
    # Level-up cure: combat XP alone reaching max_experience wipes
    # debuffs (combatant::fight does this AFTER the formula).
    if (a_cu.experience + cb.COMBAT_EXPERIENCE * d_cu.level
            >= a_cu.max_experience):
        a_pois = 0.0
    if (d_cu.experience + cb.COMBAT_EXPERIENCE * a_cu.level
            >= d_cu.max_experience):
        d_pois = 0.0
    return (_CombatantMarginals(a_death, a_avg, a_pois),
            _CombatantMarginals(d_death, d_avg, d_pois))


def _better_combat(
    us_a: _CombatantMarginals, them_a: _CombatantMarginals,
    us_b: _CombatantMarginals, them_b: _CombatantMarginals,
    harm_weight: float,
) -> bool:
    """Verbatim port of battle_context::better_combat (1.18.4
    attack.cpp): is fight A better for "us" than fight B?"""
    # Compare: P(we kill them) - P(they kill us).
    a = them_a.death - us_a.death * harm_weight
    b = them_b.death - us_b.death * harm_weight
    if a - b < -0.01:
        return False
    if a - b > 0.01:
        return True
    # Add poison, but only the mass that survives the fight.
    poison_a_us = ((us_a.poisoned - us_a.death) * cb.POISON_AMOUNT
                   if us_a.poisoned > 0 else 0.0)
    poison_a_them = ((them_a.poisoned - them_a.death) * cb.POISON_AMOUNT
                     if them_a.poisoned > 0 else 0.0)
    poison_b_us = ((us_b.poisoned - us_b.death) * cb.POISON_AMOUNT
                   if us_b.poisoned > 0 else 0.0)
    poison_b_them = ((them_b.poisoned - them_b.death) * cb.POISON_AMOUNT
                     if them_b.poisoned > 0 else 0.0)
    # Compare: damage to them - damage to us.
    a = ((us_a.avg_hp - poison_a_us) * harm_weight
         - (them_a.avg_hp - poison_a_them))
    b = ((us_b.avg_hp - poison_b_us) * harm_weight
         - (them_b.avg_hp - poison_b_them))
    if a - b < -0.01:
        return False
    if a - b > 0.01:
        return True
    # All else equal: go for most damage.
    return them_a.avg_hp < them_b.avg_hp


def _fallback_counter_weapon(d_stats_by_idx: Dict[int, object]) -> int:
    """DP-overflow fallback (huge berserk/swarm fights the engine
    itself would hand to Monte-Carlo): max damage x strikes among
    the candidates, ties to the lowest index -- the pre-port v1
    heuristic, kept deterministic where the engine is randomized."""
    best_idx, best_score = -1, -1
    for i in sorted(d_stats_by_idx):
        st = d_stats_by_idx[i]
        score = st.damage * st.n_attacks
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx


def choose_counter_weapon(gs: GameState, att: Unit, dfd: Unit,
                          a_weapon_idx: int) -> int:
    """Defender's counter-attack weapon for a sim-originated attack:
    faithful port of battle_context::choose_defender_weapon (1.18.4
    attack.cpp). Returns -1 when the defender cannot retaliate.

    History: v1 (2026-06-12, after the retaliation bug) approximated
    with max damage x strikes over matching-range weapons; this port
    replaces it so the sim retaliates with the same weapon live
    Wesnoth would pick. Exported replays record the result either
    way (playback uses the recorded index, not the engine chooser).

    Engine quirk kept verbatim: the min_rating pass assigns
    max_weight BEFORE testing `weight > max_weight`, so that test is
    always false and min_rating never leaves 0 -- the eligibility
    filter is dead code in 1.18.4, and defense_weight has no effect
    beyond its `> 0` candidate filter. Consequently we treat
    defense_weight as 1.0 everywhere: the pinned 1.18.4 scrape
    doesn't carry the attribute, and the only mainline setters
    (Giant Scorpion / Scorpling sting, defense_weight=4.0,
    wesnoth_src/data/core/units/monsters/) cannot influence the
    choice through the dead filter anyway.

    Other documented deviations, all outside the training pools:
    [disable] specials are unmodeled (no default-era weapon has
    one), and DP-overflow fights fall back to the v1 heuristic
    where the engine would switch its combatant sim to Monte-Carlo.
    """
    from tools.replay_dataset import build_attack_context

    if (not getattr(att, "attacks", None)
            or not getattr(dfd, "attacks", None)):
        return -1
    # Petrified defenders can't retaliate (their attacks are
    # stripped engine-side; build_attack_context forces -1 too).
    if "petrified" in dfd.statuses:
        return -1
    if a_weapon_idx >= len(att.attacks):
        a_weapon_idx = 0    # mirror build_attack_context's clamp

    # Snapshot once (d_weapon=-1) for range filtering; weapon lists
    # and indices match what combat will use by construction.
    base = build_attack_context(gs, att, dfd, a_weapon_idx, -1)
    if a_weapon_idx >= len(base.att_cu.weapons):
        return -1
    att_range = base.att_cu.weapons[base.a_weapon].range

    # What options does defender have? (range match; defense_weight
    # > 0 always true -- see docstring.)
    candidates = [i for i, w in enumerate(base.dfd_cu.weapons)
                  if w.range == att_range]
    if not candidates:
        return -1
    if len(candidates) == 1:
        # Only one usable weapon, don't simulate.
        return candidates[0]

    # Multiple options: simulate each candidate fight.
    ctxs = {i: build_attack_context(gs, att, dfd, a_weapon_idx, i)
            for i in candidates}
    stats = {i: _stats_pair(ctxs[i]) for i in candidates}
    d_stats_by_idx = {i: stats[i][1] for i in candidates}

    sims: Dict[int, Tuple[_CombatantMarginals, _CombatantMarginals]] = {}
    for i in candidates:
        a_stats, d_stats = stats[i]
        states = _strike_dp(a_stats, d_stats,
                            ctxs[i].att_cu, ctxs[i].dfd_cu,
                            track_touched=True)
        if states is None:
            return _fallback_counter_weapon(d_stats_by_idx)
        sims[i] = _engine_marginals(states, a_stats, d_stats,
                                    ctxs[i].att_cu, ctxs[i].dfd_cu)

    # First pass: best weight + minimum simple rating for it.
    # simple rating = blows * damage * cth * weight. Quirk kept
    # verbatim (see docstring): min_rating stays 0.
    min_rating = 0
    max_weight = 0.0
    for i in candidates:
        weight = 1.0
        if weight >= max_weight:
            st = d_stats_by_idx[i]
            max_weight = weight
            rating = int(st.n_attacks * st.damage * st.cth * weight)
            if weight > max_weight or rating < min_rating:
                min_rating = rating

    # Second pass: among eligible ratings, keep the better_defense
    # winner (us = defender, them = attacker; harm_weight 1.0).
    best_idx = -1
    for i in candidates:
        st = d_stats_by_idx[i]
        simple_rating = int(st.n_attacks * st.damage * st.cth * 1.0)
        att_m, dfd_m = sims[i]
        if simple_rating >= min_rating and (
                best_idx < 0
                or _better_combat(dfd_m, att_m,
                                  sims[best_idx][1], sims[best_idx][0],
                                  1.0)):
            best_idx = i
    return best_idx
