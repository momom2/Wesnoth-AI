"""Exact combat-outcome enumeration for MCTS chance nodes (Tier 1).

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

import combat as cb
from classes import GameState, Unit

log = logging.getLogger("combat_outcomes")

# (a_hp, d_hp, a_slowed, d_slowed, a_poisoned, d_poisoned)
OutcomeKey = Tuple[int, int, bool, bool, bool, bool]

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
    slow/poison state is meaningless and must not split outcomes."""
    a_hp, d_hp, a_sl, d_sl, a_po, d_po = key
    if a_hp <= 0:
        a_sl = a_po = False
    if d_hp <= 0:
        d_sl = d_po = False
    return (max(0, a_hp), max(0, d_hp), a_sl, d_sl, a_po, d_po)


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
    from tools.wesnoth_sim import choose_counter_weapon

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
    d_weapon = choose_counter_weapon(att, dfd, a_weapon)

    ctx = build_attack_context(gs, att, dfd, a_weapon, d_weapon)
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

    # --- soundness guards: fall back to sampling -----------------
    if a_stats.petrifies or (d_stats is not None and d_stats.petrifies):
        return None
    a_cu, d_cu = ctx.att_cu, ctx.dfd_cu
    if (a_cu.experience + _max_xp_gain(a_cu.level, d_cu.level)
            >= a_cu.max_experience):
        return None
    if (d_cu.experience + _max_xp_gain(d_cu.level, a_cu.level)
            >= d_cu.max_experience):
        return None

    schedule = _build_schedule(a_stats, d_stats)
    if schedule is None:
        return None

    # --- the DP ---------------------------------------------------
    # State: (a_hp, d_hp, a_slowed, d_slowed, a_poisoned, d_poisoned).
    # Dead states (hp == 0 either side) are absorbing.
    init: OutcomeKey = (
        a_cu.hp, d_cu.hp,
        a_cu.is_slowed, d_cu.is_slowed,
        a_cu.is_poisoned, d_cu.is_poisoned,
    )
    states: Dict[OutcomeKey, float] = {init: 1.0}

    for attacker_strikes in schedule:
        new: Dict[OutcomeKey, float] = {}

        def _add(key: OutcomeKey, p: float) -> None:
            new[key] = new.get(key, 0.0) + p

        live_mass = 0.0
        for key, p in states.items():
            a_hp, d_hp, a_sl, d_sl, a_po, d_po = key
            if a_hp <= 0 or d_hp <= 0:
                _add(key, p)          # fight already over: absorb
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
            dmg = st.slow_damage if st_sl else st.damage
            if dmg <= 0:
                # A hit that deals no damage applies no statuses
                # either (mirrors the early `return True`).
                _add(key, p * cth)
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
            if attacker_strikes:
                nkey = (st_hp_new, tg_hp_new, a_sl2, d_sl2, a_po2, d_po2)
            else:
                nkey = (tg_hp_new, st_hp_new, a_sl2, d_sl2, a_po2, d_po2)
            _add(_canonical(nkey) if (tg_hp_new <= 0) else nkey, p * cth)

        states = new
        if len(states) > MAX_DP_STATES:
            return None
        if live_mass <= PROB_EPSILON:
            break   # everything absorbed; later strikes are no-ops

    # Canonicalize, drop dust, renormalize.
    probs: Dict[OutcomeKey, float] = {}
    for key, p in states.items():
        if p < PROB_EPSILON:
            continue
        ck = _canonical(key)
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


def outcome_key_for_child(
    child_gs:    GameState,
    attacker_id: object,
    defender_id: object,
) -> OutcomeKey:
    """Extract the outcome key realized by a sampled successor
    state. A dead unit is simply absent from the child's unit set
    (hp 0, flags canonicalized)."""
    def _of(uid) -> Tuple[int, bool, bool]:
        u = next((x for x in child_gs.map.units if x.id == uid), None)
        if u is None:
            return 0, False, False
        return (u.current_hp,
                "slowed" in u.statuses,
                "poisoned" in u.statuses)
    a_hp, a_sl, a_po = _of(attacker_id)
    d_hp, d_sl, d_po = _of(defender_id)
    return _canonical((a_hp, d_hp, a_sl, d_sl, a_po, d_po))
