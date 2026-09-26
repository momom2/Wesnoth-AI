"""Neutral-side (side >= 3) combat turn: Wesnoth's default RCA AI,
restricted to STATIONARY units (user-approved scope 2026-07-14).

For a unit that cannot move, Wesnoth's default AI reduces EXACTLY to
its combat candidate action over adjacent targets, repeated while the
best rating > 0 (src/ai/default/ca.cpp combat_phase, 1.18.4). So the
substitution is exact only where the neutral units really cannot move.

A neutral unit qualifies for one of exactly three reasons, all READ
as of 2026-09-22 rather than assumed:

  1. `ai_special=guardian`. It sets STATE_GUARDIAN (1.18.4
     src/units/unit.cpp:659), and the default AI's move phase then
     hands the unit a move from its own hex to its own hex --
     "is guardian, staying still"
     (src/ai/default/ca_move_to_targets.cpp:269-277). Stashed as
     `_ai_guardian` by `scenario_events._unit_action`.
  2. No movement left: the map pins it every `turn refresh`
     ({MODIFY_UNIT (role=monster) moves 0}).
  3. No landable hex: terrain-locked.

Six pool scenarios field an acting neutral side, and the precondition
holds on all six -- checked for the first time on 2026-09-22 by
`_check_units_are_stationary`, which runs at the start of every
neutral turn and is pinned by tests/test_neutral_ai_precondition.py:

  - 2p_mini, 2p_mini_edited, Modified_Tiny_Close_Relation: guardians.
    The first two are ALSO terrain-locked; Modified_Tiny_Close_Relation
    is not -- its Tentacle has full MP and two adjacent water hexes it
    can enter, and the only thing keeping it still is the guardian
    flag. The earlier version of this docstring justified the
    substitution by the pin and the terrain lock alone, which covered
    neither that map nor the real reason for the other two.
  - enclave_micro_isar, enclave_mini_fallenstar_1v1,
    enclave_small_fallenstar_1v1: pinned to 0 MP.

If a scenario ever fields a neutral unit that is none of the three,
`_check_units_are_stationary` warns (and raises under
`WESNOTH_STRICT_WML`) instead of this AI quietly rooting a unit
Wesnoth would walk.

Rating: verbatim port of src/ai/default/attack.cpp
attack_analysis::rating (1.18.4, lines ~298-345; fetched and pinned
2026-07-14 -- see docs/wesnoth_rules.md):

    value  = chance_to_kill*target_value - avg_losses*(1-aggression)
    [exposure term: EXACTLY zero for a stationary attacker --
     terrain_quality == alternative_terrain_quality by construction]
    value += (target_starting_damage/3 + avg_damage_inflicted
              - (1-aggression)*avg_damage_taken) / 10
    [support/vulnerability gates: v1 approximates
     vulnerability = support = 0, is_surrounded = False, so the
     multiplicative term (gated on support != 0) and the -1 abort
     (gated on vulnerability > 50) are both skipped. Porting
     power_projection would make these exact -- BACKLOG; the 1%
     replay-validation pipeline is the empirical arbiter.]
    value /= (resources_used/2 + (resources_used/2)*terrain_quality)
    if leader_threat: aggression = 1.0 (before the terms above);
                      value *= 5.0

Inputs come from the sim's EXACT combat distributions
(tools/combat_outcomes.enumerate_attack_outcomes), not
approximations:
    chance_to_kill        = P(defender hp 0)
    avg_damage_inflicted  = E[defender hp lost]
                            (analyze()'s poison bonus term is NOT
                            implemented -- tentacles cannot poison;
                            port it before any poisoning monster)
    avg_damage_taken      = E[attacker hp lost]
    avg_losses            = attacker_cost * P(attacker dies)
    target_value          = cost * (1 + xp/max_xp)
    resources_used        = attacker_cost (same xp scaling)
    terrain_quality       = (defender CTH vs attacker)/100
                            * (0.5 if attacker on village)
AI parameters come from the scenario's side [ai] block
(aggression; enclaves use 0.3), falling back to the engine defaults
(0.4). caution only enters via the exposure term, which is zero
here.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

log = logging.getLogger("neutral_ai")


class MobileNeutralUnit(RuntimeError):
    """A neutral unit the default AI would move and this one
    cannot. Raised under `WESNOTH_STRICT_WML`."""

# RCA engine defaults (readonly_context defaults, 1.18.4) -- used
# when the scenario's [side][ai] block doesn't override.
DEFAULT_AGGRESSION = 0.4

_AI_PARAM_CACHE: Dict[str, float] = {}


def _side_aggression(scenario_id: str, side: int) -> float:
    key = f"{scenario_id}:{side}"
    if key in _AI_PARAM_CACHE:
        return _AI_PARAM_CACHE[key]
    aggr = DEFAULT_AGGRESSION
    try:
        from wesnoth_ai.rules.scenario_cfg import load_scenario_wml
        root = load_scenario_wml(scenario_id)
        mp = (root.first("multiplayer") or root.first("scenario")
              if root is not None else None)
        if mp is not None:
            for sn in mp.all("side"):
                if sn.attrs.get("side", "").strip() == str(side):
                    ai = sn.first("ai")
                    if ai is not None and "aggression" in ai.attrs:
                        aggr = float(ai.attrs["aggression"].strip())
                    break
    except Exception:                                 # noqa: BLE001
        pass
    _AI_PARAM_CACHE[key] = aggr
    return aggr


def _xp_scaled_cost(u) -> float:
    mx = max(1, int(getattr(u, "max_exp", 1) or 1))
    return float(u.cost) * (1.0 + float(u.current_exp) / mx)


def _attacker_on_village(gs, u) -> bool:
    from wesnoth_ai.sim.classes import Terrain
    for h in gs.map.hexes:
        if h.position.x == u.position.x and h.position.y == u.position.y:
            return Terrain.VILLAGE in h.terrain_types
    return False


def _defender_cth_vs(gs, attacker, defender, a_weapon: int) -> float:
    """Defender's chance to hit the ATTACKER (analyze()'s
    terrain_quality input: `bc->get_defender_stats().chance_to_hit`),
    from the EXACT BattleStats the combat resolver uses. Falls back
    to 0.3 (typical open-terrain CTH is 30-40%) when the defender
    has no counter-weapon."""
    try:
        from tools.combat_outcomes import (_stats_pair,
                                           choose_counter_weapon)
        from tools.replay_dataset import build_attack_context
        d_w = choose_counter_weapon(gs, attacker, defender, a_weapon)
        ctx = build_attack_context(gs, attacker, defender,
                                   a_weapon, d_w)
        _a, d_stats = _stats_pair(ctx)
        if d_stats is not None:
            return float(d_stats.cth) / 100.0
    except Exception:                                 # noqa: BLE001
        pass
    return 0.3


def rate_attack(gs, attacker, defender, action: dict,
                aggression: float) -> Optional[float]:
    """1.18.4 attack_analysis::rating for a single stationary
    attacker. None when the outcome distribution is unavailable
    (caller skips the option)."""
    from tools.combat_outcomes import enumerate_attack_outcomes
    dist = enumerate_attack_outcomes(gs, action)
    if dist is None:
        return None
    a_hp0, d_hp0 = attacker.current_hp, defender.current_hp
    ctk = 0.0
    e_d_hp = 0.0
    e_a_hp = 0.0
    p_a_dies = 0.0
    # Index rather than positional-unpack: the OutcomeKey grows over
    # time (petrify, later advancement); we only need the HP fields.
    for key, p in dist.probs.items():
        a_hp, d_hp = key[0], key[1]
        if d_hp <= 0:
            ctk += p
        if a_hp <= 0:
            p_a_dies += p
        e_d_hp += p * max(0, d_hp)
        e_a_hp += p * max(0, a_hp)
    avg_damage_inflicted = d_hp0 - e_d_hp
    avg_damage_taken = a_hp0 - e_a_hp
    avg_losses = _xp_scaled_cost(attacker) * p_a_dies

    target_value = _xp_scaled_cost(defender)
    resources_used = _xp_scaled_cost(attacker)
    # leader_threat in 1.18.4 analyze() means "the target stands
    # adjacent to a leader of the AI's OWN side" (defend my leader),
    # NOT "the target is a leader". Monster sides are no_leader=yes,
    # so for side>=3 it is constant FALSE -- no aggression=1.0, no
    # x5 (independent review 2026-07-14 M1: the first port inverted
    # this and made tentacles kamikaze into enemy leaders).

    value = ctk * target_value - avg_losses * (1.0 - aggression)
    # exposure: exactly 0 (stationary attacker; tq == alt_tq).
    target_starting_damage = defender.max_hp - d_hp0
    value += ((target_starting_damage / 3 + avg_damage_inflicted)
              - (1.0 - aggression) * avg_damage_taken) / 10.0
    # support/vulnerability gates: skipped in v1 (see module doc).
    tq = _defender_cth_vs(gs, attacker, defender,
                          int(action.get('attack_index', 0)))
    if _attacker_on_village(gs, attacker):
        tq *= 0.5
    value /= ((resources_used / 2) + (resources_used / 2) * tq)
    # (leader_threat x5 branch: unreachable for no-leader sides.)
    return value


_MOBILE_WARNED: set = set()


def _check_units_are_stationary(gs, side: int, scenario_id: str = "") -> bool:
    """The substitution's precondition, checked at the moment it is
    relied on.

    A unit this AI drives must be one Wesnoth would not move, for one
    of exactly three reasons:

      * `ai_special=guardian` -- STATE_GUARDIAN (1.18.4
        unit.cpp:659), and the move phase then hands it a move from
        its own hex to its own hex (ca_move_to_targets.cpp:269-277);
      * no movement left -- the scenario pins it every `turn refresh`
        (the enclave maps' `{MODIFY_UNIT (role=monster) moves 0}`);
      * no landable hex -- terrain-locked (2p_mini's water).

    Anything else is a unit the real AI would walk and we would not,
    silently. So it warns, and raises under `WESNOTH_STRICT_WML`.
    """
    from tools.pathfind_sim import ReachContext, unit_reach
    from wesnoth_ai.visibility import is_scenery_unit

    movers = [u for u in gs.map.units
              if u.side == side and not is_scenery_unit(u)
              and not getattr(u, "_ai_guardian", False)
              and u.current_moves > 0]
    if not movers:
        return True
    ctx = ReachContext.for_side(gs, side)
    mobile = [u for u in movers if unit_reach(u, gs, ctx).landable]
    if not mobile:
        return True
    detail = ", ".join(f"{u.name}@({u.position.x},{u.position.y})"
                       for u in mobile)
    if os.environ.get("WESNOTH_STRICT_WML"):
        raise MobileNeutralUnit(
            f"{scenario_id or 'scenario'}: side {side} has units the "
            f"default AI would move but this one cannot: {detail}")
    key = (scenario_id, side, detail)
    if key not in _MOBILE_WARNED:
        _MOBILE_WARNED.add(key)
        log.warning(
            "%s: neutral side %d has units Wesnoth's AI would MOVE and "
            "this combat-only AI will not (%s). The games it generates "
            "face a more passive opponent than the real one.",
            scenario_id or "scenario", side, detail)
    return False


def run_neutral_side_turn(sim, side: int = 3) -> int:
    """The default AI's decisions for the neutral side's turn: the RCA
    combat loop (execute the best-rated adjacent attack while
    rating > 0, re-rating after each). Returns the number of attacks
    executed. The simulator opens the turn with the side's init_side
    before calling this and closes it with the side's end_turn after
    (`WesnothSim._play_neutral_turn`)."""
    from tools.abilities import hex_neighbors
    from wesnoth_ai.visibility import is_scenery_unit
    from wesnoth_ai.sim.classes import Position

    gs = sim.gs
    aggression = _side_aggression(sim.scenario_id, side)
    _check_units_are_stationary(gs, side, sim.scenario_id)

    n_attacks = 0
    for _guard in range(32):                # hard loop bound
        best = None                          # (rating, action, a, d)
        units = {(u.position.x, u.position.y): u for u in gs.map.units}
        for a in list(gs.map.units):
            if a.side != side or is_scenery_unit(a) or a.has_attacked:
                continue
            for nx, ny in hex_neighbors(a.position.x, a.position.y):
                d = units.get((nx, ny))
                if (d is None or d.side == side
                        or d.side not in (1, 2)
                        or is_scenery_unit(d)):
                    continue
                for widx in range(len(a.attacks)):
                    action = {"type": "attack",
                              "start_hex": a.position,
                              "target_hex": Position(nx, ny),
                              "attack_index": widx}
                    r = rate_attack(gs, a, d, action, aggression)
                    if r is not None and (best is None or r > best[0]):
                        best = (r, action, a, d)
        if best is None or best[0] <= 0.0:
            break
        _, action, a, d = best
        if not sim.apply_neutral_attack(action):
            break
        n_attacks += 1
        if sim.done:
            break
    return n_attacks
