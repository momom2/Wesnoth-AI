"""Neutral-side (side >= 3) combat turn: Wesnoth's default RCA AI,
restricted to STATIONARY units (user-approved scope 2026-07-14).

The Mini_Maps_Collection tentacles are either immobilized by the
map itself (enclaves: a `turn refresh` event zeroes role=monster
moves every turn) or terrain-locked (2p_mini: the only 2 water
hexes are the spawn hexes). For a unit that cannot move, Wesnoth's
default AI reduces EXACTLY to its combat candidate action over
adjacent targets, repeated while the best rating > 0
(src/ai/default/ca.cpp combat_phase, 1.18.4).

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
                            + poison_amount*2*P(defender poisoned
                              & survives) when we can poison (per
                              analyze(); tentacles don't poison)
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
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("neutral_ai")

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
        from tools.scenario_events import load_scenario_wml
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
    from classes import Terrain
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
    for (a_hp, d_hp, _asl, _dsl, _apo, _dpo), p in dist.probs.items():
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
    leader_threat = bool(getattr(defender, "is_leader", False))
    if leader_threat:
        aggression = 1.0

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
    if leader_threat:
        value *= 5.0
    return value


def run_neutral_side_turn(sim, side: int = 3) -> int:
    """Play the neutral side's turn: init_side healing/upkeep, then
    the RCA combat loop (execute the best-rated adjacent attack
    while rating > 0, re-rating after each). Returns the number of
    attacks executed. Assumes the caller invokes this at the correct
    point of the turn cycle (after side 2's end_turn, before
    init_side(1)) and restores current_side afterwards via
    _begin_side_turn."""
    from tools.abilities import hex_neighbors
    from tools.replay_dataset import _apply_command
    from visibility import is_scenery_unit
    from classes import Position

    gs = sim.gs
    combatants = [u for u in gs.map.units
                  if u.side == side and not is_scenery_unit(u)]
    if not combatants:
        return 0
    # Side turn opens with init_side: healing (regenerate!), poison,
    # resting flags -- the same parity-verified loop players use.
    from tools.wesnoth_sim import RecordedCommand
    sim._apply_with_stats(["init_side", side])
    sim.command_history.append(RecordedCommand(
        kind="init_side", side=side, cmd=["init_side", side]))
    aggression = _side_aggression(sim.scenario_id, side)

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
