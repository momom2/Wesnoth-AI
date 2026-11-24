"""The hex the simulator struck from when an attack was ordered from a
distance, for reading game records of format 5 and earlier.

Until 2026-11-24 the legal actions offered an attack on an enemy the
attacker was not next to but could reach that turn. The simulator then
moved the unit to a hex next to the target, chosen by the rule below, and
ran the fight from there. The network did not choose that hex, so the user
removed the action (principle 3; docs/input_facts_design_20261124.md,
decision 2): the legality mask offers an attack only on an enemy on a hex
that touches the attacker's (rust/wesnoth_core/src/lib.rs
`rows_from_landable`), and the simulator refuses any other
(tools/wesnoth_sim.py `_action_to_command`).

Records played before that hold such an attack as its approach move and
the attack, under one decision. `tools/own_trajectories.py` tells that
pair from a move decision followed by an attack decision by asking which
hex the simulator would have chosen: this function. Nothing that plays
calls it.
"""
from __future__ import annotations

from typing import Optional

from wesnoth_ai.classes import GameState, Position


def attack_hex(gs: GameState, attacker, target) -> Optional[Position]:
    """The hex the simulator moved `attacker` to before an attack on
    `target` ordered from a distance.

    Returns the chosen attack hex (a neighbor of `target` the
    attacker can LAND on this turn, per the shared
    observable-state planner), or None if no such hex exists.

    Why this helper: the policy emits attack actions with
    `start_hex = unit's current position`. When `start_hex` isn't
    adjacent to `target`, Wesnoth's replay engine rejects the
    bare `[attack]` command (battle_context disables out-of-range
    weapons; the attack handler then returns without consuming the
    [random_seed] follow-up, and the next outer-loop iteration
    errors with "found dependent command in replay while
    is_synced=false"). So before recording an [attack], we have to
    emit an explicit [move] putting the attacker on a neighbor of
    the target hex -- mirroring what Wesnoth's UI does when the
    player clicks an enemy from a non-adjacent unit.

    Knowledge level: the plan runs on the ACTING SIDE'S
    OBSERVABLE state (tools/pathfind_sim.ReachContext), exactly
    like a player's own attack order -- hidden units neither
    block nor ZoC the approach; they resolve during the nested
    move's execution walk (blocked/ambush truncation aborts the
    attack, as in Wesnoth).

    Choice among candidate hexes: minimal Wesnoth float cost
    (MP + defense/ally subcosts, pathfind.cpp:815-820), i.e. the
    cheapest approach preferring defensible terrain -- the same
    preference order Wesnoth's own pathfinder applies.
    """
    from tools.abilities import hex_neighbors
    from tools.pathfind_sim import ReachContext, unit_reach

    target_neighbors = set(hex_neighbors(target.x, target.y))
    ctx = ReachContext.for_side(
        gs, attacker.side, exclude_unit=attacker)
    reach = unit_reach(attacker, gs, ctx)
    # The attacker's own hex is a free "landing" if already
    # adjacent (defensive: callers normally handle that case).
    if reach.start in target_neighbors:
        return Position(x=reach.start[0], y=reach.start[1])
    candidates = [
        pos for pos in reach.landable if pos in target_neighbors
    ]
    if not candidates:
        return None
    best = min(candidates, key=lambda pos: reach.cost[pos])
    return Position(x=best[0], y=best[1])
