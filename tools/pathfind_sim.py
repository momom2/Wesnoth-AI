#!/usr/bin/env python3
"""Wesnoth-1.18.4-faithful single-turn route planning.

PLANNING (`ReachContext` / `unit_reach` / `route_to`) runs on the ACTING
SIDE'S OBSERVABLE STATE -- exactly what Wesnoth's own client uses when
a player orders a move (`mouse_handler::get_route` builds
`shortest_path_calculator` with the moving player's team;
`get_visible_unit` / `enemy_zoc` consult only units visible to that
team -- pathfind.cpp:742-820, 134-140). Hidden units neither block nor
exert ZoC at this layer. The legality mask and the sim's order-to-path
translation BOTH read this planning, so anything the mask offers, the
sim can route.

The context and each unit's reach come from the Rust core that answers
for the state (`game_core.core_for`: `side_context`, `unit_reach`,
rust/wesnoth_core/src/core_observe.rs); this module assembles them into
`UnitReach` and picks routes from it (`route_to`).

EXECUTION is the core's move (rust/wesnoth_core/src/core_move.rs),
god-view, mirroring `unit_mover` (actions/move.cpp):
  - blocked: an invisible unit sits ON a path hex -> stop on the hex
    BEFORE it, remaining MP KEPT (post_move zeroes MP only for ambush /
    ZoC-final: move.cpp:1041-1043), blocker revealed.
  - ambush: entering a hex ADJACENT to a hidden `hides` enemy -> stop AT
    that hex, MP zeroed, ambushers revealed (check_for_ambushers,
    move.cpp:422-440; reveal_ambusher sets STATE_UNCOVERED, move.cpp:870).
  - village capture on the FINAL hex zeroes MP (move.cpp:1046-1053);
    passing THROUGH a village mid-path does NOT stop or capture.
  - ZoC landing: the planner makes ZoC hexes terminal, so a planned path
    never continues past one; a final hex in (visible-)enemy ZoC leaves
    the mover no movement (`final_loc == zoc_stop_` -> set_movement(0)).

Route preference is Wesnoth's cost model, not an explicit tie-break
(see docs/wesnoth_rules.md "Default route selection"): float cost =
MP + (defense_pct + ally_occupied)/10000 per entered hex, ZoC entry
charging all remaining MP. Residual exact ties are unspecified in the
engine (heap order) and broken deterministically by the core.

Sighted-move interrupts are deliberately NOT modelled: exported
[move] WML carries skip_sighted="all", which replay playback honours
(synced_commands.cpp:305-314), so sim, export and playback agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

Coord = Tuple[int, int]


@dataclass
class ReachContext:
    """Per-decision, per-side observable-state snapshot shared by all
    of one side's `unit_reach` calls (occupancy / ZoC are side-level,
    not unit-level), from the core that answers for the state."""
    side: int
    # Hexes holding a VISIBLE unit (any side, incl. own): cannot LAND.
    occupied_visible: Set[Coord] = field(default_factory=set)
    # Hexes holding a visible ENEMY of `side`: cannot ENTER.
    enemy_hexes: Set[Coord] = field(default_factory=set)
    # Hexes holding a visible NON-enemy unit: pass-through, +1 subcost
    # (pathfind.cpp:785).
    ally_hexes: Set[Coord] = field(default_factory=set)
    # Hexes covered by a visible enemy's ZoC.
    zoc_hexes: Set[Coord] = field(default_factory=set)
    # The core the context was read from: `unit_reach` asks it.
    core: object = field(default=None, repr=False, compare=False)

    @classmethod
    def for_side(cls, gs, side: int, *, exclude_unit=None) -> "ReachContext":
        """The observable context of `side` in `gs`. An excluded mover
        (one of `side`'s units) leaves its hex unoccupied."""
        import numpy as _np
        from wesnoth_ai.game_core import core_for
        if exclude_unit is not None and exclude_unit.side != side:
            raise ValueError(f"the excluded unit {exclude_unit.id} is not side {side}'s")
        cs = core_for(gs)
        positions = cs.geometry().keys          # the core's hex order
        occupied, enemy, ally, zoc = cs.core.side_context(side)
        ctx = cls(side=side, core=cs)
        for flags, target in ((occupied, ctx.occupied_visible), (enemy, ctx.enemy_hexes),
                              (ally, ctx.ally_hexes), (zoc, ctx.zoc_hexes)):
            target.update(positions[i] for i in _np.flatnonzero(flags))
        if exclude_unit is not None:
            pos = (exclude_unit.position.x, exclude_unit.position.y)
            ctx.occupied_visible.discard(pos)
            ctx.ally_hexes.discard(pos)
        return ctx


@dataclass
class UnitReach:
    """Single-turn reachability for one unit under one ReachContext.

    `mp[c]`   -- integer MP spent to stand on c (start: 0)
    `cost[c]` -- Wesnoth-comparable float cost (MP + subcosts) used
                 for route preference
    `prev[c]` -- predecessor hex on the preferred route
    `landable` -- hexes the unit may END a move order on: reachable,
                 not the start, and not visibly occupied (plot_turn
                 backtracks off visible-unit end hexes,
                 move.cpp:776-780; hidden occupants do NOT bar
                 landing here -- that resolves at execution).
    """
    start: Coord
    mp: Dict[Coord, int]
    cost: Dict[Coord, float]
    prev: Dict[Coord, Coord]
    landable: Set[Coord]


def unit_reach(unit, gs, ctx: ReachContext,
               budget: Optional[int] = None) -> UnitReach:
    """The single-turn reach of `unit`, one of `ctx.side`'s units, from
    the core `ctx` was read from (`budget`: its movement left by
    default)."""
    if ctx.core is None or unit.side != ctx.side:
        raise ValueError(f"unit_reach needs a context of the unit's side ({unit.side}) "
                         f"read from a core (ReachContext.for_side)")
    start = (unit.position.x, unit.position.y)
    if budget is None:
        budget = int(unit.current_moves)
    arrays = ctx.core.core.unit_reach(start[0], start[1], int(budget))
    if arrays is None:
        raise ValueError(f"no unit on the map at {start} in the context's core")
    mp_a, cost_a, prev_a = arrays
    return _reach_from_arrays(start, ctx.core.geometry().keys, mp_a.tolist(),
                              cost_a.tolist(), prev_a.tolist(), ctx)


def _reach_from_arrays(start, positions, mp_l, cost_l, prev_l,
                       ctx) -> UnitReach:
    """The core's per-hex arrays as a UnitReach."""
    mp: Dict[Coord, int] = {}
    cost: Dict[Coord, float] = {}
    prev: Dict[Coord, Coord] = {}
    for i, m in enumerate(mp_l):
        if m >= 0:
            p = positions[i]
            mp[p] = m
            cost[p] = cost_l[i]
            if prev_l[i] >= 0:
                prev[p] = positions[prev_l[i]]
    landable = {
        pos for pos in mp
        if pos != start and pos not in ctx.occupied_visible
    }
    return UnitReach(start=start, mp=mp, cost=cost, prev=prev,
                     landable=landable)


def route_to(reach: UnitReach, target: Coord) -> Optional[List[Coord]]:
    """Preferred route start..target, or None if unreached."""
    if target not in reach.mp:
        return None
    path = [target]
    while path[-1] != reach.start:
        path.append(reach.prev[path[-1]])
    path.reverse()
    return path
