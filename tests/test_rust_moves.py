"""The core's move-order planning names the hexes its moves reach.

`pathfind_sim.ReachContext.for_side` and `unit_reach` (the route of a
move order, the hex an attack is made from) come from the core that
answers for the state (`side_context`, `unit_reach`), in the core's hex
order. After a terrain change the view's hex set iterates in another
order than the core's hex indices; the planner must still name the
hexes the core's own move then reaches.
"""
from __future__ import annotations

import random

import pytest

from wesnoth_ai import game_core as gc

pytestmark = pytest.mark.skipif(gc.game_core_class() is None,
                                reason="the installed wesnoth_core wheel is older than game_core needs")


def test_reach_after_a_terrain_change_names_the_core_hexes():
    """Aethermaw's terrain changes from turn 4 give the view a new hex set,
    which iterates in another order than the core's hex indices. The core's
    context and reach must still name the right hexes: on 2026-09-29 they
    were mapped through the view's order, named other hexes, and the
    simulator refused the moves the legality mask offered from turn 4.
    Every landable hex of every side-1 unit is reached by its planned
    route on a fork of the core."""
    from tools.pathfind_sim import ReachContext, route_to, unit_reach
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.rules import scenario_pool as sp
    seed = next(s for s in range(10_000)
                if sp.random_setup(random.Random(s)).scenario_id == "multiplayer_Aethermaw")
    setup = sp.random_setup(random.Random(seed))
    sim = WesnothSim(sp.build_scenario_gamestate(setup), scenario_id=setup.scenario_id, max_turns=40)
    while not (sim.gs.global_info.turn_number == 4 and sim.gs.global_info.current_side == 1):
        sim.step({"type": "end_turn"})
    cs = sim.core.fork()
    view = gc.view_of(cs)
    core_order = list(cs.geometry().keys)
    view_order = [(h.position.x, h.position.y) for h in view.map.hexes]
    assert sorted(core_order) == sorted(view_order) and core_order != view_order, \
        "the terrain change no longer reorders the view's hexes: this test tests nothing"
    walked = 0
    for u in sorted((u for u in view.map.units if u.side == 1), key=lambda u: u.id):
        ctx = ReachContext.for_side(view, 1, exclude_unit=u)
        assert ctx.core is cs
        reach = unit_reach(u, view, ctx)
        for target in sorted(reach.landable):
            path = route_to(reach, target)
            fork = cs.fork()
            fork.core.apply_move([p[0] for p in path], [p[1] for p in path], 1)
            *_ordered, lx, ly, _reason = fork.core.last_move_walk_export()
            assert (lx, ly) == target, (u.id, target, path)
            walked += 1
    assert walked >= 10
