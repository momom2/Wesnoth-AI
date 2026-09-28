"""The core's move-order planning and visibility equal the Python's exactly.

For a view bound to the Rust core, `pathfind_sim.ReachContext.for_side`
and `unit_reach` (the route of a move order, the hex an attack is made
from) and `visibility.units_visible_to` are answered by the core
(`side_context`, `unit_reach`, `visible_ids`). On positions of games the
scenario pool sets up, played by random orders under fog, both answer
for every unit of the side to move: the same context, the same
movement points, route costs and predecessors per hex in the same
order, the same landable hexes in the same iteration order (which
decides ties between equally cheap attack hexes), and the same visible
units in the same order.
"""
from __future__ import annotations

import random

import pytest

from wesnoth_ai import game_core as gc

pytestmark = pytest.mark.skipif(gc.game_core_class() is None,
                                reason="the installed wesnoth_core wheel is older than game_core needs")


def _snapshot(sim):
    """A bound view of a fork of the simulator's core."""
    cs = sim.core.fork()
    view = cs.to_state()
    gc.bind_view(view, cs)
    return view


def _play(seed: int, side_turns: int):
    """Bound views of a game played by random orders: each side recruits
    on free castle hexes, moves its units toward the enemy leader or at
    random, attacks what stands next to them, and ends its turn; one
    view after every order."""
    from tools.abilities import hex_neighbors
    from tools.pathfind_sim import ReachContext, unit_reach
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.classes import Position
    from wesnoth_ai.rules import scenario_pool as sp
    from wesnoth_ai.visibility import leader_castle_network
    rng = random.Random(seed)
    setup = sp.random_setup(rng)
    sim = WesnothSim(sp.build_scenario_gamestate(setup), scenario_id=setup.scenario_id, max_turns=40)
    views = []
    for _ in range(side_turns):
        if sim.done:
            break
        gs = sim.gs
        side = gs.global_info.current_side
        leader = next((u for u in gs.map.units if u.side == side and u.is_leader), None)
        if leader is not None and gs.sides[side - 1].recruits:
            occupied = {(u.position.x, u.position.y) for u in gs.map.units}
            free = sorted(p for p in leader_castle_network(gs, leader)[1] if p not in occupied)
            for pos in rng.sample(free, min(3, len(free))):
                sim.step({"type": "recruit", "unit_type": rng.choice(gs.sides[side - 1].recruits),
                          "target_hex": Position(*pos)})
        foe = next((u for u in sim.gs.map.units if u.side not in (side, 3) and u.is_leader), None)
        for u in sorted((u for u in sim.gs.map.units if u.side == side and u.current_moves > 0),
                        key=lambda u: u.id):
            reach = unit_reach(u, sim.gs, ReachContext.for_side(sim.gs, side, exclude_unit=u))
            if not reach.landable:
                continue
            spots = sorted(reach.landable)
            if foe is not None and rng.random() < 0.7:
                target = min(spots, key=lambda p: abs(p[0] - foe.position.x) + abs(p[1] - foe.position.y))
            else:
                target = rng.choice(spots)
            sim.step({"type": "move", "start_hex": Position(u.position.x, u.position.y),
                      "target_hex": Position(*target)})
            views.append(_snapshot(sim))
        for u in sorted((u for u in sim.gs.map.units if u.side == side and not u.has_attacked and u.attacks),
                        key=lambda u: u.id):
            enemies = [e for e in sim.gs.map.units if e.side != side
                       and (e.position.x, e.position.y) in hex_neighbors(u.position.x, u.position.y)]
            if enemies and not sim.done:
                e = rng.choice(enemies)
                sim.step({"type": "attack", "start_hex": Position(u.position.x, u.position.y),
                          "target_hex": Position(e.position.x, e.position.y),
                          "attack_index": rng.randrange(len(u.attacks))})
                views.append(_snapshot(sim))
        if not sim.done:
            sim.step({"type": "end_turn"})
    return views


def _ordered(d: dict) -> list:
    return list(d.items())


def test_reach_and_visibility_equal_the_python():
    from tools.pathfind_sim import ReachContext, unit_reach
    from wesnoth_ai.visibility import units_visible_to, units_visible_to_python
    seen = {"views": 0, "reaches": 0, "hidden": 0, "zoc": 0, "landable": 0}
    for seed in range(4):
        for view in _play(seed, side_turns=10):
            seen["views"] += 1
            side = view.global_info.current_side
            for s in (1, 2):
                rs = [u.id for u in units_visible_to(view, s)]
                py = [u.id for u in units_visible_to_python(view, s)]
                assert rs == py, (seed, s)
                seen["hidden"] += len(rs) < len(view.map.units)
            for u in sorted(view.map.units, key=lambda u: u.id):
                if u.side != side:
                    continue
                ctx_rs = ReachContext.for_side(view, side, exclude_unit=u)
                ctx_py = ReachContext.from_units(side, units_visible_to_python(view, side), u)
                assert ctx_rs.core is not None
                for k in ("occupied_visible", "enemy_hexes", "ally_hexes", "zoc_hexes"):
                    assert getattr(ctx_rs, k) == getattr(ctx_py, k), (seed, u.id, k)
                seen["zoc"] += bool(ctx_rs.zoc_hexes)
                for budget in (None, u.max_moves):
                    a = unit_reach(u, view, ctx_rs, budget=budget)
                    b = unit_reach(u, view, ctx_py, budget=budget)
                    assert (_ordered(a.mp), _ordered(a.cost), _ordered(a.prev)) == \
                        (_ordered(b.mp), _ordered(b.cost), _ordered(b.prev)), (seed, u.id, budget)
                    assert list(a.landable) == list(b.landable), (seed, u.id, budget)
                    seen["reaches"] += 1
                    seen["landable"] += len(a.landable)
    assert seen["views"] > 100 and seen["reaches"] > 1000, seen
    assert seen["hidden"] > 50 and seen["zoc"] > 100 and seen["landable"] > 10000, seen
