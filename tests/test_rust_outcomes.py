"""`combat_outcomes` asks the core a view is bound to, not a core built
from the view."""
from __future__ import annotations

import copy
import random

import pytest

from wesnoth_ai import game_core as gc

pytestmark = pytest.mark.skipif(gc.game_core_class() is None,
                                reason="the installed wesnoth_core wheel is older than game_core needs")


def _unit_at(gs, pos):
    return next(u for u in gs.map.units if (u.position.x, u.position.y) == pos)


def test_a_bound_view_is_answered_by_its_core():
    """`combat_outcomes` routes a bound view to the core: a view whose
    core holds a different defender gets the core's answer."""
    from tools import combat_outcomes as co
    from wesnoth_ai.rules import scenario_pool as sp
    gs = sp.build_scenario_gamestate(sp.random_setup(random.Random(2)))
    gs.map.units = set()
    hexes = sorted((h.position.x, h.position.y) for h in gs.map.hexes)
    from tools.abilities import hex_neighbors
    a_pos = next(p for p in hexes if any(q in set(hexes) for q in hex_neighbors(*p)))
    d_pos = next(q for q in hex_neighbors(*a_pos) if q in set(hexes))
    gs.map.units.add(gc.build_recruit_unit("Elvish Fighter", 1, *a_pos, 1, game_id=gs.game_id))
    gs.map.units.add(gc.build_recruit_unit("Orcish Grunt", 2, *d_pos, 2, game_id=gs.game_id))
    other = copy.deepcopy(gs)
    other.map.units = {u for u in other.map.units if u.side == 1}
    other.map.units.add(gc.build_recruit_unit("Orcish Archer", 2, *d_pos, 2, game_id=gs.game_id))
    cs = gc.CoreState.from_state(other)
    gc.bind_view(gs, cs)
    try:
        att, dfd = _unit_at(gs, a_pos), _unit_at(gs, d_pos)
        # The Elvish Fighter's bow (weapon 1) meets the Archer's bow in the
        # core and nothing on the Grunt the view shows.
        assert co.choose_counter_weapon(gs, att, dfd, 1) == cs.core.counter_weapon_choice(*a_pos, *d_pos, 1)[0]
        assert co.choose_counter_weapon(gs, att, dfd, 1) >= 0
    finally:
        gc._VIEW_CORES.pop(id(gs), None)

