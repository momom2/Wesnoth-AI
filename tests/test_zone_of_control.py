"""Who holds a zone of control, asked one way by every reader.

The engine's rule is the unit's own: `unit::emits_zoc()` is
`emit_zoc_ && !incapacitated()` (src/units/unit.hpp:1352-1356, 1.18.4),
`emit_zoc_` being the type's `zoc=`, `level > 0` by default
(src/units/types.cpp:215). Attacks play no part. The planner
(`ReachContext.for_side`, which the legality mask shares) skipped every
"scenery" unit -- petrified, or attackless on a side past 2 -- while the
walker (`walk_move_path`) skipped only petrified and level-0 units, so a
move the mask offered past an attackless level-1 unit was stopped by the
walk at its first hex: the mask/sim contract broken on that board. All
of them now ask `pathfind_sim.emits_zoc`. The observation's zone flags,
which the mask's Rust reach rows read, come from the Rust kernel
(observe.rs), fed that predicate by `wesnoth_ai.observe` and the unit's
level by the Rust core.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_visibility import _hexes_grid, _state, _unit  # noqa: E402
from wesnoth_ai.sim.abilities import hex_neighbors  # noqa: E402
from tools.pathfind_sim import (ReachContext, emits_zoc, route_to,  # noqa: E402
                                unit_reach, walk_move_path)
from wesnoth_ai.sim.classes import Position  # noqa: E402
from wesnoth_ai.visibility import is_scenery_unit  # noqa: E402


def _board():
    """A Spearman at (0,0) with 4 MP on a 6x2 flat board, fog off, and an
    attackless level-1 unit of side 3 at (2,1): scenery by our
    classification, a zone of control by the engine's. Its zone covers
    (1,0), (2,0), (3,0), (1,1) and (3,1), so every way east from the
    mover enters it at x = 1."""
    stone = dataclasses.replace(_unit("stone", x=2, side=3, name="Spearman"),
                                position=Position(x=2, y=1))
    mover = _unit("mover", x=0, side=1, max_moves=4, name="Spearman")
    s = _state([mover, stone], _hexes_grid(6, 2))
    s.global_info._fog = False
    return s, next(u for u in s.map.units if u.id == "mover"), stone


def test_the_level_decides_not_the_attacks():
    s, _mover, stone = _board()
    assert is_scenery_unit(stone) and emits_zoc(stone)
    assert not emits_zoc(dataclasses.replace(stone, statuses=frozenset({"petrified"})))
    assert not emits_zoc(dataclasses.replace(stone, name="Peasant"))     # level 0


def test_the_planner_sees_the_zone_the_walker_stops_in():
    s, mover, _stone = _board()
    ctx = ReachContext.for_side(s, 1)
    assert {(1, 0), (2, 0), (3, 0), (1, 1), (3, 1)} <= ctx.zoc_hexes
    reach = unit_reach(mover, s, ctx)
    assert (4, 0) not in reach.landable
    out = walk_move_path(s, mover, [0, 1, 2, 3, 4], [0, 0, 0, 0, 0])
    assert (out.stop_reason, out.final_idx, out.mp_left) == ("zoc", 1, 0)


def test_every_planned_move_is_walked_to_its_end():
    """The mask/sim contract on this board: each hex the planner lets
    the mover land on, it reaches by the planned route."""
    s, mover, _stone = _board()
    reach = unit_reach(mover, s, ReachContext.for_side(s, 1))
    assert reach.landable
    for target in sorted(reach.landable):
        path = route_to(reach, target)
        out = walk_move_path(s, mover, [p[0] for p in path], [p[1] for p in path])
        assert out.final_idx == len(path) - 1, (target, out)


def _replayed_board():
    """A replayed three-side state, fog off, side 1 to move, whose side-3
    Dwarvish Fighter (level 1) is neither petrified nor armed: scenery by
    our classification, a zone of control by the engine's. A replayed
    state carries the real unit types and sides the Rust core needs."""
    from tests.sim_test_helpers import replayed_state, three_side_record
    s = replayed_state(three_side_record(fog=False), 1)
    stone = next(u for u in s.map.units if u.side == 3)
    stone.statuses = set(stone.statuses) - {"petrified"}
    stone.attacks = []
    assert is_scenery_unit(stone) and emits_zoc(stone)
    return s, stone


def _zones(s, stone, observation):
    """(the observation's zone of control, the planner's), on the map;
    the planner's must hold the stone's."""
    keys = observation.geometry.keys
    on_map = set(keys)
    planner = ReachContext.for_side(s, 1).zoc_hexes & on_map
    assert set(hex_neighbors(stone.position.x, stone.position.y)) & on_map <= planner
    return {keys[j] for j in np.flatnonzero(observation.zoc).tolist()}, planner


def test_the_observation_holds_the_zone_the_planner_sees():
    from wesnoth_ai import observe as obs_mod
    if obs_mod.kernel() is None:
        pytest.skip("wesnoth_core.observe_side is not available")
    s, stone = _replayed_board()
    got, want = _zones(s, stone, obs_mod.observe(s, 1))
    assert got == want


def test_the_core_observation_holds_the_zone_the_planner_sees():
    from wesnoth_ai import game_core as gc
    if gc.game_core_class() is None:
        pytest.skip("wesnoth_core.GameCore is not available")
    s, stone = _replayed_board()
    got, want = _zones(s, stone, gc.CoreState.from_state(s).observe(1))
    assert got == want
