"""Who holds a zone of control, asked one way by every reader.

The engine's rule is the unit's own: `unit::emits_zoc()` is
`emit_zoc_ && !incapacitated()` (src/units/unit.hpp:1352-1356, 1.18.4),
`emit_zoc_` being the type's `zoc=`, `level > 0` by default
(src/units/types.cpp:215). Attacks play no part. The planner
(`ReachContext.for_side`, which the legality mask shares) once skipped
every "scenery" unit -- petrified, or attackless on a side past 2 --
while the move's walk skipped only petrified and level-0 units, so a
move the mask offered past an attackless level-1 unit was stopped by the
walk at its first hex: the mask/sim contract broken on that board. The
planner, the observation's zone flags and the move all read the Rust
core (core_observe.rs, core_move.rs).
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.abilities import hex_neighbors  # noqa: E402
from tools.pathfind_sim import ReachContext, route_to, unit_reach  # noqa: E402
from wesnoth_ai import game_core as gc  # noqa: E402
from wesnoth_ai.classes import Position  # noqa: E402
from wesnoth_ai.visibility import is_scenery_unit  # noqa: E402

pytestmark = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore is not available")


def _replayed_board(**stone_changes):
    """A replayed three-side state, fog off, side 1 to move, whose side-3
    Dwarvish Fighter (level 1) is neither petrified nor armed: scenery by
    our classification, a zone of control by the engine's. A replayed
    state carries the real unit types and sides the Rust core needs.
    `stone_changes` are set on the stone."""
    from tests.sim_test_helpers import replayed_state, three_side_record
    s = copy.deepcopy(replayed_state(three_side_record(fog=False), 1))   # an unbound copy to edit
    stone = next(u for u in s.map.units if u.side == 3)
    stone.statuses = set(stone.statuses) - {"petrified"}
    stone.attacks = []
    for k, v in stone_changes.items():
        setattr(stone, k, v)
    return s, stone


def _stone_zone(stone):
    return set(hex_neighbors(stone.position.x, stone.position.y))


def test_the_level_decides_not_the_attacks():
    s, stone = _replayed_board()
    assert is_scenery_unit(stone)
    assert _stone_zone(stone) <= ReachContext.for_side(s, 1).zoc_hexes
    for changes in ({"statuses": {"petrified"}}, {"name": "Peasant"}):     # incapacitated; level 0
        s, stone = _replayed_board(**changes)
        assert not _stone_zone(stone) & ReachContext.for_side(s, 1).zoc_hexes, changes


def _walking_board():
    """The replayed board with side 1's Elvish Captain moved to (2, y),
    three hexes west of the attackless side-3 stone, and its Rust core."""
    s, stone = _replayed_board()
    mover = next(u for u in s.map.units if u.side == 1 and u.is_leader)
    mover.position = Position(x=2, y=stone.position.y)
    return s, mover, stone, gc.CoreState.from_state(s)


def _walk(cs, mover, path):
    fork = cs.fork()
    fork.core.apply_move([p[0] for p in path], [p[1] for p in path], 1)
    *_ordered, lx, ly, reason = fork.core.last_move_walk_export()
    return (lx, ly), reason, fork.core.unit_export(mover.id)["current_moves"]


def test_the_walk_stops_in_the_zone_the_planner_sees():
    s, mover, stone, cs = _walking_board()
    zone = _stone_zone(stone)
    assert zone <= ReachContext.for_side(s, 1).zoc_hexes
    y = stone.position.y
    entry = (stone.position.x - 1, y)
    beyond = next(p for p in hex_neighbors(*entry)
                  if p not in zone and p != (stone.position.x, y) and p[0] == entry[0])
    landed, reason, mp_left = _walk(cs, mover, [(2, y), (3, y), entry, beyond])
    assert (landed, reason, mp_left) == (entry, "zoc", 0)


def test_every_planned_move_is_walked_to_its_end():
    """The mask/sim contract on this board: each hex the planner lets
    the mover land on, it reaches by the planned route."""
    s, mover, _stone, cs = _walking_board()
    reach = unit_reach(mover, s, ReachContext.for_side(s, 1))
    assert len(reach.landable) > 4
    for target in sorted(reach.landable):
        path = route_to(reach, target)
        if len(path) < 2:
            continue
        landed, _reason, _mp = _walk(cs, mover, path)
        assert landed == tuple(target), (target, path)


def test_the_observation_holds_the_zone_the_planner_sees():
    """The observation's zone flags (which the mask's reach rows read)
    and the planner's context agree, and hold the stone's zone."""
    s, stone = _replayed_board()
    observation = gc.CoreState.from_state(s).observe(1)
    keys = observation.geometry.keys
    on_map = set(keys)
    planner = ReachContext.for_side(s, 1).zoc_hexes & on_map
    assert _stone_zone(stone) & on_map <= planner
    assert {keys[j] for j in np.flatnonzero(observation.zoc).tolist()} == planner
