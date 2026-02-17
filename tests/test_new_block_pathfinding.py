"""Tests for pathfinding traversal of the 10 new block types.

Verifies that PersonalPathfinder correctly handles iron bars, floodgates,
pressure plates, pipes, pumps, gold bait, heat beacon, alarm bell,
fragile floor, and steam vent -- including hazard-aware cunning cost.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from dungeon_builder.intruders.personal_pathfinder import PersonalPathfinder
from dungeon_builder.intruders.personal_map import PersonalMap
from dungeon_builder.intruders.archetypes import VANGUARD

from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_STONE,
    VOXEL_IRON_BARS,
    VOXEL_FLOODGATE,
    VOXEL_PRESSURE_PLATE,
    VOXEL_PIPE,
    VOXEL_PUMP,
    VOXEL_GOLD_BAIT,
    VOXEL_HEAT_BEACON,
    VOXEL_ALARM_BELL,
    VOXEL_FRAGILE_FLOOR,
    VOXEL_STEAM_VENT,
    HAZARD_PATH_COST,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_corridor(pmap: PersonalMap, length: int = 5, block_at: int = 2,
                   block_type: int = VOXEL_AIR, block_state: int = 0
                   ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Create a 1-D corridor of air with one block of *block_type* in the middle.

    Returns ``(start, goal)`` positions.
    """
    for i in range(length):
        if i == block_at:
            pmap.reveal(i, 0, 0, block_type, block_state)
        else:
            pmap.reveal(i, 0, 0, VOXEL_AIR)
    return (0, 0, 0), (length - 1, 0, 0)


# ---------------------------------------------------------------------------
# Iron bars -- always impassable
# ---------------------------------------------------------------------------

class TestIronBarsImpassable:

    def test_iron_bars_blocks_path(self):
        pmap = PersonalMap()
        start, goal = _make_corridor(pmap, block_type=VOXEL_IRON_BARS)
        path = PersonalPathfinder.find_path(pmap, start, goal, VANGUARD)
        assert path is None


# ---------------------------------------------------------------------------
# Floodgate -- state-dependent
# ---------------------------------------------------------------------------

class TestFloodgate:

    def test_closed_floodgate_blocks_path(self):
        """A closed floodgate (block_state=1) is impassable."""
        pmap = PersonalMap()
        start, goal = _make_corridor(pmap, block_type=VOXEL_FLOODGATE,
                                     block_state=1)
        path = PersonalPathfinder.find_path(pmap, start, goal, VANGUARD)
        assert path is None

    def test_open_floodgate_allows_path(self):
        """An open floodgate (block_state=0) is passable."""
        pmap = PersonalMap()
        start, goal = _make_corridor(pmap, block_type=VOXEL_FLOODGATE,
                                     block_state=0)
        path = PersonalPathfinder.find_path(pmap, start, goal, VANGUARD)
        assert path is not None
        assert start in path
        assert goal in path
        assert (2, 0, 0) in path  # passes through the floodgate cell


# ---------------------------------------------------------------------------
# Pressure plate -- traversable, cunning adds hazard cost
# ---------------------------------------------------------------------------

class TestPressurePlate:

    def test_pressure_plate_traversable(self):
        """Non-cunning archetype walks right through."""
        pmap = PersonalMap()
        start, goal = _make_corridor(pmap, block_type=VOXEL_PRESSURE_PLATE)
        path = PersonalPathfinder.find_path(pmap, start, goal, VANGUARD)
        assert path is not None
        assert (2, 0, 0) in path

    def test_pressure_plate_cunning_cost(self):
        """A cunning archetype still finds a path but incurs higher cost.

        With a single-corridor layout there is no alternative route, so the
        path must still go through the pressure plate.  We verify the plate
        is in the path and, when an alternative air-only route exists, the
        cunning archetype prefers it.
        """
        cunning_arch = replace(VANGUARD, cunning=0.8)
        # Single corridor -- no alternative, must pass through
        pmap = PersonalMap()
        start, goal = _make_corridor(pmap, block_type=VOXEL_PRESSURE_PLATE)
        path = PersonalPathfinder.find_path(pmap, start, goal, cunning_arch)
        assert path is not None
        assert (2, 0, 0) in path

        # Now provide an alternative air-only corridor (row y=1).
        # The cunning archetype should prefer the air bypass because the
        # pressure plate is a hazard (cost = 1 + HAZARD_PATH_COST).
        pmap2 = PersonalMap()
        # Main corridor with pressure plate at (2,0,0)
        for i in range(5):
            if i == 2:
                pmap2.reveal(i, 0, 0, VOXEL_PRESSURE_PLATE)
            else:
                pmap2.reveal(i, 0, 0, VOXEL_AIR)
        # Bypass corridor at y=1 (slightly longer: 7 cells, but no hazard)
        for i in range(5):
            pmap2.reveal(i, 1, 0, VOXEL_AIR)

        path2 = PersonalPathfinder.find_path(pmap2, (0, 0, 0), (4, 0, 0),
                                             cunning_arch)
        assert path2 is not None
        # The cunning archetype should avoid (2, 0, 0) by going via y=1
        assert (2, 0, 0) not in path2


# ---------------------------------------------------------------------------
# Pipe -- impassable solid
# ---------------------------------------------------------------------------

class TestPipeImpassable:

    def test_pipe_blocks_path(self):
        pmap = PersonalMap()
        start, goal = _make_corridor(pmap, block_type=VOXEL_PIPE)
        path = PersonalPathfinder.find_path(pmap, start, goal, VANGUARD)
        assert path is None


# ---------------------------------------------------------------------------
# Pump -- impassable solid
# ---------------------------------------------------------------------------

class TestPumpImpassable:

    def test_pump_blocks_path(self):
        pmap = PersonalMap()
        start, goal = _make_corridor(pmap, block_type=VOXEL_PUMP)
        path = PersonalPathfinder.find_path(pmap, start, goal, VANGUARD)
        assert path is None


# ---------------------------------------------------------------------------
# Gold bait -- traversable (interaction handles effect)
# ---------------------------------------------------------------------------

class TestGoldBaitTraversable:

    def test_gold_bait_allows_path(self):
        pmap = PersonalMap()
        start, goal = _make_corridor(pmap, block_type=VOXEL_GOLD_BAIT)
        path = PersonalPathfinder.find_path(pmap, start, goal, VANGUARD)
        assert path is not None
        assert (2, 0, 0) in path


# ---------------------------------------------------------------------------
# Heat beacon -- traversable
# ---------------------------------------------------------------------------

class TestHeatBeaconTraversable:

    def test_heat_beacon_allows_path(self):
        pmap = PersonalMap()
        start, goal = _make_corridor(pmap, block_type=VOXEL_HEAT_BEACON)
        path = PersonalPathfinder.find_path(pmap, start, goal, VANGUARD)
        assert path is not None
        assert (2, 0, 0) in path


# ---------------------------------------------------------------------------
# Alarm bell -- traversable
# ---------------------------------------------------------------------------

class TestAlarmBellTraversable:

    def test_alarm_bell_allows_path(self):
        pmap = PersonalMap()
        start, goal = _make_corridor(pmap, block_type=VOXEL_ALARM_BELL)
        path = PersonalPathfinder.find_path(pmap, start, goal, VANGUARD)
        assert path is not None
        assert (2, 0, 0) in path


# ---------------------------------------------------------------------------
# Fragile floor -- traversable (looks like stone to intruders)
# ---------------------------------------------------------------------------

class TestFragileFloorTraversable:

    def test_fragile_floor_allows_path(self):
        pmap = PersonalMap()
        start, goal = _make_corridor(pmap, block_type=VOXEL_FRAGILE_FLOOR)
        path = PersonalPathfinder.find_path(pmap, start, goal, VANGUARD)
        assert path is not None
        assert (2, 0, 0) in path


# ---------------------------------------------------------------------------
# Steam vent -- traversable
# ---------------------------------------------------------------------------

class TestSteamVentTraversable:

    def test_steam_vent_allows_path(self):
        pmap = PersonalMap()
        start, goal = _make_corridor(pmap, block_type=VOXEL_STEAM_VENT)
        path = PersonalPathfinder.find_path(pmap, start, goal, VANGUARD)
        assert path is not None
        assert (2, 0, 0) in path
