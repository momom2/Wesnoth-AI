"""Tests for core block placement during world initialization."""

import numpy as np
import pytest

from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.utils.rng import SeededRNG
from dungeon_builder.world.geology import GeologyGenerator
from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.world.claimed_territory import ClaimedTerritorySystem
from dungeon_builder.config import (
    CORE_X,
    CORE_Y,
    CORE_Z,
    VOXEL_AIR,
    VOXEL_CORE,
    VOXEL_STONE,
    DEFAULT_SEED,
)


def _build_world():
    """Replicate the main.py world setup without Panda3D."""
    rng = SeededRNG(DEFAULT_SEED)
    grid = VoxelGrid()
    GeologyGenerator(rng).generate(grid)

    # Carve initial dungeon (mirrors main.py _carve_initial_dungeon)
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            if grid.in_bounds(CORE_X + dx, CORE_Y + dy, CORE_Z):
                grid.grid[CORE_X + dx, CORE_Y + dy, CORE_Z] = VOXEL_AIR
            if grid.in_bounds(CORE_X + dx, CORE_Y + dy, CORE_Z - 1):
                grid.grid[CORE_X + dx, CORE_Y + dy, CORE_Z - 1] = VOXEL_AIR

    shaft_x = CORE_X
    for z in range(0, CORE_Z + 1):
        for dx in range(2):
            for dy in range(2):
                x, y = shaft_x + dx, dy
                if grid.in_bounds(x, y, z):
                    grid.grid[x, y, z] = VOXEL_AIR

    for y in range(0, CORE_Y + 1):
        for dx in range(2):
            x = shaft_x + dx
            if grid.in_bounds(x, y, CORE_Z):
                grid.grid[x, y, CORE_Z] = VOXEL_AIR

    for dx in range(-2, 4):
        for dy in range(4):
            x, y = shaft_x + dx, dy
            if grid.in_bounds(x, y, 0):
                grid.grid[x, y, 0] = VOXEL_AIR

    # Place core block (the fix)
    grid.grid[CORE_X, CORE_Y, CORE_Z] = VOXEL_CORE

    return grid


class TestCorePlacement:
    """Verify core block is properly placed in the grid."""

    def test_core_placed_as_voxel_core(self):
        grid = _build_world()
        assert grid.get(CORE_X, CORE_Y, CORE_Z) == VOXEL_CORE

    def test_air_above_core(self):
        """Headroom carved above core so top face renders."""
        grid = _build_world()
        assert grid.get(CORE_X, CORE_Y, CORE_Z - 1) == VOXEL_AIR

    def test_air_around_core_at_same_level(self):
        """Room carved around core at its z-level."""
        grid = _build_world()
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            assert grid.get(CORE_X + dx, CORE_Y + dy, CORE_Z) == VOXEL_AIR, (
                f"({CORE_X+dx},{CORE_Y+dy},{CORE_Z}) should be air"
            )

    def test_core_has_5_exposed_faces(self):
        """Core has air on 4 sides + above = 5 air neighbors."""
        grid = _build_world()
        air_count = 0
        for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,-1),(0,0,1)]:
            if grid.get(CORE_X+dx, CORE_Y+dy, CORE_Z+dz) == VOXEL_AIR:
                air_count += 1
        assert air_count >= 5, f"Expected >=5 air faces, got {air_count}"


class TestCoreClaimedTerritory:
    """Verify claiming works with core block placed."""

    def test_core_is_visible(self):
        grid = _build_world()
        eb = EventBus()
        cts = ClaimedTerritorySystem(eb, grid)
        assert grid.is_visible(CORE_X, CORE_Y, CORE_Z)

    def test_room_around_core_claimed(self):
        grid = _build_world()
        eb = EventBus()
        cts = ClaimedTerritorySystem(eb, grid)
        # Air cells in the 5x5 room should be claimed
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                x, y = CORE_X + dx, CORE_Y + dy
                if (dx, dy) != (0, 0):
                    assert grid.is_claimed(x, y, CORE_Z), (
                        f"({x},{y},{CORE_Z}) should be claimed"
                    )

    def test_headroom_above_core_claimed(self):
        grid = _build_world()
        eb = EventBus()
        cts = ClaimedTerritorySystem(eb, grid)
        # z-1 level room should be claimed
        assert grid.is_claimed(CORE_X + 1, CORE_Y, CORE_Z - 1)

    def test_shaft_to_surface_claimed(self):
        grid = _build_world()
        eb = EventBus()
        cts = ClaimedTerritorySystem(eb, grid)
        # Shaft at (CORE_X, 0) should be claimed all the way up
        for z in range(0, CORE_Z):
            assert grid.is_claimed(CORE_X, 0, z), f"Shaft at z={z} not claimed"

    def test_significant_claimed_territory(self):
        """Overall claimed count should be substantial (shaft + corridor + room)."""
        grid = _build_world()
        eb = EventBus()
        cts = ClaimedTerritorySystem(eb, grid)
        claimed = int(np.sum(grid.claimed))
        assert claimed > 100, f"Only {claimed} claimed cells (expected >100)"

    def test_significant_visible_territory(self):
        """Visible blocks should include walls of claimed areas."""
        grid = _build_world()
        eb = EventBus()
        cts = ClaimedTerritorySystem(eb, grid)
        visible = int(np.sum(grid.visible))
        assert visible > 100, f"Only {visible} visible cells (expected >100)"
