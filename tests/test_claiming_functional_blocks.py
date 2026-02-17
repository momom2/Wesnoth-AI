"""Tests for claiming propagation through functional blocks.

Verifies that slopes, stairs, doors, and other player-built blocks allow
territory to propagate through them rather than acting as barriers.
"""

from __future__ import annotations

import pytest
import numpy as np

from dungeon_builder.config import (
    VOXEL_AIR, VOXEL_STONE, VOXEL_CORE,
    VOXEL_SLOPE, VOXEL_STAIRS, VOXEL_DOOR, VOXEL_TARP,
    VOXEL_SPIKE, VOXEL_TREASURE, VOXEL_ROLLING_STONE,
    VOXEL_REINFORCED_WALL, VOXEL_LAVA,
    CORE_X, CORE_Y, CORE_Z,
)
from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.world.claimed_territory import ClaimedTerritorySystem


def _make_grid() -> VoxelGrid:
    """Create a grid filled with stone."""
    grid = VoxelGrid()
    grid.grid[:] = VOXEL_STONE
    grid.grid[CORE_X, CORE_Y, CORE_Z] = VOXEL_CORE
    return grid


def _carve_corridor_with_block(grid: VoxelGrid, block_type: int) -> tuple[int, int, int]:
    """Carve air around core, place block_type in a corridor, and air beyond it.

    Layout at z=CORE_Z:
    core @ (CORE_X, CORE_Y) -> air @ (CORE_X+1, CORE_Y) ->
    block_type @ (CORE_X+2, CORE_Y) -> air @ (CORE_X+3, CORE_Y)

    Returns the position of the air cell beyond the functional block.
    """
    cx, cy, cz = CORE_X, CORE_Y, CORE_Z
    # Air adjacent to core
    grid.grid[cx + 1, cy, cz] = VOXEL_AIR
    # Functional block in middle
    grid.grid[cx + 2, cy, cz] = block_type
    # Air beyond the block
    grid.grid[cx + 3, cy, cz] = VOXEL_AIR
    return (cx + 3, cy, cz)


class TestFunctionalBlockPropagation:
    """Claiming should propagate through all functional block types."""

    @pytest.mark.parametrize("block_type,name", [
        (VOXEL_SLOPE, "Slope"),
        (VOXEL_STAIRS, "Stairs"),
        (VOXEL_DOOR, "Door"),
        (VOXEL_TARP, "Tarp"),
        (VOXEL_SPIKE, "Spike"),
        (VOXEL_TREASURE, "Treasure"),
        (VOXEL_ROLLING_STONE, "Rolling Stone"),
        (VOXEL_REINFORCED_WALL, "Reinforced Wall"),
    ])
    def test_claiming_propagates_through_block(self, block_type: int, name: str):
        """Air beyond a functional block should be claimed."""
        grid = _make_grid()
        beyond_pos = _carve_corridor_with_block(grid, block_type)

        bus = EventBus()
        system = ClaimedTerritorySystem(bus, grid)

        bx, by, bz = beyond_pos
        assert bool(grid.claimed[bx, by, bz]) is True, (
            f"Air beyond {name} should be claimed"
        )

    @pytest.mark.parametrize("block_type,name", [
        (VOXEL_SLOPE, "Slope"),
        (VOXEL_STAIRS, "Stairs"),
        (VOXEL_DOOR, "Door"),
        (VOXEL_TARP, "Tarp"),
        (VOXEL_SPIKE, "Spike"),
        (VOXEL_TREASURE, "Treasure"),
        (VOXEL_ROLLING_STONE, "Rolling Stone"),
        (VOXEL_REINFORCED_WALL, "Reinforced Wall"),
    ])
    def test_functional_block_itself_is_claimed(self, block_type: int, name: str):
        """The functional block cell itself should be claimed."""
        grid = _make_grid()
        _carve_corridor_with_block(grid, block_type)

        bus = EventBus()
        system = ClaimedTerritorySystem(bus, grid)

        cx = CORE_X + 2
        assert bool(grid.claimed[cx, CORE_Y, CORE_Z]) is True, (
            f"{name} block itself should be claimed"
        )

    @pytest.mark.parametrize("block_type,name", [
        (VOXEL_SLOPE, "Slope"),
        (VOXEL_STAIRS, "Stairs"),
        (VOXEL_DOOR, "Door"),
        (VOXEL_SPIKE, "Spike"),
        (VOXEL_TREASURE, "Treasure"),
        (VOXEL_REINFORCED_WALL, "Reinforced Wall"),
    ])
    def test_functional_block_is_visible(self, block_type: int, name: str):
        """Claimed functional blocks should be visible (for rendering)."""
        grid = _make_grid()
        _carve_corridor_with_block(grid, block_type)

        bus = EventBus()
        system = ClaimedTerritorySystem(bus, grid)

        cx = CORE_X + 2
        assert bool(grid.visible[cx, CORE_Y, CORE_Z]) is True, (
            f"Claimed {name} should be visible"
        )


class TestNaturalSolidsStillBlock:
    """Natural solid blocks should still block claiming."""

    def test_stone_blocks_propagation(self):
        """Stone should NOT allow claiming through."""
        grid = _make_grid()
        cx, cy, cz = CORE_X, CORE_Y, CORE_Z
        grid.grid[cx + 1, cy, cz] = VOXEL_AIR
        # Stone stays at cx+2 (default fill)
        grid.grid[cx + 3, cy, cz] = VOXEL_AIR

        bus = EventBus()
        system = ClaimedTerritorySystem(bus, grid)

        # Air beyond stone should NOT be claimed
        assert bool(grid.claimed[cx + 3, cy, cz]) is False

    def test_lava_blocks_propagation(self):
        """Lava should NOT allow claiming through."""
        grid = _make_grid()
        cx, cy, cz = CORE_X, CORE_Y, CORE_Z
        grid.grid[cx + 1, cy, cz] = VOXEL_AIR
        grid.grid[cx + 2, cy, cz] = VOXEL_LAVA
        grid.grid[cx + 3, cy, cz] = VOXEL_AIR

        bus = EventBus()
        system = ClaimedTerritorySystem(bus, grid)

        assert bool(grid.claimed[cx + 3, cy, cz]) is False


class TestStairsVerticalClaiming:
    """Stairs should allow vertical territory propagation."""

    def test_stairs_bridge_z_levels(self):
        """Air above stairs should be claimable (vertical propagation)."""
        grid = _make_grid()
        cx, cy, cz = CORE_X, CORE_Y, CORE_Z
        # Air adjacent to core
        grid.grid[cx + 1, cy, cz] = VOXEL_AIR
        # Stairs at cx+2
        grid.grid[cx + 2, cy, cz] = VOXEL_STAIRS
        # Air above stairs (one z-level up, lower z-index)
        grid.grid[cx + 2, cy, cz - 1] = VOXEL_AIR

        bus = EventBus()
        system = ClaimedTerritorySystem(bus, grid)

        assert bool(grid.claimed[cx + 2, cy, cz - 1]) is True, (
            "Air above stairs should be claimed via vertical propagation"
        )
