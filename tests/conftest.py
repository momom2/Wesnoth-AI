"""Shared test fixtures."""

import pytest
from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.config import VOXEL_AIR, VOXEL_STONE


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def small_grid():
    """A small 16x16x5 grid for fast tests."""
    grid = VoxelGrid(width=16, depth=16, height=5)
    # Fill with stone, surface is air
    grid.grid[:, :, 0] = VOXEL_AIR
    grid.grid[:, :, 1:] = VOXEL_STONE
    return grid
