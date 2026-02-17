"""Tests for fog-of-war rendering in _get_color()."""

import pytest

from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.rendering.voxel_renderer import ChunkMeshBuilder
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_STONE,
    VOXEL_CORE,
    FOG_COLOR,
    RENDER_MODE_MATTER,
    RENDER_MODE_STRUCTURAL,
    RENDER_MODE_HUMIDITY,
    RENDER_MODE_HEAT,
    VOXEL_COLORS,
)


def _make_grid():
    grid = VoxelGrid(width=8, depth=8, height=3)
    grid.grid[:] = VOXEL_STONE
    return grid


class TestFogOfWar:
    def test_non_visible_block_returns_fog_color(self):
        grid = _make_grid()
        builder = ChunkMeshBuilder()
        # visible array is all False by default
        color = builder._get_color(grid, 4, 4, 1, VOXEL_STONE, RENDER_MODE_MATTER)
        assert color == FOG_COLOR

    def test_visible_block_returns_normal_color(self):
        grid = _make_grid()
        grid.visible[4, 4, 1] = True
        builder = ChunkMeshBuilder()
        color = builder._get_color(grid, 4, 4, 1, VOXEL_STONE, RENDER_MODE_MATTER)
        expected = VOXEL_COLORS[VOXEL_STONE]
        assert color == expected

    def test_fog_applies_in_structural_mode(self):
        grid = _make_grid()
        builder = ChunkMeshBuilder()
        color = builder._get_color(grid, 4, 4, 1, VOXEL_STONE, RENDER_MODE_STRUCTURAL)
        assert color == FOG_COLOR

    def test_fog_applies_in_humidity_mode(self):
        grid = _make_grid()
        builder = ChunkMeshBuilder()
        color = builder._get_color(grid, 4, 4, 1, VOXEL_STONE, RENDER_MODE_HUMIDITY)
        assert color == FOG_COLOR

    def test_fog_applies_in_heat_mode(self):
        grid = _make_grid()
        builder = ChunkMeshBuilder()
        color = builder._get_color(grid, 4, 4, 1, VOXEL_STONE, RENDER_MODE_HEAT)
        assert color == FOG_COLOR

    def test_core_visible_gets_normal_color(self):
        grid = _make_grid()
        grid.grid[4, 4, 1] = VOXEL_CORE
        grid.visible[4, 4, 1] = True
        builder = ChunkMeshBuilder()
        color = builder._get_color(grid, 4, 4, 1, VOXEL_CORE, RENDER_MODE_MATTER)
        expected = VOXEL_COLORS[VOXEL_CORE]
        assert color == expected
