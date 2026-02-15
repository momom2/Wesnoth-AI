"""Tests for intruder line-of-sight and special vision systems."""

import pytest

from dungeon_builder.intruders.vision import (
    bresenham_3d,
    compute_los,
    compute_arcane_sight,
    compute_thermal_vision,
)
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_STONE,
    VOXEL_DOOR,
    VOXEL_SLOPE,
    VOXEL_STAIRS,
    VOXEL_WATER,
    VOXEL_LAVA,
)


def _make_air_grid(w=10, d=10, h=5) -> VoxelGrid:
    """All-air grid for unobstructed LOS tests."""
    grid = VoxelGrid(width=w, depth=d, height=h)
    # Default is all air
    return grid


def _make_walled_grid() -> VoxelGrid:
    """10x10x5 grid with a stone wall at y=5."""
    grid = VoxelGrid(width=10, depth=10, height=5)
    for x in range(10):
        for z in range(5):
            grid.set(x, 5, z, VOXEL_STONE)
    return grid


# ── Bresenham 3D tests ──────────────────────────────────────────────


class TestBresenham3D:
    def test_single_cell(self):
        cells = bresenham_3d(5, 5, 5, 5, 5, 5)
        assert cells == [(5, 5, 5)]

    def test_x_axis_line(self):
        cells = bresenham_3d(0, 0, 0, 4, 0, 0)
        assert cells == [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)]

    def test_y_axis_line(self):
        cells = bresenham_3d(0, 0, 0, 0, 3, 0)
        assert cells == [(0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0)]

    def test_z_axis_line(self):
        cells = bresenham_3d(0, 0, 0, 0, 0, 2)
        assert cells == [(0, 0, 0), (0, 0, 1), (0, 0, 2)]

    def test_negative_direction(self):
        cells = bresenham_3d(3, 0, 0, 0, 0, 0)
        assert cells[0] == (3, 0, 0)
        assert cells[-1] == (0, 0, 0)
        assert len(cells) == 4

    def test_diagonal_2d(self):
        cells = bresenham_3d(0, 0, 0, 3, 3, 0)
        assert cells[0] == (0, 0, 0)
        assert cells[-1] == (3, 3, 0)
        assert len(cells) == 4  # Diagonal steps = max(dx,dy,dz)+1

    def test_diagonal_3d(self):
        cells = bresenham_3d(0, 0, 0, 2, 2, 2)
        assert cells[0] == (0, 0, 0)
        assert cells[-1] == (2, 2, 2)
        assert len(cells) == 3

    def test_contains_both_endpoints(self):
        cells = bresenham_3d(1, 2, 3, 5, 7, 1)
        assert cells[0] == (1, 2, 3)
        assert cells[-1] == (5, 7, 1)


# ── LOS through air tests ──────────────────────────────────────────


class TestLOSAir:
    def test_origin_always_visible(self):
        grid = _make_air_grid()
        visible = compute_los(grid, 5, 5, 2, 3)
        assert (5, 5, 2) in visible

    def test_adjacent_cells_visible(self):
        grid = _make_air_grid()
        visible = compute_los(grid, 5, 5, 2, 3)
        for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                            (0, 0, 1), (0, 0, -1)]:
            assert (5 + dx, 5 + dy, 2 + dz) in visible

    def test_range_1_sees_neighbors(self):
        grid = _make_air_grid()
        visible = compute_los(grid, 5, 5, 2, 1)
        # With range 1, should see all 26 neighbors + self
        assert len(visible) >= 7  # At minimum axis-aligned + self

    def test_far_cell_visible_in_air(self):
        grid = _make_air_grid()
        visible = compute_los(grid, 5, 5, 2, 4)
        assert (9, 5, 2) in visible  # 4 cells away in X

    def test_out_of_bounds_not_visible(self):
        grid = _make_air_grid(w=10, d=10, h=5)
        visible = compute_los(grid, 0, 0, 0, 3)
        # Negative coords should not appear
        for (x, y, z) in visible:
            assert x >= 0 and y >= 0 and z >= 0


# ── LOS blocked by walls ───────────────────────────────────────────


class TestLOSBlocked:
    def test_wall_blocks_los(self):
        grid = _make_walled_grid()  # Wall at y=5
        visible = compute_los(grid, 5, 3, 2, 5)
        # The wall itself should be visible
        assert (5, 5, 2) in visible
        # But cells beyond the wall should NOT be visible
        assert (5, 7, 2) not in visible

    def test_wall_face_visible(self):
        grid = _make_walled_grid()
        visible = compute_los(grid, 5, 3, 2, 4)
        assert (5, 5, 2) in visible  # Can see the wall face

    def test_stone_block_opaque(self):
        grid = _make_air_grid()
        grid.set(5, 5, 2, VOXEL_STONE)  # Single stone block
        visible = compute_los(grid, 5, 3, 2, 4)
        # Stone is visible (can see it)
        assert (5, 5, 2) in visible
        # But ray should be blocked beyond it
        assert (5, 7, 2) not in visible


# ── LOS through doors ──────────────────────────────────────────────


class TestLOSDoors:
    def test_open_door_transparent(self):
        grid = _make_air_grid()
        grid.set(5, 5, 2, VOXEL_DOOR)
        grid.block_state[5, 5, 2] = 0  # Open
        visible = compute_los(grid, 5, 3, 2, 5)
        assert (5, 5, 2) in visible   # Door visible
        assert (5, 7, 2) in visible   # Can see beyond open door

    def test_closed_door_blocks(self):
        grid = _make_air_grid()
        grid.set(5, 5, 2, VOXEL_DOOR)
        grid.block_state[5, 5, 2] = 1  # Closed
        visible = compute_los(grid, 5, 3, 2, 5)
        assert (5, 5, 2) in visible   # Door visible
        assert (5, 7, 2) not in visible  # Blocked by closed door


# ── LOS through slopes/stairs ──────────────────────────────────────


class TestLOSSlopesStairs:
    def test_slope_transparent(self):
        grid = _make_air_grid()
        grid.set(5, 5, 2, VOXEL_SLOPE)
        visible = compute_los(grid, 5, 3, 2, 5)
        assert (5, 5, 2) in visible
        assert (5, 7, 2) in visible  # Can see beyond slope

    def test_stairs_transparent(self):
        grid = _make_air_grid()
        grid.set(5, 5, 2, VOXEL_STAIRS)
        visible = compute_los(grid, 5, 3, 2, 5)
        assert (5, 5, 2) in visible
        assert (5, 7, 2) in visible


# ── LOS through water ──────────────────────────────────────────────


class TestLOSWater:
    def test_thin_water_transparent(self):
        grid = _make_air_grid()
        grid.set(5, 5, 2, VOXEL_WATER)
        visible = compute_los(grid, 5, 3, 2, 5)
        assert (5, 5, 2) in visible
        assert (5, 6, 2) in visible  # Single water cell: still transparent

    def test_thick_water_blocks(self):
        grid = _make_air_grid()
        # 3 water cells in a row
        grid.set(5, 4, 2, VOXEL_WATER)
        grid.set(5, 5, 2, VOXEL_WATER)
        grid.set(5, 6, 2, VOXEL_WATER)
        visible = compute_los(grid, 5, 2, 2, 7)
        # First 2 water cells should be visible (within WATER_LOS_DEPTH=2)
        assert (5, 4, 2) in visible
        assert (5, 5, 2) in visible
        # Third water cell and beyond should be blocked
        assert (5, 6, 2) not in visible


# ── Perception range ────────────────────────────────────────────────


class TestPerceptionRange:
    def test_range_limits_visibility(self):
        grid = _make_air_grid(w=20, d=20, h=5)
        visible = compute_los(grid, 10, 10, 2, 3)
        # Cell 4 away should NOT be visible (outside range)
        assert (10, 14, 2) not in visible
        # Cell 3 away should be visible
        assert (10, 13, 2) in visible

    def test_larger_range_sees_more(self):
        grid = _make_air_grid(w=20, d=20, h=5)
        small = compute_los(grid, 10, 10, 2, 2)
        large = compute_los(grid, 10, 10, 2, 4)
        assert len(large) > len(small)


# ── Arcane sight tests ──────────────────────────────────────────────


class TestArcaneSight:
    def test_sees_through_walls(self):
        grid = _make_walled_grid()  # Wall at y=5
        visible = compute_arcane_sight(grid, 5, 3, 2, 4)
        # Should see cells beyond the wall
        assert (5, 7, 2) in visible

    def test_manhattan_distance(self):
        grid = _make_air_grid()
        visible = compute_arcane_sight(grid, 5, 5, 2, 2)
        # Manhattan distance 2: (5+2,5,2) should be visible
        assert (7, 5, 2) in visible
        # Manhattan distance 3: should NOT be visible
        assert (8, 5, 2) not in visible

    def test_origin_visible(self):
        grid = _make_air_grid()
        visible = compute_arcane_sight(grid, 5, 5, 2, 1)
        assert (5, 5, 2) in visible

    def test_out_of_bounds_excluded(self):
        grid = _make_air_grid(w=10, d=10, h=5)
        visible = compute_arcane_sight(grid, 0, 0, 0, 3)
        for (x, y, z) in visible:
            assert grid.in_bounds(x, y, z)


# ── Thermal vision tests ───────────────────────────────────────────


class TestThermalVision:
    def test_sees_hot_blocks_through_walls(self):
        grid = _make_walled_grid()
        # Set a hot block behind the wall
        grid.set(5, 7, 2, VOXEL_LAVA)
        grid.temperature[5, 7, 2] = 1000.0
        visible = compute_thermal_vision(grid, 5, 3, 2, 6, heat_threshold=100.0)
        assert (5, 7, 2) in visible

    def test_does_not_see_cold_blocks(self):
        grid = _make_walled_grid()
        grid.temperature[5, 7, 2] = 50.0  # Below threshold
        visible = compute_thermal_vision(grid, 5, 3, 2, 6, heat_threshold=100.0)
        assert (5, 7, 2) not in visible

    def test_range_limits_thermal(self):
        grid = _make_air_grid(w=20, d=20, h=5)
        grid.temperature[10, 17, 2] = 500.0
        visible = compute_thermal_vision(grid, 10, 10, 2, 4, heat_threshold=100.0)
        # Distance 7 > range 4: not visible
        assert (10, 17, 2) not in visible

    def test_empty_when_no_heat(self):
        grid = _make_air_grid()
        visible = compute_thermal_vision(grid, 5, 5, 2, 3, heat_threshold=100.0)
        assert len(visible) == 0
