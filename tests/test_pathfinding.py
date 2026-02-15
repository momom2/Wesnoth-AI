"""Tests for A* pathfinding."""

from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.world.pathfinding import AStarPathfinder
from dungeon_builder.config import VOXEL_AIR, VOXEL_STONE


def _make_tunnel_grid():
    """Create a grid with a horizontal tunnel at z=1."""
    grid = VoxelGrid(width=10, depth=10, height=3)
    grid.grid[:] = VOXEL_STONE
    grid.grid[:, :, 0] = VOXEL_AIR  # Surface is air

    # Tunnel at z=1 from (0,0) to (9,0)
    for x in range(10):
        grid.grid[x, 0, 1] = VOXEL_AIR

    return grid


def test_horizontal_path():
    grid = _make_tunnel_grid()
    pf = AStarPathfinder(grid)
    path = pf.find_path((0, 0, 1), (9, 0, 1))
    assert path is not None
    assert path[0] == (0, 0, 1)
    assert path[-1] == (9, 0, 1)
    assert len(path) == 10


def test_no_path():
    grid = VoxelGrid(width=10, depth=10, height=3)
    grid.grid[:] = VOXEL_STONE
    # Two isolated air cells
    grid.grid[0, 0, 0] = VOXEL_AIR
    grid.grid[9, 9, 0] = VOXEL_AIR
    pf = AStarPathfinder(grid)
    path = pf.find_path((0, 0, 0), (9, 9, 0))
    assert path is None


def test_same_start_and_goal():
    grid = VoxelGrid(width=4, depth=4, height=2)
    grid.grid[:, :, 0] = VOXEL_AIR
    pf = AStarPathfinder(grid)
    path = pf.find_path((0, 0, 0), (0, 0, 0))
    assert path == [(0, 0, 0)]


def test_vertical_path():
    grid = VoxelGrid(width=4, depth=4, height=4)
    grid.grid[:] = VOXEL_STONE
    # Vertical shaft at (1,1)
    for z in range(4):
        grid.grid[1, 1, z] = VOXEL_AIR

    pf = AStarPathfinder(grid)
    path = pf.find_path((1, 1, 0), (1, 1, 3))
    assert path is not None
    assert path[0] == (1, 1, 0)
    assert path[-1] == (1, 1, 3)


def test_l_shaped_path():
    grid = VoxelGrid(width=10, depth=10, height=2)
    grid.grid[:] = VOXEL_STONE
    # Horizontal tunnel at y=0
    for x in range(5):
        grid.grid[x, 0, 0] = VOXEL_AIR
    # Turn and go along y
    for y in range(5):
        grid.grid[4, y, 0] = VOXEL_AIR

    pf = AStarPathfinder(grid)
    path = pf.find_path((0, 0, 0), (4, 4, 0))
    assert path is not None
    assert path[0] == (0, 0, 0)
    assert path[-1] == (4, 4, 0)
    # Path should go right then up (or equivalent)
    assert len(path) == 9  # 5 steps right + 4 steps up (including start)


def test_solid_start_returns_none():
    grid = VoxelGrid(width=4, depth=4, height=2)
    grid.grid[:] = VOXEL_STONE
    grid.grid[3, 3, 0] = VOXEL_AIR
    pf = AStarPathfinder(grid)
    path = pf.find_path((0, 0, 0), (3, 3, 0))
    assert path is None


def test_solid_goal_returns_none():
    grid = VoxelGrid(width=4, depth=4, height=2)
    grid.grid[:] = VOXEL_STONE
    grid.grid[0, 0, 0] = VOXEL_AIR
    pf = AStarPathfinder(grid)
    path = pf.find_path((0, 0, 0), (3, 3, 0))
    assert path is None
