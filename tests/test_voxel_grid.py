"""Tests for the VoxelGrid data structure."""

from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.config import VOXEL_AIR, VOXEL_STONE, VOXEL_BEDROCK


def test_initial_state_all_air():
    grid = VoxelGrid(width=4, depth=4, height=4)
    assert grid.get(0, 0, 0) == VOXEL_AIR
    assert grid.get(3, 3, 3) == VOXEL_AIR


def test_set_and_get():
    grid = VoxelGrid(width=4, depth=4, height=4)
    grid.set(1, 2, 3, VOXEL_STONE)
    assert grid.get(1, 2, 3) == VOXEL_STONE


def test_out_of_bounds_returns_bedrock():
    grid = VoxelGrid(width=4, depth=4, height=4)
    assert grid.get(-1, 0, 0) == VOXEL_BEDROCK
    assert grid.get(0, -1, 0) == VOXEL_BEDROCK
    assert grid.get(0, 0, -1) == VOXEL_BEDROCK
    assert grid.get(4, 0, 0) == VOXEL_BEDROCK
    assert grid.get(0, 4, 0) == VOXEL_BEDROCK
    assert grid.get(0, 0, 4) == VOXEL_BEDROCK


def test_in_bounds():
    grid = VoxelGrid(width=4, depth=4, height=4)
    assert grid.in_bounds(0, 0, 0) is True
    assert grid.in_bounds(3, 3, 3) is True
    assert grid.in_bounds(4, 0, 0) is False
    assert grid.in_bounds(-1, 0, 0) is False


def test_is_solid():
    grid = VoxelGrid(width=4, depth=4, height=4)
    assert grid.is_solid(0, 0, 0) is False  # Air
    grid.set(0, 0, 0, VOXEL_STONE)
    assert grid.is_solid(0, 0, 0) is True


def test_dirty_chunks_on_set():
    grid = VoxelGrid(width=32, depth=32, height=5)
    grid.pop_dirty_chunks()  # Clear initial state

    grid.set(0, 0, 0, VOXEL_STONE)
    dirty = grid.pop_dirty_chunks()
    # Should include the chunk containing (0,0,0) and adjacent layers
    assert (0, 0, 0) in dirty
    assert (0, 0, 1) in dirty  # Layer below


def test_set_same_value_no_dirty():
    grid = VoxelGrid(width=4, depth=4, height=4)
    grid.pop_dirty_chunks()
    grid.set(0, 0, 0, VOXEL_AIR)  # Already air
    dirty = grid.pop_dirty_chunks()
    assert len(dirty) == 0


def test_mark_all_dirty():
    grid = VoxelGrid(width=16, depth=16, height=2)
    grid.mark_all_dirty()
    dirty = grid.pop_dirty_chunks()
    assert len(dirty) == 1 * 1 * 2  # 1x1 chunks (16x16 grid with CHUNK_SIZE=16), 2 levels


def test_event_bus_integration():
    from dungeon_builder.core.event_bus import EventBus

    bus = EventBus()
    grid = VoxelGrid(width=4, depth=4, height=4)
    events = []
    bus.subscribe("voxel_changed", lambda **kw: events.append(kw))

    grid.set(1, 1, 1, VOXEL_STONE, event_bus=bus)
    assert len(events) == 1
    assert events[0]["x"] == 1
    assert events[0]["y"] == 1
    assert events[0]["z"] == 1
    assert events[0]["old_type"] == VOXEL_AIR
    assert events[0]["new_type"] == VOXEL_STONE


def test_temperature_get_set():
    grid = VoxelGrid(width=4, depth=4, height=4)
    assert grid.get_temperature(1, 1, 1) == 0.0
    grid.set_temperature(1, 1, 1, 42.5)
    assert grid.get_temperature(1, 1, 1) == 42.5


def test_temperature_out_of_bounds():
    grid = VoxelGrid(width=4, depth=4, height=4)
    assert grid.get_temperature(-1, 0, 0) == 0.0
    grid.set_temperature(-1, 0, 0, 100.0)  # should not crash


def test_loose_get_set():
    grid = VoxelGrid(width=4, depth=4, height=4)
    grid.grid[1, 1, 1] = VOXEL_STONE
    assert grid.is_loose(1, 1, 1) is False
    grid.set_loose(1, 1, 1, True)
    assert grid.is_loose(1, 1, 1) is True
    grid.set_loose(1, 1, 1, False)
    assert grid.is_loose(1, 1, 1) is False


def test_set_air_clears_loose():
    grid = VoxelGrid(width=4, depth=4, height=4)
    grid.grid[1, 1, 1] = VOXEL_STONE
    grid.set_loose(1, 1, 1, True)
    assert grid.is_loose(1, 1, 1) is True

    grid.set(1, 1, 1, VOXEL_AIR)
    assert grid.is_loose(1, 1, 1) is False


def test_humidity_get_set():
    grid = VoxelGrid(width=4, depth=4, height=4)
    assert grid.get_humidity(1, 1, 1) == 0.0
    grid.set_humidity(1, 1, 1, 0.75)
    assert grid.get_humidity(1, 1, 1) == 0.75


def test_humidity_clamped():
    grid = VoxelGrid(width=4, depth=4, height=4)
    grid.set_humidity(1, 1, 1, 2.0)
    assert grid.get_humidity(1, 1, 1) == 1.0
    grid.set_humidity(1, 1, 1, -1.0)
    assert grid.get_humidity(1, 1, 1) == 0.0
