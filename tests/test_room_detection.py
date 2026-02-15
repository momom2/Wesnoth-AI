"""Tests for room detection."""

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.world.room_detection import RoomDetector
from dungeon_builder.config import VOXEL_AIR, VOXEL_STONE


def _make_enclosed_room_grid():
    """Create a grid with a 4x4 enclosed air room at z=1."""
    grid = VoxelGrid(width=10, depth=10, height=3)
    grid.grid[:] = VOXEL_STONE
    # Carve a 4x4 room in the middle at z=1
    for x in range(3, 7):
        for y in range(3, 7):
            grid.grid[x, y, 1] = VOXEL_AIR
    return grid


def test_detect_enclosed_room():
    bus = EventBus()
    grid = _make_enclosed_room_grid()
    detector = RoomDetector(bus, grid)

    rooms_detected = []
    bus.subscribe("room_detected", lambda room: rooms_detected.append(room))

    # Trigger detection by simulating a dig_complete in the room
    detector._on_dig_complete(4, 4, 1)

    assert len(rooms_detected) == 1
    assert rooms_detected[0].size == 16  # 4x4


def test_no_room_if_open():
    """Air region touching map edge should not be detected as a room."""
    grid = VoxelGrid(width=10, depth=10, height=3)
    grid.grid[:] = VOXEL_STONE
    # Air corridor from edge to center
    for x in range(10):
        grid.grid[x, 0, 1] = VOXEL_AIR

    bus = EventBus()
    detector = RoomDetector(bus, grid)
    rooms_detected = []
    bus.subscribe("room_detected", lambda room: rooms_detected.append(room))

    detector._on_dig_complete(5, 0, 1)
    assert len(rooms_detected) == 0


def test_no_room_if_too_small():
    """Enclosed region with fewer than MIN_ROOM_SIZE cells is not a room."""
    grid = VoxelGrid(width=10, depth=10, height=3)
    grid.grid[:] = VOXEL_STONE
    # 1x2 enclosed air (only 2 cells, below minimum of 4)
    grid.grid[5, 5, 1] = VOXEL_AIR
    grid.grid[5, 6, 1] = VOXEL_AIR

    bus = EventBus()
    detector = RoomDetector(bus, grid)
    rooms_detected = []
    bus.subscribe("room_detected", lambda room: rooms_detected.append(room))

    detector._on_dig_complete(5, 5, 1)
    assert len(rooms_detected) == 0


def test_room_on_solid_voxel_ignored():
    grid = VoxelGrid(width=10, depth=10, height=3)
    grid.grid[:] = VOXEL_STONE

    bus = EventBus()
    detector = RoomDetector(bus, grid)
    rooms_detected = []
    bus.subscribe("room_detected", lambda room: rooms_detected.append(room))

    # Trying to detect at a solid voxel
    detector._on_dig_complete(5, 5, 1)
    assert len(rooms_detected) == 0
