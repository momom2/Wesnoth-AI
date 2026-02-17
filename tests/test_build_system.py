"""Tests for the dig/build system."""

import pytest

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.building.build_system import BuildSystem
from dungeon_builder.config import VOXEL_AIR, VOXEL_STONE, VOXEL_DIRT, VOXEL_BEDROCK, VOXEL_LAVA


def test_queue_dig_on_stone():
    bus = EventBus()
    grid = VoxelGrid(width=4, depth=4, height=4)
    grid.grid[1, 1, 1] = VOXEL_STONE
    grid.visible[:] = True
    bs = BuildSystem(bus, grid)

    assert bs.queue_dig(1, 1, 1) is True
    assert len(bs.dig_queue) == 1


def test_cannot_dig_air():
    bus = EventBus()
    grid = VoxelGrid(width=4, depth=4, height=4)
    grid.visible[:] = True  # must be visible for validation to fire
    bs = BuildSystem(bus, grid)

    assert bs.queue_dig(0, 0, 0) is False  # Air


def test_cannot_dig_bedrock():
    bus = EventBus()
    grid = VoxelGrid(width=4, depth=4, height=4)
    grid.grid[0, 0, 0] = VOXEL_BEDROCK
    grid.visible[:] = True
    bs = BuildSystem(bus, grid)

    assert bs.queue_dig(0, 0, 0) is False


def test_cannot_dig_lava():
    bus = EventBus()
    grid = VoxelGrid(width=4, depth=4, height=4)
    grid.grid[0, 0, 0] = VOXEL_LAVA
    grid.visible[:] = True
    bs = BuildSystem(bus, grid)

    assert bs.queue_dig(0, 0, 0) is False


def test_dig_creates_loose_material():
    """Dig completes by making material loose, not removing it."""
    bus = EventBus()
    grid = VoxelGrid(width=4, depth=4, height=4)
    grid.grid[1, 1, 1] = VOXEL_DIRT  # Dirt takes 20 ticks
    grid.visible[:] = True
    bs = BuildSystem(bus, grid)
    bs.queue_dig(1, 1, 1)

    completions = []
    bus.subscribe("dig_complete", lambda **kw: completions.append(kw))

    # Simulate 19 ticks — not yet complete
    for i in range(1, 20):
        bus.publish("tick", tick=i)
    assert len(completions) == 0
    assert grid.get(1, 1, 1) == VOXEL_DIRT
    assert not grid.is_loose(1, 1, 1)

    # 20th tick — complete, material is now loose
    bus.publish("tick", tick=20)
    assert len(completions) == 1
    assert grid.get(1, 1, 1) == VOXEL_DIRT  # Still dirt, but loose
    assert grid.is_loose(1, 1, 1)


def test_cannot_dig_already_loose():
    """Can't queue a dig on already-loose material."""
    bus = EventBus()
    grid = VoxelGrid(width=4, depth=4, height=4)
    grid.grid[1, 1, 1] = VOXEL_STONE
    grid.set_loose(1, 1, 1, True)
    grid.visible[:] = True
    bs = BuildSystem(bus, grid)

    errors = []
    bus.subscribe("error_message", lambda **kw: errors.append(kw))

    assert bs.queue_dig(1, 1, 1) is False
    assert len(errors) == 1
    assert "already loose" in errors[0]["text"]


def test_no_double_queue():
    bus = EventBus()
    grid = VoxelGrid(width=4, depth=4, height=4)
    grid.grid[1, 1, 1] = VOXEL_STONE
    grid.visible[:] = True
    bs = BuildSystem(bus, grid)

    assert bs.queue_dig(1, 1, 1) is True
    assert bs.queue_dig(1, 1, 1) is False  # Already queued


def test_stone_takes_longer_than_dirt():
    bus = EventBus()
    grid = VoxelGrid(width=4, depth=4, height=4)
    grid.grid[0, 0, 0] = VOXEL_DIRT
    grid.grid[1, 0, 0] = VOXEL_STONE
    grid.visible[:] = True
    bs = BuildSystem(bus, grid)
    bs.queue_dig(0, 0, 0)
    bs.queue_dig(1, 0, 0)

    for i in range(1, 21):
        bus.publish("tick", tick=i)
    assert grid.is_loose(0, 0, 0)  # Dirt done at 20
    assert not grid.is_loose(1, 0, 0)  # Stone not yet done

    for i in range(21, 41):
        bus.publish("tick", tick=i)
    assert grid.is_loose(1, 0, 0)  # Stone done at 40


def test_left_click_dig_mode():
    """voxel_left_clicked with mode=dig should queue a dig."""
    bus = EventBus()
    grid = VoxelGrid(width=4, depth=4, height=4)
    grid.grid[1, 1, 1] = VOXEL_STONE
    grid.visible[:] = True
    bs = BuildSystem(bus, grid)

    bus.publish("voxel_left_clicked", x=1, y=1, z=1, mode="dig")
    assert len(bs.dig_queue) == 1


# ---------------------------------------------------------------------------
# Dig overlay: is_being_dug and get_dig_progress
# ---------------------------------------------------------------------------


def test_is_being_dug_queued():
    """Queued blocks report as being dug."""
    bus = EventBus()
    grid = VoxelGrid(width=4, depth=4, height=4)
    grid.grid[1, 1, 1] = VOXEL_STONE
    grid.visible[:] = True
    bs = BuildSystem(bus, grid)
    bs.queue_dig(1, 1, 1)

    assert bs.is_being_dug(1, 1, 1) is True
    assert bs.is_being_dug(0, 0, 0) is False


def test_is_being_dug_active():
    """Active digs report as being dug."""
    bus = EventBus()
    grid = VoxelGrid(width=4, depth=4, height=4)
    grid.grid[1, 1, 1] = VOXEL_DIRT
    grid.visible[:] = True
    bs = BuildSystem(bus, grid)
    bs.queue_dig(1, 1, 1)

    # First tick promotes to active
    bus.publish("tick", tick=1)
    assert bs.is_being_dug(1, 1, 1) is True


def test_is_being_dug_completed():
    """Completed digs are no longer reported as being dug."""
    bus = EventBus()
    grid = VoxelGrid(width=4, depth=4, height=4)
    grid.grid[1, 1, 1] = VOXEL_DIRT  # 20 ticks
    grid.visible[:] = True
    bs = BuildSystem(bus, grid)
    bs.queue_dig(1, 1, 1)

    for i in range(1, 21):
        bus.publish("tick", tick=i)

    assert bs.is_being_dug(1, 1, 1) is False


def test_dig_progress_queued():
    """Queued digs have progress 0.0."""
    bus = EventBus()
    grid = VoxelGrid(width=4, depth=4, height=4)
    grid.grid[1, 1, 1] = VOXEL_STONE
    grid.visible[:] = True
    bs = BuildSystem(bus, grid)
    bs.queue_dig(1, 1, 1)

    assert bs.get_dig_progress(1, 1, 1) == pytest.approx(0.0)


def test_dig_progress_active():
    """Active digs report fractional progress."""
    bus = EventBus()
    grid = VoxelGrid(width=4, depth=4, height=4)
    grid.grid[1, 1, 1] = VOXEL_DIRT  # 20 ticks
    grid.visible[:] = True
    bs = BuildSystem(bus, grid)
    bs.queue_dig(1, 1, 1)

    # Tick 1 promotes to active and decrements once (19 remaining / 20 total)
    bus.publish("tick", tick=1)
    assert bs.get_dig_progress(1, 1, 1) == pytest.approx(1.0 / 20.0)

    # After 10 more ticks (11 total), 9 remaining / 20 total
    for i in range(2, 12):
        bus.publish("tick", tick=i)
    assert bs.get_dig_progress(1, 1, 1) == pytest.approx(11.0 / 20.0)


def test_dig_progress_not_digging():
    """Non-digging blocks return -1.0."""
    bus = EventBus()
    grid = VoxelGrid(width=4, depth=4, height=4)
    bs = BuildSystem(bus, grid)

    assert bs.get_dig_progress(0, 0, 0) == pytest.approx(-1.0)
