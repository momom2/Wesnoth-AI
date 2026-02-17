"""Tests for the move (pick up / drop) system."""

import pytest

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.core.game_state import GameState
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.building.move_system import MoveSystem
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_STONE,
    VOXEL_DIRT,
    VOXEL_GRANITE,
    DEFAULT_SEED,
)


def _setup(vtype: int = VOXEL_STONE, loose: bool = True):
    bus = EventBus()
    grid = VoxelGrid(width=4, depth=4, height=4)
    # Mark all blocks visible and claimed so territory checks don't interfere
    grid.visible[:] = True
    grid.claimed[:] = True
    grid.grid[1, 1, 1] = vtype
    if loose:
        grid.set_loose(1, 1, 1, True)
    gs = GameState(DEFAULT_SEED)
    gs.event_bus = bus
    ms = MoveSystem(bus, grid, gs)
    return bus, grid, ms


def test_pick_up_loose():
    bus, grid, ms = _setup(VOXEL_STONE, loose=True)
    assert ms.pick_up(1, 1, 1) is True
    assert ms.held_materials == {VOXEL_STONE: 1}
    assert grid.get(1, 1, 1) == VOXEL_AIR


def test_pick_up_non_loose_rejected():
    bus, grid, ms = _setup(VOXEL_STONE, loose=False)
    errors = []
    bus.subscribe("error_message", lambda **kw: errors.append(kw))

    assert ms.pick_up(1, 1, 1) is False
    assert ms.held_materials == {}
    assert len(errors) == 1
    assert "dig first" in errors[0]["text"].lower()


def test_pick_up_air_rejected():
    bus, grid, ms = _setup()
    # Pick a cell that's air
    assert ms.pick_up(0, 0, 0) is False
    assert ms.held_materials == {}


def test_pick_up_stacks_same_type():
    bus, grid, ms = _setup(VOXEL_STONE, loose=True)
    # Also set up a second loose stone
    grid.grid[2, 1, 1] = VOXEL_STONE
    grid.set_loose(2, 1, 1, True)

    assert ms.pick_up(1, 1, 1) is True
    assert ms.held_materials == {VOXEL_STONE: 1}

    assert ms.pick_up(2, 1, 1) is True
    assert ms.held_materials == {VOXEL_STONE: 2}


def test_pick_up_different_type_allowed():
    """Multi-type bag: picking up a different type adds it to the bag."""
    bus, grid, ms = _setup(VOXEL_STONE, loose=True)
    grid.grid[2, 1, 1] = VOXEL_DIRT
    grid.set_loose(2, 1, 1, True)

    assert ms.pick_up(1, 1, 1) is True
    assert ms.pick_up(2, 1, 1) is True
    assert ms.held_materials == {VOXEL_STONE: 1, VOXEL_DIRT: 1}


def test_drop_in_air():
    bus, grid, ms = _setup(VOXEL_STONE, loose=True)
    ms.pick_up(1, 1, 1)

    assert ms.drop(2, 2, 2) is True
    assert grid.get(2, 2, 2) == VOXEL_STONE
    assert grid.is_loose(2, 2, 2)
    assert ms.held_materials == {}


def test_drop_decrements_count():
    bus, grid, ms = _setup(VOXEL_STONE, loose=True)
    grid.grid[2, 1, 1] = VOXEL_STONE
    grid.set_loose(2, 1, 1, True)

    ms.pick_up(1, 1, 1)
    ms.pick_up(2, 1, 1)
    assert ms.held_materials == {VOXEL_STONE: 2}

    ms.drop(0, 0, 0)
    assert ms.held_materials == {VOXEL_STONE: 1}


def test_drop_nothing_held():
    bus, grid, ms = _setup()
    errors = []
    bus.subscribe("error_message", lambda **kw: errors.append(kw))

    assert ms.drop(0, 0, 0) is False
    assert len(errors) == 1


def test_drop_on_solid_rejected():
    """Dropping on a solid block is rejected (no auto-crafting on drop)."""
    bus, grid, ms = _setup(VOXEL_STONE, loose=True)
    grid.grid[2, 2, 2] = VOXEL_GRANITE  # non-air target

    ms.pick_up(1, 1, 1)

    errors = []
    bus.subscribe("error_message", lambda **kw: errors.append(kw))

    assert ms.drop(2, 2, 2) is False
    assert len(errors) == 1
    assert "solid" in errors[0]["text"].lower()


def test_left_click_move_mode():
    bus, grid, ms = _setup(VOXEL_STONE, loose=True)
    bus.publish("voxel_left_clicked", x=1, y=1, z=1, mode="move")
    assert ms.held_materials == {VOXEL_STONE: 1}


def test_right_click_move_mode():
    bus, grid, ms = _setup(VOXEL_STONE, loose=True)
    ms.pick_up(1, 1, 1)
    bus.publish("voxel_right_clicked", x=0, y=0, z=0, mode="move")
    assert grid.get(0, 0, 0) == VOXEL_STONE
    assert ms.held_materials == {}


def test_ignores_non_move_mode():
    bus, grid, ms = _setup(VOXEL_STONE, loose=True)
    bus.publish("voxel_left_clicked", x=1, y=1, z=1, mode="dig")
    assert ms.held_materials == {}  # move system doesn't handle dig mode


def test_pick_up_captures_temperature():
    """Picking up a hot block stores its temperature per-type."""
    bus, grid, ms = _setup(VOXEL_STONE, loose=True)
    grid.temperature[1, 1, 1] = 500.0

    ms.pick_up(1, 1, 1)

    assert ms.held_temperatures[VOXEL_STONE] == 500.0
    # Source cleared
    assert grid.temperature[1, 1, 1] == 0.0


def test_drop_restores_temperature():
    """Dropping a block restores its stored temperature."""
    bus, grid, ms = _setup(VOXEL_STONE, loose=True)
    grid.temperature[1, 1, 1] = 500.0

    ms.pick_up(1, 1, 1)
    ms.drop(2, 2, 2)

    assert grid.temperature[2, 2, 2] == 500.0


def test_pick_up_captures_humidity():
    """Picking up a wet block stores its humidity per-type."""
    bus, grid, ms = _setup(VOXEL_DIRT, loose=True)
    grid.humidity[1, 1, 1] = 0.8

    ms.pick_up(1, 1, 1)

    assert ms.held_humidities[VOXEL_DIRT] == pytest.approx(0.8)


def test_drop_restores_humidity():
    """Dropping a block restores its stored humidity."""
    bus, grid, ms = _setup(VOXEL_DIRT, loose=True)
    grid.humidity[1, 1, 1] = 0.8

    ms.pick_up(1, 1, 1)
    ms.drop(2, 2, 2)

    assert grid.humidity[2, 2, 2] == pytest.approx(0.8)


def test_stacking_averages_temperature():
    """Picking up multiple blocks of the same type averages their temperature."""
    bus, grid, ms = _setup(VOXEL_STONE, loose=True)
    grid.temperature[1, 1, 1] = 400.0
    grid.grid[2, 1, 1] = VOXEL_STONE
    grid.set_loose(2, 1, 1, True)
    grid.temperature[2, 1, 1] = 200.0

    ms.pick_up(1, 1, 1)
    ms.pick_up(2, 1, 1)

    # Average of 400 and 200 = 300
    assert ms.held_temperatures[VOXEL_STONE] == 300.0


def test_drop_clears_held_properties():
    """Dropping the last block of a type clears its temperature and humidity."""
    bus, grid, ms = _setup(VOXEL_STONE, loose=True)
    grid.temperature[1, 1, 1] = 500.0
    grid.humidity[1, 1, 1] = 0.6

    ms.pick_up(1, 1, 1)
    ms.drop(2, 2, 2)

    assert VOXEL_STONE not in ms.held_temperatures
    assert VOXEL_STONE not in ms.held_humidities


def test_multi_type_independent_temperature():
    """Different held types track temperature independently."""
    bus, grid, ms = _setup(VOXEL_STONE, loose=True)
    grid.temperature[1, 1, 1] = 500.0
    grid.grid[2, 1, 1] = VOXEL_DIRT
    grid.set_loose(2, 1, 1, True)
    grid.temperature[2, 1, 1] = 100.0

    ms.pick_up(1, 1, 1)
    ms.pick_up(2, 1, 1)

    assert ms.held_temperatures[VOXEL_STONE] == 500.0
    assert ms.held_temperatures[VOXEL_DIRT] == 100.0


def test_consume_helper():
    """consume() removes items and publishes material_dropped event."""
    bus, grid, ms = _setup(VOXEL_STONE, loose=True)
    ms.pick_up(1, 1, 1)

    events = []
    bus.subscribe("material_dropped", lambda **kw: events.append(kw))

    assert ms.consume(VOXEL_STONE, 1) is True
    assert ms.held_materials == {}
    assert len(events) == 1


def test_consume_insufficient():
    """consume() returns False when insufficient material."""
    bus, grid, ms = _setup(VOXEL_STONE, loose=True)
    ms.pick_up(1, 1, 1)

    assert ms.consume(VOXEL_STONE, 5) is False
    assert ms.held_materials == {VOXEL_STONE: 1}


def test_has_material_helper():
    bus, grid, ms = _setup(VOXEL_STONE, loose=True)
    assert ms.has_material(VOXEL_STONE) is False
    ms.pick_up(1, 1, 1)
    assert ms.has_material(VOXEL_STONE) is True
    assert ms.has_material(VOXEL_DIRT) is False


def test_total_count_helper():
    bus, grid, ms = _setup(VOXEL_STONE, loose=True)
    grid.grid[2, 1, 1] = VOXEL_DIRT
    grid.set_loose(2, 1, 1, True)

    ms.pick_up(1, 1, 1)
    ms.pick_up(2, 1, 1)
    assert ms.total_count() == 2


def test_craft_mode_blocks_left_click():
    """When craft mode is active, left-click in move mode is ignored by MoveSystem."""
    bus, grid, ms = _setup(VOXEL_STONE, loose=True)
    ms.game_state.craft_mode_active = True

    bus.publish("voxel_left_clicked", x=1, y=1, z=1, mode="move")
    # Should NOT have picked up
    assert ms.held_materials == {}


def test_craft_mode_blocks_right_click():
    """When craft mode is active, right-click in move mode is ignored by MoveSystem."""
    bus, grid, ms = _setup(VOXEL_STONE, loose=True)
    ms.pick_up(1, 1, 1)
    ms.game_state.craft_mode_active = True

    bus.publish("voxel_right_clicked", x=0, y=0, z=0, mode="move")
    # Should NOT have dropped
    assert grid.get(0, 0, 0) == VOXEL_AIR
    assert ms.held_materials == {VOXEL_STONE: 1}
