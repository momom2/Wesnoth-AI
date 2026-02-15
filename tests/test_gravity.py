"""Tests for the gravity physics system."""

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.world.physics.gravity import GravityPhysics
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_STONE,
    VOXEL_BEDROCK,
    VOXEL_MANA_CRYSTAL,
    VOXEL_DIRT,
    GRAVITY_TICK_INTERVAL,
    CONNECTIVITY_TICK_INTERVAL,
)


def _setup(width=8, depth=8, height=10):
    bus = EventBus()
    grid = VoxelGrid(width=width, depth=depth, height=height)
    grav = GravityPhysics(bus, grid)
    return bus, grid, grav


def test_loose_block_falls_through_air():
    """A loose stone block with air below falls down."""
    bus, grid, grav = _setup()
    grid.grid[4, 4, 2] = VOXEL_STONE
    grid.loose[4, 4, 2] = True

    bus.publish("tick", tick=GRAVITY_TICK_INTERVAL)

    # Block should have moved down (source cleared)
    assert grid.grid[4, 4, 2] == VOXEL_AIR
    # Should be somewhere below
    found = False
    for z in range(3, 10):
        if grid.grid[4, 4, z] == VOXEL_STONE:
            found = True
            break
    assert found


def test_loose_block_stops_on_solid():
    """A loose block stops when it reaches a solid block below."""
    bus, grid, grav = _setup()
    grid.grid[4, 4, 3] = VOXEL_STONE
    grid.loose[4, 4, 3] = True
    grid.grid[4, 4, 5] = VOXEL_BEDROCK  # floor to stop on

    bus.publish("tick", tick=GRAVITY_TICK_INTERVAL)

    # Should have fallen to z=4 (one above bedrock at z=5)
    assert grid.grid[4, 4, 4] == VOXEL_STONE
    assert bool(grid.loose[4, 4, 4]) is True
    assert grid.grid[4, 4, 3] == VOXEL_AIR


def test_non_loose_block_does_not_fall():
    """A solid non-loose block with air below does not fall on gravity tick."""
    bus, grid, grav = _setup()
    grid.grid[4, 4, 3] = VOXEL_STONE

    bus.publish("tick", tick=GRAVITY_TICK_INTERVAL)

    assert grid.grid[4, 4, 3] == VOXEL_STONE
    assert grid.grid[4, 4, 4] == VOXEL_AIR


def test_multiple_loose_blocks_column():
    """Multiple loose blocks in a column fall correctly."""
    bus, grid, grav = _setup()
    grid.grid[4, 4, 2] = VOXEL_STONE
    grid.loose[4, 4, 2] = True
    grid.grid[4, 4, 3] = VOXEL_DIRT
    grid.loose[4, 4, 3] = True
    grid.grid[4, 4, 8] = VOXEL_BEDROCK  # floor

    for i in range(1, 8):
        bus.publish("tick", tick=i * GRAVITY_TICK_INTERVAL)

    # Both should have stacked above bedrock
    stone_found = False
    dirt_found = False
    for z in range(10):
        if grid.grid[4, 4, z] == VOXEL_STONE:
            stone_found = True
        if grid.grid[4, 4, z] == VOXEL_DIRT:
            dirt_found = True
    assert stone_found
    assert dirt_found


def test_disconnected_block_becomes_loose():
    """A solid block not connected to any anchor becomes loose."""
    bus, grid, grav = _setup()
    grid.grid[4, 4, 3] = VOXEL_STONE

    bus.publish("tick", tick=CONNECTIVITY_TICK_INTERVAL)

    assert bool(grid.loose[4, 4, 3]) is True


def test_connected_through_chain_stays_stable():
    """A block connected to bedrock through a chain remains non-loose."""
    bus, grid, grav = _setup()
    grid.grid[4, 4, 9] = VOXEL_BEDROCK
    grid.grid[4, 4, 8] = VOXEL_STONE
    grid.grid[4, 4, 7] = VOXEL_STONE
    grid.grid[4, 4, 6] = VOXEL_STONE

    bus.publish("tick", tick=CONNECTIVITY_TICK_INTERVAL)

    assert bool(grid.loose[4, 4, 6]) is False
    assert bool(grid.loose[4, 4, 7]) is False
    assert bool(grid.loose[4, 4, 8]) is False


def test_mana_crystal_acts_as_anchor():
    """Blocks connected to mana crystal (but not bedrock) remain stable."""
    bus, grid, grav = _setup()
    grid.grid[4, 4, 5] = VOXEL_MANA_CRYSTAL
    grid.grid[4, 4, 4] = VOXEL_STONE
    grid.grid[3, 4, 5] = VOXEL_STONE

    bus.publish("tick", tick=CONNECTIVITY_TICK_INTERVAL)

    assert bool(grid.loose[4, 4, 4]) is False
    assert bool(grid.loose[3, 4, 5]) is False


def test_removing_connection_causes_disconnect():
    """Breaking a connecting block causes disconnected mass to become loose."""
    bus, grid, grav = _setup()
    grid.grid[4, 4, 9] = VOXEL_BEDROCK
    grid.grid[4, 4, 8] = VOXEL_STONE
    grid.grid[4, 4, 7] = VOXEL_STONE
    grid.grid[4, 4, 6] = VOXEL_STONE

    bus.publish("tick", tick=CONNECTIVITY_TICK_INTERVAL)
    assert bool(grid.loose[4, 4, 6]) is False

    # Break the chain
    grid.grid[4, 4, 8] = VOXEL_AIR

    bus.publish("tick", tick=CONNECTIVITY_TICK_INTERVAL * 2)

    assert bool(grid.loose[4, 4, 6]) is True
    assert bool(grid.loose[4, 4, 7]) is True


def test_falling_carries_temperature():
    """When a block falls, its temperature moves with it."""
    bus, grid, grav = _setup()
    grid.grid[4, 4, 2] = VOXEL_STONE
    grid.loose[4, 4, 2] = True
    grid.temperature[4, 4, 2] = 500.0

    bus.publish("tick", tick=GRAVITY_TICK_INTERVAL)

    assert grid.temperature[4, 4, 2] == 0.0
    for z in range(3, 10):
        if grid.grid[4, 4, z] == VOXEL_STONE:
            assert grid.temperature[4, 4, z] == 500.0
            break


def test_falling_carries_humidity():
    """When a block falls, its humidity moves with it."""
    bus, grid, grav = _setup()
    grid.grid[4, 4, 2] = VOXEL_DIRT
    grid.loose[4, 4, 2] = True
    grid.humidity[4, 4, 2] = 0.8

    bus.publish("tick", tick=GRAVITY_TICK_INTERVAL)

    assert grid.humidity[4, 4, 2] == 0.0
    for z in range(3, 10):
        if grid.grid[4, 4, z] == VOXEL_DIRT:
            assert grid.humidity[4, 4, z] == 0.8
            break


def test_connectivity_only_runs_on_interval():
    """Connectivity check only runs on the correct tick interval."""
    bus, grid, grav = _setup()
    grid.grid[4, 4, 3] = VOXEL_STONE  # floating, should become loose

    # Use a tick that's not a multiple of CONNECTIVITY_TICK_INTERVAL
    # but is a multiple of GRAVITY_TICK_INTERVAL (so gravity runs but not connectivity)
    non_conn_tick = GRAVITY_TICK_INTERVAL
    if non_conn_tick % CONNECTIVITY_TICK_INTERVAL == 0:
        # Connectivity and gravity share the same interval, pick one that's not
        non_conn_tick = CONNECTIVITY_TICK_INTERVAL + GRAVITY_TICK_INTERVAL
        if non_conn_tick % CONNECTIVITY_TICK_INTERVAL == 0:
            non_conn_tick += GRAVITY_TICK_INTERVAL

    # With GRAVITY_TICK_INTERVAL=1 and CONNECTIVITY_TICK_INTERVAL=10,
    # tick=1 runs gravity but not connectivity
    bus.publish("tick", tick=1)

    # Non-loose block doesn't fall (gravity only affects loose), so still there
    assert grid.grid[4, 4, 3] == VOXEL_STONE
    assert bool(grid.loose[4, 4, 3]) is False

    bus.publish("tick", tick=CONNECTIVITY_TICK_INTERVAL)
    assert bool(grid.loose[4, 4, 3]) is True
