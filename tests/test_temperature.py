"""Tests for temperature diffusion physics."""

import numpy as np

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.world.physics.temperature import TemperaturePhysics
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_STONE,
    VOXEL_LAVA,
    VOXEL_MANA_CRYSTAL,
    LAVA_TEMPERATURE,
    MANA_CRYSTAL_TEMPERATURE,
    TEMPERATURE_TICK_INTERVAL,
)


def _make_physics(width=8, depth=8, height=8):
    bus = EventBus()
    grid = VoxelGrid(width=width, depth=depth, height=height)
    phys = TemperaturePhysics(bus, grid)
    return bus, grid, phys


def test_lava_stays_hot():
    """Lava voxels maintain their temperature after diffusion."""
    bus, grid, phys = _make_physics()
    grid.grid[4, 4, 4] = VOXEL_LAVA
    grid.temperature[4, 4, 4] = LAVA_TEMPERATURE

    for i in range(1, 100):
        bus.publish("tick", tick=i)

    assert grid.temperature[4, 4, 4] == LAVA_TEMPERATURE


def test_mana_crystal_stays_cool():
    """Mana crystals maintain their fixed temperature."""
    bus, grid, phys = _make_physics()
    grid.grid[4, 4, 4] = VOXEL_MANA_CRYSTAL
    grid.temperature[4, 4, 4] = MANA_CRYSTAL_TEMPERATURE

    # Surround with hot material
    grid.temperature[3, 4, 4] = 500.0
    grid.temperature[5, 4, 4] = 500.0

    for i in range(1, 100):
        bus.publish("tick", tick=i)

    assert grid.temperature[4, 4, 4] == MANA_CRYSTAL_TEMPERATURE


def test_heat_spreads_from_lava():
    """Heat should spread from lava to adjacent voxels."""
    bus, grid, phys = _make_physics()
    # Fill with stone
    grid.grid[:] = VOXEL_STONE
    grid.temperature[:] = 20.0

    # Place lava in center
    grid.grid[4, 4, 4] = VOXEL_LAVA
    grid.temperature[4, 4, 4] = LAVA_TEMPERATURE

    initial_neighbor = float(grid.temperature[3, 4, 4])

    # Run many ticks
    for i in range(1, 200):
        bus.publish("tick", tick=i)

    # Neighbor should be warmer than initial
    assert grid.temperature[3, 4, 4] > initial_neighbor


def test_surface_cools():
    """Surface voxels should lose heat over time."""
    bus, grid, phys = _make_physics()
    grid.temperature[:, :, 0] = 100.0

    for i in range(1, 50):
        bus.publish("tick", tick=i)

    # Surface should have cooled
    avg_surface = float(np.mean(grid.temperature[:, :, 0]))
    assert avg_surface < 100.0


def test_no_negative_temperature():
    """Temperature should never go negative."""
    bus, grid, phys = _make_physics()
    grid.temperature[:] = 0.0

    for i in range(1, 50):
        bus.publish("tick", tick=i)

    assert np.all(grid.temperature >= 0.0)


def test_only_runs_on_interval():
    """Diffusion only runs on TEMPERATURE_TICK_INTERVAL ticks."""
    bus, grid, phys = _make_physics()
    grid.grid[:] = VOXEL_STONE
    grid.temperature[:] = 20.0
    grid.grid[4, 4, 4] = VOXEL_LAVA
    grid.temperature[4, 4, 4] = LAVA_TEMPERATURE

    initial = float(grid.temperature[3, 4, 4])

    # Tick 1 should not trigger diffusion (interval=5)
    bus.publish("tick", tick=1)
    assert grid.temperature[3, 4, 4] == initial

    # Tick 5 should trigger diffusion
    bus.publish("tick", tick=TEMPERATURE_TICK_INTERVAL)
    assert grid.temperature[3, 4, 4] != initial


def test_equilibrium_converges():
    """A small grid with uniform material should converge toward uniform temp."""
    bus, grid, phys = _make_physics(width=4, depth=4, height=4)
    grid.grid[:] = VOXEL_STONE
    grid.temperature[:] = 50.0
    grid.temperature[2, 2, 2] = 200.0

    # Run for many iterations
    for i in range(1, 1000):
        bus.publish("tick", tick=i)

    # All temperatures should be closer together
    temps = grid.temperature.flatten()
    assert np.std(temps) < 50.0  # significant convergence
