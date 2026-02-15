"""Tests for humidity diffusion physics."""

import numpy as np
import pytest

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.world.physics.humidity import HumidityPhysics
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_STONE,
    VOXEL_LAVA,
    VOXEL_SANDSTONE,
    VOXEL_OBSIDIAN,
    VOXEL_CHALK,
    LAVA_TEMPERATURE,
    HUMIDITY_TICK_INTERVAL,
)


def _make_physics(width=8, depth=8, height=8):
    bus = EventBus()
    grid = VoxelGrid(width=width, depth=depth, height=height)
    phys = HumidityPhysics(bus, grid)
    return bus, grid, phys


def test_humidity_spreads_through_porous_material():
    """Humidity should spread from wet sandstone to adjacent dry sandstone."""
    bus, grid, phys = _make_physics()
    grid.grid[:] = VOXEL_SANDSTONE  # porosity=0.35
    grid.humidity[4, 4, 4] = 1.0

    initial_neighbor = float(grid.humidity[3, 4, 4])

    for i in range(1, 200):
        bus.publish("tick", tick=i)

    assert grid.humidity[3, 4, 4] > initial_neighbor


def test_humidity_does_not_spread_through_obsidian():
    """Obsidian (porosity=0.0) should block humidity diffusion."""
    bus, grid, phys = _make_physics()
    grid.grid[:] = VOXEL_OBSIDIAN  # porosity=0.0
    grid.humidity[4, 4, 4] = 1.0

    for i in range(1, 200):
        bus.publish("tick", tick=i)

    # Humidity shouldn't have spread (obsidian is impermeable)
    assert grid.humidity[3, 4, 4] == 0.0


def test_surface_evaporation():
    """Surface blocks should lose humidity over time."""
    bus, grid, phys = _make_physics()
    grid.grid[:] = VOXEL_SANDSTONE
    grid.humidity[:, :, 0] = 0.8

    for i in range(1, 50):
        bus.publish("tick", tick=i)

    avg_surface = float(np.mean(grid.humidity[:, :, 0]))
    assert avg_surface < 0.8


def test_lava_generates_steam():
    """Blocks adjacent to lava should gain humidity (steam)."""
    bus, grid, phys = _make_physics()
    grid.grid[:] = VOXEL_SANDSTONE
    grid.grid[4, 4, 4] = VOXEL_LAVA

    for i in range(1, 10):
        bus.publish("tick", tick=i)

    # Adjacent sandstone (porous) should have gained humidity
    assert grid.humidity[3, 4, 4] > 0.0


def test_lava_steam_scales_with_porosity():
    """Porous materials absorb more steam than dense ones."""
    bus, grid, phys = _make_physics()
    grid.grid[:] = VOXEL_STONE  # porosity=0.005
    grid.grid[4, 4, 4] = VOXEL_LAVA
    grid.grid[3, 4, 4] = VOXEL_CHALK  # porosity=0.6
    grid.grid[5, 4, 4] = VOXEL_STONE  # porosity=0.005

    bus.publish("tick", tick=HUMIDITY_TICK_INTERVAL)

    # Chalk should absorb more steam than stone
    assert grid.humidity[3, 4, 4] > grid.humidity[5, 4, 4]


def test_humidity_clamped_to_valid_range():
    """Humidity should stay within [0.0, 1.0]."""
    bus, grid, phys = _make_physics()
    grid.grid[:] = VOXEL_SANDSTONE
    grid.humidity[:] = 0.99

    # Place lava to generate steam on already-wet blocks
    grid.grid[4, 4, 4] = VOXEL_LAVA

    for i in range(1, 100):
        bus.publish("tick", tick=i)

    assert np.all(grid.humidity >= 0.0)
    assert np.all(grid.humidity <= 1.0)


def test_only_runs_on_interval():
    """Humidity diffusion only runs on HUMIDITY_TICK_INTERVAL ticks."""
    bus, grid, phys = _make_physics()
    grid.grid[:] = VOXEL_SANDSTONE
    grid.humidity[4, 4, 4] = 1.0

    initial = float(grid.humidity[3, 4, 4])

    # Tick 1 should not trigger diffusion (interval=5)
    bus.publish("tick", tick=1)
    assert grid.humidity[3, 4, 4] == initial

    # Tick at interval should trigger diffusion
    bus.publish("tick", tick=HUMIDITY_TICK_INTERVAL)
    assert grid.humidity[3, 4, 4] != initial


def test_equilibrium_converges():
    """Humidity in a uniform porous grid should converge toward equilibrium."""
    bus, grid, phys = _make_physics(width=4, depth=4, height=4)
    grid.grid[:] = VOXEL_SANDSTONE
    grid.humidity[:] = 0.3
    grid.humidity[2, 2, 2] = 1.0

    for i in range(1, 2000):
        bus.publish("tick", tick=i)

    # All humidities should be closer together
    hums = grid.humidity.flatten()
    assert np.std(hums) < 0.2


def test_chalk_spreads_faster_than_stone():
    """Chalk (high porosity) should spread humidity faster than stone (low porosity)."""
    # Chalk grid
    bus1, grid1, phys1 = _make_physics()
    grid1.grid[:] = VOXEL_CHALK  # porosity=0.6
    grid1.humidity[4, 4, 4] = 1.0

    # Stone grid
    bus2, grid2, phys2 = _make_physics()
    grid2.grid[:] = VOXEL_STONE  # porosity=0.005
    grid2.humidity[4, 4, 4] = 1.0

    for i in range(1, 100):
        bus1.publish("tick", tick=i)
        bus2.publish("tick", tick=i)

    # Chalk neighbor should have more humidity than stone neighbor
    assert grid1.humidity[3, 4, 4] > grid2.humidity[3, 4, 4]


def test_convection_carries_heat_with_humidity():
    """When humidity flows from a hot block, it carries heat along."""
    bus, grid, phys = _make_physics()
    grid.grid[:] = VOXEL_SANDSTONE  # porosity=0.35
    # Hot wet block next to cool dry block
    grid.humidity[4, 4, 4] = 1.0
    grid.temperature[4, 4, 4] = 500.0
    grid.humidity[3, 4, 4] = 0.0
    grid.temperature[3, 4, 4] = 0.0

    initial_temp = float(grid.temperature[3, 4, 4])

    for i in range(1, 100):
        bus.publish("tick", tick=i)

    # Neighbor should have gained heat from convection
    assert grid.temperature[3, 4, 4] > initial_temp


def test_convection_no_heat_without_humidity_flow():
    """If humidity doesn't flow (impermeable), no convective heat transfer."""
    bus, grid, phys = _make_physics()
    grid.grid[:] = VOXEL_OBSIDIAN  # porosity=0.0
    grid.humidity[4, 4, 4] = 1.0
    grid.temperature[4, 4, 4] = 500.0
    grid.temperature[3, 4, 4] = 0.0

    for i in range(1, 100):
        bus.publish("tick", tick=i)

    # No humidity flow through obsidian => no convective heat
    assert grid.temperature[3, 4, 4] == 0.0


def test_temperature_stays_non_negative_after_convection():
    """Convection should not produce negative temperatures."""
    bus, grid, phys = _make_physics()
    grid.grid[:] = VOXEL_SANDSTONE
    grid.temperature[:] = 0.0
    grid.humidity[4, 4, 4] = 1.0

    for i in range(1, 200):
        bus.publish("tick", tick=i)

    assert np.all(grid.temperature >= 0.0)


def test_convection_conserves_heat_no_sources():
    """Total heat should be conserved when convection moves heat with humidity.

    No lava (no steam source), no surface effects on temperature, just
    humidity diffusion carrying heat between interior sandstone blocks.
    """
    bus, grid, phys = _make_physics(width=8, depth=8, height=8)
    grid.grid[:] = VOXEL_SANDSTONE
    grid.temperature[:] = 0.0
    grid.humidity[:] = 0.0

    # Hot wet block surrounded by cool dry blocks, deep interior
    grid.temperature[4, 4, 5] = 600.0
    grid.humidity[4, 4, 5] = 0.9
    grid.temperature[3, 4, 5] = 0.0
    grid.humidity[3, 4, 5] = 0.0

    initial_heat = float(np.sum(grid.temperature))

    # Run humidity diffusion (which also does convection)
    for i in range(1, 50):
        bus.publish("tick", tick=i)

    final_heat = float(np.sum(grid.temperature))

    # Heat should be conserved (no lava to add, no surface loss on temp)
    # Note: humidity system doesn't have surface heat loss, only humidity loss
    assert final_heat == pytest.approx(initial_heat, abs=1.0)


def test_humidity_conservation_no_sources():
    """Total humidity in a closed system (no lava) should not increase.

    Humidity may decrease from surface evaporation but should never
    be spontaneously created.
    """
    bus, grid, phys = _make_physics(width=6, depth=6, height=6)
    grid.grid[:] = VOXEL_SANDSTONE
    grid.humidity[:] = 0.0
    grid.humidity[3, 3, 3] = 1.0

    initial_hum = float(np.sum(grid.humidity))

    for i in range(1, 100):
        bus.publish("tick", tick=i)

    final_hum = float(np.sum(grid.humidity))

    # No sources, so total humidity should not increase
    assert final_hum <= initial_hum + 1e-3
