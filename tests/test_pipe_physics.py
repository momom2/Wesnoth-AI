"""Tests for the pipe & pump physics system."""

import pytest
import numpy as np

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.world.physics.pipe import PipePhysics
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_STONE,
    VOXEL_PIPE,
    VOXEL_PUMP,
    VOXEL_LAVA,
    METAL_IRON,
    METAL_COPPER,
    METAL_GOLD,
    PUMP_TICK_INTERVAL,
    PIPE_CONDUCTIVITY_BASE,
    METAL_CONDUCTIVITY_MULT,
    PUMP_CONVECTION_RATE,
)


def _make_grid(width=10, depth=10, height=5):
    """Create a small VoxelGrid filled with stone, air interior at z=1."""
    bus = EventBus()
    grid = VoxelGrid(width, depth, height)
    # Fill with stone so pipes have surrounding context
    grid.grid[:] = VOXEL_STONE
    # Hollow out z=1 interior for placing pipes/pumps
    grid.grid[1:-1, 1:-1, 1] = VOXEL_AIR
    return bus, grid


# ── Test 1: Network BFS finds connected pipes ──────────────────────────


def test_network_bfs_connected_pipes():
    """Place 3 adjacent pipes; they should form 1 network with 3 members."""
    bus, grid = _make_grid()
    # Place 3 pipes in a line at z=2
    grid.grid[3, 3, 2] = VOXEL_PIPE
    grid.grid[4, 3, 2] = VOXEL_PIPE
    grid.grid[5, 3, 2] = VOXEL_PIPE
    for x in (3, 4, 5):
        grid.metal_type[x, 3, 2] = METAL_IRON

    pp = PipePhysics(bus, grid)
    pp._update()

    assert pp._networks is not None
    assert len(pp._networks) == 1
    pipes, pumps = pp._networks[0]
    assert len(pipes) == 3
    assert len(pumps) == 0


# ── Test 2: Disconnected pipes form separate networks ──────────────────


def test_disconnected_pipes_separate_networks():
    """Two groups of pipes separated by stone form 2 networks."""
    bus, grid = _make_grid()
    # Group A: two pipes at z=2
    grid.grid[1, 1, 2] = VOXEL_PIPE
    grid.grid[2, 1, 2] = VOXEL_PIPE
    grid.metal_type[1, 1, 2] = METAL_IRON
    grid.metal_type[2, 1, 2] = METAL_IRON
    # Group B: one pipe far away at z=2
    grid.grid[8, 8, 2] = VOXEL_PIPE
    grid.metal_type[8, 8, 2] = METAL_COPPER

    pp = PipePhysics(bus, grid)
    pp._update()

    assert pp._networks is not None
    assert len(pp._networks) == 2
    sizes = sorted(len(n[0]) for n in pp._networks)
    assert sizes == [1, 2]


# ── Test 3: Pump included in network ───────────────────────────────────


def test_pump_included_in_network():
    """A pipe adjacent to a pump should share the same network, with the pump listed."""
    bus, grid = _make_grid()
    grid.grid[4, 4, 2] = VOXEL_PIPE
    grid.grid[5, 4, 2] = VOXEL_PUMP
    grid.metal_type[4, 4, 2] = METAL_IRON
    grid.metal_type[5, 4, 2] = METAL_IRON

    pp = PipePhysics(bus, grid)
    pp._update()

    assert len(pp._networks) == 1
    pipes, pumps = pp._networks[0]
    # Pumps are also in the pipes list (they conduct too)
    assert len(pipes) == 2
    assert len(pumps) == 1
    assert (5, 4, 2) in pumps


# ── Test 4: Passive heat averaging ─────────────────────────────────────


def test_passive_heat_averaging():
    """3 iron pipes with temps [100, 0, 0] equalize toward ~33 after update."""
    bus, grid = _make_grid()
    grid.grid[3, 3, 2] = VOXEL_PIPE
    grid.grid[4, 3, 2] = VOXEL_PIPE
    grid.grid[5, 3, 2] = VOXEL_PIPE
    for x in (3, 4, 5):
        grid.metal_type[x, 3, 2] = METAL_IRON
    grid.temperature[3, 3, 2] = 100.0
    grid.temperature[4, 3, 2] = 0.0
    grid.temperature[5, 3, 2] = 0.0

    pp = PipePhysics(bus, grid)
    pp._update()

    # All pipes have equal conductivity (iron), so weighted average = 100/3 ~ 33.33
    # Blend factor = min(PIPE_CONDUCTIVITY_BASE * METAL_CONDUCTIVITY_MULT[IRON], 1.0)
    # = min(0.5 * 0.8, 1.0) = 0.4
    # After one update each pipe blends toward average by 0.4
    avg = 100.0 / 3.0
    blend = PIPE_CONDUCTIVITY_BASE * METAL_CONDUCTIVITY_MULT[METAL_IRON]  # 0.4
    expected_hot = 100.0 + (avg - 100.0) * blend
    expected_cold = 0.0 + (avg - 0.0) * blend

    assert float(grid.temperature[3, 3, 2]) == pytest.approx(expected_hot, abs=0.01)
    assert float(grid.temperature[4, 3, 2]) == pytest.approx(expected_cold, abs=0.01)
    assert float(grid.temperature[5, 3, 2]) == pytest.approx(expected_cold, abs=0.01)


# ── Test 5: Passive humidity averaging ─────────────────────────────────


def test_passive_humidity_averaging():
    """Similar to heat averaging but for humidity."""
    bus, grid = _make_grid()
    grid.grid[3, 3, 2] = VOXEL_PIPE
    grid.grid[4, 3, 2] = VOXEL_PIPE
    for x in (3, 4):
        grid.metal_type[x, 3, 2] = METAL_COPPER
    grid.humidity[3, 3, 2] = 0.8
    grid.humidity[4, 3, 2] = 0.0

    pp = PipePhysics(bus, grid)
    pp._update()

    # Copper conductivity = 0.5 * 1.0 = 0.5
    avg = 0.4
    blend = PIPE_CONDUCTIVITY_BASE * METAL_CONDUCTIVITY_MULT[METAL_COPPER]  # 0.5
    expected_high = 0.8 + (avg - 0.8) * blend
    expected_low = 0.0 + (avg - 0.0) * blend

    assert float(grid.humidity[3, 3, 2]) == pytest.approx(expected_high, abs=0.001)
    assert float(grid.humidity[4, 3, 2]) == pytest.approx(expected_low, abs=0.001)


# ── Test 6: Copper pipe conducts better than iron ──────────────────────


def test_copper_conducts_better_than_iron():
    """Copper pipes (mult=1.0) equalize faster than iron (mult=0.8)."""
    bus_cu, grid_cu = _make_grid()
    bus_fe, grid_fe = _make_grid()

    for bus, grid, metal in [(bus_cu, grid_cu, METAL_COPPER), (bus_fe, grid_fe, METAL_IRON)]:
        grid.grid[3, 3, 2] = VOXEL_PIPE
        grid.grid[4, 3, 2] = VOXEL_PIPE
        for x in (3, 4):
            grid.metal_type[x, 3, 2] = metal
        grid.temperature[3, 3, 2] = 100.0
        grid.temperature[4, 3, 2] = 0.0

    pp_cu = PipePhysics(bus_cu, grid_cu)
    pp_fe = PipePhysics(bus_fe, grid_fe)
    pp_cu._update()
    pp_fe._update()

    # Copper should have equalized more (closer to 50.0 average)
    cu_diff = abs(float(grid_cu.temperature[3, 3, 2]) - float(grid_cu.temperature[4, 3, 2]))
    fe_diff = abs(float(grid_fe.temperature[3, 3, 2]) - float(grid_fe.temperature[4, 3, 2]))
    assert cu_diff < fe_diff


# ── Test 7: Pump direction — block_state=0 means +X, intake from -X ───


def test_pump_direction():
    """Pump with block_state=0 (+X direction) pulls from the -X side."""
    bus, grid = _make_grid()
    # Pump at (5,3,2) pointing +X → intake at (4,3,2)
    grid.grid[5, 3, 2] = VOXEL_PUMP
    grid.metal_type[5, 3, 2] = METAL_IRON
    grid.block_state[5, 3, 2] = 0  # +X direction

    # Hot source at intake position (4,3,2) — this is NOT a pipe, just a hot cell
    grid.temperature[4, 3, 2] = 200.0

    pp = PipePhysics(bus, grid)
    pp._update()

    # Pump should have pulled heat from (4,3,2)
    assert float(grid.temperature[4, 3, 2]) < 200.0


# ── Test 8: Pump active transfer distributes heat to network ───────────


def test_pump_active_transfer():
    """Pump pulls from intake and distributes heat to connected pipes."""
    bus, grid = _make_grid()
    # Pipe network: pipe at (6,3,2), pump at (5,3,2) pointing +X
    grid.grid[5, 3, 2] = VOXEL_PUMP
    grid.grid[6, 3, 2] = VOXEL_PIPE
    grid.metal_type[5, 3, 2] = METAL_IRON
    grid.metal_type[6, 3, 2] = METAL_IRON
    grid.block_state[5, 3, 2] = 0  # +X direction, intake from -X = (4,3,2)

    # Start with zero temp in pipes, hot source at intake
    grid.temperature[5, 3, 2] = 0.0
    grid.temperature[6, 3, 2] = 0.0
    grid.temperature[4, 3, 2] = 500.0  # intake source (stone block)

    pp = PipePhysics(bus, grid)
    pp._update()

    # Source should have lost heat
    assert float(grid.temperature[4, 3, 2]) < 500.0
    # Network pipes should have gained heat (from pumping + passive averaging)
    total_pipe_heat = float(grid.temperature[5, 3, 2]) + float(grid.temperature[6, 3, 2])
    assert total_pipe_heat > 0.0


# ── Test 9: Pump no source — no transfer when intake is cold ───────────


def test_pump_no_source():
    """No heat at pump intake means no transfer occurs."""
    bus, grid = _make_grid()
    grid.grid[5, 3, 2] = VOXEL_PUMP
    grid.grid[6, 3, 2] = VOXEL_PIPE
    grid.metal_type[5, 3, 2] = METAL_IRON
    grid.metal_type[6, 3, 2] = METAL_IRON
    grid.block_state[5, 3, 2] = 0  # +X, intake from (4,3,2)

    # Everything at zero temperature and humidity
    grid.temperature[4, 3, 2] = 0.0
    grid.temperature[5, 3, 2] = 0.0
    grid.temperature[6, 3, 2] = 0.0
    grid.humidity[4, 3, 2] = 0.0

    pp = PipePhysics(bus, grid)
    pp._update()

    # Nothing should have changed
    assert float(grid.temperature[5, 3, 2]) == pytest.approx(0.0)
    assert float(grid.temperature[6, 3, 2]) == pytest.approx(0.0)


# ── Test 10: Cache invalidation on voxel change ───────────────────────


def test_cache_invalidation():
    """Adding a pipe after initial update rebuilds the network to include it."""
    bus, grid = _make_grid()
    grid.grid[3, 3, 2] = VOXEL_PIPE
    grid.metal_type[3, 3, 2] = METAL_IRON

    pp = PipePhysics(bus, grid)
    pp._update()

    assert len(pp._networks) == 1
    assert len(pp._networks[0][0]) == 1  # 1 pipe

    # Add a second adjacent pipe and signal voxel_changed
    grid.grid[4, 3, 2] = VOXEL_PIPE
    grid.metal_type[4, 3, 2] = METAL_IRON
    bus.publish("voxel_changed", x=4, y=3, z=2, old_type=VOXEL_STONE, new_type=VOXEL_PIPE)

    pp._update()

    assert len(pp._networks) == 1
    assert len(pp._networks[0][0]) == 2  # now 2 pipes


# ── Test 11: Empty grid no error ──────────────────────────────────────


def test_empty_grid_no_error():
    """An update with no pipes/pumps does nothing and does not crash."""
    bus, grid = _make_grid()
    # grid is all stone and air — no pipes or pumps

    pp = PipePhysics(bus, grid)
    pp._update()  # should not raise

    assert pp._networks is not None
    assert len(pp._networks) == 0


# ── Test 12: Single pipe network ──────────────────────────────────────


def test_single_pipe_network():
    """A single pipe forms a network of 1."""
    bus, grid = _make_grid()
    grid.grid[5, 5, 2] = VOXEL_PIPE
    grid.metal_type[5, 5, 2] = METAL_GOLD

    pp = PipePhysics(bus, grid)
    pp._update()

    assert len(pp._networks) == 1
    pipes, pumps = pp._networks[0]
    assert len(pipes) == 1
    assert len(pumps) == 0


# ── Test 13: Pump without pipes ───────────────────────────────────────


def test_pump_without_pipes():
    """A pump alone forms a network of 1 (pump also conducts)."""
    bus, grid = _make_grid()
    grid.grid[5, 5, 2] = VOXEL_PUMP
    grid.metal_type[5, 5, 2] = METAL_IRON
    grid.block_state[5, 5, 2] = 0

    pp = PipePhysics(bus, grid)
    pp._update()

    assert len(pp._networks) == 1
    pipes, pumps = pp._networks[0]
    # Pump is in both lists: conducts (pipes list) and pumps
    assert len(pipes) == 1
    assert len(pumps) == 1


# ── Test 14: Multiple pumps in network ────────────────────────────────


def test_multiple_pumps_in_network():
    """Two pumps connected by a pipe both pump into the same network."""
    bus, grid = _make_grid()
    # Pump A at (3,3,2) pointing +X, intake from (2,3,2)
    grid.grid[3, 3, 2] = VOXEL_PUMP
    grid.metal_type[3, 3, 2] = METAL_IRON
    grid.block_state[3, 3, 2] = 0  # +X
    # Pipe at (4,3,2)
    grid.grid[4, 3, 2] = VOXEL_PIPE
    grid.metal_type[4, 3, 2] = METAL_IRON
    # Pump B at (5,3,2) pointing -X, intake from (6,3,2)
    grid.grid[5, 3, 2] = VOXEL_PUMP
    grid.metal_type[5, 3, 2] = METAL_IRON
    grid.block_state[5, 3, 2] = 1  # -X

    # Heat sources at both intakes
    grid.temperature[2, 3, 2] = 300.0  # Pump A intake
    grid.temperature[6, 3, 2] = 200.0  # Pump B intake

    pp = PipePhysics(bus, grid)
    pp._update()

    assert len(pp._networks) == 1
    pipes, pumps = pp._networks[0]
    assert len(pumps) == 2

    # Both sources should have lost heat
    assert float(grid.temperature[2, 3, 2]) < 300.0
    assert float(grid.temperature[6, 3, 2]) < 200.0

    # Network should have gained heat
    total = sum(float(grid.temperature[x, 3, 2]) for x in (3, 4, 5))
    assert total > 0.0


# ── Test 15: Lava adjacent to pipe heats it through passive conduction ─


def test_lava_adjacent_heats_pipe_network():
    """A pipe adjacent to a high-temperature cell spreads heat through the network."""
    bus, grid = _make_grid()
    # Two pipes in a line
    grid.grid[3, 3, 2] = VOXEL_PIPE
    grid.grid[4, 3, 2] = VOXEL_PIPE
    grid.metal_type[3, 3, 2] = METAL_COPPER
    grid.metal_type[4, 3, 2] = METAL_COPPER

    # Set the first pipe's temperature high (as if lava neighbor heated it)
    grid.temperature[3, 3, 2] = 500.0
    grid.temperature[4, 3, 2] = 0.0

    pp = PipePhysics(bus, grid)
    pp._update()

    # After passive conduction, pipe at (4,3,2) should have gained some heat
    assert float(grid.temperature[4, 3, 2]) > 0.0
    # And (3,3,2) should have lost some heat toward the average
    assert float(grid.temperature[3, 3, 2]) < 500.0
