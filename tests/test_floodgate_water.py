"""Tests for floodgate and door interaction with water physics.

Covers:
- Closed floodgate blocking water flow (downward and lateral)
- Open floodgate allowing water flow (downward and lateral)
- Pressure burst forcing closed floodgates/doors open
- Metal variant affecting burst threshold (gold vs iron)
- Event publication on gate burst
- Edge cases: shallow water, already-open gates
"""

import pytest
import numpy as np

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.world.physics.water import WaterPhysics
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_STONE,
    VOXEL_WATER,
    VOXEL_FLOODGATE,
    VOXEL_DOOR,
    WATER_TICK_INTERVAL,
    WATER_BURST_FACTOR,
    WATER_PRESSURE_WEIGHT,
    VOXEL_WEIGHT,
    VOXEL_SHEAR_STRENGTH,
    METAL_IRON,
    METAL_GOLD,
    METAL_STRENGTH_MULT,
    METAL_NONE,
)


@pytest.fixture
def setup():
    """Create a small 8x8x8 stone box with hollow interior."""
    eb = EventBus()
    grid = VoxelGrid(8, 8, 8)
    # Fill everything with stone
    grid.grid[:, :, :] = VOXEL_STONE
    # Hollow out interior (1..6 in each axis)
    grid.grid[1:7, 1:7, 1:7] = VOXEL_AIR
    wp = WaterPhysics(eb, grid)
    return wp, grid, eb


# ------------------------------------------------------------------
# 1. Closed floodgate blocks downward flow
# ------------------------------------------------------------------


def test_closed_floodgate_blocks_downward_flow():
    """Water above a closed floodgate does not flow down through it."""
    eb = EventBus()
    # Use a tight 1-wide shaft so water cannot spread laterally around the gate
    grid = VoxelGrid(5, 5, 6)
    grid.grid[:, :, :] = VOXEL_STONE
    # Carve a single-cell shaft at (2, 2)
    grid.grid[2, 2, 1] = VOXEL_AIR  # above water
    grid.grid[2, 2, 2] = VOXEL_AIR  # water here
    # z=3: floodgate
    # z=4: air below gate

    # Place water at (2, 2, 2)
    grid.grid[2, 2, 2] = VOXEL_WATER
    grid.water_level[2, 2, 2] = 255

    # Place closed floodgate directly below at (2, 2, 3)
    grid.grid[2, 2, 3] = VOXEL_FLOODGATE
    grid.block_state[2, 2, 3] = 1  # closed

    # Air below the floodgate at (2, 2, 4)
    grid.grid[2, 2, 4] = VOXEL_AIR

    wp = WaterPhysics(eb, grid)
    wp._flow()

    # Water should still be at z=2 -- the closed gate blocks downward flow
    assert grid.water_level[2, 2, 2] > 0
    # No water should have appeared below the gate
    assert grid.water_level[2, 2, 4] == 0
    # The gate cell itself should have no water
    assert grid.water_level[2, 2, 3] == 0


# ------------------------------------------------------------------
# 2. Open floodgate allows downward flow
# ------------------------------------------------------------------


def test_open_floodgate_allows_downward_flow():
    """Water above an open floodgate falls through it."""
    eb = EventBus()
    # Use a tight 1-wide shaft so we can track exactly where water goes
    grid = VoxelGrid(5, 5, 7)
    grid.grid[:, :, :] = VOXEL_STONE
    # Carve single-cell shaft at (2, 2): z=1 (water source), z=2 (gate), z=3..5 (air below)
    grid.grid[2, 2, 1] = VOXEL_WATER
    grid.water_level[2, 2, 1] = 255

    # Open floodgate at z=2
    grid.grid[2, 2, 2] = VOXEL_FLOODGATE
    grid.block_state[2, 2, 2] = 0  # open

    # Air below gate
    grid.grid[2, 2, 3] = VOXEL_AIR
    grid.grid[2, 2, 4] = VOXEL_AIR
    grid.grid[2, 2, 5] = VOXEL_AIR

    wp = WaterPhysics(eb, grid)
    wp._flow()

    # Water should have fallen through the open floodgate.
    # Check that water arrived somewhere below the gate (z=2..5).
    total_below = (
        int(grid.water_level[2, 2, 2])
        + int(grid.water_level[2, 2, 3])
        + int(grid.water_level[2, 2, 4])
        + int(grid.water_level[2, 2, 5])
    )
    assert total_below > 0, (
        "Water should flow through the open floodgate into cells below"
    )


# ------------------------------------------------------------------
# 3. Closed floodgate blocks lateral flow
# ------------------------------------------------------------------


def test_closed_floodgate_blocks_lateral_flow(setup):
    """Water next to a closed floodgate does not flow through laterally."""
    wp, grid, eb = setup

    # Place water at (3, 3, 3)
    grid.grid[3, 3, 3] = VOXEL_WATER
    grid.water_level[3, 3, 3] = 200

    # Place a floor below both cells so water does not fall
    grid.grid[3, 3, 4] = VOXEL_STONE
    grid.grid[4, 3, 4] = VOXEL_STONE

    # Place closed floodgate to the +x side at (4, 3, 3)
    grid.grid[4, 3, 3] = VOXEL_FLOODGATE
    grid.block_state[4, 3, 3] = 1  # closed

    # Air beyond the gate at (5, 3, 3)
    grid.grid[5, 3, 3] = VOXEL_AIR
    grid.grid[5, 3, 4] = VOXEL_STONE  # floor

    wp._flow()

    # The closed floodgate should not let water level through laterally.
    # The floodgate cell itself is not air or water or open gate, so lateral
    # flow into it is blocked.  The cell beyond should remain dry.
    assert grid.water_level[5, 3, 3] == 0


# ------------------------------------------------------------------
# 4. Open floodgate allows lateral flow
# ------------------------------------------------------------------


def test_open_floodgate_allows_lateral_flow(setup):
    """Water next to an open floodgate flows through laterally."""
    wp, grid, eb = setup

    # Place water at (3, 3, 3) with a floor below
    grid.grid[3, 3, 3] = VOXEL_WATER
    grid.water_level[3, 3, 3] = 200
    grid.grid[3, 3, 4] = VOXEL_STONE  # floor

    # Open floodgate adjacent at (4, 3, 3)
    grid.grid[4, 3, 3] = VOXEL_FLOODGATE
    grid.block_state[4, 3, 3] = 0  # open
    grid.grid[4, 3, 4] = VOXEL_STONE  # floor

    # Run flow multiple iterations so leveling has time
    for _ in range(5):
        wp._flow()

    # The open gate should have received some water
    assert grid.water_level[4, 3, 3] > 0


# ------------------------------------------------------------------
# 5. Pressure burst opens a closed floodgate
# ------------------------------------------------------------------


def _make_water_column(grid, x, y, z_start, depth):
    """Place a water column from z_start to z_start+depth-1."""
    for dz in range(depth):
        z = z_start + dz
        if grid.in_bounds(x, y, z):
            grid.grid[x, y, z] = VOXEL_WATER
            grid.water_level[x, y, z] = 255


def test_pressure_burst_opens_floodgate():
    """Deep water column creates enough pressure to force a closed floodgate open."""
    eb = EventBus()
    # Tall grid to accommodate a deep water column
    grid = VoxelGrid(5, 5, 320)
    grid.grid[:, :, :] = VOXEL_STONE

    # Hollow out a 1-wide vertical shaft at (2, 2)
    for z in range(320):
        grid.grid[2, 2, z] = VOXEL_AIR

    # Place closed iron floodgate at bottom of shaft
    gate_z = 310
    grid.grid[2, 2, gate_z] = VOXEL_FLOODGATE
    grid.block_state[2, 2, gate_z] = 1  # closed
    grid.metal_type[2, 2, gate_z] = METAL_IRON

    # Fill water column above the gate
    # Burst threshold for iron floodgate (3× buff):
    #   effective_shear = 180.0 * 1.0 = 180.0
    #   threshold_pressure = 180.0 * 1.5 = 270.0
    #   pressure = depth * 3.0 * 0.3 = depth * 0.9
    #   Need depth > 300
    _make_water_column(grid, 2, 2, 0, gate_z)  # 310 blocks of water

    wp = WaterPhysics(eb, grid)
    wp._apply_pressure()

    # The gate should have been forced open
    assert grid.block_state[2, 2, gate_z] == 0, (
        "Iron floodgate should be forced open by deep water pressure"
    )


# ------------------------------------------------------------------
# 6. Pressure burst opens a closed door
# ------------------------------------------------------------------


def test_pressure_burst_opens_door():
    """Deep water column creates enough pressure to force a closed door open."""
    eb = EventBus()
    grid = VoxelGrid(5, 5, 150)
    grid.grid[:, :, :] = VOXEL_STONE

    # Hollow out vertical shaft
    for z in range(150):
        grid.grid[2, 2, z] = VOXEL_AIR

    # Place closed door at z=140
    door_z = 140
    grid.grid[2, 2, door_z] = VOXEL_DOOR
    grid.block_state[2, 2, door_z] = 1  # closed
    grid.metal_type[2, 2, door_z] = METAL_IRON

    # Door burst threshold (3× buff):
    #   effective_shear = 75.0 * 1.0 = 75.0
    #   threshold_pressure = 75.0 * 1.5 = 112.5
    #   pressure = depth * 0.9
    #   Need depth > 125
    _make_water_column(grid, 2, 2, 0, door_z)  # 140 blocks of water

    wp = WaterPhysics(eb, grid)
    wp._apply_pressure()

    assert grid.block_state[2, 2, door_z] == 0, (
        "Iron door should be forced open by deep water pressure"
    )


# ------------------------------------------------------------------
# 7. Gold floodgate bursts easier than iron
# ------------------------------------------------------------------


def test_gold_floodgate_bursts_easier():
    """Gold floodgate (strength mult 0.4) bursts with less water than iron."""
    eb = EventBus()
    # Gold burst threshold (3× buff):
    #   effective_shear = 180.0 * 0.4 = 72.0
    #   threshold_pressure = 72.0 * 1.5 = 108.0
    #   pressure = depth * 0.9
    #   Need depth > 120
    grid = VoxelGrid(5, 5, 140)
    grid.grid[:, :, :] = VOXEL_STONE

    for z in range(140):
        grid.grid[2, 2, z] = VOXEL_AIR

    gate_z = 130
    grid.grid[2, 2, gate_z] = VOXEL_FLOODGATE
    grid.block_state[2, 2, gate_z] = 1
    grid.metal_type[2, 2, gate_z] = METAL_GOLD

    # 130 blocks of water -> pressure = 130 * 0.9 = 117.0 > 108.0 threshold
    _make_water_column(grid, 2, 2, 0, gate_z)

    wp = WaterPhysics(eb, grid)
    wp._apply_pressure()

    assert grid.block_state[2, 2, gate_z] == 0, (
        "Gold floodgate should burst with 130-deep water column"
    )


# ------------------------------------------------------------------
# 8. Iron floodgate resists the same column that bursts gold
# ------------------------------------------------------------------


def test_iron_floodgate_resists_same_column_that_bursts_gold():
    """Iron floodgate survives a water column that would burst a gold gate."""
    eb = EventBus()
    # Iron burst threshold (3× buff):
    #   effective_shear = 180.0 * 1.0 = 180.0
    #   threshold_pressure = 270.0
    #   We use 130-deep column => pressure = 117.0 < 270.0
    grid = VoxelGrid(5, 5, 140)
    grid.grid[:, :, :] = VOXEL_STONE

    for z in range(140):
        grid.grid[2, 2, z] = VOXEL_AIR

    gate_z = 130
    grid.grid[2, 2, gate_z] = VOXEL_FLOODGATE
    grid.block_state[2, 2, gate_z] = 1
    grid.metal_type[2, 2, gate_z] = METAL_IRON

    _make_water_column(grid, 2, 2, 0, gate_z)

    wp = WaterPhysics(eb, grid)
    wp._apply_pressure()

    assert grid.block_state[2, 2, gate_z] == 1, (
        "Iron floodgate should survive 130-deep water column (pressure 117 < threshold 270)"
    )


# ------------------------------------------------------------------
# 9. Enchanted metal uses same base metal strength for pressure burst
# ------------------------------------------------------------------


def test_enchanted_gate_uses_base_metal_strength():
    """Enchanted bit affects melt immunity, not strength — burst threshold same as base."""
    from dungeon_builder.config import ENCHANTED_OFFSET

    eb = EventBus()
    grid = VoxelGrid(5, 5, 140)
    grid.grid[:, :, :] = VOXEL_STONE

    for z in range(140):
        grid.grid[2, 2, z] = VOXEL_AIR

    gate_z = 130
    grid.grid[2, 2, gate_z] = VOXEL_FLOODGATE
    grid.block_state[2, 2, gate_z] = 1
    # Enchanted gold: base_metal = METAL_GOLD (0x03), strength_mult = 0.4
    grid.metal_type[2, 2, gate_z] = METAL_GOLD | ENCHANTED_OFFSET

    # Same 130-deep column that bursts regular gold should also burst enchanted gold
    _make_water_column(grid, 2, 2, 0, gate_z)

    wp = WaterPhysics(eb, grid)
    wp._apply_pressure()

    assert grid.block_state[2, 2, gate_z] == 0, (
        "Enchanted gold floodgate should burst like regular gold (enchanted only affects melt)"
    )


# ------------------------------------------------------------------
# 10. Event published on gate burst
# ------------------------------------------------------------------


def test_gate_burst_publishes_event():
    """A 'gate_pressure_burst' event is published when a gate is forced open."""
    eb = EventBus()
    grid = VoxelGrid(5, 5, 320)
    grid.grid[:, :, :] = VOXEL_STONE

    for z in range(320):
        grid.grid[2, 2, z] = VOXEL_AIR

    gate_z = 310
    grid.grid[2, 2, gate_z] = VOXEL_FLOODGATE
    grid.block_state[2, 2, gate_z] = 1
    grid.metal_type[2, 2, gate_z] = METAL_IRON

    _make_water_column(grid, 2, 2, 0, gate_z)

    events = []
    eb.subscribe("gate_pressure_burst", lambda **kw: events.append(kw))

    wp = WaterPhysics(eb, grid)
    wp._apply_pressure()

    assert len(events) == 1, "Expected exactly one gate_pressure_burst event"
    assert events[0]["x"] == 2
    assert events[0]["y"] == 2
    assert events[0]["z"] == gate_z
    assert events[0]["vtype"] == VOXEL_FLOODGATE


# ------------------------------------------------------------------
# 11. Shallow water does not burst gate
# ------------------------------------------------------------------


def test_shallow_water_does_not_burst_gate():
    """A small water column should not generate enough pressure to burst a gate."""
    eb = EventBus()
    grid = VoxelGrid(8, 8, 8)
    grid.grid[:, :, :] = VOXEL_STONE
    grid.grid[1:7, 1:7, 1:7] = VOXEL_AIR

    # Place closed floodgate at (3, 3, 4)
    grid.grid[3, 3, 4] = VOXEL_FLOODGATE
    grid.block_state[3, 3, 4] = 1
    grid.metal_type[3, 3, 4] = METAL_IRON

    # Place only 3 blocks of water above (z=1,2,3)
    # pressure = 3 * 0.9 = 2.7, threshold = 60 * 1.0 * 1.5 = 90
    _make_water_column(grid, 3, 3, 1, 3)

    wp = WaterPhysics(eb, grid)
    wp._apply_pressure()

    assert grid.block_state[3, 3, 4] == 1, (
        "Shallow water (3 blocks) should not burst iron floodgate"
    )


# ------------------------------------------------------------------
# 12. Already-open gate is unaffected by burst logic
# ------------------------------------------------------------------


def test_already_open_gate_unaffected_by_burst():
    """Burst logic only targets closed gates (block_state != 0)."""
    eb = EventBus()
    grid = VoxelGrid(5, 5, 120)
    grid.grid[:, :, :] = VOXEL_STONE

    for z in range(120):
        grid.grid[2, 2, z] = VOXEL_AIR

    gate_z = 110
    grid.grid[2, 2, gate_z] = VOXEL_FLOODGATE
    grid.block_state[2, 2, gate_z] = 0  # already open
    grid.metal_type[2, 2, gate_z] = METAL_IRON

    _make_water_column(grid, 2, 2, 0, gate_z)

    events = []
    eb.subscribe("gate_pressure_burst", lambda **kw: events.append(kw))

    wp = WaterPhysics(eb, grid)
    wp._apply_pressure()

    # No burst event should fire -- the gate is already open
    assert len(events) == 0, "No burst event expected for already-open gate"
    # block_state should still be 0
    assert grid.block_state[2, 2, gate_z] == 0
