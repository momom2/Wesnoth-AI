"""Tests for water physics: flow, pressure, lava interaction, seepage."""

import numpy as np
import pytest

from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_WATER,
    VOXEL_STONE,
    VOXEL_DIRT,
    VOXEL_LAVA,
    VOXEL_OBSIDIAN,
    VOXEL_CHALK,
    VOXEL_SANDSTONE,
    VOXEL_GRANITE,
    VOXEL_BEDROCK,
    NON_DIGGABLE,
    WATER_TICK_INTERVAL,
    WATER_FLOW_RATE,
    WATER_PRESSURE_WEIGHT,
    WATER_BURST_FACTOR,
    WATER_LAVA_PRODUCT,
    WATER_HUMIDITY_SOURCE,
    WATER_EVAPORATION_RATE,
    WATER_SEEP_RATE,
    VOXEL_POROSITY,
    VOXEL_SHEAR_STRENGTH,
    VOXEL_WEIGHT,
)
from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.world.physics.water import WaterPhysics


def _make(w=10, d=10, h=10):
    """Create event bus, grid, and water physics."""
    eb = EventBus()
    grid = VoxelGrid(w, d, h)
    wp = WaterPhysics(eb, grid)
    return eb, grid, wp


def _tick(eb, tick_num):
    """Publish a tick event."""
    eb.publish("tick", tick=tick_num)


# ──────────────────────────────────────────────────────────────────────
# Downward flow
# ──────────────────────────────────────────────────────────────────────


class TestDownwardFlow:
    def test_water_falls_through_air(self):
        """Water block above air should fall downward."""
        eb, grid, wp = _make()
        # Place water at z=3, floor at z=5, walls so it doesn't spread
        grid.grid[5, 5, 3] = VOXEL_WATER
        grid.water_level[5, 5, 3] = 200
        grid.grid[5, 5, 5] = VOXEL_STONE  # floor
        # Contain laterally at z=4 so water doesn't spread
        for z in [3, 4]:
            grid.grid[4, 5, z] = VOXEL_STONE
            grid.grid[6, 5, z] = VOXEL_STONE
            grid.grid[5, 4, z] = VOXEL_STONE
            grid.grid[5, 6, z] = VOXEL_STONE

        # Tick at correct interval
        _tick(eb, WATER_TICK_INTERVAL)

        # Water should have moved down to z=4 (above the stone floor)
        assert grid.grid[5, 5, 3] == VOXEL_AIR
        assert grid.water_level[5, 5, 3] == 0
        assert grid.grid[5, 5, 4] == VOXEL_WATER
        assert grid.water_level[5, 5, 4] == 200

    def test_water_stops_on_solid(self):
        """Water should rest on top of a solid block."""
        eb, grid, wp = _make()
        # Place stone floor at z=5
        grid.grid[5, 5, 5] = VOXEL_STONE
        # Place water above it
        grid.grid[5, 5, 4] = VOXEL_WATER
        grid.water_level[5, 5, 4] = 200

        _tick(eb, WATER_TICK_INTERVAL)

        # Water should stay at z=4 (stone below at z=5 blocks falling)
        assert grid.grid[5, 5, 4] == VOXEL_WATER
        assert grid.water_level[5, 5, 4] > 0


# ──────────────────────────────────────────────────────────────────────
# Lateral leveling
# ──────────────────────────────────────────────────────────────────────


class TestLateralLeveling:
    def test_water_levels_laterally(self):
        """Water at different levels should equalize."""
        eb, grid, wp = _make()
        # Stone floor along z=5
        for x in range(3, 8):
            grid.grid[x, 5, 5] = VOXEL_STONE

        # High water at x=4, low water at x=5
        grid.grid[4, 5, 4] = VOXEL_WATER
        grid.water_level[4, 5, 4] = 200
        grid.grid[5, 5, 4] = VOXEL_WATER
        grid.water_level[5, 5, 4] = 50

        _tick(eb, WATER_TICK_INTERVAL)

        # Water should have transferred from high to low
        lvl_a = grid.water_level[4, 5, 4]
        lvl_b = grid.water_level[5, 5, 4]
        # Difference should be smaller now
        assert abs(int(lvl_a) - int(lvl_b)) < 150  # was 150

    def test_water_does_not_flow_through_solid(self):
        """Water should not pass through solid walls."""
        eb, grid, wp = _make()
        # Stone wall at x=5, water on left side
        for z in range(3, 7):
            grid.grid[5, 5, z] = VOXEL_STONE
        # Water at x=4
        grid.grid[4, 5, 4] = VOXEL_WATER
        grid.water_level[4, 5, 4] = 200
        # Floor
        grid.grid[4, 5, 5] = VOXEL_STONE

        _tick(eb, WATER_TICK_INTERVAL)

        # Water should not appear at x=6 (other side of wall)
        assert grid.grid[6, 5, 4] != VOXEL_WATER


# ──────────────────────────────────────────────────────────────────────
# Pressure
# ──────────────────────────────────────────────────────────────────────


class TestPressure:
    def test_water_pressure_increases_with_depth(self):
        """Deeper water should exert more pressure on adjacent walls."""
        eb, grid, wp = _make()
        # Create a contained water column: stone on all sides + floor
        # Water at x=5, stone walls at x=4, x=6, y=4, y=6, floor at z=5
        for z in range(6):
            grid.grid[4, 5, z] = VOXEL_STONE
            grid.grid[6, 5, z] = VOXEL_STONE
            grid.grid[5, 4, z] = VOXEL_STONE
            grid.grid[5, 6, z] = VOXEL_STONE
        grid.grid[5, 5, 5] = VOXEL_STONE  # floor

        # Fill water column 5 blocks deep (z=0..4)
        for z in range(5):
            grid.grid[5, 5, z] = VOXEL_WATER
            grid.water_level[5, 5, z] = 255

        _tick(eb, WATER_TICK_INTERVAL)

        # Shear load on wall should increase with depth
        shallow_load = grid.shear_load[6, 5, 0]
        deep_load = grid.shear_load[6, 5, 4]
        assert deep_load > shallow_load

    def test_water_burst_weak_wall(self):
        """High pressure should burst weak walls (chalk)."""
        eb, grid, wp = _make(w=10, d=10, h=15)
        events = []
        eb.subscribe("water_burst", lambda **kw: events.append(kw))

        # Create a deep column of water (12 blocks deep)
        for z in range(12):
            grid.grid[5, 5, z] = VOXEL_WATER
            grid.water_level[5, 5, z] = 255

        # Chalk wall adjacent at depth (chalk shear_strength = 2.0)
        grid.grid[6, 5, 11] = VOXEL_CHALK

        # Pressure at depth 12: 12 * 3.0 * 0.3 = 10.8
        # Burst threshold: 2.0 * 1.5 = 3.0 → 10.8 > 3.0 → bursts!
        _tick(eb, WATER_TICK_INTERVAL)

        assert grid.grid[6, 5, 11] == VOXEL_AIR
        assert len(events) > 0

    def test_water_does_not_burst_granite(self):
        """Granite walls should resist reasonable water pressure."""
        eb, grid, wp = _make()
        # 5-block water column
        for z in range(5):
            grid.grid[5, 5, z] = VOXEL_WATER
            grid.water_level[5, 5, z] = 255

        # Granite wall (shear_strength = 20.0)
        grid.grid[6, 5, 4] = VOXEL_GRANITE

        # Pressure at depth 5: 5 * 3.0 * 0.3 = 4.5
        # Burst threshold: 20.0 * 1.5 = 30.0 → 4.5 < 30.0 → no burst
        _tick(eb, WATER_TICK_INTERVAL)

        assert grid.grid[6, 5, 4] == VOXEL_GRANITE


# ──────────────────────────────────────────────────────────────────────
# Lava interaction
# ──────────────────────────────────────────────────────────────────────


class TestLavaInteraction:
    def test_water_lava_creates_obsidian(self):
        """Water adjacent to lava should produce obsidian."""
        eb, grid, wp = _make()
        events = []
        eb.subscribe("water_lava_reaction", lambda **kw: events.append(True))

        grid.grid[5, 5, 5] = VOXEL_WATER
        grid.water_level[5, 5, 5] = 200
        grid.grid[6, 5, 5] = VOXEL_LAVA
        grid.temperature[6, 5, 5] = 1000.0

        _tick(eb, WATER_TICK_INTERVAL)

        # Both should become obsidian
        assert grid.grid[5, 5, 5] == WATER_LAVA_PRODUCT  # obsidian
        assert grid.grid[6, 5, 5] == WATER_LAVA_PRODUCT  # obsidian
        assert len(events) > 0
        # Water level should be cleared
        assert grid.water_level[5, 5, 5] == 0


# ──────────────────────────────────────────────────────────────────────
# Humidity / seepage
# ──────────────────────────────────────────────────────────────────────


class TestHumiditySeepage:
    def test_water_is_humidity_source(self):
        """Adjacent blocks should gain humidity from water."""
        eb, grid, wp = _make()
        # Floor below water so it doesn't fall
        grid.grid[5, 5, 6] = VOXEL_STONE
        grid.grid[5, 5, 5] = VOXEL_WATER
        grid.water_level[5, 5, 5] = 200
        # Sandstone adjacent (porosity=0.35)
        grid.grid[6, 5, 5] = VOXEL_SANDSTONE
        grid.grid[6, 5, 6] = VOXEL_STONE  # floor for sandstone too
        grid.humidity[6, 5, 5] = 0.0

        _tick(eb, WATER_TICK_INTERVAL)

        # Sandstone should have gained humidity via seepage
        assert grid.humidity[6, 5, 5] > 0

    def test_water_seepage_through_sandstone(self):
        """Seepage rate should scale with material porosity."""
        eb, grid, wp = _make()
        # Floor and containment so water doesn't fall/spread away
        grid.grid[5, 5, 6] = VOXEL_STONE
        grid.grid[5, 5, 5] = VOXEL_WATER
        grid.water_level[5, 5, 5] = 255

        # High porosity (sandstone, 0.35) vs low porosity (granite, 0.005)
        grid.grid[6, 5, 5] = VOXEL_SANDSTONE
        grid.grid[6, 5, 6] = VOXEL_STONE
        grid.grid[4, 5, 5] = VOXEL_GRANITE
        grid.grid[4, 5, 6] = VOXEL_STONE
        grid.humidity[6, 5, 5] = 0.0
        grid.humidity[4, 5, 5] = 0.0

        _tick(eb, WATER_TICK_INTERVAL)

        sand_hum = grid.humidity[6, 5, 5]
        gran_hum = grid.humidity[4, 5, 5]
        assert sand_hum > gran_hum


# ──────────────────────────────────────────────────────────────────────
# Evaporation
# ──────────────────────────────────────────────────────────────────────


class TestEvaporation:
    def test_water_surface_evaporation(self):
        """Water at z=0 should lose water_level over time."""
        eb, grid, wp = _make()
        grid.grid[5, 5, 0] = VOXEL_WATER
        grid.water_level[5, 5, 0] = 100

        initial = grid.water_level[5, 5, 0]
        _tick(eb, WATER_TICK_INTERVAL)

        assert grid.water_level[5, 5, 0] < initial


# ──────────────────────────────────────────────────────────────────────
# Conservation
# ──────────────────────────────────────────────────────────────────────


class TestConservation:
    def test_water_conservation_no_sinks(self):
        """In a sealed chamber with no surface, water mass is conserved."""
        eb, grid, wp = _make()
        # Create sealed chamber: stone box at z=2..7, x=3..6, y=3..6
        # Walls around interior, water inside
        for x in range(3, 7):
            for y in range(3, 7):
                grid.grid[x, y, 2] = VOXEL_STONE  # ceiling
                grid.grid[x, y, 7] = VOXEL_STONE  # floor
                for z in range(3, 7):
                    if x == 3 or x == 6 or y == 3 or y == 6:
                        grid.grid[x, y, z] = VOXEL_STONE  # walls
                    else:
                        grid.grid[x, y, z] = VOXEL_WATER
                        grid.water_level[x, y, z] = 200

        # Sum initial water
        initial_sum = int(np.sum(grid.water_level))

        # Run several ticks
        for t in range(1, 20):
            if t % WATER_TICK_INTERVAL == 0:
                _tick(eb, t)

        final_sum = int(np.sum(grid.water_level))

        # Water should be conserved (sealed, no evaporation since not z=0)
        assert final_sum == initial_sum


# ──────────────────────────────────────────────────────────────────────
# Non-diggable
# ──────────────────────────────────────────────────────────────────────


class TestWaterProperties:
    def test_water_not_diggable(self):
        """Water should be in the NON_DIGGABLE set."""
        assert VOXEL_WATER in NON_DIGGABLE


# ──────────────────────────────────────────────────────────────────────
# Tick interval
# ──────────────────────────────────────────────────────────────────────


class TestTickInterval:
    def test_water_only_runs_on_interval(self):
        """Water physics should only process on correct tick intervals."""
        eb, grid, wp = _make()
        grid.grid[5, 5, 3] = VOXEL_WATER
        grid.water_level[5, 5, 3] = 200

        # Tick that's NOT on the interval
        off_tick = WATER_TICK_INTERVAL + 1 if WATER_TICK_INTERVAL > 1 else 3
        _tick(eb, off_tick)

        # Water should not have moved
        assert grid.grid[5, 5, 3] == VOXEL_WATER
        assert grid.water_level[5, 5, 3] == 200


# ──────────────────────────────────────────────────────────────────────
# Gravity integration
# ──────────────────────────────────────────────────────────────────────


class TestGravityIntegration:
    def test_blocks_cannot_fall_through_water(self):
        """Loose blocks should not pass through water (treated as blocking)."""
        from dungeon_builder.world.physics.gravity import GravityPhysics

        eb = EventBus()
        grid = VoxelGrid(10, 10, 10)
        gp = GravityPhysics(eb, grid)

        # Water at z=5
        grid.grid[5, 5, 5] = VOXEL_WATER
        grid.water_level[5, 5, 5] = 200

        # Loose stone at z=3 (should stop at z=4, above water)
        grid.grid[5, 5, 3] = VOXEL_STONE
        grid.loose[5, 5, 3] = True

        _tick(eb, 1)  # gravity runs every tick

        # Stone should be at z=4 (resting above water) or still above water
        # It falls through air and stops when it hits water (non-air)
        stone_positions = np.where(grid.grid == VOXEL_STONE)
        if len(stone_positions[0]) > 0:
            max_z = max(stone_positions[2])
            # Stone should not have passed through water (z=5)
            assert max_z <= 4


# ──────────────────────────────────────────────────────────────────────
# Heat conduction
# ──────────────────────────────────────────────────────────────────────


class TestHeatConduction:
    def test_water_conducts_heat(self):
        """Temperature should diffuse through water (conductivity = 0.6)."""
        from dungeon_builder.world.physics.temperature import TemperaturePhysics
        from dungeon_builder.config import TEMPERATURE_TICK_INTERVAL

        eb = EventBus()
        grid = VoxelGrid(10, 10, 10)
        tp = TemperaturePhysics(eb, grid)

        # Hot block at x=4, water at x=5, cold block at x=6
        grid.grid[4, 5, 5] = VOXEL_STONE
        grid.temperature[4, 5, 5] = 500.0
        grid.grid[5, 5, 5] = VOXEL_WATER
        grid.water_level[5, 5, 5] = 200
        grid.temperature[5, 5, 5] = 20.0
        grid.grid[6, 5, 5] = VOXEL_STONE
        grid.temperature[6, 5, 5] = 20.0

        _tick(eb, TEMPERATURE_TICK_INTERVAL)

        # Water should have gained some heat from the hot stone
        assert grid.temperature[5, 5, 5] > 20.0
