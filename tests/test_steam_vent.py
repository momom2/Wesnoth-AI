"""Tests for steam vent physics (heat and humidity pulses upward through air).

Steam vents pulse heat and humidity upward through air cells above them.
In this coordinate system, z-1 = shallower/upward, z+1 = deeper/downward.
So "above" a vent at z=5 means z=4, z=3, z=2.

Heat logic lives in TemperaturePhysics._apply_steam_vent_heat().
Humidity logic lives in HumidityPhysics._apply_steam_vent_humidity().
"""

import pytest
import numpy as np

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.world.physics.temperature import TemperaturePhysics
from dungeon_builder.world.physics.humidity import HumidityPhysics
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_STONE,
    VOXEL_STEAM_VENT,
    STEAM_VENT_HEAT_PULSE,
    STEAM_VENT_HUMIDITY_PULSE,
    STEAM_VENT_RANGE,
    TEMPERATURE_TICK_INTERVAL,
    HUMIDITY_TICK_INTERVAL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid_with_vent(vent_x, vent_y, vent_z, air_zs, size=8):
    """Create an 8x8x8 stone grid with a vent and air column above it.

    Parameters
    ----------
    vent_x, vent_y, vent_z : int
        Position of the steam vent.
    air_zs : list[int]
        Z-levels to set to air above the vent (same x, y).
    size : int
        Grid side length (cube).

    Returns
    -------
    tuple[VoxelGrid, EventBus, TemperaturePhysics, HumidityPhysics]
    """
    bus = EventBus()
    grid = VoxelGrid(size, size, size)

    # Fill entire grid with stone
    grid.grid[:] = VOXEL_STONE

    # Place vent
    grid.grid[vent_x, vent_y, vent_z] = VOXEL_STEAM_VENT

    # Carve air above
    for z in air_zs:
        grid.grid[vent_x, vent_y, z] = VOXEL_AIR

    tp = TemperaturePhysics(bus, grid)
    hp = HumidityPhysics(bus, grid)
    return grid, bus, tp, hp


# ---------------------------------------------------------------------------
# 1. Heat pulse upward through air
# ---------------------------------------------------------------------------

class TestSteamVentHeatPulse:
    """Verify that _apply_steam_vent_heat adds heat to air cells above."""

    def test_heat_pulse_upward_through_air(self):
        """Vent at (3,3,5), air at z=4,3,2 -- each gets +STEAM_VENT_HEAT_PULSE."""
        grid, bus, tp, hp = _make_grid_with_vent(3, 3, 5, [4, 3, 2])

        # Ensure temperatures start at zero
        assert grid.temperature[3, 3, 4] == 0.0
        assert grid.temperature[3, 3, 3] == 0.0
        assert grid.temperature[3, 3, 2] == 0.0

        tp._apply_steam_vent_heat(grid.grid, grid.temperature)

        assert grid.temperature[3, 3, 4] == pytest.approx(STEAM_VENT_HEAT_PULSE)
        assert grid.temperature[3, 3, 3] == pytest.approx(STEAM_VENT_HEAT_PULSE)
        assert grid.temperature[3, 3, 2] == pytest.approx(STEAM_VENT_HEAT_PULSE)


# ---------------------------------------------------------------------------
# 2. Humidity pulse upward through air
# ---------------------------------------------------------------------------

class TestSteamVentHumidityPulse:
    """Verify that _apply_steam_vent_humidity adds humidity to air cells above."""

    def test_humidity_pulse_upward_through_air(self):
        """Vent at (3,3,5), air at z=4,3,2 -- each gets +STEAM_VENT_HUMIDITY_PULSE."""
        grid, bus, tp, hp = _make_grid_with_vent(3, 3, 5, [4, 3, 2])

        assert grid.humidity[3, 3, 4] == 0.0
        assert grid.humidity[3, 3, 3] == 0.0
        assert grid.humidity[3, 3, 2] == 0.0

        hp._apply_steam_vent_humidity(grid.grid, grid.humidity)

        assert grid.humidity[3, 3, 4] == pytest.approx(STEAM_VENT_HUMIDITY_PULSE)
        assert grid.humidity[3, 3, 3] == pytest.approx(STEAM_VENT_HUMIDITY_PULSE)
        assert grid.humidity[3, 3, 2] == pytest.approx(STEAM_VENT_HUMIDITY_PULSE)


# ---------------------------------------------------------------------------
# 3. Blocked by solid above
# ---------------------------------------------------------------------------

class TestSteamVentBlockedBySolid:
    """A solid block immediately above the vent blocks all propagation."""

    def test_heat_blocked_by_stone_above(self):
        """Vent at z=5, stone at z=4 (no air) -> z=4 and z=3 get nothing."""
        grid, bus, tp, hp = _make_grid_with_vent(3, 3, 5, [])  # no air at all

        tp._apply_steam_vent_heat(grid.grid, grid.temperature)

        # z=4 is stone, z=3 is stone -- neither should receive heat
        assert grid.temperature[3, 3, 4] == 0.0
        assert grid.temperature[3, 3, 3] == 0.0

    def test_humidity_blocked_by_stone_above(self):
        """Vent at z=5, stone at z=4 -> z=4 and z=3 get nothing."""
        grid, bus, tp, hp = _make_grid_with_vent(3, 3, 5, [])

        hp._apply_steam_vent_humidity(grid.grid, grid.humidity)

        assert grid.humidity[3, 3, 4] == 0.0
        assert grid.humidity[3, 3, 3] == 0.0


# ---------------------------------------------------------------------------
# 4. Partial blockage
# ---------------------------------------------------------------------------

class TestSteamVentPartialBlockage:
    """Air at z=4, stone at z=3 -> z=4 gets pulse, z=3 and z=2 get nothing."""

    def test_heat_partial_blockage(self):
        """Only the first air cell (z=4) receives heat; stone at z=3 stops it."""
        grid, bus, tp, hp = _make_grid_with_vent(3, 3, 5, [4])
        # z=3 is stone (default fill), z=2 is stone

        tp._apply_steam_vent_heat(grid.grid, grid.temperature)

        assert grid.temperature[3, 3, 4] == pytest.approx(STEAM_VENT_HEAT_PULSE)
        assert grid.temperature[3, 3, 3] == 0.0  # stone blocks
        assert grid.temperature[3, 3, 2] == 0.0  # behind the blockage

    def test_humidity_partial_blockage(self):
        """Only the first air cell (z=4) receives humidity; stone at z=3 stops it."""
        grid, bus, tp, hp = _make_grid_with_vent(3, 3, 5, [4])

        hp._apply_steam_vent_humidity(grid.grid, grid.humidity)

        assert grid.humidity[3, 3, 4] == pytest.approx(STEAM_VENT_HUMIDITY_PULSE)
        assert grid.humidity[3, 3, 3] == 0.0
        assert grid.humidity[3, 3, 2] == 0.0


# ---------------------------------------------------------------------------
# 5. Range limit
# ---------------------------------------------------------------------------

class TestSteamVentRangeLimit:
    """Only STEAM_VENT_RANGE cells above are affected; z-4 gets nothing."""

    def test_heat_range_limit(self):
        """Vent at z=5, air at z=4,3,2,1 -- only z=4,3,2 get heat (range=3)."""
        grid, bus, tp, hp = _make_grid_with_vent(3, 3, 5, [4, 3, 2, 1])

        tp._apply_steam_vent_heat(grid.grid, grid.temperature)

        # Within range
        assert grid.temperature[3, 3, 4] == pytest.approx(STEAM_VENT_HEAT_PULSE)
        assert grid.temperature[3, 3, 3] == pytest.approx(STEAM_VENT_HEAT_PULSE)
        assert grid.temperature[3, 3, 2] == pytest.approx(STEAM_VENT_HEAT_PULSE)
        # Beyond range (dz=4 > STEAM_VENT_RANGE=3)
        assert grid.temperature[3, 3, 1] == 0.0

    def test_humidity_range_limit(self):
        """Vent at z=5, air at z=4,3,2,1 -- only z=4,3,2 get humidity."""
        grid, bus, tp, hp = _make_grid_with_vent(3, 3, 5, [4, 3, 2, 1])

        hp._apply_steam_vent_humidity(grid.grid, grid.humidity)

        assert grid.humidity[3, 3, 4] == pytest.approx(STEAM_VENT_HUMIDITY_PULSE)
        assert grid.humidity[3, 3, 3] == pytest.approx(STEAM_VENT_HUMIDITY_PULSE)
        assert grid.humidity[3, 3, 2] == pytest.approx(STEAM_VENT_HUMIDITY_PULSE)
        assert grid.humidity[3, 3, 1] == 0.0


# ---------------------------------------------------------------------------
# 6. Vent at surface (z=0) -- no cells above, no crash
# ---------------------------------------------------------------------------

class TestSteamVentAtSurface:
    """Vent at z=0 -> z=-1 is out of bounds -> no crash, no effect."""

    def test_heat_vent_at_surface_no_crash(self):
        """Vent at z=0 should not crash and should not alter any temperatures."""
        bus = EventBus()
        grid = VoxelGrid(8, 8, 8)
        grid.grid[:] = VOXEL_STONE
        grid.grid[3, 3, 0] = VOXEL_STEAM_VENT

        tp = TemperaturePhysics(bus, grid)
        temp_before = grid.temperature.copy()

        tp._apply_steam_vent_heat(grid.grid, grid.temperature)

        np.testing.assert_array_equal(grid.temperature, temp_before)

    def test_humidity_vent_at_surface_no_crash(self):
        """Vent at z=0 should not crash and should not alter any humidity."""
        bus = EventBus()
        grid = VoxelGrid(8, 8, 8)
        grid.grid[:] = VOXEL_STONE
        grid.grid[3, 3, 0] = VOXEL_STEAM_VENT

        hp = HumidityPhysics(bus, grid)
        hum_before = grid.humidity.copy()

        hp._apply_steam_vent_humidity(grid.grid, grid.humidity)

        np.testing.assert_array_equal(grid.humidity, hum_before)


# ---------------------------------------------------------------------------
# 7. Multiple vents pulse independently
# ---------------------------------------------------------------------------

class TestSteamVentMultipleVents:
    """Two vents at different positions each pulse their own column."""

    def test_two_vents_heat_independently(self):
        """Vent A at (2,2,5) and vent B at (5,5,5) each pulse their air columns."""
        bus = EventBus()
        grid = VoxelGrid(8, 8, 8)
        grid.grid[:] = VOXEL_STONE

        # Vent A with air above
        grid.grid[2, 2, 5] = VOXEL_STEAM_VENT
        for z in [4, 3, 2]:
            grid.grid[2, 2, z] = VOXEL_AIR

        # Vent B with air above
        grid.grid[5, 5, 5] = VOXEL_STEAM_VENT
        for z in [4, 3, 2]:
            grid.grid[5, 5, z] = VOXEL_AIR

        tp = TemperaturePhysics(bus, grid)
        tp._apply_steam_vent_heat(grid.grid, grid.temperature)

        # Vent A column
        for z in [4, 3, 2]:
            assert grid.temperature[2, 2, z] == pytest.approx(STEAM_VENT_HEAT_PULSE)

        # Vent B column
        for z in [4, 3, 2]:
            assert grid.temperature[5, 5, z] == pytest.approx(STEAM_VENT_HEAT_PULSE)

    def test_two_vents_humidity_independently(self):
        """Two separate vents each pulse humidity in their own air columns."""
        bus = EventBus()
        grid = VoxelGrid(8, 8, 8)
        grid.grid[:] = VOXEL_STONE

        grid.grid[2, 2, 5] = VOXEL_STEAM_VENT
        grid.grid[5, 5, 5] = VOXEL_STEAM_VENT
        for z in [4, 3, 2]:
            grid.grid[2, 2, z] = VOXEL_AIR
            grid.grid[5, 5, z] = VOXEL_AIR

        hp = HumidityPhysics(bus, grid)
        hp._apply_steam_vent_humidity(grid.grid, grid.humidity)

        for z in [4, 3, 2]:
            assert grid.humidity[2, 2, z] == pytest.approx(STEAM_VENT_HUMIDITY_PULSE)
            assert grid.humidity[5, 5, z] == pytest.approx(STEAM_VENT_HUMIDITY_PULSE)


# ---------------------------------------------------------------------------
# 8. Cumulative pulses from overlapping vents
# ---------------------------------------------------------------------------

class TestSteamVentCumulativePulse:
    """Calling the pulse method twice accumulates heat; humidity is clamped."""

    def test_heat_cumulative_from_repeated_pulses(self):
        """Two calls to _apply_steam_vent_heat accumulate heat additively."""
        grid, bus, tp, hp = _make_grid_with_vent(3, 3, 5, [4, 3, 2])

        tp._apply_steam_vent_heat(grid.grid, grid.temperature)
        tp._apply_steam_vent_heat(grid.grid, grid.temperature)

        # Each call adds STEAM_VENT_HEAT_PULSE; two calls -> double
        assert grid.temperature[3, 3, 4] == pytest.approx(STEAM_VENT_HEAT_PULSE * 2)
        assert grid.temperature[3, 3, 3] == pytest.approx(STEAM_VENT_HEAT_PULSE * 2)
        assert grid.temperature[3, 3, 2] == pytest.approx(STEAM_VENT_HEAT_PULSE * 2)

    def test_humidity_cumulative_clamped_to_one(self):
        """Two humidity pulses of 0.7 each -> clamped to 1.0 per cell."""
        grid, bus, tp, hp = _make_grid_with_vent(3, 3, 5, [4, 3, 2])

        hp._apply_steam_vent_humidity(grid.grid, grid.humidity)
        hp._apply_steam_vent_humidity(grid.grid, grid.humidity)

        # The method clamps each cell to min(1.0, ...); two pulses of 0.7:
        # first call sets 0.7, second call: min(1.0, 0.7 + 0.7) = 1.0
        assert grid.humidity[3, 3, 4] == pytest.approx(1.0)
        assert grid.humidity[3, 3, 3] == pytest.approx(1.0)
        assert grid.humidity[3, 3, 2] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 9. No vents, no effect
# ---------------------------------------------------------------------------

class TestSteamVentNoVents:
    """Grid with no steam vents should have no temperature/humidity changes."""

    def test_no_vents_no_heat_change(self):
        """All-stone grid with no vents -> temperature stays at zero."""
        bus = EventBus()
        grid = VoxelGrid(8, 8, 8)
        grid.grid[:] = VOXEL_STONE

        tp = TemperaturePhysics(bus, grid)
        temp_before = grid.temperature.copy()

        tp._apply_steam_vent_heat(grid.grid, grid.temperature)

        np.testing.assert_array_equal(grid.temperature, temp_before)

    def test_no_vents_no_humidity_change(self):
        """All-stone grid with no vents -> humidity stays at zero."""
        bus = EventBus()
        grid = VoxelGrid(8, 8, 8)
        grid.grid[:] = VOXEL_STONE

        hp = HumidityPhysics(bus, grid)
        hum_before = grid.humidity.copy()

        hp._apply_steam_vent_humidity(grid.grid, grid.humidity)

        np.testing.assert_array_equal(grid.humidity, hum_before)


# ---------------------------------------------------------------------------
# 10. Vent does not affect same z-level
# ---------------------------------------------------------------------------

class TestSteamVentSameLevel:
    """Only cells above (z-1, z-2, z-3) are affected, not the vent's own z."""

    def test_heat_does_not_affect_vent_level(self):
        """Vent at z=5 -> z=5 temperature remains unchanged."""
        grid, bus, tp, hp = _make_grid_with_vent(3, 3, 5, [4, 3, 2])

        tp._apply_steam_vent_heat(grid.grid, grid.temperature)

        # Vent's own cell should not gain heat from the pulse
        assert grid.temperature[3, 3, 5] == 0.0

    def test_humidity_does_not_affect_vent_level(self):
        """Vent at z=5 -> z=5 humidity remains unchanged."""
        grid, bus, tp, hp = _make_grid_with_vent(3, 3, 5, [4, 3, 2])

        hp._apply_steam_vent_humidity(grid.grid, grid.humidity)

        assert grid.humidity[3, 3, 5] == 0.0

    def test_heat_does_not_affect_below_vent(self):
        """Vent at z=5 -> z=6 (below / deeper) gets nothing."""
        grid, bus, tp, hp = _make_grid_with_vent(3, 3, 5, [4, 3, 2])
        # Make z=6 air just in case it matters
        grid.grid[3, 3, 6] = VOXEL_AIR

        tp._apply_steam_vent_heat(grid.grid, grid.temperature)

        assert grid.temperature[3, 3, 6] == 0.0

    def test_humidity_does_not_affect_below_vent(self):
        """Vent at z=5 -> z=6 (below / deeper) gets nothing."""
        grid, bus, tp, hp = _make_grid_with_vent(3, 3, 5, [4, 3, 2])
        grid.grid[3, 3, 6] = VOXEL_AIR

        hp._apply_steam_vent_humidity(grid.grid, grid.humidity)

        assert grid.humidity[3, 3, 6] == 0.0
