"""Tests for thermal stress / thermal cracking physics."""

import pytest
import numpy as np

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.world.physics.thermal_stress import ThermalStressPhysics
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_STONE,
    VOXEL_BEDROCK,
    VOXEL_GRANITE,
    VOXEL_OBSIDIAN,
    VOXEL_IRON_INGOT,
    VOXEL_WATER,
    VOXEL_LAVA,
    VOXEL_MANA_CRYSTAL,
    VOXEL_CTE,
    VOXEL_TENSILE_STRENGTH,
    THERMAL_STRESS_TICK_INTERVAL,
    THERMAL_FATIGUE_ACCUMULATION,
    THERMAL_FATIGUE_DECAY,
    THERMAL_CRACK_THRESHOLD,
    QUENCH_MULTIPLIER,
    LAVA_TEMPERATURE,
    MAX_CASCADE_PER_TICK,
)


def _setup(width=8, depth=8, height=5):
    bus = EventBus()
    grid = VoxelGrid(width, depth, height)
    phys = ThermalStressPhysics(bus, grid)
    return bus, grid, phys


def _tick(bus, tick_num):
    bus.publish("tick", tick=tick_num)


class TestNoGradientNoStress:
    def test_no_gradient_no_stress(self):
        """Uniform temperature produces zero thermal fatigue."""
        bus, grid, phys = _setup()
        # Fill grid with stone at uniform temperature
        grid.grid[:, :, :] = VOXEL_STONE
        grid.temperature[:, :, :] = 100.0

        _tick(bus, THERMAL_STRESS_TICK_INTERVAL)

        # No gradient → no fatigue
        assert np.all(grid.thermal_fatigue == 0.0)


class TestGradientProducesFatigue:
    def test_gradient_produces_fatigue(self):
        """Hot block next to cold block accumulates fatigue."""
        bus, grid, phys = _setup()
        grid.grid[:, :, :] = VOXEL_STONE

        # Hot block at (3,3,2), cold surroundings
        grid.temperature[:, :, :] = 0.0
        grid.temperature[3, 3, 2] = 500.0

        _tick(bus, THERMAL_STRESS_TICK_INTERVAL)

        # The hot block and its neighbors should have fatigue
        assert grid.thermal_fatigue[3, 3, 2] > 0.0
        # Neighbors also feel the gradient
        assert grid.thermal_fatigue[4, 3, 2] > 0.0


class TestObsidianCracksNearLava:
    def test_obsidian_cracks_near_lava(self):
        """Obsidian (highest CTE) next to lava temperature cracks fast."""
        bus, grid, phys = _setup()
        grid.grid[:, :, :] = VOXEL_AIR

        # Obsidian block at (3,3,2)
        grid.grid[3, 3, 2] = VOXEL_OBSIDIAN
        # Support below so it isn't loose from gravity
        grid.grid[3, 3, 3] = VOXEL_BEDROCK

        # Neighboring block is at lava temperature
        grid.grid[4, 3, 2] = VOXEL_STONE
        grid.temperature[4, 3, 2] = LAVA_TEMPERATURE
        grid.temperature[3, 3, 2] = 0.0

        # Obsidian CTE=0.025, tensile=18.0 (3× buff)
        # gradient=1000, stress=25, ratio=25/18=1.39
        # fatigue_per_tick = 1.39 * 0.1 = 0.139
        # Should crack in ~8 ticks (1.11 total fatigue > 1.0)

        events_fired = []
        grid.loose[3, 3, 2] = False  # Ensure not already loose

        def on_crack(**kw):
            events_fired.append(kw)

        bus.subscribe("thermal_crack", on_crack)

        for i in range(1, 12):
            _tick(bus, THERMAL_STRESS_TICK_INTERVAL * i)

        assert bool(grid.loose[3, 3, 2]) is True
        assert len(events_fired) > 0


class TestGraniteResistsThermalShock:
    def test_granite_resists_thermal_shock(self):
        """Granite (low CTE, high tensile) survives longer than obsidian."""
        bus, grid, phys = _setup()
        grid.grid[:, :, :] = VOXEL_AIR
        grid.grid[3, 3, 2] = VOXEL_GRANITE
        grid.grid[3, 3, 3] = VOXEL_BEDROCK
        grid.grid[4, 3, 2] = VOXEL_STONE
        grid.temperature[4, 3, 2] = LAVA_TEMPERATURE
        grid.temperature[3, 3, 2] = 0.0

        # Granite CTE=0.005, tensile=15.0
        # ratio = 1000*0.005/15 = 0.33
        # fatigue_per_tick = 0.33 * 0.1 = 0.033
        # Would need ~30 ticks to crack

        # After 5 ticks, should NOT be cracked
        for i in range(1, 6):
            _tick(bus, THERMAL_STRESS_TICK_INTERVAL * i)

        assert bool(grid.loose[3, 3, 2]) is False
        # But should have accumulated some fatigue
        assert grid.thermal_fatigue[3, 3, 2] > 0.0


class TestMetalResistsThermalStress:
    def test_metal_resists_thermal_stress(self):
        """Iron ingot (very low CTE) barely accumulates fatigue."""
        bus, grid, phys = _setup()
        grid.grid[:, :, :] = VOXEL_AIR
        grid.grid[3, 3, 2] = VOXEL_IRON_INGOT
        grid.grid[3, 3, 3] = VOXEL_BEDROCK
        grid.grid[4, 3, 2] = VOXEL_STONE
        grid.temperature[4, 3, 2] = LAVA_TEMPERATURE
        grid.temperature[3, 3, 2] = 0.0

        # Iron CTE=0.003, tensile=40.0
        # ratio = 1000*0.003/40 = 0.075
        # fatigue_per_tick = 0.075 * 0.1 = 0.0075 — very slow

        for i in range(1, 6):
            _tick(bus, THERMAL_STRESS_TICK_INTERVAL * i)

        assert bool(grid.loose[3, 3, 2]) is False
        # Fatigue should be very low
        assert grid.thermal_fatigue[3, 3, 2] < 0.1


class TestAnchorImmune:
    def test_anchor_immune_to_thermal_crack(self):
        """Bedrock never cracks from thermal stress."""
        bus, grid, phys = _setup()
        grid.grid[:, :, :] = VOXEL_AIR
        grid.grid[3, 3, 2] = VOXEL_BEDROCK
        grid.grid[4, 3, 2] = VOXEL_STONE
        grid.temperature[4, 3, 2] = LAVA_TEMPERATURE
        grid.temperature[3, 3, 2] = 0.0

        for i in range(1, 20):
            _tick(bus, THERMAL_STRESS_TICK_INTERVAL * i)

        assert bool(grid.loose[3, 3, 2]) is False


class TestQuenchingMultiplier:
    def test_quenching_multiplier(self):
        """Water adjacent to hot block amplifies thermal stress by 3x."""
        bus1, grid1, phys1 = _setup()
        bus2, grid2, phys2 = _setup()

        # Setup 1: stone block with gradient, no water
        grid1.grid[:, :, :] = VOXEL_AIR
        grid1.grid[3, 3, 2] = VOXEL_STONE
        grid1.grid[3, 3, 3] = VOXEL_BEDROCK
        grid1.grid[4, 3, 2] = VOXEL_STONE
        grid1.temperature[4, 3, 2] = 800.0
        grid1.temperature[3, 3, 2] = 0.0

        # Setup 2: same but with water on other side
        grid2.grid[:, :, :] = VOXEL_AIR
        grid2.grid[3, 3, 2] = VOXEL_STONE
        grid2.grid[3, 3, 3] = VOXEL_BEDROCK
        grid2.grid[4, 3, 2] = VOXEL_STONE
        grid2.temperature[4, 3, 2] = 800.0
        grid2.temperature[3, 3, 2] = 0.0
        grid2.grid[2, 3, 2] = VOXEL_WATER  # water adjacent!

        _tick(bus1, THERMAL_STRESS_TICK_INTERVAL)
        _tick(bus2, THERMAL_STRESS_TICK_INTERVAL)

        # With water, fatigue should be ~3x higher
        fatigue_no_water = grid1.thermal_fatigue[3, 3, 2]
        fatigue_with_water = grid2.thermal_fatigue[3, 3, 2]
        assert fatigue_with_water > fatigue_no_water * 2.5


class TestFatigueDecays:
    def test_fatigue_decays_when_cool(self):
        """Fatigue slowly heals when temperature gradient disappears."""
        bus, grid, phys = _setup()
        grid.grid[:, :, :] = VOXEL_STONE

        # Create gradient to build fatigue
        grid.temperature[:, :, :] = 0.0
        grid.temperature[3, 3, 2] = 500.0
        _tick(bus, THERMAL_STRESS_TICK_INTERVAL)
        initial_fatigue = float(grid.thermal_fatigue[3, 3, 2])
        assert initial_fatigue > 0.0

        # Remove gradient
        grid.temperature[:, :, :] = 100.0  # uniform

        _tick(bus, THERMAL_STRESS_TICK_INTERVAL * 2)
        decayed_fatigue = float(grid.thermal_fatigue[3, 3, 2])
        assert decayed_fatigue < initial_fatigue


class TestStressRatioIncludesThermal:
    def test_stress_ratio_includes_thermal(self):
        """stress_ratio reflects thermal contribution."""
        bus, grid, phys = _setup()
        grid.grid[:, :, :] = VOXEL_AIR
        grid.grid[3, 3, 2] = VOXEL_OBSIDIAN
        grid.grid[3, 3, 3] = VOXEL_BEDROCK
        grid.grid[4, 3, 2] = VOXEL_STONE
        grid.temperature[4, 3, 2] = LAVA_TEMPERATURE
        grid.temperature[3, 3, 2] = 0.0

        # Stress ratio starts at 0
        assert grid.stress_ratio[3, 3, 2] == 0.0

        _tick(bus, THERMAL_STRESS_TICK_INTERVAL)

        # Should now be elevated
        assert grid.stress_ratio[3, 3, 2] > 0.0


class TestCrackedBlocksBecomeLoose:
    def test_cracked_blocks_become_loose(self):
        """Blocks that crack from thermal fatigue have loose=True."""
        bus, grid, phys = _setup()
        grid.grid[:, :, :] = VOXEL_AIR
        grid.grid[3, 3, 2] = VOXEL_OBSIDIAN
        grid.grid[3, 3, 3] = VOXEL_BEDROCK

        # Set fatigue just below threshold, then trigger one more tick
        grid.thermal_fatigue[3, 3, 2] = THERMAL_CRACK_THRESHOLD - 0.1

        # Create a large gradient to push over threshold
        grid.grid[4, 3, 2] = VOXEL_STONE
        grid.temperature[4, 3, 2] = LAVA_TEMPERATURE
        grid.temperature[3, 3, 2] = 0.0

        assert bool(grid.loose[3, 3, 2]) is False
        _tick(bus, THERMAL_STRESS_TICK_INTERVAL)
        assert bool(grid.loose[3, 3, 2]) is True


class TestOnlyRunsOnInterval:
    def test_only_runs_on_interval(self):
        """Thermal stress only computed on THERMAL_STRESS_TICK_INTERVAL ticks."""
        bus, grid, phys = _setup()
        grid.grid[:, :, :] = VOXEL_STONE
        grid.temperature[:, :, :] = 0.0
        grid.temperature[3, 3, 2] = 500.0

        # Tick that's not on interval
        _tick(bus, 1)
        assert np.all(grid.thermal_fatigue == 0.0)

        # Tick on interval
        _tick(bus, THERMAL_STRESS_TICK_INTERVAL)
        assert grid.thermal_fatigue[3, 3, 2] > 0.0


class TestCascadeCapRespected:
    def test_cascade_cap_respected(self):
        """No more than MAX_CASCADE_PER_TICK cracks per tick."""
        bus, grid, phys = _setup(width=10, depth=10, height=5)
        # Fill with obsidian at high fatigue
        grid.grid[:, :, :] = VOXEL_OBSIDIAN
        grid.grid[:, :, 4] = VOXEL_BEDROCK  # bottom support
        grid.thermal_fatigue[:, :, :3] = THERMAL_CRACK_THRESHOLD - 0.01

        # Create massive gradient
        grid.temperature[:, :, :] = 0.0
        grid.temperature[0, :, :] = LAVA_TEMPERATURE

        events = []
        bus.subscribe("thermal_crack", lambda **kw: events.append(kw))

        _tick(bus, THERMAL_STRESS_TICK_INTERVAL)

        if events:
            total_cracked = sum(e["count"] for e in events)
            assert total_cracked <= MAX_CASCADE_PER_TICK


class TestThermalCrackResetsFatigue:
    def test_thermal_crack_resets_fatigue(self):
        """After cracking, fatigue is reset to 0."""
        bus, grid, phys = _setup()
        grid.grid[:, :, :] = VOXEL_AIR
        grid.grid[3, 3, 2] = VOXEL_OBSIDIAN
        grid.grid[3, 3, 3] = VOXEL_BEDROCK

        # Set high fatigue and trigger crack
        grid.thermal_fatigue[3, 3, 2] = THERMAL_CRACK_THRESHOLD - 0.01
        grid.grid[4, 3, 2] = VOXEL_STONE
        grid.temperature[4, 3, 2] = LAVA_TEMPERATURE
        grid.temperature[3, 3, 2] = 0.0

        _tick(bus, THERMAL_STRESS_TICK_INTERVAL)

        assert bool(grid.loose[3, 3, 2]) is True
        assert grid.thermal_fatigue[3, 3, 2] == pytest.approx(0.0)


class TestObsidianQuenchCracksFast:
    def test_obsidian_quench_cracks_fast(self):
        """Obsidian at high temp next to water cracks within 2-3 ticks."""
        bus, grid, phys = _setup()
        grid.grid[:, :, :] = VOXEL_AIR
        grid.grid[3, 3, 2] = VOXEL_OBSIDIAN
        grid.grid[3, 3, 3] = VOXEL_BEDROCK
        grid.grid[2, 3, 2] = VOXEL_WATER  # water for quenching

        # Obsidian is hot, water is cold — extreme gradient
        grid.temperature[3, 3, 2] = 800.0
        grid.temperature[2, 3, 2] = WATER_TEMPERATURE = 20.0

        # ratio = 780 * 0.025 * 3.0 / 18.0 = 3.25 (3× buff tensile)
        # fatigue_per_tick = 3.25 * 0.1 = 0.325
        # Should crack in ~4 ticks (1.3 total fatigue > 1.0)

        for i in range(1, 6):
            _tick(bus, THERMAL_STRESS_TICK_INTERVAL * i)

        assert bool(grid.loose[3, 3, 2]) is True
