"""Tests for the metal melting system in TemperaturePhysics._check_melting()."""

import pytest
import numpy as np

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.world.physics.temperature import TemperaturePhysics
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_STONE,
    VOXEL_PIPE,
    VOXEL_SPIKE,
    VOXEL_DOOR,
    VOXEL_REINFORCED_WALL,
    VOXEL_IRON_BARS,
    VOXEL_FLOODGATE,
    VOXEL_PRESSURE_PLATE,
    VOXEL_HEAT_BEACON,
    VOXEL_ALARM_BELL,
    VOXEL_GOLD_BAIT,
    METAL_IRON,
    METAL_COPPER,
    METAL_GOLD,
    METAL_MELT_TEMPERATURE,
    ENCHANTED_OFFSET,
    TEMPERATURE_TICK_INTERVAL,
)


def _setup(width=8, depth=8, height=8):
    """Create a minimal event bus, voxel grid, and temperature physics instance."""
    eb = EventBus()
    grid = VoxelGrid(width=width, depth=depth, height=height)
    tp = TemperaturePhysics(eb, grid)
    return eb, grid, tp


# ---------------------------------------------------------------------------
# Basic copper pipe melting threshold
# ---------------------------------------------------------------------------


class TestCopperPipeMelting:
    def test_copper_pipe_melts_at_threshold(self):
        """Copper pipe at exactly 800 C should melt to air."""
        eb, grid, tp = _setup()
        grid.set(3, 3, 3, VOXEL_PIPE)
        grid.set_metal_type(3, 3, 3, METAL_COPPER)
        grid.temperature[3, 3, 3] = 800.0

        tp._check_melting(grid.grid, grid.temperature)

        assert grid.get(3, 3, 3) == VOXEL_AIR

    def test_copper_pipe_survives_below_threshold(self):
        """Copper pipe at 799 C (below threshold) should remain."""
        eb, grid, tp = _setup()
        grid.set(3, 3, 3, VOXEL_PIPE)
        grid.set_metal_type(3, 3, 3, METAL_COPPER)
        grid.temperature[3, 3, 3] = 799.0

        tp._check_melting(grid.grid, grid.temperature)

        assert grid.get(3, 3, 3) == VOXEL_PIPE


# ---------------------------------------------------------------------------
# Iron pipe melting threshold
# ---------------------------------------------------------------------------


class TestIronPipeMelting:
    def test_iron_pipe_melts_at_threshold(self):
        """Iron pipe at exactly 1200 C should melt to air."""
        eb, grid, tp = _setup()
        grid.set(3, 3, 3, VOXEL_PIPE)
        grid.set_metal_type(3, 3, 3, METAL_IRON)
        grid.temperature[3, 3, 3] = 1200.0

        tp._check_melting(grid.grid, grid.temperature)

        assert grid.get(3, 3, 3) == VOXEL_AIR

    def test_iron_pipe_survives_below_threshold(self):
        """Iron pipe at 1199 C (below threshold) should remain."""
        eb, grid, tp = _setup()
        grid.set(3, 3, 3, VOXEL_PIPE)
        grid.set_metal_type(3, 3, 3, METAL_IRON)
        grid.temperature[3, 3, 3] = 1199.0

        tp._check_melting(grid.grid, grid.temperature)

        assert grid.get(3, 3, 3) == VOXEL_PIPE


# ---------------------------------------------------------------------------
# Gold pipe melting threshold
# ---------------------------------------------------------------------------


class TestGoldPipeMelting:
    def test_gold_pipe_melts_at_threshold(self):
        """Gold pipe at exactly 600 C should melt to air."""
        eb, grid, tp = _setup()
        grid.set(3, 3, 3, VOXEL_PIPE)
        grid.set_metal_type(3, 3, 3, METAL_GOLD)
        grid.temperature[3, 3, 3] = 600.0

        tp._check_melting(grid.grid, grid.temperature)

        assert grid.get(3, 3, 3) == VOXEL_AIR


# ---------------------------------------------------------------------------
# Enchanted immunity
# ---------------------------------------------------------------------------


class TestEnchantedImmunity:
    def test_enchanted_copper_pipe_immune(self):
        """Enchanted copper pipe at 800 C should NOT melt."""
        eb, grid, tp = _setup()
        grid.set(3, 3, 3, VOXEL_PIPE)
        grid.set_metal_type(3, 3, 3, METAL_COPPER | ENCHANTED_OFFSET)
        grid.temperature[3, 3, 3] = 800.0

        tp._check_melting(grid.grid, grid.temperature)

        assert grid.get(3, 3, 3) == VOXEL_PIPE

    def test_enchanted_iron_immune(self):
        """Enchanted iron pipe at 1200 C should NOT melt."""
        eb, grid, tp = _setup()
        grid.set(3, 3, 3, VOXEL_PIPE)
        grid.set_metal_type(3, 3, 3, METAL_IRON | ENCHANTED_OFFSET)
        grid.temperature[3, 3, 3] = 1200.0

        tp._check_melting(grid.grid, grid.temperature)

        assert grid.get(3, 3, 3) == VOXEL_PIPE

    def test_enchanted_gold_immune(self):
        """Enchanted gold pipe at 600 C should NOT melt."""
        eb, grid, tp = _setup()
        grid.set(3, 3, 3, VOXEL_PIPE)
        grid.set_metal_type(3, 3, 3, METAL_GOLD | ENCHANTED_OFFSET)
        grid.temperature[3, 3, 3] = 600.0

        tp._check_melting(grid.grid, grid.temperature)

        assert grid.get(3, 3, 3) == VOXEL_PIPE


# ---------------------------------------------------------------------------
# Other meltable block types
# ---------------------------------------------------------------------------


class TestOtherMeltableBlocks:
    def test_spike_melts(self):
        """Iron spike at 1200 C should melt to air."""
        eb, grid, tp = _setup()
        grid.set(3, 3, 3, VOXEL_SPIKE)
        grid.set_metal_type(3, 3, 3, METAL_IRON)
        grid.temperature[3, 3, 3] = 1200.0

        tp._check_melting(grid.grid, grid.temperature)

        assert grid.get(3, 3, 3) == VOXEL_AIR

    def test_door_melts(self):
        """Gold door at 600 C should melt to air."""
        eb, grid, tp = _setup()
        grid.set(3, 3, 3, VOXEL_DOOR)
        grid.set_metal_type(3, 3, 3, METAL_GOLD)
        grid.temperature[3, 3, 3] = 600.0

        tp._check_melting(grid.grid, grid.temperature)

        assert grid.get(3, 3, 3) == VOXEL_AIR

    def test_reinforced_wall_melts(self):
        """Copper reinforced wall at 800 C should melt to air."""
        eb, grid, tp = _setup()
        grid.set(3, 3, 3, VOXEL_REINFORCED_WALL)
        grid.set_metal_type(3, 3, 3, METAL_COPPER)
        grid.temperature[3, 3, 3] = 800.0

        tp._check_melting(grid.grid, grid.temperature)

        assert grid.get(3, 3, 3) == VOXEL_AIR

    def test_iron_bars_melt(self):
        """Iron bars at 1200 C should melt to air."""
        eb, grid, tp = _setup()
        grid.set(3, 3, 3, VOXEL_IRON_BARS)
        grid.set_metal_type(3, 3, 3, METAL_IRON)
        grid.temperature[3, 3, 3] = 1200.0

        tp._check_melting(grid.grid, grid.temperature)

        assert grid.get(3, 3, 3) == VOXEL_AIR


# ---------------------------------------------------------------------------
# Event publishing
# ---------------------------------------------------------------------------


class TestMeltingEvents:
    def test_metal_melted_event_published(self):
        """The 'metal_melted' event should be published when a block melts."""
        eb, grid, tp = _setup()
        grid.set(3, 3, 3, VOXEL_PIPE)
        grid.set_metal_type(3, 3, 3, METAL_COPPER)
        grid.temperature[3, 3, 3] = 800.0

        events = []
        eb.subscribe("metal_melted", lambda **kw: events.append(True))

        tp._check_melting(grid.grid, grid.temperature)

        assert len(events) == 1

    def test_no_event_when_nothing_melts(self):
        """No 'metal_melted' event when no blocks reach their melt temperature."""
        eb, grid, tp = _setup()
        grid.set(3, 3, 3, VOXEL_PIPE)
        grid.set_metal_type(3, 3, 3, METAL_IRON)
        grid.temperature[3, 3, 3] = 500.0  # Well below iron's 1200 C

        events = []
        eb.subscribe("metal_melted", lambda **kw: events.append(True))

        tp._check_melting(grid.grid, grid.temperature)

        assert len(events) == 0


# ---------------------------------------------------------------------------
# Multiple simultaneous melts
# ---------------------------------------------------------------------------


class TestMultipleMelts:
    def test_multiple_blocks_melt_simultaneously(self):
        """Two blocks at their melt temperatures should both become air."""
        eb, grid, tp = _setup()

        # Copper pipe at 800 C
        grid.set(2, 2, 2, VOXEL_PIPE)
        grid.set_metal_type(2, 2, 2, METAL_COPPER)
        grid.temperature[2, 2, 2] = 800.0

        # Gold door at 600 C
        grid.set(5, 5, 5, VOXEL_DOOR)
        grid.set_metal_type(5, 5, 5, METAL_GOLD)
        grid.temperature[5, 5, 5] = 600.0

        tp._check_melting(grid.grid, grid.temperature)

        assert grid.get(2, 2, 2) == VOXEL_AIR
        assert grid.get(5, 5, 5) == VOXEL_AIR
