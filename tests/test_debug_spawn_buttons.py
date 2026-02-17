"""Tests for debug spawn buttons (temporary testing feature).

Covers:
- IntruderAI subscribes to debug spawn events
- Publishing debug events triggers party spawning
- HUD has spawn buttons (source inspection)
"""

from __future__ import annotations

import inspect

import pytest

from dungeon_builder.config import (
    VOXEL_AIR, SURFACE_Z, CORE_X, CORE_Y, CORE_Z,
    UNDERWORLD_SPAWN_Z_MIN, UNDERWORLD_SPAWN_Z_MAX,
)
from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.world.pathfinding import AStarPathfinder
from dungeon_builder.dungeon_core.core import DungeonCore
from dungeon_builder.utils.rng import SeededRNG
from dungeon_builder.intruders.decision import IntruderAI


# ── Helpers ────────────────────────────────────────────────────────────


def _make_grid_with_air_edges() -> VoxelGrid:
    """Create a grid with air cells at surface edges and deep edges."""
    grid = VoxelGrid()
    # Surface edge air for surface spawning
    for x in range(grid.width):
        grid.grid[x, 0, SURFACE_Z] = VOXEL_AIR
        grid.grid[x, grid.depth - 1, SURFACE_Z] = VOXEL_AIR
    # Deep edge air for underworld spawning
    for z in range(UNDERWORLD_SPAWN_Z_MIN, UNDERWORLD_SPAWN_Z_MAX + 1):
        for x in range(grid.width):
            grid.grid[x, 0, z] = VOXEL_AIR
            grid.grid[x, grid.depth - 1, z] = VOXEL_AIR
    # Air path from surface to core so pathfinding works
    for y in range(grid.depth):
        grid.grid[CORE_X, y, SURFACE_Z] = VOXEL_AIR
    return grid


# ── Test: Debug spawn event subscriptions ──────────────────────────────


class TestDebugSpawnSubscriptions:
    """IntruderAI should subscribe to debug spawn events."""

    def test_subscribes_to_debug_spawn_party(self):
        """IntruderAI.__init__ should subscribe to 'debug_spawn_party'."""
        src = inspect.getsource(IntruderAI.__init__)
        assert "debug_spawn_party" in src

    def test_subscribes_to_debug_spawn_underworld(self):
        """IntruderAI.__init__ should subscribe to 'debug_spawn_underworld_party'."""
        src = inspect.getsource(IntruderAI.__init__)
        assert "debug_spawn_underworld_party" in src

    def test_debug_spawn_handler_exists(self):
        """IntruderAI should have a _on_debug_spawn_party method."""
        assert hasattr(IntruderAI, "_on_debug_spawn_party")

    def test_debug_spawn_uw_handler_exists(self):
        """IntruderAI should have a _on_debug_spawn_uw method."""
        assert hasattr(IntruderAI, "_on_debug_spawn_uw")


# ── Test: Debug spawn via event bus ────────────────────────────────────


class TestDebugSpawnViaEventBus:
    """Publishing debug events should trigger spawning."""

    def test_debug_spawn_surface_party(self):
        """Publishing debug_spawn_party should create surface intruders."""
        event_bus = EventBus()
        grid = _make_grid_with_air_edges()
        pathfinder = AStarPathfinder(grid)
        core = DungeonCore(event_bus, CORE_X, CORE_Y, CORE_Z, hp=100)
        rng = SeededRNG(42)
        ai = IntruderAI(event_bus, grid, pathfinder, core, rng)

        assert len(ai.intruders) == 0
        assert len(ai.parties) == 0

        event_bus.publish("debug_spawn_party")

        assert len(ai.intruders) > 0
        assert len(ai.parties) == 1
        # All should be surface intruders
        for i in ai.intruders:
            assert not i.is_underworlder

    def test_debug_spawn_underworld_party(self):
        """Publishing debug_spawn_underworld_party should create underworlders."""
        event_bus = EventBus()
        grid = _make_grid_with_air_edges()
        pathfinder = AStarPathfinder(grid)
        core = DungeonCore(event_bus, CORE_X, CORE_Y, CORE_Z, hp=100)
        rng = SeededRNG(42)
        ai = IntruderAI(event_bus, grid, pathfinder, core, rng)

        assert len(ai.intruders) == 0
        assert len(ai._underworld_parties) == 0

        event_bus.publish("debug_spawn_underworld_party")

        assert len(ai.intruders) > 0
        assert len(ai._underworld_parties) == 1
        # All should be underworlders
        for i in ai.intruders:
            assert i.is_underworlder

    def test_debug_spawn_enables_spawning(self):
        """Debug spawn should enable spawning_enabled flag."""
        event_bus = EventBus()
        grid = _make_grid_with_air_edges()
        pathfinder = AStarPathfinder(grid)
        core = DungeonCore(event_bus, CORE_X, CORE_Y, CORE_Z, hp=100)
        rng = SeededRNG(42)
        ai = IntruderAI(event_bus, grid, pathfinder, core, rng)

        assert ai.spawning_enabled is False
        event_bus.publish("debug_spawn_party")
        assert ai.spawning_enabled is True

    def test_debug_spawn_multiple_times(self):
        """Can spawn multiple parties via repeated debug events."""
        event_bus = EventBus()
        grid = _make_grid_with_air_edges()
        pathfinder = AStarPathfinder(grid)
        core = DungeonCore(event_bus, CORE_X, CORE_Y, CORE_Z, hp=100)
        rng = SeededRNG(42)
        ai = IntruderAI(event_bus, grid, pathfinder, core, rng)

        event_bus.publish("debug_spawn_party")
        event_bus.publish("debug_spawn_party")

        assert len(ai.parties) == 2


# ── Test: HUD has spawn buttons (source inspection) ───────────────────


class TestHUDSpawnButtons:
    """HUD should have debug spawn buttons."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        from dungeon_builder.ui.hud import HUD
        self.class_src = inspect.getsource(HUD)

    def test_surface_spawn_button_exists(self):
        """HUD should create a 'Spawn Surface' button."""
        assert "Spawn Surface" in self.class_src

    def test_underworld_spawn_button_exists(self):
        """HUD should create a 'Spawn Underworld' button."""
        assert "Spawn Underworld" in self.class_src

    def test_surface_button_publishes_event(self):
        """Surface button should publish debug_spawn_party event."""
        assert "debug_spawn_party" in self.class_src

    def test_underworld_button_publishes_event(self):
        """Underworld button should publish debug_spawn_underworld_party event."""
        assert "debug_spawn_underworld_party" in self.class_src
