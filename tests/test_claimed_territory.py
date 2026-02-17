"""Tests for claimed territory flood-fill and fog-of-war visibility."""

import numpy as np
import pytest

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.world.claimed_territory import ClaimedTerritorySystem
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_STONE,
    VOXEL_CORE,
    VOXEL_WATER,
    VOXEL_LAVA,
    CORE_X,
    CORE_Y,
    CORE_Z,
    CLAIMED_TICK_INTERVAL,
    FOG_COLOR,
)


def _setup():
    """Create a small grid with stone everywhere except a core block."""
    bus = EventBus()
    grid = VoxelGrid(width=16, depth=16, height=5)
    grid.grid[:] = VOXEL_STONE
    # Place core at a sensible location within this small grid
    cx, cy, cz = min(CORE_X, 8), min(CORE_Y, 8), min(CORE_Z, 2)
    grid.grid[cx, cy, cz] = VOXEL_CORE
    return bus, grid, cx, cy, cz


# ── Core with no air ─────────────────────────────────────────────────


class TestNoAir:
    def test_no_air_neighbors_means_no_claimed(self):
        bus, grid, cx, cy, cz = _setup()
        # Core surrounded by stone — no air at all
        sys = ClaimedTerritorySystem(bus, grid, cx, cy, cz)
        assert not np.any(grid.claimed)

    def test_core_always_visible_even_without_air(self):
        bus, grid, cx, cy, cz = _setup()
        sys = ClaimedTerritorySystem(bus, grid, cx, cy, cz)
        assert grid.is_visible(cx, cy, cz)


# ── Room around core ─────────────────────────────────────────────────


class TestRoomAroundCore:
    def test_air_room_claimed(self):
        bus, grid, cx, cy, cz = _setup()
        # Carve a 3x3 air room around core
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = cx + dx, cy + dy
                if grid.in_bounds(nx, ny, cz) and (dx, dy) != (0, 0):
                    grid.grid[nx, ny, cz] = VOXEL_AIR
        sys = ClaimedTerritorySystem(bus, grid, cx, cy, cz)
        # All air cells should be claimed
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = cx + dx, cy + dy
                if (dx, dy) != (0, 0) and grid.in_bounds(nx, ny, cz):
                    assert grid.is_claimed(nx, ny, cz), f"({nx},{ny},{cz}) not claimed"

    def test_walls_adjacent_to_room_are_visible(self):
        bus, grid, cx, cy, cz = _setup()
        # Carve 3x3 air room
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = cx + dx, cy + dy
                if grid.in_bounds(nx, ny, cz) and (dx, dy) != (0, 0):
                    grid.grid[nx, ny, cz] = VOXEL_AIR
        sys = ClaimedTerritorySystem(bus, grid, cx, cy, cz)
        # Stone walls 1 cell beyond the air room should be visible
        wall_x = cx + 2
        if grid.in_bounds(wall_x, cy, cz):
            assert grid.is_visible(wall_x, cy, cz)

    def test_distant_stone_not_visible(self):
        bus, grid, cx, cy, cz = _setup()
        # Carve single air cell next to core
        grid.grid[cx + 1, cy, cz] = VOXEL_AIR
        sys = ClaimedTerritorySystem(bus, grid, cx, cy, cz)
        # Stone far away should not be visible
        far_x = cx + 5
        if grid.in_bounds(far_x, cy, cz):
            assert not grid.is_visible(far_x, cy, cz)


# ── Connectivity ──────────────────────────────────────────────────────


class TestConnectivity:
    def test_disconnected_pocket_not_claimed(self):
        bus, grid, cx, cy, cz = _setup()
        # Air next to core
        grid.grid[cx + 1, cy, cz] = VOXEL_AIR
        # Disconnected air pocket far away (no air path to core)
        grid.grid[0, 0, 0] = VOXEL_AIR
        grid.grid[1, 0, 0] = VOXEL_AIR

        sys = ClaimedTerritorySystem(bus, grid, cx, cy, cz)
        assert grid.is_claimed(cx + 1, cy, cz)
        assert not grid.is_claimed(0, 0, 0)
        assert not grid.is_claimed(1, 0, 0)

    def test_corridor_connects_chambers(self):
        bus, grid, cx, cy, cz = _setup()
        # Chamber 1: around core
        grid.grid[cx + 1, cy, cz] = VOXEL_AIR
        # Corridor
        grid.grid[cx + 2, cy, cz] = VOXEL_AIR
        grid.grid[cx + 3, cy, cz] = VOXEL_AIR
        # Chamber 2
        grid.grid[cx + 4, cy, cz] = VOXEL_AIR
        if grid.in_bounds(cx + 4, cy + 1, cz):
            grid.grid[cx + 4, cy + 1, cz] = VOXEL_AIR

        sys = ClaimedTerritorySystem(bus, grid, cx, cy, cz)
        assert grid.is_claimed(cx + 4, cy, cz)

    def test_blocking_corridor_disconnects(self):
        bus, grid, cx, cy, cz = _setup()
        # Air path: core -> cx+1 -> cx+2 -> cx+3
        grid.grid[cx + 1, cy, cz] = VOXEL_AIR
        grid.grid[cx + 2, cy, cz] = VOXEL_AIR
        grid.grid[cx + 3, cy, cz] = VOXEL_AIR

        sys = ClaimedTerritorySystem(bus, grid, cx, cy, cz)
        assert grid.is_claimed(cx + 3, cy, cz)

        # Block the corridor
        grid.grid[cx + 2, cy, cz] = VOXEL_STONE
        sys.recompute()
        assert not grid.is_claimed(cx + 3, cy, cz)
        # Near side still claimed
        assert grid.is_claimed(cx + 1, cy, cz)


# ── Vertical ──────────────────────────────────────────────────────────


class TestVertical:
    def test_vertical_shaft_claimed(self):
        bus, grid, cx, cy, cz = _setup()
        # Air below core (shaft going deeper)
        grid.grid[cx, cy + 1, cz] = VOXEL_AIR  # air next to core
        for z in range(cz, min(cz + 3, grid.height)):
            grid.grid[cx, cy + 1, z] = VOXEL_AIR

        sys = ClaimedTerritorySystem(bus, grid, cx, cy, cz)
        for z in range(cz, min(cz + 3, grid.height)):
            assert grid.is_claimed(cx, cy + 1, z), f"z={z} not claimed"


# ── Tick-based recompute ──────────────────────────────────────────────


class TestTickRecompute:
    def test_recompute_on_tick_interval(self):
        bus, grid, cx, cy, cz = _setup()
        grid.grid[cx + 1, cy, cz] = VOXEL_AIR
        sys = ClaimedTerritorySystem(bus, grid, cx, cy, cz)
        assert grid.is_claimed(cx + 1, cy, cz)

        # Add new air at cx+2 (not yet claimed since no recompute)
        grid.grid[cx + 2, cy, cz] = VOXEL_AIR

        # Wrong tick — no recompute
        bus.publish("tick", tick=1)
        assert not grid.is_claimed(cx + 2, cy, cz)

        # Right tick — recompute happens
        bus.publish("tick", tick=CLAIMED_TICK_INTERVAL)
        assert grid.is_claimed(cx + 2, cy, cz)


# ── Event publishing ──────────────────────────────────────────────────


class TestEvents:
    def test_event_fires_on_change(self):
        bus, grid, cx, cy, cz = _setup()
        sys = ClaimedTerritorySystem(bus, grid, cx, cy, cz)
        # No claimed territory yet

        events = []
        bus.subscribe("claimed_territory_changed", lambda **kw: events.append(True))

        # Add air and recompute — territory changes
        grid.grid[cx + 1, cy, cz] = VOXEL_AIR
        sys.recompute()
        assert len(events) == 1

    def test_no_event_when_unchanged(self):
        bus, grid, cx, cy, cz = _setup()
        grid.grid[cx + 1, cy, cz] = VOXEL_AIR
        sys = ClaimedTerritorySystem(bus, grid, cx, cy, cz)

        events = []
        bus.subscribe("claimed_territory_changed", lambda **kw: events.append(True))

        # Recompute with no changes
        sys.recompute()
        assert len(events) == 0


# ── Core block visibility ─────────────────────────────────────────────


class TestCoreBlockVisibility:
    """Core placed as VOXEL_CORE renders correctly in claimed territory."""

    def test_core_block_is_visible(self):
        """Core block (solid) with air neighbors is visible."""
        bus, grid, cx, cy, cz = _setup()
        # Carve air around core (simulates _carve_initial_dungeon)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if (dx, dy) != (0, 0) and grid.in_bounds(cx + dx, cy + dy, cz):
                    grid.grid[cx + dx, cy + dy, cz] = VOXEL_AIR
        sys = ClaimedTerritorySystem(bus, grid, cx, cy, cz)
        assert grid.is_visible(cx, cy, cz), "Core block should be visible"

    def test_core_block_not_claimed(self):
        """Core block is solid, so it should not be claimed (only air/water are)."""
        bus, grid, cx, cy, cz = _setup()
        grid.grid[cx + 1, cy, cz] = VOXEL_AIR
        sys = ClaimedTerritorySystem(bus, grid, cx, cy, cz)
        assert not grid.is_claimed(cx, cy, cz), "Solid core should not be claimed"

    def test_air_above_core_is_claimed(self):
        """Air above the core (headroom) should be claimed."""
        bus, grid, cx, cy, cz = _setup()
        # Air next to core at same level
        grid.grid[cx + 1, cy, cz] = VOXEL_AIR
        # Air above core
        if grid.in_bounds(cx, cy, cz - 1):
            grid.grid[cx, cy, cz - 1] = VOXEL_AIR
            grid.grid[cx + 1, cy, cz - 1] = VOXEL_AIR
            sys = ClaimedTerritorySystem(bus, grid, cx, cy, cz)
            assert grid.is_claimed(cx, cy, cz - 1), "Air above core should be claimed"


# ── Water propagation ──────────────────────────────────────────────────


class TestWaterPropagation:
    """Tests for claiming through water (but not lava)."""

    def test_water_corridor_claimed(self):
        """Water corridor between core and air pocket is claimed."""
        bus, grid, cx, cy, cz = _setup()
        # Air next to core
        grid.grid[cx + 1, cy, cz] = VOXEL_AIR
        # Water corridor
        grid.grid[cx + 2, cy, cz] = VOXEL_WATER
        grid.grid[cx + 3, cy, cz] = VOXEL_WATER
        # Air pocket beyond water
        grid.grid[cx + 4, cy, cz] = VOXEL_AIR

        sys = ClaimedTerritorySystem(bus, grid, cx, cy, cz)
        assert grid.is_claimed(cx + 2, cy, cz)
        assert grid.is_claimed(cx + 3, cy, cz)
        assert grid.is_claimed(cx + 4, cy, cz)

    def test_lava_blocks_propagation(self):
        """Lava corridor blocks territory propagation."""
        bus, grid, cx, cy, cz = _setup()
        # Air next to core
        grid.grid[cx + 1, cy, cz] = VOXEL_AIR
        # Lava barrier
        grid.grid[cx + 2, cy, cz] = VOXEL_LAVA
        # Air beyond lava
        grid.grid[cx + 3, cy, cz] = VOXEL_AIR

        sys = ClaimedTerritorySystem(bus, grid, cx, cy, cz)
        assert grid.is_claimed(cx + 1, cy, cz)
        assert not grid.is_claimed(cx + 3, cy, cz)

    def test_mixed_air_water_corridor(self):
        """Alternating air and water cells are all claimed."""
        bus, grid, cx, cy, cz = _setup()
        grid.grid[cx + 1, cy, cz] = VOXEL_AIR
        grid.grid[cx + 2, cy, cz] = VOXEL_WATER
        grid.grid[cx + 3, cy, cz] = VOXEL_AIR
        grid.grid[cx + 4, cy, cz] = VOXEL_WATER

        sys = ClaimedTerritorySystem(bus, grid, cx, cy, cz)
        for dx in range(1, 5):
            assert grid.is_claimed(cx + dx, cy, cz), f"dx={dx} not claimed"

    def test_water_not_visible(self):
        """Water cells that are claimed should NOT be marked visible."""
        bus, grid, cx, cy, cz = _setup()
        grid.grid[cx + 1, cy, cz] = VOXEL_AIR
        grid.grid[cx + 2, cy, cz] = VOXEL_WATER

        sys = ClaimedTerritorySystem(bus, grid, cx, cy, cz)
        assert grid.is_claimed(cx + 2, cy, cz)
        assert not grid.is_visible(cx + 2, cy, cz)

    def test_stone_adjacent_to_water_is_visible(self):
        """Stone block adjacent to claimed water should be visible."""
        bus, grid, cx, cy, cz = _setup()
        grid.grid[cx + 1, cy, cz] = VOXEL_AIR
        grid.grid[cx + 2, cy, cz] = VOXEL_WATER
        # cx+3 remains stone

        sys = ClaimedTerritorySystem(bus, grid, cx, cy, cz)
        assert grid.is_visible(cx + 3, cy, cz)

    def test_submerged_chamber_claimed(self):
        """Fully water-filled room connected to core is claimed."""
        bus, grid, cx, cy, cz = _setup()
        grid.grid[cx + 1, cy, cz] = VOXEL_AIR  # entry
        # 2x2 water chamber
        for dx in (2, 3):
            for dy in (0, 1):
                grid.grid[cx + dx, cy + dy, cz] = VOXEL_WATER

        sys = ClaimedTerritorySystem(bus, grid, cx, cy, cz)
        for dx in (2, 3):
            for dy in (0, 1):
                assert grid.is_claimed(cx + dx, cy + dy, cz), (
                    f"({cx+dx},{cy+dy},{cz}) not claimed"
                )

    def test_water_adjacent_to_core_seeds_flood_fill(self):
        """Water directly next to the core seeds the flood-fill."""
        bus, grid, cx, cy, cz = _setup()
        grid.grid[cx + 1, cy, cz] = VOXEL_WATER
        grid.grid[cx + 2, cy, cz] = VOXEL_AIR

        sys = ClaimedTerritorySystem(bus, grid, cx, cy, cz)
        assert grid.is_claimed(cx + 1, cy, cz)
        assert grid.is_claimed(cx + 2, cy, cz)
