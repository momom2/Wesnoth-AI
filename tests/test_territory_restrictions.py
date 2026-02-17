"""Tests for territory-based restrictions on digging and moving."""

import pytest

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.core.game_state import GameState
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.building.build_system import BuildSystem
from dungeon_builder.building.move_system import MoveSystem
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_STONE,
    VOXEL_DIRT,
    VOXEL_DOOR,
    DEFAULT_SEED,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _dig_setup():
    """Grid with a stone block that is NOT visible (not near claimed territory)."""
    bus = EventBus()
    grid = VoxelGrid(width=8, depth=8, height=4)
    grid.grid[:] = VOXEL_STONE
    # visible is all False by default
    bs = BuildSystem(bus, grid)
    return bus, grid, bs


def _move_setup():
    """Grid for move tests."""
    bus = EventBus()
    grid = VoxelGrid(width=8, depth=8, height=4)
    grid.grid[:] = VOXEL_STONE
    # visible and claimed are all False by default
    gs = GameState(DEFAULT_SEED)
    gs.event_bus = bus
    ms = MoveSystem(bus, grid, gs)
    return bus, grid, ms


# ── Digging territory restrictions ───────────────────────────────────


class TestDigTerritoryRestrictions:
    def test_invisible_block_becomes_pending_dig(self):
        """Digging a non-visible block goes to pending_digs (not rejected)."""
        bus, grid, bs = _dig_setup()
        # Block at (4,4,2) is stone but not visible

        assert bs.queue_dig(4, 4, 2) is True
        assert len(bs.pending_digs) == 1
        assert len(bs.dig_queue) == 0  # NOT in dig_queue yet

    def test_can_dig_visible_block(self):
        """Digging a visible block (border of claimed territory) succeeds."""
        bus, grid, bs = _dig_setup()
        grid.visible[4, 4, 2] = True

        assert bs.queue_dig(4, 4, 2) is True
        assert len(bs.dig_queue) == 1

    def test_dig_border_block_adjacent_to_claimed_air(self):
        """A solid block next to a claimed air cell is visible and diggable."""
        bus, grid, bs = _dig_setup()
        # Carve air cell and mark it claimed
        grid.grid[4, 4, 2] = VOXEL_AIR
        grid.claimed[4, 4, 2] = True
        # The adjacent stone block should be visible (border)
        grid.visible[5, 4, 2] = True

        assert bs.queue_dig(5, 4, 2) is True

    def test_dig_deeply_buried_block_becomes_pending(self):
        """A solid block far from any claimed air goes to pending_digs."""
        bus, grid, bs = _dig_setup()
        # Everything is stone, nothing is claimed or visible

        assert bs.queue_dig(0, 0, 0) is True
        assert len(bs.pending_digs) == 1

    def test_left_click_dig_outside_territory_goes_pending(self):
        """Left-click dig on non-visible block goes to pending_digs."""
        bus, grid, bs = _dig_setup()

        bus.publish("voxel_left_clicked", x=4, y=4, z=2, mode="dig")
        assert len(bs.dig_queue) == 0
        assert len(bs.pending_digs) == 1


# ── Pick up territory restrictions ───────────────────────────────────


class TestPickUpTerritoryRestrictions:
    def test_cannot_pick_up_outside_territory(self):
        """Picking up a loose block outside visible territory is rejected."""
        bus, grid, ms = _move_setup()
        grid.grid[4, 4, 2] = VOXEL_STONE
        grid.loose[4, 4, 2] = True
        # visible is False

        errors = []
        bus.subscribe("error_message", lambda **kw: errors.append(kw))

        assert ms.pick_up(4, 4, 2) is False
        assert len(errors) == 1
        assert "territory" in errors[0]["text"].lower()

    def test_can_pick_up_in_territory(self):
        """Picking up a loose visible block succeeds."""
        bus, grid, ms = _move_setup()
        grid.grid[4, 4, 2] = VOXEL_STONE
        grid.loose[4, 4, 2] = True
        grid.visible[4, 4, 2] = True

        assert ms.pick_up(4, 4, 2) is True
        assert ms.held_materials == {VOXEL_STONE: 1}


# ── Drop territory restrictions ──────────────────────────────────────


class TestDropTerritoryRestrictions:
    def test_cannot_drop_on_unclaimed_air(self):
        """Dropping onto an unclaimed air cell is rejected."""
        bus, grid, ms = _move_setup()
        grid.grid[4, 4, 2] = VOXEL_STONE
        grid.loose[4, 4, 2] = True
        grid.visible[4, 4, 2] = True
        ms.pick_up(4, 4, 2)

        # Target is air but not claimed
        grid.grid[3, 3, 2] = VOXEL_AIR

        errors = []
        bus.subscribe("error_message", lambda **kw: errors.append(kw))

        assert ms.drop(3, 3, 2) is False
        assert len(errors) == 1
        assert "territory" in errors[0]["text"].lower()

    def test_can_drop_on_claimed_air(self):
        """Dropping onto a claimed air cell succeeds."""
        bus, grid, ms = _move_setup()
        grid.grid[4, 4, 2] = VOXEL_STONE
        grid.loose[4, 4, 2] = True
        grid.visible[4, 4, 2] = True
        ms.pick_up(4, 4, 2)

        grid.grid[3, 3, 2] = VOXEL_AIR
        grid.claimed[3, 3, 2] = True

        assert ms.drop(3, 3, 2) is True
        assert grid.get(3, 3, 2) == VOXEL_STONE

    def test_drop_on_solid_rejected(self):
        """Dropping onto a solid block is rejected (no auto-crafting)."""
        bus, grid, ms = _move_setup()
        grid.grid[4, 4, 2] = VOXEL_STONE
        grid.loose[4, 4, 2] = True
        grid.visible[4, 4, 2] = True
        ms.pick_up(4, 4, 2)

        # Target is stone and visible
        grid.grid[0, 0, 0] = VOXEL_STONE
        grid.visible[0, 0, 0] = True

        errors = []
        bus.subscribe("error_message", lambda **kw: errors.append(kw))

        assert ms.drop(0, 0, 0) is False
        assert len(errors) == 1
        assert "solid" in errors[0]["text"].lower()


# ── Door toggle territory restrictions ────────────────────────────────


class TestDoorToggleTerritoryRestrictions:
    def test_cannot_toggle_door_outside_territory(self):
        """Non-visible door cannot be toggled."""
        bus, grid, ms = _move_setup()
        grid.grid[4, 4, 2] = VOXEL_DOOR
        grid.loose[4, 4, 2] = False
        grid.block_state[4, 4, 2] = 1  # closed
        # visible is False

        bus.publish("voxel_left_clicked", x=4, y=4, z=2, mode="move")

        # Door should NOT have been toggled
        assert grid.get_block_state(4, 4, 2) == 1

    def test_can_toggle_visible_door(self):
        """Visible door can be toggled."""
        bus, grid, ms = _move_setup()
        grid.grid[4, 4, 2] = VOXEL_DOOR
        grid.loose[4, 4, 2] = False
        grid.block_state[4, 4, 2] = 1  # closed
        grid.visible[4, 4, 2] = True

        events = []
        bus.subscribe("door_toggled", lambda **kw: events.append(kw))

        bus.publish("voxel_left_clicked", x=4, y=4, z=2, mode="move")

        assert grid.get_block_state(4, 4, 2) == 0  # now open
        assert len(events) == 1
