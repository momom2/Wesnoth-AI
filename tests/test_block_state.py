"""Tests for block_state array and door toggle behavior."""

import numpy as np

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.core.game_state import GameState
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.building.move_system import MoveSystem
from dungeon_builder.building.crafting_system import CraftingSystem
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_STONE,
    VOXEL_DOOR,
    VOXEL_SPIKE,
    VOXEL_ENCHANTED_METAL,
    DEFAULT_SEED,
)


def _setup(width=8, depth=8, height=8):
    bus = EventBus()
    grid = VoxelGrid(width=width, depth=depth, height=height)
    # Mark all blocks visible and claimed so territory checks don't interfere
    grid.visible[:] = True
    grid.claimed[:] = True
    gs = GameState(DEFAULT_SEED)
    gs.event_bus = bus
    ms = MoveSystem(bus, grid, gs)
    cs = CraftingSystem(bus, grid, ms, gs)
    gs.move_system = ms
    return bus, grid, ms, cs


# ---------------------------------------------------------------------------
# block_state array tests
# ---------------------------------------------------------------------------


class TestBlockStateDefault:
    def test_block_state_default_zero(self):
        """All block states default to 0."""
        grid = VoxelGrid(width=4, depth=4, height=4)
        assert np.all(grid.block_state == 0)


class TestBlockStateSetGet:
    def test_block_state_set_get(self):
        """Can set and get block_state values."""
        grid = VoxelGrid(width=4, depth=4, height=4)
        grid.set_block_state(1, 2, 3, 42)
        assert grid.get_block_state(1, 2, 3) == 42

    def test_block_state_out_of_bounds(self):
        """Out-of-bounds get returns 0, set is no-op."""
        grid = VoxelGrid(width=4, depth=4, height=4)
        assert grid.get_block_state(100, 0, 0) == 0
        grid.set_block_state(100, 0, 0, 5)  # no crash


class TestBlockStateClearedOnSet:
    def test_block_state_cleared_on_set(self):
        """block_state resets to 0 when voxel type changes."""
        bus = EventBus()
        grid = VoxelGrid(width=4, depth=4, height=4)
        grid.grid[2, 2, 2] = VOXEL_DOOR
        grid.set_block_state(2, 2, 2, 1)
        assert grid.get_block_state(2, 2, 2) == 1

        # Change the voxel type
        grid.set(2, 2, 2, VOXEL_STONE)
        assert grid.get_block_state(2, 2, 2) == 0


class TestBlockStateMarksDirty:
    def test_block_state_marks_chunk_dirty(self):
        """Setting block_state marks the chunk dirty for re-render."""
        grid = VoxelGrid(width=16, depth=16, height=4)
        grid.pop_dirty_chunks()  # clear initial dirty state

        grid.set_block_state(1, 2, 3, 1)

        dirty = grid.pop_dirty_chunks()
        assert len(dirty) > 0
        # Should include the chunk containing (1, 2, 3)
        cx, cy = 1 // 16, 2 // 16
        assert (cx, cy, 3) in dirty


# ---------------------------------------------------------------------------
# Door toggle tests
# ---------------------------------------------------------------------------


class TestDoorToggle:
    def test_door_toggle_closed_to_open(self):
        """Left-click on closed door opens it (state 1 -> 0)."""
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_DOOR
        grid.set_block_state(4, 4, 4, 1)  # closed

        events = []
        bus.subscribe("door_toggled", lambda **kw: events.append(kw))

        bus.publish("voxel_left_clicked", x=4, y=4, z=4, mode="move")

        assert grid.get_block_state(4, 4, 4) == 0  # now open
        assert len(events) == 1
        assert events[0]["state"] == 0

    def test_door_toggle_open_to_closed(self):
        """Left-click on open door closes it (state 0 -> 1)."""
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_DOOR
        grid.set_block_state(4, 4, 4, 0)  # open

        bus.publish("voxel_left_clicked", x=4, y=4, z=4, mode="move")

        assert grid.get_block_state(4, 4, 4) == 1  # now closed

    def test_door_toggle_does_not_pick_up(self):
        """Clicking a non-loose door toggles it instead of picking up."""
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_DOOR
        grid.set_block_state(4, 4, 4, 1)  # closed
        grid.loose[4, 4, 4] = False

        bus.publish("voxel_left_clicked", x=4, y=4, z=4, mode="move")

        # Door still present, not picked up
        assert grid.get(4, 4, 4) == VOXEL_DOOR
        assert ms.held_materials == {}

    def test_loose_door_can_be_picked_up(self):
        """A loose door is picked up normally (not toggled)."""
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_DOOR
        grid.loose[4, 4, 4] = True

        bus.publish("voxel_left_clicked", x=4, y=4, z=4, mode="move")

        # Door was picked up
        assert grid.get(4, 4, 4) == VOXEL_AIR
        assert ms.has_material(VOXEL_DOOR)
        assert ms.held_materials[VOXEL_DOOR] == 1


class TestDoorCraftState:
    def test_crafted_door_starts_closed(self):
        """Crafting a door sets block_state=1 (closed)."""
        bus, grid, ms, cs = _setup()
        grid.grid[3, 4, 4] = VOXEL_STONE
        grid.grid[5, 4, 4] = VOXEL_STONE
        ms.held_materials = {VOXEL_ENCHANTED_METAL: 1}

        cs._current_z = 4
        bus.publish("craft_recipe_selected", recipe_name="Door")
        bus.publish("craft_at_position", x=4, y=4, z=4)

        assert grid.get(4, 4, 4) == VOXEL_DOOR
        assert grid.get_block_state(4, 4, 4) == 1
