"""Tests for pressure plate chain activation logic in the decision engine.

Covers: IntruderAI._activate_pressure_plate() and the pressure plate trigger
within _move_to().  Verifies that stepping on a pressure plate (or grabbing
gold bait) triggers adjacent spikes, doors, and floodgates as expected.
"""

from __future__ import annotations

import pytest

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.world.pathfinding import AStarPathfinder
from dungeon_builder.dungeon_core.core import DungeonCore
from dungeon_builder.utils.rng import SeededRNG
from dungeon_builder.intruders.decision import IntruderAI
from dungeon_builder.intruders.agent import Intruder, IntruderState
from dungeon_builder.intruders.archetypes import (
    VANGUARD,
    SHADOWBLADE,
    IntruderObjective,
)
from dungeon_builder.intruders.personal_map import PersonalMap
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_STONE,
    VOXEL_SPIKE,
    VOXEL_DOOR,
    VOXEL_FLOODGATE,
    VOXEL_PRESSURE_PLATE,
    VOXEL_GOLD_BAIT,
    PRESSURE_PLATE_TRIGGER_RANGE,
)


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_ai(width=16, depth=16, height=5, core_pos=(7, 7, 1), seed=42):
    """Create an IntruderAI with a small stone-filled grid and an air corridor."""
    bus = EventBus()
    grid = VoxelGrid(width=width, depth=depth, height=height)
    # Fill entirely with stone
    grid.grid[:] = VOXEL_STONE
    # Carve an air corridor along y=0, z=1
    for x in range(width):
        grid.set(x, 0, 1, VOXEL_AIR)
    # Also open y=1, z=1 for cross-corridor tests (door adjacency, etc.)
    for x in range(width):
        grid.set(x, 1, 1, VOXEL_AIR)
    grid.visible[:] = True
    grid.claimed[:] = True
    pf = AStarPathfinder(grid)
    core = DungeonCore(bus, *core_pos, hp=100)
    rng = SeededRNG(seed)
    ai = IntruderAI(bus, grid, pf, core, rng)
    return ai, bus, grid


def _make_intruder(x, y, z, pmap=None, archetype=VANGUARD):
    """Create a minimal intruder for testing."""
    if pmap is None:
        pmap = PersonalMap()
    intruder = Intruder(
        intruder_id=1,
        x=x, y=y, z=z,
        archetype=archetype,
        objective=IntruderObjective.DESTROY_CORE,
        personal_map=pmap,
    )
    intruder.state = IntruderState.ADVANCING
    return intruder


# ═══════════════════════════════════════════════════════════════════════
#  Pressure Plate Chain Tests
# ═══════════════════════════════════════════════════════════════════════


class TestPressurePlateToSpike:
    """Plate at (3,0,1) adjacent to spike at (4,0,1): spike should extend."""

    def test_adjacent_spike_extends(self):
        ai, bus, grid = _make_ai()
        # Place pressure plate and spike
        grid.set(3, 0, 1, VOXEL_PRESSURE_PLATE)
        grid.set_block_state(3, 0, 1, 0)  # unarmed
        grid.set(4, 0, 1, VOXEL_SPIKE)
        grid.set_block_state(4, 0, 1, 0)  # retracted

        intruder = _make_intruder(2, 0, 1)
        ai.intruders.append(intruder)

        # Step onto the pressure plate
        ai._move_to(intruder, (3, 0, 1))

        assert int(grid.block_state[4, 0, 1]) == 1, "Spike should be extended"


class TestPressurePlateToDoor:
    """Plate at (3,0,1) adjacent to door at (3,1,1): door should close."""

    def test_adjacent_door_closes(self):
        ai, bus, grid = _make_ai()
        grid.set(3, 0, 1, VOXEL_PRESSURE_PLATE)
        grid.set_block_state(3, 0, 1, 0)
        grid.set(3, 1, 1, VOXEL_DOOR)
        grid.set_block_state(3, 1, 1, 0)  # open

        intruder = _make_intruder(2, 0, 1)
        ai.intruders.append(intruder)

        ai._move_to(intruder, (3, 0, 1))

        assert int(grid.block_state[3, 1, 1]) == 1, "Door should be closed"


class TestPressurePlateFloodgateToggle:
    """Floodgate toggles open/closed each activation."""

    def test_floodgate_toggles_open_to_closed(self):
        ai, bus, grid = _make_ai()
        grid.set(3, 0, 1, VOXEL_PRESSURE_PLATE)
        grid.set_block_state(3, 0, 1, 0)
        grid.set(4, 0, 1, VOXEL_FLOODGATE)
        grid.set_block_state(4, 0, 1, 0)  # open

        intruder = _make_intruder(2, 0, 1)
        ai.intruders.append(intruder)

        ai._move_to(intruder, (3, 0, 1))

        assert int(grid.block_state[4, 0, 1]) == 1, "Floodgate should toggle to closed"

    def test_floodgate_toggles_closed_to_open(self):
        ai, bus, grid = _make_ai()
        grid.set(3, 0, 1, VOXEL_PRESSURE_PLATE)
        grid.set_block_state(3, 0, 1, 0)
        grid.set(4, 0, 1, VOXEL_FLOODGATE)
        grid.set_block_state(4, 0, 1, 1)  # closed

        intruder = _make_intruder(2, 0, 1)
        ai.intruders.append(intruder)

        ai._move_to(intruder, (3, 0, 1))

        assert int(grid.block_state[4, 0, 1]) == 0, "Floodgate should toggle to open"


class TestPressurePlateOnlyTriggersOnce:
    """A plate with block_state=1 (already triggered) should not re-fire."""

    def test_second_step_does_not_retrigger(self):
        ai, bus, grid = _make_ai()
        grid.set(3, 0, 1, VOXEL_PRESSURE_PLATE)
        grid.set_block_state(3, 0, 1, 0)  # armed
        grid.set(4, 0, 1, VOXEL_FLOODGATE)
        grid.set_block_state(4, 0, 1, 0)  # open

        intruder = _make_intruder(2, 0, 1)
        ai.intruders.append(intruder)

        # First step: triggers the plate
        ai._move_to(intruder, (3, 0, 1))
        assert int(grid.block_state[3, 0, 1]) == 1, "Plate should be armed"
        assert int(grid.block_state[4, 0, 1]) == 1, "Floodgate toggled to closed"

        # Move away and back — plate already triggered (state=1)
        ai._move_to(intruder, (2, 0, 1))
        # Reset floodgate to open manually to detect re-trigger
        grid.set_block_state(4, 0, 1, 0)
        ai._move_to(intruder, (3, 0, 1))

        # Floodgate should remain open because the plate is already state=1
        assert int(grid.block_state[4, 0, 1]) == 0, (
            "Floodgate should NOT toggle again — plate already triggered"
        )


class TestPressurePlateMultiTarget:
    """A plate adjacent to both a spike and a door activates both."""

    def test_spike_and_door_both_activate(self):
        ai, bus, grid = _make_ai()
        grid.set(3, 0, 1, VOXEL_PRESSURE_PLATE)
        grid.set_block_state(3, 0, 1, 0)
        # Spike to the right
        grid.set(4, 0, 1, VOXEL_SPIKE)
        grid.set_block_state(4, 0, 1, 0)
        # Door to the front (y+1)
        grid.set(3, 1, 1, VOXEL_DOOR)
        grid.set_block_state(3, 1, 1, 0)

        intruder = _make_intruder(2, 0, 1)
        ai.intruders.append(intruder)

        ai._move_to(intruder, (3, 0, 1))

        assert int(grid.block_state[4, 0, 1]) == 1, "Spike should be extended"
        assert int(grid.block_state[3, 1, 1]) == 1, "Door should be closed"


class TestPressurePlateOutOfRange:
    """A spike 2 cells away from the plate should NOT be activated."""

    def test_spike_two_cells_away_not_triggered(self):
        ai, bus, grid = _make_ai()
        grid.set(3, 0, 1, VOXEL_PRESSURE_PLATE)
        grid.set_block_state(3, 0, 1, 0)
        # Spike at Manhattan distance 2 (two cells to the right)
        grid.set(5, 0, 1, VOXEL_SPIKE)
        grid.set_block_state(5, 0, 1, 0)

        intruder = _make_intruder(2, 0, 1)
        ai.intruders.append(intruder)

        ai._move_to(intruder, (3, 0, 1))

        assert int(grid.block_state[5, 0, 1]) == 0, (
            "Spike at distance 2 should NOT be triggered (range is 1)"
        )


class TestGoldBaitTriggersTraps:
    """Grabbing gold bait calls _activate_pressure_plate at the bait position."""

    def test_bait_grab_activates_adjacent_spike(self):
        ai, bus, grid = _make_ai()
        # Place gold bait and adjacent spike
        grid.set(5, 0, 1, VOXEL_GOLD_BAIT)
        grid.set(6, 0, 1, VOXEL_SPIKE)
        grid.set_block_state(6, 0, 1, 0)  # retracted

        # Use Shadowblade (greed > 0) so it will grab the bait
        pmap = PersonalMap()
        intruder = _make_intruder(4, 0, 1, pmap=pmap, archetype=SHADOWBLADE)
        ai.intruders.append(intruder)

        # Simulate the bait grab interaction completing:
        # Set up intruder as INTERACTING with "grab_bait" target at (5,0,1)
        intruder.state = IntruderState.INTERACTING
        intruder.interaction_type = "grab_bait"
        intruder.interaction_target = (5, 0, 1)
        intruder.interaction_ticks = 1  # will complete on next call

        ai._update_interacting(intruder)

        assert int(grid.block_state[6, 0, 1]) == 1, (
            "Adjacent spike should extend when gold bait is grabbed"
        )

    def test_bait_grab_activates_adjacent_floodgate(self):
        ai, bus, grid = _make_ai()
        grid.set(5, 0, 1, VOXEL_GOLD_BAIT)
        grid.set(5, 1, 1, VOXEL_FLOODGATE)
        grid.set_block_state(5, 1, 1, 0)  # open

        pmap = PersonalMap()
        intruder = _make_intruder(4, 0, 1, pmap=pmap, archetype=SHADOWBLADE)
        ai.intruders.append(intruder)

        intruder.state = IntruderState.INTERACTING
        intruder.interaction_type = "grab_bait"
        intruder.interaction_target = (5, 0, 1)
        intruder.interaction_ticks = 1

        ai._update_interacting(intruder)

        assert int(grid.block_state[5, 1, 1]) == 1, (
            "Adjacent floodgate should toggle when gold bait is grabbed"
        )


class TestPressurePlateEvent:
    """Verify that 'pressure_plate_activated' event is published."""

    def test_event_published_on_step(self):
        ai, bus, grid = _make_ai()
        grid.set(3, 0, 1, VOXEL_PRESSURE_PLATE)
        grid.set_block_state(3, 0, 1, 0)

        received = []
        bus.subscribe("pressure_plate_activated", lambda **kw: received.append(kw))

        intruder = _make_intruder(2, 0, 1)
        ai.intruders.append(intruder)

        ai._move_to(intruder, (3, 0, 1))

        assert len(received) == 1, "Expected exactly one pressure_plate_activated event"
        assert received[0]["x"] == 3
        assert received[0]["y"] == 0
        assert received[0]["z"] == 1
        assert received[0]["intruder"] is intruder

    def test_no_event_on_already_triggered_plate(self):
        ai, bus, grid = _make_ai()
        grid.set(3, 0, 1, VOXEL_PRESSURE_PLATE)
        grid.set_block_state(3, 0, 1, 1)  # already triggered

        received = []
        bus.subscribe("pressure_plate_activated", lambda **kw: received.append(kw))

        intruder = _make_intruder(2, 0, 1)
        ai.intruders.append(intruder)

        ai._move_to(intruder, (3, 0, 1))

        assert len(received) == 0, "No event should fire on already-triggered plate"


class TestPressurePlateState:
    """After stepping on a pressure plate, its block_state should be 1."""

    def test_plate_state_set_to_one(self):
        ai, bus, grid = _make_ai()
        grid.set(3, 0, 1, VOXEL_PRESSURE_PLATE)
        grid.set_block_state(3, 0, 1, 0)

        intruder = _make_intruder(2, 0, 1)
        ai.intruders.append(intruder)

        ai._move_to(intruder, (3, 0, 1))

        assert int(grid.block_state[3, 0, 1]) == 1, (
            "Pressure plate block_state should be 1 after being stepped on"
        )


class TestPressurePlateTriggerRange:
    """The trigger range constant should be 1 (document assumption)."""

    def test_trigger_range_is_one(self):
        assert PRESSURE_PLATE_TRIGGER_RANGE == 1, (
            "Tests assume PRESSURE_PLATE_TRIGGER_RANGE == 1"
        )

    def test_diagonal_neighbor_within_range(self):
        """Diagonal neighbor (dx=1, dy=1, dz=0) is within the 3x3x3 cube scan."""
        ai, bus, grid = _make_ai()
        grid.set(3, 0, 1, VOXEL_PRESSURE_PLATE)
        grid.set_block_state(3, 0, 1, 0)
        # Spike at diagonal: (4,1,1) — Chebyshev distance 1
        grid.set(4, 1, 1, VOXEL_SPIKE)
        grid.set_block_state(4, 1, 1, 0)

        intruder = _make_intruder(2, 0, 1)
        ai.intruders.append(intruder)

        ai._move_to(intruder, (3, 0, 1))

        assert int(grid.block_state[4, 1, 1]) == 1, (
            "Diagonal spike (Chebyshev distance 1) should be activated"
        )

    def test_vertical_neighbor_within_range(self):
        """Spike one z-level away (dz=1) is within range."""
        ai, bus, grid = _make_ai()
        grid.set(3, 0, 1, VOXEL_PRESSURE_PLATE)
        grid.set_block_state(3, 0, 1, 0)
        # Spike at (3,0,2) — one level deeper
        grid.set(3, 0, 2, VOXEL_SPIKE)
        grid.set_block_state(3, 0, 2, 0)

        intruder = _make_intruder(2, 0, 1)
        ai.intruders.append(intruder)

        ai._move_to(intruder, (3, 0, 1))

        assert int(grid.block_state[3, 0, 2]) == 1, (
            "Spike one z-level below the plate should be activated"
        )
