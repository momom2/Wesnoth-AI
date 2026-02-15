"""Tests for the intruder personal fog-of-war map."""

import pytest

from dungeon_builder.intruders.personal_map import PersonalMap
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_STONE,
    VOXEL_SPIKE,
    VOXEL_LAVA,
    VOXEL_TARP,
    VOXEL_TREASURE,
    VOXEL_DOOR,
    VOXEL_DIRT,
)


class TestPersonalMapReveal:
    """Test cell revelation and type tracking."""

    def test_initially_empty(self):
        pm = PersonalMap()
        assert len(pm) == 0
        assert not pm.is_revealed(0, 0, 0)

    def test_reveal_single_cell(self):
        pm = PersonalMap()
        pm.reveal(5, 10, 3, VOXEL_STONE)
        assert pm.is_revealed(5, 10, 3)
        assert pm.get_type(5, 10, 3) == VOXEL_STONE

    def test_reveal_updates_type(self):
        pm = PersonalMap()
        pm.reveal(1, 1, 1, VOXEL_STONE)
        pm.reveal(1, 1, 1, VOXEL_AIR)
        assert pm.get_type(1, 1, 1) == VOXEL_AIR

    def test_unrevealed_returns_none(self):
        pm = PersonalMap()
        assert pm.get_type(0, 0, 0) is None

    def test_reveal_multiple_cells(self):
        pm = PersonalMap()
        for i in range(10):
            pm.reveal(i, 0, 0, VOXEL_AIR)
        assert len(pm) == 10


class TestPersonalMapHazards:
    """Test automatic hazard tracking."""

    def test_spike_tracked_as_hazard(self):
        pm = PersonalMap()
        pm.reveal(3, 3, 0, VOXEL_SPIKE)
        assert (3, 3, 0) in pm.hazards

    def test_lava_tracked_as_hazard(self):
        pm = PersonalMap()
        pm.reveal(5, 5, 2, VOXEL_LAVA)
        assert (5, 5, 2) in pm.hazards

    def test_tarp_tracked_as_hazard(self):
        pm = PersonalMap()
        pm.reveal(1, 1, 0, VOXEL_TARP)
        assert (1, 1, 0) in pm.hazards

    def test_non_hazard_not_tracked(self):
        pm = PersonalMap()
        pm.reveal(0, 0, 0, VOXEL_STONE)
        assert (0, 0, 0) not in pm.hazards

    def test_hazard_removed_when_type_changes(self):
        pm = PersonalMap()
        pm.reveal(2, 2, 0, VOXEL_SPIKE)
        assert (2, 2, 0) in pm.hazards
        pm.reveal(2, 2, 0, VOXEL_AIR)  # Spike was destroyed
        assert (2, 2, 0) not in pm.hazards

    def test_mark_hazard_manually(self):
        pm = PersonalMap()
        pm.reveal(4, 4, 0, VOXEL_AIR)
        pm.mark_hazard(4, 4, 0)
        assert (4, 4, 0) in pm.hazards


class TestPersonalMapTreasures:
    """Test treasure position tracking."""

    def test_treasure_tracked(self):
        pm = PersonalMap()
        pm.reveal(8, 8, 5, VOXEL_TREASURE)
        assert (8, 8, 5) in pm.treasures

    def test_treasure_removed_when_collected(self):
        pm = PersonalMap()
        pm.reveal(8, 8, 5, VOXEL_TREASURE)
        pm.reveal(8, 8, 5, VOXEL_AIR)  # Treasure collected
        assert (8, 8, 5) not in pm.treasures

    def test_remove_treasure_explicitly(self):
        pm = PersonalMap()
        pm.reveal(8, 8, 5, VOXEL_TREASURE)
        pm.remove_treasure(8, 8, 5)
        assert (8, 8, 5) not in pm.treasures

    def test_remove_treasure_nonexistent_safe(self):
        pm = PersonalMap()
        pm.remove_treasure(0, 0, 0)  # No error

    def test_multiple_treasures(self):
        pm = PersonalMap()
        pm.reveal(1, 1, 1, VOXEL_TREASURE)
        pm.reveal(2, 2, 2, VOXEL_TREASURE)
        assert len(pm.treasures) == 2


class TestPersonalMapDoors:
    """Test door state tracking."""

    def test_door_tracked_with_state(self):
        pm = PersonalMap()
        pm.reveal(3, 3, 1, VOXEL_DOOR, block_state=1)  # Closed
        assert pm.get_door_state(3, 3, 1) == 1

    def test_door_state_updates(self):
        pm = PersonalMap()
        pm.reveal(3, 3, 1, VOXEL_DOOR, block_state=1)
        pm.reveal(3, 3, 1, VOXEL_DOOR, block_state=0)  # Now open
        assert pm.get_door_state(3, 3, 1) == 0

    def test_door_removed_when_destroyed(self):
        pm = PersonalMap()
        pm.reveal(3, 3, 1, VOXEL_DOOR, block_state=1)
        pm.reveal(3, 3, 1, VOXEL_AIR)
        assert pm.get_door_state(3, 3, 1) is None

    def test_non_door_returns_none(self):
        pm = PersonalMap()
        pm.reveal(0, 0, 0, VOXEL_STONE)
        assert pm.get_door_state(0, 0, 0) is None


class TestPersonalMapFrontier:
    """Test exploration frontier detection."""

    def test_single_cell_is_frontier(self):
        pm = PersonalMap()
        pm.reveal(5, 5, 5, VOXEL_AIR)
        frontier = pm.get_frontier()
        assert (5, 5, 5) in frontier

    def test_surrounded_cell_not_frontier(self):
        pm = PersonalMap()
        # Reveal center and all 6 neighbors
        pm.reveal(5, 5, 5, VOXEL_AIR)
        pm.reveal(6, 5, 5, VOXEL_STONE)
        pm.reveal(4, 5, 5, VOXEL_STONE)
        pm.reveal(5, 6, 5, VOXEL_STONE)
        pm.reveal(5, 4, 5, VOXEL_STONE)
        pm.reveal(5, 5, 6, VOXEL_STONE)
        pm.reveal(5, 5, 4, VOXEL_STONE)
        frontier = pm.get_frontier()
        # Center has all neighbors revealed — NOT frontier
        assert (5, 5, 5) not in frontier
        # But the stone neighbors themselves ARE frontier (they have unrevealed neighbors)
        assert len(frontier) == 6

    def test_empty_map_no_frontier(self):
        pm = PersonalMap()
        assert pm.get_frontier() == []

    def test_line_frontier(self):
        pm = PersonalMap()
        for x in range(5):
            pm.reveal(x, 0, 0, VOXEL_AIR)
        frontier = pm.get_frontier()
        # All cells in a line have unrevealed neighbors (y±1, z±1)
        assert len(frontier) == 5


class TestPersonalMapMerge:
    """Test map merging for cooperative allies."""

    def test_merge_adds_unseen_cells(self):
        pm1 = PersonalMap()
        pm2 = PersonalMap()
        pm1.reveal(0, 0, 0, VOXEL_AIR)
        pm2.reveal(1, 1, 1, VOXEL_STONE)
        pm1.merge(pm2)
        assert pm1.is_revealed(0, 0, 0)
        assert pm1.is_revealed(1, 1, 1)
        assert len(pm1) == 2

    def test_merge_updates_existing_cells(self):
        pm1 = PersonalMap()
        pm2 = PersonalMap()
        pm1.reveal(0, 0, 0, VOXEL_STONE)
        pm2.reveal(0, 0, 0, VOXEL_AIR)  # Dug out
        pm1.merge(pm2)
        assert pm1.get_type(0, 0, 0) == VOXEL_AIR

    def test_merge_combines_hazards(self):
        pm1 = PersonalMap()
        pm2 = PersonalMap()
        pm1.reveal(0, 0, 0, VOXEL_SPIKE)
        pm2.reveal(1, 1, 1, VOXEL_LAVA)
        pm1.merge(pm2)
        assert (0, 0, 0) in pm1.hazards
        assert (1, 1, 1) in pm1.hazards

    def test_merge_combines_treasures(self):
        pm1 = PersonalMap()
        pm2 = PersonalMap()
        pm1.reveal(0, 0, 0, VOXEL_TREASURE)
        pm2.reveal(5, 5, 5, VOXEL_TREASURE)
        pm1.merge(pm2)
        assert len(pm1.treasures) == 2

    def test_merge_combines_doors(self):
        pm1 = PersonalMap()
        pm2 = PersonalMap()
        pm1.reveal(0, 0, 0, VOXEL_DOOR, block_state=1)
        pm2.reveal(5, 5, 5, VOXEL_DOOR, block_state=0)
        pm1.merge(pm2)
        assert pm1.get_door_state(0, 0, 0) == 1
        assert pm1.get_door_state(5, 5, 5) == 0

    def test_merge_is_idempotent(self):
        pm1 = PersonalMap()
        pm2 = PersonalMap()
        pm2.reveal(1, 1, 1, VOXEL_STONE)
        pm1.merge(pm2)
        pm1.merge(pm2)  # Second merge should be a no-op
        assert len(pm1) == 1

    def test_merge_does_not_modify_source(self):
        pm1 = PersonalMap()
        pm2 = PersonalMap()
        pm2.reveal(1, 1, 1, VOXEL_STONE)
        pm1.reveal(0, 0, 0, VOXEL_AIR)
        pm1.merge(pm2)
        # pm2 should still only have its original cell
        assert len(pm2) == 1
        assert not pm2.is_revealed(0, 0, 0)


class TestPersonalMapRepr:
    def test_repr(self):
        pm = PersonalMap()
        pm.reveal(0, 0, 0, VOXEL_SPIKE)
        pm.reveal(1, 1, 1, VOXEL_TREASURE)
        r = repr(pm)
        assert "seen=2" in r
        assert "hazards=1" in r
        assert "treasures=1" in r
