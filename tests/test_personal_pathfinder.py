"""Tests for fog-of-war A* pathfinder operating on PersonalMap."""

import pytest

from dungeon_builder.intruders.personal_pathfinder import PersonalPathfinder
from dungeon_builder.intruders.personal_map import PersonalMap
from dungeon_builder.intruders.archetypes import (
    VANGUARD, SHADOWBLADE, TUNNELER, PYREMANCER, WINDCALLER,
    GORECLAW, GLOOMSEER,
)
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_STONE,
    VOXEL_DOOR,
    VOXEL_LAVA,
    VOXEL_WATER,
    VOXEL_REINFORCED_WALL,
    VOXEL_BEDROCK,
    VOXEL_SPIKE,
    VOXEL_SLOPE,
    VOXEL_STAIRS,
)


def _reveal_line(pm: PersonalMap, y_start: int, y_end: int, z: int = 0,
                 x: int = 0, vtype: int = VOXEL_AIR):
    """Reveal a horizontal line of cells in the personal map."""
    for y in range(y_start, y_end + 1):
        pm.reveal(x, y, z, vtype)


# ── Basic pathfinding ───────────────────────────────────────────────


class TestBasicPaths:
    def test_same_start_and_goal(self):
        pm = PersonalMap()
        pm.reveal(0, 0, 0, VOXEL_AIR)
        path = PersonalPathfinder.find_path(pm, (0, 0, 0), (0, 0, 0), VANGUARD)
        assert path == [(0, 0, 0)]

    def test_straight_line(self):
        pm = PersonalMap()
        _reveal_line(pm, 0, 5)
        path = PersonalPathfinder.find_path(pm, (0, 0, 0), (0, 5, 0), VANGUARD)
        assert path is not None
        assert path[0] == (0, 0, 0)
        assert path[-1] == (0, 5, 0)
        assert len(path) == 6

    def test_no_path_returns_none(self):
        pm = PersonalMap()
        pm.reveal(0, 0, 0, VOXEL_AIR)
        pm.reveal(0, 5, 0, VOXEL_AIR)
        # Gap between — no intermediate cells revealed
        path = PersonalPathfinder.find_path(pm, (0, 0, 0), (0, 5, 0), VANGUARD)
        assert path is None

    def test_goal_unrevealed_returns_none(self):
        pm = PersonalMap()
        pm.reveal(0, 0, 0, VOXEL_AIR)
        path = PersonalPathfinder.find_path(pm, (0, 0, 0), (0, 5, 0), VANGUARD)
        assert path is None

    def test_path_around_wall(self):
        pm = PersonalMap()
        # Corridor: (0,0,0) → (0,1,0) → (1,1,0) → (1,2,0) → (0,2,0)
        pm.reveal(0, 0, 0, VOXEL_AIR)
        pm.reveal(0, 1, 0, VOXEL_AIR)
        pm.reveal(1, 1, 0, VOXEL_AIR)
        pm.reveal(1, 2, 0, VOXEL_AIR)
        pm.reveal(0, 2, 0, VOXEL_AIR)
        # Stone wall at (0,1,0)… wait, that's air. Let's do it differently:
        # Direct path (0,0)→(0,2) would go through (0,1) which is air, so there IS a direct path
        # Let me create a proper wall scenario
        pm2 = PersonalMap()
        pm2.reveal(0, 0, 0, VOXEL_AIR)
        pm2.reveal(0, 1, 0, VOXEL_STONE)  # Wall!
        pm2.reveal(1, 0, 0, VOXEL_AIR)
        pm2.reveal(1, 1, 0, VOXEL_AIR)
        pm2.reveal(1, 2, 0, VOXEL_AIR)
        pm2.reveal(0, 2, 0, VOXEL_AIR)
        path = PersonalPathfinder.find_path(pm2, (0, 0, 0), (0, 2, 0), VANGUARD)
        assert path is not None
        assert (0, 1, 0) not in path  # Doesn't go through wall
        assert path[-1] == (0, 2, 0)


# ── Unrevealed cells ────────────────────────────────────────────────


class TestUnrevealed:
    def test_unrevealed_cells_impassable(self):
        pm = PersonalMap()
        pm.reveal(0, 0, 0, VOXEL_AIR)
        pm.reveal(0, 2, 0, VOXEL_AIR)
        # (0,1,0) not revealed — should block path
        path = PersonalPathfinder.find_path(pm, (0, 0, 0), (0, 2, 0), VANGUARD)
        assert path is None

    def test_partial_reveal_finds_path(self):
        pm = PersonalMap()
        pm.reveal(0, 0, 0, VOXEL_AIR)
        pm.reveal(0, 1, 0, VOXEL_AIR)
        pm.reveal(0, 2, 0, VOXEL_AIR)
        path = PersonalPathfinder.find_path(pm, (0, 0, 0), (0, 2, 0), VANGUARD)
        assert path is not None
        assert len(path) == 3


# ── Door traversal ──────────────────────────────────────────────────


class TestDoorTraversal:
    def test_open_door_traversable_by_all(self):
        pm = PersonalMap()
        pm.reveal(0, 0, 0, VOXEL_AIR)
        pm.reveal(0, 1, 0, VOXEL_DOOR, block_state=0)  # Open
        pm.reveal(0, 2, 0, VOXEL_AIR)
        path = PersonalPathfinder.find_path(pm, (0, 0, 0), (0, 2, 0), VANGUARD)
        assert path is not None

    def test_closed_door_blocked_for_non_interacters(self):
        pm = PersonalMap()
        pm.reveal(0, 0, 0, VOXEL_AIR)
        pm.reveal(0, 1, 0, VOXEL_DOOR, block_state=1)  # Closed
        pm.reveal(0, 2, 0, VOXEL_AIR)
        # Pyremancer can't bash or lockpick
        path = PersonalPathfinder.find_path(pm, (0, 0, 0), (0, 2, 0), PYREMANCER)
        assert path is None

    def test_closed_door_lockpicked_by_shadowblade(self):
        pm = PersonalMap()
        pm.reveal(0, 0, 0, VOXEL_AIR)
        pm.reveal(0, 1, 0, VOXEL_DOOR, block_state=1)
        pm.reveal(0, 2, 0, VOXEL_AIR)
        path = PersonalPathfinder.find_path(pm, (0, 0, 0), (0, 2, 0), SHADOWBLADE)
        assert path is not None
        assert (0, 1, 0) in path

    def test_closed_door_bashed_by_vanguard(self):
        pm = PersonalMap()
        pm.reveal(0, 0, 0, VOXEL_AIR)
        pm.reveal(0, 1, 0, VOXEL_DOOR, block_state=1)
        pm.reveal(0, 2, 0, VOXEL_AIR)
        path = PersonalPathfinder.find_path(pm, (0, 0, 0), (0, 2, 0), VANGUARD)
        assert path is not None


# ── Digger traversal ────────────────────────────────────────────────


class TestDiggerTraversal:
    def test_tunneler_through_stone(self):
        pm = PersonalMap()
        pm.reveal(0, 0, 0, VOXEL_AIR)
        pm.reveal(0, 1, 0, VOXEL_STONE)
        pm.reveal(0, 2, 0, VOXEL_AIR)
        path = PersonalPathfinder.find_path(pm, (0, 0, 0), (0, 2, 0), TUNNELER)
        assert path is not None
        assert (0, 1, 0) in path  # Goes through stone

    def test_non_digger_blocked_by_stone(self):
        pm = PersonalMap()
        pm.reveal(0, 0, 0, VOXEL_AIR)
        pm.reveal(0, 1, 0, VOXEL_STONE)
        pm.reveal(0, 2, 0, VOXEL_AIR)
        path = PersonalPathfinder.find_path(pm, (0, 0, 0), (0, 2, 0), VANGUARD)
        assert path is None

    def test_tunneler_cannot_dig_reinforced(self):
        pm = PersonalMap()
        pm.reveal(0, 0, 0, VOXEL_AIR)
        pm.reveal(0, 1, 0, VOXEL_REINFORCED_WALL)
        pm.reveal(0, 2, 0, VOXEL_AIR)
        path = PersonalPathfinder.find_path(pm, (0, 0, 0), (0, 2, 0), TUNNELER)
        assert path is None

    def test_tunneler_cannot_dig_bedrock(self):
        pm = PersonalMap()
        pm.reveal(0, 0, 0, VOXEL_AIR)
        pm.reveal(0, 1, 0, VOXEL_BEDROCK)
        pm.reveal(0, 2, 0, VOXEL_AIR)
        path = PersonalPathfinder.find_path(pm, (0, 0, 0), (0, 2, 0), TUNNELER)
        assert path is None

    def test_dig_has_high_cost(self):
        """Tunneler prefers air over digging even if air path is longer."""
        pm = PersonalMap()
        # Direct path through stone: 3 cells
        pm.reveal(0, 0, 0, VOXEL_AIR)
        pm.reveal(0, 1, 0, VOXEL_STONE)
        pm.reveal(0, 2, 0, VOXEL_AIR)
        # Air detour: 5 cells
        pm.reveal(1, 0, 0, VOXEL_AIR)
        pm.reveal(1, 1, 0, VOXEL_AIR)
        pm.reveal(1, 2, 0, VOXEL_AIR)
        path = PersonalPathfinder.find_path(pm, (0, 0, 0), (0, 2, 0), TUNNELER)
        assert path is not None
        # Should prefer the air detour since dig cost is high
        assert (0, 1, 0) not in path


# ── Flyer traversal ─────────────────────────────────────────────────


class TestFlyerTraversal:
    def test_flyer_moves_vertically_through_air(self):
        pm = PersonalMap()
        pm.reveal(5, 5, 0, VOXEL_AIR)
        pm.reveal(5, 5, 1, VOXEL_AIR)
        pm.reveal(5, 5, 2, VOXEL_AIR)
        path = PersonalPathfinder.find_path(pm, (5, 5, 0), (5, 5, 2), WINDCALLER)
        assert path is not None
        assert len(path) == 3

    def test_non_flyer_cannot_go_up_through_air(self):
        pm = PersonalMap()
        pm.reveal(5, 5, 0, VOXEL_AIR)
        pm.reveal(5, 5, 1, VOXEL_AIR)
        # Vanguard can't fly up
        path = PersonalPathfinder.find_path(pm, (5, 5, 1), (5, 5, 0), VANGUARD)
        assert path is None

    def test_non_flyer_uses_slope_to_go_up(self):
        pm = PersonalMap()
        pm.reveal(5, 5, 1, VOXEL_AIR)
        pm.reveal(5, 5, 0, VOXEL_SLOPE)  # Slope at z=0
        path = PersonalPathfinder.find_path(pm, (5, 5, 1), (5, 5, 0), VANGUARD)
        assert path is not None

    def test_non_flyer_uses_stairs_to_go_up(self):
        pm = PersonalMap()
        pm.reveal(5, 5, 1, VOXEL_AIR)
        pm.reveal(5, 5, 0, VOXEL_STAIRS)
        path = PersonalPathfinder.find_path(pm, (5, 5, 1), (5, 5, 0), VANGUARD)
        assert path is not None

    def test_non_flyer_can_fall_down(self):
        pm = PersonalMap()
        pm.reveal(5, 5, 0, VOXEL_AIR)
        pm.reveal(5, 5, 1, VOXEL_AIR)
        # Going deeper (z+1) is falling — should be allowed
        path = PersonalPathfinder.find_path(pm, (5, 5, 0), (5, 5, 1), VANGUARD)
        assert path is not None


# ── Fire-immune traversal ───────────────────────────────────────────


class TestFireImmuneTraversal:
    def test_pyremancer_through_lava(self):
        pm = PersonalMap()
        pm.reveal(0, 0, 0, VOXEL_AIR)
        pm.reveal(0, 1, 0, VOXEL_LAVA)
        pm.reveal(0, 2, 0, VOXEL_AIR)
        path = PersonalPathfinder.find_path(pm, (0, 0, 0), (0, 2, 0), PYREMANCER)
        assert path is not None

    def test_non_immune_blocked_by_lava(self):
        pm = PersonalMap()
        pm.reveal(0, 0, 0, VOXEL_AIR)
        pm.reveal(0, 1, 0, VOXEL_LAVA)
        pm.reveal(0, 2, 0, VOXEL_AIR)
        path = PersonalPathfinder.find_path(pm, (0, 0, 0), (0, 2, 0), VANGUARD)
        assert path is None


# ── Water / reinforced wall ─────────────────────────────────────────


class TestImpassables:
    def test_water_blocks_all(self):
        pm = PersonalMap()
        pm.reveal(0, 0, 0, VOXEL_AIR)
        pm.reveal(0, 1, 0, VOXEL_WATER)
        pm.reveal(0, 2, 0, VOXEL_AIR)
        for arch in [VANGUARD, SHADOWBLADE, TUNNELER, PYREMANCER, WINDCALLER]:
            path = PersonalPathfinder.find_path(pm, (0, 0, 0), (0, 2, 0), arch)
            assert path is None, f"{arch.name} should not pass through water"

    def test_reinforced_wall_blocks_all(self):
        pm = PersonalMap()
        pm.reveal(0, 0, 0, VOXEL_AIR)
        pm.reveal(0, 1, 0, VOXEL_REINFORCED_WALL)
        pm.reveal(0, 2, 0, VOXEL_AIR)
        for arch in [VANGUARD, TUNNELER, GORECLAW]:
            path = PersonalPathfinder.find_path(pm, (0, 0, 0), (0, 2, 0), arch)
            assert path is None, f"{arch.name} should not pass through reinforced wall"


# ── Hazard avoidance ────────────────────────────────────────────────


class TestHazardAvoidance:
    def test_cunning_avoids_hazards_if_alternative(self):
        """High-cunning intruder avoids spike if safe detour exists."""
        pm = PersonalMap()
        # Direct: (0,0) → (0,1)[spike] → (0,2)
        pm.reveal(0, 0, 0, VOXEL_AIR)
        pm.reveal(0, 1, 0, VOXEL_AIR)
        pm.hazards.add((0, 1, 0))  # Marked as hazard
        pm.reveal(0, 2, 0, VOXEL_AIR)
        # Detour: (0,0) → (1,0) → (1,1) → (1,2) → (0,2)
        pm.reveal(1, 0, 0, VOXEL_AIR)
        pm.reveal(1, 1, 0, VOXEL_AIR)
        pm.reveal(1, 2, 0, VOXEL_AIR)
        # Gloomseer has cunning=0.7 (≥0.5)
        path = PersonalPathfinder.find_path(pm, (0, 0, 0), (0, 2, 0), GLOOMSEER)
        assert path is not None
        # Should avoid the hazard cell
        assert (0, 1, 0) not in path

    def test_low_cunning_walks_through_hazard(self):
        """Low-cunning intruder ignores hazards, takes direct path."""
        pm = PersonalMap()
        pm.reveal(0, 0, 0, VOXEL_AIR)
        pm.reveal(0, 1, 0, VOXEL_AIR)
        pm.hazards.add((0, 1, 0))
        pm.reveal(0, 2, 0, VOXEL_AIR)
        # Goreclaw has cunning=0.0
        path = PersonalPathfinder.find_path(pm, (0, 0, 0), (0, 2, 0), GORECLAW)
        assert path is not None
        assert (0, 1, 0) in path  # Direct path through hazard


# ── Max iterations ──────────────────────────────────────────────────


class TestMaxIterations:
    def test_returns_none_when_exceeded(self):
        pm = PersonalMap()
        for y in range(50):
            pm.reveal(0, y, 0, VOXEL_AIR)
        path = PersonalPathfinder.find_path(
            pm, (0, 0, 0), (0, 49, 0), VANGUARD, max_iterations=5
        )
        assert path is None  # Not enough iterations to find long path
