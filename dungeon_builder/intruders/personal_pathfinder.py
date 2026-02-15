"""A* pathfinder operating on an intruder's PersonalMap.

Unlike the global :class:`AStarPathfinder` in ``world/pathfinding.py``,
this pathfinder only knows about cells the intruder has *seen*.  Unrevealed
cells are treated as impassable.  Traversability depends on the intruder's
archetype (diggers can path through solid, flyers can move vertically
without slopes, fire-immune can cross lava, etc.).
"""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING

from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_SLOPE,
    VOXEL_STAIRS,
    VOXEL_DOOR,
    VOXEL_SPIKE,
    VOXEL_TREASURE,
    VOXEL_TARP,
    VOXEL_ROLLING_STONE,
    VOXEL_LAVA,
    VOXEL_WATER,
    VOXEL_REINFORCED_WALL,
    VOXEL_BEDROCK,
    VOXEL_CORE,
    NON_DIGGABLE,
    PERSONAL_PATHFINDER_MAX_ITERATIONS,
    HAZARD_PATH_COST,
    LAVA_ADJACENT_PATH_COST,
    PATHFINDING_VERTICAL_COST,
)

if TYPE_CHECKING:
    from dungeon_builder.intruders.archetypes import ArchetypeStats
    from dungeon_builder.intruders.personal_map import PersonalMap


# Block types universally traversable (when revealed)
_WALK_TYPES = frozenset({VOXEL_AIR, VOXEL_SLOPE, VOXEL_STAIRS})


class PersonalPathfinder:
    """Fog-of-war A* for a single intruder."""

    @staticmethod
    def find_path(
        personal_map: PersonalMap,
        start: tuple[int, int, int],
        goal: tuple[int, int, int],
        archetype: ArchetypeStats,
        *,
        max_iterations: int = PERSONAL_PATHFINDER_MAX_ITERATIONS,
    ) -> list[tuple[int, int, int]] | None:
        """Return a path from *start* to *goal*, or *None*.

        Only uses cells present in *personal_map.seen*.  Traversability
        and move costs depend on the *archetype*'s abilities.
        """
        if start == goal:
            return [start]

        # Goal must be revealed (or we can't know if we can reach it)
        if not personal_map.is_revealed(*goal):
            return None

        open_set: list[tuple[float, int, tuple[int, int, int]]] = []
        counter = 0
        g_score: dict[tuple[int, int, int], float] = {start: 0.0}
        came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        f0 = _heuristic(start, goal)
        heapq.heappush(open_set, (f0, counter, start))
        counter += 1

        iterations = 0
        while open_set and iterations < max_iterations:
            iterations += 1
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                return _reconstruct(came_from, current)

            current_g = g_score[current]

            for neighbor, move_cost in _get_neighbors(
                personal_map, current, archetype
            ):
                tentative_g = current_g + move_cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f = tentative_g + _heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, counter, neighbor))
                    counter += 1

        return None  # No path found


def _heuristic(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])


def _get_neighbors(
    personal_map: PersonalMap,
    pos: tuple[int, int, int],
    archetype: ArchetypeStats,
) -> list[tuple[tuple[int, int, int], float]]:
    """Return (neighbor, cost) pairs reachable from *pos*."""
    x, y, z = pos
    result: list[tuple[tuple[int, int, int], float]] = []

    # 4 horizontal + 2 vertical
    for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                        (0, 0, 1), (0, 0, -1)):
        nx, ny, nz = x + dx, y + dy, z + dz
        npos = (nx, ny, nz)

        vtype = personal_map.get_type(nx, ny, nz)
        if vtype is None:
            continue  # Unrevealed → impassable

        cost = _move_cost(personal_map, npos, vtype, archetype, dz)
        if cost is not None:
            result.append((npos, cost))

    return result


def _move_cost(
    personal_map: PersonalMap,
    pos: tuple[int, int, int],
    vtype: int,
    archetype: ArchetypeStats,
    dz: int,
) -> float | None:
    """Return the move cost into *pos*, or *None* if impassable.

    *dz* is -1 (up in array = shallower), +1 (down = deeper), or 0 (horizontal).
    """
    base_cost = 1.0

    # --- Universal traversals ---
    if vtype in _WALK_TYPES:
        # Vertical movement without slope/stairs: only flyers
        if dz != 0 and vtype == VOXEL_AIR and not archetype.can_fly:
            # Non-flyers need slope/stairs for vertical movement
            # But going "down" (z+1) through air is falling — allowed
            if dz == -1:
                return None  # Can't go up through plain air without flying
        cost = base_cost if dz == 0 else PATHFINDING_VERTICAL_COST
        # Hazard modifiers
        if pos in personal_map.hazards and archetype.cunning >= 0.5:
            cost += HAZARD_PATH_COST
        return cost

    # --- Doors ---
    if vtype == VOXEL_DOOR:
        door_state = personal_map.get_door_state(*pos)
        if door_state == 0:  # Open
            return base_cost
        # Closed door
        if archetype.can_lockpick:
            return base_cost + 5.0  # Lockpick cost
        if archetype.can_bash_door:
            return base_cost + 15.0  # Bash cost
        return None  # Can't get through

    # --- Functional blocks (spike, treasure, tarp, rolling stone) ---
    # These are traversable (the interaction system handles what happens)
    if vtype in (VOXEL_SPIKE, VOXEL_TREASURE, VOXEL_TARP, VOXEL_ROLLING_STONE):
        cost = base_cost
        if pos in personal_map.hazards and archetype.cunning >= 0.5:
            cost += HAZARD_PATH_COST
        return cost

    # --- Lava ---
    if vtype == VOXEL_LAVA:
        if archetype.fire_immune:
            return base_cost + 2.0  # Slight preference for non-lava
        return None  # Death

    # --- Water ---
    if vtype == VOXEL_WATER:
        return None  # Nobody can swim (for now)

    # --- Reinforced wall / bedrock / core ---
    if vtype in (VOXEL_REINFORCED_WALL, VOXEL_BEDROCK, VOXEL_CORE):
        return None  # Never traversable

    # --- Other solid blocks ---
    if archetype.can_dig and vtype not in NON_DIGGABLE:
        # Digger can tunnel through diggable solids
        # Cost = high (reflects time to dig)
        return base_cost + 20.0
    return None  # Solid, can't traverse


def _reconstruct(
    came_from: dict[tuple[int, int, int], tuple[int, int, int]],
    current: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
