"""A* pathfinding on the 3D voxel grid."""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING

from dungeon_builder.config import (
    VOXEL_AIR,
    PATHFINDING_MAX_ITERATIONS,
    PATHFINDING_VERTICAL_COST,
)

if TYPE_CHECKING:
    from dungeon_builder.world.voxel_grid import VoxelGrid


class AStarPathfinder:
    """3D A* pathfinding through air voxels.

    Movement: 4-directional horizontal (N/S/E/W) plus up/down.
    Intruders can only traverse air voxels.
    Vertical movement costs slightly more to prefer horizontal paths.
    """

    def __init__(self, voxel_grid: VoxelGrid) -> None:
        self.voxel_grid = voxel_grid

    def find_path(
        self,
        start: tuple[int, int, int],
        goal: tuple[int, int, int],
        max_iterations: int = PATHFINDING_MAX_ITERATIONS,
    ) -> list[tuple[int, int, int]] | None:
        """Find a path from start to goal through air voxels.

        Returns list of (x, y, z) waypoints including start and goal,
        or None if no path exists.
        """
        if start == goal:
            return [start]

        # Verify start and goal are traversable
        if self.voxel_grid.get(*start) != VOXEL_AIR:
            return None
        if self.voxel_grid.get(*goal) != VOXEL_AIR:
            return None

        open_set: list[tuple[float, int, tuple[int, int, int]]] = []
        counter = 0  # Tiebreaker for heap
        heapq.heappush(open_set, (0.0, counter, start))
        counter += 1

        came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        g_score: dict[tuple[int, int, int], float] = {start: 0.0}
        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                return self._reconstruct(came_from, current)

            # Skip if we've already found a better path to this node
            current_g = g_score.get(current, float("inf"))

            for neighbor in self._get_neighbors(current):
                cost = self._move_cost(current, neighbor)
                tentative_g = current_g + cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, counter, neighbor))
                    counter += 1

        return None  # No path found within iteration budget

    def _heuristic(self, a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
        """Manhattan distance in 3D."""
        return float(abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2]))

    def _get_neighbors(self, pos: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        x, y, z = pos
        neighbors = []
        grid = self.voxel_grid

        # Horizontal movement (4-directional)
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if grid.in_bounds(nx, ny, z) and grid.get(nx, ny, z) == VOXEL_AIR:
                neighbors.append((nx, ny, z))

        # Vertical movement: down (z+1 = deeper in array)
        if grid.in_bounds(x, y, z + 1) and grid.get(x, y, z + 1) == VOXEL_AIR:
            neighbors.append((x, y, z + 1))

        # Vertical movement: up (z-1 = toward surface)
        if z > 0 and grid.in_bounds(x, y, z - 1) and grid.get(x, y, z - 1) == VOXEL_AIR:
            neighbors.append((x, y, z - 1))

        return neighbors

    def _move_cost(
        self, current: tuple[int, int, int], neighbor: tuple[int, int, int]
    ) -> float:
        if current[2] != neighbor[2]:
            return PATHFINDING_VERTICAL_COST
        return 1.0

    def _reconstruct(
        self,
        came_from: dict[tuple[int, int, int], tuple[int, int, int]],
        current: tuple[int, int, int],
    ) -> list[tuple[int, int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
