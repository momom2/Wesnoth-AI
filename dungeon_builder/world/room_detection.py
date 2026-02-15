"""Dynamic room recognition from voxel layout via BFS flood fill."""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING

from dungeon_builder.config import VOXEL_AIR

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.world.voxel_grid import VoxelGrid

logger = logging.getLogger("dungeon_builder.rooms")


class Room:
    """An enclosed air region on a single Z-level."""

    def __init__(self, room_id: int, cells: set[tuple[int, int, int]], z: int) -> None:
        self.room_id = room_id
        self.cells = cells
        self.z = z
        self.size = len(cells)

    def __repr__(self) -> str:
        return f"Room(id={self.room_id}, z={self.z}, size={self.size})"


class RoomDetector:
    """Detects enclosed rooms via BFS flood fill on each Z-level.

    A "room" is a connected region of air voxels on a single Z-level that
    is fully enclosed by solid voxels (no path to the map edge through air).
    Minimum size is 4 cells.
    """

    MIN_ROOM_SIZE = 4

    def __init__(self, event_bus: EventBus, voxel_grid: VoxelGrid) -> None:
        self.event_bus = event_bus
        self.voxel_grid = voxel_grid
        self.rooms: dict[int, Room] = {}
        self._next_id = 1
        # Track which cells belong to which room for efficient updates
        self._cell_to_room: dict[tuple[int, int, int], int] = {}

        event_bus.subscribe("dig_complete", self._on_dig_complete)

    def _on_dig_complete(self, x: int, y: int, z: int) -> None:
        """Re-scan for rooms near the changed voxel."""
        # Remove any existing room that contained or was adjacent to this cell
        self._invalidate_nearby_rooms(x, y, z)
        # Re-detect rooms at this Z-level starting from this cell
        self._detect_room_at(x, y, z)

    def _invalidate_nearby_rooms(self, x: int, y: int, z: int) -> None:
        """Remove rooms that might be affected by a change at (x, y, z)."""
        to_remove = set()
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                key = (x + dx, y + dy, z)
                rid = self._cell_to_room.get(key)
                if rid is not None:
                    to_remove.add(rid)

        for rid in to_remove:
            room = self.rooms.pop(rid, None)
            if room:
                for cell in room.cells:
                    self._cell_to_room.pop(cell, None)

    def _detect_room_at(self, start_x: int, start_y: int, z: int) -> None:
        grid = self.voxel_grid
        if grid.get(start_x, start_y, z) != VOXEL_AIR:
            return
        if (start_x, start_y, z) in self._cell_to_room:
            return  # Already part of a detected room

        visited: set[tuple[int, int]] = set()
        queue: deque[tuple[int, int]] = deque([(start_x, start_y)])
        cells: set[tuple[int, int, int]] = set()
        is_enclosed = True

        while queue:
            x, y = queue.popleft()
            if (x, y) in visited:
                continue
            visited.add((x, y))

            if not grid.in_bounds(x, y, z):
                is_enclosed = False
                continue

            if grid.get(x, y, z) != VOXEL_AIR:
                continue

            cells.add((x, y, z))

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) not in visited:
                    if not grid.in_bounds(nx, ny, z):
                        is_enclosed = False
                    else:
                        queue.append((nx, ny))

        if is_enclosed and len(cells) >= self.MIN_ROOM_SIZE:
            room = Room(self._next_id, cells, z)
            self._next_id += 1
            self.rooms[room.room_id] = room
            for cell in cells:
                self._cell_to_room[cell] = room.room_id
            self.event_bus.publish("room_detected", room=room)
            logger.info("Detected %s", room)
