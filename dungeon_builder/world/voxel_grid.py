"""3D voxel grid data structure backed by a NumPy array."""

from __future__ import annotations

import numpy as np

from dungeon_builder.config import (
    GRID_WIDTH,
    GRID_DEPTH,
    GRID_HEIGHT,
    CHUNK_SIZE,
    VOXEL_AIR,
    VOXEL_BEDROCK,
)


class VoxelGrid:
    """The foundational 3D grid storing voxel types.

    Coordinates: grid[x, y, z] where:
      - x: 0..width-1 (east-west)
      - y: 0..depth-1 (north-south)
      - z: 0..height-1 where z=0 is surface, z=height-1 is deepest

    Stored as uint8 (256 possible voxel types).
    """

    def __init__(
        self,
        width: int = GRID_WIDTH,
        depth: int = GRID_DEPTH,
        height: int = GRID_HEIGHT,
    ) -> None:
        self.width = width
        self.depth = depth
        self.height = height
        self.grid = np.zeros((width, depth, height), dtype=np.uint8)
        self.humidity = np.zeros((width, depth, height), dtype=np.float32)
        self.temperature = np.zeros((width, depth, height), dtype=np.float32)
        self.loose = np.zeros((width, depth, height), dtype=np.bool_)
        self.load = np.zeros((width, depth, height), dtype=np.float32)
        self.stress_ratio = np.zeros((width, depth, height), dtype=np.float32)
        self._dirty_chunks: set[tuple[int, int, int]] = set()

        # Number of chunks per axis
        self.chunks_x = (width + CHUNK_SIZE - 1) // CHUNK_SIZE
        self.chunks_y = (depth + CHUNK_SIZE - 1) // CHUNK_SIZE

    def get(self, x: int, y: int, z: int) -> int:
        if not self.in_bounds(x, y, z):
            return VOXEL_BEDROCK  # Out of bounds = impassable
        return int(self.grid[x, y, z])

    def set(self, x: int, y: int, z: int, voxel_type: int, event_bus=None) -> None:
        if not self.in_bounds(x, y, z):
            return
        old = int(self.grid[x, y, z])
        if old == voxel_type:
            return
        self.grid[x, y, z] = voxel_type
        if voxel_type == VOXEL_AIR:
            self.loose[x, y, z] = False

        # Mark affected chunks dirty
        cx, cy = x // CHUNK_SIZE, y // CHUNK_SIZE
        self._dirty_chunks.add((cx, cy, z))

        # If on a chunk boundary, also mark the adjacent chunk
        lx = x % CHUNK_SIZE
        ly = y % CHUNK_SIZE
        if lx == 0 and cx > 0:
            self._dirty_chunks.add((cx - 1, cy, z))
        if lx == CHUNK_SIZE - 1 and cx < self.chunks_x - 1:
            self._dirty_chunks.add((cx + 1, cy, z))
        if ly == 0 and cy > 0:
            self._dirty_chunks.add((cx, cy - 1, z))
        if ly == CHUNK_SIZE - 1 and cy < self.chunks_y - 1:
            self._dirty_chunks.add((cx, cy + 1, z))

        # Also mark layers above and below (exposed faces change)
        if z > 0:
            self._dirty_chunks.add((cx, cy, z - 1))
        if z < self.height - 1:
            self._dirty_chunks.add((cx, cy, z + 1))

        if event_bus:
            event_bus.publish(
                "voxel_changed", x=x, y=y, z=z, old_type=old, new_type=voxel_type
            )

    def in_bounds(self, x: int, y: int, z: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.depth and 0 <= z < self.height

    def is_solid(self, x: int, y: int, z: int) -> bool:
        return self.get(x, y, z) != VOXEL_AIR

    def get_humidity(self, x: int, y: int, z: int) -> float:
        if not self.in_bounds(x, y, z):
            return 0.0
        return float(self.humidity[x, y, z])

    def set_humidity(self, x: int, y: int, z: int, value: float) -> None:
        if self.in_bounds(x, y, z):
            self.humidity[x, y, z] = max(0.0, min(1.0, value))

    def get_temperature(self, x: int, y: int, z: int) -> float:
        if not self.in_bounds(x, y, z):
            return 0.0
        return float(self.temperature[x, y, z])

    def set_temperature(self, x: int, y: int, z: int, value: float) -> None:
        if self.in_bounds(x, y, z):
            self.temperature[x, y, z] = value

    def is_loose(self, x: int, y: int, z: int) -> bool:
        if not self.in_bounds(x, y, z):
            return False
        return bool(self.loose[x, y, z])

    def get_load(self, x: int, y: int, z: int) -> float:
        if not self.in_bounds(x, y, z):
            return 0.0
        return float(self.load[x, y, z])

    def get_stress_ratio(self, x: int, y: int, z: int) -> float:
        if not self.in_bounds(x, y, z):
            return 0.0
        return float(self.stress_ratio[x, y, z])

    def set_loose(self, x: int, y: int, z: int, value: bool) -> None:
        if self.in_bounds(x, y, z):
            self.loose[x, y, z] = value

    def pop_dirty_chunks(self) -> set[tuple[int, int, int]]:
        dirty = self._dirty_chunks.copy()
        self._dirty_chunks.clear()
        return dirty

    def mark_all_dirty(self) -> None:
        """Mark every chunk as needing re-mesh (used at initial load)."""
        for cx in range(self.chunks_x):
            for cy in range(self.chunks_y):
                for z in range(self.height):
                    self._dirty_chunks.add((cx, cy, z))
