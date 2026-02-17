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
        self.shear_load = np.zeros((width, depth, height), dtype=np.float32)
        self.stress_ratio = np.zeros((width, depth, height), dtype=np.float32)
        self.fall_distance = np.zeros((width, depth, height), dtype=np.uint8)
        self.water_level = np.zeros((width, depth, height), dtype=np.uint8)
        self.thermal_fatigue = np.zeros((width, depth, height), dtype=np.float32)
        self.block_state = np.zeros((width, depth, height), dtype=np.uint8)
        self.metal_type = np.zeros((width, depth, height), dtype=np.uint8)
        self.claimed = np.zeros((width, depth, height), dtype=np.bool_)
        self.visible = np.zeros((width, depth, height), dtype=np.bool_)
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
        self.block_state[x, y, z] = 0  # Reset state when type changes
        self.metal_type[x, y, z] = 0  # Reset metal type when type changes
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

    def is_claimed(self, x: int, y: int, z: int) -> bool:
        """Return True if the air voxel at (x,y,z) is claimed territory."""
        if not self.in_bounds(x, y, z):
            return False
        return bool(self.claimed[x, y, z])

    def is_visible(self, x: int, y: int, z: int) -> bool:
        """Return True if the block at (x,y,z) is adjacent to claimed air."""
        if not self.in_bounds(x, y, z):
            return False
        return bool(self.visible[x, y, z])

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

    def get_water_level(self, x: int, y: int, z: int) -> int:
        if not self.in_bounds(x, y, z):
            return 0
        return int(self.water_level[x, y, z])

    def set_water_level(self, x: int, y: int, z: int, level: int) -> None:
        if self.in_bounds(x, y, z):
            self.water_level[x, y, z] = max(0, min(255, level))

    def get_thermal_fatigue(self, x: int, y: int, z: int) -> float:
        if not self.in_bounds(x, y, z):
            return 0.0
        return float(self.thermal_fatigue[x, y, z])

    def get_block_state(self, x: int, y: int, z: int) -> int:
        if not self.in_bounds(x, y, z):
            return 0
        return int(self.block_state[x, y, z])

    def set_block_state(self, x: int, y: int, z: int, state: int) -> None:
        if not self.in_bounds(x, y, z):
            return
        self.block_state[x, y, z] = state
        # Mark chunk dirty for re-render
        cx, cy = x // CHUNK_SIZE, y // CHUNK_SIZE
        self._dirty_chunks.add((cx, cy, z))

    def get_metal_type(self, x: int, y: int, z: int) -> int:
        """Return the metal type at (x,y,z), or 0 (METAL_NONE) if out of bounds."""
        if not self.in_bounds(x, y, z):
            return 0
        return int(self.metal_type[x, y, z])

    def set_metal_type(self, x: int, y: int, z: int, metal: int) -> None:
        """Set the metal type at (x,y,z). Marks chunk dirty for re-render."""
        if not self.in_bounds(x, y, z):
            return
        self.metal_type[x, y, z] = metal
        cx, cy = x // CHUNK_SIZE, y // CHUNK_SIZE
        self._dirty_chunks.add((cx, cy, z))

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

    def mark_chunk_dirty(self, cx: int, cy: int, z: int) -> None:
        """Mark a single chunk as needing re-mesh."""
        self._dirty_chunks.add((cx, cy, z))

    def mark_block_dirty(self, x: int, y: int, z: int) -> None:
        """Mark the chunk containing (x,y,z) as dirty, including neighbour z layers."""
        cx, cy = x // CHUNK_SIZE, y // CHUNK_SIZE
        self._dirty_chunks.add((cx, cy, z))
        if z > 0:
            self._dirty_chunks.add((cx, cy, z - 1))
        if z < self.height - 1:
            self._dirty_chunks.add((cx, cy, z + 1))

    def mark_blocks_dirty(self, xs, ys, zs) -> None:
        """Mark chunks for arrays of block positions as dirty.

        *xs*, *ys*, *zs* are NumPy int arrays of equal length.
        """
        if len(xs) == 0:
            return
        cxs = xs // CHUNK_SIZE
        cys = ys // CHUNK_SIZE
        unique = set(zip(cxs.tolist(), cys.tolist(), zs.tolist()))
        self._dirty_chunks.update(unique)
        h = self.height
        for cx_val, cy_val, z_val in list(unique):
            if z_val > 0:
                self._dirty_chunks.add((cx_val, cy_val, z_val - 1))
            if z_val < h - 1:
                self._dirty_chunks.add((cx_val, cy_val, z_val + 1))
