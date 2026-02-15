"""Line-of-sight and special vision systems for intruders.

Vision is based on **straight-line Bresenham ray-casting**, not Manhattan
distance.  Rays are blocked by solid opaque blocks (stone, dirt, etc.)
but pass through air, open doors, slopes, and stairs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_DOOR,
    VOXEL_SLOPE,
    VOXEL_STAIRS,
    VOXEL_WATER,
    VOXEL_LAVA,
    WATER_LOS_DEPTH,
)

if TYPE_CHECKING:
    from dungeon_builder.world.voxel_grid import VoxelGrid


# Voxel types that are always transparent to LOS
_TRANSPARENT = frozenset({VOXEL_AIR, VOXEL_SLOPE, VOXEL_STAIRS})


def bresenham_3d(
    x0: int, y0: int, z0: int,
    x1: int, y1: int, z1: int,
) -> list[tuple[int, int, int]]:
    """3D Bresenham line from (x0,y0,z0) to (x1,y1,z1).

    Returns the full list of cells along the ray, including both endpoints.
    """
    cells: list[tuple[int, int, int]] = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)

    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    sz = 1 if z1 > z0 else -1

    # Driving axis is the one with the largest delta
    if dx >= dy and dx >= dz:
        # X-driven
        ey = 2 * dy - dx
        ez = 2 * dz - dx
        x, y, z = x0, y0, z0
        for _ in range(dx + 1):
            cells.append((x, y, z))
            if ey > 0:
                y += sy
                ey -= 2 * dx
            if ez > 0:
                z += sz
                ez -= 2 * dx
            ey += 2 * dy
            ez += 2 * dz
            x += sx
    elif dy >= dx and dy >= dz:
        # Y-driven
        ex = 2 * dx - dy
        ez = 2 * dz - dy
        x, y, z = x0, y0, z0
        for _ in range(dy + 1):
            cells.append((x, y, z))
            if ex > 0:
                x += sx
                ex -= 2 * dy
            if ez > 0:
                z += sz
                ez -= 2 * dy
            ex += 2 * dx
            ez += 2 * dz
            y += sy
    else:
        # Z-driven
        ex = 2 * dx - dz
        ey = 2 * dy - dz
        x, y, z = x0, y0, z0
        for _ in range(dz + 1):
            cells.append((x, y, z))
            if ex > 0:
                x += sx
                ex -= 2 * dz
            if ey > 0:
                y += sy
                ey -= 2 * dz
            ex += 2 * dx
            ey += 2 * dy
            z += sz

    return cells


def _is_los_transparent(
    voxel_grid: VoxelGrid,
    x: int, y: int, z: int,
) -> bool:
    """Return True if LOS passes through this cell.

    * Air, slopes, stairs are always transparent.
    * Open doors (block_state == 0) are transparent; closed doors block.
    * Water is semi-transparent (counted separately for depth check).
    """
    if not voxel_grid.in_bounds(x, y, z):
        return False

    vtype = voxel_grid.get(x, y, z)
    if vtype in _TRANSPARENT:
        return True
    if vtype == VOXEL_DOOR and voxel_grid.block_state[x, y, z] == 0:
        return True
    # Water is handled separately — callers use _is_water
    return False


def _is_water(voxel_grid: VoxelGrid, x: int, y: int, z: int) -> bool:
    if not voxel_grid.in_bounds(x, y, z):
        return False
    return voxel_grid.get(x, y, z) == VOXEL_WATER


def compute_los(
    voxel_grid: VoxelGrid,
    ox: int, oy: int, oz: int,
    perception_range: int,
) -> set[tuple[int, int, int]]:
    """Compute all cells visible from *(ox, oy, oz)* via straight-line LOS.

    Casts rays to every cell on the surface of a cube of side
    ``2 * perception_range + 1`` centered on the origin.  Each ray walks
    the 3D Bresenham line; opaque cells stop the ray.

    Water is semi-transparent: LOS passes through at most
    :data:`WATER_LOS_DEPTH` consecutive water cells before being blocked.
    """
    visible: set[tuple[int, int, int]] = set()
    visible.add((ox, oy, oz))

    r = perception_range
    # Collect unique target cells on the surface of the cube
    targets: set[tuple[int, int, int]] = set()
    for x in range(ox - r, ox + r + 1):
        for y in range(oy - r, oy + r + 1):
            for z in range(oz - r, oz + r + 1):
                if abs(x - ox) == r or abs(y - oy) == r or abs(z - oz) == r:
                    targets.add((x, y, z))

    for tx, ty, tz in targets:
        ray = bresenham_3d(ox, oy, oz, tx, ty, tz)
        water_count = 0
        for cx, cy, cz in ray:
            if (cx, cy, cz) == (ox, oy, oz):
                continue  # Skip origin
            if not voxel_grid.in_bounds(cx, cy, cz):
                break  # Out of bounds — stop this ray

            if _is_water(voxel_grid, cx, cy, cz):
                water_count += 1
                if water_count > WATER_LOS_DEPTH:
                    break  # Too much water, ray blocked
                visible.add((cx, cy, cz))
            elif _is_los_transparent(voxel_grid, cx, cy, cz):
                water_count = 0
                visible.add((cx, cy, cz))
            else:
                # Opaque solid — mark it visible (you can see the wall face)
                # but stop the ray beyond it
                visible.add((cx, cy, cz))
                break

    return visible


def compute_arcane_sight(
    voxel_grid: VoxelGrid,
    ox: int, oy: int, oz: int,
    arcane_range: int,
) -> set[tuple[int, int, int]]:
    """Compute cells visible via arcane sight (Gloomseer).

    Arcane sight uses **Manhattan distance** and ignores all blocks — the
    intruder can see voxel types through solid walls within *arcane_range*.
    """
    visible: set[tuple[int, int, int]] = set()
    for x in range(ox - arcane_range, ox + arcane_range + 1):
        for y in range(oy - arcane_range, oy + arcane_range + 1):
            for z in range(oz - arcane_range, oz + arcane_range + 1):
                if abs(x - ox) + abs(y - oy) + abs(z - oz) <= arcane_range:
                    if voxel_grid.in_bounds(x, y, z):
                        visible.add((x, y, z))
    return visible


def compute_thermal_vision(
    voxel_grid: VoxelGrid,
    ox: int, oy: int, oz: int,
    thermal_range: int,
    heat_threshold: float = 100.0,
) -> set[tuple[int, int, int]]:
    """Compute cells visible via thermal vision (Pyremancer).

    Sees through walls within *thermal_range* (Manhattan), but only reveals
    cells whose temperature exceeds *heat_threshold*.
    """
    visible: set[tuple[int, int, int]] = set()
    for x in range(ox - thermal_range, ox + thermal_range + 1):
        for y in range(oy - thermal_range, oy + thermal_range + 1):
            for z in range(oz - thermal_range, oz + thermal_range + 1):
                if abs(x - ox) + abs(y - oy) + abs(z - oz) <= thermal_range:
                    if voxel_grid.in_bounds(x, y, z):
                        if voxel_grid.temperature[x, y, z] >= heat_threshold:
                            visible.add((x, y, z))
    return visible
