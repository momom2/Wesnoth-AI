"""Claimed territory: 3D flood-fill from core through traversable voxels.

Air, water, and player-built functional blocks (slopes, stairs, doors,
spikes, treasure, tarps, rolling stones, reinforced walls) reachable from
the dungeon core via 6-connected flood-fill are marked as "claimed".
Natural solid blocks and lava act as barriers.  Solid blocks adjacent to
claimed territory are marked as "visible" (for fog-of-war rendering).

Follows the same iterative NumPy dilation pattern as
``GravityPhysics._check_connectivity()`` in ``gravity.py``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

import dungeon_builder.config as _cfg
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_WATER,
    VOXEL_SLOPE,
    VOXEL_STAIRS,
    VOXEL_DOOR,
    VOXEL_TARP,
    VOXEL_SPIKE,
    VOXEL_TREASURE,
    VOXEL_ROLLING_STONE,
    VOXEL_REINFORCED_WALL,
    CORE_X,
    CORE_Y,
    CORE_Z,
    CLAIMED_TICK_INTERVAL,
)

# Functional blocks that claiming can propagate through
_CLAIM_TRAVERSABLE = frozenset((
    VOXEL_AIR, VOXEL_WATER,
    VOXEL_SLOPE, VOXEL_STAIRS, VOXEL_DOOR, VOXEL_TARP,
    VOXEL_SPIKE, VOXEL_TREASURE, VOXEL_ROLLING_STONE,
    VOXEL_REINFORCED_WALL,
))

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.world.voxel_grid import VoxelGrid

logger = logging.getLogger("dungeon_builder.claimed_territory")


class ClaimedTerritorySystem:
    """Computes claimed territory via 6-connected flood-fill from core.

    Propagates through air, water, and functional blocks.
    Runs every ``CLAIMED_TICK_INTERVAL`` ticks.  After computing claimed
    territory, it also computes the *visible* mask for solid blocks: a
    solid block is visible if any of its 6 neighbours is claimed.
    """

    def __init__(
        self,
        event_bus: EventBus,
        voxel_grid: VoxelGrid,
        core_x: int = CORE_X,
        core_y: int = CORE_Y,
        core_z: int = CORE_Z,
    ) -> None:
        self.event_bus = event_bus
        self.voxel_grid = voxel_grid
        self.core_x = core_x
        self.core_y = core_y
        self.core_z = core_z

        event_bus.subscribe("tick", self._on_tick)

        # Initial computation
        self.recompute()

    def _on_tick(self, tick: int, **kw) -> None:
        if tick % CLAIMED_TICK_INTERVAL == 0:
            self.recompute()

    def recompute(self) -> None:
        """Full recomputation of claimed territory and visibility."""
        grid = self.voxel_grid
        voxels = grid.grid
        w, d, h = grid.width, grid.depth, grid.height

        # Traversable mask: air, water, and functional blocks propagate claims
        traversable = np.zeros_like(voxels, dtype=np.bool_)
        for vtype in _CLAIM_TRAVERSABLE:
            traversable |= (voxels == vtype)

        # Seed from air cells 6-adjacent to the core block
        cx, cy, cz = self.core_x, self.core_y, self.core_z
        seed = np.zeros((w, d, h), dtype=np.bool_)

        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0),
                           (0, 1, 0), (0, -1, 0),
                           (0, 0, 1), (0, 0, -1)):
            nx, ny, nz = cx + dx, cy + dy, cz + dz
            if 0 <= nx < w and 0 <= ny < d and 0 <= nz < h:
                if traversable[nx, ny, nz]:
                    seed[nx, ny, nz] = True

        if not np.any(seed):
            grid.claimed[:] = False
            grid.visible[:] = False
            # Core itself is always visible
            if grid.in_bounds(cx, cy, cz):
                grid.visible[cx, cy, cz] = True
            return

        # Iterative 6-connected dilation through air
        connected = seed.copy()
        max_iter = w + d + h

        for _ in range(max_iter):
            expanded = connected.copy()

            if w > 1:
                expanded[1:, :, :] |= connected[:-1, :, :]
                expanded[:-1, :, :] |= connected[1:, :, :]
            if d > 1:
                expanded[:, 1:, :] |= connected[:, :-1, :]
                expanded[:, :-1, :] |= connected[:, 1:, :]
            if h > 1:
                expanded[:, :, 1:] |= connected[:, :, :-1]
                expanded[:, :, :-1] |= connected[:, :, 1:]

            # Only keep expansion into traversable cells
            expanded &= traversable

            if np.array_equal(expanded, connected):
                break

            connected = expanded

        # Store claimed territory
        old_claimed = grid.claimed.copy()
        grid.claimed[:] = connected

        # Compute visibility: solid blocks adjacent to at least one claimed cell
        # The "air-only" mask is needed so natural solids adjacent to claimed
        # air/functional blocks are visible (fog-of-war boundary).
        air_mask = (voxels == VOXEL_AIR) | (voxels == VOXEL_WATER)
        vis = np.zeros((w, d, h), dtype=np.bool_)
        if w > 1:
            vis[1:, :, :] |= connected[:-1, :, :]
            vis[:-1, :, :] |= connected[1:, :, :]
        if d > 1:
            vis[:, 1:, :] |= connected[:, :-1, :]
            vis[:, :-1, :] |= connected[:, 1:, :]
        if h > 1:
            vis[:, :, 1:] |= connected[:, :, :-1]
            vis[:, :, :-1] |= connected[:, :, 1:]

        # Natural solids adjacent to claimed territory are visible
        vis &= ~air_mask
        # Functional blocks that are themselves claimed are also visible
        vis |= (connected & ~air_mask)

        # Core block is always visible
        if grid.in_bounds(cx, cy, cz):
            vis[cx, cy, cz] = True

        # Ore x-ray dilation: ores/crystals within PLAYER_XRAY_RANGE of
        # visible blocks become visible (seeps through solid stone).
        xray_range = _cfg.PLAYER_XRAY_RANGE
        if xray_range > 0 and _cfg.XRAY_VISIBLE_TYPES:
            ore_mask = np.zeros((w, d, h), dtype=np.bool_)
            for vtype in _cfg.XRAY_VISIBLE_TYPES:
                ore_mask |= (voxels == vtype)
            xray_front = vis.copy()
            for _ in range(xray_range):
                expanded = xray_front.copy()
                # 6-connected dilation through solid blocks only
                if w > 1:
                    expanded[1:] |= xray_front[:-1]
                    expanded[:-1] |= xray_front[1:]
                if d > 1:
                    expanded[:, 1:] |= xray_front[:, :-1]
                    expanded[:, :-1] |= xray_front[:, 1:]
                if h > 1:
                    expanded[:, :, 1:] |= xray_front[:, :, :-1]
                    expanded[:, :, :-1] |= xray_front[:, :, 1:]
                # Only expand through solid (not air/water)
                expanded &= ~air_mask
                xray_front = expanded
            vis |= (xray_front & ore_mask)

        grid.visible[:] = vis

        # If claimed territory changed, trigger chunk re-renders
        if not np.array_equal(old_claimed, connected):
            grid.mark_all_dirty()
            self.event_bus.publish("claimed_territory_changed")
            claimed_count = int(np.sum(connected))
            logger.debug("Claimed territory updated: %d air cells", claimed_count)
