"""Gravity physics: loose blocks fall, disconnected blocks become loose."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from dungeon_builder.config import (
    CHUNK_SIZE,
    VOXEL_AIR,
    STRUCTURAL_ANCHORS,
    GRAVITY_TICK_INTERVAL,
    CONNECTIVITY_TICK_INTERVAL,
    MAX_FALL_PER_TICK,
)

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.world.voxel_grid import VoxelGrid


class GravityPhysics:
    """Handles gravity for loose blocks and structural connectivity.

    Two responsibilities:
    1. Every GRAVITY_TICK_INTERVAL ticks: loose blocks fall through air.
    2. Every CONNECTIVITY_TICK_INTERVAL ticks: flood-fill from anchors
       to detect disconnected solid masses, which become loose.
    """

    def __init__(self, event_bus: EventBus, voxel_grid: VoxelGrid) -> None:
        self.event_bus = event_bus
        self.voxel_grid = voxel_grid

        # Pre-build anchor lookup (True if anchor type)
        self._anchor_lut = np.zeros(256, dtype=np.bool_)
        for vtype in STRUCTURAL_ANCHORS:
            if 0 <= vtype < 256:
                self._anchor_lut[vtype] = True

        event_bus.subscribe("tick", self._on_tick)

    def _on_tick(self, tick: int, **kw) -> None:
        if tick % GRAVITY_TICK_INTERVAL == 0:
            self._process_falling()
        if tick % CONNECTIVITY_TICK_INTERVAL == 0:
            self._check_connectivity()

    def _process_falling(self) -> None:
        """Move loose blocks downward through air."""
        grid = self.voxel_grid
        voxels = grid.grid
        loose = grid.loose
        temp = grid.temperature
        humidity = grid.humidity

        fell_any = False

        for _ in range(MAX_FALL_PER_TICK):
            # Boolean mask: non-air, loose blocks with air directly below
            # z+1 is deeper (downward in our coordinate system)
            can_fall = (
                (voxels[:, :, :-1] != VOXEL_AIR)
                & loose[:, :, :-1]
                & (voxels[:, :, 1:] == VOXEL_AIR)
            )

            if not np.any(can_fall):
                break

            fell_any = True

            # Copy falling blocks downward
            voxels[:, :, 1:][can_fall] = voxels[:, :, :-1][can_fall]
            loose[:, :, 1:][can_fall] = True
            temp[:, :, 1:][can_fall] = temp[:, :, :-1][can_fall]
            humidity[:, :, 1:][can_fall] = humidity[:, :, :-1][can_fall]

            # Clear source positions
            voxels[:, :, :-1][can_fall] = VOXEL_AIR
            loose[:, :, :-1][can_fall] = False
            temp[:, :, :-1][can_fall] = 0.0
            humidity[:, :, :-1][can_fall] = 0.0

        if fell_any:
            grid.mark_all_dirty()
            self.event_bus.publish("blocks_fell")

    def _check_connectivity(self) -> None:
        """Mark blocks as loose if not connected to any anchor."""
        grid = self.voxel_grid
        voxels = grid.grid
        loose = grid.loose

        # Solid, non-loose blocks need connectivity checking
        solid_nonloose = (voxels != VOXEL_AIR) & (~loose)

        # Seed: anchor blocks are always connected
        connected = self._anchor_lut[voxels].copy()

        # Iterative 6-connected dilation through solid non-loose blocks
        max_iter = grid.width + grid.depth + grid.height

        for _ in range(max_iter):
            expanded = connected.copy()

            # Expand in 6 directions
            expanded[1:, :, :] |= connected[:-1, :, :]
            expanded[:-1, :, :] |= connected[1:, :, :]
            expanded[:, 1:, :] |= connected[:, :-1, :]
            expanded[:, :-1, :] |= connected[:, 1:, :]
            expanded[:, :, 1:] |= connected[:, :, :-1]
            expanded[:, :, :-1] |= connected[:, :, 1:]

            # Only keep expansion into solid non-loose blocks
            expanded &= solid_nonloose

            if np.array_equal(expanded, connected):
                break

            connected = expanded

        # Any solid non-loose block not connected becomes loose
        newly_loose = solid_nonloose & (~connected)

        if np.any(newly_loose):
            loose[newly_loose] = True
            grid.mark_all_dirty()
            count = int(np.sum(newly_loose))
            self.event_bus.publish("structural_disconnect", count=count)
