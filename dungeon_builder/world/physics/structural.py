"""Structural integrity physics: load distribution and collapse detection.

Models real architectural principles:
- Compression: load accumulates downward through columns
- Arches: air gaps redirect load laterally to flanking blocks
- Buttressing: adjacent solid blocks reduce effective transmitted load
- Cantilevers: unsupported spans limited to ~2-3 blocks
- Foundation: deeper blocks bear accumulated weight from above
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_WEIGHT,
    VOXEL_MAX_LOAD,
    STRUCTURAL_ANCHORS,
    STRUCTURAL_TICK_INTERVAL,
    MAX_CASCADE_PER_TICK,
    LOAD_DIST_BELOW,
    LOAD_DIST_LATERAL,
    BUTTRESS_FACTOR,
    HUMIDITY_WEAKNESS,
    TEMP_WEAKNESS_MIN,
    TEMP_WEAKNESS_MAX,
    TEMP_WEAKNESS_FACTOR,
)

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.world.voxel_grid import VoxelGrid


class StructuralIntegrityPhysics:
    """Computes load distribution and detects structural failure.

    Load flows downward (+z direction) with lateral distribution.
    When accumulated load exceeds a block's capacity (modified by
    temperature and humidity), the block becomes loose.
    """

    def __init__(self, event_bus: EventBus, voxel_grid: VoxelGrid) -> None:
        self.event_bus = event_bus
        self.voxel_grid = voxel_grid

        # Pre-build LUTs (256 entries each)
        self._weight_lut = np.zeros(256, dtype=np.float32)
        self._capacity_lut = np.zeros(256, dtype=np.float32)
        self._anchor_lut = np.zeros(256, dtype=np.bool_)

        for vtype, w in VOXEL_WEIGHT.items():
            if 0 <= vtype < 256:
                self._weight_lut[vtype] = w

        for vtype, cap in VOXEL_MAX_LOAD.items():
            if 0 <= vtype < 256:
                # Use large finite value for inf (NumPy operations)
                self._capacity_lut[vtype] = min(cap, 1e9)

        for vtype in STRUCTURAL_ANCHORS:
            if 0 <= vtype < 256:
                self._anchor_lut[vtype] = True

        event_bus.subscribe("tick", self._on_tick)

    def _on_tick(self, tick: int, **kw) -> None:
        if tick % STRUCTURAL_TICK_INTERVAL != 0:
            return
        self._compute_load()
        self._check_failures()

    def _compute_load(self) -> None:
        """Top-down load accumulation with lateral distribution."""
        grid = self.voxel_grid
        voxels = grid.grid
        loose = grid.loose
        w, d, h = grid.width, grid.depth, grid.height

        # Weight of each block from LUT
        self_weight = self._weight_lut[voxels]

        # Solid mask (non-air, non-loose blocks bear load)
        solid = (voxels != VOXEL_AIR) & (~loose)

        # Anchor mask
        anchors = self._anchor_lut[voxels]

        # Accumulated load array (reset each computation)
        load = np.zeros((w, d, h), dtype=np.float32)

        # Process top-to-bottom (z=0 is surface, z=h-1 is deepest)
        for z in range(h):
            # Each block starts with its own weight
            load[:, :, z] += self_weight[:, :, z]

            # Anchors absorb all load
            load[:, :, z][anchors[:, :, z]] = 0.0

            # Non-solid blocks can't transmit load
            load[:, :, z][~solid[:, :, z]] = 0.0

            if z >= h - 1:
                continue

            current_load = load[:, :, z]

            # Buttressing: count solid cardinal neighbors at same z-level
            solid_z = solid[:, :, z].astype(np.float32)
            neighbor_count = np.zeros((w, d), dtype=np.float32)
            neighbor_count[1:, :] += solid_z[:-1, :]
            neighbor_count[:-1, :] += solid_z[1:, :]
            neighbor_count[:, 1:] += solid_z[:, :-1]
            neighbor_count[:, :-1] += solid_z[:, 1:]

            buttress_mult = np.clip(
                1.0 - neighbor_count * BUTTRESS_FACTOR, 0.5, 1.0
            )
            effective_load = current_load * buttress_mult

            # Direct downward distribution
            direct_below = effective_load * LOAD_DIST_BELOW
            solid_below = solid[:, :, z + 1]

            # Track how much load was successfully distributed
            distributed = np.zeros((w, d), dtype=np.float32)

            direct_accepted = np.where(solid_below, direct_below, 0.0)
            load[:, :, z + 1] += direct_accepted
            distributed += direct_accepted

            # Lateral-below distribution (4 cardinal-below neighbors)
            lateral = effective_load * LOAD_DIST_LATERAL
            if w > 1:
                mask = solid[1:, :, z + 1]
                contrib = np.where(mask, lateral[:-1, :], 0.0)
                load[1:, :, z + 1] += contrib
                distributed[:-1, :] += contrib
                mask = solid[:-1, :, z + 1]
                contrib = np.where(mask, lateral[1:, :], 0.0)
                load[:-1, :, z + 1] += contrib
                distributed[1:, :] += contrib
            if d > 1:
                mask = solid[:, 1:, z + 1]
                contrib = np.where(mask, lateral[:, :-1], 0.0)
                load[:, 1:, z + 1] += contrib
                distributed[:, :-1] += contrib
                mask = solid[:, :-1, z + 1]
                contrib = np.where(mask, lateral[:, 1:], 0.0)
                load[:, :-1, z + 1] += contrib
                distributed[:, 1:] += contrib

            # Arch effect: if block directly below is air, redistribute
            # 25% of direct load laterally to cardinal-below neighbors
            air_below = (~solid_below) & (voxels[:, :, z + 1] == VOXEL_AIR)
            redistributed = np.where(air_below, direct_below * 0.25, 0.0)
            if w > 1:
                contrib = np.where(
                    solid[1:, :, z + 1], redistributed[:-1, :], 0.0
                )
                load[1:, :, z + 1] += contrib
                distributed[:-1, :] += contrib
                contrib = np.where(
                    solid[:-1, :, z + 1], redistributed[1:, :], 0.0
                )
                load[:-1, :, z + 1] += contrib
                distributed[1:, :] += contrib
            if d > 1:
                contrib = np.where(
                    solid[:, 1:, z + 1], redistributed[:, :-1], 0.0
                )
                load[:, 1:, z + 1] += contrib
                distributed[:, :-1] += contrib
                contrib = np.where(
                    solid[:, :-1, z + 1], redistributed[:, 1:], 0.0
                )
                load[:, :-1, z + 1] += contrib
                distributed[:, 1:] += contrib

            # Undistributed load stays on the current block as retained stress.
            # This models cantilever stress: a block over air with no supports
            # bears all the weight itself.
            intended = effective_load * (LOAD_DIST_BELOW + 4 * LOAD_DIST_LATERAL)
            undistributed = np.maximum(intended - distributed, 0.0)
            load[:, :, z] += undistributed

        grid.load[:] = load

    def _compute_effective_capacity(self) -> np.ndarray:
        """Compute capacity modified by temperature and humidity."""
        grid = self.voxel_grid
        voxels = grid.grid

        base_cap = self._capacity_lut[voxels]

        # Humidity modifier: capacity * (1 - humidity * HUMIDITY_WEAKNESS)
        humidity_mod = np.clip(
            1.0 - grid.humidity * HUMIDITY_WEAKNESS, 0.5, 1.0
        )

        # Temperature modifier: linear ramp between threshold and max
        temp_factor = np.clip(
            (grid.temperature - TEMP_WEAKNESS_MIN)
            / (TEMP_WEAKNESS_MAX - TEMP_WEAKNESS_MIN),
            0.0,
            1.0,
        )
        temp_mod = 1.0 - temp_factor * TEMP_WEAKNESS_FACTOR

        return base_cap * humidity_mod * temp_mod

    def _check_failures(self) -> None:
        """Compare load against capacity. Overloaded blocks become loose."""
        grid = self.voxel_grid
        voxels = grid.grid
        loose = grid.loose

        effective_cap = self._compute_effective_capacity()
        load = grid.load

        # Compute stress ratio for rendering
        safe_cap = np.where(effective_cap > 0, effective_cap, 1.0)
        grid.stress_ratio[:] = np.where(
            effective_cap > 0, load / safe_cap, 0.0
        )

        # Find overloaded blocks
        solid = (voxels != VOXEL_AIR) & (~loose)
        anchors = self._anchor_lut[voxels]
        overloaded = (
            (load > effective_cap) & solid & (~anchors) & (effective_cap > 0)
        )

        if not np.any(overloaded):
            return

        # Cap cascading: prioritize highest stress first
        xs, ys, zs = np.where(overloaded)
        count = min(len(xs), MAX_CASCADE_PER_TICK)

        if len(xs) > MAX_CASCADE_PER_TICK:
            stress_vals = grid.stress_ratio[xs, ys, zs]
            worst_indices = np.argpartition(stress_vals, -count)[-count:]
            xs, ys, zs = xs[worst_indices], ys[worst_indices], zs[worst_indices]

        loose[xs, ys, zs] = True
        grid.mark_all_dirty()

        self.event_bus.publish("structural_failure", count=int(count))
