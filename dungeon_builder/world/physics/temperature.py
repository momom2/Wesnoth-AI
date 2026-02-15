"""Temperature diffusion physics using vectorized NumPy operations."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from dungeon_builder.config import (
    VOXEL_LAVA,
    VOXEL_MANA_CRYSTAL,
    VOXEL_CONDUCTIVITY,
    LAVA_TEMPERATURE,
    MANA_CRYSTAL_TEMPERATURE,
    SURFACE_HEAT_LOSS,
    TEMPERATURE_TICK_INTERVAL,
    DIFFUSION_RATE,
)

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.world.voxel_grid import VoxelGrid


class TemperaturePhysics:
    """Diffuses heat through the voxel grid based on conductivity.

    Runs every TEMPERATURE_TICK_INTERVAL ticks.
    - Lava voxels stay at LAVA_TEMPERATURE.
    - Mana crystals stay at MANA_CRYSTAL_TEMPERATURE.
    - Surface (z=0) loses heat each tick.
    - Heat flows between neighbours proportional to min(cond_self, cond_neighbor).
    """

    def __init__(self, event_bus: EventBus, voxel_grid: VoxelGrid) -> None:
        self.event_bus = event_bus
        self.voxel_grid = voxel_grid

        # Pre-build conductivity lookup (256 entries for all possible uint8 types)
        self._cond_lut = np.zeros(256, dtype=np.float32)
        for vtype, cond in VOXEL_CONDUCTIVITY.items():
            if 0 <= vtype < 256:
                self._cond_lut[vtype] = cond

        event_bus.subscribe("tick", self._on_tick)

    def _on_tick(self, tick: int) -> None:
        if tick % TEMPERATURE_TICK_INTERVAL != 0:
            return
        self._diffuse()

    def _diffuse(self) -> None:
        """Discrete heat equation: T_new = T + α·Δt·∇²T.

        With α·Δt = DIFFUSION_RATE * min(cond_a, cond_b) and 6 neighbors,
        the CFL condition requires DIFFUSION_RATE * max_cond * 6 < 1.0.
        Currently 0.1 * 1.0 * 6 = 0.6 < 1, so the scheme is unconditionally
        stable and non-negative — no per-flow clamping needed.
        """
        grid = self.voxel_grid
        temp = grid.temperature
        voxels = grid.grid

        # Build conductivity array from voxel types (vectorized lookup)
        cond = self._cond_lut[voxels]

        # Compute heat flow for each of 6 directions using explicit slicing
        # Flow = (T_neighbor - T_self) * min(cond_self, cond_neighbor) * rate
        total_flow = np.zeros_like(temp)

        # +X direction
        delta = temp[1:, :, :] - temp[:-1, :, :]
        min_cond = np.minimum(cond[:-1, :, :], cond[1:, :, :])
        flow = delta * min_cond * DIFFUSION_RATE
        total_flow[:-1, :, :] += flow
        total_flow[1:, :, :] -= flow

        # +Y direction
        delta = temp[:, 1:, :] - temp[:, :-1, :]
        min_cond = np.minimum(cond[:, :-1, :], cond[:, 1:, :])
        flow = delta * min_cond * DIFFUSION_RATE
        total_flow[:, :-1, :] += flow
        total_flow[:, 1:, :] -= flow

        # +Z direction
        delta = temp[:, :, 1:] - temp[:, :, :-1]
        min_cond = np.minimum(cond[:, :, :-1], cond[:, :, 1:])
        flow = delta * min_cond * DIFFUSION_RATE
        total_flow[:, :, :-1] += flow
        total_flow[:, :, 1:] -= flow

        # Apply flow (conservative: sum of total_flow is zero)
        temp += total_flow

        # Surface heat loss (z=0) — environmental sink
        temp[:, :, 0] *= (1.0 - SURFACE_HEAT_LOSS)

        # Fixed-temperature voxels — explicit sources/sinks
        lava_mask = voxels == VOXEL_LAVA
        temp[lava_mask] = LAVA_TEMPERATURE

        mana_mask = voxels == VOXEL_MANA_CRYSTAL
        temp[mana_mask] = MANA_CRYSTAL_TEMPERATURE

        # Safety clamp (CFL guarantees non-negativity, but float rounding)
        np.maximum(temp, 0.0, out=temp)
