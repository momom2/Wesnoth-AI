"""Temperature diffusion physics using vectorized NumPy operations."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_LAVA,
    VOXEL_MANA_CRYSTAL,
    VOXEL_HEAT_BEACON,
    VOXEL_STEAM_VENT,
    VOXEL_CONDUCTIVITY,
    LAVA_TEMPERATURE,
    MANA_CRYSTAL_TEMPERATURE,
    HEAT_BEACON_TEMPERATURE,
    STEAM_VENT_HEAT_PULSE,
    STEAM_VENT_RANGE,
    SURFACE_HEAT_LOSS,
    TEMPERATURE_TICK_INTERVAL,
    DIFFUSION_RATE,
    MELTABLE_BLOCKS,
    METAL_MELT_TEMPERATURE,
    ENCHANTED_OFFSET,
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

        # Heat Beacon — fixed-temperature source (like lava but lower temp)
        beacon_mask = voxels == VOXEL_HEAT_BEACON
        if np.any(beacon_mask):
            temp[beacon_mask] = HEAT_BEACON_TEMPERATURE

        # Steam Vent — pulse heat upward through air above
        self._apply_steam_vent_heat(voxels, temp)

        # Metal melting — metallic blocks melt when temp >= melt threshold
        self._check_melting(voxels, temp)

        # Safety clamp (CFL guarantees non-negativity, but float rounding)
        np.maximum(temp, 0.0, out=temp)

    def _apply_steam_vent_heat(
        self, voxels: np.ndarray, temp: np.ndarray,
    ) -> None:
        """Steam vents push heat upward through air cells above them."""
        grid = self.voxel_grid
        vent_positions = np.argwhere(voxels == VOXEL_STEAM_VENT)
        if len(vent_positions) == 0:
            return

        for pos in vent_positions:
            x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
            # Pulse heat upward (z-1 = shallower/up in our coordinate system)
            for dz in range(1, STEAM_VENT_RANGE + 1):
                nz = z - dz
                if nz < 0:
                    break
                if voxels[x, y, nz] != VOXEL_AIR:
                    break  # Blocked by solid
                temp[x, y, nz] += STEAM_VENT_HEAT_PULSE

    def _check_melting(
        self, voxels: np.ndarray, temp: np.ndarray,
    ) -> None:
        """Melt metallic blocks when temperature exceeds their melt threshold.

        Enchanted blocks (metal_type & ENCHANTED_OFFSET) are immune.
        Melted blocks become air.
        """
        grid = self.voxel_grid
        metal_type_arr = grid.metal_type
        melted_any = False

        for vtype in MELTABLE_BLOCKS:
            mask = (voxels == vtype)
            if not np.any(mask):
                continue

            mt = metal_type_arr[mask]
            # Enchanted blocks are immune
            enchanted = (mt & ENCHANTED_OFFSET) != 0
            base = mt & 0x7F

            # Look up melt temperature for each base metal
            melt_temps = np.array(
                [METAL_MELT_TEMPERATURE.get(int(b), 99999.0) for b in base],
                dtype=np.float32,
            )
            melt_temps[enchanted] = 99999.0  # Immune

            # Check which blocks should melt
            should_melt = temp[mask] >= melt_temps
            if not np.any(should_melt):
                continue

            # Apply melting: set to AIR
            positions = np.argwhere(mask)
            for i, pos in enumerate(positions):
                if should_melt[i]:
                    x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
                    grid.set(x, y, z, VOXEL_AIR, event_bus=self.event_bus)
                    melted_any = True

        if melted_any:
            self.event_bus.publish("metal_melted")
