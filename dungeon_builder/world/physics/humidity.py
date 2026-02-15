"""Humidity diffusion physics using vectorized NumPy operations.

Moisture seeps through porous materials over time, creating wet zones
around water sources. Lava voxels generate steam (humidity) in adjacent
blocks. Surface blocks lose humidity through evaporation.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_LAVA,
    VOXEL_WATER,
    VOXEL_POROSITY,
    LAVA_TEMPERATURE,
    HUMIDITY_TICK_INTERVAL,
    HUMIDITY_DIFFUSION_RATE,
    HUMIDITY_SURFACE_LOSS,
    HUMIDITY_SOURCE_LEVEL,
    WATER_HUMIDITY_SOURCE,
    CONVECTION_RATE,
)

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.world.voxel_grid import VoxelGrid


class HumidityPhysics:
    """Diffuses humidity through the voxel grid based on porosity.

    Runs every HUMIDITY_TICK_INTERVAL ticks.
    - Humidity flows between neighbors proportional to min porosity.
    - Lava-adjacent blocks gain humidity (steam).
    - Surface (z=0) loses humidity (evaporation).
    - Humidity is clamped to [0.0, 1.0].
    """

    def __init__(self, event_bus: EventBus, voxel_grid: VoxelGrid) -> None:
        self.event_bus = event_bus
        self.voxel_grid = voxel_grid

        # Pre-build porosity lookup (256 entries for all possible uint8 types)
        self._porosity_lut = np.zeros(256, dtype=np.float32)
        for vtype, poro in VOXEL_POROSITY.items():
            if 0 <= vtype < 256:
                self._porosity_lut[vtype] = poro

        event_bus.subscribe("tick", self._on_tick)

    def _on_tick(self, tick: int, **kw) -> None:
        if tick % HUMIDITY_TICK_INTERVAL != 0:
            return
        self._diffuse()

    def _diffuse(self) -> None:
        """Diffuse humidity and carry heat via convection.

        Humidity follows the discrete heat equation:
          H_new = H + rate·∇²H  (with porosity as diffusivity)
        CFL: HUMIDITY_DIFFUSION_RATE * max_porosity * 6 = 0.05 * 1.0 * 6 = 0.3 < 1

        Convection couples humidity flow to heat transport:
          heat_carry = |hum_flow| * CONVECTION_RATE * T_source
        Max convective heat loss per cell per tick:
          6 * 0.05 * 0.3 * T = 0.09T (well under T)
        Combined with temperature diffusion (max 0.6T loss), total max = 0.69T < T.
        Both systems are CFL-stable, so no per-flow clamping is needed.
        """
        grid = self.voxel_grid
        hum = grid.humidity
        temp = grid.temperature
        voxels = grid.grid

        # Build porosity array from voxel types (vectorized lookup)
        poro = self._porosity_lut[voxels]

        # Compute humidity and heat flow for each of 6 directions
        # Humidity: Flow = (H_neighbor - H_self) * min(poro_self, poro_neighbor) * rate
        # Convection: heat_carry = |hum_flow| * CONVECTION_RATE * T_source
        total_hum_flow = np.zeros_like(hum)
        total_heat_flow = np.zeros_like(temp)

        # +X direction
        # delta > 0 means hum[1:] > hum[:-1], so humidity flows from [1:] to [:-1]
        delta = hum[1:, :, :] - hum[:-1, :, :]
        min_poro = np.minimum(poro[:-1, :, :], poro[1:, :, :])
        flow = delta * min_poro * HUMIDITY_DIFFUSION_RATE
        total_hum_flow[:-1, :, :] += flow
        total_hum_flow[1:, :, :] -= flow
        # Convection: heat carried from the source (block losing humidity)
        # flow > 0 → source is [1:]; flow < 0 → source is [:-1]
        heat_carry = np.abs(flow) * CONVECTION_RATE * np.where(
            flow > 0, temp[1:, :, :], temp[:-1, :, :]
        )
        # Heat moves in the same direction as humidity (conservative: +dest, -source)
        total_heat_flow[:-1, :, :] += np.where(flow > 0, heat_carry, 0.0)
        total_heat_flow[1:, :, :] += np.where(flow < 0, heat_carry, 0.0)
        total_heat_flow[1:, :, :] -= np.where(flow > 0, heat_carry, 0.0)
        total_heat_flow[:-1, :, :] -= np.where(flow < 0, heat_carry, 0.0)

        # +Y direction
        delta = hum[:, 1:, :] - hum[:, :-1, :]
        min_poro = np.minimum(poro[:, :-1, :], poro[:, 1:, :])
        flow = delta * min_poro * HUMIDITY_DIFFUSION_RATE
        total_hum_flow[:, :-1, :] += flow
        total_hum_flow[:, 1:, :] -= flow
        heat_carry = np.abs(flow) * CONVECTION_RATE * np.where(
            flow > 0, temp[:, 1:, :], temp[:, :-1, :]
        )
        total_heat_flow[:, :-1, :] += np.where(flow > 0, heat_carry, 0.0)
        total_heat_flow[:, 1:, :] += np.where(flow < 0, heat_carry, 0.0)
        total_heat_flow[:, 1:, :] -= np.where(flow > 0, heat_carry, 0.0)
        total_heat_flow[:, :-1, :] -= np.where(flow < 0, heat_carry, 0.0)

        # +Z direction
        delta = hum[:, :, 1:] - hum[:, :, :-1]
        min_poro = np.minimum(poro[:, :, :-1], poro[:, :, 1:])
        flow = delta * min_poro * HUMIDITY_DIFFUSION_RATE
        total_hum_flow[:, :, :-1] += flow
        total_hum_flow[:, :, 1:] -= flow
        heat_carry = np.abs(flow) * CONVECTION_RATE * np.where(
            flow > 0, temp[:, :, 1:], temp[:, :, :-1]
        )
        total_heat_flow[:, :, :-1] += np.where(flow > 0, heat_carry, 0.0)
        total_heat_flow[:, :, 1:] += np.where(flow < 0, heat_carry, 0.0)
        total_heat_flow[:, :, 1:] -= np.where(flow > 0, heat_carry, 0.0)
        total_heat_flow[:, :, :-1] -= np.where(flow < 0, heat_carry, 0.0)

        # Apply flows (both are conservative: sum of each total_flow is zero)
        hum += total_hum_flow
        temp += total_heat_flow

        # Surface evaporation (z=0) — environmental sink
        hum[:, :, 0] *= (1.0 - HUMIDITY_SURFACE_LOSS)

        # Steam from lava: blocks adjacent to lava gain humidity
        # (lava is an explicit source — allowed to create humidity)
        lava_mask = voxels == VOXEL_LAVA
        if np.any(lava_mask):
            steam = np.zeros_like(hum)
            # Expand lava mask to 6-connected neighbors
            if grid.width > 1:
                steam[1:, :, :] = np.maximum(steam[1:, :, :], lava_mask[:-1, :, :])
                steam[:-1, :, :] = np.maximum(steam[:-1, :, :], lava_mask[1:, :, :])
            if grid.depth > 1:
                steam[:, 1:, :] = np.maximum(steam[:, 1:, :], lava_mask[:, :-1, :])
                steam[:, :-1, :] = np.maximum(steam[:, :-1, :], lava_mask[:, 1:, :])
            if grid.height > 1:
                steam[:, :, 1:] = np.maximum(steam[:, :, 1:], lava_mask[:, :, :-1])
                steam[:, :, :-1] = np.maximum(steam[:, :, :-1], lava_mask[:, :, 1:])

            # Only non-lava, non-air blocks receive steam
            receives_steam = steam.astype(bool) & (voxels != VOXEL_LAVA) & (voxels != VOXEL_AIR)
            steam_level = HUMIDITY_SOURCE_LEVEL * poro[receives_steam]
            hum[receives_steam] = np.maximum(hum[receives_steam], steam_level)

        # Humidity from water blocks: adjacent blocks gain humidity (scaled by porosity)
        water_mask = voxels == VOXEL_WATER
        if np.any(water_mask):
            water_adj = np.zeros_like(hum, dtype=np.bool_)
            if grid.width > 1:
                water_adj[1:, :, :] |= water_mask[:-1, :, :]
                water_adj[:-1, :, :] |= water_mask[1:, :, :]
            if grid.depth > 1:
                water_adj[:, 1:, :] |= water_mask[:, :-1, :]
                water_adj[:, :-1, :] |= water_mask[:, 1:, :]
            if grid.height > 1:
                water_adj[:, :, 1:] |= water_mask[:, :, :-1]
                water_adj[:, :, :-1] |= water_mask[:, :, 1:]

            # Non-water, non-air blocks adjacent to water gain humidity
            receives_moisture = (
                water_adj
                & (voxels != VOXEL_WATER)
                & (voxels != VOXEL_AIR)
            )
            if np.any(receives_moisture):
                moisture_level = WATER_HUMIDITY_SOURCE * poro[receives_moisture]
                hum[receives_moisture] = np.maximum(
                    hum[receives_moisture], moisture_level
                )

        # Clamp humidity to [0.0, 1.0]
        np.clip(hum, 0.0, 1.0, out=hum)

        # Safety clamp (CFL guarantees non-negativity, but float rounding)
        np.maximum(temp, 0.0, out=temp)
