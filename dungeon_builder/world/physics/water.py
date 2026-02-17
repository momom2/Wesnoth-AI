"""Water physics: cellular automata fluid flow, pressure, and lava interaction.

Water flows downward through air (gravity), levels laterally between
adjacent water blocks, seeps humidity through porous solids, and exerts
hydrostatic pressure on adjacent walls.  When water meets lava both are
consumed and obsidian is produced.

Water level is tracked per-voxel as a uint8 (0-255), where 0 = empty
and 255 = full block.  A voxel is typed VOXEL_WATER when water_level > 0
and typed VOXEL_AIR when water_level reaches 0.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_WATER,
    VOXEL_LAVA,
    VOXEL_DOOR,
    VOXEL_FLOODGATE,
    VOXEL_POROSITY,
    VOXEL_SHEAR_STRENGTH,
    VOXEL_WEIGHT,
    WATER_TICK_INTERVAL,
    WATER_FLOW_RATE,
    WATER_SEEP_RATE,
    WATER_PRESSURE_WEIGHT,
    WATER_BURST_FACTOR,
    WATER_HUMIDITY_SOURCE,
    WATER_TEMPERATURE,
    WATER_EVAPORATION_RATE,
    MAX_WATER_FLOW_PER_TICK,
    WATER_LAVA_PRODUCT,
    MAX_CASCADE_PER_TICK,
    METAL_STRENGTH_MULT,
    ENCHANTED_OFFSET,
)

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.world.voxel_grid import VoxelGrid


class WaterPhysics:
    """Cellular automata water flow, pressure mechanics, and lava interaction.

    Runs every WATER_TICK_INTERVAL ticks.
    - Downward flow: water above air falls (gravity)
    - Lateral leveling: water seeks its own level
    - Seepage: water increases humidity of adjacent porous solids
    - Pressure: column depth exerts shear load on adjacent solid walls
    - Burst: if pressure > shear_strength * BURST_FACTOR, wall bursts
    - Lava interaction: water + lava = obsidian + steam
    """

    def __init__(self, event_bus: EventBus, voxel_grid: VoxelGrid) -> None:
        self.event_bus = event_bus
        self.voxel_grid = voxel_grid

        # Pre-build LUTs
        self._porosity_lut = np.zeros(256, dtype=np.float32)
        for vtype, poro in VOXEL_POROSITY.items():
            if 0 <= vtype < 256:
                self._porosity_lut[vtype] = poro

        self._shear_lut = np.zeros(256, dtype=np.float32)
        for vtype, shear in VOXEL_SHEAR_STRENGTH.items():
            if 0 <= vtype < 256:
                self._shear_lut[vtype] = min(shear, 1e9)

        self._weight_lut = np.zeros(256, dtype=np.float32)
        for vtype, w in VOXEL_WEIGHT.items():
            if 0 <= vtype < 256:
                self._weight_lut[vtype] = w

        event_bus.subscribe("tick", self._on_tick)

    def _on_tick(self, tick: int, **kw) -> None:
        if tick % WATER_TICK_INTERVAL != 0:
            return
        self._lava_interaction()
        self._flow()
        self._apply_seepage()
        self._apply_pressure()
        self._surface_evaporation()
        self._cleanup()

    # ------------------------------------------------------------------
    # Flow
    # ------------------------------------------------------------------

    def _flow(self) -> None:
        """Gravity-driven downward flow + lateral leveling."""
        grid = self.voxel_grid
        voxels = grid.grid
        water = grid.water_level
        w, d, h = grid.width, grid.depth, grid.height

        flowed = False

        for _ in range(MAX_WATER_FLOW_PER_TICK):
            moved = False

            # --- Downward flow (z increases = deeper) ---
            if h > 1:
                # Water above air (or open floodgate): transfer down
                water_above = (voxels[:, :, :-1] == VOXEL_WATER)
                air_below = (voxels[:, :, 1:] == VOXEL_AIR)
                gate_below = (
                    (voxels[:, :, 1:] == VOXEL_FLOODGATE)
                    & (grid.block_state[:, :, 1:] == 0)
                )
                can_flow_down = water_above & (air_below | gate_below)

                if np.any(can_flow_down):
                    moved = True
                    # Transfer: target below gets source's water level
                    level_above = water[:, :, :-1].copy()
                    transfer = np.where(can_flow_down, level_above, np.uint8(0))

                    # Add to below
                    water[:, :, 1:] = np.where(
                        can_flow_down,
                        np.minimum(
                            water[:, :, 1:].astype(np.int16) + transfer.astype(np.int16),
                            255,
                        ).astype(np.uint8),
                        water[:, :, 1:],
                    )
                    voxels[:, :, 1:] = np.where(
                        can_flow_down & (water[:, :, 1:] > 0),
                        VOXEL_WATER,
                        voxels[:, :, 1:],
                    )

                    # Remove from above
                    water[:, :, :-1] = np.where(can_flow_down, np.uint8(0), water[:, :, :-1])
                    voxels[:, :, :-1] = np.where(
                        can_flow_down, VOXEL_AIR, voxels[:, :, :-1]
                    )

            # --- Lateral leveling ---
            # Transfer water from higher-level to lower-level neighbors
            water_mask = (voxels == VOXEL_WATER)
            # Neighbors that are water, air, or open floodgates can receive
            # Open floodgate: block_state == 0
            open_gate = (voxels == VOXEL_FLOODGATE) & (grid.block_state == 0)
            for axis, slc_hi, slc_lo in [
                (0, (slice(None, -1), slice(None), slice(None)),
                    (slice(1, None), slice(None), slice(None))),
                (0, (slice(1, None), slice(None), slice(None)),
                    (slice(None, -1), slice(None), slice(None))),
                (1, (slice(None), slice(None, -1), slice(None)),
                    (slice(None), slice(1, None), slice(None))),
                (1, (slice(None), slice(1, None), slice(None)),
                    (slice(None), slice(None, -1), slice(None))),
            ]:
                src_water = water_mask[slc_hi]
                dst_ok = (
                    (voxels[slc_lo] == VOXEL_AIR)
                    | (voxels[slc_lo] == VOXEL_WATER)
                    | open_gate[slc_lo]
                )
                can_level = src_water & dst_ok

                if not np.any(can_level):
                    continue

                src_level = water[slc_hi].astype(np.int16)
                dst_level = water[slc_lo].astype(np.int16)
                diff = src_level - dst_level

                # Only flow if source is higher
                flow_mask = can_level & (diff > 0)
                if not np.any(flow_mask):
                    continue

                # Transfer amount: fraction of the difference
                transfer = np.where(
                    flow_mask,
                    np.maximum((diff * int(WATER_FLOW_RATE * 100) // 100), 1),
                    0,
                ).astype(np.int16)
                # Ensure we don't transfer more than source has
                transfer = np.minimum(transfer, src_level)
                # Ensure destination doesn't overflow
                transfer = np.minimum(transfer, 255 - dst_level)

                if not np.any(transfer > 0):
                    continue

                moved = True

                # Apply transfer
                new_src = (src_level - transfer).astype(np.uint8)
                new_dst = (dst_level + transfer).astype(np.uint8)

                water[slc_hi] = np.where(flow_mask, new_src, water[slc_hi])
                water[slc_lo] = np.where(flow_mask, new_dst, water[slc_lo])

                # Update voxel types
                voxels[slc_hi] = np.where(
                    flow_mask & (new_src == 0), VOXEL_AIR, voxels[slc_hi]
                )
                voxels[slc_lo] = np.where(
                    flow_mask & (new_dst > 0), VOXEL_WATER, voxels[slc_lo]
                )

            if not moved:
                break
            flowed = True

        if flowed:
            self.voxel_grid.mark_all_dirty()

    # ------------------------------------------------------------------
    # Seepage
    # ------------------------------------------------------------------

    def _apply_seepage(self) -> None:
        """Water increases humidity of adjacent porous solid blocks."""
        grid = self.voxel_grid
        voxels = grid.grid
        hum = grid.humidity
        w, d, h = grid.width, grid.depth, grid.height

        water_mask = (voxels == VOXEL_WATER)
        if not np.any(water_mask):
            return

        poro = self._porosity_lut[voxels]

        # Find solid blocks adjacent to water
        adj_water = np.zeros((w, d, h), dtype=np.bool_)
        if w > 1:
            adj_water[1:, :, :] |= water_mask[:-1, :, :]
            adj_water[:-1, :, :] |= water_mask[1:, :, :]
        if d > 1:
            adj_water[:, 1:, :] |= water_mask[:, :-1, :]
            adj_water[:, :-1, :] |= water_mask[:, 1:, :]
        if h > 1:
            adj_water[:, :, 1:] |= water_mask[:, :, :-1]
            adj_water[:, :, :-1] |= water_mask[:, :, 1:]

        # Only solid non-water blocks with porosity > 0 get seepage
        receives = adj_water & (voxels != VOXEL_AIR) & (voxels != VOXEL_WATER) & (poro > 0)
        if np.any(receives):
            seepage = WATER_SEEP_RATE * poro[receives]
            hum[receives] = np.minimum(
                hum[receives] + seepage, 1.0
            )

    # ------------------------------------------------------------------
    # Pressure
    # ------------------------------------------------------------------

    def _apply_pressure(self) -> None:
        """Compute hydrostatic pressure from water column depth.

        Pressure increases with depth (column of water above).
        Applies shear load to adjacent solid walls.
        If pressure exceeds shear_strength * BURST_FACTOR, wall bursts.
        """
        grid = self.voxel_grid
        voxels = grid.grid
        w, d, h = grid.width, grid.depth, grid.height

        water_mask = (voxels == VOXEL_WATER)
        if not np.any(water_mask):
            return

        # Compute water depth at each position (count water blocks above)
        # depth[x,y,z] = number of contiguous water blocks from surface down to z
        depth = np.zeros((w, d, h), dtype=np.float32)
        for z in range(h):
            if z == 0:
                depth[:, :, z] = np.where(water_mask[:, :, z], 1.0, 0.0)
            else:
                depth[:, :, z] = np.where(
                    water_mask[:, :, z],
                    depth[:, :, z - 1] + 1.0,
                    0.0,
                )

        # Pressure = depth * water_weight * pressure_weight
        water_weight = self._weight_lut[VOXEL_WATER]
        pressure = depth * water_weight * WATER_PRESSURE_WEIGHT

        if not np.any(pressure > 0):
            return

        # Apply pressure as shear load to adjacent solid walls
        shear = self._shear_lut[voxels]
        solid = (voxels != VOXEL_AIR) & (voxels != VOXEL_WATER)

        # Find solid blocks adjacent to pressurized water
        # Pressure at the water block is applied to all adjacent solid neighbors
        adj_pressure = np.zeros((w, d, h), dtype=np.float32)
        if w > 1:
            adj_pressure[1:, :, :] = np.maximum(adj_pressure[1:, :, :], pressure[:-1, :, :])
            adj_pressure[:-1, :, :] = np.maximum(adj_pressure[:-1, :, :], pressure[1:, :, :])
        if d > 1:
            adj_pressure[:, 1:, :] = np.maximum(adj_pressure[:, 1:, :], pressure[:, :-1, :])
            adj_pressure[:, :-1, :] = np.maximum(adj_pressure[:, :-1, :], pressure[:, 1:, :])
        if h > 1:
            adj_pressure[:, :, 1:] = np.maximum(adj_pressure[:, :, 1:], pressure[:, :, :-1])
            adj_pressure[:, :, :-1] = np.maximum(adj_pressure[:, :, :-1], pressure[:, :, 1:])

        # Apply to solid blocks only
        wall_pressure = np.where(solid, adj_pressure, 0.0)
        grid.shear_load += wall_pressure

        # --- Gate/door pressure burst ---
        # Closed floodgates and doors under water pressure can be forced open.
        # Effective shear = base_shear * METAL_STRENGTH_MULT[base_metal].
        self._check_gate_burst(voxels, wall_pressure, shear, grid)

        # Burst check: if pressure > shear_strength * BURST_FACTOR → wall breaks
        safe_shear = np.where(shear > 0, shear, 1e9)
        should_burst = (
            solid
            & (wall_pressure > safe_shear * WATER_BURST_FACTOR)
            & (shear > 0)  # Don't burst anchors with inf shear
            & (~grid.loose)
        )

        if not np.any(should_burst):
            return

        xs, ys, zs = np.where(should_burst)
        count = min(len(xs), MAX_CASCADE_PER_TICK)
        if len(xs) > count:
            # Burst the most pressurized walls first
            worst = np.argpartition(wall_pressure[xs, ys, zs], -count)[-count:]
            xs, ys, zs = xs[worst], ys[worst], zs[worst]

        voxels[xs, ys, zs] = VOXEL_AIR
        grid.loose[xs, ys, zs] = False
        grid.mark_all_dirty()
        self.event_bus.publish("water_burst", count=int(count))

    def _check_gate_burst(
        self,
        voxels: np.ndarray,
        wall_pressure: np.ndarray,
        shear: np.ndarray,
        grid: VoxelGrid,
    ) -> None:
        """Force open closed doors/floodgates under water pressure.

        Closed doors/floodgates under pressure exceeding their effective shear
        (base_shear × METAL_STRENGTH_MULT[metal]) are forced open (block_state=0).
        """
        gate_types = frozenset({VOXEL_DOOR, VOXEL_FLOODGATE})
        metal_type_arr = grid.metal_type
        block_state_arr = grid.block_state

        for gate_type in gate_types:
            mask = (voxels == gate_type) & (block_state_arr != 0)  # Closed only
            if not np.any(mask):
                continue

            positions = np.argwhere(mask)
            for pos in positions:
                x, y, z = int(pos[0]), int(pos[1]), int(pos[2])
                pressure = wall_pressure[x, y, z]
                if pressure <= 0:
                    continue

                # Get effective shear strength
                base_shear = float(shear[x, y, z])
                mt = int(metal_type_arr[x, y, z])
                base_mt = mt & 0x7F
                strength_mult = METAL_STRENGTH_MULT.get(base_mt, 1.0)
                effective_shear = base_shear * strength_mult

                if pressure > effective_shear * WATER_BURST_FACTOR:
                    # Force open
                    grid.set_block_state(x, y, z, 0)
                    self.event_bus.publish(
                        "gate_pressure_burst",
                        x=x, y=y, z=z, vtype=gate_type,
                    )

    # ------------------------------------------------------------------
    # Lava interaction
    # ------------------------------------------------------------------

    def _lava_interaction(self) -> None:
        """Water adjacent to lava: both become obsidian + steam.

        Checked before flow so new water doesn't teleport through lava.
        """
        grid = self.voxel_grid
        voxels = grid.grid
        temp = grid.temperature
        hum = grid.humidity
        w, d, h = grid.width, grid.depth, grid.height

        water_mask = (voxels == VOXEL_WATER)
        lava_mask = (voxels == VOXEL_LAVA)

        if not (np.any(water_mask) and np.any(lava_mask)):
            return

        # Find water blocks adjacent to lava
        water_near_lava = np.zeros((w, d, h), dtype=np.bool_)
        if w > 1:
            water_near_lava[1:, :, :] |= lava_mask[:-1, :, :]
            water_near_lava[:-1, :, :] |= lava_mask[1:, :, :]
        if d > 1:
            water_near_lava[:, 1:, :] |= lava_mask[:, :-1, :]
            water_near_lava[:, :-1, :] |= lava_mask[:, 1:, :]
        if h > 1:
            water_near_lava[:, :, 1:] |= lava_mask[:, :, :-1]
            water_near_lava[:, :, :-1] |= lava_mask[:, :, 1:]
        water_near_lava &= water_mask

        # Find lava blocks adjacent to water
        lava_near_water = np.zeros((w, d, h), dtype=np.bool_)
        if w > 1:
            lava_near_water[1:, :, :] |= water_mask[:-1, :, :]
            lava_near_water[:-1, :, :] |= water_mask[1:, :, :]
        if d > 1:
            lava_near_water[:, 1:, :] |= water_mask[:, :-1, :]
            lava_near_water[:, :-1, :] |= water_mask[:, 1:, :]
        if h > 1:
            lava_near_water[:, :, 1:] |= water_mask[:, :, :-1]
            lava_near_water[:, :, :-1] |= water_mask[:, :, 1:]
        lava_near_water &= lava_mask

        reacted = False

        # Convert water near lava to obsidian
        if np.any(water_near_lava):
            reacted = True
            # Average temperature: midpoint between water (20) and lava (1000)
            avg_temp = (temp[water_near_lava] + WATER_TEMPERATURE) / 2.0
            voxels[water_near_lava] = WATER_LAVA_PRODUCT
            temp[water_near_lava] = avg_temp
            grid.water_level[water_near_lava] = 0
            grid.loose[water_near_lava] = False
            # Steam: set high humidity on adjacent blocks
            self._generate_steam(water_near_lava)

        # Convert lava near water to obsidian
        if np.any(lava_near_water):
            reacted = True
            avg_temp = (temp[lava_near_water] + WATER_TEMPERATURE) / 2.0
            voxels[lava_near_water] = WATER_LAVA_PRODUCT
            temp[lava_near_water] = avg_temp
            grid.loose[lava_near_water] = False

        if reacted:
            grid.mark_all_dirty()
            self.event_bus.publish("water_lava_reaction")

    def _generate_steam(self, source_mask: np.ndarray) -> None:
        """Set high humidity on blocks adjacent to steam source positions."""
        grid = self.voxel_grid
        voxels = grid.grid
        hum = grid.humidity
        w, d, h = grid.width, grid.depth, grid.height

        adj = np.zeros((w, d, h), dtype=np.bool_)
        if w > 1:
            adj[1:, :, :] |= source_mask[:-1, :, :]
            adj[:-1, :, :] |= source_mask[1:, :, :]
        if d > 1:
            adj[:, 1:, :] |= source_mask[:, :-1, :]
            adj[:, :-1, :] |= source_mask[:, 1:, :]
        if h > 1:
            adj[:, :, 1:] |= source_mask[:, :, :-1]
            adj[:, :, :-1] |= source_mask[:, :, 1:]

        receives_steam = adj & (voxels != VOXEL_LAVA) & (voxels != VOXEL_WATER)
        if np.any(receives_steam):
            hum[receives_steam] = np.maximum(
                hum[receives_steam], WATER_HUMIDITY_SOURCE
            )

    # ------------------------------------------------------------------
    # Surface evaporation
    # ------------------------------------------------------------------

    def _surface_evaporation(self) -> None:
        """Water at z=0 (surface) slowly evaporates."""
        grid = self.voxel_grid
        water = grid.water_level

        surface_water = (grid.grid[:, :, 0] == VOXEL_WATER)
        if not np.any(surface_water):
            return

        evap_amount = max(1, int(WATER_EVAPORATION_RATE * 255))
        current = water[:, :, 0].astype(np.int16)
        new_level = np.where(
            surface_water,
            np.maximum(current - evap_amount, 0),
            current,
        ).astype(np.uint8)
        water[:, :, 0] = new_level

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        """Remove VOXEL_WATER where water_level has dropped to 0."""
        grid = self.voxel_grid
        voxels = grid.grid
        water = grid.water_level

        empty_water = (voxels == VOXEL_WATER) & (water == 0)
        if np.any(empty_water):
            voxels[empty_water] = VOXEL_AIR
            grid.mark_all_dirty()
