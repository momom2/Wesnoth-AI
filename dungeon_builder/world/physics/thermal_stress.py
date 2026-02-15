"""Thermal stress physics: temperature gradients cause cracking.

Large temperature differences between adjacent blocks create thermal
stress proportional to the material's coefficient of thermal expansion
(CTE). Cumulative thermal fatigue accumulates over time; when it
exceeds THERMAL_CRACK_THRESHOLD the block cracks (becomes loose).

Quenching (water adjacent to hot blocks) dramatically amplifies
thermal stress, enabling explosive cracking gameplay.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_WATER,
    VOXEL_CTE,
    VOXEL_TENSILE_STRENGTH,
    STRUCTURAL_ANCHORS,
    THERMAL_STRESS_TICK_INTERVAL,
    THERMAL_FATIGUE_ACCUMULATION,
    THERMAL_FATIGUE_DECAY,
    QUENCH_MULTIPLIER,
    THERMAL_CRACK_THRESHOLD,
    MAX_CASCADE_PER_TICK,
)

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.world.voxel_grid import VoxelGrid


class ThermalStressPhysics:
    """Computes thermal stress from temperature gradients and tracks fatigue.

    Runs every THERMAL_STRESS_TICK_INTERVAL ticks.
    - Computes max |T_self - T_neighbor| across 6 neighbors
    - thermal_stress = max_gradient × CTE
    - Quenching: water-adjacent blocks get stress × QUENCH_MULTIPLIER
    - thermal_ratio = thermal_stress / tensile_strength
    - Fatigue accumulates: fatigue += ratio × ACCUMULATION_RATE
    - Fatigue decays when stress is low: fatigue -= DECAY_RATE
    - When fatigue >= THRESHOLD: block becomes loose (cracks)
    """

    def __init__(self, event_bus: EventBus, voxel_grid: VoxelGrid) -> None:
        self.event_bus = event_bus
        self.voxel_grid = voxel_grid

        # Pre-build LUTs (256 entries each)
        self._cte_lut = np.zeros(256, dtype=np.float32)
        self._tensile_lut = np.zeros(256, dtype=np.float32)
        self._anchor_lut = np.zeros(256, dtype=np.bool_)

        for vtype, cte in VOXEL_CTE.items():
            if 0 <= vtype < 256:
                self._cte_lut[vtype] = cte

        for vtype, tens in VOXEL_TENSILE_STRENGTH.items():
            if 0 <= vtype < 256:
                self._tensile_lut[vtype] = min(tens, 1e9)

        for vtype in STRUCTURAL_ANCHORS:
            if 0 <= vtype < 256:
                self._anchor_lut[vtype] = True

        event_bus.subscribe("tick", self._on_tick)

    def _on_tick(self, tick: int, **kw) -> None:
        if tick % THERMAL_STRESS_TICK_INTERVAL != 0:
            return
        self._compute_thermal_stress()

    def _compute_thermal_stress(self) -> None:
        """Compute thermal gradient stress and update fatigue."""
        grid = self.voxel_grid
        temp = grid.temperature
        voxels = grid.grid
        fatigue = grid.thermal_fatigue
        w, d, h = grid.width, grid.depth, grid.height

        # Step 1: Compute max absolute temperature gradient per block
        max_gradient = np.zeros((w, d, h), dtype=np.float32)

        # +X/-X
        if w > 1:
            diff_x = np.abs(temp[1:, :, :] - temp[:-1, :, :])
            max_gradient[:-1, :, :] = np.maximum(max_gradient[:-1, :, :], diff_x)
            max_gradient[1:, :, :] = np.maximum(max_gradient[1:, :, :], diff_x)
        # +Y/-Y
        if d > 1:
            diff_y = np.abs(temp[:, 1:, :] - temp[:, :-1, :])
            max_gradient[:, :-1, :] = np.maximum(max_gradient[:, :-1, :], diff_y)
            max_gradient[:, 1:, :] = np.maximum(max_gradient[:, 1:, :], diff_y)
        # +Z/-Z
        if h > 1:
            diff_z = np.abs(temp[:, :, 1:] - temp[:, :, :-1])
            max_gradient[:, :, :-1] = np.maximum(max_gradient[:, :, :-1], diff_z)
            max_gradient[:, :, 1:] = np.maximum(max_gradient[:, :, 1:], diff_z)

        # Step 2: Compute instantaneous thermal stress
        cte = self._cte_lut[voxels]
        thermal_stress = max_gradient * cte

        # Step 3: Quenching bonus - water adjacent to hot blocks amplifies stress
        water_mask = (voxels == VOXEL_WATER)
        if np.any(water_mask):
            has_adjacent_water = np.zeros((w, d, h), dtype=np.bool_)
            if w > 1:
                has_adjacent_water[1:, :, :] |= water_mask[:-1, :, :]
                has_adjacent_water[:-1, :, :] |= water_mask[1:, :, :]
            if d > 1:
                has_adjacent_water[:, 1:, :] |= water_mask[:, :-1, :]
                has_adjacent_water[:, :-1, :] |= water_mask[:, 1:, :]
            if h > 1:
                has_adjacent_water[:, :, 1:] |= water_mask[:, :, :-1]
                has_adjacent_water[:, :, :-1] |= water_mask[:, :, 1:]
            thermal_stress = np.where(
                has_adjacent_water,
                thermal_stress * QUENCH_MULTIPLIER,
                thermal_stress,
            )

        # Step 4: Compute stress ratio (thermal_stress / tensile_strength)
        tensile = self._tensile_lut[voxels]
        safe_tensile = np.where(tensile > 0, tensile, 1.0)
        thermal_ratio = np.where(tensile > 0, thermal_stress / safe_tensile, 0.0)

        # Step 5: Update cumulative fatigue
        fatigue_delta = thermal_ratio * THERMAL_FATIGUE_ACCUMULATION
        fatigue += fatigue_delta
        # Decay: fatigue decreases when ratio is below 0.1
        low_stress = thermal_ratio < 0.1
        fatigue[low_stress] -= THERMAL_FATIGUE_DECAY
        np.maximum(fatigue, 0.0, out=fatigue)

        # Step 6: Update stress_ratio (max with existing structural stress)
        grid.stress_ratio[:] = np.maximum(grid.stress_ratio, thermal_ratio)

        # Step 7: Crack blocks where fatigue >= threshold
        solid = (voxels != VOXEL_AIR) & (~grid.loose)
        anchors = self._anchor_lut[voxels]
        cracked = (
            (fatigue >= THERMAL_CRACK_THRESHOLD)
            & solid
            & (~anchors)
            & (cte > 0)  # Only materials with CTE > 0 can crack
        )

        if not np.any(cracked):
            return

        xs, ys, zs = np.where(cracked)
        count = min(len(xs), MAX_CASCADE_PER_TICK)
        if len(xs) > count:
            worst = np.argpartition(fatigue[xs, ys, zs], -count)[-count:]
            xs, ys, zs = xs[worst], ys[worst], zs[worst]

        grid.loose[xs, ys, zs] = True
        fatigue[xs, ys, zs] = 0.0  # Reset fatigue on cracked blocks
        grid.mark_all_dirty()
        self.event_bus.publish("thermal_crack", count=int(count))
