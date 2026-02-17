"""Gravity physics: loose blocks fall, disconnected blocks become loose."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from dungeon_builder.config import (
    CHUNK_SIZE,
    VOXEL_AIR,
    VOXEL_WATER,
    VOXEL_WEIGHT,
    VOXEL_MAX_LOAD,
    VOXEL_POROSITY,
    VOXEL_SHOCK_TRANSMIT,
    VOXEL_BRITTLENESS,
    STRUCTURAL_ANCHORS,
    GRAVITY_TICK_INTERVAL,
    CONNECTIVITY_TICK_INTERVAL,
    MAX_FALL_PER_TICK,
    MAX_CASCADE_PER_TICK,
    IMPACT_DAMAGE_THRESHOLD,
    IMPACT_DAMAGE_FACTOR,
    SHOCK_ATTENUATION,
    SHOCK_STRUCTURAL_FACTOR,
    MAX_SHOCK_PROPAGATION_STEPS,
    SHATTER_THRESHOLD,
    MAX_CASCADE_DEPTH,
    GRANULAR_POROSITY_THRESHOLD,
    REPOSE_TICK_INTERVAL,
    MAX_SPREAD_PER_TICK,
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

        # Weight LUT for impact damage calculation
        self._weight_lut = np.zeros(256, dtype=np.float32)
        for vtype, w in VOXEL_WEIGHT.items():
            if 0 <= vtype < 256:
                self._weight_lut[vtype] = w

        # Shock propagation LUTs
        self._shock_transmit_lut = np.zeros(256, dtype=np.float32)
        for vtype, st in VOXEL_SHOCK_TRANSMIT.items():
            if 0 <= vtype < 256:
                self._shock_transmit_lut[vtype] = st

        self._brittleness_lut = np.zeros(256, dtype=np.float32)
        for vtype, brit in VOXEL_BRITTLENESS.items():
            if 0 <= vtype < 256:
                self._brittleness_lut[vtype] = brit

        self._capacity_lut = np.zeros(256, dtype=np.float32)
        for vtype, cap in VOXEL_MAX_LOAD.items():
            if 0 <= vtype < 256:
                self._capacity_lut[vtype] = min(cap, 1e9)

        # Granular LUT: materials with porosity >= threshold spread laterally
        self._granular_lut = np.zeros(256, dtype=np.bool_)
        for vtype, poro in VOXEL_POROSITY.items():
            if 0 <= vtype < 256 and poro >= GRANULAR_POROSITY_THRESHOLD:
                # Air is porous but not granular
                if vtype != VOXEL_AIR:
                    self._granular_lut[vtype] = True

        # Snapshot of grid+loose at last connectivity check for skip-if-unchanged
        self._last_connectivity_grid: np.ndarray | None = None
        self._last_connectivity_loose: np.ndarray | None = None

        event_bus.subscribe("tick", self._on_tick)

    def _on_tick(self, tick: int, **kw) -> None:
        if tick % GRAVITY_TICK_INTERVAL == 0:
            self._process_falling()
        if tick % REPOSE_TICK_INTERVAL == 0:
            self._spread_granular()
        if tick % CONNECTIVITY_TICK_INTERVAL == 0:
            self._check_connectivity()

    def _process_falling(self) -> None:
        """Move loose blocks downward through air, tracking fall distance."""
        grid = self.voxel_grid
        voxels = grid.grid
        loose = grid.loose
        temp = grid.temperature
        humidity = grid.humidity
        fall_dist = grid.fall_distance

        fell_any = False

        for _ in range(MAX_FALL_PER_TICK):
            # Boolean mask: non-air, loose blocks with air directly below
            # z+1 is deeper (downward in our coordinate system)
            # Water blocks are treated as solid (blocks can't fall through water)
            can_fall = (
                (voxels[:, :, :-1] != VOXEL_AIR)
                & loose[:, :, :-1]
                & (voxels[:, :, 1:] == VOXEL_AIR)
            )

            if not np.any(can_fall):
                break

            fell_any = True

            # Copy falling blocks downward (including accumulated fall distance)
            voxels[:, :, 1:][can_fall] = voxels[:, :, :-1][can_fall]
            loose[:, :, 1:][can_fall] = True
            temp[:, :, 1:][can_fall] = temp[:, :, :-1][can_fall]
            humidity[:, :, 1:][can_fall] = humidity[:, :, :-1][can_fall]
            fall_dist[:, :, 1:][can_fall] = fall_dist[:, :, :-1][can_fall] + 1

            # Clear source positions
            voxels[:, :, :-1][can_fall] = VOXEL_AIR
            loose[:, :, :-1][can_fall] = False
            temp[:, :, :-1][can_fall] = 0.0
            humidity[:, :, :-1][can_fall] = 0.0
            fall_dist[:, :, :-1][can_fall] = 0

        if fell_any:
            self._apply_impacts()
            grid.mark_all_dirty()
            self.event_bus.publish("blocks_fell")

    def _apply_impacts(self) -> None:
        """Apply impact damage when falling blocks land on solid ground.

        A block that has fallen IMPACT_DAMAGE_THRESHOLD+ cells applies
        impact load = fall_distance * weight * IMPACT_DAMAGE_FACTOR
        to the block directly below it.
        """
        grid = self.voxel_grid
        voxels = grid.grid
        loose = grid.loose
        fall_dist = grid.fall_distance

        # Find blocks that have stopped: loose, non-air, with solid below
        # (they couldn't fall further this tick)
        h = grid.height
        if h < 2:
            return

        stopped = (
            (voxels[:, :, :-1] != VOXEL_AIR)
            & loose[:, :, :-1]
            & (voxels[:, :, 1:] != VOXEL_AIR)
            & (fall_dist[:, :, :-1] >= IMPACT_DAMAGE_THRESHOLD)
        )

        if not np.any(stopped):
            return

        # Compute impact load
        weight = self._weight_lut[voxels[:, :, :-1]]
        impact = fall_dist[:, :, :-1].astype(np.float32) * weight * IMPACT_DAMAGE_FACTOR

        # Add impact load to the block below (at z+1)
        impact_on_below = np.where(stopped, impact, 0.0)
        grid.load[:, :, 1:] += impact_on_below

        # Propagate shock waves from impact points
        self._propagate_shock(impact_on_below)

        # Clear fall distance for stopped blocks
        fall_dist[:, :, :-1][stopped] = 0

        # Publish impact event with count of impacted blocks
        count = int(np.sum(stopped))
        self.event_bus.publish("impact", count=count)

    def _spread_granular(self) -> None:
        """Spread loose granular materials laterally (angle of repose).

        A loose granular block (dirt, sand, chalk, sandstone) sitting on a
        solid surface will slide to an adjacent cardinal position if:
        1. The adjacent position at the same z-level is air
        2. The position below the adjacent position (z+1) is also air
           (so the block will fall after sliding)

        This creates natural-looking piles instead of vertical columns.
        """
        grid = self.voxel_grid
        voxels = grid.grid
        loose = grid.loose
        temp = grid.temperature
        humidity = grid.humidity
        w, d, h = grid.width, grid.depth, grid.height

        spread_any = False

        for _ in range(MAX_SPREAD_PER_TICK):
            # Find loose granular blocks resting on solid (can't fall down)
            granular = self._granular_lut[voxels]
            resting = np.zeros((w, d, h), dtype=np.bool_)
            # Resting = loose, granular, non-air, with solid or boundary below
            resting[:, :, :-1] = (
                (voxels[:, :, :-1] != VOXEL_AIR)
                & loose[:, :, :-1]
                & granular[:, :, :-1]
                & (voxels[:, :, 1:] != VOXEL_AIR)
            )
            # Bottom row: resting on grid boundary
            resting[:, :, -1] = (
                (voxels[:, :, -1] != VOXEL_AIR)
                & loose[:, :, -1]
                & granular[:, :, -1]
            )

            if not np.any(resting):
                break

            # For each resting block, check if any cardinal neighbor at same z
            # is air AND has air below it (so the block will fall into the gap)
            # We process one direction per pass to avoid conflicts
            moved = np.zeros((w, d, h), dtype=np.bool_)

            # Try +X direction
            if w > 1:
                can_slide = np.zeros((w, d, h), dtype=np.bool_)
                # Block at [:-1] can slide to [1:]
                can_slide[:-1, :, :] = (
                    resting[:-1, :, :]
                    & (voxels[1:, :, :] == VOXEL_AIR)
                )
                # Also need air below target (z+1 of target) for it to make sense
                # (block slides to the edge and falls)
                can_slide[:-1, :, :-1] &= (voxels[1:, :, 1:] == VOXEL_AIR)
                # Don't slide from bottom row unless there's a reason
                can_slide[:, :, -1] = False

                if np.any(can_slide):
                    # Move blocks: copy to target, clear source
                    xs, ys, zs = np.where(can_slide)
                    # Only process blocks that haven't been touched yet
                    for i in range(len(xs)):
                        sx, sy, sz = xs[i], ys[i], zs[i]
                        tx = sx + 1
                        if moved[sx, sy, sz] or voxels[tx, sy, sz] != VOXEL_AIR:
                            continue
                        voxels[tx, sy, sz] = voxels[sx, sy, sz]
                        loose[tx, sy, sz] = True
                        temp[tx, sy, sz] = temp[sx, sy, sz]
                        humidity[tx, sy, sz] = humidity[sx, sy, sz]
                        voxels[sx, sy, sz] = VOXEL_AIR
                        loose[sx, sy, sz] = False
                        temp[sx, sy, sz] = 0.0
                        humidity[sx, sy, sz] = 0.0
                        moved[sx, sy, sz] = True
                        moved[tx, sy, sz] = True
                        spread_any = True

            # Try -X direction
            if w > 1:
                can_slide = np.zeros((w, d, h), dtype=np.bool_)
                can_slide[1:, :, :] = (
                    resting[1:, :, :]
                    & (~moved[1:, :, :])
                    & (voxels[:-1, :, :] == VOXEL_AIR)
                )
                can_slide[1:, :, :-1] &= (voxels[:-1, :, 1:] == VOXEL_AIR)
                can_slide[:, :, -1] = False

                if np.any(can_slide):
                    xs, ys, zs = np.where(can_slide)
                    for i in range(len(xs)):
                        sx, sy, sz = xs[i], ys[i], zs[i]
                        tx = sx - 1
                        if moved[sx, sy, sz] or voxels[tx, sy, sz] != VOXEL_AIR:
                            continue
                        voxels[tx, sy, sz] = voxels[sx, sy, sz]
                        loose[tx, sy, sz] = True
                        temp[tx, sy, sz] = temp[sx, sy, sz]
                        humidity[tx, sy, sz] = humidity[sx, sy, sz]
                        voxels[sx, sy, sz] = VOXEL_AIR
                        loose[sx, sy, sz] = False
                        temp[sx, sy, sz] = 0.0
                        humidity[sx, sy, sz] = 0.0
                        moved[sx, sy, sz] = True
                        moved[tx, sy, sz] = True
                        spread_any = True

            # Try +Y direction
            if d > 1:
                can_slide = np.zeros((w, d, h), dtype=np.bool_)
                can_slide[:, :-1, :] = (
                    resting[:, :-1, :]
                    & (~moved[:, :-1, :])
                    & (voxels[:, 1:, :] == VOXEL_AIR)
                )
                can_slide[:, :-1, :-1] &= (voxels[:, 1:, 1:] == VOXEL_AIR)
                can_slide[:, :, -1] = False

                if np.any(can_slide):
                    xs, ys, zs = np.where(can_slide)
                    for i in range(len(xs)):
                        sx, sy, sz = xs[i], ys[i], zs[i]
                        ty = sy + 1
                        if moved[sx, sy, sz] or voxels[sx, ty, sz] != VOXEL_AIR:
                            continue
                        voxels[sx, ty, sz] = voxels[sx, sy, sz]
                        loose[sx, ty, sz] = True
                        temp[sx, ty, sz] = temp[sx, sy, sz]
                        humidity[sx, ty, sz] = humidity[sx, sy, sz]
                        voxels[sx, sy, sz] = VOXEL_AIR
                        loose[sx, sy, sz] = False
                        temp[sx, sy, sz] = 0.0
                        humidity[sx, sy, sz] = 0.0
                        moved[sx, sy, sz] = True
                        moved[sx, ty, sz] = True
                        spread_any = True

            # Try -Y direction
            if d > 1:
                can_slide = np.zeros((w, d, h), dtype=np.bool_)
                can_slide[:, 1:, :] = (
                    resting[:, 1:, :]
                    & (~moved[:, 1:, :])
                    & (voxels[:, :-1, :] == VOXEL_AIR)
                )
                can_slide[:, 1:, :-1] &= (voxels[:, :-1, 1:] == VOXEL_AIR)
                can_slide[:, :, -1] = False

                if np.any(can_slide):
                    xs, ys, zs = np.where(can_slide)
                    for i in range(len(xs)):
                        sx, sy, sz = xs[i], ys[i], zs[i]
                        ty = sy - 1
                        if moved[sx, sy, sz] or voxels[sx, ty, sz] != VOXEL_AIR:
                            continue
                        voxels[sx, ty, sz] = voxels[sx, sy, sz]
                        loose[sx, ty, sz] = True
                        temp[sx, ty, sz] = temp[sx, sy, sz]
                        humidity[sx, ty, sz] = humidity[sx, sy, sz]
                        voxels[sx, sy, sz] = VOXEL_AIR
                        loose[sx, sy, sz] = False
                        temp[sx, sy, sz] = 0.0
                        humidity[sx, sy, sz] = 0.0
                        moved[sx, sy, sz] = True
                        moved[sx, ty, sz] = True
                        spread_any = True

            if not spread_any:
                break

        if spread_any:
            grid.mark_all_dirty()
            self.event_bus.publish("blocks_spread")

    def _propagate_shock(self, impact_energy: np.ndarray) -> None:
        """Propagate shock waves from impact points through solid blocks.

        Impact energy arrives as a (w, d, h-1) array representing load
        added at z+1 positions. Shock accumulates as it propagates
        outward from impact points, attenuated by SHOCK_ATTENUATION per
        step. When accumulated shock + existing load exceeds capacity,
        blocks crack. Extreme shock shatters brittle materials to air.
        """
        grid = self.voxel_grid
        voxels = grid.grid
        loose = grid.loose
        w, d, h = grid.width, grid.depth, grid.height

        # Initialize shock energy at impact points
        initial_shock = np.zeros((w, d, h), dtype=np.float32)
        initial_shock[:, :, 1:] = impact_energy

        if not np.any(initial_shock > 0):
            return

        transmit = self._shock_transmit_lut[voxels]
        capacity = self._capacity_lut[voxels]
        anchors = self._anchor_lut[voxels]
        brittleness = self._brittleness_lut[voxels]
        solid = (voxels != VOXEL_AIR) & (~loose)

        attenuation = 1.0 - SHOCK_ATTENUATION

        # Accumulate total shock received by each block across all steps
        accumulated_shock = initial_shock.copy()
        current_wave = initial_shock.copy()

        for _step in range(MAX_SHOCK_PROPAGATION_STEPS):
            outgoing = current_wave * transmit
            next_wave = np.zeros_like(current_wave)

            # 6-directional propagation
            if w > 1:
                next_wave[1:, :, :] += outgoing[:-1, :, :] * attenuation
                next_wave[:-1, :, :] += outgoing[1:, :, :] * attenuation
            if d > 1:
                next_wave[:, 1:, :] += outgoing[:, :-1, :] * attenuation
                next_wave[:, :-1, :] += outgoing[:, 1:, :] * attenuation
            if h > 1:
                next_wave[:, :, 1:] += outgoing[:, :, :-1] * attenuation
                next_wave[:, :, :-1] += outgoing[:, :, 1:] * attenuation

            # Only propagate through solid non-anchor blocks
            next_wave *= solid.astype(np.float32)
            next_wave[anchors] = 0.0

            accumulated_shock += next_wave
            current_wave = next_wave

            if not np.any(current_wave > 0.01):
                break

        # Check for failures from accumulated shock + existing load
        total_force = grid.load + accumulated_shock * SHOCK_STRUCTURAL_FACTOR
        safe_cap = np.where(capacity > 0, capacity, 1.0)

        # Shatter: extreme shock on brittle materials → air
        shatter_ratio = np.where(
            capacity > 0, accumulated_shock / safe_cap, 0.0
        )
        should_shatter = (
            (shatter_ratio > SHATTER_THRESHOLD)
            & solid
            & (~anchors)
            & (capacity > 0)
            & (brittleness >= 0.5)
        )

        # Crack: shock + load exceeds capacity → loose
        should_crack = (
            (total_force > capacity)
            & solid
            & (~anchors)
            & (capacity > 0)
            & (~should_shatter)
        )

        total_new_loose = 0
        total_shattered = 0
        remaining = MAX_CASCADE_PER_TICK
        dirty_pos = []  # Collect (xs, ys, zs) arrays for dirty marking

        # Apply shattering
        if np.any(should_shatter) and remaining > 0:
            sxs, sys_, szs = np.where(should_shatter)
            count = min(len(sxs), remaining)
            if len(sxs) > count:
                worst = np.argpartition(
                    shatter_ratio[sxs, sys_, szs], -count
                )[-count:]
                sxs, sys_, szs = sxs[worst], sys_[worst], szs[worst]
            else:
                sxs, sys_, szs = sxs[:count], sys_[:count], szs[:count]

            voxels[sxs, sys_, szs] = VOXEL_AIR
            loose[sxs, sys_, szs] = False
            total_shattered += count
            remaining -= count
            dirty_pos.append((sxs, sys_, szs))

        # Apply cracking
        if np.any(should_crack) and remaining > 0:
            cxs, cys, czs = np.where(should_crack)
            count = min(len(cxs), remaining)
            if len(cxs) > count:
                worst = np.argpartition(
                    total_force[cxs, cys, czs], -count
                )[-count:]
                cxs, cys, czs = cxs[worst], cys[worst], czs[worst]
            else:
                cxs, cys, czs = cxs[:count], cys[:count], czs[:count]

            loose[cxs, cys, czs] = True
            total_new_loose += count
            dirty_pos.append((cxs, cys, czs))

        if total_new_loose > 0 or total_shattered > 0:
            all_xs = np.concatenate([p[0] for p in dirty_pos])
            all_ys = np.concatenate([p[1] for p in dirty_pos])
            all_zs = np.concatenate([p[2] for p in dirty_pos])
            grid.mark_blocks_dirty(all_xs, all_ys, all_zs)
            self.event_bus.publish(
                "shock_cascade",
                cracked=total_new_loose,
                shattered=total_shattered,
            )

    def _check_connectivity(self) -> None:
        """Mark blocks as loose if not connected to any anchor.

        Skips the expensive flood-fill if the grid and loose arrays haven't
        changed since the last check.
        """
        grid = self.voxel_grid
        voxels = grid.grid
        loose = grid.loose

        # Skip if grid+loose unchanged since last check
        if (
            self._last_connectivity_grid is not None
            and np.array_equal(voxels, self._last_connectivity_grid)
            and np.array_equal(loose, self._last_connectivity_loose)
        ):
            return

        # Solid, non-loose blocks need connectivity checking
        # Water blocks are excluded (they flow, not structural)
        solid_nonloose = (voxels != VOXEL_AIR) & (voxels != VOXEL_WATER) & (~loose)

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
            xs, ys, zs = np.where(newly_loose)
            grid.mark_blocks_dirty(xs, ys, zs)
            count = int(len(xs))
            self.event_bus.publish("structural_disconnect", count=count)

        # Save snapshot for skip-if-unchanged on next call
        self._last_connectivity_grid = voxels.copy()
        self._last_connectivity_loose = loose.copy()
