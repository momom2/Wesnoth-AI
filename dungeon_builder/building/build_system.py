"""Dig queue and construction system."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dungeon_builder.config import (
    VOXEL_AIR,
    NON_DIGGABLE,
    DIG_DURATION,
    MAX_CONCURRENT_DIGS,
)

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.world.voxel_grid import VoxelGrid

logger = logging.getLogger("dungeon_builder.building")


class DigJob:
    """A queued dig operation on a single voxel."""

    __slots__ = ("x", "y", "z", "ticks_remaining", "total_ticks")

    def __init__(self, x: int, y: int, z: int, duration_ticks: int) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.ticks_remaining = duration_ticks
        self.total_ticks = duration_ticks


class BuildSystem:
    """Manages the dig queue: queues dig jobs, processes them over ticks.

    Dig completion makes material loose (instead of removing it).

    Three dig lists:
    - ``pending_digs``: blocks not yet visible (through fog); skip ALL
      validation except duplicate check.  When they become visible they are
      either promoted to ``dig_queue`` (valid) or cancelled (invalid).
    - ``dig_queue``: validated, waiting for a concurrent-dig slot.
    - ``active_digs``: currently progressing.
    """

    def __init__(self, event_bus: EventBus, voxel_grid: VoxelGrid) -> None:
        self.event_bus = event_bus
        self.voxel_grid = voxel_grid
        self.dig_queue: list[DigJob] = []
        self.active_digs: list[DigJob] = []
        self.pending_digs: list[DigJob] = []  # invisible blocks awaiting reveal

        event_bus.subscribe("tick", self._on_tick)
        event_bus.subscribe("voxel_left_clicked", self._on_voxel_left_clicked)
        event_bus.subscribe("claimed_territory_changed", self._check_pending_digs)
        event_bus.subscribe("dig_complete", self._on_dig_complete_check_pending)

    # ── Public API ────────────────────────────────────────────────────

    def queue_dig(self, x: int, y: int, z: int) -> bool:
        """Queue a dig at the given voxel position.  Returns True if queued.

        If the block is not yet visible (fog-of-war), it goes into
        ``pending_digs`` — no validation except duplicate check.
        """
        # Don't double-queue across any list
        if self._is_in_any_list(x, y, z):
            return False

        # If not visible, add as pending (skip all validation)
        if not self.voxel_grid.is_visible(x, y, z):
            # Use a placeholder duration; real duration set on promotion
            job = DigJob(x, y, z, 0)
            self.pending_digs.append(job)
            self.event_bus.publish("dig_pending", x=x, y=y, z=z)
            logger.debug("Dig pending (invisible) at (%d, %d, %d)", x, y, z)
            return True

        vtype = self.voxel_grid.get(x, y, z)

        # Can't dig non-diggable types
        if vtype in NON_DIGGABLE:
            return False

        # Can't dig already-loose material
        if self.voxel_grid.is_loose(x, y, z):
            self.event_bus.publish(
                "error_message", text="Material is already loose"
            )
            return False

        duration = DIG_DURATION.get(vtype, 40)
        job = DigJob(x, y, z, duration)
        self.dig_queue.append(job)
        self.event_bus.publish("dig_queued", x=x, y=y, z=z, duration=duration)
        logger.debug("Dig queued at (%d, %d, %d), duration=%d ticks", x, y, z, duration)
        return True

    def cancel_dig(self, x: int, y: int, z: int) -> bool:
        """Cancel a dig at (x, y, z).  Returns True if cancelled.

        Searches ``pending_digs``, ``dig_queue``, and ``active_digs``.
        Block is fully restored (no partial damage — dig only makes loose
        on completion).
        """
        for lst in (self.pending_digs, self.dig_queue, self.active_digs):
            for job in lst:
                if job.x == x and job.y == y and job.z == z:
                    lst.remove(job)
                    self.event_bus.publish("dig_cancelled", x=x, y=y, z=z)
                    logger.debug("Dig cancelled at (%d, %d, %d)", x, y, z)
                    return True
        return False

    def is_being_dug(self, x: int, y: int, z: int) -> bool:
        """Return True if the voxel is queued, active, *or* pending."""
        return self._is_in_any_list(x, y, z)

    def is_pending_dig(self, x: int, y: int, z: int) -> bool:
        """Return True if the voxel is in the pending (invisible) list."""
        for job in self.pending_digs:
            if job.x == x and job.y == y and job.z == z:
                return True
        return False

    def get_dig_progress(self, x: int, y: int, z: int) -> float:
        """Return dig progress as 0.0 (not started) to 1.0 (complete).

        Queued and pending jobs return 0.0.  Active jobs return fraction
        completed.  Returns -1.0 if the voxel is not being dug.
        """
        for job in self.pending_digs:
            if job.x == x and job.y == y and job.z == z:
                return 0.0
        for job in self.dig_queue:
            if job.x == x and job.y == y and job.z == z:
                return 0.0
        for job in self.active_digs:
            if job.x == x and job.y == y and job.z == z:
                return 1.0 - (job.ticks_remaining / job.total_ticks)
        return -1.0

    # ── Tick processing ───────────────────────────────────────────────

    def _on_tick(self, tick: int, **kw) -> None:
        # Promote queued jobs to active
        while len(self.active_digs) < MAX_CONCURRENT_DIGS and self.dig_queue:
            job = self.dig_queue.pop(0)
            self.active_digs.append(job)

        # Process active digs
        completed = []
        for job in self.active_digs:
            job.ticks_remaining -= 1
            if job.ticks_remaining <= 0:
                # Mark material as loose instead of removing it
                self.voxel_grid.set_loose(job.x, job.y, job.z, True)
                completed.append(job)
                self.event_bus.publish("dig_complete", x=job.x, y=job.y, z=job.z)
                logger.debug("Dig complete at (%d, %d, %d) — now loose", job.x, job.y, job.z)

        for job in completed:
            self.active_digs.remove(job)

    # ── Pending dig promotion ─────────────────────────────────────────

    def _check_pending_digs(self, **kw) -> None:
        """Promote or cancel pending digs whose blocks have become visible.

        Called on ``claimed_territory_changed`` and ``dig_complete``.
        Per design: pending digs skip ALL validation until visible.  Once
        visible, re-validate: if invalid (non-diggable, already loose, air)
        → cancel with ``dig_cancelled``.  If valid → promote to ``dig_queue``.
        """
        still_pending: list[DigJob] = []
        for job in self.pending_digs:
            x, y, z = job.x, job.y, job.z
            if not self.voxel_grid.is_visible(x, y, z):
                still_pending.append(job)
                continue

            # Now visible — validate
            vtype = self.voxel_grid.get(x, y, z)
            if (
                vtype == VOXEL_AIR
                or vtype in NON_DIGGABLE
                or self.voxel_grid.is_loose(x, y, z)
            ):
                # Invalid — cancel (player can now see why)
                self.event_bus.publish("dig_cancelled", x=x, y=y, z=z)
                logger.debug(
                    "Pending dig cancelled at (%d, %d, %d): invalid (vtype=%d)",
                    x, y, z, vtype,
                )
                continue

            # Valid — set real duration and promote
            duration = DIG_DURATION.get(vtype, 40)
            job.ticks_remaining = duration
            job.total_ticks = duration
            self.dig_queue.append(job)
            self.event_bus.publish("dig_queued", x=x, y=y, z=z, duration=duration)
            logger.debug(
                "Pending dig promoted at (%d, %d, %d), duration=%d", x, y, z, duration
            )

        self.pending_digs = still_pending

    def _on_dig_complete_check_pending(self, **kw) -> None:
        """After a dig completes, check if any pending digs can be promoted."""
        self._check_pending_digs()

    # ── Event handlers ────────────────────────────────────────────────

    def _on_voxel_left_clicked(self, x: int, y: int, z: int, mode: str, **kw) -> None:
        if mode == "dig":
            # If already being dug → cancel
            if self.is_being_dug(x, y, z):
                self.cancel_dig(x, y, z)
            else:
                self.queue_dig(x, y, z)

    # ── Internal helpers ──────────────────────────────────────────────

    def _is_in_any_list(self, x: int, y: int, z: int) -> bool:
        """Check if (x,y,z) appears in any of the three dig lists."""
        for job in self.pending_digs:
            if job.x == x and job.y == y and job.z == z:
                return True
        for job in self.dig_queue:
            if job.x == x and job.y == y and job.z == z:
                return True
        for job in self.active_digs:
            if job.x == x and job.y == y and job.z == z:
                return True
        return False
