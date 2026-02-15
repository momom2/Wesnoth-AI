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
    """

    def __init__(self, event_bus: EventBus, voxel_grid: VoxelGrid) -> None:
        self.event_bus = event_bus
        self.voxel_grid = voxel_grid
        self.dig_queue: list[DigJob] = []
        self.active_digs: list[DigJob] = []

        event_bus.subscribe("tick", self._on_tick)
        event_bus.subscribe("voxel_left_clicked", self._on_voxel_left_clicked)

    def queue_dig(self, x: int, y: int, z: int) -> bool:
        """Queue a dig at the given voxel position. Returns True if queued."""
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

        # Don't double-queue
        for job in self.dig_queue + self.active_digs:
            if job.x == x and job.y == y and job.z == z:
                return False

        duration = DIG_DURATION.get(vtype, 40)
        job = DigJob(x, y, z, duration)
        self.dig_queue.append(job)
        self.event_bus.publish("dig_queued", x=x, y=y, z=z, duration=duration)
        logger.debug("Dig queued at (%d, %d, %d), duration=%d ticks", x, y, z, duration)
        return True

    def _on_tick(self, tick: int) -> None:
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

    def _on_voxel_left_clicked(self, x: int, y: int, z: int, mode: str) -> None:
        if mode == "dig":
            self.queue_dig(x, y, z)
