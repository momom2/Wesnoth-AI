"""Pick up and drop loose materials."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dungeon_builder.config import VOXEL_AIR

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.world.voxel_grid import VoxelGrid

logger = logging.getLogger("dungeon_builder.building")


class MoveSystem:
    """Lets the player pick up loose material and drop it elsewhere.

    Held material is a (voxel_type, count) tuple or None.
    Dropping onto a non-air voxel triggers an "attempt_craft" event
    so the crafting system can check for recipes.
    """

    def __init__(self, event_bus: EventBus, voxel_grid: VoxelGrid) -> None:
        self.event_bus = event_bus
        self.voxel_grid = voxel_grid
        self.held_material: tuple[int, int] | None = None  # (vtype, count)

        event_bus.subscribe("voxel_left_clicked", self._on_left_click)
        event_bus.subscribe("voxel_right_clicked", self._on_right_click)

    def pick_up(self, x: int, y: int, z: int) -> bool:
        """Pick up loose material at (x, y, z). Returns True on success."""
        if not self.voxel_grid.in_bounds(x, y, z):
            return False

        vtype = self.voxel_grid.get(x, y, z)
        if vtype == VOXEL_AIR:
            return False

        if not self.voxel_grid.is_loose(x, y, z):
            self.event_bus.publish(
                "error_message", text="Not loose, dig first"
            )
            return False

        if self.held_material is not None:
            held_type, held_count = self.held_material
            if held_type == vtype:
                self.held_material = (held_type, held_count + 1)
            else:
                self.event_bus.publish(
                    "error_message", text="Already holding a different material"
                )
                return False
        else:
            self.held_material = (vtype, 1)

        # Remove the voxel (set to AIR clears loose flag automatically)
        self.voxel_grid.set(x, y, z, VOXEL_AIR, event_bus=self.event_bus)
        self.event_bus.publish(
            "material_picked_up",
            vtype=self.held_material[0],
            count=self.held_material[1],
        )
        logger.debug("Picked up type=%d at (%d, %d, %d)", vtype, x, y, z)
        return True

    def drop(self, x: int, y: int, z: int) -> bool:
        """Drop held material at (x, y, z). Returns True on success."""
        if self.held_material is None:
            self.event_bus.publish("error_message", text="Nothing to drop")
            return False

        if not self.voxel_grid.in_bounds(x, y, z):
            return False

        target = self.voxel_grid.get(x, y, z)
        held_type, held_count = self.held_material

        if target == VOXEL_AIR:
            # Place as loose material
            self.voxel_grid.set(x, y, z, held_type, event_bus=self.event_bus)
            self.voxel_grid.set_loose(x, y, z, True)
            if held_count > 1:
                self.held_material = (held_type, held_count - 1)
            else:
                self.held_material = None
            self.event_bus.publish(
                "material_dropped", x=x, y=y, z=z, vtype=held_type
            )
            logger.debug("Dropped type=%d at (%d, %d, %d)", held_type, x, y, z)
            return True

        # Target is not air — attempt crafting
        self.event_bus.publish(
            "attempt_craft",
            x=x, y=y, z=z,
            held_type=held_type,
            held_count=held_count,
            target_type=target,
        )
        return True

    def _on_left_click(self, x: int, y: int, z: int, mode: str) -> None:
        if mode == "move":
            self.pick_up(x, y, z)

    def _on_right_click(self, x: int, y: int, z: int, mode: str) -> None:
        if mode == "move":
            self.drop(x, y, z)
