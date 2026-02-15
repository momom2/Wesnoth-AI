"""Pick up and drop loose materials."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dungeon_builder.config import VOXEL_AIR, VOXEL_DOOR

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.world.voxel_grid import VoxelGrid
    from dungeon_builder.building.crafting_system import CraftingSystem

logger = logging.getLogger("dungeon_builder.building")


class MoveSystem:
    """Lets the player pick up loose material and drop it elsewhere.

    Held material is a (voxel_type, count) tuple or None.
    Temperature and humidity are averaged across picked-up blocks and
    restored when dropped, so moving hot blocks carries their heat.
    Dropping onto a non-air voxel triggers an "attempt_craft" event
    so the crafting system can check for recipes.
    Dropping onto air first checks for recipes (air-target crafts like
    spikes, doors, slopes) before falling through to place-as-loose.
    """

    def __init__(self, event_bus: EventBus, voxel_grid: VoxelGrid) -> None:
        self.event_bus = event_bus
        self.voxel_grid = voxel_grid
        self.held_material: tuple[int, int] | None = None  # (vtype, count)
        self.held_temperature: float = 0.0  # running average temperature
        self.held_humidity: float = 0.0     # running average humidity
        self.crafting_system: CraftingSystem | None = None  # set after construction

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

        # Capture temperature and humidity before removing the block
        block_temp = self.voxel_grid.get_temperature(x, y, z)
        block_hum = self.voxel_grid.get_humidity(x, y, z)

        if self.held_material is not None:
            held_type, held_count = self.held_material
            if held_type == vtype:
                # Running average of temperature and humidity
                new_count = held_count + 1
                self.held_temperature = (
                    (self.held_temperature * held_count + block_temp) / new_count
                )
                self.held_humidity = (
                    (self.held_humidity * held_count + block_hum) / new_count
                )
                self.held_material = (held_type, new_count)
            else:
                self.event_bus.publish(
                    "error_message", text="Already holding a different material"
                )
                return False
        else:
            self.held_material = (vtype, 1)
            self.held_temperature = block_temp
            self.held_humidity = block_hum

        # Remove the voxel (set to AIR clears loose flag automatically)
        self.voxel_grid.set(x, y, z, VOXEL_AIR, event_bus=self.event_bus)
        # Clear temperature and humidity at source (carried in held state)
        self.voxel_grid.set_temperature(x, y, z, 0.0)
        self.voxel_grid.set_humidity(x, y, z, 0.0)
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
            # Try crafting first (air-target recipes like spikes, doors, slopes)
            held_before = self.held_material
            self.event_bus.publish(
                "attempt_craft",
                x=x, y=y, z=z,
                held_type=held_type,
                held_count=held_count,
                target_type=VOXEL_AIR,
            )
            # Check if craft system handled it:
            # 1. Material was consumed (auto-executed single match)
            if self.held_material != held_before:
                return True
            # 2. Menu is open (multiple matches, waiting for selection)
            if self.crafting_system is not None and self.crafting_system.has_pending_craft():
                return True

            # No recipe matched — fall through to place-as-loose
            self.voxel_grid.set(x, y, z, held_type, event_bus=self.event_bus)
            self.voxel_grid.set_loose(x, y, z, True)
            self.voxel_grid.set_temperature(x, y, z, self.held_temperature)
            self.voxel_grid.set_humidity(x, y, z, self.held_humidity)
            if held_count > 1:
                self.held_material = (held_type, held_count - 1)
            else:
                self.held_material = None
                self.held_temperature = 0.0
                self.held_humidity = 0.0
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
        if mode != "move":
            return

        vtype = self.voxel_grid.get(x, y, z)

        # Door toggle: clicking a non-loose door toggles its state
        if vtype == VOXEL_DOOR and not self.voxel_grid.is_loose(x, y, z):
            current = self.voxel_grid.get_block_state(x, y, z)
            new_state = 0 if current == 1 else 1
            self.voxel_grid.set_block_state(x, y, z, new_state)
            self.event_bus.publish(
                "door_toggled", x=x, y=y, z=z, state=new_state
            )
            logger.debug(
                "Door toggled at (%d, %d, %d) -> state=%d", x, y, z, new_state
            )
            return

        # Default: pick up loose material
        self.pick_up(x, y, z)

    def _on_right_click(self, x: int, y: int, z: int, mode: str) -> None:
        if mode == "move":
            self.drop(x, y, z)
