"""Pick up and drop loose materials.

Inventory is a multi-type bag: the player can hold multiple material types
simultaneously with no limit on quantity.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dungeon_builder.config import VOXEL_AIR, VOXEL_DOOR, VOXEL_PUMP, METAL_NONE

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.core.game_state import GameState
    from dungeon_builder.world.voxel_grid import VoxelGrid

logger = logging.getLogger("dungeon_builder.building")


class MoveSystem:
    """Lets the player pick up loose material and drop it elsewhere.

    Held materials are stored as a dict mapping voxel_type -> count.
    Temperature and humidity are tracked per-type as running averages,
    so moving hot blocks carries their heat.

    Dropping onto air places a loose block.  Crafting is handled
    separately via the recipe panel (no auto-craft on drop).
    """

    def __init__(
        self, event_bus: EventBus, voxel_grid: VoxelGrid, game_state: GameState,
    ) -> None:
        self.event_bus = event_bus
        self.voxel_grid = voxel_grid
        self.game_state = game_state

        # Multi-type bag: vtype -> count
        self.held_materials: dict[int, int] = {}
        # Per-type running average temperature/humidity
        self.held_temperatures: dict[int, float] = {}
        self.held_humidities: dict[int, float] = {}
        # Per-type metal variant: vtype -> metal_type (uint8)
        self.held_metal_types: dict[int, int] = {}
        # Track which type to drop on right-click (last picked up type)
        self._last_picked_type: int | None = None

        event_bus.subscribe("voxel_left_clicked", self._on_left_click)
        event_bus.subscribe("voxel_right_clicked", self._on_right_click)

    # ── Helpers ──────────────────────────────────────────────────────

    def has_material(self, vtype: int) -> bool:
        """Return True if the bag contains at least one of this type."""
        return self.held_materials.get(vtype, 0) > 0

    def get_count(self, vtype: int) -> int:
        """Return count of a specific material type in the bag."""
        return self.held_materials.get(vtype, 0)

    def total_count(self) -> int:
        """Return total number of items across all types."""
        return sum(self.held_materials.values())

    def consume(self, vtype: int, count: int = 1) -> bool:
        """Remove *count* units of *vtype* from the bag.

        Returns True if successful, False if insufficient.
        """
        current = self.held_materials.get(vtype, 0)
        if current < count:
            return False
        new_count = current - count
        if new_count == 0:
            del self.held_materials[vtype]
            self.held_temperatures.pop(vtype, None)
            self.held_humidities.pop(vtype, None)
            self.held_metal_types.pop(vtype, None)
            if self._last_picked_type == vtype:
                # Fall back to any remaining type
                self._last_picked_type = (
                    next(iter(self.held_materials)) if self.held_materials else None
                )
        else:
            self.held_materials[vtype] = new_count
        self.event_bus.publish(
            "material_dropped",
            materials=dict(self.held_materials),
        )
        return True

    def get_held_metal_type(self, vtype: int) -> int:
        """Return the metal_type of the held block of *vtype*, or METAL_NONE."""
        return self.held_metal_types.get(vtype, METAL_NONE)

    # ── Pick up ──────────────────────────────────────────────────────

    def pick_up(self, x: int, y: int, z: int) -> bool:
        """Pick up loose material at (x, y, z). Returns True on success."""
        if not self.voxel_grid.in_bounds(x, y, z):
            return False

        vtype = self.voxel_grid.get(x, y, z)
        if vtype == VOXEL_AIR:
            return False

        # Can only pick up within claimed territory
        if not self.voxel_grid.is_visible(x, y, z):
            self.event_bus.publish(
                "error_message", text="Too far from claimed territory"
            )
            return False

        if not self.voxel_grid.is_loose(x, y, z):
            self.event_bus.publish(
                "error_message", text="Not loose, dig first"
            )
            return False

        # Capture temperature, humidity, and metal_type before removing the block
        block_temp = self.voxel_grid.get_temperature(x, y, z)
        block_hum = self.voxel_grid.get_humidity(x, y, z)
        block_metal = self.voxel_grid.get_metal_type(x, y, z)

        # Accumulate into bag with running average for temp/humidity
        old_count = self.held_materials.get(vtype, 0)
        new_count = old_count + 1
        if old_count > 0:
            old_temp = self.held_temperatures.get(vtype, 0.0)
            old_hum = self.held_humidities.get(vtype, 0.0)
            self.held_temperatures[vtype] = (
                (old_temp * old_count + block_temp) / new_count
            )
            self.held_humidities[vtype] = (
                (old_hum * old_count + block_hum) / new_count
            )
        else:
            self.held_temperatures[vtype] = block_temp
            self.held_humidities[vtype] = block_hum
        # Store metal_type (most recent pick-up wins for same vtype)
        if block_metal != METAL_NONE:
            self.held_metal_types[vtype] = block_metal
        self.held_materials[vtype] = new_count
        self._last_picked_type = vtype

        # Remove the voxel (set to AIR clears loose flag automatically)
        self.voxel_grid.set(x, y, z, VOXEL_AIR, event_bus=self.event_bus)
        # Clear temperature and humidity at source (carried in held state)
        self.voxel_grid.set_temperature(x, y, z, 0.0)
        self.voxel_grid.set_humidity(x, y, z, 0.0)
        self.event_bus.publish(
            "material_picked_up",
            materials=dict(self.held_materials),
        )
        logger.debug("Picked up type=%d at (%d, %d, %d)", vtype, x, y, z)
        return True

    # ── Drop ─────────────────────────────────────────────────────────

    def drop(self, x: int, y: int, z: int) -> bool:
        """Drop one unit of the last-picked material at (x, y, z).

        Returns True on success.  Dropping is simple placement — no
        auto-crafting.  Crafting goes through the recipe panel instead.
        """
        if not self.held_materials:
            self.event_bus.publish("error_message", text="Nothing to drop")
            return False

        if not self.voxel_grid.in_bounds(x, y, z):
            return False

        target = self.voxel_grid.get(x, y, z)

        if target != VOXEL_AIR:
            self.event_bus.publish(
                "error_message", text="Can't drop on a solid block"
            )
            return False

        # Must be within claimed territory
        if not self.voxel_grid.is_claimed(x, y, z):
            self.event_bus.publish(
                "error_message", text="Too far from claimed territory"
            )
            return False

        # Determine which type to drop
        drop_type = self._last_picked_type
        if drop_type is None or drop_type not in self.held_materials:
            # Fall back to first available type
            drop_type = next(iter(self.held_materials))

        # Place as loose block
        self.voxel_grid.set(x, y, z, drop_type, event_bus=self.event_bus)
        self.voxel_grid.set_loose(x, y, z, True)
        # Restore carried temperature, humidity, and metal_type
        temp = self.held_temperatures.get(drop_type, 0.0)
        hum = self.held_humidities.get(drop_type, 0.0)
        metal = self.held_metal_types.get(drop_type, METAL_NONE)
        self.voxel_grid.set_temperature(x, y, z, temp)
        self.voxel_grid.set_humidity(x, y, z, hum)
        if metal != METAL_NONE:
            self.voxel_grid.set_metal_type(x, y, z, metal)

        # Decrement count
        self.consume(drop_type, 1)

        self.event_bus.publish(
            "material_dropped",
            x=x, y=y, z=z, vtype=drop_type,
            materials=dict(self.held_materials),
        )
        logger.debug("Dropped type=%d at (%d, %d, %d)", drop_type, x, y, z)
        return True

    # ── Event handlers ───────────────────────────────────────────────

    def _on_left_click(self, x: int, y: int, z: int, mode: str) -> None:
        if mode != "move":
            return

        # Craft mode: left-clicks handled by camera → craft_at_position
        if self.game_state.craft_mode_active:
            return

        vtype = self.voxel_grid.get(x, y, z)

        # Door toggle: clicking a non-loose door toggles its state
        if vtype == VOXEL_DOOR and not self.voxel_grid.is_loose(x, y, z) \
                and self.voxel_grid.is_visible(x, y, z):
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

        # Pump direction: clicking a non-loose pump cycles direction (0-5)
        if vtype == VOXEL_PUMP and not self.voxel_grid.is_loose(x, y, z) \
                and self.voxel_grid.is_visible(x, y, z):
            current = self.voxel_grid.get_block_state(x, y, z)
            new_state = (current + 1) % 6
            self.voxel_grid.set_block_state(x, y, z, new_state)
            self.event_bus.publish(
                "pump_direction_changed", x=x, y=y, z=z, direction=new_state
            )
            logger.debug(
                "Pump direction at (%d, %d, %d) -> %d", x, y, z, new_state
            )
            return

        # Default: pick up loose material
        self.pick_up(x, y, z)

    def _on_right_click(self, x: int, y: int, z: int, mode: str) -> None:
        if mode != "move":
            return
        # In craft mode, right-click cancels (handled by camera)
        if self.game_state.craft_mode_active:
            return
        self.drop(x, y, z)
