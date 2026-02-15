"""Crafting system: matches recipes and executes crafts."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dungeon_builder.building.crafting_book import CraftingBook

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.world.voxel_grid import VoxelGrid
    from dungeon_builder.building.move_system import MoveSystem

logger = logging.getLogger("dungeon_builder.building")


class CraftingSystem:
    """Listens for attempt_craft events and executes matching recipes."""

    def __init__(
        self,
        event_bus: EventBus,
        voxel_grid: VoxelGrid,
        move_system: MoveSystem,
    ) -> None:
        self.event_bus = event_bus
        self.voxel_grid = voxel_grid
        self.move_system = move_system
        self.crafting_book = CraftingBook()

        event_bus.subscribe("attempt_craft", self._on_attempt_craft)

    def _on_attempt_craft(
        self,
        x: int,
        y: int,
        z: int,
        held_type: int,
        held_count: int,
        target_type: int,
    ) -> None:
        recipe = self.crafting_book.find_recipe(
            self.voxel_grid, x, y, z, held_type
        )
        if recipe is None:
            self.event_bus.publish(
                "error_message", text="No recipe matches"
            )
            return

        success = recipe.craft_fn(
            self.voxel_grid, x, y, z, held_type, self.event_bus
        )
        if success:
            # Consume one unit of held material
            if held_count > 1:
                self.move_system.held_material = (held_type, held_count - 1)
            else:
                self.move_system.held_material = None
            self.event_bus.publish(
                "craft_success", recipe=recipe.name, x=x, y=y, z=z
            )
            logger.info("Crafted '%s' at (%d, %d, %d)", recipe.name, x, y, z)
