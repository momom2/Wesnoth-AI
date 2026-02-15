"""Crafting system: matches recipes and executes crafts.

Supports an event-driven craft menu:
- 0 matches: publish error (non-air target) or do nothing (air target)
- 1 match: auto-execute the recipe
- 2+ matches: publish "craft_menu" event with all options, wait for "craft_selected"
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dungeon_builder.building.crafting_book import CraftingBook
from dungeon_builder.config import VOXEL_AIR

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.world.voxel_grid import VoxelGrid
    from dungeon_builder.building.move_system import MoveSystem

logger = logging.getLogger("dungeon_builder.building")


class CraftingSystem:
    """Listens for attempt_craft events and executes matching recipes.

    When multiple recipes match, publishes a craft_menu event and waits
    for a craft_selected event to execute the chosen recipe.
    """

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

        # Pending craft state for menu selection
        self._pending_craft: tuple | None = None

        # Flag set after each attempt_craft so MoveSystem can check results
        self._last_craft_matched: bool = False

        event_bus.subscribe("attempt_craft", self._on_attempt_craft)
        event_bus.subscribe("craft_selected", self._on_craft_selected)

    def has_pending_craft(self) -> bool:
        """Return True if a craft menu is open and waiting for selection."""
        return self._pending_craft is not None

    def _on_attempt_craft(
        self,
        x: int,
        y: int,
        z: int,
        held_type: int,
        held_count: int,
        target_type: int,
    ) -> None:
        matches = self.crafting_book.find_all_recipes(
            self.voxel_grid, x, y, z, held_type
        )

        self._last_craft_matched = len(matches) > 0

        if len(matches) == 0:
            # Only publish error if target is non-air (air targets fall through
            # to place-as-loose in MoveSystem)
            if target_type != VOXEL_AIR:
                self.event_bus.publish(
                    "error_message", text="No recipe matches"
                )
            return

        if len(matches) == 1:
            # Auto-execute the single matching recipe
            self._execute_recipe(matches[0], x, y, z, held_type, held_count)
            return

        # Multiple matches — open craft menu for player selection
        self._pending_craft = (x, y, z, held_type, held_count, matches)
        recipe_info = [
            {"name": r.name, "description": r.description} for r in matches
        ]
        self.event_bus.publish(
            "craft_menu", recipes=recipe_info, x=x, y=y, z=z
        )
        logger.info(
            "Craft menu opened at (%d, %d, %d) with %d options",
            x, y, z, len(matches),
        )

    def _on_craft_selected(self, recipe_index: int, **kw) -> None:
        """Player selected a recipe from the menu."""
        if self._pending_craft is None:
            return

        x, y, z, held_type, held_count, recipes = self._pending_craft
        self._pending_craft = None

        if 0 <= recipe_index < len(recipes):
            self._execute_recipe(
                recipes[recipe_index], x, y, z, held_type, held_count
            )
        else:
            logger.warning("Invalid recipe_index %d (max %d)", recipe_index, len(recipes) - 1)

    def _execute_recipe(self, recipe, x, y, z, held_type, held_count) -> None:
        """Execute a recipe and consume one unit of held material."""
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
