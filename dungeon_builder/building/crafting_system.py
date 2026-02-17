"""Crafting system: highlight-mode state machine for recipe placement.

Flow:
1. Player clicks a recipe in the panel -> "craft_recipe_selected" event
2. System scans for valid positions and highlights them
3. Player clicks a highlighted voxel -> "craft_at_position" event
4. System executes the recipe, consumes material, re-scans highlights
5. Player cancels (ESC/right-click) or runs out of material -> exit craft mode
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dungeon_builder.building.crafting_book import CraftingBook, CraftingRecipe

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.core.game_state import GameState
    from dungeon_builder.world.voxel_grid import VoxelGrid
    from dungeon_builder.building.move_system import MoveSystem

logger = logging.getLogger("dungeon_builder.building")


class CraftingSystem:
    """Manages recipe selection, position highlighting, and craft execution.

    Replaces the old auto-craft-on-drop flow with an explicit
    panel → highlight → click workflow.
    """

    def __init__(
        self,
        event_bus: EventBus,
        voxel_grid: VoxelGrid,
        move_system: MoveSystem,
        game_state: GameState,
    ) -> None:
        self.event_bus = event_bus
        self.voxel_grid = voxel_grid
        self.move_system = move_system
        self.game_state = game_state
        self.crafting_book = CraftingBook()

        # Highlight-mode state
        self._active_recipe: CraftingRecipe | None = None
        self._active_held_type: int | None = None
        self._highlighted_positions: set[tuple[int, int, int]] = set()
        self._current_z: int = 0

        event_bus.subscribe("craft_recipe_selected", self._on_recipe_selected)
        event_bus.subscribe("craft_cancel", self._on_craft_cancel)
        event_bus.subscribe("craft_at_position", self._on_craft_at_position)
        event_bus.subscribe("z_level_changed", self._on_z_changed)

    @property
    def is_craft_mode_active(self) -> bool:
        """Return True if a recipe is selected and highlights are showing."""
        return self._active_recipe is not None

    # ── Recipe selection ─────────────────────────────────────────────

    def _on_recipe_selected(self, recipe_name: str, **kw) -> None:
        """Player clicked a recipe in the panel."""
        # If already in craft mode for this recipe, toggle it off
        if self._active_recipe is not None and self._active_recipe.name == recipe_name:
            self._exit_craft_mode()
            return

        recipe = self.crafting_book.get_recipe_by_name(recipe_name)
        if recipe is None:
            logger.warning("Unknown recipe: %s", recipe_name)
            return

        # Find a matching held type
        held_keys = set(self.move_system.held_materials.keys())
        matching = held_keys & recipe.required_inputs
        if not matching:
            self.event_bus.publish(
                "error_message", text="Missing required material"
            )
            return

        # Pick the first matching held type
        held_type = next(iter(matching))

        self._active_recipe = recipe
        self._active_held_type = held_type

        # Scan and highlight
        self._scan_and_highlight(self._current_z)

        self.game_state.craft_mode_active = True
        self.event_bus.publish(
            "craft_mode_entered", recipe_name=recipe.name
        )
        logger.info(
            "Craft mode entered: %s (held_type=%d)", recipe.name, held_type
        )

    # ── Position scanning ────────────────────────────────────────────

    def _scan_and_highlight(self, z_level: int) -> None:
        """Find all valid craft positions at z_level and publish highlights."""
        if self._active_recipe is None or self._active_held_type is None:
            return

        positions = self.crafting_book.find_valid_positions(
            self._active_recipe, self.voxel_grid, self._active_held_type,
            z_level,
        )
        self._highlighted_positions = set(positions)

        self.event_bus.publish(
            "craft_highlights_updated",
            positions=self._highlighted_positions,
        )

    # ── Craft execution ──────────────────────────────────────────────

    def _on_craft_at_position(self, x: int, y: int, z: int, **kw) -> None:
        """Player clicked a position while in craft mode."""
        if self._active_recipe is None:
            return

        if (x, y, z) not in self._highlighted_positions:
            # Clicked non-highlighted position — cancel craft mode
            self._exit_craft_mode()
            return

        # Execute the recipe, passing held metal_type if available
        held_metal = self.move_system.get_held_metal_type(self._active_held_type)
        success = self._active_recipe.craft_fn(
            self.voxel_grid, x, y, z, self._active_held_type, self.event_bus,
            held_metal=held_metal,
        )
        if not success:
            self.event_bus.publish(
                "error_message", text="Craft failed"
            )
            return

        # Consume one unit of material
        self.move_system.consume(self._active_held_type, 1)

        self.event_bus.publish(
            "craft_success",
            recipe=self._active_recipe.name,
            x=x, y=y, z=z,
        )
        logger.info(
            "Crafted '%s' at (%d, %d, %d)", self._active_recipe.name, x, y, z
        )

        # Check if material is depleted
        if not self.move_system.has_material(self._active_held_type):
            self._exit_craft_mode()
            return

        # Re-scan highlights (crafted position may no longer be valid,
        # other positions may open up)
        self._scan_and_highlight(self._current_z)

    # ── Cancellation ─────────────────────────────────────────────────

    def _on_craft_cancel(self, **kw) -> None:
        """Player cancelled craft mode (ESC, right-click, panel toggle)."""
        if self._active_recipe is not None:
            self._exit_craft_mode()

    def _exit_craft_mode(self) -> None:
        """Clear craft mode state and notify subscribers."""
        self._active_recipe = None
        self._active_held_type = None
        self._highlighted_positions.clear()

        self.game_state.craft_mode_active = False
        self.event_bus.publish("craft_highlights_cleared")
        self.event_bus.publish("craft_mode_exited")
        logger.info("Craft mode exited")

    # ── Z-level change ───────────────────────────────────────────────

    def _on_z_changed(self, z: int, **kw) -> None:
        """Re-scan valid positions when the view layer changes."""
        self._current_z = z
        if self._active_recipe is not None:
            self._scan_and_highlight(z)
