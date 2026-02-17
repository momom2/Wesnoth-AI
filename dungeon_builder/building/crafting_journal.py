"""Crafting recipe discovery tracking.

Records which recipes the player has successfully crafted at least once.
Provides a pure-data API for UI display — no Panda3D dependency.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.building.crafting_book import CraftingBook

logger = logging.getLogger("dungeon_builder.building")


class CraftingJournal:
    """Tracks which crafting recipes the player has discovered.

    Subscribes to ``craft_success`` events on the EventBus and records
    each unique recipe name.  Publishes ``recipe_discovered`` the first
    time a recipe is crafted.
    """

    def __init__(self, event_bus: EventBus, crafting_book: CraftingBook) -> None:
        self._event_bus = event_bus
        self._crafting_book = crafting_book
        self._discovered: set[str] = set()

        event_bus.subscribe("craft_success", self._on_craft_success)

    def _on_craft_success(self, recipe: str, **kwargs) -> None:
        """Record a newly discovered recipe and publish event if first time."""
        is_new = recipe not in self._discovered
        self._discovered.add(recipe)
        if is_new:
            self._event_bus.publish(
                "recipe_discovered",
                recipe=recipe,
                total=len(self._discovered),
            )
            logger.debug("Recipe discovered: %s (%d total)", recipe, len(self._discovered))

    def is_discovered(self, recipe_name: str) -> bool:
        """Return True if the recipe has been crafted at least once."""
        return recipe_name in self._discovered

    @property
    def discovered_count(self) -> int:
        """Number of unique recipes discovered so far."""
        return len(self._discovered)

    @property
    def total_recipes(self) -> int:
        """Total number of recipes in the crafting book."""
        return len(self._crafting_book.recipes)

    @property
    def discovered_names(self) -> frozenset[str]:
        """Return an immutable copy of all discovered recipe names."""
        return frozenset(self._discovered)

    def get_all_recipes_display(self) -> list[dict]:
        """Return all recipes with discovery status for UI display.

        Returns a list of dicts in crafting book order::

            {"name": str, "description": str, "discovered": bool}

        Undiscovered recipes show ``"???"`` as name and a placeholder
        description.
        """
        result = []
        for recipe in self._crafting_book.recipes:
            if recipe.name in self._discovered:
                result.append({
                    "name": recipe.name,
                    "description": recipe.description,
                    "discovered": True,
                })
            else:
                result.append({
                    "name": "???",
                    "description": "[Craft this recipe to reveal]",
                    "discovered": False,
                })
        return result

    def discover_all(self) -> None:
        """Debug helper: mark every recipe as discovered."""
        for recipe in self._crafting_book.recipes:
            self._discovered.add(recipe.name)
