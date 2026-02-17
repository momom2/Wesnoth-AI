"""Right-side recipe panel for the crafting system.

Shows all known recipes.  Recipes whose required input is in the player's
hand are enabled (green); others are greyed out.  Clicking an enabled
recipe enters craft-highlight mode.

Features:
- **Recipe pinning**: clicking a recipe toggles its pinned state (underlined
  name).  When the panel is closed, a floating overlay shows the pinned
  recipe.  Clicking the overlay (or re-clicking the recipe) unpins.
- **Ingredient highlighting**: clicking the "Requires: ..." label scans
  the current z-level for visible matching blocks and publishes
  ``ingredient_highlight`` (cyan tint for 3 s) or an error message.

Opened with the B key or the "Book [B]" button on the HUD.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from direct.gui.DirectGui import (
    DirectFrame,
    DirectLabel,
    DirectButton,
    DirectScrolledFrame,
    DGG,
)
from panda3d.core import TextNode
from direct.showbase.ShowBase import ShowBase

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.core.game_state import GameState
    from dungeon_builder.building.crafting_journal import CraftingJournal
    from dungeon_builder.building.move_system import MoveSystem

logger = logging.getLogger("dungeon_builder.ui")

# Map voxel type int -> friendly name (duplicated from hud.py to avoid
# coupling; could be moved to a shared module if desired).
_VTYPE_NAMES = {
    1: "Dirt", 2: "Stone", 3: "Bedrock", 4: "Core",
    10: "Sandstone", 11: "Limestone", 12: "Shale", 13: "Chalk",
    20: "Slate", 21: "Marble", 22: "Gneiss",
    30: "Granite", 31: "Basalt", 32: "Obsidian",
    40: "Iron Ore", 41: "Copper Ore", 42: "Gold Ore", 43: "Mana Crystal",
    50: "Lava", 51: "Water",
    60: "Iron Ingot", 61: "Copper Ingot", 62: "Gold Ingot",
    63: "Enchanted Metal",
    70: "Reinforced Wall", 71: "Spike", 72: "Door", 73: "Treasure",
    74: "Rolling Stone", 75: "Tarp", 76: "Slope", 77: "Stairs",
}


def _input_names(required_inputs: frozenset[int]) -> str:
    """Return a comma-separated string of material names for the inputs."""
    names = [_VTYPE_NAMES.get(v, f"Type {v}") for v in sorted(required_inputs)]
    return ", ".join(names)


class CraftingBookPanel:
    """Right-side interactive recipe panel.

    Discovered recipes show their name and required input.  Undiscovered
    recipes show as ``???``.  Enabled recipes (player has the input
    material) are green and clickable; disabled ones are grey.
    """

    def __init__(
        self,
        app: ShowBase,
        event_bus: EventBus,
        crafting_journal: CraftingJournal,
        move_system: MoveSystem,
        game_state: GameState | None = None,
    ) -> None:
        self.app = app
        self.event_bus = event_bus
        self.journal = crafting_journal
        self.move_system = move_system
        self.game_state = game_state
        self._visible = False
        self._selected_recipe: str | None = None  # name of recipe in craft mode
        self._pinned_recipe: str | None = None     # name of pinned recipe
        self._current_z: int = 0                   # track z-level for ingredient scan

        # Build DirectGui elements (hidden initially)
        self._build_panel()
        self.panel.hide()

        # Floating pinned-recipe overlay (hidden initially)
        self._build_pinned_overlay()
        self._pinned_frame.hide()

        # Keyboard toggle
        app.accept("b", self.toggle)

        # HUD button toggle (via event)
        event_bus.subscribe("toggle_crafting_book", self._on_toggle_event)

        # Live updates
        event_bus.subscribe("recipe_discovered", self._on_recipe_discovered)
        event_bus.subscribe("material_picked_up", self._on_material_changed)
        event_bus.subscribe("material_dropped", self._on_material_changed)
        event_bus.subscribe("craft_success", self._on_craft_success)
        event_bus.subscribe("craft_mode_entered", self._on_craft_mode_entered)
        event_bus.subscribe("craft_mode_exited", self._on_craft_mode_exited)
        event_bus.subscribe("z_level_changed", self._on_z_level_changed)

    # ── Panel construction ────────────────────────────────────────────

    def _build_panel(self) -> None:
        """Construct the DirectGui elements for the panel."""
        a2d = self.app.aspect2d

        # Right-side panel background
        self.panel = DirectFrame(
            frameColor=(0.05, 0.05, 0.1, 0.92),
            frameSize=(0.55, 1.35, -0.92, 0.92),
            pos=(0, 0, 0),
            parent=a2d,
            sortOrder=50,
        )

        # Title
        self.title_label = DirectLabel(
            text="Crafting [B]",
            text_fg=(0.9, 0.8, 0.4, 1),
            text_scale=0.05,
            text_align=TextNode.A_left,
            pos=(0.6, 0, 0.84),
            frameColor=(0, 0, 0, 0),
            parent=self.panel,
        )

        # Close button
        self.close_btn = DirectButton(
            text="X",
            text_scale=0.045,
            text_fg=(0.9, 0.3, 0.3, 1),
            frameSize=(-0.035, 0.035, -0.025, 0.04),
            frameColor=(0.15, 0.15, 0.2, 0.8),
            pos=(1.28, 0, 0.86),
            command=self.toggle,
            parent=self.panel,
        )

        # Discovered count
        self.count_label = DirectLabel(
            text="",
            text_fg=(0.6, 0.6, 0.6, 1),
            text_scale=0.035,
            text_align=TextNode.A_left,
            pos=(0.6, 0, 0.77),
            frameColor=(0, 0, 0, 0),
            parent=self.panel,
        )

        # Scrollable content frame for recipe entries
        self.scroll_frame = DirectScrolledFrame(
            frameColor=(0, 0, 0, 0),
            frameSize=(0.55, 1.33, -0.90, 0.73),
            canvasSize=(0.55, 1.30, -2.5, 0),
            scrollBarWidth=0.03,
            pos=(0, 0, 0),
            parent=self.panel,
        )
        # Hide horizontal scrollbar
        self.scroll_frame.horizontalScroll.hide()

        # Recipe entry widgets: list of (button, desc_btn, recipe_name)
        # desc_btn is now a DirectButton (clickable) for ingredient highlighting
        self._recipe_widgets: list[tuple[DirectButton, DirectButton, str]] = []
        self._create_recipe_entries()

    def _build_pinned_overlay(self) -> None:
        """Build the small floating overlay shown when the panel is closed
        and a recipe is pinned."""
        a2d = self.app.aspect2d
        self._pinned_frame = DirectFrame(
            frameColor=(0.08, 0.08, 0.14, 0.88),
            frameSize=(0.0, 0.45, -0.07, 0.04),
            pos=(1.1, 0, -0.85),
            parent=a2d,
            sortOrder=55,
        )
        self._pinned_label = DirectButton(
            text="",
            text_fg=(0.9, 0.8, 0.4, 1),
            text_scale=0.032,
            text_align=TextNode.A_left,
            frameSize=(0.0, 0.42, -0.065, 0.035),
            frameColor=(0, 0, 0, 0),
            pos=(0.015, 0, 0.0),
            parent=self._pinned_frame,
            command=self._on_pinned_overlay_click,
        )

    def _create_recipe_entries(self) -> None:
        """Create a button + clickable description for each recipe slot."""
        recipes_data = self.journal.get_all_recipes_display()
        canvas = self.scroll_frame.getCanvas()
        y_start = -0.04
        y_step = 0.10

        for i, entry in enumerate(recipes_data):
            y = y_start - i * y_step
            recipe_name = entry["name"]

            btn = DirectButton(
                text=recipe_name,
                text_scale=0.038,
                text_fg=(0.5, 0.5, 0.5, 1),
                text_align=TextNode.A_left,
                frameSize=(0.0, 0.72, -0.025, 0.045),
                frameColor=(0.12, 0.12, 0.18, 0.7),
                pos=(0.02, 0, y),
                parent=canvas,
                command=self._on_recipe_click,
                extraArgs=[recipe_name],
                state=DGG.DISABLED,
            )

            # Clickable description label for ingredient highlighting
            desc_btn = DirectButton(
                text=entry["description"],
                text_fg=(0.4, 0.4, 0.4, 1),
                text_scale=0.028,
                text_align=TextNode.A_left,
                frameSize=(0.0, 0.72, -0.018, 0.022),
                frameColor=(0, 0, 0, 0),
                pos=(0.04, 0, y - 0.04),
                parent=canvas,
                command=self._on_ingredient_click,
                extraArgs=[recipe_name],
                state=DGG.DISABLED,
            )
            self._recipe_widgets.append((btn, desc_btn, recipe_name))

        # Adjust canvas size to fit all entries
        total_h = len(recipes_data) * y_step + 0.1
        self.scroll_frame["canvasSize"] = (0.0, 0.75, -total_h, 0)

    # ── Click handlers ────────────────────────────────────────────────

    def _on_recipe_click(self, recipe_name: str) -> None:
        """Handle click on a recipe button — enter craft mode + toggle pin."""
        if recipe_name == "???":
            return

        # Toggle pin
        if self._pinned_recipe == recipe_name:
            self._pinned_recipe = None
        else:
            self._pinned_recipe = recipe_name
        self._update_pinned_overlay()

        # Also enter craft mode
        self.event_bus.publish(
            "craft_recipe_selected", recipe_name=recipe_name
        )

    def _on_ingredient_click(self, recipe_name: str) -> None:
        """Scan current z-level for visible instances of the recipe's inputs.

        Publishes ``ingredient_highlight`` with the set of positions, or
        ``error_message`` if none found.
        """
        if recipe_name == "???":
            return
        if self.game_state is None:
            return

        recipe = self.journal._crafting_book.get_recipe_by_name(recipe_name)
        if recipe is None:
            return

        grid = self.game_state.voxel_grid
        if grid is None:
            return

        z = self._current_z
        positions: set[tuple[int, int, int]] = set()
        for vtype in recipe.required_inputs:
            for x in range(grid.width):
                for y in range(grid.depth):
                    if grid.get(x, y, z) == vtype and grid.is_visible(x, y, z):
                        positions.add((x, y, z))

        if positions:
            self.event_bus.publish("ingredient_highlight", positions=positions)
        else:
            names = [_VTYPE_NAMES.get(v, "?") for v in recipe.required_inputs]
            self.event_bus.publish(
                "error_message",
                text=f"You can't find any {'/'.join(names)} nearby",
            )

    def _on_pinned_overlay_click(self) -> None:
        """Clicking the floating overlay unpins the recipe."""
        self._pinned_recipe = None
        self._update_pinned_overlay()
        if self._visible:
            self._refresh()

    # ── Refresh ───────────────────────────────────────────────────────

    def _refresh(self) -> None:
        """Update all recipe entries from current journal + bag state."""
        recipes_data = self.journal.get_all_recipes_display()
        self.count_label["text"] = (
            f"Discovered: {self.journal.discovered_count} / {self.journal.total_recipes}"
        )

        held_keys = set(self.move_system.held_materials.keys())

        for i, (btn, desc_btn, _old_name) in enumerate(self._recipe_widgets):
            entry = recipes_data[i]
            actual_name = entry["name"]

            # Get the recipe object for required_inputs display
            recipe_obj = None
            if entry["discovered"]:
                from dungeon_builder.building.crafting_book import CraftingBook
                # Access via journal's reference
                recipe_obj = self.journal._crafting_book.recipes[i]

            is_pinned = (self._pinned_recipe == actual_name)

            if entry["discovered"]:
                # Name with underline for pinned recipe
                display_name = f"\1underline\1{actual_name}\2" if is_pinned else actual_name
                btn["text"] = display_name
                has_input = (
                    recipe_obj is not None
                    and bool(held_keys & recipe_obj.required_inputs)
                )
                req_text = (
                    f"Requires: {_input_names(recipe_obj.required_inputs)}"
                    if recipe_obj is not None else ""
                )
                desc_btn["text"] = req_text

                # Enable the ingredient label (clickable) if discovered
                desc_btn["state"] = DGG.NORMAL

                if self._selected_recipe == actual_name:
                    # Active craft mode — golden highlight
                    btn["text_fg"] = (1.0, 0.9, 0.3, 1)
                    btn["frameColor"] = (0.25, 0.22, 0.08, 0.9)
                    btn["state"] = DGG.NORMAL
                elif has_input:
                    # Enabled — has material
                    btn["text_fg"] = (0.3, 1.0, 0.3, 1)
                    btn["frameColor"] = (0.12, 0.18, 0.12, 0.8)
                    desc_btn["text_fg"] = (0.7, 0.9, 0.7, 1)
                    btn["state"] = DGG.NORMAL
                else:
                    # Disabled — no material (but still clickable for pin/ingredient)
                    btn["text_fg"] = (0.5, 0.5, 0.5, 1)
                    btn["frameColor"] = (0.12, 0.12, 0.18, 0.7)
                    desc_btn["text_fg"] = (0.4, 0.4, 0.4, 1)
                    btn["state"] = DGG.NORMAL  # always clickable for pin toggle
            else:
                btn["text"] = "???"
                desc_btn["text"] = "[Craft this recipe to reveal]"
                btn["text_fg"] = (0.35, 0.35, 0.35, 1)
                btn["frameColor"] = (0.1, 0.1, 0.1, 0.6)
                desc_btn["text_fg"] = (0.3, 0.3, 0.3, 1)
                btn["state"] = DGG.DISABLED
                desc_btn["state"] = DGG.DISABLED

            # Update stored name reference
            self._recipe_widgets[i] = (btn, desc_btn, actual_name)

    # ── Pinned overlay ────────────────────────────────────────────────

    def _update_pinned_overlay(self) -> None:
        """Show/hide the floating pinned overlay based on current state."""
        if self._pinned_recipe and not self._visible:
            # Get recipe info for display
            recipe = self.journal._crafting_book.get_recipe_by_name(self._pinned_recipe)
            if recipe is not None:
                req = _input_names(recipe.required_inputs)
                self._pinned_label["text"] = f"{self._pinned_recipe}\n  {req}"
            else:
                self._pinned_label["text"] = self._pinned_recipe
            self._pinned_frame.show()
        else:
            self._pinned_frame.hide()

    # ── Visibility ────────────────────────────────────────────────────

    def toggle(self) -> None:
        """Toggle panel visibility."""
        if self._visible:
            self.panel.hide()
            self._visible = False
            # Show floating overlay if a recipe is pinned
            self._update_pinned_overlay()
        else:
            self._refresh()
            self.panel.show()
            self._visible = True
            # Hide overlay when panel is open
            self._pinned_frame.hide()

    @property
    def is_visible(self) -> bool:
        """Whether the panel is currently shown."""
        return self._visible

    # ── Event handlers ────────────────────────────────────────────────

    def _on_toggle_event(self, **kwargs) -> None:
        """Handle toggle_crafting_book event from HUD button."""
        self.toggle()

    def _on_recipe_discovered(self, **kwargs) -> None:
        """Live-update the panel if it is currently visible."""
        if self._visible:
            self._refresh()

    def _on_material_changed(self, **kwargs) -> None:
        """Refresh enabled states when inventory changes."""
        if self._visible:
            self._refresh()

    def _on_craft_success(self, **kwargs) -> None:
        """Refresh after a craft completes."""
        if self._visible:
            self._refresh()

    def _on_craft_mode_entered(self, recipe_name: str, **kwargs) -> None:
        """Highlight the selected recipe button."""
        self._selected_recipe = recipe_name
        if self._visible:
            self._refresh()

    def _on_craft_mode_exited(self, **kwargs) -> None:
        """Un-highlight all recipe buttons."""
        self._selected_recipe = None
        if self._visible:
            self._refresh()

    def _on_z_level_changed(self, z: int, **kwargs) -> None:
        """Track z-level for ingredient scanning."""
        self._current_z = z
