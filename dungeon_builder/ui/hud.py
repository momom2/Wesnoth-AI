"""Main HUD overlay: core HP, tick counter, speed controls, Z-level, tool/hand info."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from direct.gui.DirectGui import DirectFrame, DirectLabel, DirectButton
from panda3d.core import TextNode
from direct.showbase.ShowBase import ShowBase
from direct.task.Task import Task

from dungeon_builder.config import VOXEL_COLORS, ARCHETYPE_COLORS

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.core.game_state import GameState

logger = logging.getLogger("dungeon_builder.ui")

# Map voxel type int to a friendly name
_VTYPE_NAMES = {
    1: "Dirt", 2: "Stone", 3: "Bedrock", 4: "Core",
    10: "Sandstone", 11: "Limestone", 12: "Shale", 13: "Chalk",
    20: "Slate", 21: "Marble", 22: "Gneiss",
    30: "Granite", 31: "Basalt", 32: "Obsidian",
    40: "Iron Ore", 41: "Copper Ore", 42: "Gold Ore", 43: "Mana Crystal",
    50: "Lava", 51: "Water",
    60: "Iron Ingot", 61: "Copper Ingot", 62: "Gold Ingot", 63: "Enchanted Metal",
    70: "Reinforced Wall", 71: "Spike", 72: "Door", 73: "Treasure",
    74: "Rolling Stone", 75: "Tarp", 76: "Slope", 77: "Stairs",
}


class HUD:
    """HUD showing game state, tool info, held material, and error messages."""

    def __init__(self, app: ShowBase, event_bus: EventBus, game_state: GameState) -> None:
        self.app = app
        self.event_bus = event_bus
        self.game_state = game_state

        a2d = app.aspect2d

        # Top bar background
        self.top_bar = DirectFrame(
            frameColor=(0, 0, 0, 0.7),
            frameSize=(-2.0, 2.0, -0.07, 0.07),
            pos=(0, 0, 0.93),
            parent=a2d,
        )

        # Core HP
        self.core_hp_label = DirectLabel(
            text="Core: 100/100",
            text_fg=(1, 0.3, 0.3, 1),
            text_scale=0.05,
            text_align=TextNode.A_left,
            pos=(-1.7, 0, 0.91),
            frameColor=(0, 0, 0, 0),
            parent=a2d,
        )

        # Tick counter
        self.time_label = DirectLabel(
            text="Tick: 0",
            text_fg=(1, 1, 1, 1),
            text_scale=0.05,
            text_align=TextNode.A_center,
            pos=(0, 0, 0.91),
            frameColor=(0, 0, 0, 0),
            parent=a2d,
        )

        # Speed controls
        self.speed_label = DirectLabel(
            text="PLAYING",
            text_fg=(0, 1, 0, 1),
            text_scale=0.05,
            text_align=TextNode.A_center,
            pos=(0.8, 0, 0.91),
            frameColor=(0, 0, 0, 0),
            parent=a2d,
        )

        btn_style = dict(
            text_scale=0.05,
            frameSize=(-0.08, 0.08, -0.03, 0.05),
            relief=1,
            parent=a2d,
        )
        self.pause_btn = DirectButton(
            text="||", pos=(1.1, 0, 0.92),
            command=self._set_speed, extraArgs=[0], **btn_style
        )
        self.play_btn = DirectButton(
            text=">", pos=(1.3, 0, 0.92),
            command=self._set_speed, extraArgs=[1], **btn_style
        )
        self.fast_btn = DirectButton(
            text=">>", pos=(1.5, 0, 0.92),
            command=self._set_speed, extraArgs=[2], **btn_style
        )

        # Z-level indicator
        self.z_label = DirectLabel(
            text="Z: -1",
            text_fg=(0.7, 0.9, 1, 1),
            text_scale=0.05,
            text_align=TextNode.A_right,
            pos=(1.7, 0, 0.91),
            frameColor=(0, 0, 0, 0),
            parent=a2d,
        )

        # Intruder count
        self.intruder_label = DirectLabel(
            text="Intruders: 0",
            text_fg=(1, 0.6, 0.6, 1),
            text_scale=0.05,
            text_align=TextNode.A_left,
            pos=(-1.7, 0, 0.76),
            frameColor=(0, 0, 0, 0),
            parent=a2d,
        )

        # Underworlder count
        self.underworld_label = DirectLabel(
            text="Underworlders: 0",
            text_fg=(0.6, 1, 0.6, 1),
            text_scale=0.05,
            text_align=TextNode.A_left,
            pos=(-1.7, 0, 0.69),
            frameColor=(0, 0, 0, 0),
            parent=a2d,
        )

        # Dungeon reputation
        self.reputation_label = DirectLabel(
            text="Reputation: Unknown",
            text_fg=(0.6, 0.6, 0.6, 1),
            text_scale=0.05,
            text_align=TextNode.A_left,
            pos=(-1.7, 0, 0.62),
            frameColor=(0, 0, 0, 0),
            parent=a2d,
        )

        # Party / archetype breakdown
        self.party_label = DirectLabel(
            text="",
            text_fg=(0.8, 0.8, 0.8, 1),
            text_scale=0.04,
            text_align=TextNode.A_left,
            pos=(-1.2, 0, 0.55),
            frameColor=(0, 0, 0, 0),
            parent=a2d,
        )

        # ── Bottom bar: Tool + Hand ──
        self.bottom_bar = DirectFrame(
            frameColor=(0, 0, 0, 0.7),
            frameSize=(-2.0, 2.0, -0.07, 0.07),
            pos=(0, 0, -0.93),
            parent=a2d,
        )

        # Tool indicator
        self.tool_label = DirectLabel(
            text="Tool: Dig [X]",
            text_fg=(0.8, 0.8, 1, 1),
            text_scale=0.05,
            text_align=TextNode.A_left,
            pos=(-1.7, 0, -0.95),
            frameColor=(0, 0, 0, 0),
            parent=a2d,
        )

        # Hand contents
        self.hand_label = DirectLabel(
            text="Hand: Empty",
            text_fg=(0.9, 0.9, 0.7, 1),
            text_scale=0.05,
            text_align=TextNode.A_left,
            pos=(-0.5, 0, -0.95),
            frameColor=(0, 0, 0, 0),
            parent=a2d,
        )

        # Hover info (shows voxel type under cursor)
        self.hover_label = DirectLabel(
            text="",
            text_fg=(0.7, 0.85, 1, 1),
            text_scale=0.045,
            text_align=TextNode.A_left,
            pos=(0.2, 0, -0.95),
            frameColor=(0, 0, 0, 0),
            parent=a2d,
        )

        # Book of Crafting button
        self.book_btn = DirectButton(
            text="Book [B]",
            text_scale=0.04,
            text_fg=(0.9, 0.8, 0.4, 1),
            frameSize=(-0.12, 0.12, -0.03, 0.04),
            frameColor=(0.15, 0.15, 0.2, 0.8),
            pos=(0.9, 0, -0.95),
            command=lambda: event_bus.publish("toggle_crafting_book"),
            parent=a2d,
        )

        # Debug: spawn buttons (temporary, for testing)
        self.spawn_surface_btn = DirectButton(
            text="Spawn Surface",
            text_scale=0.035,
            text_fg=(1, 0.6, 0.6, 1),
            frameSize=(-0.16, 0.16, -0.03, 0.04),
            frameColor=(0.2, 0.1, 0.1, 0.8),
            pos=(1.2, 0, -0.95),
            command=lambda: event_bus.publish("debug_spawn_party"),
            parent=a2d,
        )
        self.spawn_underworld_btn = DirectButton(
            text="Spawn Underworld",
            text_scale=0.035,
            text_fg=(0.6, 1, 0.6, 1),
            frameSize=(-0.18, 0.18, -0.03, 0.04),
            frameColor=(0.1, 0.2, 0.1, 0.8),
            pos=(1.6, 0, -0.95),
            command=lambda: event_bus.publish("debug_spawn_underworld_party"),
            parent=a2d,
        )

        # Error message (center, red, fades)
        self.error_label = DirectLabel(
            text="",
            text_fg=(1, 0.3, 0.3, 1),
            text_scale=0.06,
            text_align=TextNode.A_center,
            pos=(0, 0, -0.75),
            frameColor=(0, 0, 0, 0),
            parent=a2d,
        )
        self._error_task_name = "hud_error_fade"

        # Game over overlay (hidden initially)
        self.game_over_frame = DirectFrame(
            frameColor=(0, 0, 0, 0.8),
            frameSize=(-0.8, 0.8, -0.15, 0.15),
            pos=(0, 0, 0),
            parent=a2d,
        )
        self.game_over_label = DirectLabel(
            text="GAME OVER",
            text_fg=(1, 0, 0, 1),
            text_scale=0.12,
            pos=(0, 0, -0.04),
            frameColor=(0, 0, 0, 0),
            parent=self.game_over_frame,
        )
        self.game_over_frame.hide()

        # Intruder tracking
        self._intruder_count = 0
        self._underworld_count = 0
        self._archetype_counts: dict[str, int] = {}  # name → alive count

        # Subscribe to events
        event_bus.subscribe("core_damaged", self._on_core_damaged)
        event_bus.subscribe("tick", self._on_tick)
        event_bus.subscribe("speed_changed", self._on_speed_changed)
        event_bus.subscribe("z_level_changed", self._on_z_changed)
        event_bus.subscribe("game_over", self._on_game_over)
        event_bus.subscribe("intruder_spawned", self._on_intruder_spawned)
        event_bus.subscribe("intruder_died", self._on_intruder_removed)
        event_bus.subscribe("intruder_escaped", self._on_intruder_removed)
        event_bus.subscribe("tool_changed", self._on_tool_changed)
        event_bus.subscribe("material_picked_up", self._on_material_picked_up)
        event_bus.subscribe("material_dropped", self._on_material_dropped)
        event_bus.subscribe("craft_success", self._on_craft_success)
        event_bus.subscribe("recipe_discovered", self._on_recipe_discovered)
        event_bus.subscribe("error_message", self._on_error_message)
        event_bus.subscribe("reputation_changed", self._on_reputation_changed)
        event_bus.subscribe("voxel_hover", self._on_voxel_hover)
        event_bus.subscribe("voxel_hover_clear", self._on_voxel_hover_clear)
        event_bus.subscribe("craft_mode_entered", self._on_craft_mode_entered)
        event_bus.subscribe("craft_mode_exited", self._on_craft_mode_exited)

        # Keyboard shortcuts
        app.accept("space", self._toggle_pause)
        app.accept("+", self._speed_up)
        app.accept("=", self._speed_up)  # Unshifted + key
        app.accept("-", self._speed_down)

    def _set_speed(self, speed: int) -> None:
        self.game_state.time_manager.set_speed(speed)

    def _toggle_pause(self) -> None:
        tm = self.game_state.time_manager
        if tm.paused:
            tm.set_speed(1)
        else:
            tm.set_speed(0)

    def _speed_up(self) -> None:
        tm = self.game_state.time_manager
        new_speed = min(2, tm.speed + 1)
        tm.set_speed(new_speed)

    def _speed_down(self) -> None:
        tm = self.game_state.time_manager
        new_speed = max(0, tm.speed - 1)
        tm.set_speed(new_speed)

    def _on_core_damaged(self, hp: int, max_hp: int) -> None:
        self.core_hp_label["text"] = f"Core: {hp}/{max_hp}"

    def _on_tick(self, tick: int) -> None:
        # Update every 10 ticks to reduce overhead
        if tick % 10 == 0:
            self.time_label["text"] = f"Tick: {tick}"
            # Refresh hand display (in case held material changed)
            self._refresh_hand()

    def _on_speed_changed(self, speed: int, **kwargs) -> None:
        labels = {0: "PAUSED", 1: "PLAYING", 2: "FAST >>"}
        self.speed_label["text"] = labels.get(speed, "???")
        colors = {0: (1, 1, 0, 1), 1: (0, 1, 0, 1), 2: (1, 0.5, 0, 1)}
        self.speed_label["text_fg"] = colors.get(speed, (1, 1, 1, 1))

    def _on_z_changed(self, z: int) -> None:
        self.z_label["text"] = f"Z: {-z}"

    def _on_game_over(self, **kwargs) -> None:
        self.game_over_frame.show()
        self.game_state.game_over = True

    def _on_intruder_spawned(self, intruder=None, **kwargs) -> None:
        is_uw = intruder is not None and getattr(intruder, "is_underworlder", False)
        if is_uw:
            self._underworld_count += 1
            self.underworld_label["text"] = f"Underworlders: {self._underworld_count}"
        else:
            self._intruder_count += 1
            self.intruder_label["text"] = f"Intruders: {self._intruder_count}"
        if intruder is not None:
            name = intruder.archetype.name
            self._archetype_counts[name] = self._archetype_counts.get(name, 0) + 1
            self._refresh_party_label()

    def _on_intruder_removed(self, intruder=None, **kwargs) -> None:
        is_uw = intruder is not None and getattr(intruder, "is_underworlder", False)
        if is_uw:
            self._underworld_count = max(0, self._underworld_count - 1)
            self.underworld_label["text"] = f"Underworlders: {self._underworld_count}"
        else:
            self._intruder_count = max(0, self._intruder_count - 1)
            self.intruder_label["text"] = f"Intruders: {self._intruder_count}"
        if intruder is not None:
            name = intruder.archetype.name
            self._archetype_counts[name] = max(
                0, self._archetype_counts.get(name, 1) - 1
            )
            # Remove zero-count entries
            if self._archetype_counts.get(name, 0) == 0:
                self._archetype_counts.pop(name, None)
            self._refresh_party_label()

    def _refresh_party_label(self) -> None:
        """Update the archetype breakdown display."""
        if not self._archetype_counts:
            self.party_label["text"] = ""
            return
        parts = []
        for name in sorted(self._archetype_counts):
            count = self._archetype_counts[name]
            if count > 0:
                parts.append(f"{name}: {count}")
        self.party_label["text"] = " | ".join(parts)

    def _on_tool_changed(self, mode: str) -> None:
        label = "Dig" if mode == "dig" else "Move"
        self.tool_label["text"] = f"Tool: {label} [X]"

    def _on_material_picked_up(self, **kwargs) -> None:
        self._refresh_hand()

    def _on_material_dropped(self, **kwargs) -> None:
        self._refresh_hand()

    def _on_craft_success(self, recipe: str, **kwargs) -> None:
        self._refresh_hand()
        self._show_error(f"Crafted: {recipe}", color=(0.3, 1, 0.3, 1))

    def _on_recipe_discovered(self, recipe: str, total: int, **kwargs) -> None:
        self._show_error(f"Recipe discovered: {recipe}!", color=(0.4, 0.9, 0.4, 1))

    def _refresh_hand(self) -> None:
        ms = self.game_state.move_system
        if ms is None or not ms.held_materials:
            self.hand_label["text"] = "Hand: Empty"
        else:
            parts = []
            for vtype, count in sorted(ms.held_materials.items()):
                name = _VTYPE_NAMES.get(vtype, f"Type {vtype}")
                parts.append(f"{name} x{count}")
            self.hand_label["text"] = "Hand: " + ", ".join(parts)

    def _on_error_message(self, text: str) -> None:
        self._show_error(text)

    def _show_error(self, text: str, color: tuple = (1, 0.3, 0.3, 1)) -> None:
        self.error_label["text"] = text
        self.error_label["text_fg"] = color

        # Cancel any existing fade task
        self.app.taskMgr.remove(self._error_task_name)

        # Schedule fade after 2 seconds
        def clear_error(task: Task):
            self.error_label["text"] = ""
            return task.done

        self.app.taskMgr.doMethodLater(
            2.0, clear_error, self._error_task_name
        )

    def _on_reputation_changed(
        self, lethality: float = 0.5, richness: float = 0.0, **kwargs,
    ) -> None:
        """Update the reputation label based on dungeon profile."""
        if lethality > 0.7 and richness < 0.4:
            self.reputation_label["text"] = "Reputation: Deadly"
            self.reputation_label["text_fg"] = (1, 0.2, 0.2, 1)
        elif richness > 0.4:
            self.reputation_label["text"] = "Reputation: Treasure Hoard"
            self.reputation_label["text_fg"] = (1, 0.85, 0.2, 1)
        else:
            self.reputation_label["text"] = "Reputation: Moderate"
            self.reputation_label["text_fg"] = (0.6, 0.6, 0.6, 1)

    def _on_voxel_hover(self, x: int, y: int, z: int, **kwargs) -> None:
        """Display voxel type and coordinates when hovering."""
        grid = self.game_state.voxel_grid
        if grid is None:
            return
        vtype = grid.get(x, y, z)
        name = _VTYPE_NAMES.get(vtype, f"Type {vtype}")
        self.hover_label["text"] = f"[{name}] ({x}, {y}, {-z})"

    def _on_voxel_hover_clear(self, **kwargs) -> None:
        """Clear hover display when cursor leaves voxels."""
        self.hover_label["text"] = ""

    def _on_craft_mode_entered(self, recipe_name: str, **kwargs) -> None:
        """Show craft mode indicator in the tool label area."""
        self.tool_label["text"] = f"Crafting: {recipe_name} (click to place, ESC to cancel)"
        self.tool_label["text_fg"] = (0.3, 1.0, 0.3, 1)

    def _on_craft_mode_exited(self, **kwargs) -> None:
        """Restore normal tool display."""
        mode = self.game_state.build_mode
        label = "Dig" if mode == "dig" else "Move"
        self.tool_label["text"] = f"Tool: {label} [X]"
        self.tool_label["text_fg"] = (0.8, 0.8, 1, 1)
