"""Main HUD overlay: core HP, tick counter, speed controls, Z-level, tool/hand info."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from direct.gui.DirectGui import DirectFrame, DirectLabel, DirectButton
from panda3d.core import TextNode
from direct.showbase.ShowBase import ShowBase
from direct.task.Task import Task

from dungeon_builder.config import VOXEL_COLORS

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
    50: "Lava",
    60: "Iron Ingot", 61: "Copper Ingot", 62: "Gold Ingot", 63: "Enchanted Metal",
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
            text="PAUSED",
            text_fg=(1, 1, 0, 1),
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
        event_bus.subscribe("error_message", self._on_error_message)

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

    def _on_intruder_spawned(self, **kwargs) -> None:
        self._intruder_count += 1
        self.intruder_label["text"] = f"Intruders: {self._intruder_count}"

    def _on_intruder_removed(self, **kwargs) -> None:
        self._intruder_count = max(0, self._intruder_count - 1)
        self.intruder_label["text"] = f"Intruders: {self._intruder_count}"

    def _on_tool_changed(self, mode: str) -> None:
        label = "Dig" if mode == "dig" else "Move"
        self.tool_label["text"] = f"Tool: {label} [X]"

    def _on_material_picked_up(self, vtype: int, count: int) -> None:
        name = _VTYPE_NAMES.get(vtype, f"Type {vtype}")
        self.hand_label["text"] = f"Hand: {name} x{count}"

    def _on_material_dropped(self, **kwargs) -> None:
        self._refresh_hand()

    def _on_craft_success(self, recipe: str, **kwargs) -> None:
        self._refresh_hand()
        self._show_error(f"Crafted: {recipe}", color=(0.3, 1, 0.3, 1))

    def _refresh_hand(self) -> None:
        ms = self.game_state.move_system
        if ms is None or ms.held_material is None:
            self.hand_label["text"] = "Hand: Empty"
        else:
            vtype, count = ms.held_material
            name = _VTYPE_NAMES.get(vtype, f"Type {vtype}")
            self.hand_label["text"] = f"Hand: {name} x{count}"

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
