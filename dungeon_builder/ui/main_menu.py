"""Main menu overlay: title screen, options, and submenu navigation.

Appears on startup and when the player presses Escape during gameplay.
The game pauses while the menu is open.  Options submenus allow runtime
modification of game constants via sliders.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from direct.gui.DirectGui import (
    DirectFrame,
    DirectLabel,
    DirectButton,
    DirectSlider,
)
from panda3d.core import TextNode
from direct.showbase.ShowBase import ShowBase

import dungeon_builder.config as _cfg
from dungeon_builder.ui.menu_constants import (
    DIFFICULTY_SETTINGS,
    VISIBILITY_SETTINGS,
    FOG_COLOR_SETTINGS,
    capture_defaults,
    capture_fog_defaults,
)

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.core.game_state import GameState

logger = logging.getLogger("dungeon_builder.ui")

# ── Style constants (matching existing HUD/CraftingBookPanel) ────────────
_BG_COLOR = (0.05, 0.05, 0.1, 0.92)
_TITLE_COLOR = (0.9, 0.8, 0.4, 1)
_TEXT_COLOR = (0.9, 0.9, 0.9, 1)
_MUTED_COLOR = (0.5, 0.5, 0.55, 1)
_BUTTON_FG = (0.9, 0.9, 0.9, 1)
_BUTTON_BG = (0.15, 0.15, 0.2, 0.9)
_BUTTON_SIZE = (-0.3, 0.3, -0.045, 0.055)
_SORT_ORDER = 100  # Above everything else

# Menu states
_STATES = (
    "main", "options", "game_constants",
    "difficulty", "visibility",
    "keybinding", "sound",
)


class MainMenu:
    """Full-screen menu overlay with title, options, and submenu navigation.

    Navigation state machine with a stack for back-navigation.
    """

    def __init__(
        self,
        app: ShowBase,
        event_bus: EventBus,
        game_state: GameState,
    ) -> None:
        self.app = app
        self.event_bus = event_bus
        self.game_state = game_state

        self._visible: bool = True
        self._first_play: bool = True
        self._speed_before_menu: int = 0
        self._state: str = "main"
        self._nav_stack: list[str] = []

        # Snapshot defaults for reset buttons
        self._defaults_difficulty = capture_defaults(DIFFICULTY_SETTINGS)
        self._defaults_visibility = capture_defaults(VISIBILITY_SETTINGS)
        self._fog_defaults = capture_fog_defaults()

        # Slider widgets for refreshing on reset
        self._sliders: dict[str, tuple[DirectSlider, DirectLabel]] = {}
        self._fog_sliders: dict[int, tuple[DirectSlider, DirectLabel]] = {}

        # Build the overlay and all frames
        a2d = app.aspect2d
        self._overlay = DirectFrame(
            frameColor=(0, 0, 0, 0.85),
            frameSize=(-2.5, 2.5, -1.5, 1.5),
            pos=(0, 0, 0),
            parent=a2d,
            sortOrder=_SORT_ORDER,
        )

        self._frames: dict[str, DirectFrame] = {}
        self._build_main_frame()
        self._build_options_frame()
        self._build_game_constants_frame()
        self._build_difficulty_frame()
        self._build_visibility_frame()
        self._build_keybinding_frame()
        self._build_sound_frame()

        # Show main frame initially
        self._set_state("main")

        # Bind Escape to toggle
        app.accept("escape", self.toggle)

    # ── Navigation ───────────────────────────────────────────────────

    def show(self) -> None:
        """Show the menu overlay and pause the game."""
        if self._visible:
            return
        self._visible = True
        self.game_state.menu_open = True
        # Save current speed and pause
        if self.game_state.time_manager is not None:
            self._speed_before_menu = self.game_state.time_manager.speed
            if self._speed_before_menu != 0:
                self.game_state.time_manager.set_speed(0)
        self._overlay.show()
        self._set_state("main")

    def hide(self) -> None:
        """Hide the menu and resume the game."""
        if not self._visible:
            return
        self._visible = False
        self.game_state.menu_open = False
        self._overlay.hide()
        self._nav_stack.clear()
        # Resume game speed
        if self.game_state.time_manager is not None:
            if self._speed_before_menu > 0:
                self.game_state.time_manager.set_speed(self._speed_before_menu)
            elif not self._first_play:
                self.game_state.time_manager.set_speed(1)

    def toggle(self) -> None:
        """Toggle the menu on/off (bound to Escape)."""
        if self._visible:
            if self._first_play:
                return  # Can't dismiss before first Play
            self.hide()
        else:
            self.show()

    def _on_play(self) -> None:
        self._first_play = False
        self._speed_before_menu = 1
        self.hide()

    def _on_quit(self) -> None:
        self.app.userExit()

    def _navigate_to(self, state: str) -> None:
        self._nav_stack.append(self._state)
        self._set_state(state)

    def _navigate_back(self) -> None:
        if self._nav_stack:
            prev = self._nav_stack.pop()
            self._set_state(prev)
        else:
            self.hide()

    def _set_state(self, state: str) -> None:
        for frame in self._frames.values():
            frame.hide()
        self._frames[state].show()
        self._state = state

    # ── Frame builders ───────────────────────────────────────────────

    def _make_frame(self, name: str) -> DirectFrame:
        """Create a submenu content frame inside the overlay."""
        frame = DirectFrame(
            frameColor=(0, 0, 0, 0),
            frameSize=(-1.0, 1.0, -0.9, 0.9),
            pos=(0, 0, 0),
            parent=self._overlay,
        )
        frame.hide()
        self._frames[name] = frame
        return frame

    def _make_title(self, parent: DirectFrame, text: str) -> DirectLabel:
        return DirectLabel(
            text=text,
            text_fg=_TITLE_COLOR,
            text_scale=0.1,
            text_align=TextNode.A_center,
            pos=(0, 0, 0.7),
            frameColor=(0, 0, 0, 0),
            parent=parent,
        )

    def _make_button(
        self, parent: DirectFrame, text: str, y: float, command,
    ) -> DirectButton:
        return DirectButton(
            text=text,
            text_fg=_BUTTON_FG,
            text_scale=0.06,
            frameColor=_BUTTON_BG,
            frameSize=_BUTTON_SIZE,
            relief=1,
            pos=(0, 0, y),
            command=command,
            parent=parent,
        )

    def _make_back_button(self, parent: DirectFrame, y: float = -0.8) -> DirectButton:
        return self._make_button(parent, "Back", y, self._navigate_back)

    # ── Main menu ────────────────────────────────────────────────────

    def _build_main_frame(self) -> None:
        f = self._make_frame("main")
        self._make_title(f, "DUNGEON BUILDER")
        self._make_button(f, "Play", 0.2, self._on_play)
        self._make_button(f, "Options", 0.05, lambda: self._navigate_to("options"))
        self._make_button(f, "Quit", -0.1, self._on_quit)

    # ── Options menu ─────────────────────────────────────────────────

    def _build_options_frame(self) -> None:
        f = self._make_frame("options")
        self._make_title(f, "OPTIONS")
        self._make_button(f, "Keybinding", 0.3, lambda: self._navigate_to("keybinding"))
        self._make_button(f, "Game Constants", 0.15, lambda: self._navigate_to("game_constants"))
        self._make_button(f, "Sound", 0.0, lambda: self._navigate_to("sound"))
        self._make_back_button(f)

    # ── Game constants hub ───────────────────────────────────────────

    def _build_game_constants_frame(self) -> None:
        f = self._make_frame("game_constants")
        self._make_title(f, "GAME CONSTANTS")
        self._make_button(f, "Difficulty", 0.2, lambda: self._navigate_to("difficulty"))
        self._make_button(f, "Visibility", 0.05, lambda: self._navigate_to("visibility"))
        self._make_back_button(f)

    # ── Difficulty sliders ───────────────────────────────────────────

    def _build_difficulty_frame(self) -> None:
        f = self._make_frame("difficulty")
        self._make_title(f, "DIFFICULTY")
        y_start = 0.5
        for i, setting in enumerate(DIFFICULTY_SETTINGS):
            slider, label = self._build_slider_row(f, i, setting, y_start)
            self._sliders[setting["attr"]] = (slider, label)
        btn_y = y_start - len(DIFFICULTY_SETTINGS) * 0.09 - 0.05
        self._make_button(
            f, "Reset Defaults", btn_y,
            lambda: self._reset_defaults(
                DIFFICULTY_SETTINGS, self._defaults_difficulty,
            ),
        )
        self._make_back_button(f, min(btn_y - 0.15, -0.8))

    # ── Visibility sliders ───────────────────────────────────────────

    def _build_visibility_frame(self) -> None:
        f = self._make_frame("visibility")
        self._make_title(f, "VISIBILITY")
        y_start = 0.5
        idx = 0
        for setting in VISIBILITY_SETTINGS:
            slider, label = self._build_slider_row(f, idx, setting, y_start)
            self._sliders[setting["attr"]] = (slider, label)
            idx += 1
        # Fog colour sliders
        for fog_setting in FOG_COLOR_SETTINGS:
            slider, label = self._build_fog_slider(f, idx, fog_setting, y_start)
            self._fog_sliders[fog_setting["index"]] = (slider, label)
            idx += 1
        btn_y = y_start - idx * 0.09 - 0.05
        self._make_button(
            f, "Reset Defaults", btn_y,
            lambda: self._reset_visibility_defaults(),
        )
        self._make_back_button(f, min(btn_y - 0.15, -0.8))

    # ── Stub menus ───────────────────────────────────────────────────

    def _build_keybinding_frame(self) -> None:
        f = self._make_frame("keybinding")
        self._make_title(f, "KEYBINDING")
        DirectLabel(
            text="Coming soon",
            text_fg=_MUTED_COLOR,
            text_scale=0.06,
            text_align=TextNode.A_center,
            pos=(0, 0, 0.1),
            frameColor=(0, 0, 0, 0),
            parent=f,
        )
        self._make_back_button(f)

    def _build_sound_frame(self) -> None:
        f = self._make_frame("sound")
        self._make_title(f, "SOUND")
        DirectLabel(
            text="Coming soon",
            text_fg=_MUTED_COLOR,
            text_scale=0.06,
            text_align=TextNode.A_center,
            pos=(0, 0, 0.1),
            frameColor=(0, 0, 0, 0),
            parent=f,
        )
        self._make_back_button(f)

    # ── Slider helpers ───────────────────────────────────────────────

    def _build_slider_row(
        self,
        parent: DirectFrame,
        index: int,
        setting: dict,
        y_start: float,
    ) -> tuple[DirectSlider, DirectLabel]:
        """Build one slider row: label + slider + value readout."""
        y = y_start - index * 0.09
        attr = setting["attr"]
        current_val = getattr(_cfg, attr)
        fmt = ".2f" if setting["type"] is float else "d"

        # Label
        DirectLabel(
            text=setting["label"],
            text_fg=_TEXT_COLOR,
            text_scale=0.04,
            text_align=TextNode.A_left,
            pos=(-0.55, 0, y),
            frameColor=(0, 0, 0, 0),
            parent=parent,
        )

        # Slider
        slider = DirectSlider(
            range=(setting["min"], setting["max"]),
            value=current_val,
            pageSize=setting["step"],
            scale=0.35,
            pos=(0.15, 0, y + 0.015),
            parent=parent,
        )

        # Value readout
        value_label = DirectLabel(
            text=f"{current_val:{fmt}}",
            text_fg=_TITLE_COLOR,
            text_scale=0.04,
            text_align=TextNode.A_right,
            pos=(0.65, 0, y),
            frameColor=(0, 0, 0, 0),
            parent=parent,
        )

        # Closure for callback
        def on_change(s=slider, vl=value_label, st=setting, f=fmt):
            raw = s["value"]
            step = st["step"]
            val = round(raw / step) * step
            val = st["type"](val)
            # Clamp
            val = max(st["min"], min(st["max"], val))
            setattr(_cfg, st["attr"], val)
            vl["text"] = f"{val:{f}}"
            self.event_bus.publish("config_changed", key=st["attr"], value=val)

        slider["command"] = on_change
        return slider, value_label

    def _build_fog_slider(
        self,
        parent: DirectFrame,
        index: int,
        setting: dict,
        y_start: float,
    ) -> tuple[DirectSlider, DirectLabel]:
        """Build a fog colour component slider."""
        y = y_start - index * 0.09
        fog_idx = setting["index"]
        current_val = _cfg.FOG_COLOR[fog_idx]

        DirectLabel(
            text=setting["label"],
            text_fg=_TEXT_COLOR,
            text_scale=0.04,
            text_align=TextNode.A_left,
            pos=(-0.55, 0, y),
            frameColor=(0, 0, 0, 0),
            parent=parent,
        )

        slider = DirectSlider(
            range=(setting["min"], setting["max"]),
            value=current_val,
            pageSize=setting["step"],
            scale=0.35,
            pos=(0.15, 0, y + 0.015),
            parent=parent,
        )

        value_label = DirectLabel(
            text=f"{current_val:.2f}",
            text_fg=_TITLE_COLOR,
            text_scale=0.04,
            text_align=TextNode.A_right,
            pos=(0.65, 0, y),
            frameColor=(0, 0, 0, 0),
            parent=parent,
        )

        def on_fog_change(s=slider, vl=value_label, fi=fog_idx, st=setting):
            raw = s["value"]
            step = st["step"]
            val = round(raw / step) * step
            val = max(st["min"], min(st["max"], val))
            fog = list(_cfg.FOG_COLOR)
            fog[fi] = val
            _cfg.FOG_COLOR = tuple(fog)
            vl["text"] = f"{val:.2f}"
            self.event_bus.publish("config_changed", key="FOG_COLOR", value=_cfg.FOG_COLOR)

        slider["command"] = on_fog_change
        return slider, value_label

    # ── Reset defaults ───────────────────────────────────────────────

    def _reset_defaults(
        self,
        settings: list[dict],
        defaults: dict[str, object],
    ) -> None:
        """Restore config values and refresh slider positions."""
        for setting in settings:
            attr = setting["attr"]
            default_val = defaults[attr]
            setattr(_cfg, attr, default_val)
            # Refresh slider + label
            if attr in self._sliders:
                slider, label = self._sliders[attr]
                slider["value"] = default_val
                fmt = ".2f" if setting["type"] is float else "d"
                label["text"] = f"{default_val:{fmt}}"
        self.event_bus.publish("config_changed", key="all_difficulty")

    def _reset_visibility_defaults(self) -> None:
        """Reset visibility settings AND fog colour."""
        self._reset_defaults(
            VISIBILITY_SETTINGS, self._defaults_visibility,
        )
        # Reset fog
        _cfg.FOG_COLOR = self._fog_defaults
        for fog_setting in FOG_COLOR_SETTINGS:
            fi = fog_setting["index"]
            val = self._fog_defaults[fi]
            if fi in self._fog_sliders:
                slider, label = self._fog_sliders[fi]
                slider["value"] = val
                label["text"] = f"{val:.2f}"
        self.event_bus.publish("config_changed", key="all_visibility")
