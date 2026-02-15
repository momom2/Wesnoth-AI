"""Dropdown menu for switching between render modes (matter/humidity/heat)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from direct.gui.DirectGui import DirectOptionMenu
from direct.showbase.ShowBase import ShowBase

from dungeon_builder.config import (
    RENDER_MODE_MATTER,
    RENDER_MODE_HUMIDITY,
    RENDER_MODE_HEAT,
)

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.rendering.voxel_renderer import VoxelWorldRenderer

_MODES = [RENDER_MODE_MATTER, RENDER_MODE_HUMIDITY, RENDER_MODE_HEAT]
_LABELS = {"matter": "Matter", "humidity": "Humidity", "heat": "Heat"}


class RenderModeSelector:
    """Dropdown and keyboard shortcut (V) for switching render overlays."""

    def __init__(
        self,
        app: ShowBase,
        event_bus: EventBus,
        world_renderer: VoxelWorldRenderer,
    ) -> None:
        self.app = app
        self.event_bus = event_bus
        self.world_renderer = world_renderer
        self._current_index = 0

        self.menu = DirectOptionMenu(
            text="View",
            scale=0.06,
            items=[_LABELS[m] for m in _MODES],
            initialitem=0,
            highlightColor=(0.65, 0.65, 0.65, 1),
            command=self._on_select,
            pos=(1.1, 0, 0.9),
        )

        app.accept("v", self._cycle_mode)

    def _on_select(self, label: str) -> None:
        for mode, lbl in _LABELS.items():
            if lbl == label:
                self.world_renderer.set_render_mode(mode)
                self._current_index = _MODES.index(mode)
                self.event_bus.publish("render_mode_changed", mode=mode)
                return

    def _cycle_mode(self) -> None:
        self._current_index = (self._current_index + 1) % len(_MODES)
        mode = _MODES[self._current_index]
        self.world_renderer.set_render_mode(mode)
        self.menu.set(self._current_index)
        self.event_bus.publish("render_mode_changed", mode=mode)
