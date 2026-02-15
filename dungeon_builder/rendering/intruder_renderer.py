"""Intruder visual rendering using simple procedural cubes.

Each archetype gets a distinct color so the player can identify types at a
glance.  Frenzied Goreclaws flash bright red.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
)
from direct.showbase.ShowBase import ShowBase

from dungeon_builder.config import (
    ARCHETYPE_COLORS,
    ARCHETYPE_DEFAULT_COLOR,
    ARCHETYPE_FRENZY_COLOR,
)

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.intruders.agent import Intruder

logger = logging.getLogger("dungeon_builder.rendering.intruders")


# ── Geometry helpers ──────────────────────────────────────────────────

def _make_cube_geom(r: float, g: float, b: float, size: float = 0.4) -> GeomNode:
    """Create a small colored cube GeomNode."""
    vdata = GeomVertexData("intruder", GeomVertexFormat.get_v3n3c4(), Geom.UH_static)
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    color = GeomVertexWriter(vdata, "color")
    prim = GeomTriangles(Geom.UH_static)

    s = size / 2.0
    # 6 faces, 4 verts each
    faces = [
        # (normal, 4 corners)
        ((0, 0, 1), [(-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)]),
        ((0, 0, -1), [(-s, s, -s), (s, s, -s), (s, -s, -s), (-s, -s, -s)]),
        ((1, 0, 0), [(s, -s, -s), (s, s, -s), (s, s, s), (s, -s, s)]),
        ((-1, 0, 0), [(-s, s, -s), (-s, -s, -s), (-s, -s, s), (-s, s, s)]),
        ((0, 1, 0), [(s, s, -s), (-s, s, -s), (-s, s, s), (s, s, s)]),
        ((0, -1, 0), [(-s, -s, -s), (s, -s, -s), (s, -s, s), (-s, -s, s)]),
    ]

    vi = 0
    for nrm, corners in faces:
        for cx, cy, cz in corners:
            vertex.add_data3f(cx, cy, cz)
            normal.add_data3f(*nrm)
            color.add_data4f(r, g, b, 1.0)
        prim.add_vertices(vi, vi + 1, vi + 2)
        prim.add_vertices(vi, vi + 2, vi + 3)
        vi += 4

    geom = Geom(vdata)
    geom.add_primitive(prim)
    node = GeomNode("intruder_cube")
    node.add_geom(geom)
    return node


def _archetype_color(intruder: Intruder) -> tuple[float, float, float]:
    """Return the (R, G, B) color for an intruder based on archetype."""
    return ARCHETYPE_COLORS.get(
        intruder.archetype.name, ARCHETYPE_DEFAULT_COLOR
    )


# ── Renderer ──────────────────────────────────────────────────────────

class IntruderRenderer:
    """Renders intruders as colored cubes, updates positions via events.

    Each archetype gets a unique color.  Frenzied intruders swap to the
    frenzy color.
    """

    def __init__(self, app: ShowBase, event_bus: EventBus) -> None:
        self.app = app
        self.event_bus = event_bus
        self._models: dict[int, NodePath] = {}
        self._base_colors: dict[int, tuple[float, float, float]] = {}

        event_bus.subscribe("intruder_spawned", self._on_spawn)
        event_bus.subscribe("intruder_moved", self._on_moved)
        event_bus.subscribe("intruder_died", self._on_died)
        event_bus.subscribe("intruder_escaped", self._on_escaped)

    # ── Event handlers ────────────────────────────────────────────────

    def _on_spawn(self, intruder: Intruder) -> None:
        rgb = _archetype_color(intruder)
        self._base_colors[intruder.id] = rgb
        node = _make_cube_geom(*rgb, size=0.6)
        np = self.app.render.attach_new_node(node)
        np.set_pos(intruder.x + 0.5, intruder.y + 0.5, -intruder.z + 0.5)
        self._models[intruder.id] = np

    def _on_moved(self, intruder: Intruder, **kwargs) -> None:
        np = self._models.get(intruder.id)
        if not np:
            return
        np.set_pos(intruder.x + 0.5, intruder.y + 0.5, -intruder.z + 0.5)

        # Frenzy visual: swap the model color if frenzy just changed
        if intruder.frenzy_active:
            self._swap_color(intruder.id, ARCHETYPE_FRENZY_COLOR)
        else:
            base = self._base_colors.get(intruder.id, ARCHETYPE_DEFAULT_COLOR)
            self._swap_color(intruder.id, base)

    def _on_died(self, intruder: Intruder, **kwargs) -> None:
        np = self._models.pop(intruder.id, None)
        if np:
            np.remove_node()
        self._base_colors.pop(intruder.id, None)

    def _on_escaped(self, intruder: Intruder, **kwargs) -> None:
        np = self._models.pop(intruder.id, None)
        if np:
            np.remove_node()
        self._base_colors.pop(intruder.id, None)

    # ── Internal helpers ──────────────────────────────────────────────

    def _swap_color(self, intruder_id: int, rgb: tuple[float, float, float]) -> None:
        """Replace the intruder's cube with a new color (cheap since it's
        just one small 24-vertex geom)."""
        old = self._models.get(intruder_id)
        if not old:
            return
        pos = old.get_pos()
        old.remove_node()
        node = _make_cube_geom(*rgb, size=0.6)
        np = self.app.render.attach_new_node(node)
        np.set_pos(pos)
        self._models[intruder_id] = np
