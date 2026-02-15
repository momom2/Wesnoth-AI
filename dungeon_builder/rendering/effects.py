"""Visual effects for the dungeon core and dig indicators."""

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

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus

logger = logging.getLogger("dungeon_builder.rendering.effects")


def _make_core_marker() -> GeomNode:
    """Create a glowing cube for the dungeon core."""
    vdata = GeomVertexData("core", GeomVertexFormat.get_v3n3c4(), Geom.UH_static)
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    color = GeomVertexWriter(vdata, "color")
    prim = GeomTriangles(Geom.UH_static)

    s = 0.35
    c = 0.5  # center offset
    r, g, b = 0.9, 0.1, 0.2

    faces = [
        ((0, 0, 1), [(c-s, c-s, c+s), (c+s, c-s, c+s), (c+s, c+s, c+s), (c-s, c+s, c+s)]),
        ((0, 0, -1), [(c-s, c+s, c-s), (c+s, c+s, c-s), (c+s, c-s, c-s), (c-s, c-s, c-s)]),
        ((1, 0, 0), [(c+s, c-s, c-s), (c+s, c+s, c-s), (c+s, c+s, c+s), (c+s, c-s, c+s)]),
        ((-1, 0, 0), [(c-s, c+s, c-s), (c-s, c-s, c-s), (c-s, c-s, c+s), (c-s, c+s, c+s)]),
        ((0, 1, 0), [(c+s, c+s, c-s), (c-s, c+s, c-s), (c-s, c+s, c+s), (c+s, c+s, c+s)]),
        ((0, -1, 0), [(c-s, c-s, c-s), (c+s, c-s, c-s), (c+s, c-s, c+s), (c-s, c-s, c+s)]),
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
    node = GeomNode("core_marker")
    node.add_geom(geom)
    return node


class EffectsRenderer:
    """Renders visual indicators for the core."""

    def __init__(self, app: ShowBase, event_bus: EventBus) -> None:
        self.app = app
        self.event_bus = event_bus
        self._core_np: NodePath | None = None

    def place_core_marker(self, x: int, y: int, z: int) -> None:
        """Place a visual marker at the dungeon core position."""
        node = _make_core_marker()
        np = self.app.render.attach_new_node(node)
        np.set_pos(x, y, -z)
        self._core_np = np
