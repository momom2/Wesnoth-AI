"""Intruder visual rendering using simple procedural cubes."""

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
    from dungeon_builder.intruders.agent import Intruder

logger = logging.getLogger("dungeon_builder.rendering.intruders")


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


class IntruderRenderer:
    """Renders intruders as colored cubes, updates positions via events."""

    def __init__(self, app: ShowBase, event_bus: EventBus) -> None:
        self.app = app
        self.event_bus = event_bus
        self._models: dict[int, NodePath] = {}

        event_bus.subscribe("intruder_spawned", self._on_spawn)
        event_bus.subscribe("intruder_moved", self._on_moved)
        event_bus.subscribe("intruder_died", self._on_died)
        event_bus.subscribe("intruder_escaped", self._on_escaped)

    def _on_spawn(self, intruder: Intruder) -> None:
        node = _make_cube_geom(1.0, 0.2, 0.2, size=0.6)
        np = self.app.render.attach_new_node(node)
        np.set_pos(intruder.x + 0.5, intruder.y + 0.5, -intruder.z + 0.5)
        self._models[intruder.id] = np

    def _on_moved(self, intruder: Intruder, **kwargs) -> None:
        np = self._models.get(intruder.id)
        if np:
            np.set_pos(intruder.x + 0.5, intruder.y + 0.5, -intruder.z + 0.5)

    def _on_died(self, intruder: Intruder, **kwargs) -> None:
        np = self._models.pop(intruder.id, None)
        if np:
            np.remove_node()

    def _on_escaped(self, intruder: Intruder, **kwargs) -> None:
        np = self._models.pop(intruder.id, None)
        if np:
            np.remove_node()
