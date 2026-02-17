"""Visual effects for the dungeon core, hover highlight, craft markers, and dig indicators."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from panda3d.core import (
    Geom,
    GeomLines,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
    TransparencyAttrib,
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


def _make_highlight_cube() -> GeomNode:
    """Create a wireframe cube for hover highlighting.

    The cube spans from (-m, -m, -m) to (1+m, 1+m, 1+m), slightly larger
    than a unit voxel so it renders outside the block faces.
    """
    m = 0.02  # margin outside the voxel
    vdata = GeomVertexData("highlight", GeomVertexFormat.get_v3c4(), Geom.UH_static)
    vertex = GeomVertexWriter(vdata, "vertex")
    color = GeomVertexWriter(vdata, "color")
    prim = GeomLines(Geom.UH_static)

    # 8 corners of the cube
    corners = [
        (-m, -m, -m),          # 0: bottom-SW
        (1 + m, -m, -m),       # 1: bottom-SE
        (1 + m, 1 + m, -m),    # 2: bottom-NE
        (-m, 1 + m, -m),       # 3: bottom-NW
        (-m, -m, 1 + m),       # 4: top-SW
        (1 + m, -m, 1 + m),    # 5: top-SE
        (1 + m, 1 + m, 1 + m), # 6: top-NE
        (-m, 1 + m, 1 + m),    # 7: top-NW
    ]

    r, g, b, a = 1.0, 1.0, 1.0, 0.85

    for cx, cy, cz in corners:
        vertex.add_data3f(cx, cy, cz)
        color.add_data4f(r, g, b, a)

    # 12 edges of a cube
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
    ]
    for v0, v1 in edges:
        prim.add_vertices(v0, v1)

    geom = Geom(vdata)
    geom.add_primitive(prim)
    node = GeomNode("highlight_cube")
    node.add_geom(geom)
    return node


def _make_craft_marker() -> GeomNode:
    """Create a semi-transparent green cube for craft-valid air positions.

    Uses the same geometry as the core marker but green with alpha=0.35.
    """
    vdata = GeomVertexData("craft", GeomVertexFormat.get_v3n3c4(), Geom.UH_static)
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    color = GeomVertexWriter(vdata, "color")
    prim = GeomTriangles(Geom.UH_static)

    # Slightly smaller than a full voxel to avoid z-fighting
    m = 0.05
    r, g, b, a = 0.2, 0.9, 0.3, 0.35

    faces = [
        ((0, 0, 1), [(m, m, 1 - m), (1 - m, m, 1 - m), (1 - m, 1 - m, 1 - m), (m, 1 - m, 1 - m)]),
        ((0, 0, -1), [(m, 1 - m, m), (1 - m, 1 - m, m), (1 - m, m, m), (m, m, m)]),
        ((1, 0, 0), [(1 - m, m, m), (1 - m, 1 - m, m), (1 - m, 1 - m, 1 - m), (1 - m, m, 1 - m)]),
        ((-1, 0, 0), [(m, 1 - m, m), (m, m, m), (m, m, 1 - m), (m, 1 - m, 1 - m)]),
        ((0, 1, 0), [(1 - m, 1 - m, m), (m, 1 - m, m), (m, 1 - m, 1 - m), (1 - m, 1 - m, 1 - m)]),
        ((0, -1, 0), [(m, m, m), (1 - m, m, m), (1 - m, m, 1 - m), (m, m, 1 - m)]),
    ]

    vi = 0
    for nrm, corners in faces:
        for cx, cy, cz in corners:
            vertex.add_data3f(cx, cy, cz)
            normal.add_data3f(*nrm)
            color.add_data4f(r, g, b, a)
        prim.add_vertices(vi, vi + 1, vi + 2)
        prim.add_vertices(vi, vi + 2, vi + 3)
        vi += 4

    geom = Geom(vdata)
    geom.add_primitive(prim)
    node = GeomNode("craft_marker")
    node.add_geom(geom)
    return node


class EffectsRenderer:
    """Renders visual indicators for the core, hover highlight, and craft markers."""

    def __init__(self, app: ShowBase, event_bus: EventBus) -> None:
        self.app = app
        self.event_bus = event_bus
        self._core_np: NodePath | None = None
        self._highlight_np: NodePath | None = None
        self._craft_marker_nps: list[NodePath] = []  # Pool of reusable markers

        # Create the reusable highlight cube (hidden initially)
        self._init_highlight()

        event_bus.subscribe("voxel_hover", self._on_voxel_hover)
        event_bus.subscribe("voxel_hover_clear", self._on_voxel_hover_clear)
        event_bus.subscribe("craft_highlights_updated", self._on_craft_highlights_updated)
        event_bus.subscribe("craft_highlights_cleared", self._on_craft_highlights_cleared)

    def _init_highlight(self) -> None:
        """Create the wireframe highlight cube, hidden by default."""
        node = _make_highlight_cube()
        np = self.app.render.attach_new_node(node)
        np.set_transparency(TransparencyAttrib.M_alpha)
        np.set_render_mode_thickness(2.0)
        np.set_light_off()   # Unaffected by scene lighting
        np.set_bin("fixed", 50)  # Render on top of voxels
        np.set_depth_test(False)
        np.set_depth_write(False)
        np.hide()
        self._highlight_np = np

    def _on_voxel_hover(self, x: int, y: int, z: int) -> None:
        """Move highlight cube to the hovered voxel."""
        if self._highlight_np is not None:
            self._highlight_np.set_pos(x, y, -z)
            self._highlight_np.show()

    def _on_voxel_hover_clear(self, **kwargs) -> None:
        """Hide highlight cube when no voxel is hovered."""
        if self._highlight_np is not None:
            self._highlight_np.hide()

    def place_core_marker(self, x: int, y: int, z: int) -> None:
        """Place a visual marker at the dungeon core position."""
        node = _make_core_marker()
        np = self.app.render.attach_new_node(node)
        np.set_pos(x, y, -z)
        self._core_np = np

    # ── Craft highlight markers ──────────────────────────────────────

    def _on_craft_highlights_updated(self, positions: set, **kwargs) -> None:
        """Show green markers at all valid craft positions."""
        # Hide all existing markers first
        for np in self._craft_marker_nps:
            np.hide()

        # Show/create markers for each position
        for i, (x, y, z) in enumerate(positions):
            if i >= len(self._craft_marker_nps):
                # Create a new marker node and add to pool
                node = _make_craft_marker()
                np = self.app.render.attach_new_node(node)
                np.set_transparency(TransparencyAttrib.M_alpha)
                np.set_light_off()
                np.set_bin("fixed", 45)
                np.set_depth_write(False)
                self._craft_marker_nps.append(np)
            marker = self._craft_marker_nps[i]
            marker.set_pos(x, y, -z)
            marker.show()

    def _on_craft_highlights_cleared(self, **kwargs) -> None:
        """Hide all craft markers."""
        for np in self._craft_marker_nps:
            np.hide()
