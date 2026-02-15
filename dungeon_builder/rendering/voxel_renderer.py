"""Chunk-based voxel mesh generation and rendering using Panda3D GeomNode."""

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
    LVector4f,
)

from dungeon_builder.config import (
    CHUNK_SIZE,
    VOXEL_AIR,
    VOXEL_COLORS,
    GRID_HEIGHT,
    RENDER_MODE_MATTER,
    RENDER_MODE_HUMIDITY,
    RENDER_MODE_HEAT,
)

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.rendering.layer_slice import LayerSliceManager
    from dungeon_builder.world.voxel_grid import VoxelGrid

logger = logging.getLogger("dungeon_builder.rendering")

# Vertex format: position + normal + color
VOXEL_VERTEX_FORMAT = GeomVertexFormat.get_v3n3c4()

# Six face directions: (dx, dy, dz, face_name)
FACES = [
    (0, 0, -1, "top"),     # Toward surface (z decreasing = up in world)
    (0, 0, 1, "bottom"),   # Toward depth (z increasing = down in world)
    (1, 0, 0, "east"),
    (-1, 0, 0, "west"),
    (0, 1, 0, "north"),
    (0, -1, 0, "south"),
]

# Normal vectors for each face (in world space, where Z is up)
FACE_NORMALS = {
    "top": (0, 0, 1),
    "bottom": (0, 0, -1),
    "east": (1, 0, 0),
    "west": (-1, 0, 0),
    "north": (0, 1, 0),
    "south": (0, -1, 0),
}

# Quad vertices for each face (offsets within a unit cube)
# The cube goes from (0,0,0) to (1,1,1) in local voxel space
# World mapping: voxel at grid (gx, gy, gz) renders at world (gx, gy, -gz)
# So the "top" face (toward surface) is at higher world-Z
FACE_VERTICES = {
    "top": [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
    "bottom": [(0, 1, 0), (1, 1, 0), (1, 0, 0), (0, 0, 0)],
    "east": [(1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)],
    "west": [(0, 1, 0), (0, 1, 1), (0, 0, 1), (0, 0, 0)],
    "north": [(1, 1, 0), (1, 1, 1), (0, 1, 1), (0, 1, 0)],
    "south": [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)],
}

# Slight color variation per face for pseudo-lighting
FACE_SHADE = {
    "top": 1.0,
    "bottom": 0.5,
    "east": 0.8,
    "west": 0.7,
    "north": 0.9,
    "south": 0.6,
}


def humidity_to_color(h: float) -> tuple[float, float, float, float]:
    """Map humidity [0..1] to a blue-cyan-green gradient."""
    h = max(0.0, min(1.0, h))
    if h < 0.5:
        t = h / 0.5
        # Blue (0,0,1) -> Cyan (0,1,1)
        return (0.0, t, 1.0, 1.0)
    else:
        t = (h - 0.5) / 0.5
        # Cyan (0,1,1) -> Green (0,1,0)
        return (0.0, 1.0, 1.0 - t, 1.0)


def temperature_to_color(temp: float) -> tuple[float, float, float, float]:
    """Map temperature to blue-white-red gradient.

    0-20: deep blue
    20-100: blue to white
    100-500: white to red
    500+: bright red
    """
    if temp <= 20.0:
        return (0.0, 0.0, 0.5, 1.0)
    elif temp <= 100.0:
        t = (temp - 20.0) / 80.0
        # Blue to white
        return (t, t, 0.5 + 0.5 * t, 1.0)
    elif temp <= 500.0:
        t = (temp - 100.0) / 400.0
        # White to red
        return (1.0, 1.0 - t, 1.0 - t, 1.0)
    else:
        return (1.0, 0.0, 0.0, 1.0)


class ChunkMeshBuilder:
    """Builds a GeomNode for one 16x16x1 chunk."""

    def build(
        self,
        voxel_grid: VoxelGrid,
        chunk_x: int,
        chunk_y: int,
        z_level: int,
        render_mode: str = RENDER_MODE_MATTER,
    ) -> GeomNode | None:
        """Build mesh for chunk at (chunk_x, chunk_y, z_level).

        Returns GeomNode or None if chunk has no visible faces.
        """
        vdata = GeomVertexData(
            f"chunk_{chunk_x}_{chunk_y}_{z_level}",
            VOXEL_VERTEX_FORMAT,
            Geom.UH_static,
        )
        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        color = GeomVertexWriter(vdata, "color")
        prim = GeomTriangles(Geom.UH_static)

        vertex_count = 0
        x_off = chunk_x * CHUNK_SIZE
        y_off = chunk_y * CHUNK_SIZE

        for lx in range(CHUNK_SIZE):
            gx = x_off + lx
            if gx >= voxel_grid.width:
                continue
            for ly in range(CHUNK_SIZE):
                gy = y_off + ly
                if gy >= voxel_grid.depth:
                    continue

                vtype = voxel_grid.get(gx, gy, z_level)
                if vtype == VOXEL_AIR:
                    continue

                # Choose color based on render mode
                base_color = self._get_color(
                    voxel_grid, gx, gy, z_level, vtype, render_mode
                )

                for dx, dy, dz, face_name in FACES:
                    nx, ny, nz = gx + dx, gy + dy, z_level + dz
                    # Only render face if neighbor is air or out-of-bounds-above
                    neighbor = voxel_grid.get(nx, ny, nz)
                    if neighbor != VOXEL_AIR:
                        continue

                    shade = FACE_SHADE[face_name]
                    fc = (
                        base_color[0] * shade,
                        base_color[1] * shade,
                        base_color[2] * shade,
                        base_color[3],
                    )
                    nrm = FACE_NORMALS[face_name]
                    offsets = FACE_VERTICES[face_name]
                    base_idx = vertex_count

                    for ox, oy, oz in offsets:
                        # World position: gx+ox, gy+oy, -(z_level) + oz
                        vertex.add_data3f(gx + ox, gy + oy, -z_level + oz)
                        normal.add_data3f(*nrm)
                        color.add_data4f(*fc)
                        vertex_count += 1

                    # Two triangles for the quad
                    prim.add_vertices(base_idx, base_idx + 1, base_idx + 2)
                    prim.add_vertices(base_idx, base_idx + 2, base_idx + 3)

        if vertex_count == 0:
            return None

        geom = Geom(vdata)
        geom.add_primitive(prim)
        node = GeomNode(f"chunk_{chunk_x}_{chunk_y}_{z_level}")
        node.add_geom(geom)
        return node

    def _get_color(
        self,
        voxel_grid: VoxelGrid,
        x: int,
        y: int,
        z: int,
        vtype: int,
        render_mode: str,
    ) -> tuple[float, float, float, float]:
        """Get the color for a voxel based on render mode."""
        if render_mode == RENDER_MODE_HUMIDITY:
            h = voxel_grid.get_humidity(x, y, z)
            return humidity_to_color(h)

        if render_mode == RENDER_MODE_HEAT:
            temp = voxel_grid.get_temperature(x, y, z)
            return temperature_to_color(temp)

        # Matter mode (default)
        base = VOXEL_COLORS.get(vtype, (1.0, 0.0, 1.0, 1.0))
        # Dim loose material slightly
        if voxel_grid.is_loose(x, y, z):
            return (base[0] * 0.7, base[1] * 0.7, base[2] * 0.7, base[3])
        return base


class VoxelWorldRenderer:
    """Manages all chunk meshes and updates them when voxels change."""

    def __init__(
        self,
        event_bus: EventBus,
        voxel_grid: VoxelGrid,
        layer_manager: LayerSliceManager,
    ) -> None:
        self.event_bus = event_bus
        self.voxel_grid = voxel_grid
        self.layer_manager = layer_manager
        self.mesh_builder = ChunkMeshBuilder()
        self.render_mode: str = RENDER_MODE_MATTER

        # chunk_key (cx, cy, z) -> NodePath
        self._chunk_nodes: dict[tuple[int, int, int], NodePath] = {}

        event_bus.subscribe("voxel_changed", self._on_voxel_changed)

    def set_render_mode(self, mode: str) -> None:
        """Switch render mode and rebuild all chunks."""
        if mode == self.render_mode:
            return
        self.render_mode = mode
        self.voxel_grid.mark_all_dirty()
        self.update_dirty_chunks()

    def build_all_chunks(self) -> None:
        """Build meshes for the entire world. Called once at startup."""
        logger.info("Building all chunk meshes...")
        count = 0
        for z in range(self.voxel_grid.height):
            for cx in range(self.voxel_grid.chunks_x):
                for cy in range(self.voxel_grid.chunks_y):
                    self._rebuild_chunk(cx, cy, z)
                    count += 1
        logger.info("Built %d chunks", count)

    def update_dirty_chunks(self) -> None:
        """Rebuild any chunks that were marked dirty."""
        dirty = self.voxel_grid.pop_dirty_chunks()
        for cx, cy, z in dirty:
            self._rebuild_chunk(cx, cy, z)

    def _rebuild_chunk(self, cx: int, cy: int, z: int) -> None:
        key = (cx, cy, z)

        # Remove old mesh if it exists
        old = self._chunk_nodes.pop(key, None)
        if old is not None:
            old.remove_node()

        # Build new mesh
        geom_node = self.mesh_builder.build(
            self.voxel_grid, cx, cy, z, render_mode=self.render_mode
        )
        if geom_node is None:
            return

        # Attach to the appropriate layer
        layer_np = self.layer_manager.get_layer(z)
        if layer_np is None:
            return

        np = layer_np.attach_new_node(geom_node)
        self._chunk_nodes[key] = np

    def _on_voxel_changed(self, x: int, y: int, z: int, **kwargs) -> None:
        """Schedule chunk rebuilds when a voxel changes."""
        # The VoxelGrid already marks dirty chunks, so we just need to
        # rebuild on next update. We could do it immediately, but batching
        # via update_dirty_chunks is more efficient during bulk operations.
        self.update_dirty_chunks()
