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

import dungeon_builder.config as _cfg
from dungeon_builder.config import (
    CHUNK_SIZE,
    VOXEL_AIR,
    VOXEL_COLORS,
    VOXEL_DOOR,
    VOXEL_FLOODGATE,
    VOXEL_IRON_BARS,
    VOXEL_STEAM_VENT,
    VOXEL_NOISE,
    VERTEX_NOISE_AMPLITUDE,
    GRID_HEIGHT,
    RENDER_MODE_MATTER,
    RENDER_MODE_HUMIDITY,
    RENDER_MODE_HEAT,
    RENDER_MODE_STRUCTURAL,
    METALLIC_BLOCKS,
    METAL_COLORS,
    ENCHANTED_OFFSET,
)

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.building.build_system import BuildSystem
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


def _vertex_noise(
    gx: int, gy: int, gz: int, face_idx: int, vert_idx: int,
) -> float:
    """Deterministic per-vertex color noise in [-1, +1] range.

    Uses an FNV-1a-inspired hash for speed (no imports needed).
    Same inputs always produce the same output (stable across frames).
    """
    h = 2166136261
    for v in (gx, gy, gz, face_idx, vert_idx):
        h ^= v & 0xFFFFFFFF
        h = (h * 16777619) & 0xFFFFFFFF
    return (h / 2147483647.0) - 1.0


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


def stress_to_color(stress: float) -> tuple[float, float, float, float]:
    """Map stress ratio [0..1+] to green-yellow-orange-red gradient."""
    stress = max(0.0, stress)
    if stress <= 0.5:
        t = stress / 0.5
        # Green -> Yellow
        return (0.9 * t, 0.7 + 0.2 * t, 0.0, 1.0)
    elif stress <= 0.8:
        t = (stress - 0.5) / 0.3
        # Yellow -> Orange
        return (0.9 + 0.1 * t, 0.9 - 0.4 * t, 0.0, 1.0)
    elif stress <= 1.0:
        t = (stress - 0.8) / 0.2
        # Orange -> Red
        return (1.0, 0.5 - 0.5 * t, 0.0, 1.0)
    else:
        return (1.0, 0.0, 0.0, 1.0)


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
        build_system: BuildSystem | None = None,
        craft_highlights: set | None = None,
        ingredient_highlights: set | None = None,
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
                    voxel_grid, gx, gy, z_level, vtype, render_mode,
                    build_system, craft_highlights, ingredient_highlights,
                )

                # Per-material noise amplitude
                use_noise = (
                    render_mode == RENDER_MODE_MATTER
                    and base_color != _cfg.FOG_COLOR
                )
                noise_amp = (
                    VOXEL_NOISE.get(vtype, VERTEX_NOISE_AMPLITUDE)
                    if use_noise else 0.0
                )

                for face_idx, (dx, dy, dz, face_name) in enumerate(FACES):
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

                    for vi, (ox, oy, oz) in enumerate(offsets):
                        # World position: gx+ox, gy+oy, -(z_level) + oz
                        vertex.add_data3f(gx + ox, gy + oy, -z_level + oz)
                        normal.add_data3f(*nrm)
                        if noise_amp > 0.0:
                            n = _vertex_noise(gx, gy, z_level, face_idx, vi)
                            nc = (
                                max(0.0, min(1.0, fc[0] + n * noise_amp)),
                                max(0.0, min(1.0, fc[1] + n * noise_amp * 0.8)),
                                max(0.0, min(1.0, fc[2] + n * noise_amp * 0.6)),
                                fc[3],
                            )
                            color.add_data4f(*nc)
                        else:
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
        build_system: BuildSystem | None = None,
        craft_highlights: set | None = None,
        ingredient_highlights: set | None = None,
    ) -> tuple[float, float, float, float]:
        """Get the color for a voxel based on render mode."""
        # Fog of war: blocks not adjacent to claimed territory are dark
        # Exception: pending digs show dim gold even through fog
        if not voxel_grid.is_visible(x, y, z):
            if build_system is not None and build_system.is_pending_dig(x, y, z):
                t = 0.25
                fog = _cfg.FOG_COLOR
                return (
                    fog[0] * (1 - t) + 0.8 * t,
                    fog[1] * (1 - t) + 0.65 * t,
                    fog[2] * (1 - t) + 0.0 * t,
                    fog[3],
                )
            return _cfg.FOG_COLOR

        if render_mode == RENDER_MODE_STRUCTURAL:
            s = voxel_grid.get_stress_ratio(x, y, z)
            return stress_to_color(s)

        if render_mode == RENDER_MODE_HUMIDITY:
            h = voxel_grid.get_humidity(x, y, z)
            return humidity_to_color(h)

        if render_mode == RENDER_MODE_HEAT:
            temp = voxel_grid.get_temperature(x, y, z)
            return temperature_to_color(temp)

        # Matter mode (default)
        base = VOXEL_COLORS.get(vtype, (1.0, 0.0, 1.0, 1.0))

        # Metal-variant tinting for metallic blocks
        if vtype in METALLIC_BLOCKS:
            mt = voxel_grid.get_metal_type(x, y, z)
            base_mt = mt & 0x7F  # strip enchanted bit
            if base_mt in METAL_COLORS:
                tint = METAL_COLORS[base_mt]
                # Blend 40% toward the metal color
                t = 0.4
                br = base[0] * (1 - t) + tint[0] * t
                bg = base[1] * (1 - t) + tint[1] * t
                bb = base[2] * (1 - t) + tint[2] * t
                base = (br, bg, bb, base[3])
            # Enchanted blocks get a purple shimmer
            if mt & ENCHANTED_OFFSET:
                t = 0.15
                br = base[0] * (1 - t) + 0.6 * t
                bg = base[1] * (1 - t) + 0.2 * t
                bb = base[2] * (1 - t) + 0.9 * t
                base = (br, bg, bb, base[3])

        # Golden overlay for blocks being dug
        if build_system is not None and build_system.is_being_dug(x, y, z):
            progress = build_system.get_dig_progress(x, y, z)
            # Blend from gold (queued) to brighter gold (near complete)
            # Gold: (1.0, 0.85, 0.0) — lerp from base toward gold
            t = 0.4 + 0.3 * max(0.0, progress)  # 0.4 blend at start, 0.7 near done
            r = base[0] * (1 - t) + 1.0 * t
            g = base[1] * (1 - t) + 0.85 * t
            b = base[2] * (1 - t) + 0.0 * t
            return (r, g, b, base[3])

        # Green overlay for craft-valid positions (solid blocks)
        if craft_highlights is not None and (x, y, z) in craft_highlights:
            t = 0.45
            r = base[0] * (1 - t) + 0.2 * t
            g = base[1] * (1 - t) + 1.0 * t
            b = base[2] * (1 - t) + 0.3 * t
            return (r, g, b, base[3])

        # Cyan overlay for ingredient highlights (temporary, from crafting panel)
        if ingredient_highlights is not None and (x, y, z) in ingredient_highlights:
            t = 0.5
            r = base[0] * (1 - t) + 0.3 * t
            g = base[1] * (1 - t) + 0.9 * t
            b = base[2] * (1 - t) + 1.0 * t
            return (r, g, b, base[3])

        # Open doors and open floodgates are semi-transparent
        if vtype == VOXEL_DOOR and voxel_grid.get_block_state(x, y, z) == 0:
            return (base[0], base[1], base[2], 0.3)
        if vtype == VOXEL_FLOODGATE and voxel_grid.get_block_state(x, y, z) == 0:
            return (base[0], base[1], base[2], 0.3)
        # Iron bars: semi-transparent (you can see through)
        if vtype == VOXEL_IRON_BARS:
            return (base[0], base[1], base[2], 0.7)
        # Steam vent: slight transparency
        if vtype == VOXEL_STEAM_VENT:
            return (base[0], base[1], base[2], 0.8)
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
        self.build_system: BuildSystem | None = None  # set after construction
        self.craft_highlights: set[tuple[int, int, int]] | None = None

        # chunk_key (cx, cy, z) -> NodePath
        self._chunk_nodes: dict[tuple[int, int, int], NodePath] = {}

        # Track chunks containing active digs for periodic progress refresh
        self._dig_chunks: set[tuple[int, int, int]] = set()

        # Ingredient highlight (temporary cyan tint from crafting panel)
        self._ingredient_highlights: set[tuple[int, int, int]] | None = None
        self._ingredient_clear_tick: int = 0
        self._current_tick: int = 0

        event_bus.subscribe("voxel_changed", self._on_voxel_changed)
        event_bus.subscribe("dig_queued", self._on_dig_state_changed)
        event_bus.subscribe("dig_complete", self._on_dig_state_changed)
        event_bus.subscribe("dig_cancelled", self._on_dig_state_changed)
        event_bus.subscribe("dig_pending", self._on_dig_state_changed)
        event_bus.subscribe("tick", self._on_tick_refresh_digs)
        event_bus.subscribe(
            "claimed_territory_changed",
            lambda **kw: self.update_dirty_chunks(),
        )
        event_bus.subscribe("craft_highlights_updated", self._on_craft_highlights_updated)
        event_bus.subscribe("craft_highlights_cleared", self._on_craft_highlights_cleared)
        event_bus.subscribe("ingredient_highlight", self._on_ingredient_highlight)

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
            self.voxel_grid, cx, cy, z, render_mode=self.render_mode,
            build_system=self.build_system,
            craft_highlights=self.craft_highlights,
            ingredient_highlights=self._ingredient_highlights,
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

    def _on_dig_state_changed(self, x: int, y: int, z: int, **kwargs) -> None:
        """Rebuild the chunk when a dig is queued or completed."""
        cx, cy = x // CHUNK_SIZE, y // CHUNK_SIZE
        key = (cx, cy, z)
        self._dig_chunks.discard(key)
        # Track chunk if dig is still active
        if self.build_system is not None and self.build_system.is_being_dug(x, y, z):
            self._dig_chunks.add(key)
        self._rebuild_chunk(cx, cy, z)

    def _on_tick_refresh_digs(self, tick: int, **kwargs) -> None:
        """Periodically rebuild dig chunks to show progress animation."""
        self._current_tick = tick

        # Safety: repopulate _dig_chunks if digs exist but tracking was lost
        if not self._dig_chunks and self.build_system is not None:
            for job in self.build_system.active_digs + self.build_system.dig_queue:
                cx, cy = job.x // CHUNK_SIZE, job.y // CHUNK_SIZE
                self._dig_chunks.add((cx, cy, job.z))
            # Also track pending digs (they need chunk rebuilds too)
            if hasattr(self.build_system, 'pending_digs'):
                for job in self.build_system.pending_digs:
                    cx, cy = job.x // CHUNK_SIZE, job.y // CHUNK_SIZE
                    self._dig_chunks.add((cx, cy, job.z))

        if not self._dig_chunks:
            # Clear ingredient highlights if timer expired
            if self._ingredient_highlights and tick >= self._ingredient_clear_tick:
                self._clear_ingredient_highlights()
            return
        # Refresh every 5 ticks (~4x/sec) for smooth progress updates
        if tick % 5 != 0:
            # Still check ingredient highlight timeout
            if self._ingredient_highlights and tick >= self._ingredient_clear_tick:
                self._clear_ingredient_highlights()
            return
        # Rebuild all chunks that have active digs
        stale = set()
        for key in list(self._dig_chunks):
            cx, cy, z = key
            self._rebuild_chunk(cx, cy, z)
            # Check if any digs remain in this chunk
            has_digs = False
            if self.build_system is not None:
                x_off = cx * CHUNK_SIZE
                y_off = cy * CHUNK_SIZE
                for lx in range(CHUNK_SIZE):
                    for ly in range(CHUNK_SIZE):
                        if self.build_system.is_being_dug(x_off + lx, y_off + ly, z):
                            has_digs = True
                            break
                    if has_digs:
                        break
            if not has_digs:
                stale.add(key)
        self._dig_chunks -= stale

        # Check ingredient highlight timeout
        if self._ingredient_highlights and tick >= self._ingredient_clear_tick:
            self._clear_ingredient_highlights()

    def _on_craft_highlights_updated(self, positions: set, **kwargs) -> None:
        """Store highlights and rebuild affected chunks."""
        old = self.craft_highlights or set()
        self.craft_highlights = positions if positions else None
        # Rebuild chunks that gained or lost highlights
        affected_chunks: set[tuple[int, int, int]] = set()
        for x, y, z in old | (positions or set()):
            affected_chunks.add((x // CHUNK_SIZE, y // CHUNK_SIZE, z))
        for cx, cy, z in affected_chunks:
            self._rebuild_chunk(cx, cy, z)

    def _on_craft_highlights_cleared(self, **kwargs) -> None:
        """Clear highlights and rebuild affected chunks."""
        if self.craft_highlights:
            old = self.craft_highlights
            self.craft_highlights = None
            affected_chunks: set[tuple[int, int, int]] = set()
            for x, y, z in old:
                affected_chunks.add((x // CHUNK_SIZE, y // CHUNK_SIZE, z))
            for cx, cy, z in affected_chunks:
                self._rebuild_chunk(cx, cy, z)

    # ── Ingredient highlighting (temporary cyan) ─────────────────────

    def _on_ingredient_highlight(self, positions: set, **kwargs) -> None:
        """Show cyan highlights on ingredient blocks for 3 seconds."""
        old = self._ingredient_highlights or set()
        self._ingredient_highlights = positions if positions else None
        # Clear after 60 ticks (3 seconds at 20 TPS)
        self._ingredient_clear_tick = self._current_tick + 60
        # Rebuild affected chunks
        affected: set[tuple[int, int, int]] = set()
        for x, y, z in old | (positions or set()):
            affected.add((x // CHUNK_SIZE, y // CHUNK_SIZE, z))
        for cx, cy, z in affected:
            self._rebuild_chunk(cx, cy, z)

    def _clear_ingredient_highlights(self) -> None:
        """Remove ingredient highlights and rebuild affected chunks."""
        if not self._ingredient_highlights:
            return
        old = self._ingredient_highlights
        self._ingredient_highlights = None
        affected: set[tuple[int, int, int]] = set()
        for x, y, z in old:
            affected.add((x // CHUNK_SIZE, y // CHUNK_SIZE, z))
        for cx, cy, z in affected:
            self._rebuild_chunk(cx, cy, z)
