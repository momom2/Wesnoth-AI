"""Procedural geology generation with realistic strata, ores, and a surface river."""

from __future__ import annotations

import math

from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_BEDROCK,
    VOXEL_CORE,
    VOXEL_DIRT,
    VOXEL_SANDSTONE,
    VOXEL_LIMESTONE,
    VOXEL_SHALE,
    VOXEL_CHALK,
    VOXEL_SLATE,
    VOXEL_MARBLE,
    VOXEL_GNEISS,
    VOXEL_GRANITE,
    VOXEL_BASALT,
    VOXEL_OBSIDIAN,
    VOXEL_IRON_ORE,
    VOXEL_COPPER_ORE,
    VOXEL_GOLD_ORE,
    VOXEL_MANA_CRYSTAL,
    VOXEL_LAVA,
    VOXEL_POROSITY,
    LAVA_TEMPERATURE,
    MANA_CRYSTAL_TEMPERATURE,
)
from dungeon_builder.utils.rng import SeededRNG
from dungeon_builder.world.voxel_grid import VoxelGrid


# ---------------------------------------------------------------------------
# Value noise (no external deps)
# ---------------------------------------------------------------------------

class ValueNoise2D:
    """Hash-based 2D value noise with smoothstep interpolation."""

    def __init__(self, seed: int, scale: float = 16.0) -> None:
        self.seed = seed
        self.scale = scale

    def _hash(self, ix: int, iy: int) -> float:
        """Deterministic hash from grid coords to [0, 1]."""
        # Combine seed with coords using large primes
        h = (ix * 374761393 + iy * 668265263 + self.seed * 1274126177) & 0xFFFFFFFF
        h = ((h ^ (h >> 13)) * 1103515245 + 12345) & 0xFFFFFFFF
        return (h & 0x7FFFFFFF) / 0x7FFFFFFF

    @staticmethod
    def _smoothstep(t: float) -> float:
        return t * t * (3.0 - 2.0 * t)

    def sample(self, x: float, y: float) -> float:
        """Sample noise at world position. Returns value in [0, 1]."""
        sx = x / self.scale
        sy = y / self.scale
        ix0 = int(math.floor(sx))
        iy0 = int(math.floor(sy))
        fx = sx - ix0
        fy = sy - iy0
        fx = self._smoothstep(fx)
        fy = self._smoothstep(fy)

        v00 = self._hash(ix0, iy0)
        v10 = self._hash(ix0 + 1, iy0)
        v01 = self._hash(ix0, iy0 + 1)
        v11 = self._hash(ix0 + 1, iy0 + 1)

        v0 = v00 + (v10 - v00) * fx
        v1 = v01 + (v11 - v01) * fx
        return v0 + (v1 - v0) * fy

    def fbm(self, x: float, y: float, octaves: int = 3) -> float:
        """Fractional Brownian Motion — layered noise for organic shapes."""
        value = 0.0
        amplitude = 1.0
        frequency = 1.0
        total_amp = 0.0
        for _ in range(octaves):
            value += self.sample(x * frequency, y * frequency) * amplitude
            total_amp += amplitude
            amplitude *= 0.5
            frequency *= 2.0
        return value / total_amp


# ---------------------------------------------------------------------------
# Geology generator
# ---------------------------------------------------------------------------

class GeologyGenerator:
    """Fills a VoxelGrid with geologically-inspired strata, ores, and rivers.

    Layer structure (by proportional depth, z=0 is surface, z=h-1 is bedrock):
      0.00:           Air (surface)
      0.00-0.10:      Dirt (topsoil, noise-varied thickness)
      0.10-0.30:      Sedimentary (sandstone, limestone, shale, chalk)
      0.30-0.55:      Metamorphic (slate, marble, gneiss)
      0.55-0.90:      Igneous (granite, basalt, rare obsidian)
      0.90-0.95:      Dense granite/basalt
      0.95-1.00:      Bedrock (indestructible)

    Adapts to any grid height — not all layers appear on shallow maps.
    """

    def __init__(self, rng: SeededRNG) -> None:
        self.rng = rng.fork("geology")

    def generate(self, voxel_grid: VoxelGrid) -> None:
        w, d, h = voxel_grid.width, voxel_grid.depth, voxel_grid.height
        grid = voxel_grid.grid

        # Noise layers for strata boundaries and variation
        strata_noise = ValueNoise2D(self.rng.randint(0, 2**31), scale=20.0)
        detail_noise = ValueNoise2D(self.rng.randint(0, 2**31), scale=10.0)
        ore_noise = ValueNoise2D(self.rng.randint(0, 2**31), scale=8.0)

        # --- Surface ---
        grid[:, :, 0] = VOXEL_AIR

        # --- Fill each column based on proportional depth + noise ---
        max_z = max(h - 1, 1)
        for x in range(w):
            for y in range(d):
                n = strata_noise.fbm(float(x), float(y), octaves=3)
                d2 = detail_noise.fbm(float(x), float(y), octaves=2)
                # Noise offset as proportion of total depth (±5%)
                offset = (n - 0.5) * 0.10

                for z in range(1, h):
                    depth_ratio = z / max_z
                    vtype = self._pick_rock_type(depth_ratio, offset, d2)
                    grid[x, y, z] = vtype

        # --- Bedrock floor (last layer always bedrock) ---
        grid[:, :, h - 1] = VOXEL_BEDROCK

        # --- Ore deposits ---
        self._place_ores(voxel_grid, ore_noise)

        # --- Natural caves ---
        self._carve_caves(voxel_grid)

        # --- Surface river ---
        self._generate_river(voxel_grid)

        # --- Lava river (deep) ---
        self._generate_lava_river(voxel_grid)

        # --- Humidity ---
        self._initialize_humidity(voxel_grid)

        # --- Temperature ---
        self._initialize_temperature(voxel_grid)

    def _pick_rock_type(self, depth_ratio: float, offset: float, detail: float) -> int:
        """Choose voxel type for a given proportional depth, with noise variation.

        depth_ratio: 0.0 = surface, 1.0 = deepest.
        offset: noise-based shift (±0.05).
        detail: secondary noise value in [0, 1].
        """
        eff = depth_ratio + offset

        # Dirt layer (0.00-0.10)
        if eff < 0.10 + detail * 0.02:
            return VOXEL_DIRT

        # Sedimentary zone (0.10-0.30)
        if eff < 0.30 + detail * 0.02:
            band = (eff - 0.10) / 0.20  # 0..1 through the zone
            if band < 0.3:
                return VOXEL_SANDSTONE
            elif band < 0.55:
                return VOXEL_LIMESTONE
            elif band < 0.75:
                return VOXEL_CHALK if detail > 0.5 else VOXEL_SANDSTONE
            else:
                return VOXEL_SHALE

        # Metamorphic zone (0.30-0.55)
        if eff < 0.55 + detail * 0.02:
            band = (eff - 0.30) / 0.25
            if band < 0.35:
                return VOXEL_SLATE
            elif band < 0.65:
                return VOXEL_MARBLE if detail > 0.6 else VOXEL_GNEISS
            else:
                return VOXEL_GNEISS

        # Igneous zone (0.55-0.90)
        if eff < 0.90:
            band = (eff - 0.55) / 0.35
            if detail > 0.85 and eff > 0.75:
                return VOXEL_OBSIDIAN
            elif band < 0.5:
                return VOXEL_GRANITE
            else:
                return VOXEL_BASALT

        # Dense zone (0.90-0.95)
        if eff < 0.95:
            return VOXEL_BASALT if detail > 0.5 else VOXEL_GRANITE

        return VOXEL_BEDROCK

    # -------------------------------------------------------------------
    # Ore placement
    # -------------------------------------------------------------------

    def _place_ores(self, voxel_grid: VoxelGrid, ore_noise: ValueNoise2D) -> None:
        """Scatter ore clusters at geologically appropriate depths."""
        w, d, h = voxel_grid.width, voxel_grid.depth, voxel_grid.height
        max_z = max(h - 1, 1)

        # (ore_type, min_ratio, max_ratio, cluster_count, cluster_size_range)
        ore_defs = [
            (VOXEL_IRON_ORE, 0.10, 0.50, 12, (3, 7)),     # iron in sedimentary/metamorphic
            (VOXEL_COPPER_ORE, 0.30, 0.65, 10, (3, 6)),    # copper in metamorphic
            (VOXEL_GOLD_ORE, 0.55, 0.90, 6, (2, 5)),       # gold in igneous
            (VOXEL_MANA_CRYSTAL, 0.70, 0.95, 4, (2, 4)),   # mana crystals deep
        ]

        for ore_type, r_min, r_max, count, (sz_min, sz_max) in ore_defs:
            z_min = max(1, int(r_min * max_z))
            z_max = min(h - 2, int(r_max * max_z))
            if z_min >= z_max:
                continue
            for _ in range(count):
                cx = self.rng.randint(4, w - 4)
                cy = self.rng.randint(4, d - 4)
                cz = self.rng.randint(z_min, z_max)
                size = self.rng.randint(sz_min, sz_max)
                self._place_ore_cluster(voxel_grid, ore_type, cx, cy, cz, size)

    def _place_ore_cluster(
        self, voxel_grid: VoxelGrid, ore_type: int,
        cx: int, cy: int, cz: int, size: int,
    ) -> None:
        """Place a small irregular cluster of ore voxels."""
        grid = voxel_grid.grid
        placed = 0
        x, y, z = cx, cy, cz
        for _ in range(size * 3):  # attempts
            if placed >= size:
                break
            if voxel_grid.in_bounds(x, y, z):
                current = int(grid[x, y, z])
                # Only replace solid rock, not air/bedrock/core/other ores/lava
                if current not in (VOXEL_AIR, VOXEL_BEDROCK, VOXEL_CORE, VOXEL_LAVA) and current < 40:
                    grid[x, y, z] = ore_type
                    placed += 1
            # Random walk to next position
            direction = self.rng.randint(0, 5)
            if direction == 0: x += 1
            elif direction == 1: x -= 1
            elif direction == 2: y += 1
            elif direction == 3: y -= 1
            elif direction == 4: z += 1
            else: z -= 1

    # -------------------------------------------------------------------
    # Cave carving
    # -------------------------------------------------------------------

    def _carve_caves(self, voxel_grid: VoxelGrid) -> None:
        """Carve a few small natural caverns."""
        h = voxel_grid.height
        num_caves = self.rng.randint(3, 8)
        for _ in range(num_caves):
            cx = self.rng.randint(8, voxel_grid.width - 8)
            cy = self.rng.randint(8, voxel_grid.depth - 8)
            # Caves in the 30-80% depth range
            cz = self.rng.randint(max(2, int(0.30 * h)), max(3, int(0.80 * h)))
            radius = self.rng.randint(2, 4)
            self._carve_sphere(voxel_grid, cx, cy, cz, radius)

    def _carve_sphere(
        self, voxel_grid: VoxelGrid, cx: int, cy: int, cz: int, radius: int,
    ) -> None:
        """Carve a roughly spherical air pocket."""
        for x in range(cx - radius, cx + radius + 1):
            for y in range(cy - radius, cy + radius + 1):
                for z in range(cz - radius, cz + radius + 1):
                    if not voxel_grid.in_bounds(x, y, z):
                        continue
                    dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
                    if dist < radius + self.rng.uniform(-0.5, 0.5):
                        if voxel_grid.grid[x, y, z] not in (VOXEL_BEDROCK, VOXEL_LAVA):
                            voxel_grid.grid[x, y, z] = VOXEL_AIR

    # -------------------------------------------------------------------
    # Surface river generation
    # -------------------------------------------------------------------

    def _generate_river(self, voxel_grid: VoxelGrid) -> None:
        """Carve a meandering river channel across the surface (Z=0).

        The river is air voxels with maximum humidity. When the physics
        engine is built, water will flow through these channels.
        """
        w, d = voxel_grid.width, voxel_grid.depth
        grid = voxel_grid.grid

        # River flows roughly west-to-east (x=0 to x=w-1)
        # Pick start and end Y positions in the middle third
        third = d // 3
        start_y = self.rng.randint(third, 2 * third)
        end_y = self.rng.randint(third, 2 * third)

        # Generate waypoints with random deviation
        num_waypoints = 6
        waypoints: list[tuple[float, float]] = [(0.0, float(start_y))]
        for i in range(1, num_waypoints):
            t = i / num_waypoints
            base_x = t * (w - 1)
            base_y = start_y + t * (end_y - start_y)
            deviation = self.rng.uniform(-8.0, 8.0)
            waypoints.append((base_x, base_y + deviation))
        waypoints.append((float(w - 1), float(end_y)))

        # Interpolate smooth path between waypoints
        river_cells: set[tuple[int, int]] = set()
        for i in range(len(waypoints) - 1):
            x0, y0 = waypoints[i]
            x1, y1 = waypoints[i + 1]
            steps = int(max(abs(x1 - x0), abs(y1 - y0))) * 2 + 1
            for s in range(steps + 1):
                t = s / max(steps, 1)
                px = x0 + (x1 - x0) * t
                py = y0 + (y1 - y0) * t
                ix, iy = int(round(px)), int(round(py))
                # Carve width of 2-3 blocks around center
                half_w = self.rng.choice([1, 1, 1, 2])
                for dx in range(-half_w, half_w + 1):
                    for dy in range(-half_w, half_w + 1):
                        rx, ry = ix + dx, iy + dy
                        if 0 <= rx < w and 0 <= ry < d:
                            river_cells.add((rx, ry))

        # Carve river at Z=0 (already air) and Z=1 (dig into dirt for riverbed)
        for rx, ry in river_cells:
            grid[rx, ry, 0] = VOXEL_AIR
            if voxel_grid.in_bounds(rx, ry, 1):
                grid[rx, ry, 1] = VOXEL_AIR  # riverbed carved into dirt

        # Store river cells on the grid for humidity initialization
        voxel_grid._river_cells = river_cells  # type: ignore[attr-defined]

    # -------------------------------------------------------------------
    # Lava river generation (deep)
    # -------------------------------------------------------------------

    def _generate_lava_river(self, voxel_grid: VoxelGrid) -> None:
        """Generate a meandering lava river at 80-90% depth."""
        w, d, h = voxel_grid.width, voxel_grid.depth, voxel_grid.height
        grid = voxel_grid.grid

        if h < 5:
            return  # too shallow for lava

        # Lava river z-level: ~85% depth
        lava_z = max(2, int(0.85 * (h - 1)))
        if lava_z >= h - 1:
            lava_z = h - 2  # don't overwrite bedrock

        # River flows roughly north-to-south (y=0 to y=d-1)
        third = w // 3
        start_x = self.rng.randint(third, 2 * third)
        end_x = self.rng.randint(third, 2 * third)

        num_waypoints = 5
        waypoints: list[tuple[float, float]] = [(float(start_x), 0.0)]
        for i in range(1, num_waypoints):
            t = i / num_waypoints
            base_y = t * (d - 1)
            base_x = start_x + t * (end_x - start_x)
            deviation = self.rng.uniform(-6.0, 6.0)
            waypoints.append((base_x + deviation, base_y))
        waypoints.append((float(end_x), float(d - 1)))

        # Interpolate path
        lava_cells: set[tuple[int, int]] = set()
        for i in range(len(waypoints) - 1):
            x0, y0 = waypoints[i]
            x1, y1 = waypoints[i + 1]
            steps = int(max(abs(x1 - x0), abs(y1 - y0))) * 2 + 1
            for s in range(steps + 1):
                t = s / max(steps, 1)
                px = x0 + (x1 - x0) * t
                py = y0 + (y1 - y0) * t
                ix, iy = int(round(px)), int(round(py))
                # Lava channel width: 1-2 blocks
                half_w = self.rng.choice([1, 1, 1, 2])
                for dx in range(-half_w, half_w + 1):
                    for dy in range(-half_w, half_w + 1):
                        rx, ry = ix + dx, iy + dy
                        if 0 <= rx < w and 0 <= ry < d:
                            lava_cells.add((rx, ry))

        # Place lava voxels
        for rx, ry in lava_cells:
            if voxel_grid.in_bounds(rx, ry, lava_z):
                current = int(grid[rx, ry, lava_z])
                if current != VOXEL_BEDROCK:
                    grid[rx, ry, lava_z] = VOXEL_LAVA

        # Store for temperature initialization
        voxel_grid._lava_cells = lava_cells  # type: ignore[attr-defined]
        voxel_grid._lava_z = lava_z  # type: ignore[attr-defined]

    # -------------------------------------------------------------------
    # Humidity initialization
    # -------------------------------------------------------------------

    def _initialize_humidity(self, voxel_grid: VoxelGrid) -> None:
        """Set initial humidity: river=1.0, gradient near river, depth-based baseline."""
        w, d, h = voxel_grid.width, voxel_grid.depth, voxel_grid.height
        humidity = voxel_grid.humidity
        grid = voxel_grid.grid

        # Baseline humidity: increases with depth and porosity
        for z in range(h):
            depth_factor = z / max(h - 1, 1)  # 0 at surface, 1 at bottom
            base_humidity = 0.05 + depth_factor * 0.25  # 0.05..0.30
            for x in range(w):
                for y in range(d):
                    vtype = int(grid[x, y, z])
                    porosity = VOXEL_POROSITY.get(vtype, 0.0)
                    # Porous rock retains more moisture
                    humidity[x, y, z] = base_humidity * porosity

        # River channel: max humidity
        river_cells = getattr(voxel_grid, '_river_cells', set())
        for rx, ry in river_cells:
            # River surface and riverbed
            for z in range(min(2, h)):
                humidity[rx, ry, z] = 1.0

        # Propagate humidity gradient around river
        max_spread = 5
        for rx, ry in river_cells:
            for dx in range(-max_spread, max_spread + 1):
                for dy in range(-max_spread, max_spread + 1):
                    nx, ny = rx + dx, ry + dy
                    if not (0 <= nx < w and 0 <= ny < d):
                        continue
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < 1.0:
                        continue
                    # Humidity falls off with distance, scaled by porosity
                    for z in range(min(4, h)):
                        vtype = int(grid[nx, ny, z])
                        porosity = VOXEL_POROSITY.get(vtype, 0.0)
                        falloff = max(0.0, 1.0 - dist / max_spread) * porosity
                        if falloff > humidity[nx, ny, z]:
                            humidity[nx, ny, z] = falloff

    # -------------------------------------------------------------------
    # Temperature initialization
    # -------------------------------------------------------------------

    def _initialize_temperature(self, voxel_grid: VoxelGrid) -> None:
        """Set initial temperature: baseline increases with depth, lava=hot, mana=fixed."""
        w, d, h = voxel_grid.width, voxel_grid.depth, voxel_grid.height
        temperature = voxel_grid.temperature
        grid = voxel_grid.grid

        # Baseline: 20 at surface, 100 at deepest
        for z in range(h):
            depth_factor = z / max(h - 1, 1)
            base_temp = 20.0 + depth_factor * 80.0
            for x in range(w):
                for y in range(d):
                    temperature[x, y, z] = base_temp

        # Lava voxels: fixed high temperature
        for x in range(w):
            for y in range(d):
                for z in range(h):
                    vtype = int(grid[x, y, z])
                    if vtype == VOXEL_LAVA:
                        temperature[x, y, z] = LAVA_TEMPERATURE
                    elif vtype == VOXEL_MANA_CRYSTAL:
                        temperature[x, y, z] = MANA_CRYSTAL_TEMPERATURE
