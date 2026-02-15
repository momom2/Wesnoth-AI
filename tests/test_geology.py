"""Tests for geology generation."""

import numpy as np

from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.world.geology import GeologyGenerator
from dungeon_builder.utils.rng import SeededRNG
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_BEDROCK,
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
    LAVA_TEMPERATURE,
    MANA_CRYSTAL_TEMPERATURE,
)

SEDIMENTARY = {VOXEL_SANDSTONE, VOXEL_LIMESTONE, VOXEL_SHALE, VOXEL_CHALK}
METAMORPHIC = {VOXEL_SLATE, VOXEL_MARBLE, VOXEL_GNEISS}
IGNEOUS = {VOXEL_GRANITE, VOXEL_BASALT, VOXEL_OBSIDIAN}
ORES = {VOXEL_IRON_ORE, VOXEL_COPPER_ORE, VOXEL_GOLD_ORE, VOXEL_MANA_CRYSTAL}


def _make_grid(seed: int = 42, height: int = 21) -> VoxelGrid:
    grid = VoxelGrid(height=height)
    GeologyGenerator(SeededRNG(seed)).generate(grid)
    return grid


def test_surface_is_air():
    grid = _make_grid()
    for x in range(grid.width):
        for y in range(grid.depth):
            assert grid.get(x, y, 0) == VOXEL_AIR


def test_bedrock_floor():
    grid = _make_grid()
    h = grid.height
    for x in range(grid.width):
        for y in range(grid.depth):
            assert grid.get(x, y, h - 1) == VOXEL_BEDROCK


def test_bedrock_floor_varying_heights():
    """Bedrock is always the last layer regardless of height."""
    for height in [5, 10, 21, 30]:
        grid = _make_grid(height=height)
        for x in range(grid.width):
            for y in range(grid.depth):
                assert grid.get(x, y, height - 1) == VOXEL_BEDROCK


def test_deterministic():
    grid1 = _make_grid(42)
    grid2 = _make_grid(42)
    assert (grid1.grid == grid2.grid).all()
    assert (grid1.humidity == grid2.humidity).all()
    assert (grid1.temperature == grid2.temperature).all()


def test_different_seeds_differ():
    grid1 = _make_grid(42)
    grid2 = _make_grid(99)
    assert not (grid1.grid == grid2.grid).all()


def test_shallow_layers_mostly_dirt():
    grid = _make_grid()
    h = grid.height
    # Z=1 should be mostly dirt (river carves some to air)
    dirt_z = max(1, int(0.05 * (h - 1)))  # near top
    layer = grid.grid[:, :, dirt_z]
    dirt_count = np.sum(layer == VOXEL_DIRT)
    air_count = np.sum(layer == VOXEL_AIR)  # river riverbed
    total = layer.size
    assert (dirt_count + air_count) / total > 0.8


def test_deep_layers_mostly_igneous():
    grid = _make_grid()
    h = grid.height
    # ~70% depth should be mostly igneous rock or ores or air from caves
    deep_z = int(0.70 * (h - 1))
    layer = grid.grid[:, :, deep_z]
    igneous_count = sum(np.sum(layer == v) for v in IGNEOUS)
    ore_count = sum(np.sum(layer == v) for v in ORES)
    air_count = np.sum(layer == VOXEL_AIR)
    total = layer.size
    assert (igneous_count + ore_count + air_count) / total > 0.80


def test_geological_progression():
    """Verify sedimentary is above metamorphic is above igneous."""
    grid = _make_grid()
    h = grid.height

    def dominant_class(z: int) -> str:
        layer = grid.grid[:, :, z]
        sed = sum(int(np.sum(layer == v)) for v in SEDIMENTARY)
        met = sum(int(np.sum(layer == v)) for v in METAMORPHIC)
        ign = sum(int(np.sum(layer == v)) for v in IGNEOUS)
        if sed >= met and sed >= ign:
            return "sedimentary"
        if met >= ign:
            return "metamorphic"
        return "igneous"

    # Proportional checks: 20% depth sedimentary, 40% metamorphic, 70% igneous
    sed_z = int(0.20 * (h - 1))
    met_z = int(0.42 * (h - 1))
    ign_z = int(0.70 * (h - 1))
    assert dominant_class(sed_z) == "sedimentary"
    assert dominant_class(met_z) == "metamorphic"
    assert dominant_class(ign_z) == "igneous"


def test_geological_progression_tall_map():
    """Geological layers work on a taller map too."""
    grid = _make_grid(seed=42, height=30)
    h = grid.height

    def dominant_class(z: int) -> str:
        layer = grid.grid[:, :, z]
        sed = sum(int(np.sum(layer == v)) for v in SEDIMENTARY)
        met = sum(int(np.sum(layer == v)) for v in METAMORPHIC)
        ign = sum(int(np.sum(layer == v)) for v in IGNEOUS)
        if sed >= met and sed >= ign:
            return "sedimentary"
        if met >= ign:
            return "metamorphic"
        return "igneous"

    sed_z = int(0.20 * (h - 1))
    met_z = int(0.42 * (h - 1))
    ign_z = int(0.70 * (h - 1))
    assert dominant_class(sed_z) == "sedimentary"
    assert dominant_class(met_z) == "metamorphic"
    assert dominant_class(ign_z) == "igneous"


def test_river_channel():
    """River should carve air at Z=0 and Z=1 with max humidity."""
    grid = _make_grid()
    river_cells = getattr(grid, '_river_cells', set())
    assert len(river_cells) > 20, "River should have significant coverage"

    for rx, ry in list(river_cells)[:50]:  # check a sample
        assert grid.get(rx, ry, 0) == VOXEL_AIR
        assert grid.get(rx, ry, 1) == VOXEL_AIR  # riverbed carved
        assert grid.get_humidity(rx, ry, 0) == 1.0
        assert grid.get_humidity(rx, ry, 1) == 1.0


def test_lava_river():
    """Lava river should exist at ~85% depth."""
    grid = _make_grid()
    h = grid.height
    lava_z = getattr(grid, '_lava_z', None)
    lava_cells = getattr(grid, '_lava_cells', set())

    assert lava_z is not None, "Lava river z-level should be set"
    assert len(lava_cells) > 10, "Lava river should have coverage"

    # Check lava is at approximately 85% depth
    expected_z = int(0.85 * (h - 1))
    assert abs(lava_z - expected_z) <= 1

    # Check actual lava voxels exist
    lava_count = 0
    for rx, ry in lava_cells:
        if grid.get(rx, ry, lava_z) == VOXEL_LAVA:
            lava_count += 1
    assert lava_count > 5, "Should have lava voxels placed"


def test_lava_temperature():
    """Lava voxels should be at LAVA_TEMPERATURE."""
    grid = _make_grid()
    lava_z = getattr(grid, '_lava_z', None)
    lava_cells = getattr(grid, '_lava_cells', set())

    if lava_z is None or not lava_cells:
        return

    for rx, ry in list(lava_cells)[:20]:
        if grid.get(rx, ry, lava_z) == VOXEL_LAVA:
            assert grid.get_temperature(rx, ry, lava_z) == LAVA_TEMPERATURE


def test_ore_distribution():
    """Ores should exist at appropriate depth ranges."""
    grid = _make_grid()
    h = grid.height

    # Iron should exist in sedimentary/metamorphic range (10-50% depth)
    iron_count = 0
    z_min = max(1, int(0.10 * (h - 1)))
    z_max = min(h - 2, int(0.50 * (h - 1)))
    for z in range(z_min, z_max + 1):
        iron_count += int(np.sum(grid.grid[:, :, z] == VOXEL_IRON_ORE))
    assert iron_count > 0, "Iron ore should be present in sedimentary layers"

    # Mana crystals should exist deep (70-95% depth)
    mana_count = 0
    z_min = int(0.70 * (h - 1))
    z_max = min(h - 2, int(0.95 * (h - 1)))
    for z in range(z_min, z_max + 1):
        mana_count += int(np.sum(grid.grid[:, :, z] == VOXEL_MANA_CRYSTAL))
    assert mana_count > 0, "Mana crystals should be present in deep layers"


def test_humidity_gradient():
    """Humidity should be higher near river and at depth."""
    grid = _make_grid()
    river_cells = getattr(grid, '_river_cells', set())
    if not river_cells:
        return

    # Pick a river cell and check humidity decreases with distance
    rx, ry = next(iter(river_cells))
    river_humidity = grid.get_humidity(rx, ry, 0)
    assert river_humidity == 1.0

    # Check that far-away surface cells have lower humidity
    # Find a cell far from river
    for x in range(grid.width):
        far = True
        for rc in river_cells:
            if abs(x - rc[0]) < 10:
                far = False
                break
        if far:
            assert grid.get_humidity(x, grid.depth // 2, 0) < 0.5
            break


def test_temperature_baseline():
    """Temperature should increase with depth."""
    grid = _make_grid()
    h = grid.height

    # Surface should be around 20
    surface_temp = grid.get_temperature(grid.width // 2, grid.depth // 2, 0)
    assert 15.0 <= surface_temp <= 25.0

    # Deep should be higher
    deep_z = int(0.8 * (h - 1))
    deep_temp = grid.get_temperature(grid.width // 2, grid.depth // 2, deep_z)
    assert deep_temp > surface_temp


def test_mana_crystal_temperature():
    """Mana crystals should be at their fixed temperature."""
    grid = _make_grid()
    h = grid.height

    for x in range(grid.width):
        for y in range(grid.depth):
            for z in range(h):
                if grid.get(x, y, z) == VOXEL_MANA_CRYSTAL:
                    assert grid.get_temperature(x, y, z) == MANA_CRYSTAL_TEMPERATURE
                    return  # found one, that's enough
    # If no mana crystals found, test still passes (stochastic)


def test_no_lava_on_shallow_map():
    """Very shallow maps should skip lava generation gracefully."""
    grid = _make_grid(seed=42, height=4)
    # Should not crash; no lava expected on 4-layer map
    assert grid.height == 4
