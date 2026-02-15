"""Tests for the crafting system."""

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.building.move_system import MoveSystem
from dungeon_builder.building.crafting_system import CraftingSystem
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_STONE,
    VOXEL_MARBLE,
    VOXEL_LIMESTONE,
    VOXEL_BASALT,
    VOXEL_GRANITE,
    VOXEL_SANDSTONE,
    VOXEL_CHALK,
    VOXEL_LAVA,
    VOXEL_OBSIDIAN,
    VOXEL_IRON_ORE,
    VOXEL_COPPER_ORE,
    VOXEL_IRON_INGOT,
    VOXEL_COPPER_INGOT,
    VOXEL_MANA_CRYSTAL,
    VOXEL_ENCHANTED_METAL,
)


def _setup(width=8, depth=8, height=8):
    bus = EventBus()
    grid = VoxelGrid(width=width, depth=depth, height=height)
    ms = MoveSystem(bus, grid)
    cs = CraftingSystem(bus, grid, ms)
    return bus, grid, ms, cs


def test_marble_wall():
    """Marble on air with stone behind -> marble wall."""
    bus, grid, ms, cs = _setup()
    # Stone on one side, air on the other
    grid.grid[3, 4, 4] = VOXEL_STONE  # west neighbor
    # (4,4,4) is air, (5,4,4) is air
    ms.held_material = (VOXEL_MARBLE, 1)

    successes = []
    bus.subscribe("craft_success", lambda **kw: successes.append(kw))

    bus.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_MARBLE,
        held_count=1,
        target_type=VOXEL_AIR,
    )

    assert len(successes) == 1
    assert successes[0]["recipe"] == "Marble Wall"
    assert grid.get(4, 4, 4) == VOXEL_MARBLE
    assert ms.held_material is None


def test_ore_smelting():
    """Iron ore on air with high temperature -> iron ingot."""
    bus, grid, ms, cs = _setup()
    grid.temperature[4, 4, 4] = 900.0  # hot enough
    ms.held_material = (VOXEL_IRON_ORE, 2)

    successes = []
    bus.subscribe("craft_success", lambda **kw: successes.append(kw))

    bus.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_IRON_ORE,
        held_count=2,
        target_type=VOXEL_AIR,
    )

    assert len(successes) == 1
    assert grid.get(4, 4, 4) == VOXEL_IRON_INGOT
    assert grid.is_loose(4, 4, 4)
    # Should have 1 remaining
    assert ms.held_material == (VOXEL_IRON_ORE, 1)


def test_ore_smelting_too_cold():
    """Ore in low temperature -> no recipe."""
    bus, grid, ms, cs = _setup()
    grid.temperature[4, 4, 4] = 100.0  # not hot enough
    ms.held_material = (VOXEL_IRON_ORE, 1)

    errors = []
    bus.subscribe("error_message", lambda **kw: errors.append(kw))

    bus.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_IRON_ORE,
        held_count=1,
        target_type=VOXEL_AIR,
    )

    assert len(errors) == 1
    assert "no recipe" in errors[0]["text"].lower()


def test_obsidian_forge():
    """Basalt on lava -> obsidian."""
    bus, grid, ms, cs = _setup()
    grid.grid[4, 4, 4] = VOXEL_LAVA
    ms.held_material = (VOXEL_BASALT, 1)

    successes = []
    bus.subscribe("craft_success", lambda **kw: successes.append(kw))

    bus.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_BASALT,
        held_count=1,
        target_type=VOXEL_LAVA,
    )

    assert len(successes) == 1
    assert grid.get(4, 4, 4) == VOXEL_OBSIDIAN
    assert grid.is_loose(4, 4, 4)


def test_mana_infusion():
    """Mana crystal on iron ingot near lava -> enchanted metal."""
    bus, grid, ms, cs = _setup()
    grid.grid[4, 4, 4] = VOXEL_IRON_INGOT
    grid.grid[6, 4, 4] = VOXEL_LAVA  # within 3 blocks
    ms.held_material = (VOXEL_MANA_CRYSTAL, 1)

    successes = []
    bus.subscribe("craft_success", lambda **kw: successes.append(kw))

    bus.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_MANA_CRYSTAL,
        held_count=1,
        target_type=VOXEL_IRON_INGOT,
    )

    assert len(successes) == 1
    assert grid.get(4, 4, 4) == VOXEL_ENCHANTED_METAL


def test_mana_infusion_no_lava_nearby():
    """Mana crystal on ingot without lava nearby -> no recipe."""
    bus, grid, ms, cs = _setup()
    grid.grid[4, 4, 4] = VOXEL_IRON_INGOT
    ms.held_material = (VOXEL_MANA_CRYSTAL, 1)

    errors = []
    bus.subscribe("error_message", lambda **kw: errors.append(kw))

    bus.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_MANA_CRYSTAL,
        held_count=1,
        target_type=VOXEL_IRON_INGOT,
    )

    assert len(errors) == 1


def test_stone_brick():
    """Limestone on air near a wall -> limestone block."""
    bus, grid, ms, cs = _setup()
    grid.grid[3, 4, 4] = VOXEL_STONE  # neighboring wall
    ms.held_material = (VOXEL_LIMESTONE, 1)

    successes = []
    bus.subscribe("craft_success", lambda **kw: successes.append(kw))

    bus.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_LIMESTONE,
        held_count=1,
        target_type=VOXEL_AIR,
    )

    assert len(successes) == 1
    assert grid.get(4, 4, 4) == VOXEL_LIMESTONE


def test_glass():
    """Sandstone on hot air -> chalk (glass)."""
    bus, grid, ms, cs = _setup()
    grid.temperature[4, 4, 4] = 700.0
    ms.held_material = (VOXEL_SANDSTONE, 1)

    successes = []
    bus.subscribe("craft_success", lambda **kw: successes.append(kw))

    bus.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_SANDSTONE,
        held_count=1,
        target_type=VOXEL_AIR,
    )

    assert len(successes) == 1
    assert grid.get(4, 4, 4) == VOXEL_CHALK


def test_granite_pillar():
    """Granite on air above solid ground -> granite pillar."""
    bus, grid, ms, cs = _setup()
    grid.grid[4, 4, 5] = VOXEL_STONE  # solid below
    ms.held_material = (VOXEL_GRANITE, 1)

    successes = []
    bus.subscribe("craft_success", lambda **kw: successes.append(kw))

    bus.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_GRANITE,
        held_count=1,
        target_type=VOXEL_AIR,
    )

    assert len(successes) == 1
    assert grid.get(4, 4, 4) == VOXEL_GRANITE


def test_no_recipe():
    """Arbitrary material with no matching recipe -> error."""
    bus, grid, ms, cs = _setup()
    ms.held_material = (VOXEL_STONE, 1)

    errors = []
    bus.subscribe("error_message", lambda **kw: errors.append(kw))

    bus.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_STONE,
        held_count=1,
        target_type=VOXEL_AIR,
    )

    assert len(errors) == 1
