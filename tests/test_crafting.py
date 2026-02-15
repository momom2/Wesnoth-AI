"""Tests for the crafting system."""

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.building.move_system import MoveSystem
from dungeon_builder.building.crafting_system import CraftingSystem
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_DIRT,
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
    VOXEL_GOLD_INGOT,
    VOXEL_MANA_CRYSTAL,
    VOXEL_ENCHANTED_METAL,
    VOXEL_REINFORCED_WALL,
    VOXEL_SPIKE,
    VOXEL_DOOR,
    VOXEL_TREASURE,
    VOXEL_ROLLING_STONE,
    VOXEL_TARP,
    VOXEL_SLOPE,
    VOXEL_STAIRS,
)


def _setup(width=8, depth=8, height=8):
    bus = EventBus()
    grid = VoxelGrid(width=width, depth=depth, height=height)
    ms = MoveSystem(bus, grid)
    cs = CraftingSystem(bus, grid, ms)
    ms.crafting_system = cs
    return bus, grid, ms, cs


# ---------------------------------------------------------------------------
# Original recipe tests (7 existing recipes)
# ---------------------------------------------------------------------------


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
    """Ore in low temperature -> no recipe (no error for air targets)."""
    bus, grid, ms, cs = _setup()
    grid.temperature[4, 4, 4] = 100.0  # not hot enough
    ms.held_material = (VOXEL_IRON_ORE, 1)

    successes = []
    bus.subscribe("craft_success", lambda **kw: successes.append(kw))

    bus.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_IRON_ORE,
        held_count=1,
        target_type=VOXEL_AIR,
    )

    # No craft happened — air targets silently fail (MoveSystem falls through)
    assert len(successes) == 0


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


def test_no_recipe_non_air_target():
    """Arbitrary material on non-air with no matching recipe -> error."""
    bus, grid, ms, cs = _setup()
    grid.grid[4, 4, 4] = VOXEL_STONE
    ms.held_material = (VOXEL_DIRT, 1)

    errors = []
    bus.subscribe("error_message", lambda **kw: errors.append(kw))

    bus.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_DIRT,
        held_count=1,
        target_type=VOXEL_STONE,
    )

    assert len(errors) == 1


def test_no_recipe_air_target():
    """No matching recipe on air -> no error (falls through to place-as-loose)."""
    bus, grid, ms, cs = _setup()
    ms.held_material = (VOXEL_STONE, 1)

    errors = []
    successes = []
    bus.subscribe("error_message", lambda **kw: errors.append(kw))
    bus.subscribe("craft_success", lambda **kw: successes.append(kw))

    bus.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_STONE,
        held_count=1,
        target_type=VOXEL_AIR,
    )

    # No error for air targets (MoveSystem handles fallthrough)
    assert len(errors) == 0
    assert len(successes) == 0


# ---------------------------------------------------------------------------
# Functional block recipe tests (8 new recipes)
# ---------------------------------------------------------------------------


def test_reinforced_wall():
    """Iron ingot on stone -> reinforced wall."""
    bus, grid, ms, cs = _setup()
    grid.grid[4, 4, 4] = VOXEL_STONE
    ms.held_material = (VOXEL_IRON_INGOT, 1)

    successes = []
    bus.subscribe("craft_success", lambda **kw: successes.append(kw))

    bus.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_IRON_INGOT,
        held_count=1,
        target_type=VOXEL_STONE,
    )

    assert len(successes) == 1
    assert successes[0]["recipe"] == "Reinforced Wall"
    assert grid.get(4, 4, 4) == VOXEL_REINFORCED_WALL
    assert ms.held_material is None


def test_reinforced_wall_wrong_target():
    """Iron ingot on dirt -> no match (must be stone)."""
    bus, grid, ms, cs = _setup()
    grid.grid[4, 4, 4] = VOXEL_DIRT
    ms.held_material = (VOXEL_IRON_INGOT, 1)

    errors = []
    bus.subscribe("error_message", lambda **kw: errors.append(kw))

    bus.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_IRON_INGOT,
        held_count=1,
        target_type=VOXEL_DIRT,
    )

    assert len(errors) == 1
    assert ms.held_material == (VOXEL_IRON_INGOT, 1)


def test_treasure():
    """Gold ingot on stone -> treasure."""
    bus, grid, ms, cs = _setup()
    grid.grid[4, 4, 4] = VOXEL_STONE
    ms.held_material = (VOXEL_GOLD_INGOT, 1)

    successes = []
    bus.subscribe("craft_success", lambda **kw: successes.append(kw))

    bus.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_GOLD_INGOT,
        held_count=1,
        target_type=VOXEL_STONE,
    )

    assert len(successes) == 1
    assert successes[0]["recipe"] == "Treasure"
    assert grid.get(4, 4, 4) == VOXEL_TREASURE


def test_spike_trap():
    """Iron ingot on air with solid below -> spike with state=1 (extended)."""
    bus, grid, ms, cs = _setup()
    grid.grid[4, 4, 5] = VOXEL_STONE  # solid below
    ms.held_material = (VOXEL_IRON_INGOT, 1)

    successes = []
    bus.subscribe("craft_success", lambda **kw: successes.append(kw))

    bus.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_IRON_INGOT,
        held_count=1,
        target_type=VOXEL_AIR,
    )

    assert len(successes) == 1
    assert successes[0]["recipe"] == "Spike Trap"
    assert grid.get(4, 4, 4) == VOXEL_SPIKE
    assert grid.get_block_state(4, 4, 4) == 1  # extended


def test_spike_trap_no_floor():
    """Iron ingot on air, air below -> no match."""
    bus, grid, ms, cs = _setup()
    # No solid below (4,4,5) is air
    ms.held_material = (VOXEL_IRON_INGOT, 1)

    successes = []
    bus.subscribe("craft_success", lambda **kw: successes.append(kw))

    bus.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_IRON_INGOT,
        held_count=1,
        target_type=VOXEL_AIR,
    )

    assert len(successes) == 0


def test_door():
    """Enchanted metal on air between two opposite walls -> door (closed)."""
    bus, grid, ms, cs = _setup()
    grid.grid[3, 4, 4] = VOXEL_STONE  # -X wall
    grid.grid[5, 4, 4] = VOXEL_STONE  # +X wall
    ms.held_material = (VOXEL_ENCHANTED_METAL, 1)

    successes = []
    bus.subscribe("craft_success", lambda **kw: successes.append(kw))

    bus.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_ENCHANTED_METAL,
        held_count=1,
        target_type=VOXEL_AIR,
    )

    assert len(successes) == 1
    assert successes[0]["recipe"] == "Door"
    assert grid.get(4, 4, 4) == VOXEL_DOOR
    assert grid.get_block_state(4, 4, 4) == 1  # closed


def test_door_no_opposite_walls():
    """Enchanted metal with only one wall -> no door."""
    bus, grid, ms, cs = _setup()
    grid.grid[3, 4, 4] = VOXEL_STONE  # only -X wall, no +X wall
    ms.held_material = (VOXEL_ENCHANTED_METAL, 1)

    successes = []
    bus.subscribe("craft_success", lambda **kw: successes.append(kw))

    bus.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_ENCHANTED_METAL,
        held_count=1,
        target_type=VOXEL_AIR,
    )

    assert len(successes) == 0


def test_tarp():
    """Dirt on air between two opposite walls -> tarp."""
    bus, grid, ms, cs = _setup()
    grid.grid[4, 3, 4] = VOXEL_STONE  # -Y wall
    grid.grid[4, 5, 4] = VOXEL_STONE  # +Y wall
    ms.held_material = (VOXEL_DIRT, 1)

    successes = []
    bus.subscribe("craft_success", lambda **kw: successes.append(kw))

    bus.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_DIRT,
        held_count=1,
        target_type=VOXEL_AIR,
    )

    assert len(successes) == 1
    assert successes[0]["recipe"] == "Tarp"
    assert grid.get(4, 4, 4) == VOXEL_TARP


def test_tarp_one_wall():
    """Dirt on air with only one wall -> no tarp."""
    bus, grid, ms, cs = _setup()
    grid.grid[4, 3, 4] = VOXEL_STONE  # only -Y wall
    ms.held_material = (VOXEL_DIRT, 1)

    successes = []
    bus.subscribe("craft_success", lambda **kw: successes.append(kw))

    bus.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_DIRT,
        held_count=1,
        target_type=VOXEL_AIR,
    )

    assert len(successes) == 0


def test_slope():
    """Stone on air with solid below and one solid side -> slope.

    Block above is solid to prevent stairs from also matching.
    """
    bus, grid, ms, cs = _setup()
    grid.grid[4, 4, 5] = VOXEL_STONE  # solid below
    grid.grid[3, 4, 4] = VOXEL_STONE  # one solid side (-X)
    grid.grid[4, 4, 3] = VOXEL_STONE  # solid above (blocks stairs, slope-only)
    ms.held_material = (VOXEL_STONE, 1)

    successes = []
    bus.subscribe("craft_success", lambda **kw: successes.append(kw))

    bus.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_STONE,
        held_count=1,
        target_type=VOXEL_AIR,
    )

    assert len(successes) == 1
    assert successes[0]["recipe"] == "Slope"
    assert grid.get(4, 4, 4) == VOXEL_SLOPE


def test_stairs():
    """Stone on air with solid below, one side, air above -> stairs."""
    bus, grid, ms, cs = _setup()
    grid.grid[4, 4, 5] = VOXEL_STONE  # solid below
    grid.grid[3, 4, 4] = VOXEL_STONE  # one solid side (-X)
    # z=3 is above z=4 (shallower), ensure it's air (default)
    ms.held_material = (VOXEL_STONE, 1)

    successes = []
    bus.subscribe("craft_success", lambda **kw: successes.append(kw))

    bus.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_STONE,
        held_count=1,
        target_type=VOXEL_AIR,
    )

    # Both slope and stairs match this scenario (solid below + 1 side + air above)
    # Should auto-execute... wait, both match so it should open a menu
    # Actually: slope requires exactly 1 solid side, stairs requires exactly 1 + air above
    # Both conditions are met here, so 2 matches -> menu
    menus = []
    bus.subscribe("craft_menu", lambda **kw: menus.append(kw))

    # Re-test with fresh setup since first attempt was consumed
    bus2, grid2, ms2, cs2 = _setup()
    grid2.grid[4, 4, 5] = VOXEL_STONE
    grid2.grid[3, 4, 4] = VOXEL_STONE
    ms2.held_material = (VOXEL_STONE, 1)

    menus2 = []
    bus2.subscribe("craft_menu", lambda **kw: menus2.append(kw))

    bus2.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_STONE,
        held_count=1,
        target_type=VOXEL_AIR,
    )

    # Both Slope and Stairs match -> menu opens
    assert len(menus2) == 1
    recipe_names = [r["name"] for r in menus2[0]["recipes"]]
    assert "Slope" in recipe_names
    assert "Stairs" in recipe_names

    # Select Stairs from the menu
    stairs_idx = recipe_names.index("Stairs")
    bus2.publish("craft_selected", recipe_index=stairs_idx)

    assert grid2.get(4, 4, 4) == VOXEL_STAIRS


def test_stairs_blocked_above():
    """Stone on air with solid below and side but solid above -> only slope (not stairs)."""
    bus, grid, ms, cs = _setup()
    grid.grid[4, 4, 5] = VOXEL_STONE  # solid below
    grid.grid[3, 4, 4] = VOXEL_STONE  # one solid side (-X)
    grid.grid[4, 4, 3] = VOXEL_STONE  # solid above -> blocks stairs

    ms.held_material = (VOXEL_STONE, 1)

    successes = []
    bus.subscribe("craft_success", lambda **kw: successes.append(kw))

    bus.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_STONE,
        held_count=1,
        target_type=VOXEL_AIR,
    )

    # Only slope matches (stairs requires air above)
    assert len(successes) == 1
    assert successes[0]["recipe"] == "Slope"
    assert grid.get(4, 4, 4) == VOXEL_SLOPE


def test_rolling_stone():
    """Granite on air above slope -> rolling stone (loose)."""
    bus, grid, ms, cs = _setup()
    grid.grid[4, 4, 5] = VOXEL_SLOPE  # slope below
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
    assert successes[0]["recipe"] == "Rolling Stone"
    assert grid.get(4, 4, 4) == VOXEL_ROLLING_STONE
    assert grid.is_loose(4, 4, 4)


def test_rolling_stone_above_stairs():
    """Granite on air above stairs -> rolling stone."""
    bus, grid, ms, cs = _setup()
    grid.grid[4, 4, 5] = VOXEL_STAIRS  # stairs below
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
    assert grid.get(4, 4, 4) == VOXEL_ROLLING_STONE


def test_rolling_stone_no_slope():
    """Granite on air above regular stone -> granite pillar, NOT rolling stone."""
    bus, grid, ms, cs = _setup()
    grid.grid[4, 4, 5] = VOXEL_STONE  # regular stone below (not slope/stairs)
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

    # Should match granite pillar (solid below), NOT rolling stone
    assert len(successes) == 1
    assert successes[0]["recipe"] == "Granite Pillar"
    assert grid.get(4, 4, 4) == VOXEL_GRANITE


# ---------------------------------------------------------------------------
# Craft menu tests
# ---------------------------------------------------------------------------


def test_craft_menu_multiple_matches():
    """When multiple recipes match, craft_menu is published."""
    bus, grid, ms, cs = _setup()
    # Iron ingot on air with solid below: both Spike Trap and possibly others
    # Iron on stone: Reinforced Wall (but that's non-air target)
    # Iron on air + solid below: Spike Trap
    # Need a scenario with 2+ air-target matches for the same held material
    # Stone on air with solid below + 1 side = Slope + Stairs (if air above)
    grid.grid[4, 4, 5] = VOXEL_STONE  # solid below
    grid.grid[3, 4, 4] = VOXEL_STONE  # one solid side
    # air above at z=3 (default) -> both Slope and Stairs match
    ms.held_material = (VOXEL_STONE, 1)

    menus = []
    bus.subscribe("craft_menu", lambda **kw: menus.append(kw))

    bus.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_STONE,
        held_count=1,
        target_type=VOXEL_AIR,
    )

    assert len(menus) == 1
    assert len(menus[0]["recipes"]) >= 2


def test_craft_menu_selection_executes():
    """Publishing craft_selected after a menu executes the chosen recipe."""
    bus, grid, ms, cs = _setup()
    grid.grid[4, 4, 5] = VOXEL_STONE
    grid.grid[3, 4, 4] = VOXEL_STONE
    ms.held_material = (VOXEL_STONE, 1)

    bus.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_STONE,
        held_count=1,
        target_type=VOXEL_AIR,
    )

    # Menu should be pending
    assert cs.has_pending_craft()

    # Select index 0 (should be Slope based on recipe order)
    successes = []
    bus.subscribe("craft_success", lambda **kw: successes.append(kw))
    bus.publish("craft_selected", recipe_index=0)

    assert len(successes) == 1
    assert not cs.has_pending_craft()


def test_single_match_auto_executes():
    """Exactly 1 match skips the menu and auto-executes."""
    bus, grid, ms, cs = _setup()
    grid.grid[4, 4, 4] = VOXEL_STONE
    ms.held_material = (VOXEL_IRON_INGOT, 1)

    menus = []
    successes = []
    bus.subscribe("craft_menu", lambda **kw: menus.append(kw))
    bus.subscribe("craft_success", lambda **kw: successes.append(kw))

    bus.publish(
        "attempt_craft",
        x=4, y=4, z=4,
        held_type=VOXEL_IRON_INGOT,
        held_count=1,
        target_type=VOXEL_STONE,
    )

    assert len(menus) == 0  # no menu
    assert len(successes) == 1  # auto-executed
    assert grid.get(4, 4, 4) == VOXEL_REINFORCED_WALL


def test_air_target_craft_via_move_system():
    """MoveSystem.drop on air with matching recipe crafts instead of placing."""
    bus, grid, ms, cs = _setup()
    grid.grid[4, 4, 5] = VOXEL_STONE  # solid below for spike
    ms.held_material = (VOXEL_IRON_INGOT, 2)

    result = ms.drop(4, 4, 4)

    assert result is True
    assert grid.get(4, 4, 4) == VOXEL_SPIKE
    assert ms.held_material == (VOXEL_IRON_INGOT, 1)


def test_air_target_no_recipe_places_loose():
    """MoveSystem.drop on air with no matching recipe places as loose."""
    bus, grid, ms, cs = _setup()
    # No special conditions — just bare air, copper ingot has no air-target recipe
    ms.held_material = (VOXEL_COPPER_INGOT, 1)

    result = ms.drop(4, 4, 4)

    assert result is True
    assert grid.get(4, 4, 4) == VOXEL_COPPER_INGOT
    assert grid.is_loose(4, 4, 4)
    assert ms.held_material is None
