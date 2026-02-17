"""Tests for the crafting system (highlight-mode state machine)."""

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.core.game_state import GameState
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
    DEFAULT_SEED,
)


def _setup(width=8, depth=8, height=8):
    bus = EventBus()
    grid = VoxelGrid(width=width, depth=depth, height=height)
    # Mark all blocks visible and claimed so territory checks don't interfere
    grid.visible[:] = True
    grid.claimed[:] = True
    gs = GameState(DEFAULT_SEED)
    gs.event_bus = bus
    ms = MoveSystem(bus, grid, gs)
    cs = CraftingSystem(bus, grid, ms, gs)
    gs.build_system = None
    gs.move_system = ms
    return bus, grid, ms, cs, gs


def _craft(bus, cs, recipe_name, x, y, z):
    """Helper: set z-level, select recipe, then craft at position. Returns success events."""
    successes = []
    bus.subscribe("craft_success", lambda **kw: successes.append(kw))

    # Ensure the crafting system scans at the target z-level
    cs._current_z = z
    bus.publish("craft_recipe_selected", recipe_name=recipe_name)
    bus.publish("craft_at_position", x=x, y=y, z=z)
    return successes


# ---------------------------------------------------------------------------
# Original recipe tests (7 existing recipes)
# ---------------------------------------------------------------------------


def test_marble_wall():
    """Marble on air with stone behind -> marble wall."""
    bus, grid, ms, cs, gs = _setup()
    # Stone on one side, air on the other
    grid.grid[3, 4, 4] = VOXEL_STONE  # west neighbor
    # (4,4,4) is air, (5,4,4) is air
    ms.held_materials = {VOXEL_MARBLE: 1}

    successes = _craft(bus, cs, "Marble Wall", 4, 4, 4)

    assert len(successes) == 1
    assert successes[0]["recipe"] == "Marble Wall"
    assert grid.get(4, 4, 4) == VOXEL_MARBLE
    assert not ms.has_material(VOXEL_MARBLE)


def test_ore_smelting():
    """Iron ore on air with high temperature -> iron ingot."""
    bus, grid, ms, cs, gs = _setup()
    grid.temperature[4, 4, 4] = 900.0  # hot enough
    ms.held_materials = {VOXEL_IRON_ORE: 2}

    successes = _craft(bus, cs, "Ore Smelting", 4, 4, 4)

    assert len(successes) == 1
    assert grid.get(4, 4, 4) == VOXEL_IRON_INGOT
    assert grid.is_loose(4, 4, 4)
    # Should have 1 remaining
    assert ms.held_materials == {VOXEL_IRON_ORE: 1}


def test_ore_smelting_too_cold():
    """Ore in low temperature -> no valid positions."""
    bus, grid, ms, cs, gs = _setup()
    grid.temperature[4, 4, 4] = 100.0  # not hot enough
    ms.held_materials = {VOXEL_IRON_ORE: 1}

    bus.publish("craft_recipe_selected", recipe_name="Ore Smelting")

    # Craft mode entered but no highlighted positions at z=4
    # (The position (4,4,4) doesn't satisfy the temperature check)
    # Trying to craft there should cancel (not in highlighted set)
    highlights = []
    bus.subscribe("craft_highlights_updated", lambda **kw: highlights.append(kw))

    # Set z-level to scan at z=4
    cs._current_z = 4
    cs._scan_and_highlight(4)

    # No valid positions at this z-level with low temperature
    assert len(cs._highlighted_positions) == 0


def test_obsidian_forge():
    """Basalt on lava -> obsidian."""
    bus, grid, ms, cs, gs = _setup()
    grid.grid[4, 4, 4] = VOXEL_LAVA
    ms.held_materials = {VOXEL_BASALT: 1}

    successes = _craft(bus, cs, "Obsidian Forge", 4, 4, 4)

    assert len(successes) == 1
    assert grid.get(4, 4, 4) == VOXEL_OBSIDIAN
    assert grid.is_loose(4, 4, 4)


def test_mana_infusion():
    """Mana crystal on iron ingot near lava -> enchanted metal."""
    bus, grid, ms, cs, gs = _setup()
    grid.grid[4, 4, 4] = VOXEL_IRON_INGOT
    grid.grid[6, 4, 4] = VOXEL_LAVA  # within 3 blocks
    ms.held_materials = {VOXEL_MANA_CRYSTAL: 1}

    successes = _craft(bus, cs, "Mana Infusion", 4, 4, 4)

    assert len(successes) == 1
    assert grid.get(4, 4, 4) == VOXEL_ENCHANTED_METAL


def test_mana_infusion_no_lava_nearby():
    """Mana crystal on ingot without lava nearby -> not in highlights."""
    bus, grid, ms, cs, gs = _setup()
    grid.grid[4, 4, 4] = VOXEL_IRON_INGOT
    ms.held_materials = {VOXEL_MANA_CRYSTAL: 1}

    bus.publish("craft_recipe_selected", recipe_name="Mana Infusion")

    # Position (4,4,4) should NOT be in highlights since no lava nearby
    cs._current_z = 4
    cs._scan_and_highlight(4)
    assert (4, 4, 4) not in cs._highlighted_positions


def test_stone_brick():
    """Limestone on air near a wall -> limestone block."""
    bus, grid, ms, cs, gs = _setup()
    grid.grid[3, 4, 4] = VOXEL_STONE  # neighboring wall
    ms.held_materials = {VOXEL_LIMESTONE: 1}

    successes = _craft(bus, cs, "Stone Brick", 4, 4, 4)

    assert len(successes) == 1
    assert grid.get(4, 4, 4) == VOXEL_LIMESTONE


def test_glass():
    """Sandstone on hot air -> chalk (glass)."""
    bus, grid, ms, cs, gs = _setup()
    grid.temperature[4, 4, 4] = 700.0
    ms.held_materials = {VOXEL_SANDSTONE: 1}

    successes = _craft(bus, cs, "Glass", 4, 4, 4)

    assert len(successes) == 1
    assert grid.get(4, 4, 4) == VOXEL_CHALK


def test_granite_pillar():
    """Granite on air above solid ground -> granite pillar."""
    bus, grid, ms, cs, gs = _setup()
    grid.grid[4, 4, 5] = VOXEL_STONE  # solid below
    ms.held_materials = {VOXEL_GRANITE: 1}

    successes = _craft(bus, cs, "Granite Pillar", 4, 4, 4)

    assert len(successes) == 1
    assert grid.get(4, 4, 4) == VOXEL_GRANITE


# ---------------------------------------------------------------------------
# Functional block recipe tests (8 new recipes)
# ---------------------------------------------------------------------------


def test_reinforced_wall():
    """Iron ingot on stone -> reinforced wall."""
    bus, grid, ms, cs, gs = _setup()
    grid.grid[4, 4, 4] = VOXEL_STONE
    ms.held_materials = {VOXEL_IRON_INGOT: 1}

    successes = _craft(bus, cs, "Reinforced Wall", 4, 4, 4)

    assert len(successes) == 1
    assert successes[0]["recipe"] == "Reinforced Wall"
    assert grid.get(4, 4, 4) == VOXEL_REINFORCED_WALL
    assert not ms.has_material(VOXEL_IRON_INGOT)


def test_reinforced_wall_wrong_target():
    """Iron ingot on dirt -> not in highlights (must be stone)."""
    bus, grid, ms, cs, gs = _setup()
    grid.grid[4, 4, 4] = VOXEL_DIRT
    ms.held_materials = {VOXEL_IRON_INGOT: 1}

    bus.publish("craft_recipe_selected", recipe_name="Reinforced Wall")

    cs._current_z = 4
    cs._scan_and_highlight(4)
    # Dirt is not a valid target for Reinforced Wall
    assert (4, 4, 4) not in cs._highlighted_positions
    # Material still held
    assert ms.held_materials == {VOXEL_IRON_INGOT: 1}


def test_treasure():
    """Gold ingot on stone -> treasure."""
    bus, grid, ms, cs, gs = _setup()
    grid.grid[4, 4, 4] = VOXEL_STONE
    ms.held_materials = {VOXEL_GOLD_INGOT: 1}

    successes = _craft(bus, cs, "Treasure", 4, 4, 4)

    assert len(successes) == 1
    assert successes[0]["recipe"] == "Treasure"
    assert grid.get(4, 4, 4) == VOXEL_TREASURE


def test_spike_trap():
    """Iron ingot on air with solid below -> spike with state=1 (extended)."""
    bus, grid, ms, cs, gs = _setup()
    grid.grid[4, 4, 5] = VOXEL_STONE  # solid below
    ms.held_materials = {VOXEL_IRON_INGOT: 1}

    successes = _craft(bus, cs, "Spike Trap", 4, 4, 4)

    assert len(successes) == 1
    assert successes[0]["recipe"] == "Spike Trap"
    assert grid.get(4, 4, 4) == VOXEL_SPIKE
    assert grid.get_block_state(4, 4, 4) == 1  # extended


def test_spike_trap_no_floor():
    """Iron ingot on air, air below -> not in highlights."""
    bus, grid, ms, cs, gs = _setup()
    # No solid below (4,4,5) is air
    ms.held_materials = {VOXEL_IRON_INGOT: 1}

    bus.publish("craft_recipe_selected", recipe_name="Spike Trap")
    cs._current_z = 4
    cs._scan_and_highlight(4)

    assert (4, 4, 4) not in cs._highlighted_positions


def test_door():
    """Enchanted metal on air between two opposite walls -> door (closed)."""
    bus, grid, ms, cs, gs = _setup()
    grid.grid[3, 4, 4] = VOXEL_STONE  # -X wall
    grid.grid[5, 4, 4] = VOXEL_STONE  # +X wall
    ms.held_materials = {VOXEL_ENCHANTED_METAL: 1}

    successes = _craft(bus, cs, "Door", 4, 4, 4)

    assert len(successes) == 1
    assert successes[0]["recipe"] == "Door"
    assert grid.get(4, 4, 4) == VOXEL_DOOR
    assert grid.get_block_state(4, 4, 4) == 1  # closed


def test_door_no_opposite_walls():
    """Enchanted metal with only one wall -> not in highlights."""
    bus, grid, ms, cs, gs = _setup()
    grid.grid[3, 4, 4] = VOXEL_STONE  # only -X wall, no +X wall
    ms.held_materials = {VOXEL_ENCHANTED_METAL: 1}

    bus.publish("craft_recipe_selected", recipe_name="Door")
    cs._current_z = 4
    cs._scan_and_highlight(4)

    assert (4, 4, 4) not in cs._highlighted_positions


def test_tarp():
    """Dirt on air between two opposite walls -> tarp."""
    bus, grid, ms, cs, gs = _setup()
    grid.grid[4, 3, 4] = VOXEL_STONE  # -Y wall
    grid.grid[4, 5, 4] = VOXEL_STONE  # +Y wall
    ms.held_materials = {VOXEL_DIRT: 1}

    successes = _craft(bus, cs, "Tarp", 4, 4, 4)

    assert len(successes) == 1
    assert successes[0]["recipe"] == "Tarp"
    assert grid.get(4, 4, 4) == VOXEL_TARP


def test_tarp_one_wall():
    """Dirt on air with only one wall -> not in highlights."""
    bus, grid, ms, cs, gs = _setup()
    grid.grid[4, 3, 4] = VOXEL_STONE  # only -Y wall
    ms.held_materials = {VOXEL_DIRT: 1}

    bus.publish("craft_recipe_selected", recipe_name="Tarp")
    cs._current_z = 4
    cs._scan_and_highlight(4)

    assert (4, 4, 4) not in cs._highlighted_positions


def test_slope():
    """Stone on air with solid below and one solid side -> slope.

    Block above is solid to prevent stairs from also matching.
    """
    bus, grid, ms, cs, gs = _setup()
    grid.grid[4, 4, 5] = VOXEL_STONE  # solid below
    grid.grid[3, 4, 4] = VOXEL_STONE  # one solid side (-X)
    grid.grid[4, 4, 3] = VOXEL_STONE  # solid above (blocks stairs, slope-only)
    ms.held_materials = {VOXEL_STONE: 1}

    successes = _craft(bus, cs, "Slope", 4, 4, 4)

    assert len(successes) == 1
    assert successes[0]["recipe"] == "Slope"
    assert grid.get(4, 4, 4) == VOXEL_SLOPE


def test_stairs():
    """Stone on air with solid below, one side, air above -> stairs via explicit recipe selection."""
    bus, grid, ms, cs, gs = _setup()
    grid.grid[4, 4, 5] = VOXEL_STONE  # solid below
    grid.grid[3, 4, 4] = VOXEL_STONE  # one solid side (-X)
    # z=3 is above z=4 (shallower), ensure it's air (default)
    ms.held_materials = {VOXEL_STONE: 1}

    # With the new system, the player explicitly selects "Stairs" from the panel
    successes = _craft(bus, cs, "Stairs", 4, 4, 4)

    assert len(successes) == 1
    assert successes[0]["recipe"] == "Stairs"
    assert grid.get(4, 4, 4) == VOXEL_STAIRS


def test_stairs_blocked_above():
    """Stone on air with solid below and side but solid above -> only slope valid (not stairs)."""
    bus, grid, ms, cs, gs = _setup()
    grid.grid[4, 4, 5] = VOXEL_STONE  # solid below
    grid.grid[3, 4, 4] = VOXEL_STONE  # one solid side (-X)
    grid.grid[4, 4, 3] = VOXEL_STONE  # solid above -> blocks stairs

    ms.held_materials = {VOXEL_STONE: 1}

    # Stairs should not be valid at this position
    bus.publish("craft_recipe_selected", recipe_name="Stairs")
    cs._current_z = 4
    cs._scan_and_highlight(4)

    assert (4, 4, 4) not in cs._highlighted_positions

    # But slope works
    bus.publish("craft_cancel")
    successes = _craft(bus, cs, "Slope", 4, 4, 4)

    assert len(successes) == 1
    assert successes[0]["recipe"] == "Slope"
    assert grid.get(4, 4, 4) == VOXEL_SLOPE


def test_rolling_stone():
    """Granite on air above slope -> rolling stone (loose)."""
    bus, grid, ms, cs, gs = _setup()
    grid.grid[4, 4, 5] = VOXEL_SLOPE  # slope below
    ms.held_materials = {VOXEL_GRANITE: 1}

    successes = _craft(bus, cs, "Rolling Stone", 4, 4, 4)

    assert len(successes) == 1
    assert successes[0]["recipe"] == "Rolling Stone"
    assert grid.get(4, 4, 4) == VOXEL_ROLLING_STONE
    assert grid.is_loose(4, 4, 4)


def test_rolling_stone_above_stairs():
    """Granite on air above stairs -> rolling stone."""
    bus, grid, ms, cs, gs = _setup()
    grid.grid[4, 4, 5] = VOXEL_STAIRS  # stairs below
    ms.held_materials = {VOXEL_GRANITE: 1}

    successes = _craft(bus, cs, "Rolling Stone", 4, 4, 4)

    assert len(successes) == 1
    assert grid.get(4, 4, 4) == VOXEL_ROLLING_STONE


def test_rolling_stone_no_slope():
    """Granite on air above regular stone -> granite pillar, NOT rolling stone."""
    bus, grid, ms, cs, gs = _setup()
    grid.grid[4, 4, 5] = VOXEL_STONE  # regular stone below (not slope/stairs)
    ms.held_materials = {VOXEL_GRANITE: 1}

    # Rolling Stone requires slope/stairs below — should not be in highlights
    bus.publish("craft_recipe_selected", recipe_name="Rolling Stone")
    cs._current_z = 4
    cs._scan_and_highlight(4)
    assert (4, 4, 4) not in cs._highlighted_positions

    # But Granite Pillar matches
    bus.publish("craft_cancel")
    successes = _craft(bus, cs, "Granite Pillar", 4, 4, 4)
    assert len(successes) == 1
    assert successes[0]["recipe"] == "Granite Pillar"
    assert grid.get(4, 4, 4) == VOXEL_GRANITE


# ---------------------------------------------------------------------------
# Highlight-mode state machine tests
# ---------------------------------------------------------------------------


def test_recipe_selection_enters_craft_mode():
    """Selecting a recipe enters craft mode with highlights."""
    bus, grid, ms, cs, gs = _setup()
    grid.grid[4, 4, 4] = VOXEL_STONE
    ms.held_materials = {VOXEL_IRON_INGOT: 1}

    mode_events = []
    highlight_events = []
    bus.subscribe("craft_mode_entered", lambda **kw: mode_events.append(kw))
    bus.subscribe("craft_highlights_updated", lambda **kw: highlight_events.append(kw))

    cs._current_z = 4
    bus.publish("craft_recipe_selected", recipe_name="Reinforced Wall")

    assert gs.craft_mode_active is True
    assert cs.is_craft_mode_active is True
    assert len(mode_events) == 1
    assert mode_events[0]["recipe_name"] == "Reinforced Wall"
    assert len(highlight_events) >= 1


def test_craft_cancel_exits_mode():
    """Cancelling clears highlights and exits craft mode."""
    bus, grid, ms, cs, gs = _setup()
    grid.grid[4, 4, 4] = VOXEL_STONE
    ms.held_materials = {VOXEL_IRON_INGOT: 1}

    cs._current_z = 4
    bus.publish("craft_recipe_selected", recipe_name="Reinforced Wall")
    assert gs.craft_mode_active is True

    exit_events = []
    clear_events = []
    bus.subscribe("craft_mode_exited", lambda **kw: exit_events.append(kw))
    bus.subscribe("craft_highlights_cleared", lambda **kw: clear_events.append(kw))

    bus.publish("craft_cancel")

    assert gs.craft_mode_active is False
    assert cs.is_craft_mode_active is False
    assert len(exit_events) == 1
    assert len(clear_events) == 1


def test_click_non_highlighted_cancels():
    """Clicking a non-highlighted position cancels craft mode."""
    bus, grid, ms, cs, gs = _setup()
    grid.grid[4, 4, 4] = VOXEL_STONE
    ms.held_materials = {VOXEL_IRON_INGOT: 1}

    cs._current_z = 4
    bus.publish("craft_recipe_selected", recipe_name="Reinforced Wall")
    assert gs.craft_mode_active is True

    # Click a position that's NOT highlighted (0,0,0) is air, not a valid target
    bus.publish("craft_at_position", x=0, y=0, z=0)

    assert gs.craft_mode_active is False


def test_material_depleted_exits_craft_mode():
    """When material runs out after crafting, craft mode exits automatically."""
    bus, grid, ms, cs, gs = _setup()
    grid.grid[4, 4, 4] = VOXEL_STONE
    ms.held_materials = {VOXEL_IRON_INGOT: 1}  # exactly 1

    cs._current_z = 4
    bus.publish("craft_recipe_selected", recipe_name="Reinforced Wall")
    bus.publish("craft_at_position", x=4, y=4, z=4)

    # Should have exited craft mode since iron is depleted
    assert gs.craft_mode_active is False
    assert not ms.has_material(VOXEL_IRON_INGOT)


def test_toggle_same_recipe_exits():
    """Selecting the same recipe again exits craft mode."""
    bus, grid, ms, cs, gs = _setup()
    grid.grid[4, 4, 4] = VOXEL_STONE
    ms.held_materials = {VOXEL_IRON_INGOT: 1}

    cs._current_z = 4
    bus.publish("craft_recipe_selected", recipe_name="Reinforced Wall")
    assert gs.craft_mode_active is True

    bus.publish("craft_recipe_selected", recipe_name="Reinforced Wall")
    assert gs.craft_mode_active is False


def test_no_matching_material_error():
    """Selecting a recipe without matching material produces an error."""
    bus, grid, ms, cs, gs = _setup()
    ms.held_materials = {VOXEL_STONE: 1}  # no iron ingot

    errors = []
    bus.subscribe("error_message", lambda **kw: errors.append(kw))

    bus.publish("craft_recipe_selected", recipe_name="Reinforced Wall")

    assert len(errors) == 1
    assert "material" in errors[0]["text"].lower()
    assert gs.craft_mode_active is False


def test_z_level_change_rescans():
    """Changing z-level while in craft mode rescans highlights."""
    bus, grid, ms, cs, gs = _setup()
    # Set up valid position at z=4
    grid.grid[4, 4, 4] = VOXEL_STONE
    # Set up valid position at z=2
    grid.grid[4, 4, 2] = VOXEL_STONE
    ms.held_materials = {VOXEL_IRON_INGOT: 2}

    cs._current_z = 4
    bus.publish("craft_recipe_selected", recipe_name="Reinforced Wall")

    # Change z-level
    highlight_events = []
    bus.subscribe("craft_highlights_updated", lambda **kw: highlight_events.append(kw))

    bus.publish("z_level_changed", z=2)

    assert len(highlight_events) >= 1
    # Should have scanned z=2 and found the stone there
    assert (4, 4, 2) in cs._highlighted_positions


def test_multiple_crafts_in_sequence():
    """Can craft multiple times in a row from the same recipe selection."""
    bus, grid, ms, cs, gs = _setup()
    # Two stone blocks
    grid.grid[4, 4, 4] = VOXEL_STONE
    grid.grid[5, 5, 4] = VOXEL_STONE
    ms.held_materials = {VOXEL_IRON_INGOT: 2}

    successes = []
    bus.subscribe("craft_success", lambda **kw: successes.append(kw))

    cs._current_z = 4
    bus.publish("craft_recipe_selected", recipe_name="Reinforced Wall")

    # Craft at first position
    bus.publish("craft_at_position", x=4, y=4, z=4)
    assert len(successes) == 1
    assert grid.get(4, 4, 4) == VOXEL_REINFORCED_WALL

    # Still in craft mode with 1 iron remaining
    assert gs.craft_mode_active is True
    assert ms.held_materials == {VOXEL_IRON_INGOT: 1}

    # Craft at second position
    bus.publish("craft_at_position", x=5, y=5, z=4)
    assert len(successes) == 2
    assert grid.get(5, 5, 4) == VOXEL_REINFORCED_WALL

    # Now depleted — should auto-exit craft mode
    assert gs.craft_mode_active is False


def test_find_valid_positions_basic():
    """CraftingBook.find_valid_positions scans z-level for valid spots."""
    bus, grid, ms, cs, gs = _setup()
    grid.grid[2, 2, 4] = VOXEL_STONE
    grid.grid[6, 6, 4] = VOXEL_STONE

    recipe = cs.crafting_book.get_recipe_by_name("Reinforced Wall")
    positions = cs.crafting_book.find_valid_positions(
        recipe, grid, VOXEL_IRON_INGOT, z_level=4
    )

    assert (2, 2, 4) in positions
    assert (6, 6, 4) in positions


def test_craft_success_event_has_position():
    """craft_success event includes the position of the craft."""
    bus, grid, ms, cs, gs = _setup()
    grid.grid[4, 4, 4] = VOXEL_STONE
    ms.held_materials = {VOXEL_IRON_INGOT: 1}

    successes = _craft(bus, cs, "Reinforced Wall", 4, 4, 4)

    assert successes[0]["x"] == 4
    assert successes[0]["y"] == 4
    assert successes[0]["z"] == 4


def test_drop_no_longer_triggers_crafting():
    """MoveSystem.drop on air places as loose — no crafting."""
    bus, grid, ms, cs, gs = _setup()
    grid.grid[4, 4, 5] = VOXEL_STONE  # solid below for spike
    ms.held_materials = {VOXEL_IRON_INGOT: 2}

    # Drop should place as loose, NOT craft a spike
    result = ms.drop(4, 4, 4)

    assert result is True
    assert grid.get(4, 4, 4) == VOXEL_IRON_INGOT
    assert grid.is_loose(4, 4, 4)
    assert ms.held_materials == {VOXEL_IRON_INGOT: 1}


def test_no_recipe_drop_places_loose():
    """MoveSystem.drop on air with no matching recipe places as loose."""
    bus, grid, ms, cs, gs = _setup()
    # No special conditions — just bare air, copper ingot has no air-target recipe
    ms.held_materials = {VOXEL_COPPER_INGOT: 1}

    result = ms.drop(4, 4, 4)

    assert result is True
    assert grid.get(4, 4, 4) == VOXEL_COPPER_INGOT
    assert grid.is_loose(4, 4, 4)
    assert ms.held_materials == {}


def test_required_inputs_on_recipes():
    """All recipes have required_inputs set."""
    book = cs_book()
    for recipe in book.recipes:
        assert isinstance(recipe.required_inputs, frozenset)
        assert len(recipe.required_inputs) > 0, f"Recipe {recipe.name} has empty required_inputs"


def cs_book():
    """Helper to get a CraftingBook instance."""
    from dungeon_builder.building.crafting_book import CraftingBook
    return CraftingBook()
