"""Tests for 10 new functional block recipes and metal_type on crafted blocks."""

import pytest

from dungeon_builder.core.event_bus import EventBus
from dungeon_builder.core.game_state import GameState
from dungeon_builder.world.voxel_grid import VoxelGrid
from dungeon_builder.building.move_system import MoveSystem
from dungeon_builder.building.crafting_system import CraftingSystem
from dungeon_builder.config import (
    VOXEL_AIR,
    VOXEL_STONE,
    VOXEL_LAVA,
    VOXEL_WATER,
    VOXEL_CHALK,
    VOXEL_GRANITE,
    VOXEL_OBSIDIAN,
    VOXEL_IRON_INGOT,
    VOXEL_COPPER_INGOT,
    VOXEL_GOLD_INGOT,
    VOXEL_ENCHANTED_METAL,
    VOXEL_REINFORCED_WALL,
    VOXEL_SPIKE,
    VOXEL_DOOR,
    VOXEL_GOLD_BAIT,
    VOXEL_HEAT_BEACON,
    VOXEL_PRESSURE_PLATE,
    VOXEL_IRON_BARS,
    VOXEL_FLOODGATE,
    VOXEL_ALARM_BELL,
    VOXEL_FRAGILE_FLOOR,
    VOXEL_PIPE,
    VOXEL_PUMP,
    VOXEL_STEAM_VENT,
    METAL_NONE,
    METAL_IRON,
    METAL_COPPER,
    METAL_GOLD,
    METAL_ENCH_IRON,
    METAL_ENCH_COPPER,
    METAL_ENCH_GOLD,
    ENCHANTED_OFFSET,
    DEFAULT_SEED,
)


# ---------------------------------------------------------------------------
# Setup helper
# ---------------------------------------------------------------------------

def _setup(width=10, depth=10, height=10):
    bus = EventBus()
    grid = VoxelGrid(width=width, depth=depth, height=height)
    grid.visible[:] = True
    grid.claimed[:] = True
    gs = GameState(DEFAULT_SEED)
    gs.event_bus = bus
    ms = MoveSystem(bus, grid, gs)
    cs = CraftingSystem(bus, grid, ms, gs)
    gs.move_system = ms
    return bus, grid, ms, cs


def _craft(bus, cs, recipe_name, x, y, z, ms, held_materials, held_metal_types=None):
    """Helper: set held materials, select recipe, craft at position."""
    ms.held_materials = held_materials.copy()
    if held_metal_types:
        ms.held_metal_types = held_metal_types.copy()
    cs._current_z = z
    bus.publish("craft_recipe_selected", recipe_name=recipe_name)
    bus.publish("craft_at_position", x=x, y=y, z=z)


# ---------------------------------------------------------------------------
# Existing recipes now set metal_type
# ---------------------------------------------------------------------------

class TestExistingRecipesMetalType:
    def test_reinforced_wall_iron(self):
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_STONE
        _craft(bus, cs, "Reinforced Wall", 4, 4, 4, ms,
               {VOXEL_IRON_INGOT: 1}, {VOXEL_IRON_INGOT: METAL_IRON})
        assert grid.get(4, 4, 4) == VOXEL_REINFORCED_WALL
        assert grid.get_metal_type(4, 4, 4) == METAL_IRON

    def test_reinforced_wall_copper(self):
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_STONE
        _craft(bus, cs, "Reinforced Wall", 4, 4, 4, ms,
               {VOXEL_COPPER_INGOT: 1}, {VOXEL_COPPER_INGOT: METAL_COPPER})
        assert grid.get(4, 4, 4) == VOXEL_REINFORCED_WALL
        assert grid.get_metal_type(4, 4, 4) == METAL_COPPER

    def test_reinforced_wall_gold(self):
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_STONE
        _craft(bus, cs, "Reinforced Wall", 4, 4, 4, ms,
               {VOXEL_GOLD_INGOT: 1}, {VOXEL_GOLD_INGOT: METAL_GOLD})
        assert grid.get(4, 4, 4) == VOXEL_REINFORCED_WALL
        assert grid.get_metal_type(4, 4, 4) == METAL_GOLD

    def test_spike_trap_metal_type(self):
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_AIR
        grid.grid[4, 4, 5] = VOXEL_STONE  # solid below
        _craft(bus, cs, "Spike Trap", 4, 4, 4, ms,
               {VOXEL_COPPER_INGOT: 1}, {VOXEL_COPPER_INGOT: METAL_COPPER})
        assert grid.get(4, 4, 4) == VOXEL_SPIKE
        assert grid.get_metal_type(4, 4, 4) == METAL_COPPER

    def test_door_metal_type(self):
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_AIR
        grid.grid[3, 4, 4] = VOXEL_STONE
        grid.grid[5, 4, 4] = VOXEL_STONE
        _craft(bus, cs, "Door", 4, 4, 4, ms,
               {VOXEL_ENCHANTED_METAL: 1}, {VOXEL_ENCHANTED_METAL: METAL_ENCH_COPPER})
        assert grid.get(4, 4, 4) == VOXEL_DOOR
        assert grid.get_metal_type(4, 4, 4) == METAL_ENCH_COPPER


# ---------------------------------------------------------------------------
# Gold Bait
# ---------------------------------------------------------------------------

class TestGoldBait:
    def test_craft_gold_bait(self):
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_AIR
        grid.grid[4, 4, 5] = VOXEL_STONE  # solid below
        grid.grid[5, 4, 4] = VOXEL_STONE  # wall
        _craft(bus, cs, "Gold Bait", 4, 4, 4, ms,
               {VOXEL_ENCHANTED_METAL: 1}, {VOXEL_ENCHANTED_METAL: METAL_ENCH_GOLD})
        assert grid.get(4, 4, 4) == VOXEL_GOLD_BAIT
        assert grid.get_metal_type(4, 4, 4) == METAL_ENCH_GOLD

    def test_gold_bait_requires_gold(self):
        """Gold bait fails if held enchanted metal is not gold."""
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_AIR
        grid.grid[4, 4, 5] = VOXEL_STONE
        grid.grid[5, 4, 4] = VOXEL_STONE
        _craft(bus, cs, "Gold Bait", 4, 4, 4, ms,
               {VOXEL_ENCHANTED_METAL: 1}, {VOXEL_ENCHANTED_METAL: METAL_ENCH_IRON})
        # Craft should fail (iron is not gold)
        assert grid.get(4, 4, 4) == VOXEL_AIR

    def test_gold_bait_needs_floor(self):
        """Gold bait on air without solid below fails."""
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_AIR
        grid.grid[4, 4, 5] = VOXEL_AIR  # no floor
        grid.grid[5, 4, 4] = VOXEL_STONE
        ms.held_materials = {VOXEL_ENCHANTED_METAL: 1}
        ms.held_metal_types = {VOXEL_ENCHANTED_METAL: METAL_ENCH_GOLD}
        cs._current_z = 4
        bus.publish("craft_recipe_selected", recipe_name="Gold Bait")
        # Position should not be highlighted — check highlighted_positions
        assert (4, 4, 4) not in cs._highlighted_positions

    def test_gold_bait_needs_wall(self):
        """Gold bait on air without lateral support fails."""
        bus, grid, ms, cs = _setup()
        # All air around
        grid.grid[:] = VOXEL_AIR
        grid.grid[4, 4, 5] = VOXEL_STONE  # floor only
        ms.held_materials = {VOXEL_ENCHANTED_METAL: 1}
        ms.held_metal_types = {VOXEL_ENCHANTED_METAL: METAL_ENCH_GOLD}
        cs._current_z = 4
        bus.publish("craft_recipe_selected", recipe_name="Gold Bait")
        assert (4, 4, 4) not in cs._highlighted_positions


# ---------------------------------------------------------------------------
# Heat Beacon
# ---------------------------------------------------------------------------

class TestHeatBeacon:
    def test_craft_heat_beacon(self):
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_STONE
        grid.temperature[4, 4, 4] = 300.0
        _craft(bus, cs, "Heat Beacon", 4, 4, 4, ms,
               {VOXEL_COPPER_INGOT: 1}, {VOXEL_COPPER_INGOT: METAL_COPPER})
        assert grid.get(4, 4, 4) == VOXEL_HEAT_BEACON
        assert grid.get_metal_type(4, 4, 4) == METAL_COPPER

    def test_heat_beacon_requires_temperature(self):
        """Heat beacon fails if stone temperature <= 200."""
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_STONE
        grid.temperature[4, 4, 4] = 100.0  # too cold
        ms.held_materials = {VOXEL_COPPER_INGOT: 1}
        cs._current_z = 4
        bus.publish("craft_recipe_selected", recipe_name="Heat Beacon")
        assert (4, 4, 4) not in cs._highlighted_positions

    def test_heat_beacon_requires_copper(self):
        """Heat beacon only accepts copper ingot."""
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_STONE
        grid.temperature[4, 4, 4] = 300.0
        ms.held_materials = {VOXEL_IRON_INGOT: 1}
        cs._current_z = 4
        bus.publish("craft_recipe_selected", recipe_name="Heat Beacon")
        # Iron ingot not in required_inputs for heat beacon
        # This should fail at recipe selection (no matching held type)
        assert cs._active_recipe is None or cs._active_recipe.name != "Heat Beacon"


# ---------------------------------------------------------------------------
# Pressure Plate
# ---------------------------------------------------------------------------

class TestPressurePlate:
    def test_craft_pressure_plate(self):
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_STONE
        grid.grid[4, 4, 3] = VOXEL_AIR  # air above
        _craft(bus, cs, "Pressure Plate", 4, 4, 4, ms,
               {VOXEL_ENCHANTED_METAL: 1}, {VOXEL_ENCHANTED_METAL: METAL_ENCH_IRON})
        assert grid.get(4, 4, 4) == VOXEL_PRESSURE_PLATE
        assert grid.get_metal_type(4, 4, 4) == METAL_ENCH_IRON

    def test_pressure_plate_needs_air_above(self):
        """Pressure plate fails if no air above."""
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_STONE
        grid.grid[4, 4, 3] = VOXEL_STONE  # solid above
        ms.held_materials = {VOXEL_ENCHANTED_METAL: 1}
        cs._current_z = 4
        bus.publish("craft_recipe_selected", recipe_name="Pressure Plate")
        assert (4, 4, 4) not in cs._highlighted_positions


# ---------------------------------------------------------------------------
# Iron Bars
# ---------------------------------------------------------------------------

class TestIronBars:
    def test_craft_iron_bars(self):
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_AIR
        grid.grid[3, 4, 4] = VOXEL_STONE
        grid.grid[5, 4, 4] = VOXEL_STONE
        _craft(bus, cs, "Iron Bars", 4, 4, 4, ms,
               {VOXEL_ENCHANTED_METAL: 1}, {VOXEL_ENCHANTED_METAL: METAL_ENCH_GOLD})
        assert grid.get(4, 4, 4) == VOXEL_IRON_BARS
        assert grid.get_metal_type(4, 4, 4) == METAL_ENCH_GOLD

    def test_iron_bars_needs_opposite_walls(self):
        """Iron bars fail without 2 opposite walls."""
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_AIR
        grid.grid[3, 4, 4] = VOXEL_STONE  # only one wall
        ms.held_materials = {VOXEL_ENCHANTED_METAL: 1}
        cs._current_z = 4
        bus.publish("craft_recipe_selected", recipe_name="Iron Bars")
        assert (4, 4, 4) not in cs._highlighted_positions


# ---------------------------------------------------------------------------
# Floodgate
# ---------------------------------------------------------------------------

class TestFloodgate:
    def test_craft_floodgate(self):
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_STONE
        grid.grid[5, 4, 4] = VOXEL_WATER  # adjacent water
        _craft(bus, cs, "Floodgate", 4, 4, 4, ms,
               {VOXEL_ENCHANTED_METAL: 1}, {VOXEL_ENCHANTED_METAL: METAL_ENCH_IRON})
        assert grid.get(4, 4, 4) == VOXEL_FLOODGATE
        assert grid.get_metal_type(4, 4, 4) == METAL_ENCH_IRON
        assert grid.get_block_state(4, 4, 4) == 1  # starts closed

    def test_floodgate_needs_adjacent_water(self):
        """Floodgate fails without adjacent water."""
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_STONE
        # No water nearby
        ms.held_materials = {VOXEL_ENCHANTED_METAL: 1}
        cs._current_z = 4
        bus.publish("craft_recipe_selected", recipe_name="Floodgate")
        assert (4, 4, 4) not in cs._highlighted_positions

    def test_floodgate_detects_water_level(self):
        """Floodgate recognizes water_level > 0 as adjacent water."""
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_STONE
        grid.grid[5, 4, 4] = VOXEL_AIR
        grid.water_level[5, 4, 4] = 100  # water level but not VOXEL_WATER
        _craft(bus, cs, "Floodgate", 4, 4, 4, ms,
               {VOXEL_ENCHANTED_METAL: 1}, {VOXEL_ENCHANTED_METAL: METAL_ENCH_COPPER})
        assert grid.get(4, 4, 4) == VOXEL_FLOODGATE


# ---------------------------------------------------------------------------
# Alarm Bell
# ---------------------------------------------------------------------------

class TestAlarmBell:
    def test_craft_alarm_bell(self):
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_AIR
        grid.grid[4, 4, 5] = VOXEL_STONE  # solid below
        grid.grid[5, 4, 4] = VOXEL_STONE  # wall
        _craft(bus, cs, "Alarm Bell", 4, 4, 4, ms,
               {VOXEL_ENCHANTED_METAL: 1}, {VOXEL_ENCHANTED_METAL: METAL_ENCH_COPPER})
        assert grid.get(4, 4, 4) == VOXEL_ALARM_BELL
        assert grid.get_metal_type(4, 4, 4) == METAL_ENCH_COPPER

    def test_alarm_bell_needs_floor_and_wall(self):
        """Alarm bell fails without solid below + wall."""
        bus, grid, ms, cs = _setup()
        grid.grid[:] = VOXEL_AIR
        grid.grid[4, 4, 5] = VOXEL_STONE  # floor only, no walls
        ms.held_materials = {VOXEL_ENCHANTED_METAL: 1}
        cs._current_z = 4
        bus.publish("craft_recipe_selected", recipe_name="Alarm Bell")
        assert (4, 4, 4) not in cs._highlighted_positions


# ---------------------------------------------------------------------------
# Fragile Floor
# ---------------------------------------------------------------------------

class TestFragileFloor:
    def test_craft_fragile_floor(self):
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_AIR
        grid.grid[4, 4, 5] = VOXEL_STONE  # solid below
        grid.grid[3, 4, 4] = VOXEL_STONE  # wall 1
        grid.grid[5, 4, 4] = VOXEL_STONE  # wall 2
        _craft(bus, cs, "Fragile Floor", 4, 4, 4, ms, {VOXEL_CHALK: 1})
        assert grid.get(4, 4, 4) == VOXEL_FRAGILE_FLOOR
        assert grid.get_metal_type(4, 4, 4) == METAL_NONE  # non-metallic

    def test_fragile_floor_needs_two_walls(self):
        """Fragile floor fails with fewer than 2 solid sides."""
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_AIR
        grid.grid[4, 4, 5] = VOXEL_STONE
        grid.grid[3, 4, 4] = VOXEL_STONE  # only 1 wall
        ms.held_materials = {VOXEL_CHALK: 1}
        cs._current_z = 4
        bus.publish("craft_recipe_selected", recipe_name="Fragile Floor")
        assert (4, 4, 4) not in cs._highlighted_positions

    def test_fragile_floor_needs_solid_below(self):
        """Fragile floor fails without solid below."""
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_AIR
        grid.grid[4, 4, 5] = VOXEL_AIR  # no floor
        grid.grid[3, 4, 4] = VOXEL_STONE
        grid.grid[5, 4, 4] = VOXEL_STONE
        ms.held_materials = {VOXEL_CHALK: 1}
        cs._current_z = 4
        bus.publish("craft_recipe_selected", recipe_name="Fragile Floor")
        assert (4, 4, 4) not in cs._highlighted_positions


# ---------------------------------------------------------------------------
# Pipe
# ---------------------------------------------------------------------------

class TestPipe:
    def test_craft_pipe_in_air(self):
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_AIR
        grid.grid[5, 4, 4] = VOXEL_STONE  # solid neighbor
        _craft(bus, cs, "Pipe", 4, 4, 4, ms,
               {VOXEL_COPPER_INGOT: 1}, {VOXEL_COPPER_INGOT: METAL_COPPER})
        assert grid.get(4, 4, 4) == VOXEL_PIPE
        assert grid.get_metal_type(4, 4, 4) == METAL_COPPER

    def test_craft_pipe_in_stone(self):
        """Pipes can be built into non-loose stone."""
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_STONE
        grid.loose[4, 4, 4] = False
        _craft(bus, cs, "Pipe", 4, 4, 4, ms,
               {VOXEL_IRON_INGOT: 1}, {VOXEL_IRON_INGOT: METAL_IRON})
        assert grid.get(4, 4, 4) == VOXEL_PIPE
        assert grid.get_metal_type(4, 4, 4) == METAL_IRON

    def test_pipe_in_air_needs_solid_neighbor(self):
        """Pipe in air fails without any solid neighbor."""
        bus, grid, ms, cs = _setup()
        grid.grid[:] = VOXEL_AIR  # all air
        ms.held_materials = {VOXEL_IRON_INGOT: 1}
        cs._current_z = 4
        bus.publish("craft_recipe_selected", recipe_name="Pipe")
        assert (4, 4, 4) not in cs._highlighted_positions

    def test_pipe_cannot_build_into_loose_stone(self):
        """Pipe can't be built into loose stone."""
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_STONE
        grid.loose[4, 4, 4] = True  # loose!
        ms.held_materials = {VOXEL_IRON_INGOT: 1}
        cs._current_z = 4
        bus.publish("craft_recipe_selected", recipe_name="Pipe")
        assert (4, 4, 4) not in cs._highlighted_positions


# ---------------------------------------------------------------------------
# Pump
# ---------------------------------------------------------------------------

class TestPump:
    def test_craft_pump_on_pipe(self):
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_PIPE
        _craft(bus, cs, "Pump", 4, 4, 4, ms,
               {VOXEL_IRON_INGOT: 1}, {VOXEL_IRON_INGOT: METAL_IRON})
        assert grid.get(4, 4, 4) == VOXEL_PUMP
        assert grid.get_metal_type(4, 4, 4) == METAL_IRON
        assert grid.get_block_state(4, 4, 4) == 0  # default direction

    def test_pump_requires_pipe_target(self):
        """Pump fails on non-pipe targets."""
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_STONE  # not a pipe
        ms.held_materials = {VOXEL_IRON_INGOT: 1}
        cs._current_z = 4
        bus.publish("craft_recipe_selected", recipe_name="Pump")
        assert (4, 4, 4) not in cs._highlighted_positions

    def test_pump_direction_cycle(self):
        """Left-clicking a pump cycles its direction 0-5."""
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_PUMP
        grid.loose[4, 4, 4] = False
        grid.set_block_state(4, 4, 4, 0)

        events = []
        bus.subscribe("pump_direction_changed", lambda **kw: events.append(kw))

        for expected_dir in [1, 2, 3, 4, 5, 0]:
            bus.publish("voxel_left_clicked", x=4, y=4, z=4, mode="move")
            assert grid.get_block_state(4, 4, 4) == expected_dir
        assert len(events) == 6


# ---------------------------------------------------------------------------
# Steam Vent
# ---------------------------------------------------------------------------

class TestSteamVent:
    def test_craft_steam_vent(self):
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_AIR
        grid.grid[4, 4, 5] = VOXEL_LAVA  # lava 1 below
        _craft(bus, cs, "Steam Vent", 4, 4, 4, ms, {VOXEL_OBSIDIAN: 1})
        assert grid.get(4, 4, 4) == VOXEL_STEAM_VENT
        assert grid.get_metal_type(4, 4, 4) == METAL_NONE

    def test_steam_vent_lava_2_deep(self):
        """Steam vent works with lava 2 cells below."""
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_AIR
        grid.grid[4, 4, 5] = VOXEL_STONE  # not lava at depth 1
        grid.grid[4, 4, 6] = VOXEL_LAVA   # lava at depth 2
        _craft(bus, cs, "Steam Vent", 4, 4, 4, ms, {VOXEL_OBSIDIAN: 1})
        assert grid.get(4, 4, 4) == VOXEL_STEAM_VENT

    def test_steam_vent_no_lava(self):
        """Steam vent fails without lava below."""
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_AIR
        grid.grid[4, 4, 5] = VOXEL_STONE
        grid.grid[4, 4, 6] = VOXEL_STONE
        ms.held_materials = {VOXEL_OBSIDIAN: 1}
        cs._current_z = 4
        bus.publish("craft_recipe_selected", recipe_name="Steam Vent")
        assert (4, 4, 4) not in cs._highlighted_positions

    def test_steam_vent_lava_too_deep(self):
        """Steam vent fails if lava is 3+ cells below (only checks 2)."""
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_AIR
        grid.grid[4, 4, 5] = VOXEL_STONE
        grid.grid[4, 4, 6] = VOXEL_STONE
        grid.grid[4, 4, 7] = VOXEL_LAVA  # too deep
        ms.held_materials = {VOXEL_OBSIDIAN: 1}
        cs._current_z = 4
        bus.publish("craft_recipe_selected", recipe_name="Steam Vent")
        assert (4, 4, 4) not in cs._highlighted_positions


# ---------------------------------------------------------------------------
# Metal_type fallback when held_metal_types not set
# ---------------------------------------------------------------------------

class TestMetalTypeFallback:
    def test_reinforced_wall_fallback_from_held_type(self):
        """When no held_metal_types, metal_type derived from held voxel type."""
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_STONE
        # Don't set held_metal_types — test fallback
        ms.held_materials = {VOXEL_GOLD_INGOT: 1}
        cs._current_z = 4
        bus.publish("craft_recipe_selected", recipe_name="Reinforced Wall")
        bus.publish("craft_at_position", x=4, y=4, z=4)
        assert grid.get(4, 4, 4) == VOXEL_REINFORCED_WALL
        assert grid.get_metal_type(4, 4, 4) == METAL_GOLD

    def test_pipe_fallback_from_held_type(self):
        """Pipe with no held_metal_types uses HELD_TO_METAL fallback."""
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_AIR
        grid.grid[5, 4, 4] = VOXEL_STONE
        ms.held_materials = {VOXEL_COPPER_INGOT: 1}
        cs._current_z = 4
        bus.publish("craft_recipe_selected", recipe_name="Pipe")
        bus.publish("craft_at_position", x=4, y=4, z=4)
        assert grid.get(4, 4, 4) == VOXEL_PIPE
        assert grid.get_metal_type(4, 4, 4) == METAL_COPPER
