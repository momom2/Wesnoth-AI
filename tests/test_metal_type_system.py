"""Tests for the metal_type system: array, helpers, smelting, infusion, pick_up/drop."""

import numpy as np
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
    VOXEL_IRON_ORE,
    VOXEL_COPPER_ORE,
    VOXEL_GOLD_ORE,
    VOXEL_MANA_CRYSTAL,
    VOXEL_IRON_INGOT,
    VOXEL_COPPER_INGOT,
    VOXEL_GOLD_INGOT,
    VOXEL_ENCHANTED_METAL,
    METAL_NONE,
    METAL_IRON,
    METAL_COPPER,
    METAL_GOLD,
    ENCHANTED_OFFSET,
    METAL_ENCH_IRON,
    METAL_ENCH_COPPER,
    METAL_ENCH_GOLD,
    is_enchanted_metal,
    base_metal_of,
    make_enchanted,
    METAL_MELT_TEMPERATURE,
    METAL_STRENGTH_MULT,
    METAL_GREED_APPEAL,
    METAL_COLORS,
    METAL_CONDUCTIVITY_MULT,
    DEFAULT_SEED,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup(width=8, depth=8, height=8):
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


# ---------------------------------------------------------------------------
# metal_type array on VoxelGrid
# ---------------------------------------------------------------------------

class TestMetalTypeArray:
    def test_default_zero(self):
        """All metal_types default to METAL_NONE (0)."""
        grid = VoxelGrid(width=4, depth=4, height=4)
        assert np.all(grid.metal_type == 0)

    def test_set_get(self):
        """Can set and get metal_type values."""
        grid = VoxelGrid(width=4, depth=4, height=4)
        grid.set_metal_type(1, 2, 3, METAL_COPPER)
        assert grid.get_metal_type(1, 2, 3) == METAL_COPPER

    def test_out_of_bounds_returns_zero(self):
        """Out-of-bounds get returns 0, set is no-op."""
        grid = VoxelGrid(width=4, depth=4, height=4)
        assert grid.get_metal_type(100, 0, 0) == 0
        grid.set_metal_type(100, 0, 0, METAL_IRON)  # no crash

    def test_reset_on_voxel_type_change(self):
        """metal_type resets to 0 when voxel type changes via set()."""
        bus = EventBus()
        grid = VoxelGrid(width=4, depth=4, height=4)
        grid.grid[2, 2, 2] = VOXEL_IRON_INGOT
        grid.set_metal_type(2, 2, 2, METAL_IRON)
        assert grid.get_metal_type(2, 2, 2) == METAL_IRON

        grid.set(2, 2, 2, VOXEL_STONE)
        assert grid.get_metal_type(2, 2, 2) == 0

    def test_marks_chunk_dirty(self):
        """Setting metal_type marks the chunk dirty."""
        grid = VoxelGrid(width=16, depth=16, height=4)
        grid.pop_dirty_chunks()

        grid.set_metal_type(1, 2, 3, METAL_GOLD)

        dirty = grid.pop_dirty_chunks()
        assert len(dirty) > 0


# ---------------------------------------------------------------------------
# Helper functions: is_enchanted_metal, base_metal_of, make_enchanted
# ---------------------------------------------------------------------------

class TestMetalHelpers:
    def test_is_enchanted_false_for_base(self):
        assert is_enchanted_metal(METAL_IRON) is False
        assert is_enchanted_metal(METAL_COPPER) is False
        assert is_enchanted_metal(METAL_GOLD) is False
        assert is_enchanted_metal(METAL_NONE) is False

    def test_is_enchanted_true_for_enchanted(self):
        assert is_enchanted_metal(METAL_ENCH_IRON) is True
        assert is_enchanted_metal(METAL_ENCH_COPPER) is True
        assert is_enchanted_metal(METAL_ENCH_GOLD) is True

    def test_base_metal_of_strips_enchanted(self):
        assert base_metal_of(METAL_ENCH_IRON) == METAL_IRON
        assert base_metal_of(METAL_ENCH_COPPER) == METAL_COPPER
        assert base_metal_of(METAL_ENCH_GOLD) == METAL_GOLD

    def test_base_metal_of_passes_through_base(self):
        assert base_metal_of(METAL_IRON) == METAL_IRON
        assert base_metal_of(METAL_NONE) == METAL_NONE

    def test_make_enchanted(self):
        assert make_enchanted(METAL_IRON) == METAL_ENCH_IRON
        assert make_enchanted(METAL_COPPER) == METAL_ENCH_COPPER
        assert make_enchanted(METAL_GOLD) == METAL_ENCH_GOLD

    def test_make_enchanted_idempotent(self):
        """Enchanting already-enchanted metal keeps enchanted bit."""
        assert make_enchanted(METAL_ENCH_IRON) == METAL_ENCH_IRON


# ---------------------------------------------------------------------------
# Metal property LUT lookups
# ---------------------------------------------------------------------------

class TestMetalPropertyLUTs:
    def test_melt_temperature_by_metal(self):
        assert METAL_MELT_TEMPERATURE[METAL_IRON] == 1200.0
        assert METAL_MELT_TEMPERATURE[METAL_COPPER] == 800.0
        assert METAL_MELT_TEMPERATURE[METAL_GOLD] == 600.0
        assert METAL_MELT_TEMPERATURE[METAL_NONE] == 0.0

    def test_strength_mult_by_metal(self):
        assert METAL_STRENGTH_MULT[METAL_IRON] == 1.0
        assert METAL_STRENGTH_MULT[METAL_COPPER] == 0.7
        assert METAL_STRENGTH_MULT[METAL_GOLD] == 0.4

    def test_greed_appeal_by_metal(self):
        assert METAL_GREED_APPEAL[METAL_GOLD] == 0.8
        assert METAL_GREED_APPEAL[METAL_COPPER] == 0.2
        assert METAL_GREED_APPEAL[METAL_IRON] == 0.0

    def test_colors_by_metal(self):
        assert len(METAL_COLORS[METAL_IRON]) == 3
        assert len(METAL_COLORS[METAL_COPPER]) == 3
        assert len(METAL_COLORS[METAL_GOLD]) == 3

    def test_conductivity_by_metal(self):
        # Copper is best conductor
        assert METAL_CONDUCTIVITY_MULT[METAL_COPPER] >= METAL_CONDUCTIVITY_MULT[METAL_IRON]
        assert METAL_CONDUCTIVITY_MULT[METAL_COPPER] >= METAL_CONDUCTIVITY_MULT[METAL_GOLD]


# ---------------------------------------------------------------------------
# Ore smelting sets metal_type
# ---------------------------------------------------------------------------

class TestOreSmeltingMetalType:
    def test_smelt_iron_ore_sets_metal_type(self):
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_AIR
        grid.temperature[4, 4, 4] = 900.0
        ms.held_materials = {VOXEL_IRON_ORE: 1}

        cs._current_z = 4
        bus.publish("craft_recipe_selected", recipe_name="Ore Smelting")
        bus.publish("craft_at_position", x=4, y=4, z=4)

        assert grid.get(4, 4, 4) == VOXEL_IRON_INGOT
        assert grid.get_metal_type(4, 4, 4) == METAL_IRON

    def test_smelt_copper_ore_sets_metal_type(self):
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_AIR
        grid.temperature[4, 4, 4] = 900.0
        ms.held_materials = {VOXEL_COPPER_ORE: 1}

        cs._current_z = 4
        bus.publish("craft_recipe_selected", recipe_name="Ore Smelting")
        bus.publish("craft_at_position", x=4, y=4, z=4)

        assert grid.get(4, 4, 4) == VOXEL_COPPER_INGOT
        assert grid.get_metal_type(4, 4, 4) == METAL_COPPER

    def test_smelt_gold_ore_sets_metal_type(self):
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_AIR
        grid.temperature[4, 4, 4] = 900.0
        ms.held_materials = {VOXEL_GOLD_ORE: 1}

        cs._current_z = 4
        bus.publish("craft_recipe_selected", recipe_name="Ore Smelting")
        bus.publish("craft_at_position", x=4, y=4, z=4)

        assert grid.get(4, 4, 4) == VOXEL_GOLD_INGOT
        assert grid.get_metal_type(4, 4, 4) == METAL_GOLD


# ---------------------------------------------------------------------------
# Mana infusion produces enchanted metal_type
# ---------------------------------------------------------------------------

class TestManaInfusionMetalType:
    def _infuse_setup(self):
        bus, grid, ms, cs = _setup()
        # Place lava within 3 blocks
        grid.grid[7, 4, 4] = VOXEL_LAVA
        return bus, grid, ms, cs

    def test_infuse_iron_ingot(self):
        bus, grid, ms, cs = self._infuse_setup()
        grid.grid[4, 4, 4] = VOXEL_IRON_INGOT
        grid.set_metal_type(4, 4, 4, METAL_IRON)
        grid.loose[4, 4, 4] = True
        ms.held_materials = {VOXEL_MANA_CRYSTAL: 1}

        cs._current_z = 4
        bus.publish("craft_recipe_selected", recipe_name="Mana Infusion")
        bus.publish("craft_at_position", x=4, y=4, z=4)

        assert grid.get(4, 4, 4) == VOXEL_ENCHANTED_METAL
        assert grid.get_metal_type(4, 4, 4) == METAL_ENCH_IRON

    def test_infuse_copper_ingot(self):
        bus, grid, ms, cs = self._infuse_setup()
        grid.grid[4, 4, 4] = VOXEL_COPPER_INGOT
        grid.set_metal_type(4, 4, 4, METAL_COPPER)
        grid.loose[4, 4, 4] = True
        ms.held_materials = {VOXEL_MANA_CRYSTAL: 1}

        cs._current_z = 4
        bus.publish("craft_recipe_selected", recipe_name="Mana Infusion")
        bus.publish("craft_at_position", x=4, y=4, z=4)

        assert grid.get(4, 4, 4) == VOXEL_ENCHANTED_METAL
        assert grid.get_metal_type(4, 4, 4) == METAL_ENCH_COPPER

    def test_infuse_gold_ingot(self):
        bus, grid, ms, cs = self._infuse_setup()
        grid.grid[4, 4, 4] = VOXEL_GOLD_INGOT
        grid.set_metal_type(4, 4, 4, METAL_GOLD)
        grid.loose[4, 4, 4] = True
        ms.held_materials = {VOXEL_MANA_CRYSTAL: 1}

        cs._current_z = 4
        bus.publish("craft_recipe_selected", recipe_name="Mana Infusion")
        bus.publish("craft_at_position", x=4, y=4, z=4)

        assert grid.get(4, 4, 4) == VOXEL_ENCHANTED_METAL
        assert grid.get_metal_type(4, 4, 4) == METAL_ENCH_GOLD

    def test_cannot_infuse_already_enchanted(self):
        """Mana infusion rejects already-enchanted ingots."""
        bus, grid, ms, cs = self._infuse_setup()
        grid.grid[4, 4, 4] = VOXEL_ENCHANTED_METAL
        grid.set_metal_type(4, 4, 4, METAL_ENCH_IRON)
        ms.held_materials = {VOXEL_MANA_CRYSTAL: 1}

        cs._current_z = 4
        bus.publish("craft_recipe_selected", recipe_name="Mana Infusion")
        # Should not find valid positions
        bus.publish("craft_at_position", x=4, y=4, z=4)

        # Enchanted metal should remain unchanged (not double-enchanted)
        assert grid.get(4, 4, 4) == VOXEL_ENCHANTED_METAL
        assert grid.get_metal_type(4, 4, 4) == METAL_ENCH_IRON

    def test_infuse_legacy_ingot_without_metal_type(self):
        """Mana infusion on an ingot with no metal_type falls back to HELD_TO_METAL."""
        bus, grid, ms, cs = self._infuse_setup()
        grid.grid[4, 4, 4] = VOXEL_GOLD_INGOT
        # Don't set metal_type — legacy ingot
        ms.held_materials = {VOXEL_MANA_CRYSTAL: 1}

        cs._current_z = 4
        bus.publish("craft_recipe_selected", recipe_name="Mana Infusion")
        bus.publish("craft_at_position", x=4, y=4, z=4)

        assert grid.get(4, 4, 4) == VOXEL_ENCHANTED_METAL
        assert grid.get_metal_type(4, 4, 4) == METAL_ENCH_GOLD


# ---------------------------------------------------------------------------
# Pick up / drop preserves metal_type
# ---------------------------------------------------------------------------

class TestPickUpDropMetalType:
    def test_pick_up_preserves_metal_type(self):
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_IRON_INGOT
        grid.set_metal_type(4, 4, 4, METAL_IRON)
        grid.loose[4, 4, 4] = True

        ms.pick_up(4, 4, 4)
        assert ms.held_metal_types[VOXEL_IRON_INGOT] == METAL_IRON

    def test_drop_restores_metal_type(self):
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_COPPER_INGOT
        grid.set_metal_type(4, 4, 4, METAL_COPPER)
        grid.loose[4, 4, 4] = True

        ms.pick_up(4, 4, 4)
        # Drop at a different location
        grid.grid[3, 3, 3] = VOXEL_AIR
        ms.drop(3, 3, 3)

        assert grid.get(3, 3, 3) == VOXEL_COPPER_INGOT
        assert grid.get_metal_type(3, 3, 3) == METAL_COPPER

    def test_pick_up_enchanted_preserves_enchanted_metal_type(self):
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_ENCHANTED_METAL
        grid.set_metal_type(4, 4, 4, METAL_ENCH_GOLD)
        grid.loose[4, 4, 4] = True

        ms.pick_up(4, 4, 4)
        assert ms.held_metal_types[VOXEL_ENCHANTED_METAL] == METAL_ENCH_GOLD

    def test_consume_cleans_up_metal_type(self):
        bus, grid, ms, cs = _setup()
        grid.grid[4, 4, 4] = VOXEL_IRON_INGOT
        grid.set_metal_type(4, 4, 4, METAL_IRON)
        grid.loose[4, 4, 4] = True

        ms.pick_up(4, 4, 4)
        assert VOXEL_IRON_INGOT in ms.held_metal_types

        ms.consume(VOXEL_IRON_INGOT)
        assert VOXEL_IRON_INGOT not in ms.held_metal_types
