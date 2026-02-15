"""Crafting recipe definitions — the CRAFTING BOOK.

Each recipe has:
  - name: human-readable recipe name
  - description: what happens
  - check_fn(grid, x, y, z, held_type) -> bool: can the craft happen here?
  - craft_fn(grid, x, y, z, held_type, event_bus) -> bool: execute the craft
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

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
    VOXEL_GOLD_ORE,
    VOXEL_MANA_CRYSTAL,
    VOXEL_IRON_INGOT,
    VOXEL_COPPER_INGOT,
    VOXEL_GOLD_INGOT,
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

if TYPE_CHECKING:
    from dungeon_builder.core.event_bus import EventBus
    from dungeon_builder.world.voxel_grid import VoxelGrid

# Ore -> Metal ingot mapping
ORE_TO_INGOT = {
    VOXEL_IRON_ORE: VOXEL_IRON_INGOT,
    VOXEL_COPPER_ORE: VOXEL_COPPER_INGOT,
    VOXEL_GOLD_ORE: VOXEL_GOLD_INGOT,
}

# Metal ingot types (for pile stacking)
METAL_INGOTS = frozenset({
    VOXEL_IRON_INGOT,
    VOXEL_COPPER_INGOT,
    VOXEL_GOLD_INGOT,
    VOXEL_ENCHANTED_METAL,
})

# 6-connected neighbor offsets
NEIGHBORS_6 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


@dataclass
class CraftingRecipe:
    name: str
    description: str
    check_fn: Callable[..., bool]
    craft_fn: Callable[..., bool]


# ---------------------------------------------------------------------------
# Recipe implementations
# ---------------------------------------------------------------------------

def _check_marble_wall(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Marble on air with air on one side and stone on the opposite."""
    if held_type != VOXEL_MARBLE:
        return False
    if grid.get(x, y, z) != VOXEL_AIR:
        return False
    # Check for air/stone opposing pair in XY plane
    pairs = [
        ((1, 0, 0), (-1, 0, 0)),
        ((0, 1, 0), (0, -1, 0)),
    ]
    for (dx1, dy1, dz1), (dx2, dy2, dz2) in pairs:
        n1 = grid.get(x + dx1, y + dy1, z + dz1)
        n2 = grid.get(x + dx2, y + dy2, z + dz2)
        if n1 == VOXEL_AIR and n2 != VOXEL_AIR:
            return True
        if n2 == VOXEL_AIR and n1 != VOXEL_AIR:
            return True
    return False


def _craft_marble_wall(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus
) -> bool:
    grid.set(x, y, z, VOXEL_MARBLE, event_bus=event_bus)
    return True


def _check_ore_smelting(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Drop ore on air where temperature > 800."""
    if held_type not in ORE_TO_INGOT:
        return False
    if grid.get(x, y, z) != VOXEL_AIR:
        return False
    return grid.get_temperature(x, y, z) > 800.0


def _craft_ore_smelting(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus
) -> bool:
    ingot = ORE_TO_INGOT[held_type]
    grid.set(x, y, z, ingot, event_bus=event_bus)
    grid.set_loose(x, y, z, True)
    return True


def _check_obsidian_forge(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Drop basalt on lava -> obsidian."""
    if held_type != VOXEL_BASALT:
        return False
    return grid.get(x, y, z) == VOXEL_LAVA


def _craft_obsidian_forge(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus
) -> bool:
    grid.set(x, y, z, VOXEL_OBSIDIAN, event_bus=event_bus)
    grid.set_loose(x, y, z, True)
    return True


def _check_mana_infusion(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Drop mana crystal on a metal ingot near lava (within 3 blocks)."""
    if held_type != VOXEL_MANA_CRYSTAL:
        return False
    target = grid.get(x, y, z)
    if target not in METAL_INGOTS:
        return False
    # Check for lava within 3 blocks
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            for dz in range(-3, 4):
                if grid.get(x + dx, y + dy, z + dz) == VOXEL_LAVA:
                    return True
    return False


def _craft_mana_infusion(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus
) -> bool:
    grid.set(x, y, z, VOXEL_ENCHANTED_METAL, event_bus=event_bus)
    grid.set_loose(x, y, z, True)
    return True


def _check_stone_brick(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Drop limestone on air near a wall (any solid neighbor) -> limestone block."""
    if held_type != VOXEL_LIMESTONE:
        return False
    if grid.get(x, y, z) != VOXEL_AIR:
        return False
    # Must have at least one solid neighbor
    for dx, dy, dz in NEIGHBORS_6:
        n = grid.get(x + dx, y + dy, z + dz)
        if n != VOXEL_AIR:
            return True
    return False


def _craft_stone_brick(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus
) -> bool:
    grid.set(x, y, z, VOXEL_LIMESTONE, event_bus=event_bus)
    return True


def _check_glass(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Drop sandstone on air where temperature > 600 -> chalk (glass-like)."""
    if held_type != VOXEL_SANDSTONE:
        return False
    if grid.get(x, y, z) != VOXEL_AIR:
        return False
    return grid.get_temperature(x, y, z) > 600.0


def _craft_glass(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus
) -> bool:
    grid.set(x, y, z, VOXEL_CHALK, event_bus=event_bus)
    grid.set_loose(x, y, z, True)
    return True


def _check_granite_pillar(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Drop granite on air with solid below -> granite pillar.

    Does not match if below is slope/stairs (rolling stone takes priority).
    """
    if held_type != VOXEL_GRANITE:
        return False
    if grid.get(x, y, z) != VOXEL_AIR:
        return False
    # Must have solid below (z+1 = deeper), but not slope/stairs
    below = grid.get(x, y, z + 1)
    if below == VOXEL_AIR:
        return False
    if below in (VOXEL_SLOPE, VOXEL_STAIRS):
        return False
    return True


def _craft_granite_pillar(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus
) -> bool:
    grid.set(x, y, z, VOXEL_GRANITE, event_bus=event_bus)
    return True


# ---------------------------------------------------------------------------
# Functional block recipe implementations
# ---------------------------------------------------------------------------

def _check_reinforced_wall(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Iron ingot on stone -> reinforced wall."""
    if held_type != VOXEL_IRON_INGOT:
        return False
    return grid.get(x, y, z) == VOXEL_STONE


def _craft_reinforced_wall(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus
) -> bool:
    grid.set(x, y, z, VOXEL_REINFORCED_WALL, event_bus=event_bus)
    return True


def _check_treasure(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Gold ingot on stone -> treasure."""
    if held_type != VOXEL_GOLD_INGOT:
        return False
    return grid.get(x, y, z) == VOXEL_STONE


def _craft_treasure(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus
) -> bool:
    grid.set(x, y, z, VOXEL_TREASURE, event_bus=event_bus)
    return True


def _count_solid_sides_xy(grid: VoxelGrid, x: int, y: int, z: int) -> int:
    """Count how many of the 4 cardinal XY neighbors are solid (non-air)."""
    count = 0
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        if grid.get(x + dx, y + dy, z) != VOXEL_AIR:
            count += 1
    return count


def _has_opposite_walls(grid: VoxelGrid, x: int, y: int, z: int) -> bool:
    """Check if there are solid blocks on 2 opposite sides (X or Y axis)."""
    # Check X axis
    if grid.get(x + 1, y, z) != VOXEL_AIR and grid.get(x - 1, y, z) != VOXEL_AIR:
        return True
    # Check Y axis
    if grid.get(x, y + 1, z) != VOXEL_AIR and grid.get(x, y - 1, z) != VOXEL_AIR:
        return True
    return False


def _check_slope(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Stone on air with solid below + solid on exactly 1 side (X or Y)."""
    if held_type != VOXEL_STONE:
        return False
    if grid.get(x, y, z) != VOXEL_AIR:
        return False
    # Solid below (z+1 is deeper)
    if grid.get(x, y, z + 1) == VOXEL_AIR:
        return False
    return _count_solid_sides_xy(grid, x, y, z) == 1


def _craft_slope(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus
) -> bool:
    grid.set(x, y, z, VOXEL_SLOPE, event_bus=event_bus)
    return True


def _check_stairs(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Stone on air with solid below + solid on exactly 1 side + air above."""
    if held_type != VOXEL_STONE:
        return False
    if grid.get(x, y, z) != VOXEL_AIR:
        return False
    # Solid below (z+1 is deeper)
    if grid.get(x, y, z + 1) == VOXEL_AIR:
        return False
    # Air above (z-1 is shallower)
    if z == 0 or grid.get(x, y, z - 1) != VOXEL_AIR:
        return False
    return _count_solid_sides_xy(grid, x, y, z) == 1


def _craft_stairs(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus
) -> bool:
    grid.set(x, y, z, VOXEL_STAIRS, event_bus=event_bus)
    return True


def _check_door(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Enchanted metal on air between 2 opposite walls."""
    if held_type != VOXEL_ENCHANTED_METAL:
        return False
    if grid.get(x, y, z) != VOXEL_AIR:
        return False
    return _has_opposite_walls(grid, x, y, z)


def _craft_door(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus
) -> bool:
    grid.set(x, y, z, VOXEL_DOOR, event_bus=event_bus)
    # Doors start closed (state=1)
    grid.set_block_state(x, y, z, 1)
    return True


def _check_spike_trap(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Iron ingot on air with solid below."""
    if held_type != VOXEL_IRON_INGOT:
        return False
    if grid.get(x, y, z) != VOXEL_AIR:
        return False
    # Solid below (z+1 is deeper)
    return grid.get(x, y, z + 1) != VOXEL_AIR


def _craft_spike_trap(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus
) -> bool:
    grid.set(x, y, z, VOXEL_SPIKE, event_bus=event_bus)
    # Spikes start extended (state=1)
    grid.set_block_state(x, y, z, 1)
    return True


def _check_tarp(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Dirt on air between 2 opposite walls."""
    if held_type != VOXEL_DIRT:
        return False
    if grid.get(x, y, z) != VOXEL_AIR:
        return False
    return _has_opposite_walls(grid, x, y, z)


def _craft_tarp(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus
) -> bool:
    grid.set(x, y, z, VOXEL_TARP, event_bus=event_bus)
    return True


def _check_rolling_stone(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Granite on air above slope or stairs."""
    if held_type != VOXEL_GRANITE:
        return False
    if grid.get(x, y, z) != VOXEL_AIR:
        return False
    # Must have slope or stairs below (z+1 is deeper)
    below = grid.get(x, y, z + 1)
    return below in (VOXEL_SLOPE, VOXEL_STAIRS)


def _craft_rolling_stone(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus
) -> bool:
    grid.set(x, y, z, VOXEL_ROLLING_STONE, event_bus=event_bus)
    grid.set_loose(x, y, z, True)  # Rolling stone starts loose
    return True


# ---------------------------------------------------------------------------
# The crafting book
# ---------------------------------------------------------------------------

class CraftingBook:
    """Contains all crafting recipes. Recipes are checked in order."""

    def __init__(self) -> None:
        self.recipes: list[CraftingRecipe] = [
            CraftingRecipe(
                "Marble Wall",
                "Place marble against stone to build a wall",
                _check_marble_wall,
                _craft_marble_wall,
            ),
            CraftingRecipe(
                "Ore Smelting",
                "Drop ore in high heat (>800) to smelt into metal ingot",
                _check_ore_smelting,
                _craft_ore_smelting,
            ),
            CraftingRecipe(
                "Obsidian Forge",
                "Drop basalt on lava to create obsidian",
                _check_obsidian_forge,
                _craft_obsidian_forge,
            ),
            CraftingRecipe(
                "Mana Infusion",
                "Drop mana crystal on metal ingot near lava to enchant",
                _check_mana_infusion,
                _craft_mana_infusion,
            ),
            CraftingRecipe(
                "Stone Brick",
                "Place limestone next to a wall to build a stone brick",
                _check_stone_brick,
                _craft_stone_brick,
            ),
            CraftingRecipe(
                "Glass",
                "Drop sandstone in high heat (>600) to create glass",
                _check_glass,
                _craft_glass,
            ),
            CraftingRecipe(
                "Granite Pillar",
                "Place granite on solid ground to build a pillar",
                _check_granite_pillar,
                _craft_granite_pillar,
            ),
            # --- Functional block recipes ---
            CraftingRecipe(
                "Reinforced Wall",
                "Apply iron ingot to stone to reinforce it",
                _check_reinforced_wall,
                _craft_reinforced_wall,
            ),
            CraftingRecipe(
                "Treasure",
                "Apply gold ingot to stone to create a treasure",
                _check_treasure,
                _craft_treasure,
            ),
            CraftingRecipe(
                "Slope",
                "Place stone on air with solid below and one wall to form a slope",
                _check_slope,
                _craft_slope,
            ),
            CraftingRecipe(
                "Stairs",
                "Place stone on air with solid below, one wall, and air above to form stairs",
                _check_stairs,
                _craft_stairs,
            ),
            CraftingRecipe(
                "Door",
                "Place enchanted metal between two opposite walls to build a door",
                _check_door,
                _craft_door,
            ),
            CraftingRecipe(
                "Spike Trap",
                "Place iron ingot on air above solid ground to set a spike trap",
                _check_spike_trap,
                _craft_spike_trap,
            ),
            CraftingRecipe(
                "Tarp",
                "Stretch dirt between two opposite walls to cover a pit",
                _check_tarp,
                _craft_tarp,
            ),
            CraftingRecipe(
                "Rolling Stone",
                "Place granite above a slope or stairs to create a rolling stone",
                _check_rolling_stone,
                _craft_rolling_stone,
            ),
        ]

    def find_recipe(
        self, grid: VoxelGrid, x: int, y: int, z: int, held_type: int
    ) -> CraftingRecipe | None:
        """Return the first matching recipe, or None."""
        for recipe in self.recipes:
            if recipe.check_fn(grid, x, y, z, held_type):
                return recipe
        return None

    def find_all_recipes(
        self, grid: VoxelGrid, x: int, y: int, z: int, held_type: int
    ) -> list[CraftingRecipe]:
        """Return ALL matching recipes (not just the first)."""
        return [r for r in self.recipes if r.check_fn(grid, x, y, z, held_type)]
