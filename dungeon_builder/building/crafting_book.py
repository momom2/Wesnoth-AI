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
    """Drop granite on air with solid below -> granite pillar."""
    if held_type != VOXEL_GRANITE:
        return False
    if grid.get(x, y, z) != VOXEL_AIR:
        return False
    # Must have solid below (z+1 = deeper)
    below = grid.get(x, y, z + 1)
    return below != VOXEL_AIR


def _craft_granite_pillar(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus
) -> bool:
    grid.set(x, y, z, VOXEL_GRANITE, event_bus=event_bus)
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
        ]

    def find_recipe(
        self, grid: VoxelGrid, x: int, y: int, z: int, held_type: int
    ) -> CraftingRecipe | None:
        """Return the first matching recipe, or None."""
        for recipe in self.recipes:
            if recipe.check_fn(grid, x, y, z, held_type):
                return recipe
        return None
