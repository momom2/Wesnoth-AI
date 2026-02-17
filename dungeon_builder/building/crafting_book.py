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
    VOXEL_WATER,
    METAL_NONE,
    METAL_IRON,
    METAL_COPPER,
    METAL_GOLD,
    ENCHANTED_OFFSET,
    HELD_TO_METAL,
    ORE_TO_METAL,
    PIPEABLE_STONE_TYPES,
    base_metal_of,
    is_enchanted_metal,
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
    required_inputs: frozenset[int]   # Voxel types the player must hold (any match)
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
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus,
    **kwargs,
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
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus,
    **kwargs,
) -> bool:
    ingot = ORE_TO_INGOT[held_type]
    metal = ORE_TO_METAL[held_type]
    grid.set(x, y, z, ingot, event_bus=event_bus)
    grid.set_metal_type(x, y, z, metal)
    grid.set_loose(x, y, z, True)
    return True


def _check_obsidian_forge(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Drop basalt on lava -> obsidian."""
    if held_type != VOXEL_BASALT:
        return False
    return grid.get(x, y, z) == VOXEL_LAVA


def _craft_obsidian_forge(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus,
    **kwargs,
) -> bool:
    grid.set(x, y, z, VOXEL_OBSIDIAN, event_bus=event_bus)
    grid.set_loose(x, y, z, True)
    return True


def _check_mana_infusion(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Drop mana crystal on a base metal ingot near lava (within 3 blocks).

    Only base (non-enchanted) ingots can be infused.
    """
    if held_type != VOXEL_MANA_CRYSTAL:
        return False
    target = grid.get(x, y, z)
    # Only base ingots, not already-enchanted
    if target not in (VOXEL_IRON_INGOT, VOXEL_COPPER_INGOT, VOXEL_GOLD_INGOT):
        return False
    # Check for lava within 3 blocks
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            for dz in range(-3, 4):
                if grid.get(x + dx, y + dy, z + dz) == VOXEL_LAVA:
                    return True
    return False


def _craft_mana_infusion(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus,
    **kwargs,
) -> bool:
    # Determine source metal from the ingot being enchanted
    source_metal = grid.get_metal_type(x, y, z)
    # If the ingot has no metal_type set (legacy), derive from voxel type
    if source_metal == METAL_NONE:
        target = grid.get(x, y, z)
        source_metal = HELD_TO_METAL.get(target, METAL_IRON)
    grid.set(x, y, z, VOXEL_ENCHANTED_METAL, event_bus=event_bus)
    grid.set_metal_type(x, y, z, source_metal | ENCHANTED_OFFSET)
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
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus,
    **kwargs,
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
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus,
    **kwargs,
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
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus,
    **kwargs,
) -> bool:
    grid.set(x, y, z, VOXEL_GRANITE, event_bus=event_bus)
    return True


# ---------------------------------------------------------------------------
# Functional block recipe implementations
# ---------------------------------------------------------------------------

def _check_reinforced_wall(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Any metal ingot on stone -> reinforced wall."""
    if held_type not in (VOXEL_IRON_INGOT, VOXEL_COPPER_INGOT, VOXEL_GOLD_INGOT):
        return False
    return grid.get(x, y, z) == VOXEL_STONE


def _craft_reinforced_wall(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus,
    held_metal: int = METAL_NONE,
) -> bool:
    metal = held_metal if held_metal != METAL_NONE else HELD_TO_METAL.get(held_type, METAL_IRON)
    grid.set(x, y, z, VOXEL_REINFORCED_WALL, event_bus=event_bus)
    grid.set_metal_type(x, y, z, metal)
    return True


def _check_treasure(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Gold ingot on stone -> treasure."""
    if held_type != VOXEL_GOLD_INGOT:
        return False
    return grid.get(x, y, z) == VOXEL_STONE


def _craft_treasure(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus,
    held_metal: int = METAL_NONE,
) -> bool:
    metal = held_metal if held_metal != METAL_NONE else METAL_GOLD
    grid.set(x, y, z, VOXEL_TREASURE, event_bus=event_bus)
    grid.set_metal_type(x, y, z, metal)
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
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus,
    **kwargs,
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
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus,
    **kwargs,
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
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus,
    held_metal: int = METAL_NONE,
) -> bool:
    metal = held_metal if held_metal != METAL_NONE else HELD_TO_METAL.get(held_type, METAL_IRON)
    # If held type is enchanted, the metal should already have the enchanted bit
    grid.set(x, y, z, VOXEL_DOOR, event_bus=event_bus)
    grid.set_metal_type(x, y, z, metal)
    # Doors start closed (state=1)
    grid.set_block_state(x, y, z, 1)
    return True


def _check_spike_trap(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Any metal ingot on air with solid below."""
    if held_type not in (VOXEL_IRON_INGOT, VOXEL_COPPER_INGOT, VOXEL_GOLD_INGOT):
        return False
    if grid.get(x, y, z) != VOXEL_AIR:
        return False
    # Solid below (z+1 is deeper)
    return grid.get(x, y, z + 1) != VOXEL_AIR


def _craft_spike_trap(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus,
    held_metal: int = METAL_NONE,
) -> bool:
    metal = held_metal if held_metal != METAL_NONE else HELD_TO_METAL.get(held_type, METAL_IRON)
    grid.set(x, y, z, VOXEL_SPIKE, event_bus=event_bus)
    grid.set_metal_type(x, y, z, metal)
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
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus,
    **kwargs,
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
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus,
    **kwargs,
) -> bool:
    grid.set(x, y, z, VOXEL_ROLLING_STONE, event_bus=event_bus)
    grid.set_loose(x, y, z, True)  # Rolling stone starts loose
    return True


# ---------------------------------------------------------------------------
# New functional block helpers
# ---------------------------------------------------------------------------

def _has_any_solid_neighbor(grid: VoxelGrid, x: int, y: int, z: int) -> bool:
    """Check if any of the 6 neighbors is solid (non-air)."""
    for dx, dy, dz in NEIGHBORS_6:
        if grid.get(x + dx, y + dy, z + dz) != VOXEL_AIR:
            return True
    return False


def _has_lava_below(grid: VoxelGrid, x: int, y: int, z: int, max_depth: int = 2) -> bool:
    """Check if there is lava within *max_depth* cells below (z+1..z+max_depth)."""
    for dz in range(1, max_depth + 1):
        if grid.get(x, y, z + dz) == VOXEL_LAVA:
            return True
    return False


def _has_adjacent_water(grid: VoxelGrid, x: int, y: int, z: int) -> bool:
    """Check if any of the 6 neighbors contains water."""
    for dx, dy, dz in NEIGHBORS_6:
        nx, ny, nz = x + dx, y + dy, z + dz
        if grid.get(nx, ny, nz) == VOXEL_WATER:
            return True
        if grid.in_bounds(nx, ny, nz) and grid.get_water_level(nx, ny, nz) > 0:
            return True
    return False


# ---------------------------------------------------------------------------
# New functional block recipe implementations (IDs 78-87)
# ---------------------------------------------------------------------------

# ── Gold Bait (78) ────────────────────────────────────────────────────

def _check_gold_bait(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Enchanted gold ingot on air with solid below + ≥1 wall."""
    if held_type != VOXEL_ENCHANTED_METAL:
        return False
    if grid.get(x, y, z) != VOXEL_AIR:
        return False
    # Must be solid below
    if grid.get(x, y, z + 1) == VOXEL_AIR:
        return False
    # Lateral support: at least 1 solid XY neighbor
    if _count_solid_sides_xy(grid, x, y, z) < 1:
        return False
    return True


def _craft_gold_bait(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus,
    held_metal: int = METAL_NONE,
) -> bool:
    metal = held_metal if held_metal != METAL_NONE else HELD_TO_METAL.get(held_type, METAL_GOLD) | ENCHANTED_OFFSET
    # Gold bait requires gold — validate base metal is gold
    if base_metal_of(metal) != METAL_GOLD:
        return False
    grid.set(x, y, z, VOXEL_GOLD_BAIT, event_bus=event_bus)
    grid.set_metal_type(x, y, z, metal)
    return True


# ── Heat Beacon (79) ──────────────────────────────────────────────────

def _check_heat_beacon(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Copper ingot on stone with temperature > 200."""
    if held_type != VOXEL_COPPER_INGOT:
        return False
    if grid.get(x, y, z) != VOXEL_STONE:
        return False
    return grid.get_temperature(x, y, z) > 200.0


def _craft_heat_beacon(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus,
    held_metal: int = METAL_NONE,
) -> bool:
    metal = held_metal if held_metal != METAL_NONE else METAL_COPPER
    grid.set(x, y, z, VOXEL_HEAT_BEACON, event_bus=event_bus)
    grid.set_metal_type(x, y, z, metal)
    return True


# ── Pressure Plate (80) ──────────────────────────────────────────────

def _check_pressure_plate(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Any enchanted ingot on stone with air above."""
    if held_type != VOXEL_ENCHANTED_METAL:
        return False
    if grid.get(x, y, z) != VOXEL_STONE:
        return False
    # Air above (z-1 is shallower)
    if z == 0:
        return True  # surface counts as "air above"
    return grid.get(x, y, z - 1) == VOXEL_AIR


def _craft_pressure_plate(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus,
    held_metal: int = METAL_NONE,
) -> bool:
    metal = held_metal if held_metal != METAL_NONE else HELD_TO_METAL.get(held_type, METAL_IRON) | ENCHANTED_OFFSET
    grid.set(x, y, z, VOXEL_PRESSURE_PLATE, event_bus=event_bus)
    grid.set_metal_type(x, y, z, metal)
    # Pressure plate starts unarmed (state=0)
    return True


# ── Iron Bars (81) ───────────────────────────────────────────────────

def _check_iron_bars(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Any enchanted ingot on air with 2 opposite walls."""
    if held_type != VOXEL_ENCHANTED_METAL:
        return False
    if grid.get(x, y, z) != VOXEL_AIR:
        return False
    return _has_opposite_walls(grid, x, y, z)


def _craft_iron_bars(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus,
    held_metal: int = METAL_NONE,
) -> bool:
    metal = held_metal if held_metal != METAL_NONE else HELD_TO_METAL.get(held_type, METAL_IRON) | ENCHANTED_OFFSET
    grid.set(x, y, z, VOXEL_IRON_BARS, event_bus=event_bus)
    grid.set_metal_type(x, y, z, metal)
    return True


# ── Floodgate (82) ───────────────────────────────────────────────────

def _check_floodgate(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Any enchanted ingot on stone with adjacent water."""
    if held_type != VOXEL_ENCHANTED_METAL:
        return False
    if grid.get(x, y, z) != VOXEL_STONE:
        return False
    return _has_adjacent_water(grid, x, y, z)


def _craft_floodgate(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus,
    held_metal: int = METAL_NONE,
) -> bool:
    metal = held_metal if held_metal != METAL_NONE else HELD_TO_METAL.get(held_type, METAL_IRON) | ENCHANTED_OFFSET
    grid.set(x, y, z, VOXEL_FLOODGATE, event_bus=event_bus)
    grid.set_metal_type(x, y, z, metal)
    # Floodgate starts closed (state=1)
    grid.set_block_state(x, y, z, 1)
    return True


# ── Alarm Bell (83) ──────────────────────────────────────────────────

def _check_alarm_bell(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Any enchanted ingot on air with solid below + ≥1 wall."""
    if held_type != VOXEL_ENCHANTED_METAL:
        return False
    if grid.get(x, y, z) != VOXEL_AIR:
        return False
    # Solid below
    if grid.get(x, y, z + 1) == VOXEL_AIR:
        return False
    # Lateral support
    if _count_solid_sides_xy(grid, x, y, z) < 1:
        return False
    return True


def _craft_alarm_bell(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus,
    held_metal: int = METAL_NONE,
) -> bool:
    metal = held_metal if held_metal != METAL_NONE else HELD_TO_METAL.get(held_type, METAL_IRON) | ENCHANTED_OFFSET
    grid.set(x, y, z, VOXEL_ALARM_BELL, event_bus=event_bus)
    grid.set_metal_type(x, y, z, metal)
    return True


# ── Fragile Floor (84) ──────────────────────────────────────────────

def _check_fragile_floor(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Chalk on air with ≥2 walls + solid below."""
    if held_type != VOXEL_CHALK:
        return False
    if grid.get(x, y, z) != VOXEL_AIR:
        return False
    # Solid below
    if grid.get(x, y, z + 1) == VOXEL_AIR:
        return False
    # At least 2 solid sides for lateral support
    if _count_solid_sides_xy(grid, x, y, z) < 2:
        return False
    return True


def _craft_fragile_floor(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus,
    held_metal: int = METAL_NONE,
) -> bool:
    grid.set(x, y, z, VOXEL_FRAGILE_FLOOR, event_bus=event_bus)
    # No metal_type (non-metallic)
    return True


# ── Pipe (85) ────────────────────────────────────────────────────────

def _check_pipe(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Any base metal ingot on air (with solid neighbor) or non-loose stone."""
    if held_type not in (VOXEL_IRON_INGOT, VOXEL_COPPER_INGOT, VOXEL_GOLD_INGOT):
        return False
    target = grid.get(x, y, z)
    if target == VOXEL_AIR:
        return _has_any_solid_neighbor(grid, x, y, z)
    # Can also build into non-loose pipeable stone
    if target in PIPEABLE_STONE_TYPES and not grid.is_loose(x, y, z):
        return True
    return False


def _craft_pipe(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus,
    held_metal: int = METAL_NONE,
) -> bool:
    metal = held_metal if held_metal != METAL_NONE else HELD_TO_METAL.get(held_type, METAL_IRON)
    grid.set(x, y, z, VOXEL_PIPE, event_bus=event_bus)
    grid.set_metal_type(x, y, z, metal)
    return True


# ── Pump (86) ────────────────────────────────────────────────────────

def _check_pump(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Any base metal ingot on an existing pipe."""
    if held_type not in (VOXEL_IRON_INGOT, VOXEL_COPPER_INGOT, VOXEL_GOLD_INGOT):
        return False
    return grid.get(x, y, z) == VOXEL_PIPE


def _craft_pump(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus,
    held_metal: int = METAL_NONE,
) -> bool:
    metal = held_metal if held_metal != METAL_NONE else HELD_TO_METAL.get(held_type, METAL_IRON)
    grid.set(x, y, z, VOXEL_PUMP, event_bus=event_bus)
    grid.set_metal_type(x, y, z, metal)
    # Pump starts with direction +X (state=0); player clicks to cycle
    grid.set_block_state(x, y, z, 0)
    return True


# ── Steam Vent (87) ──────────────────────────────────────────────────

def _check_steam_vent(grid: VoxelGrid, x: int, y: int, z: int, held_type: int) -> bool:
    """Obsidian on air with lava within 2 cells below."""
    if held_type != VOXEL_OBSIDIAN:
        return False
    if grid.get(x, y, z) != VOXEL_AIR:
        return False
    return _has_lava_below(grid, x, y, z, max_depth=2)


def _craft_steam_vent(
    grid: VoxelGrid, x: int, y: int, z: int, held_type: int, event_bus: EventBus,
    held_metal: int = METAL_NONE,
) -> bool:
    grid.set(x, y, z, VOXEL_STEAM_VENT, event_bus=event_bus)
    # No metal_type (non-metallic, obsidian-derived)
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
                frozenset({VOXEL_MARBLE}),
                _check_marble_wall,
                _craft_marble_wall,
            ),
            CraftingRecipe(
                "Ore Smelting",
                "Drop ore in high heat (>800) to smelt into metal ingot",
                frozenset({VOXEL_IRON_ORE, VOXEL_COPPER_ORE, VOXEL_GOLD_ORE}),
                _check_ore_smelting,
                _craft_ore_smelting,
            ),
            CraftingRecipe(
                "Obsidian Forge",
                "Drop basalt on lava to create obsidian",
                frozenset({VOXEL_BASALT}),
                _check_obsidian_forge,
                _craft_obsidian_forge,
            ),
            CraftingRecipe(
                "Mana Infusion",
                "Drop mana crystal on metal ingot near lava to enchant",
                frozenset({VOXEL_MANA_CRYSTAL}),
                _check_mana_infusion,
                _craft_mana_infusion,
            ),
            CraftingRecipe(
                "Stone Brick",
                "Place limestone next to a wall to build a stone brick",
                frozenset({VOXEL_LIMESTONE}),
                _check_stone_brick,
                _craft_stone_brick,
            ),
            CraftingRecipe(
                "Glass",
                "Drop sandstone in high heat (>600) to create glass",
                frozenset({VOXEL_SANDSTONE}),
                _check_glass,
                _craft_glass,
            ),
            CraftingRecipe(
                "Granite Pillar",
                "Place granite on solid ground to build a pillar",
                frozenset({VOXEL_GRANITE}),
                _check_granite_pillar,
                _craft_granite_pillar,
            ),
            # --- Functional block recipes ---
            CraftingRecipe(
                "Reinforced Wall",
                "Apply any metal ingot to stone to reinforce it",
                frozenset({VOXEL_IRON_INGOT, VOXEL_COPPER_INGOT, VOXEL_GOLD_INGOT}),
                _check_reinforced_wall,
                _craft_reinforced_wall,
            ),
            CraftingRecipe(
                "Treasure",
                "Apply gold ingot to stone to create a treasure",
                frozenset({VOXEL_GOLD_INGOT}),
                _check_treasure,
                _craft_treasure,
            ),
            CraftingRecipe(
                "Slope",
                "Place stone on air with solid below and one wall to form a slope",
                frozenset({VOXEL_STONE}),
                _check_slope,
                _craft_slope,
            ),
            CraftingRecipe(
                "Stairs",
                "Place stone on air with solid below, one wall, and air above to form stairs",
                frozenset({VOXEL_STONE}),
                _check_stairs,
                _craft_stairs,
            ),
            CraftingRecipe(
                "Door",
                "Place enchanted metal between two opposite walls to build a door",
                frozenset({VOXEL_ENCHANTED_METAL}),
                _check_door,
                _craft_door,
            ),
            CraftingRecipe(
                "Spike Trap",
                "Place any metal ingot on air above solid ground to set a spike trap",
                frozenset({VOXEL_IRON_INGOT, VOXEL_COPPER_INGOT, VOXEL_GOLD_INGOT}),
                _check_spike_trap,
                _craft_spike_trap,
            ),
            CraftingRecipe(
                "Tarp",
                "Stretch dirt between two opposite walls to cover a pit",
                frozenset({VOXEL_DIRT}),
                _check_tarp,
                _craft_tarp,
            ),
            CraftingRecipe(
                "Rolling Stone",
                "Place granite above a slope or stairs to create a rolling stone",
                frozenset({VOXEL_GRANITE}),
                _check_rolling_stone,
                _craft_rolling_stone,
            ),
            # --- New functional block recipes (Phase 2) ---
            CraftingRecipe(
                "Gold Bait",
                "Place enchanted gold in air with floor + wall to lure greedy intruders",
                frozenset({VOXEL_ENCHANTED_METAL}),
                _check_gold_bait,
                _craft_gold_bait,
            ),
            CraftingRecipe(
                "Heat Beacon",
                "Apply copper ingot to hot stone (>200°) to create a heat source",
                frozenset({VOXEL_COPPER_INGOT}),
                _check_heat_beacon,
                _craft_heat_beacon,
            ),
            CraftingRecipe(
                "Pressure Plate",
                "Apply enchanted ingot to stone with air above to set a trigger",
                frozenset({VOXEL_ENCHANTED_METAL}),
                _check_pressure_plate,
                _craft_pressure_plate,
            ),
            CraftingRecipe(
                "Iron Bars",
                "Place enchanted ingot between opposite walls to create bars",
                frozenset({VOXEL_ENCHANTED_METAL}),
                _check_iron_bars,
                _craft_iron_bars,
            ),
            CraftingRecipe(
                "Floodgate",
                "Apply enchanted ingot to stone near water to create a floodgate",
                frozenset({VOXEL_ENCHANTED_METAL}),
                _check_floodgate,
                _craft_floodgate,
            ),
            CraftingRecipe(
                "Alarm Bell",
                "Place enchanted ingot in air with floor + wall to detect intruders",
                frozenset({VOXEL_ENCHANTED_METAL}),
                _check_alarm_bell,
                _craft_alarm_bell,
            ),
            CraftingRecipe(
                "Fragile Floor",
                "Place chalk in air with floor + 2 walls to disguise a weak floor",
                frozenset({VOXEL_CHALK}),
                _check_fragile_floor,
                _craft_fragile_floor,
            ),
            CraftingRecipe(
                "Pipe",
                "Place metal ingot in air or stone to build pipe network",
                frozenset({VOXEL_IRON_INGOT, VOXEL_COPPER_INGOT, VOXEL_GOLD_INGOT}),
                _check_pipe,
                _craft_pipe,
            ),
            CraftingRecipe(
                "Pump",
                "Place metal ingot on a pipe to drive flow (click to change direction)",
                frozenset({VOXEL_IRON_INGOT, VOXEL_COPPER_INGOT, VOXEL_GOLD_INGOT}),
                _check_pump,
                _craft_pump,
            ),
            CraftingRecipe(
                "Steam Vent",
                "Place obsidian in air above lava to create a steam vent",
                frozenset({VOXEL_OBSIDIAN}),
                _check_steam_vent,
                _craft_steam_vent,
            ),
        ]

        # Name → recipe lookup for quick access
        self._by_name: dict[str, CraftingRecipe] = {
            r.name: r for r in self.recipes
        }

    def get_recipe_by_name(self, name: str) -> CraftingRecipe | None:
        """Return recipe by name, or None if not found."""
        return self._by_name.get(name)

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

    def find_valid_positions(
        self,
        recipe: CraftingRecipe,
        grid: VoxelGrid,
        held_type: int,
        z_level: int,
    ) -> list[tuple[int, int, int]]:
        """Scan visible/claimed voxels at *z_level* for valid craft positions.

        Only checks the given z-level (the player's current view) for
        performance — 64×64 = 4096 cells per scan.
        """
        results: list[tuple[int, int, int]] = []
        w, d = grid.width, grid.depth
        if z_level < 0 or z_level >= grid.height:
            return results
        for x in range(w):
            for y in range(d):
                if not grid.is_visible(x, y, z_level) and not grid.is_claimed(x, y, z_level):
                    continue
                if recipe.check_fn(grid, x, y, z_level, held_type):
                    results.append((x, y, z_level))
        return results
