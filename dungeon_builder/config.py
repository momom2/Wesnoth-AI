"""Game constants and balance parameters. No magic numbers elsewhere."""

# Grid dimensions
GRID_WIDTH = 64
GRID_DEPTH = 64
GRID_HEIGHT = 21  # Z index 0 (surface) to Z index 20 (deepest)
CHUNK_SIZE = 16   # 16x16x1 per chunk

# Simulation timing
TICKS_PER_SECOND = 20
SPEED_MULTIPLIERS = {0: 0.0, 1: 1.0, 2: 3.0}  # pause / play / fast

# RNG
DEFAULT_SEED = 42

# Z-level mapping: array index 0 = surface (world Z=0), index 20 = deepest (world Z=-20)
SURFACE_Z = 0
DEEPEST_Z = 20

# Dungeon core
CORE_DEFAULT_HP = 100
CORE_X = 32
CORE_Y = 32
CORE_Z = 10  # Array index (world Z = -10)

# Layer-slice transparency (asymmetric: above vs below focus)
# Above focus (toward surface, lower z-index): ceiling context, barely visible
LAYER_ALPHA_ABOVE = {1: 1.0, 2: 0.15}
LAYER_MAX_VISIBLE_ABOVE = 2
# Below focus (deeper, higher z-index): extended depth view
LAYER_ALPHA_BELOW = {1: 0.7, 2: 0.5, 3: 0.35, 4: 0.2, 5: 0.1}
LAYER_MAX_VISIBLE_BELOW = 5

# Voxel types (uint8 values for numpy array)
VOXEL_AIR = 0
VOXEL_DIRT = 1
VOXEL_STONE = 2       # generic stone (treated as granite for gameplay)
VOXEL_BEDROCK = 3
VOXEL_CORE = 4

# Sedimentary rocks (upper layers)
VOXEL_SANDSTONE = 10
VOXEL_LIMESTONE = 11
VOXEL_SHALE = 12
VOXEL_CHALK = 13

# Metamorphic rocks (mid layers)
VOXEL_SLATE = 20
VOXEL_MARBLE = 21
VOXEL_GNEISS = 22

# Igneous rocks (deep layers)
VOXEL_GRANITE = 30
VOXEL_BASALT = 31
VOXEL_OBSIDIAN = 32

# Ores
VOXEL_IRON_ORE = 40
VOXEL_COPPER_ORE = 41
VOXEL_GOLD_ORE = 42
VOXEL_MANA_CRYSTAL = 43

# Liquids / special
VOXEL_LAVA = 50
VOXEL_WATER = 51

# Crafted materials
VOXEL_IRON_INGOT = 60
VOXEL_COPPER_INGOT = 61
VOXEL_GOLD_INGOT = 62
VOXEL_ENCHANTED_METAL = 63

# Crafted functional blocks
VOXEL_REINFORCED_WALL = 70
VOXEL_SPIKE = 71
VOXEL_DOOR = 72
VOXEL_TREASURE = 73
VOXEL_ROLLING_STONE = 74
VOXEL_TARP = 75
VOXEL_SLOPE = 76
VOXEL_STAIRS = 77

# New functional blocks (dungeon expansion)
VOXEL_GOLD_BAIT = 78
VOXEL_HEAT_BEACON = 79
VOXEL_PRESSURE_PLATE = 80
VOXEL_IRON_BARS = 81
VOXEL_FLOODGATE = 82
VOXEL_ALARM_BELL = 83
VOXEL_FRAGILE_FLOOR = 84
VOXEL_PIPE = 85
VOXEL_PUMP = 86
VOXEL_STEAM_VENT = 87

# ── Metal Type System ────────────────────────────────────────────────
# Per-voxel uint8 array specifying what metal a block is made of.
# Properties (melt temp, strength, greed, color) derived compositionally
# from metal type × block type.
METAL_NONE = 0       # Non-metallic blocks (stone, dirt, etc.)
METAL_IRON = 1
METAL_COPPER = 2
METAL_GOLD = 3
# Future: METAL_BRONZE = 4, METAL_STEEL = 5, METAL_MITHRIL = 6, ...

# Enchanted variants: bit 7 set → melt-immune, otherwise same base metal
ENCHANTED_OFFSET = 128
METAL_ENCH_IRON = METAL_IRON | ENCHANTED_OFFSET      # 129
METAL_ENCH_COPPER = METAL_COPPER | ENCHANTED_OFFSET   # 130
METAL_ENCH_GOLD = METAL_GOLD | ENCHANTED_OFFSET       # 131


def is_enchanted_metal(metal_type: int) -> bool:
    """True if metal_type represents an enchanted variant (melt-immune)."""
    return (metal_type & ENCHANTED_OFFSET) != 0


def base_metal_of(metal_type: int) -> int:
    """Strip enchanted bit to get base metal (METAL_IRON/COPPER/GOLD)."""
    return metal_type & 0x7F


def make_enchanted(metal_type: int) -> int:
    """Return the enchanted version of a base metal type."""
    return metal_type | ENCHANTED_OFFSET


# Map held voxel type → metal constant for crafting
HELD_TO_METAL = {
    VOXEL_IRON_INGOT: METAL_IRON,
    VOXEL_COPPER_INGOT: METAL_COPPER,
    VOXEL_GOLD_INGOT: METAL_GOLD,
    VOXEL_ENCHANTED_METAL: METAL_NONE,  # generic enchanted — caller sets from source
}

# Map ore voxel → base metal type
ORE_TO_METAL = {
    VOXEL_IRON_ORE: METAL_IRON,
    VOXEL_COPPER_ORE: METAL_COPPER,
    VOXEL_GOLD_ORE: METAL_GOLD,
}

# Metal property tables (independent of block type)
METAL_MELT_TEMPERATURE = {
    METAL_NONE: 0.0,       # non-metallic → never melts via metal system
    METAL_IRON: 1200.0,
    METAL_COPPER: 800.0,
    METAL_GOLD: 600.0,
}

METAL_STRENGTH_MULT = {    # multiplier on block's base shear/tensile/max_load
    METAL_NONE: 1.0,
    METAL_IRON: 1.0,       # baseline
    METAL_COPPER: 0.7,
    METAL_GOLD: 0.4,
}

METAL_GREED_APPEAL = {     # added to intruder greed calculation when visible
    METAL_NONE: 0.0,
    METAL_IRON: 0.0,
    METAL_COPPER: 0.2,
    METAL_GOLD: 0.8,
}

METAL_COLORS: dict[int, tuple[float, float, float]] = {
    METAL_NONE: (0.5, 0.5, 0.5),
    METAL_IRON: (0.55, 0.55, 0.60),
    METAL_COPPER: (0.72, 0.52, 0.35),
    METAL_GOLD: (0.95, 0.85, 0.20),
}

METAL_CONDUCTIVITY_MULT = {  # thermal conductivity multiplier for pipes
    METAL_NONE: 0.0,
    METAL_IRON: 0.8,
    METAL_COPPER: 1.0,       # best conductor
    METAL_GOLD: 0.9,
}

# Blocks that use metal_type (metallic objects)
METALLIC_BLOCKS = frozenset({
    VOXEL_IRON_INGOT, VOXEL_COPPER_INGOT, VOXEL_GOLD_INGOT,
    VOXEL_ENCHANTED_METAL,
    VOXEL_REINFORCED_WALL, VOXEL_SPIKE, VOXEL_DOOR,
    VOXEL_GOLD_BAIT, VOXEL_HEAT_BEACON, VOXEL_PRESSURE_PLATE,
    VOXEL_IRON_BARS, VOXEL_FLOODGATE, VOXEL_ALARM_BELL,
    VOXEL_PIPE, VOXEL_PUMP,
})

# Blocks that can melt (enchanted metal_type is immune regardless)
MELTABLE_BLOCKS = frozenset({
    VOXEL_IRON_INGOT, VOXEL_COPPER_INGOT, VOXEL_GOLD_INGOT,
    VOXEL_REINFORCED_WALL, VOXEL_SPIKE,
    VOXEL_HEAT_BEACON, VOXEL_PIPE, VOXEL_PUMP,
    VOXEL_DOOR, VOXEL_IRON_BARS, VOXEL_FLOODGATE,
    VOXEL_PRESSURE_PLATE, VOXEL_ALARM_BELL, VOXEL_GOLD_BAIT,
})

# Non-diggable voxel types
NON_DIGGABLE = frozenset({
    VOXEL_AIR, VOXEL_BEDROCK, VOXEL_CORE, VOXEL_LAVA, VOXEL_WATER,
    VOXEL_REINFORCED_WALL, VOXEL_IRON_BARS, VOXEL_FLOODGATE,
})

# Porosity per voxel type (0.0 = impermeable, 1.0 = fully permeable)
VOXEL_POROSITY = {
    VOXEL_AIR: 1.0,
    VOXEL_DIRT: 0.4,
    VOXEL_STONE: 0.005,
    VOXEL_BEDROCK: 0.0,
    VOXEL_CORE: 0.0,
    VOXEL_SANDSTONE: 0.35,
    VOXEL_LIMESTONE: 0.25,
    VOXEL_SHALE: 0.05,
    VOXEL_CHALK: 0.6,
    VOXEL_SLATE: 0.02,
    VOXEL_MARBLE: 0.01,
    VOXEL_GNEISS: 0.01,
    VOXEL_GRANITE: 0.005,
    VOXEL_BASALT: 0.01,
    VOXEL_OBSIDIAN: 0.0,
    VOXEL_IRON_ORE: 0.05,
    VOXEL_COPPER_ORE: 0.03,
    VOXEL_GOLD_ORE: 0.02,
    VOXEL_MANA_CRYSTAL: 0.0,
    VOXEL_LAVA: 0.0,
    VOXEL_WATER: 1.0,
    VOXEL_IRON_INGOT: 0.0,
    VOXEL_COPPER_INGOT: 0.0,
    VOXEL_GOLD_INGOT: 0.0,
    VOXEL_ENCHANTED_METAL: 0.0,
    VOXEL_REINFORCED_WALL: 0.0,
    VOXEL_SPIKE: 0.0,
    VOXEL_DOOR: 0.0,
    VOXEL_TREASURE: 0.0,
    VOXEL_ROLLING_STONE: 0.3,     # granular (rolls via angle of repose)
    VOXEL_TARP: 0.8,              # porous fabric
    VOXEL_SLOPE: 0.01,
    VOXEL_STAIRS: 0.01,
    VOXEL_GOLD_BAIT: 0.0,
    VOXEL_HEAT_BEACON: 0.0,
    VOXEL_PRESSURE_PLATE: 0.0,
    VOXEL_IRON_BARS: 0.8,           # transparent (LOS passes through)
    VOXEL_FLOODGATE: 0.0,
    VOXEL_ALARM_BELL: 0.0,
    VOXEL_FRAGILE_FLOOR: 0.6,       # chalky / weak
    VOXEL_PIPE: 0.0,
    VOXEL_PUMP: 0.0,
    VOXEL_STEAM_VENT: 0.3,          # obsidian-derived, some porosity
}

# Thermal conductivity per voxel type (0.0 = insulator, 1.0 = perfect conductor)
VOXEL_CONDUCTIVITY = {
    VOXEL_AIR: 0.05,
    VOXEL_DIRT: 0.3,
    VOXEL_STONE: 0.6,
    VOXEL_BEDROCK: 0.4,
    VOXEL_CORE: 0.1,
    VOXEL_SANDSTONE: 0.4,
    VOXEL_LIMESTONE: 0.45,
    VOXEL_SHALE: 0.35,
    VOXEL_CHALK: 0.3,
    VOXEL_SLATE: 0.55,
    VOXEL_MARBLE: 0.6,
    VOXEL_GNEISS: 0.65,
    VOXEL_GRANITE: 0.7,
    VOXEL_BASALT: 0.75,
    VOXEL_OBSIDIAN: 0.8,
    VOXEL_IRON_ORE: 0.5,
    VOXEL_COPPER_ORE: 0.55,
    VOXEL_GOLD_ORE: 0.6,
    VOXEL_MANA_CRYSTAL: 0.0,   # absorbs heat, does not conduct
    VOXEL_LAVA: 1.0,
    VOXEL_WATER: 0.6,
    VOXEL_IRON_INGOT: 0.8,
    VOXEL_COPPER_INGOT: 0.85,
    VOXEL_GOLD_INGOT: 0.9,
    VOXEL_ENCHANTED_METAL: 0.5,
    VOXEL_REINFORCED_WALL: 0.75,
    VOXEL_SPIKE: 0.8,
    VOXEL_DOOR: 0.5,
    VOXEL_TREASURE: 0.9,
    VOXEL_ROLLING_STONE: 0.7,
    VOXEL_TARP: 0.1,
    VOXEL_SLOPE: 0.6,
    VOXEL_STAIRS: 0.6,
    VOXEL_GOLD_BAIT: 0.85,
    VOXEL_HEAT_BEACON: 0.90,
    VOXEL_PRESSURE_PLATE: 0.85,
    VOXEL_IRON_BARS: 0.80,
    VOXEL_FLOODGATE: 0.80,
    VOXEL_ALARM_BELL: 0.85,
    VOXEL_FRAGILE_FLOOR: 0.15,       # chalky insulator
    VOXEL_PIPE: 0.90,                 # good conductor (metal tube)
    VOXEL_PUMP: 0.80,
    VOXEL_STEAM_VENT: 0.80,
}

# Temperature physics
LAVA_TEMPERATURE = 1000.0
MANA_CRYSTAL_TEMPERATURE = 20.0
SURFACE_HEAT_LOSS = 0.05
TEMPERATURE_TICK_INTERVAL = 5   # run diffusion every N ticks
DIFFUSION_RATE = 0.1

# Gravity / structural physics
GRAVITY_TICK_INTERVAL = 1        # Loose-fall runs every tick (responsive)
STRUCTURAL_TICK_INTERVAL = 10    # Load calc every 0.5s
CONNECTIVITY_TICK_INTERVAL = 10  # Connectivity flood-fill every 0.5s
MAX_FALL_PER_TICK = 5            # Loose blocks fall up to 5 cells per tick
MAX_CASCADE_PER_TICK = 64        # Cap structural failures per tick

# Structural anchors (absorb all load, infinite capacity)
STRUCTURAL_ANCHORS = frozenset({3, 4, 43})  # BEDROCK, CORE, MANA_CRYSTAL

# Render modes
RENDER_MODE_MATTER = "matter"
RENDER_MODE_HUMIDITY = "humidity"
RENDER_MODE_HEAT = "heat"
RENDER_MODE_STRUCTURAL = "structural"

# Dig durations in ticks (at 20 ticks/sec)
DIG_DURATION = {
    VOXEL_DIRT: 20,         # 1 second
    VOXEL_STONE: 40,        # 2 seconds (legacy)
    VOXEL_SANDSTONE: 30,    # 1.5 seconds
    VOXEL_LIMESTONE: 35,    # 1.75 seconds
    VOXEL_SHALE: 25,        # 1.25 seconds (brittle)
    VOXEL_CHALK: 20,        # 1 second (soft)
    VOXEL_SLATE: 60,        # 3 seconds
    VOXEL_MARBLE: 70,       # 3.5 seconds
    VOXEL_GNEISS: 65,       # 3.25 seconds
    VOXEL_GRANITE: 100,     # 5 seconds
    VOXEL_BASALT: 90,       # 4.5 seconds
    VOXEL_OBSIDIAN: 200,    # 10 seconds
    VOXEL_IRON_ORE: 50,     # 2.5 seconds
    VOXEL_COPPER_ORE: 55,   # 2.75 seconds
    VOXEL_GOLD_ORE: 60,     # 3 seconds
    VOXEL_MANA_CRYSTAL: 80, # 4 seconds
    VOXEL_SPIKE: 60,        # 3 seconds
    VOXEL_DOOR: 50,         # 2.5 seconds
    VOXEL_TREASURE: 30,     # 1.5 seconds
    VOXEL_ROLLING_STONE: 80, # 4 seconds
    VOXEL_TARP: 10,         # 0.5 seconds (flimsy)
    VOXEL_SLOPE: 45,        # 2.25 seconds
    VOXEL_STAIRS: 45,       # 2.25 seconds
    VOXEL_GOLD_BAIT: 60,    # 3 seconds
    VOXEL_HEAT_BEACON: 50,   # 2.5 seconds
    VOXEL_PRESSURE_PLATE: 50, # 2.5 seconds
    VOXEL_ALARM_BELL: 40,    # 2 seconds
    VOXEL_FRAGILE_FLOOR: 30, # 1.5 seconds (weak)
    VOXEL_PIPE: 40,           # 2 seconds
    VOXEL_PUMP: 50,           # 2.5 seconds
    VOXEL_STEAM_VENT: 80,    # 4 seconds (obsidian-like)
    # VOXEL_IRON_BARS and VOXEL_FLOODGATE are NON_DIGGABLE
}
MAX_CONCURRENT_DIGS = 999  # effectively unlimited — all reachable digs run simultaneously

# Colors per voxel type (RGBA floats)
VOXEL_COLORS = {
    VOXEL_DIRT: (0.55, 0.35, 0.17, 1.0),
    VOXEL_STONE: (0.5, 0.5, 0.5, 1.0),
    VOXEL_BEDROCK: (0.2, 0.2, 0.2, 1.0),
    VOXEL_CORE: (0.8, 0.1, 0.1, 1.0),
    VOXEL_SANDSTONE: (0.85, 0.75, 0.50, 1.0),
    VOXEL_LIMESTONE: (0.80, 0.80, 0.72, 1.0),
    VOXEL_SHALE: (0.40, 0.40, 0.45, 1.0),
    VOXEL_CHALK: (0.92, 0.91, 0.88, 1.0),
    VOXEL_SLATE: (0.45, 0.50, 0.55, 1.0),
    VOXEL_MARBLE: (0.90, 0.88, 0.85, 1.0),
    VOXEL_GNEISS: (0.55, 0.50, 0.48, 1.0),
    VOXEL_GRANITE: (0.65, 0.62, 0.60, 1.0),
    VOXEL_BASALT: (0.30, 0.30, 0.33, 1.0),
    VOXEL_OBSIDIAN: (0.10, 0.08, 0.12, 1.0),
    VOXEL_IRON_ORE: (0.60, 0.35, 0.25, 1.0),
    VOXEL_COPPER_ORE: (0.45, 0.65, 0.50, 1.0),
    VOXEL_GOLD_ORE: (0.85, 0.75, 0.20, 1.0),
    VOXEL_MANA_CRYSTAL: (0.55, 0.30, 0.85, 1.0),
    VOXEL_LAVA: (1.0, 0.30, 0.0, 1.0),
    VOXEL_WATER: (0.15, 0.40, 0.85, 0.7),
    VOXEL_IRON_INGOT: (0.70, 0.55, 0.50, 1.0),
    VOXEL_COPPER_INGOT: (0.75, 0.50, 0.30, 1.0),
    VOXEL_GOLD_INGOT: (0.95, 0.85, 0.30, 1.0),
    VOXEL_ENCHANTED_METAL: (0.40, 0.60, 0.90, 1.0),
    VOXEL_REINFORCED_WALL: (0.50, 0.52, 0.55, 1.0),   # steel gray
    VOXEL_SPIKE: (0.35, 0.30, 0.28, 1.0),              # dark metallic
    VOXEL_DOOR: (0.45, 0.35, 0.25, 1.0),               # wood/metal
    VOXEL_TREASURE: (0.95, 0.85, 0.20, 1.0),            # bright gold
    VOXEL_ROLLING_STONE: (0.60, 0.58, 0.55, 1.0),       # granite-like
    VOXEL_TARP: (0.65, 0.55, 0.35, 0.8),                # semi-transparent brown
    VOXEL_SLOPE: (0.55, 0.53, 0.50, 1.0),               # stone with slight warmth
    VOXEL_STAIRS: (0.58, 0.55, 0.52, 1.0),              # stone, slightly lighter
    VOXEL_GOLD_BAIT: (0.95, 0.85, 0.20, 1.0),         # same as treasure (deception!)
    VOXEL_HEAT_BEACON: (0.85, 0.45, 0.15, 1.0),       # glowing orange-copper
    VOXEL_PRESSURE_PLATE: (0.45, 0.45, 0.48, 1.0),    # dark steel (base, tinted by metal)
    VOXEL_IRON_BARS: (0.50, 0.50, 0.55, 0.7),         # semi-transparent metal
    VOXEL_FLOODGATE: (0.40, 0.50, 0.65, 1.0),         # blue-steel (base, tinted by metal)
    VOXEL_ALARM_BELL: (0.80, 0.70, 0.30, 1.0),        # brass (base, tinted by metal)
    VOXEL_FRAGILE_FLOOR: (0.5, 0.5, 0.5, 1.0),        # same as stone (deception!)
    VOXEL_PIPE: (0.72, 0.52, 0.35, 0.9),              # copper-ish (base, tinted by metal)
    VOXEL_PUMP: (0.55, 0.55, 0.60, 1.0),              # iron-ish (base, tinted by metal)
    VOXEL_STEAM_VENT: (0.15, 0.12, 0.18, 0.8),        # dark obsidian, semi-transparent
}

# ── Vertex noise: per-material color grain amplitude ──
VERTEX_NOISE_AMPLITUDE = 0.07  # Default amplitude (0.0 = flat, 0.1 = subtle)
VOXEL_NOISE: dict[int, float] = {
    VOXEL_DIRT: 0.10,           # Rough, earthy
    VOXEL_STONE: 0.06,          # Moderate grain
    VOXEL_SANDSTONE: 0.09,      # Sandy variation
    VOXEL_LIMESTONE: 0.05,      # Smooth-ish
    VOXEL_SHALE: 0.07,          # Layered
    VOXEL_CHALK: 0.04,          # Powdery, smooth
    VOXEL_SLATE: 0.06,          # Layered stone
    VOXEL_MARBLE: 0.12,         # Veined (high variation)
    VOXEL_GNEISS: 0.08,         # Banded
    VOXEL_GRANITE: 0.08,        # Speckled
    VOXEL_BASALT: 0.05,         # Dense, smooth
    VOXEL_OBSIDIAN: 0.03,       # Glassy, smooth
    VOXEL_LAVA: 0.12,           # Roiling surface
    VOXEL_WATER: 0.04,          # Gentle ripple
    VOXEL_MANA_CRYSTAL: 0.10,   # Glowing facets
    VOXEL_GOLD_BAIT: 0.04,      # smooth metallic
    VOXEL_HEAT_BEACON: 0.05,    # glowing variation
    VOXEL_PRESSURE_PLATE: 0.03, # machined metal
    VOXEL_IRON_BARS: 0.03,      # uniform bars
    VOXEL_FLOODGATE: 0.03,      # machined metal
    VOXEL_ALARM_BELL: 0.03,     # polished bell
    VOXEL_FRAGILE_FLOOR: 0.06,  # same as stone (deception)
    VOXEL_PIPE: 0.03,           # smooth tube
    VOXEL_PUMP: 0.04,           # mechanical
    VOXEL_STEAM_VENT: 0.03,     # glassy
}

# Intruder defaults (legacy — used by decision.py until Phase 6 rewrite)
INTRUDER_DEFAULT_HP = 50
INTRUDER_DEFAULT_SPEED = 1
INTRUDER_DEFAULT_DAMAGE = 5
INTRUDER_RETREAT_THRESHOLD = 0.2  # Retreat at 20% HP
INTRUDER_SPAWN_INTERVAL = 200     # Ticks between spawns (10 seconds)
INTRUDER_ATTACK_INTERVAL = 20     # Ticks between attacks on core (1 second)
MAX_INTRUDERS = 10

# ── Intruder archetype system ────────────────────────────────────────

# Party spawning
INTRUDER_PARTY_SPAWN_INTERVAL = 400   # 20 seconds between party spawns
MAX_PARTIES = 3                        # Max concurrent parties
MAX_INTRUDERS_TOTAL = 24               # Hard cap on alive intruders

# Map sharing
MAP_SHARE_RANGE = 3                    # Allies within N cells share maps
MAP_SHARE_INTERVAL = 10                # Share maps every N ticks (throttle)

# Interaction durations (ticks)
DOOR_BASH_TICKS = 15
DOOR_BASH_TICKS_GORECLAW = 10
DOOR_LOCKPICK_TICKS = 5
TREASURE_GRAB_TICKS = 10

# Trap damage
SPIKE_DAMAGE = 20
ROLLING_STONE_DAMAGE = 30
TARP_DETECT_CUNNING = 0.5             # Cunning threshold to detect tarps

# Warden abilities
WARDEN_HEAL_AMOUNT = 5
WARDEN_HEAL_INTERVAL = 20             # Ticks between heals
WARDEN_LOYALTY_BONUS = 0.2            # Loyalty boost to allies within range
WARDEN_DEATH_LOYALTY_PENALTY = 0.3    # Loyalty drop when warden dies

# Pyremancer abilities
PYREMANCER_HEAT_AMOUNT = 200.0        # Temperature added to adjacent blocks
PYREMANCER_HEAT_INTERVAL = 10         # Ticks between heat applications
PYREMANCER_ATTACK_RANGE = 3           # Ranged attack distance

# Goreclaw frenzy
GORECLAW_FRENZY_SPEED_MULT = 2
GORECLAW_FRENZY_DAMAGE_MULT = 1.5
GORECLAW_FRENZY_RANDOM_CHANCE = 0.5   # Chance of random move each step in frenzy

# Archetype render colors (R, G, B) — distinct per archetype
ARCHETYPE_COLORS: dict[str, tuple[float, float, float]] = {
    "Vanguard":    (0.7, 0.7, 0.9),   # Steel blue (armored tank)
    "Shadowblade": (0.4, 0.1, 0.5),   # Dark purple (stealthy rogue)
    "Tunneler":    (0.6, 0.4, 0.2),   # Brown (earthy digger)
    "Pyremancer":  (1.0, 0.35, 0.0),  # Bright orange (fire mage)
    "Windcaller":  (0.3, 0.9, 0.9),   # Cyan (air/wind)
    "Warden":      (1.0, 1.0, 0.4),   # Gold-yellow (holy healer)
    "Goreclaw":    (0.8, 0.1, 0.1),   # Blood red (berserker)
    "Gloomseer":   (0.2, 0.1, 0.3),   # Deep indigo (diviner)
    # Underworlders
    "Magmawraith":       (1.0, 0.5, 0.1),   # Molten orange
    "Boremite":          (0.5, 0.4, 0.3),   # Muddy brown
    "Stoneskin Brute":   (0.4, 0.45, 0.4),  # Dark gray-green
    "Tremorstalker":     (0.6, 0.3, 0.6),   # Purple-gray
    "Corrosive Crawler": (0.3, 0.7, 0.2),   # Acid green
}
ARCHETYPE_DEFAULT_COLOR: tuple[float, float, float] = (1.0, 0.2, 0.2)  # Fallback red
ARCHETYPE_FRENZY_COLOR: tuple[float, float, float] = (1.0, 0.0, 0.0)  # Bright red for frenzy

# Vision
WATER_LOS_DEPTH = 2                   # Water blocks LOS after N cells

# Party composition weights
PARTY_WEIGHT_STANDARD_RAID = 0.40
PARTY_WEIGHT_SCOUTING_PARTY = 0.25
PARTY_WEIGHT_SIEGE_FORCE = 0.20
PARTY_WEIGHT_WAR_BAND = 0.15

# Underworld party composition weights
PARTY_WEIGHT_UNDERWORLD_HORDE = 0.30
PARTY_WEIGHT_UNDERWORLD_OVERSEER = 0.25
PARTY_WEIGHT_UNDERWORLD_INFERNAL = 0.20
PARTY_WEIGHT_UNDERWORLD_SOLITARY = 0.25

# Underworlder spawning
UNDERWORLD_SPAWN_INTERVAL = 600         # 30 seconds between underworld party spawns
MAX_UNDERWORLD_PARTIES = 2              # Max concurrent underworld parties
MAX_UNDERWORLDERS_TOTAL = 16            # Hard cap on alive underworlders
UNDERWORLD_SPAWN_Z_MIN = 10            # Shallowest underworld spawn (CORE_Z)
UNDERWORLD_SPAWN_Z_MAX = 18            # Deepest underworld spawn (2 above bedrock)

# Corrosive Crawler
CORROSIVE_DAMAGE_FACTOR = 0.5           # stress_ratio added to adjacent blocks after dig

# Magmawraith
MAGMAWRAITH_HEAT_AMOUNT = 100.0         # Temperature added to adjacent blocks
MAGMAWRAITH_HEAT_INTERVAL = 15          # Ticks between heat applications

# Boremite
BOREMITE_DIG_DIVISOR = 3                # Dig at 1/3 duration (vs Tunneler 1/2)

# ── Social Dynamics: Level & Status ──
LEVEL_HP_SCALE = 0.15                   # Per-level HP multiplier increment (level 5 = 1.6x)
LEVEL_DAMAGE_SCALE = 0.10               # Per-level damage multiplier increment (level 5 = 1.4x)
LEVEL_WEIGHTS = (0.40, 0.30, 0.20, 0.08, 0.02)  # Probability for levels 1-5
LEVEL_DEADLY_SHIFT = 0.10               # Weight shifted from level 1 to higher per 0.1 lethality above 0.5

# ── Social Dynamics: Knowledge Archive ──
KNOWLEDGE_STALE_TICKS = 4000            # ~3.3 minutes — intel older than this is filtered at injection
KNOWLEDGE_UNCERTAIN_THRESHOLD = 0.7     # Skip cells above this uncertainty at injection
KNOWLEDGE_CONTRADICTION_BASE = 0.3      # Uncertainty added per contradiction
KNOWLEDGE_CONFIRM_DECAY = 0.5           # Uncertainty multiplier on confirmation
KNOWLEDGE_CHANGE_UNCERTAINTY = 0.4      # Uncertainty added when player changes a cell

# ── Social Dynamics: Reputation ──
REPUTATION_DEADLY_LETHALITY = 0.7       # Lethality threshold for "deadly" profile
REPUTATION_RICH_RICHNESS = 0.4          # Richness threshold for "rich" profile
REPUTATION_UNKNOWN_THRESHOLD = 5        # Fewer total outcomes → "unknown" profile
REPUTATION_DEADLY_MODIFIER = (0.3, 0.1, -0.2)   # (destroy, explore, pillage) offsets
REPUTATION_RICH_MODIFIER = (-0.1, 0.1, 0.3)
REPUTATION_UNKNOWN_MODIFIER = (0.0, 0.2, 0.0)
REPUTATION_DEADLY_LOYALTY = -0.1        # Loyalty modifier for deadly dungeon
REPUTATION_RICH_LOYALTY = -0.15         # Loyalty modifier for rich dungeon

# ── Social Dynamics: Morale ──
MORALE_BASE = 0.7                       # Starting morale for all intruders
MORALE_LOW_THRESHOLD = 0.3             # Below this: slower movement, earlier retreat
MORALE_HIGH_THRESHOLD = 0.8            # Above this: faster movement, damage bonus
MORALE_FLEE_THRESHOLD = 0.1            # Below this: abandon party and flee solo
MORALE_ALLY_DEATH_PENALTY = 0.15       # Morale loss per ally death witnessed
MORALE_DAMAGE_PENALTY = 0.03           # Morale loss per hit taken
MORALE_HAZARD_PENALTY = 0.01           # Morale loss per new hazard seen
MORALE_TREASURE_BONUS = 0.1            # Morale gain per treasure collected
MORALE_LEADER_BONUS = 0.001            # Morale gain per tick while leader alive
MORALE_WARDEN_TICK = 0.002             # Morale gain per tick near warden
MORALE_DRIFT_RATE = 0.001              # Drift toward MORALE_BASE per tick
MORALE_SLOW_FACTOR = 1.5               # Move interval multiplier when low morale
MORALE_FAST_FACTOR = 0.8               # Move interval multiplier when high morale
MORALE_DAMAGE_BONUS = 1.2              # Damage multiplier when high morale
MORALE_RETREAT_MULTIPLIER = 2.0        # Retreat threshold multiplier when low morale

# ── Social Dynamics: Faction encounters ──
FACTION_ENCOUNTER_INTERVAL = 5         # Check for cross-faction combat every N ticks

# Personal pathfinder
PERSONAL_PATHFINDER_MAX_ITERATIONS = 5000
HAZARD_PATH_COST = 100.0              # Extra cost for detected spike cells
LAVA_ADJACENT_PATH_COST = 10.0        # Extra cost for lava-adjacent cells

# Pathfinding
PATHFINDING_MAX_ITERATIONS = 10000
PATHFINDING_VERTICAL_COST = 1.5

# Camera defaults
CAMERA_DEFAULT_DISTANCE = 40.0
CAMERA_MIN_DISTANCE = 10.0
CAMERA_MAX_DISTANCE = 100.0
CAMERA_DEFAULT_HEADING = 45.0
CAMERA_DEFAULT_PITCH = -60.0
CAMERA_PAN_SPEED = 30.0
CAMERA_ROTATE_SPEED = 90.0
CAMERA_ZOOM_STEP = 5.0

# Claimed territory
CLAIMED_TICK_INTERVAL = 10  # Recompute every 0.5s (same as connectivity)

# Fog of war
FOG_COLOR = (0.03, 0.03, 0.04, 1.0)  # Near-black for unexplored blocks

# Ore / crystal x-ray visibility — seeps through this many solid blocks
PLAYER_XRAY_RANGE = 3  # mutable, future spells/upgrades increase this
XRAY_VISIBLE_TYPES = frozenset({
    VOXEL_IRON_ORE, VOXEL_COPPER_ORE, VOXEL_GOLD_ORE, VOXEL_MANA_CRYSTAL,
})

# Material weight (arbitrary load units, granite=10.0 baseline)
VOXEL_WEIGHT = {
    VOXEL_AIR: 0.0,
    VOXEL_DIRT: 5.0,
    VOXEL_STONE: 8.0,
    VOXEL_BEDROCK: 0.0,
    VOXEL_CORE: 0.0,
    VOXEL_SANDSTONE: 6.0,
    VOXEL_LIMESTONE: 7.0,
    VOXEL_SHALE: 5.5,
    VOXEL_CHALK: 4.0,
    VOXEL_SLATE: 7.5,
    VOXEL_MARBLE: 8.5,
    VOXEL_GNEISS: 8.0,
    VOXEL_GRANITE: 10.0,
    VOXEL_BASALT: 11.0,
    VOXEL_OBSIDIAN: 9.0,
    VOXEL_IRON_ORE: 12.0,
    VOXEL_COPPER_ORE: 11.0,
    VOXEL_GOLD_ORE: 14.0,
    VOXEL_MANA_CRYSTAL: 0.0,
    VOXEL_LAVA: 0.0,
    VOXEL_WATER: 3.0,
    VOXEL_IRON_INGOT: 15.0,
    VOXEL_COPPER_INGOT: 14.0,
    VOXEL_GOLD_INGOT: 18.0,
    VOXEL_ENCHANTED_METAL: 10.0,
    VOXEL_REINFORCED_WALL: 12.0,
    VOXEL_SPIKE: 8.0,
    VOXEL_DOOR: 6.0,
    VOXEL_TREASURE: 20.0,
    VOXEL_ROLLING_STONE: 12.0,
    VOXEL_TARP: 0.5,
    VOXEL_SLOPE: 7.0,
    VOXEL_STAIRS: 7.0,
    VOXEL_GOLD_BAIT: 18.0,
    VOXEL_HEAT_BEACON: 14.0,
    VOXEL_PRESSURE_PLATE: 8.0,
    VOXEL_IRON_BARS: 10.0,
    VOXEL_FLOODGATE: 12.0,
    VOXEL_ALARM_BELL: 6.0,
    VOXEL_FRAGILE_FLOOR: 4.0,
    VOXEL_PIPE: 10.0,
    VOXEL_PUMP: 14.0,
    VOXEL_STEAM_VENT: 9.0,
}

# Max load capacity (compressive strength) per voxel type
# Compressive capacity (3× base values — cave-ins should be rare, deliberate action only)
VOXEL_MAX_LOAD = {
    VOXEL_AIR: 0.0,
    VOXEL_DIRT: 60.0,
    VOXEL_STONE: 240.0,
    VOXEL_BEDROCK: float("inf"),
    VOXEL_CORE: float("inf"),
    VOXEL_SANDSTONE: 120.0,
    VOXEL_LIMESTONE: 150.0,
    VOXEL_SHALE: 90.0,
    VOXEL_CHALK: 45.0,
    VOXEL_SLATE: 210.0,
    VOXEL_MARBLE: 255.0,
    VOXEL_GNEISS: 225.0,
    VOXEL_GRANITE: 360.0,
    VOXEL_BASALT: 330.0,
    VOXEL_OBSIDIAN: 180.0,
    VOXEL_IRON_ORE: 195.0,
    VOXEL_COPPER_ORE: 165.0,
    VOXEL_GOLD_ORE: 135.0,
    VOXEL_MANA_CRYSTAL: float("inf"),
    VOXEL_LAVA: 0.0,
    VOXEL_WATER: 0.0,
    VOXEL_IRON_INGOT: 300.0,
    VOXEL_COPPER_INGOT: 240.0,
    VOXEL_GOLD_INGOT: 150.0,
    VOXEL_ENCHANTED_METAL: 450.0,
    VOXEL_REINFORCED_WALL: 600.0,  # strongest buildable block
    VOXEL_SPIKE: 120.0,
    VOXEL_DOOR: 240.0,
    VOXEL_TREASURE: 90.0,
    VOXEL_ROLLING_STONE: 300.0,
    VOXEL_TARP: 15.0,             # still fragile (breaks under a few blocks)
    VOXEL_SLOPE: 240.0,
    VOXEL_STAIRS: 240.0,
    VOXEL_GOLD_BAIT: 90.0,
    VOXEL_HEAT_BEACON: 240.0,
    VOXEL_PRESSURE_PLATE: 240.0,
    VOXEL_IRON_BARS: 300.0,
    VOXEL_FLOODGATE: 450.0,
    VOXEL_ALARM_BELL: 90.0,
    VOXEL_FRAGILE_FLOOR: 24.0,       # still deliberately weak
    VOXEL_PIPE: 240.0,
    VOXEL_PUMP: 270.0,
    VOXEL_STEAM_VENT: 120.0,
}

# Load distribution (legacy fixed ratios, kept for reference)
LOAD_DIST_BELOW = 0.80       # 80% to block directly below
LOAD_DIST_LATERAL = 0.05     # 5% to each of 4 cardinal-below neighbors (20% total)
BUTTRESS_FACTOR = 0.03       # Each solid same-level neighbor reduces load 3%

# Stiffness per voxel type (load attraction: stiffer receivers take more load)
# In redundant structures, F_i = k_i / Σk_j × F_total (direct stiffness method)
VOXEL_STIFFNESS = {
    VOXEL_AIR: 0.0,
    VOXEL_DIRT: 1.0,
    VOXEL_STONE: 6.0,
    VOXEL_BEDROCK: 100.0,
    VOXEL_CORE: 100.0,
    VOXEL_SANDSTONE: 2.0,
    VOXEL_LIMESTONE: 3.5,
    VOXEL_SHALE: 1.5,
    VOXEL_CHALK: 0.5,
    VOXEL_SLATE: 5.0,
    VOXEL_MARBLE: 7.0,
    VOXEL_GNEISS: 6.5,
    VOXEL_GRANITE: 10.0,
    VOXEL_BASALT: 9.0,
    VOXEL_OBSIDIAN: 4.0,
    VOXEL_IRON_ORE: 5.0,
    VOXEL_COPPER_ORE: 4.5,
    VOXEL_GOLD_ORE: 3.0,
    VOXEL_MANA_CRYSTAL: 100.0,
    VOXEL_LAVA: 0.0,
    VOXEL_WATER: 0.0,
    VOXEL_IRON_INGOT: 12.0,
    VOXEL_COPPER_INGOT: 10.0,
    VOXEL_GOLD_INGOT: 6.0,
    VOXEL_ENCHANTED_METAL: 15.0,
    VOXEL_REINFORCED_WALL: 20.0,
    VOXEL_SPIKE: 8.0,
    VOXEL_DOOR: 10.0,
    VOXEL_TREASURE: 3.0,
    VOXEL_ROLLING_STONE: 10.0,
    VOXEL_TARP: 0.1,
    VOXEL_SLOPE: 6.0,
    VOXEL_STAIRS: 6.0,
    VOXEL_GOLD_BAIT: 3.0,
    VOXEL_HEAT_BEACON: 8.0,
    VOXEL_PRESSURE_PLATE: 9.0,
    VOXEL_IRON_BARS: 8.0,
    VOXEL_FLOODGATE: 10.0,
    VOXEL_ALARM_BELL: 4.0,
    VOXEL_FRAGILE_FLOOR: 1.0,
    VOXEL_PIPE: 7.0,
    VOXEL_PUMP: 8.0,
    VOXEL_STEAM_VENT: 4.0,
}

# Tensile strength per voxel type (governs cantilever/bending failure)
# Failure: load × span / 2 > tensile_strength
# Stone has very low tensile strength; metals are far superior
# Tensile strength (3× base — cantilevers are more forgiving)
VOXEL_TENSILE_STRENGTH = {
    VOXEL_AIR: 0.0,
    VOXEL_DIRT: 6.0,
    VOXEL_STONE: 30.0,
    VOXEL_BEDROCK: float("inf"),
    VOXEL_CORE: float("inf"),
    VOXEL_SANDSTONE: 15.0,
    VOXEL_LIMESTONE: 24.0,
    VOXEL_SHALE: 12.0,
    VOXEL_CHALK: 6.0,
    VOXEL_SLATE: 36.0,
    VOXEL_MARBLE: 30.0,
    VOXEL_GNEISS: 33.0,
    VOXEL_GRANITE: 45.0,
    VOXEL_BASALT: 42.0,
    VOXEL_OBSIDIAN: 18.0,      # brittle glass, snaps more easily
    VOXEL_IRON_ORE: 24.0,
    VOXEL_COPPER_ORE: 21.0,
    VOXEL_GOLD_ORE: 15.0,
    VOXEL_MANA_CRYSTAL: float("inf"),
    VOXEL_LAVA: 0.0,
    VOXEL_WATER: 0.0,
    VOXEL_IRON_INGOT: 120.0,   # wrought iron, excellent in tension
    VOXEL_COPPER_INGOT: 90.0,
    VOXEL_GOLD_INGOT: 45.0,
    VOXEL_ENCHANTED_METAL: 150.0,
    VOXEL_REINFORCED_WALL: 180.0,
    VOXEL_SPIKE: 60.0,
    VOXEL_DOOR: 90.0,
    VOXEL_TREASURE: 15.0,
    VOXEL_ROLLING_STONE: 45.0,
    VOXEL_TARP: 3.0,
    VOXEL_SLOPE: 30.0,
    VOXEL_STAIRS: 30.0,
    VOXEL_GOLD_BAIT: 45.0,
    VOXEL_HEAT_BEACON: 165.0,
    VOXEL_PRESSURE_PLATE: 180.0,
    VOXEL_IRON_BARS: 195.0,
    VOXEL_FLOODGATE: 240.0,
    VOXEL_ALARM_BELL: 60.0,
    VOXEL_FRAGILE_FLOOR: 9.0,
    VOXEL_PIPE: 150.0,
    VOXEL_PUMP: 165.0,
    VOXEL_STEAM_VENT: 18.0,
}

# Shear strength per voxel type (lateral load capacity)
# Typically ~15-20% of compressive for stone, ~30-40% for ductile metals
# Shear strength (3× base — lateral loads need deliberate force to cause failure)
VOXEL_SHEAR_STRENGTH = {
    VOXEL_AIR: 0.0,
    VOXEL_DIRT: 12.0,
    VOXEL_STONE: 48.0,
    VOXEL_BEDROCK: float("inf"),
    VOXEL_CORE: float("inf"),
    VOXEL_SANDSTONE: 18.0,
    VOXEL_LIMESTONE: 24.0,
    VOXEL_SHALE: 13.5,
    VOXEL_CHALK: 6.0,
    VOXEL_SLATE: 30.0,
    VOXEL_MARBLE: 36.0,
    VOXEL_GNEISS: 33.0,
    VOXEL_GRANITE: 60.0,
    VOXEL_BASALT: 54.0,
    VOXEL_OBSIDIAN: 24.0,
    VOXEL_IRON_ORE: 39.0,
    VOXEL_COPPER_ORE: 33.0,
    VOXEL_GOLD_ORE: 21.0,
    VOXEL_MANA_CRYSTAL: float("inf"),
    VOXEL_LAVA: 0.0,
    VOXEL_WATER: 0.0,
    VOXEL_IRON_INGOT: 120.0,
    VOXEL_COPPER_INGOT: 96.0,
    VOXEL_GOLD_INGOT: 45.0,
    VOXEL_ENCHANTED_METAL: 180.0,
    VOXEL_REINFORCED_WALL: 150.0,
    VOXEL_SPIKE: 45.0,
    VOXEL_DOOR: 75.0,
    VOXEL_TREASURE: 15.0,
    VOXEL_ROLLING_STONE: 60.0,
    VOXEL_TARP: 3.0,
    VOXEL_SLOPE: 36.0,
    VOXEL_STAIRS: 36.0,
    VOXEL_GOLD_BAIT: 36.0,
    VOXEL_HEAT_BEACON: 105.0,
    VOXEL_PRESSURE_PLATE: 120.0,
    VOXEL_IRON_BARS: 135.0,
    VOXEL_FLOODGATE: 180.0,
    VOXEL_ALARM_BELL: 45.0,
    VOXEL_FRAGILE_FLOOR: 6.0,
    VOXEL_PIPE: 90.0,
    VOXEL_PUMP: 105.0,
    VOXEL_STEAM_VENT: 24.0,
}

# Multi-block arch detection
MAX_ARCH_SPAN = 5             # Maximum scanning distance for arch detection

# Environmental weakness factors
HUMIDITY_WEAKNESS = 0.3       # Base humidity weakness (scaled by porosity per material)
TEMP_WEAKNESS_MIN = 400.0     # Below this, no thermal weakening
TEMP_WEAKNESS_MAX = 800.0     # At this temp, maximum weakening
TEMP_WEAKNESS_FACTOR = 0.5    # At max temp, capacity drops to 50%

# Humidity diffusion
HUMIDITY_TICK_INTERVAL = 5    # Run humidity diffusion every N ticks
HUMIDITY_DIFFUSION_RATE = 0.05  # Slower than heat (water seeps, not flows)
HUMIDITY_SURFACE_LOSS = 0.03  # Evaporation at surface
HUMIDITY_SOURCE_LEVEL = 0.8   # Blocks adjacent to lava produce steam -> humidity

# Impact damage
IMPACT_DAMAGE_THRESHOLD = 3   # Minimum fall distance (cells) to cause impact damage
IMPACT_DAMAGE_FACTOR = 0.5    # Fraction of (fall_distance * weight) applied as impact load

# Heat convection: humidity movement carries heat
CONVECTION_RATE = 0.3         # Fraction of humidity flow that carries proportional heat

# Angle of repose: loose granular materials spread laterally
# Materials with porosity >= this threshold are considered "granular"
GRANULAR_POROSITY_THRESHOLD = 0.2  # dirt, sand, chalk, sandstone
REPOSE_TICK_INTERVAL = 2     # Run lateral spreading every N ticks (responsive)
MAX_SPREAD_PER_TICK = 3       # Max lateral moves per tick

# Coefficient of Thermal Expansion per voxel type
# Higher = more susceptible to thermal shock (gradient × CTE = stress)
# Glass/obsidian extremely vulnerable; metals are ductile and resist
VOXEL_CTE = {
    VOXEL_AIR: 0.0,
    VOXEL_DIRT: 0.005,
    VOXEL_STONE: 0.008,
    VOXEL_BEDROCK: 0.0,
    VOXEL_CORE: 0.0,
    VOXEL_SANDSTONE: 0.010,
    VOXEL_LIMESTONE: 0.009,
    VOXEL_SHALE: 0.012,
    VOXEL_CHALK: 0.015,
    VOXEL_SLATE: 0.007,
    VOXEL_MARBLE: 0.008,
    VOXEL_GNEISS: 0.006,
    VOXEL_GRANITE: 0.005,
    VOXEL_BASALT: 0.006,
    VOXEL_OBSIDIAN: 0.025,        # glass! extreme thermal shock vulnerability
    VOXEL_IRON_ORE: 0.004,
    VOXEL_COPPER_ORE: 0.005,
    VOXEL_GOLD_ORE: 0.006,
    VOXEL_MANA_CRYSTAL: 0.0,
    VOXEL_LAVA: 0.0,
    VOXEL_WATER: 0.0,
    VOXEL_IRON_INGOT: 0.003,
    VOXEL_COPPER_INGOT: 0.004,
    VOXEL_GOLD_INGOT: 0.005,
    VOXEL_ENCHANTED_METAL: 0.001,
    VOXEL_REINFORCED_WALL: 0.003,
    VOXEL_SPIKE: 0.003,
    VOXEL_DOOR: 0.001,
    VOXEL_TREASURE: 0.005,
    VOXEL_ROLLING_STONE: 0.005,
    VOXEL_TARP: 0.001,
    VOXEL_SLOPE: 0.008,
    VOXEL_STAIRS: 0.008,
    VOXEL_GOLD_BAIT: 0.004,
    VOXEL_HEAT_BEACON: 0.003,
    VOXEL_PRESSURE_PLATE: 0.002,
    VOXEL_IRON_BARS: 0.003,
    VOXEL_FLOODGATE: 0.002,
    VOXEL_ALARM_BELL: 0.004,
    VOXEL_FRAGILE_FLOOR: 0.015,      # chalky, thermally vulnerable
    VOXEL_PIPE: 0.003,
    VOXEL_PUMP: 0.003,
    VOXEL_STEAM_VENT: 0.025,         # obsidian-level thermal cycling
}

# Thermal stress constants
THERMAL_STRESS_TICK_INTERVAL = 10  # Same as structural (runs alongside)
THERMAL_FATIGUE_ACCUMULATION = 0.1 # Fraction of instantaneous ratio added to fatigue
THERMAL_FATIGUE_DECAY = 0.02       # Fatigue heals slowly when gradient is low
QUENCH_MULTIPLIER = 3.0            # Multiplier when water is adjacent (rapid cooling)
THERMAL_CRACK_THRESHOLD = 1.0      # When fatigue >= 1.0, block cracks

# Shock wave transmissivity per voxel type (inverse of absorption)
# 0.0 = absorbs all shock (ductile), 1.0 = transmits all (rigid/brittle)
VOXEL_SHOCK_TRANSMIT = {
    VOXEL_AIR: 0.0,
    VOXEL_DIRT: 0.1,
    VOXEL_STONE: 0.5,
    VOXEL_BEDROCK: 0.0,
    VOXEL_CORE: 0.0,
    VOXEL_SANDSTONE: 0.3,
    VOXEL_LIMESTONE: 0.4,
    VOXEL_SHALE: 0.6,
    VOXEL_CHALK: 0.2,
    VOXEL_SLATE: 0.55,
    VOXEL_MARBLE: 0.5,
    VOXEL_GNEISS: 0.45,
    VOXEL_GRANITE: 0.6,
    VOXEL_BASALT: 0.65,
    VOXEL_OBSIDIAN: 0.8,          # glass transmits shock extremely well
    VOXEL_IRON_ORE: 0.4,
    VOXEL_COPPER_ORE: 0.35,
    VOXEL_GOLD_ORE: 0.3,
    VOXEL_MANA_CRYSTAL: 0.0,
    VOXEL_LAVA: 0.0,
    VOXEL_WATER: 0.0,             # liquid absorbs shock
    VOXEL_IRON_INGOT: 0.2,
    VOXEL_COPPER_INGOT: 0.15,
    VOXEL_GOLD_INGOT: 0.1,
    VOXEL_ENCHANTED_METAL: 0.05,
    VOXEL_REINFORCED_WALL: 0.15,
    VOXEL_SPIKE: 0.3,
    VOXEL_DOOR: 0.1,
    VOXEL_TREASURE: 0.1,
    VOXEL_ROLLING_STONE: 0.5,
    VOXEL_TARP: 0.0,
    VOXEL_SLOPE: 0.5,
    VOXEL_STAIRS: 0.5,
    VOXEL_GOLD_BAIT: 0.3,
    VOXEL_HEAT_BEACON: 0.6,
    VOXEL_PRESSURE_PLATE: 0.7,
    VOXEL_IRON_BARS: 0.6,
    VOXEL_FLOODGATE: 0.7,
    VOXEL_ALARM_BELL: 0.4,
    VOXEL_FRAGILE_FLOOR: 0.2,
    VOXEL_PIPE: 0.5,
    VOXEL_PUMP: 0.6,
    VOXEL_STEAM_VENT: 0.8,
}

# Brittleness per voxel type (shatter vs crack on impact)
# 0.0 = always cracks (ductile), 1.0 = always shatters to air (brittle)
VOXEL_BRITTLENESS = {
    VOXEL_AIR: 0.0,
    VOXEL_DIRT: 0.0,
    VOXEL_STONE: 0.3,
    VOXEL_BEDROCK: 0.0,
    VOXEL_CORE: 0.0,
    VOXEL_SANDSTONE: 0.4,
    VOXEL_LIMESTONE: 0.35,
    VOXEL_SHALE: 0.6,
    VOXEL_CHALK: 0.8,
    VOXEL_SLATE: 0.5,
    VOXEL_MARBLE: 0.4,
    VOXEL_GNEISS: 0.3,
    VOXEL_GRANITE: 0.2,
    VOXEL_BASALT: 0.25,
    VOXEL_OBSIDIAN: 0.95,         # glass shatters spectacularly
    VOXEL_IRON_ORE: 0.3,
    VOXEL_COPPER_ORE: 0.25,
    VOXEL_GOLD_ORE: 0.2,
    VOXEL_MANA_CRYSTAL: 0.0,
    VOXEL_LAVA: 0.0,
    VOXEL_WATER: 0.0,
    VOXEL_IRON_INGOT: 0.05,
    VOXEL_COPPER_INGOT: 0.05,
    VOXEL_GOLD_INGOT: 0.02,
    VOXEL_ENCHANTED_METAL: 0.01,
    VOXEL_REINFORCED_WALL: 0.05,
    VOXEL_SPIKE: 0.1,
    VOXEL_DOOR: 0.05,
    VOXEL_TREASURE: 0.2,
    VOXEL_ROLLING_STONE: 0.2,
    VOXEL_TARP: 0.0,
    VOXEL_SLOPE: 0.3,
    VOXEL_STAIRS: 0.3,
    VOXEL_GOLD_BAIT: 0.15,
    VOXEL_HEAT_BEACON: 0.08,
    VOXEL_PRESSURE_PLATE: 0.05,
    VOXEL_IRON_BARS: 0.05,
    VOXEL_FLOODGATE: 0.05,
    VOXEL_ALARM_BELL: 0.10,
    VOXEL_FRAGILE_FLOOR: 0.8,       # chalky, shatters easily
    VOXEL_PIPE: 0.08,
    VOXEL_PUMP: 0.05,
    VOXEL_STEAM_VENT: 0.80,         # obsidian-level brittleness
}

# Impact cascade constants
SHOCK_ATTENUATION = 0.7           # Fraction of shock absorbed by each block
SHOCK_STRUCTURAL_FACTOR = 0.5     # How much shock contributes to structural load
MAX_SHOCK_PROPAGATION_STEPS = 5   # Max BFS depth for shock wave
SHATTER_THRESHOLD = 2.0           # shock/capacity ratio to shatter instead of crack
MAX_CASCADE_DEPTH = 3             # Max chain reaction levels per tick

# Water physics
WATER_TICK_INTERVAL = 2           # Run water flow every 2 ticks (responsive)
WATER_FLOW_RATE = 0.4             # Fraction of water_level transferred per tick laterally
WATER_SEEP_RATE = 0.02            # Rate at which water seeps through porous solids
WATER_PRESSURE_WEIGHT = 0.3       # Lateral pressure per unit of water depth
WATER_BURST_FACTOR = 1.5          # Pressure must exceed shear_strength * factor to burst
WATER_HUMIDITY_SOURCE = 0.9       # Water blocks set adjacent humidity (scaled by porosity)
WATER_TEMPERATURE = 20.0          # Default temperature of water blocks
WATER_EVAPORATION_RATE = 0.01     # Water level loss at surface (z=0) per tick
MAX_WATER_FLOW_PER_TICK = 3       # Max flow iterations per tick
WATER_LAVA_PRODUCT = 32           # VOXEL_OBSIDIAN produced when water meets lava

# ── New Block Gameplay Constants ─────────────────────────────────────
GOLD_BAIT_INTERACT_TICKS = 10     # Same as treasure grab
HEAT_BEACON_TEMPERATURE = 500.0   # Source temp (< lava 1000)
HEAT_BEACON_DAMAGE = 15           # Burn damage to non-immune intruders
PRESSURE_PLATE_TRIGGER_RANGE = 1  # Adjacent activation distance
ALARM_BELL_DETECTION_RANGE = 2    # LOS detection range (cells)
ALARM_BELL_COOLDOWN = 40          # Ticks between alarm triggers
FRAGILE_FLOOR_WEIGHT_THRESHOLD = 10  # block_state accumulation before collapse
STEAM_VENT_DAMAGE = 10            # Steam burn damage
STEAM_VENT_HEAT_PULSE = 150.0     # Heat added to cells above
STEAM_VENT_HUMIDITY_PULSE = 0.7   # Humidity added to cells above
STEAM_VENT_RANGE = 3              # Cells above to affect

# Pipe & pump constants
PIPE_CONDUCTIVITY_BASE = 0.5      # Base heat/humidity transfer rate through pipes
PUMP_CONVECTION_RATE = 0.3        # Active pumping multiplier
PUMP_TICK_INTERVAL = 5            # Pump operates every N ticks

# Pump direction encoding (stored in block_state)
PUMP_DIR_POS_X = 0
PUMP_DIR_NEG_X = 1
PUMP_DIR_POS_Y = 2
PUMP_DIR_NEG_Y = 3
PUMP_DIR_POS_Z = 4   # up (shallower)
PUMP_DIR_NEG_Z = 5   # down (deeper)

# Stone types that pipes can be built into
PIPEABLE_STONE_TYPES = frozenset({
    VOXEL_STONE, VOXEL_SANDSTONE, VOXEL_LIMESTONE, VOXEL_SHALE, VOXEL_CHALK,
    VOXEL_SLATE, VOXEL_MARBLE, VOXEL_GNEISS, VOXEL_GRANITE, VOXEL_BASALT,
    VOXEL_OBSIDIAN,
})
