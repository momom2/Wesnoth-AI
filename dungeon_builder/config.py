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

# Layer-slice transparency: offset from focus -> alpha
LAYER_ALPHA = {0: 1.0, 1: 0.6, 2: 0.3}
LAYER_MAX_VISIBLE_OFFSET = 2  # +/-2 visible, beyond hidden

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

# Non-diggable voxel types
NON_DIGGABLE = frozenset({VOXEL_AIR, VOXEL_BEDROCK, VOXEL_CORE, VOXEL_LAVA, VOXEL_WATER, VOXEL_REINFORCED_WALL})

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
}
MAX_CONCURRENT_DIGS = 5

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
}

# Max load capacity (compressive strength) per voxel type
VOXEL_MAX_LOAD = {
    VOXEL_AIR: 0.0,
    VOXEL_DIRT: 20.0,
    VOXEL_STONE: 80.0,
    VOXEL_BEDROCK: float("inf"),
    VOXEL_CORE: float("inf"),
    VOXEL_SANDSTONE: 40.0,
    VOXEL_LIMESTONE: 50.0,
    VOXEL_SHALE: 30.0,
    VOXEL_CHALK: 15.0,
    VOXEL_SLATE: 70.0,
    VOXEL_MARBLE: 85.0,
    VOXEL_GNEISS: 75.0,
    VOXEL_GRANITE: 120.0,
    VOXEL_BASALT: 110.0,
    VOXEL_OBSIDIAN: 60.0,
    VOXEL_IRON_ORE: 65.0,
    VOXEL_COPPER_ORE: 55.0,
    VOXEL_GOLD_ORE: 45.0,
    VOXEL_MANA_CRYSTAL: float("inf"),
    VOXEL_LAVA: 0.0,
    VOXEL_WATER: 0.0,
    VOXEL_IRON_INGOT: 100.0,
    VOXEL_COPPER_INGOT: 80.0,
    VOXEL_GOLD_INGOT: 50.0,
    VOXEL_ENCHANTED_METAL: 150.0,
    VOXEL_REINFORCED_WALL: 200.0,  # strongest buildable block
    VOXEL_SPIKE: 40.0,
    VOXEL_DOOR: 80.0,
    VOXEL_TREASURE: 30.0,
    VOXEL_ROLLING_STONE: 100.0,
    VOXEL_TARP: 5.0,              # breaks under almost any block weight
    VOXEL_SLOPE: 80.0,
    VOXEL_STAIRS: 80.0,
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
}

# Tensile strength per voxel type (governs cantilever/bending failure)
# Failure: load × span / 2 > tensile_strength
# Stone has very low tensile strength; metals are far superior
VOXEL_TENSILE_STRENGTH = {
    VOXEL_AIR: 0.0,
    VOXEL_DIRT: 2.0,
    VOXEL_STONE: 10.0,
    VOXEL_BEDROCK: float("inf"),
    VOXEL_CORE: float("inf"),
    VOXEL_SANDSTONE: 5.0,
    VOXEL_LIMESTONE: 8.0,
    VOXEL_SHALE: 4.0,
    VOXEL_CHALK: 2.0,
    VOXEL_SLATE: 12.0,
    VOXEL_MARBLE: 10.0,
    VOXEL_GNEISS: 11.0,
    VOXEL_GRANITE: 15.0,
    VOXEL_BASALT: 14.0,
    VOXEL_OBSIDIAN: 6.0,       # brittle glass, snaps easily
    VOXEL_IRON_ORE: 8.0,
    VOXEL_COPPER_ORE: 7.0,
    VOXEL_GOLD_ORE: 5.0,
    VOXEL_MANA_CRYSTAL: float("inf"),
    VOXEL_LAVA: 0.0,
    VOXEL_WATER: 0.0,
    VOXEL_IRON_INGOT: 40.0,    # wrought iron, excellent in tension
    VOXEL_COPPER_INGOT: 30.0,
    VOXEL_GOLD_INGOT: 15.0,
    VOXEL_ENCHANTED_METAL: 50.0,
    VOXEL_REINFORCED_WALL: 60.0,
    VOXEL_SPIKE: 20.0,
    VOXEL_DOOR: 30.0,
    VOXEL_TREASURE: 5.0,
    VOXEL_ROLLING_STONE: 15.0,
    VOXEL_TARP: 1.0,
    VOXEL_SLOPE: 10.0,
    VOXEL_STAIRS: 10.0,
}

# Shear strength per voxel type (lateral load capacity)
# Typically ~15-20% of compressive for stone, ~30-40% for ductile metals
VOXEL_SHEAR_STRENGTH = {
    VOXEL_AIR: 0.0,
    VOXEL_DIRT: 4.0,
    VOXEL_STONE: 16.0,
    VOXEL_BEDROCK: float("inf"),
    VOXEL_CORE: float("inf"),
    VOXEL_SANDSTONE: 6.0,
    VOXEL_LIMESTONE: 8.0,
    VOXEL_SHALE: 4.5,
    VOXEL_CHALK: 2.0,
    VOXEL_SLATE: 10.0,
    VOXEL_MARBLE: 12.0,
    VOXEL_GNEISS: 11.0,
    VOXEL_GRANITE: 20.0,
    VOXEL_BASALT: 18.0,
    VOXEL_OBSIDIAN: 8.0,
    VOXEL_IRON_ORE: 13.0,
    VOXEL_COPPER_ORE: 11.0,
    VOXEL_GOLD_ORE: 7.0,
    VOXEL_MANA_CRYSTAL: float("inf"),
    VOXEL_LAVA: 0.0,
    VOXEL_WATER: 0.0,
    VOXEL_IRON_INGOT: 40.0,
    VOXEL_COPPER_INGOT: 32.0,
    VOXEL_GOLD_INGOT: 15.0,
    VOXEL_ENCHANTED_METAL: 60.0,
    VOXEL_REINFORCED_WALL: 50.0,
    VOXEL_SPIKE: 15.0,
    VOXEL_DOOR: 25.0,
    VOXEL_TREASURE: 5.0,
    VOXEL_ROLLING_STONE: 20.0,
    VOXEL_TARP: 1.0,
    VOXEL_SLOPE: 12.0,
    VOXEL_STAIRS: 12.0,
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
