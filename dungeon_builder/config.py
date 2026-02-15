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

# Crafted materials
VOXEL_IRON_INGOT = 60
VOXEL_COPPER_INGOT = 61
VOXEL_GOLD_INGOT = 62
VOXEL_ENCHANTED_METAL = 63

# Non-diggable voxel types
NON_DIGGABLE = frozenset({VOXEL_AIR, VOXEL_BEDROCK, VOXEL_CORE, VOXEL_LAVA})

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
    VOXEL_IRON_INGOT: 0.0,
    VOXEL_COPPER_INGOT: 0.0,
    VOXEL_GOLD_INGOT: 0.0,
    VOXEL_ENCHANTED_METAL: 0.0,
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
    VOXEL_IRON_INGOT: 0.8,
    VOXEL_COPPER_INGOT: 0.85,
    VOXEL_GOLD_INGOT: 0.9,
    VOXEL_ENCHANTED_METAL: 0.5,
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
    VOXEL_IRON_INGOT: (0.70, 0.55, 0.50, 1.0),
    VOXEL_COPPER_INGOT: (0.75, 0.50, 0.30, 1.0),
    VOXEL_GOLD_INGOT: (0.95, 0.85, 0.30, 1.0),
    VOXEL_ENCHANTED_METAL: (0.40, 0.60, 0.90, 1.0),
}

# Intruder defaults
INTRUDER_DEFAULT_HP = 50
INTRUDER_DEFAULT_SPEED = 1
INTRUDER_DEFAULT_DAMAGE = 5
INTRUDER_RETREAT_THRESHOLD = 0.2  # Retreat at 20% HP
INTRUDER_SPAWN_INTERVAL = 200     # Ticks between spawns (10 seconds)
INTRUDER_ATTACK_INTERVAL = 20     # Ticks between attacks on core (1 second)
MAX_INTRUDERS = 10

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
    VOXEL_IRON_INGOT: 15.0,
    VOXEL_COPPER_INGOT: 14.0,
    VOXEL_GOLD_INGOT: 18.0,
    VOXEL_ENCHANTED_METAL: 10.0,
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
    VOXEL_IRON_INGOT: 100.0,
    VOXEL_COPPER_INGOT: 80.0,
    VOXEL_GOLD_INGOT: 50.0,
    VOXEL_ENCHANTED_METAL: 150.0,
}

# Load distribution weights
LOAD_DIST_BELOW = 0.60       # 60% to block directly below
LOAD_DIST_LATERAL = 0.05     # 5% to each of 4 cardinal-below neighbors
BUTTRESS_FACTOR = 0.03       # Each solid same-level neighbor reduces load 3%

# Environmental weakness factors
HUMIDITY_WEAKNESS = 0.3       # At humidity=1.0, capacity drops to 70%
TEMP_WEAKNESS_MIN = 200.0     # Below this, no thermal weakening
TEMP_WEAKNESS_MAX = 800.0     # At this temp, maximum weakening
TEMP_WEAKNESS_FACTOR = 0.5    # At max temp, capacity drops to 50%
