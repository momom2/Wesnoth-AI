# constants.py
# Configuration constants for Wesnoth AI training

from pathlib import Path

# ============================================================================
# Paths
# ============================================================================

# Path to Wesnoth executable (Steam version on Windows)
WESNOTH_PATH = Path(r"C:\Program Files (x86)\Steam\steamapps\common\wesnoth\wesnoth.exe")

# Wesnoth userdata directory. This is where add-ons live, where Wesnoth
# writes its own log files, and where our file-based IPC happens.
# On Windows the default is "<user>/Documents/My Games/Wesnoth<version>".
WESNOTH_USERDATA_PATH = Path.home() / "Documents" / "My Games" / "Wesnoth1.18"
WESNOTH_LOGS_PATH = WESNOTH_USERDATA_PATH / "logs"

# Base directory for project
BASE_PATH = Path(__file__).parent  # Uses directory where constants.py is located

# Paths to project directories (source of truth — Wesnoth reads these via
# a directory junction installed into WESNOTH_USERDATA_PATH/data/add-ons/).
LOGS_PATH = BASE_PATH / "logs"
ADDONS_PATH = BASE_PATH / "add-ons" / "wesnoth_ai"
SCENARIOS_PATH = ADDONS_PATH / "scenarios"
LUA_PATH = ADDONS_PATH / "lua"
GAMES_PATH = ADDONS_PATH / "games"

# Where the add-on is installed for Wesnoth to find. Created as a junction
# (on Windows) or symlink (POSIX) pointing back at ADDONS_PATH so edits
# in the project tree show up live in Wesnoth.
ADDON_INSTALL_PATH = WESNOTH_USERDATA_PATH / "data" / "add-ons" / "wesnoth_ai"

# Training data paths
CHECKPOINTS_PATH = BASE_PATH / "training" / "checkpoints"
REPLAYS_PATH = BASE_PATH / "training" / "replays"

# ============================================================================
# Training Configuration
# ============================================================================

# Number of parallel games to run
NUM_PARALLEL_GAMES = 4

# Maximum actions per game before timeout
MAX_ACTIONS_PER_GAME = 2000

# Penalties
ACTION_PENALTY = 0.001  # Small penalty per action to encourage efficiency
TIMEOUT_PENALTY = 1.0   # Large penalty for timing out

# ============================================================================
# AI Model Configuration
# ============================================================================

# Transformer architecture
TRANSFORMER_D_MODEL = 256
TRANSFORMER_NUM_LAYERS = 6
TRANSFORMER_NUM_HEADS = 8
TRANSFORMER_D_FF = 1024
TRANSFORMER_DROPOUT = 0.1

# Memory
TRANSFORMER_MEMORY_SIZE = 128
MEMORY_STATE_SIZE = 256

# Map dimensions (for Caves of the Basilisk and similar)
MAX_MAP_WIDTH = 50
MAX_MAP_HEIGHT = 50

# Unit and action limits
MAX_UNIT_TYPES = 200  # More than enough for Knalgan + Drakes
MAX_ATTACKS = 4       # Max attacks per unit
MAX_RECRUITS = 10     # Max recruit list size

# Feature dimensions
TERRAIN_EMBEDDING_DIM = 16
UNIT_TYPE_EMBEDDING_DIM = 32
SPECIAL_EMBEDDING_DIM = 16

# ============================================================================
# Training Hyperparameters
# ============================================================================

# Experience replay
REPLAY_BUFFER_SIZE = 100000
REPLAY_BATCH_SIZE = 512

# Learning
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001

# Loss weights
POLICY_LOSS_WEIGHT = 1.0
VALUE_LOSS_WEIGHT = 0.5
CONSISTENCY_LOSS_WEIGHT = 0.1

# Training schedule
CHECKPOINT_FREQUENCY = 100   # Save every N games
REPLAY_SAVE_FREQUENCY = 100  # Save replay every N games
LOG_FREQUENCY = 10           # Log stats every N games

# ============================================================================
# IPC Configuration (mixed transport — see wesnoth_interface.py)
# ============================================================================
#
# Lua → Python (state): Lua std_print()s a framed block; it lands in
# <userdata>/logs/wesnoth-*.out.log; Python tails the file.
#
# Python → Lua (action): Python atomically writes a Lua chunk to
# ADDONS_PATH/games/<game_id>/action.lua. Lua reads it via
# wesnoth.read_file. Each action carries a monotonic `seq` field; Lua
# tracks last-executed seq and ignores stale payloads.

ACTION_FILE_NAME = "action.lua"

# How long either side will wait for the other before giving up.
STATE_TIMEOUT_SECONDS = 30.0
ACTION_TIMEOUT_SECONDS = 30.0

# How often the Python side polls for new log content (seconds). Small
# enough to keep turn latency low; large enough to avoid hogging CPU.
STATE_POLL_INTERVAL = 0.05

# ============================================================================
# Game Configuration
# ============================================================================

# Initial training configuration
FIRST_MAP = "2p_Caves_of_the_Basilisk"
FIRST_FACTIONS = ["Knalgan Alliance", "Drakes"]

# Faction recruit lists (for reference)
KNALGAN_RECRUITS = [
    "Dwarvish Guardsman", "Dwarvish Fighter", "Dwarvish Ulfserker",
    "Dwarvish Thunderer", "Thief", "Poacher", "Footpad", "Gryphon Rider"
]

DRAKE_RECRUITS = [
    "Drake Burner", "Drake Clasher", "Drake Glider",
    "Drake Fighter", "Saurian Skirmisher", "Saurian Augur"
]

# Starting gold
STARTING_GOLD = 100
VILLAGE_GOLD = 2

# ============================================================================
# Terrain Codes (Wesnoth 1.18.5)
# ============================================================================

# Basic terrain types (base layer)
TERRAIN_BASES = {
    'Aa': 'snow',
    'Ai': 'ice',
    'Gg': 'grassland',
    'Gs': 'semi_dry_grass',
    'Gd': 'dirt',
    'Gll': 'leaf_litter',
    'Ql': 'stones',
    'Qxu': 'unwalkable',
    'Xu': 'impassable_unwalkable',
    'Uu': 'underground_unwalkable',
    'Uh': 'hills_unwalkable',
    'Hh': 'hills',
    'Ha': 'snowy_hills',
    'Ms': 'snowy_mountains',
    'Mm': 'mountains',
    'Md': 'dry_mountains',
    'Xu': 'impassable',
    'Ql': 'cave',
    'Qxe': 'encampment',
    'Rr': 'road',
    'Re': 'dirt_road',
    'Ww': 'shallow_water',
    'Wo': 'deep_water',
    'Wot': 'deep_water_submerged',
    'Ss': 'swamp',
    'Ds': 'desert',
    'Dd': 'desert_village',
}

# Overlay terrain (structures, features)
TERRAIN_OVERLAYS = {
    'Fp': 'pine_forest',
    'Fpa': 'snowy_pine_forest',
    'Fms': 'snowy_mixed_forest',
    'Fdf': 'deciduous_forest',
    'Ft': 'tropical_forest',
    'Vh': 'human_village',
    'Vhh': 'human_hill_village',
    'Vhc': 'human_cave_village',
    'Vhr': 'human_snow_village',
    'Ve': 'elven_village',
    'Vea': 'elven_snow_village',
    'Vd': 'desert_village',
    'Vl': 'underground_village',
    'Vu': 'dwarven_village',
    'Vud': 'dwarven_mountain_village',
    'Vo': 'orcish_village',
    'Vda': 'drake_village',
    'Vaa': 'snow_village',
    'Vwm': 'merman_village',
    'Vhs': 'swamp_village',
    'Ce': 'castle',
    'Ch': 'castle_human',
    'Cv': 'castle_village',
    'Kh': 'keep_human',
    'Ke': 'keep',
    'Chs': 'snow_castle',
    'Chr': 'snow_keep',
}

# Special modifiers
TERRAIN_MODIFIERS = {
    'illuminated': 'ILLUMINATED',
    'shadowed': 'SHADOWED',
}
