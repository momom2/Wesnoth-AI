"""Tunables for Wesnoth AI.

Deliberately small. Training hyperparameters, model sizes, and scenario
reference data from the earlier (deleted) ML code are gone — Phase 3
reintroduces them alongside the policy code that actually uses them.
"""

from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

# Wesnoth executable. Steam install on Windows by default.
WESNOTH_PATH = Path(
    r"C:\Program Files (x86)\Steam\steamapps\common\wesnoth\wesnoth.exe"
)

# Wesnoth userdata — where add-ons live, where Wesnoth writes its logs,
# and where our Lua side reads the action.lua file via the add-on
# directory junction.
WESNOTH_USERDATA_PATH = Path.home() / "Documents" / "My Games" / "Wesnoth1.18"
WESNOTH_LOGS_PATH = WESNOTH_USERDATA_PATH / "logs"

# Project layout — all source files of the add-on live here. Wesnoth
# sees them via the junction installed by main.install_addon().
BASE_PATH       = Path(__file__).parent
LOGS_PATH       = BASE_PATH / "logs"
ADDONS_PATH     = BASE_PATH / "add-ons" / "wesnoth_ai"
SCENARIOS_PATH  = ADDONS_PATH / "scenarios"
LUA_PATH        = ADDONS_PATH / "lua"
GAMES_PATH      = ADDONS_PATH / "games"

# Installed-add-on location; a directory junction / symlink back to
# ADDONS_PATH, managed by main.install_addon().
ADDON_INSTALL_PATH = WESNOTH_USERDATA_PATH / "data" / "add-ons" / "wesnoth_ai"

# Artifacts from training runs. Phase 1 doesn't produce checkpoints;
# kept so the layout is ready for Phase 3.
CHECKPOINTS_PATH = BASE_PATH / "training" / "checkpoints"
REPLAYS_PATH     = BASE_PATH / "training" / "replays"

# ----------------------------------------------------------------------
# Run configuration
# ----------------------------------------------------------------------

NUM_PARALLEL_GAMES   = 4      # Each game is a separate Wesnoth process.
                              # Lua generates a random per-process game_id
                              # so parallel games don't collide on the
                              # IPC directory. See training_scenario.cfg
                              # preload.
# Was 2000 — with a 16-way actor pool (units + recruits + end_turn),
# ~1/16 of random actions end the turn, so 2000 actions ≈ 60 turns.
# That's far longer than necessary, and a random policy never finds a
# leader kill before hitting the cap — meaning every game terminated
# on TIMEOUT (terminal reward 0) and the ±1 training signal never
# fired. 500 actions ≈ 15-30 turns, still enough room for proper
# engagement but produces way more terminations per hour and raises
# the probability of a stumbled-upon kill registering in the queue.
MAX_ACTIONS_PER_GAME = 500

# How often to emit aggregated stats / save checkpoints. Only
# checkpoints fire when the policy is trainable; stats always.
LOG_FREQUENCY        = 10
CHECKPOINT_FREQUENCY = 100

# ----------------------------------------------------------------------
# IPC (see wesnoth_interface.py)
# ----------------------------------------------------------------------
#
# Lua → Python (state): Lua std_print()s a framed block; it lands in
#   <userdata>/logs/wesnoth-*.out.log; Python tails the file.
# Python → Lua (action): Python atomically writes a Lua chunk to
#   <game_dir>/action.lua; Lua reads via wesnoth.read_file; a monotonic
#   `seq` field lets Lua distinguish fresh from stale.

ACTION_FILE_NAME       = "action.lua"
STATE_TIMEOUT_SECONDS  = 30.0
ACTION_TIMEOUT_SECONDS = 30.0
STATE_POLL_INTERVAL    = 0.01   # lowered from 0.05 — Python's 50 ms tick
                                # added an avg 25 ms tail to every
                                # read_state. 10 ms cuts that to ~5 ms
                                # and matches the Lua-side POLL_MS.

# ----------------------------------------------------------------------
# Encoder feature normalization
# ----------------------------------------------------------------------
# Each scalar feature is divided by its NORM constant before being
# concatenated into the unit / global token. The NORMs are chosen so
# default-era values land in the [0, ~1] range (the transformer's
# input scale). Override at config-time to support custom eras whose
# unit costs / HP exceed these (a cost-200 unit feeds in as 2.5 with
# the default COST_NORM=80; clipping is fine but you might want to
# tune COST_NORM up so the spread of cost values is more uniform).

HP_NORM       = 80.0    # full Walking-Corpse's HP =~22; ladder Drake max =~70
MOVES_NORM    = 10.0    # ladder caps around 7 (Wolf Rider with quick)
EXP_NORM      = 150.0   # 4-level units (e.g. Lich) need ~150 XP to AMLA
COST_NORM     = 80.0    # ladder caps around 60 (Yeti, Lich)
GOLD_NORM     = 500.0   # default-era 2p starts at 100; 500 covers
                        # late-game gold accumulation
INCOME_NORM   = 50.0
VILLAGES_NORM = 30.0    # large 4p maps; 2p ladder caps ~15
TURN_NORM     = 60.0    # default 2p ladder turn limit ~30, 60 for safety

# Combat-oracle bias on attack-action logits. Raises the prior on
# attacks where expected net damage is positive (so the policy
# learns to attack when the trade is favorable). 0.1 = "moderate
# bias"; 0 = oracle off; >0.3 = aggressive override of learned
# logits. See action_sampler's call site.
COMBAT_LOGIT_ALPHA = 0.1

# Pre-seeded faction vocab. The encoder pre-binds these to specific
# embedding rows so cross-replay supervised training stays
# consistent (a state that happens to feed Rebels first wouldn't
# alone determine its row). Empty string is reserved for
# "unknown / unset" -> id 0; the six default-era factions follow.
# Era mods can extend the list (the encoder's `MAX_FACTIONS=32`
# bound leaves headroom). Order MATTERS for backwards compatibility
# with saved checkpoints; appending is safe, reordering is not.
DEFAULT_FACTIONS = (
    "",
    "Drakes",
    "Knalgan Alliance",
    "Rebels",
    "Loyalists",
    "Northerners",
    "Undead",
)
