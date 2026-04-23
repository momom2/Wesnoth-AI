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

NUM_PARALLEL_GAMES   = 1      # Phase 1 only tested with one; Phase 3 can scale.
MAX_ACTIONS_PER_GAME = 2000   # Safety rail against runaway games.

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
STATE_POLL_INTERVAL    = 0.05
