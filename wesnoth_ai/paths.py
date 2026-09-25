"""The repository root and the data files the code reads from it.

Library code finds every repo-relative location here. A module that
counts `Path(__file__).parent` hops to reach the root points somewhere
else the day it moves one directory deeper, and nothing fails at the
move; this file computes the root once, from its own fixed place in the
package, and tests/test_paths.py keeps every other package module off
`__file__`.

It imports nothing but pathlib: the torch-free drivers (run_elo_batch,
eval_procedure, turn_search_config, host_resources) import it.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Pinned 1.18.4 scrapes (CLAUDE.md, "Wesnoth data provenance").
UNIT_STATS_PATH = REPO_ROOT / "unit_stats.json"
TERRAIN_DB_PATH = REPO_ROOT / "terrain_db.json"

# The WML-only copy of the Wesnoth data tree the simulator reads.
WESNOTH_SRC_DIR = REPO_ROOT / "wesnoth_src"

# The project's own add-ons: the live bridge's Lua side and the drills.
ADDONS_DIR = REPO_ROOT / "add-ons"

CONFIGS_DIR = REPO_ROOT / "configs"

# The command-line scripts. Code that launches a script as a subprocess
# names it here: a script keeps its path in tools/ when its code moves
# into the package.
TOOLS_DIR = REPO_ROOT / "tools"

# The save scaffold and the per-scenario templates that replay export
# composes and tools/build_scenario_templates.py writes.
TEMPLATES_DIR = TOOLS_DIR / "templates"
SCENARIO_TEMPLATES_DIR = TEMPLATES_DIR / "scenarios"

# The Rust core's source, read for the `__phase__` it declares.
RUST_CORE_SRC_DIR = REPO_ROOT / "rust" / "wesnoth_core" / "src"

# The imitation corpus of human games (not in git).
IMITATION_DATASET_DIR = REPO_ROOT / "replays_dataset_imitation"
