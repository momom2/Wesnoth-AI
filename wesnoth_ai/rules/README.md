# wesnoth_ai/rules: the game's rules as data

This package reads what Wesnoth's own data says about a game: the terrain
table, a scenario's `.cfg` and map, and the WML that describes a starting
state. From them it builds the starting `GameState` of a generated game,
and it checks its reading of the scenarios against the game's own. Playing
a game from that state is the simulator's work (`tools/wesnoth_sim.py`).

## Quickstart

```python
import random

from wesnoth_ai.rules.scenario_cfg import load_scenario_wml
from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate, random_setup
from wesnoth_ai.rules.terrain_resolver import mvt_cost

setup = random_setup(random.Random(0))           # a Ladder map, two factions and leaders
gs = build_scenario_gamestate(setup)             # its starting GameState
root = load_scenario_wml("multiplayer_Hamlets")  # the .cfg, preprocessed and parsed
mvt_cost("Gs^Fp", {"flat": 1, "forest": 2})      # 2: forest on grass costs the worse of the two
```

`WesnothSim(gs, scenario_id=setup.scenario_id)` then fires the scenario's
prestart and start events and plays the game.

The command lines:

    python tools/analysis/expansion_diff.py     # our macro expansion against the game's
    python tools/analysis/scenario_surface.py   # every declared attribute and what reads it
    python tools/build_scenario_templates.py    # rebuild tools/templates/scenarios/

## Modules

| module | what it holds | entry points |
|---|---|---|
| `terrain_resolver.py` | A terrain code's movement cost, defense, healing, light, hide cover and terrain set, resolved through the engine's alias rules from `terrain_db.json`. | `mvt_cost`, `def_pct`, `terrain_heals`, `terrain_light_bonus`, `hides_cover`, `terrain_members`, `split_start_position` |
| `wml_state.py` | The one reader of the WML that describes a starting state: sides, units, villages, time of day, the scenario's economy and map. Replay reconstruction and generation both read through it. | `read_side`, `read_unit`, `read_villages`, `read_tod`, `scenario_economy`, `resolve_map_file`, `split_map_grid`, `map_starting_positions` |
| `scenario_cfg.py` | A scenario's `.cfg` found by its id and read the way the game reads it: preprocessor conditionals, macro expansion (the core macros, the add-on's utility files, the file's own), parse. | `load_scenario_wml`, `parse_scenario_cfg`, `find_scenario_cfg_path`, `evaluate_conditionals` |
| `scenario_pool.py` | The scenarios generated games are played on (the 21-map Ladder whitelist, the mini maps), the default era's factions and leaders, and the starting `GameState` of one game. | `random_setup`, `ScenarioSetup`, `build_scenario_gamestate`, `load_factions`, `FORCED_FACTION` |
| `scenarios.py` | The competitive 2p scenarios of the data tree, which filter the human-replay corpus. | `COMPETITIVE_2P_SCENARIOS`, `is_competitive_2p` |
| `build_scenario_templates.py` | The per-scenario save templates the replay exporter composes, built by the game's own preprocessor (`wesnoth -p`). | `main` (the command line), `run_preprocessor`, `transform` |
| `expansion_diff.py` | Our expansion of each pool scenario compared with the game's (the templates), the differences grouped into clusters. | `compare`, `clusters`, `main` |
| `scenario_surface.py` | Every tag and attribute of the scenarios we build or rebuild, each classified in `tests/data/scenario_surface.json` and, when we read it, bound to the code that does. | `surface`, `unknowns`, `missing_readers`, `main` |

Related code outside the package: the WML parser (`parse_wml`, `WMLNode`)
is in `tools/replay_extract.py`; the unit database (`unit_stats.json`,
read by `_stats_for`) and the map parser (`parse_map_data`,
`parse_terrain_codes`) are in `tools/replay_dataset.py`; the scenario
event interpreter is `tools/scenario_events.py`.
docs/refactor_plan_20260925.md gives each its place.

## Invariants

- `unit_stats.json` and `terrain_db.json` are pinned scrapes of Wesnoth
  1.18.4 (CLAUDE.md, "Wesnoth data provenance"). `tools/scrape_terrain.py`
  and `tools/scrape_unit_stats.py` run only on a checkout of the 1.18.4
  tag, never on `wesnoth_src/`.
- Every data location comes from `wesnoth_ai.paths`; no module here reads
  its own `__file__` (tests/test_paths.py).
- WML is 1-indexed. `wml_state` and `scenario_pool` return the 0-indexed
  positions the rest of the code uses; `scenario_cfg` returns the WML tree
  as written.
- One reading of each thing: a scenario through `scenario_cfg`, the side,
  unit and village WML through `wml_state`, a terrain code through
  `terrain_resolver`, whose `hides_cover` is also the source of the Rust
  core's hide flags.
- Nothing is dropped silently: an unknown macro, a non-default board
  schedule or a scenario that sets a quick-leader gate warns once, and
  raises under `WESNOTH_STRICT_WML=1`.
- Our expansion matches the game's except for the divergences
  `tests/data/expansion_diff_expected.json` records with a reason. Every
  declared attribute is classified in `tests/data/scenario_surface.json`,
  and a MODELLED one names its reader as `path:symbol`: a reader that moves
  takes its manifest entries with it.
- `__init__.py` imports nothing, so importing one module loads that module
  and what it imports; the torch-free drivers rely on it.

## Tests

- Terrain: `test_terrain_overlay_resolution`, `test_hide_cover`,
  `test_terrain_multi_hot`, `test_start_positions`.
- Reading WML and scenarios: `test_wml_state`, `test_scenario_economy`,
  `test_scenario_cfg_case`, `test_preprocessor_conditionals`,
  `test_bare_clone_time_areas`, `test_time_areas`,
  `test_mini_tentacle_spawns`, `test_scenario_init_reads`.
- The detectors: `test_expansion_diff`, `test_scenario_surface`,
  `test_template_builder`.
- The pool's starting states: `test_scenario_state_snapshot` (a
  fingerprint of every pool scenario's starting state),
  `test_elo_ladder_maps`.

```
pytest tests/test_terrain_overlay_resolution.py tests/test_hide_cover.py tests/test_wml_state.py tests/test_scenario_cfg_case.py tests/test_preprocessor_conditionals.py tests/test_expansion_diff.py tests/test_scenario_surface.py tests/test_scenario_state_snapshot.py
```

`tools/scenario_init_oracle.py` compares each pool scenario's starting
state with the one real Wesnoth builds. It launches the game, so it runs
only by hand, with the user's agreement.
