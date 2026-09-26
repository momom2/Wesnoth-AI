# Refactor plan: one package per system (2026-09-25)

The user's standing backlog item: refactor the project for easier
navigation, documentation and separation of each system. This plan comes
from an inventory of the tree at 0.6.7, commit 90ff321
(docs/refactor_inventory_20260925.md: an import graph over the 423 tracked
Python files, every reference in text files, every argparse flag; a
read-only crawl, 2026-09-25).

## What is wrong now

- Library code lives in `tools/` beside one-off scripts: 165 files and
  76.1k lines there, of which about 50k are library code (the simulator,
  the search, the actor pool, the whole evaluation stack), against 30
  files in `wesnoth_ai/`. `wesnoth_ai` imports `tools` in 23 module pairs,
  so the "library" does not run without `tools/` on `sys.path`.
- The de facto API is private names: `replay_dataset._apply_command` has
  24 production importers, `_build_initial_gamestate` 24, `eval_sim._load_policy`
  18; production code imports 116 private names across modules.
- The live generation path runs through the legacy CLI: the actor pool
  plays its games with `_play_one_game_safe` from the 4,088-line
  `sim_self_play.py`, whose trainer is quarantined.
- 42 files exceed the 600-line target, mostly because of 26 functions of
  250 lines or more (`sim_self_play.main` 1,895, `replay_extract.extract_replay`
  1,211, `supervised_train.train` 1,099, `run_elo_batch.main` 947,
  `replay_dataset._apply_command` 783). Splitting files without splitting
  these functions only moves the problem.
- Duplicates that can drift: five loaders of `unit_stats.json` (each with
  its own fallback), six checkpoint-to-policy loaders, eight AUC functions,
  the az trainer recipe copied line for line into a benchmark that six
  tests use.
- No entry document for a system: the Architecture section of CLAUDE.md
  is 85 lines beside a 780-line status log.

## Target layout

All library code under `wesnoth_ai/<system>/`; no library module imports
`tools/`. A module with a command line keeps its `tools/` file name as a
thin wrapper (argparse, then one call into the package), so every command
in the docs and the box scripts keeps working; a library-only module
leaves `tools/` entirely, since a re-export shim would take the tests'
patches into its own namespace (step 0's finding, "Moving a module"
item 6). `tools/analysis/` and
`scripts/` keep their paths. Each system package has a README: what it
does, its entry points, its invariants, its tests.

```
wesnoth_ai/
  paths.py      the repo root and its data files (replaces 24 __file__-depth lookups)
  constants.py  model constants, OBSERVATION_EPOCH
  rules/        unit and terrain data (one unit_db), map data, WML, scenario building
  sim/          game state, combat, abilities, traits, setup, advancement, the command
                applier, events and effects, pathfinding, vision and observation,
                the simulator, the Rust-core adapter, game records
  replays/      imitation labels and datasets, extraction, export to Wesnoth saves,
                the value corpus
  model/        encoder, network, outputs, packed trunk, unit vocabulary
  inference/    the inference seam, server priors, leaf wire, graphed serving, devices
  policy/       action sampling and legality, the transformer policy and checkpoint I/O,
                the raw player
  search/       MCTS, the MCTS policy, draw tiebreak
  training/     the trainer step, signal telemetry, the imitation trainer and pre-encoding
  selfplay/     the game loop, the actor pool and its protocol, host resources
  eval/         provenance, the match driver, players, workers, the inference server,
                ratings, the catalog, the reference player, the turn-level gap
  bridge/       live Wesnoth: the interface, state conversion, the RCA eval
tools/          command-line wrappers only, same file names; tools/analysis unchanged
scripts/        box scripts and their shared library (scripts/box/)
tests/          helpers/ (shared fixtures), oracles/ (reference implementations)
```

Quarantined code (quarantine/INVENTORY.md) stays where it is, as the
quarantine rule says, and imports from the new packages; live code stops
importing it where it can (the actor loop imports the plan tournament and
the turn-commit policy today; eval's searched players default to TCS).

## What a move must not break

- Paths computed from `__file__` depth (24 in the modules that move):
  moving a module one directory deeper silently retargets `parent.parent`.
  `wesnoth_ai/paths.py` first, every lookup through it.
- Pickles that name module paths: pre-encoded corpus records
  (`wesnoth_ai.encoder.RawEncoded`, `tools.replay_dataset.ActionIndices`),
  holdout sidecars, anchor caches, bench experiences. Every loader goes
  through one unpickler that maps old module paths to new ones, with a
  test that loads a record written under the old paths.
- 93 `monkeypatch.setattr(module, name)` calls in tests: a function that
  leaves its module is no longer patched, and the test may pass without
  testing anything. The codemod rewrites the targets; each moved module's
  tests run with a check that every patched name exists where patched.
- 15 tests read source by path or glob; `tests/data/scenario_surface.json`
  names 75 readers by `tools/<file>.py:symbol`: both resolve through the
  module after the move.
- Torch-free drivers (`run_elo_batch`, `eval_procedure`,
  `turn_search_config`, `host_resources`) must stay torch-free: package
  `__init__.py` files import nothing.
- Rust doc comments name Python paths (22); the build does not depend on
  them, the text is updated with the move.

## Order of steps

Each step is one branch that ends with `ruff check .` clean, the fast
tier green and a green CI run; a step moves or extracts, it never changes
behaviour, and the parity, sweep and snapshot tests guard it.

0. Make moves safe, no moves yet: `paths.py`; source-scanning tests
   through the module; shared test fixtures into `tests/helpers/`; the
   mapping unpickler; the import codemod (an AST rewrite of import
   statements and monkeypatch targets, with a dry-run report).
1. Extractions that remove the worst couplings, no directory moves:
   the game loop out of `sim_self_play`; `_load_policy` and the match
   helpers into one eval module; provenance helpers out of
   `run_elo_batch`/`elo_eval_game` (breaks their import cycle); one
   `configure_az_trainer` (removes the benchmark's copy); one unit_db
   (each call site keeps its fallback); `harvest_states`/`load_states`
   out of the benchmark scripts; the actor protocol constants into one
   module (breaks the actor_pool/actor_stream cycle).
2. The deletions the owner approves (BACKLOG "Decision: deletions"), so
   nothing dead is moved.
3-8. The moves, one system per step: rules, sim (with the
   `replay_dataset` split), replays, model and inference, policy, search,
   training and selfplay, eval and bridge.
9. Function splits, one per commit, pure moves.
10. Docs: each package README, CLAUDE.md's Architecture and run lines,
   README's layout, `docs/README.md`.

About 1,000 import statements and 260 files change in the moves of
library code out of `tools/`, and about 760 more statements if the flat
`wesnoth_ai` modules also move into subpackages; the codemod makes them
mechanical, the tests and CI make them checked.

## Open branches

Steps 3 and later conflict with `exp/turn-value` (it adds about 1,800
lines of library code to `tools/` and edits `game_record.py` and
`turn_gap.py`) and with the parked `exp/xod-dominance` (new files importing
`replay_dataset`, `combat_outcomes`, `game_record`, `swap_detector`,
`rewards`). The moves start after `exp/turn-value` lands its result;
`exp/xod-dominance` takes a rewrite pass with the codemod when it is
revived. Steps 0 to 2 conflict with neither.

## Decisions for the user

- Deletions: the candidates and their evidence are in BACKLOG
  "Decision: deletions" (extended 2026-09-25): 48 entry points nobody runs
  (10,111 lines), 23 functions and 6 methods with no caller, unread
  configs and constants, about 40 test files that test only dead or
  quarantined code.
- The quarantine rule: moving or deleting quarantined code needs a ruling
  that lifts "the code stays where it is". Until then it stays in
  `tools/`.
- Scope: moving the flat `wesnoth_ai` modules (encoder, model, ...) into
  subpackages is optional; this plan moves the simulator's (classes,
  combat, visibility, observe, game_core) with the sim step and leaves the
  rest to the user's choice.

## Step 0, delivered (2026-09-26, branch refactor/step0-safe-moves)

- `wesnoth_ai/paths.py` holds the repo root and every location library
  code reads from it (unit_stats.json, terrain_db.json, wesnoth_src/,
  add-ons/, configs/, tools/ for launched scripts, the save templates, the
  Rust source, the imitation corpus). The 24 library modules that counted
  `__file__` hops take their locations from it; in them `__file__` is left
  only in `sys.path` bootstraps. tests/test_paths.py fails when a
  `wesnoth_ai` module other than paths.py reads `__file__`, when paths.py
  names a location that does not exist, or when a torch-free driver
  (run_elo_batch, eval_procedure, turn_search_config, host_resources)
  imports torch.
- The tests that read source take it from the imported module
  (`inspect.getsource`), or walk a directory recursively through
  `tests/helpers/source_tree.source_files`, which fails on a directory
  with no matching file.
- `tests/helpers/` holds the shared fixtures, imported as
  `helpers.<module>`.
- Every reader of pickled project objects goes through
  `wesnoth_ai.unpickle.load` / `loads`, which map module paths through
  `MOVED_MODULES`; tests/test_unpickle.py fails when a production tree
  reads a pickle any other way.
- `tools/dev/move_module.py` is the codemod.

## Step 1, delivered (2026-09-26, branch refactor/step1-extractions)

Six extractions, one commit each. The names keep their spelling, private
ones included, and their bodies byte for byte; every importer imports
from the new module and the old one re-exports nothing.

- `tools/selfplay_game.py`: the game loop out of sim_self_play
  (`GameOutcome`, `play_one_game`, `_play_one_game_safe`, `_worker_loop`,
  `k_median_of`, `_roll_max_turns`, the recruit-cost and bounce helpers,
  `_leader_of`, `_update_closest_approach`, `_outcome_for`) and
  `VALIDATION_EXPORTER`, which sim_self_play's main sets on the module
  (`from tools import selfplay_game`).
- `tools/eval_players.py`: `_load_policy`, `peek_checkpoint_arch`,
  `CHECKPOINT_STRUCT_FLAGS`, `_PolicyPair`, `_play_one_eval_game` and
  `GameResult` out of eval_sim.
- `tools/eval_provenance.py`, torch-free: the estimand fields and their
  refusals, `file_sha256` and `spec_sha256` out of run_elo_batch,
  `_pt_config` out of elo_eval_game.
- `tools/az_recipe.py`: `configure_az_trainer`, the one copy of the az
  loop's loss settings (az_loop, bench_train_step, six tests).
- `tools/bench_states.py`: `load_states`, `reconstruct_boundary` and the
  default manifest and dataset out of bench_pipeline, `harvest_states`
  out of bench_infer.
- `tools/actor_protocol.py`, standard library only: the actor and serve
  messages and `ActorFatalError`.

Cycles 4 (the actor modules) and 5 (the eval driver) of the inventory's
(c2) are gone; 1 to 3 remain. The one unit_db of step 1 waits: another
branch edits replay_dataset and replay_extract.

For the moves: the new modules log under their own names
("selfplay_game", "eval_players", "bench_states"), which
tools/sim_dummy_smoke.py and tests/test_player_sides.py spell. actor_pool
still re-exports the workers' other names (`_IPCInferenceClient`,
`_actor_loop`, `_serve_loop`, `_BatchPicker`, ...) through `__all__`;
tests import them from there and test_actor_pool_lifecycle patches
`actor_pool._serve_loop`. test_plan_tournament assigns
`elo_eval_game._load_policy` by hand, a patch of the importer's binding
that no dry run on the defining module lists.

## Moving a module (step 3 onwards)

1. Create the destination package: an `__init__.py` with a docstring and
   no imports, since importing any module of the package runs it, in the
   torch-free drivers too.
2. `python tools/dev/move_module.py OLD NEW`, for example
   `tools.replay_dataset wesnoth_ai.sim.replay_dataset`. The dry run
   changes nothing and prints every rewrite, every refusal with its
   file:line, the patches made through the module object (a name not bound
   at the module's top level is flagged), the moved module's `__file__`,
   `__name__` and `sys.path` lines, and every other mention of the module.
   Resolve the refusals by hand and run it again until there are none.
3. The same command with `--apply` rewrites the importers, moves the file
   with `git mv` (the rename is staged) and adds the move to
   `MOVED_MODULES`.
4. In the moved module, delete the `sys.path` bootstrap lines the report
   lists (tests/test_paths.py fails while any remain) and take any data
   path from `wesnoth_ai.paths`. A logger named by `__name__` changes name
   with the module (graphed_serve, packed_trunk and inference_seam use
   one; tests/test_packed_compile.py spells `wesnoth_ai.packed_trunk`).
5. Go through the other mentions: Python code inside strings
   (`run_elo_batch._PEEK_FLAGS` imports `tools.eval_players` in a child
   interpreter), `sys.modules` keys and `__import__` lists in strings, the
   readers of `tests/data/scenario_surface.json`, Rust doc comments, live
   docs. Records (docs/archive, quarantine, dated docs) stay as written.
6. A module with a command line keeps a thin wrapper at its `tools/` path:
   argparse, then one call into the package. A library-only module keeps
   nothing in `tools/`: the codemod has rewritten every importer, and a
   re-export shim at the old path would take `monkeypatch.setattr(shim,
   ...)` into the shim's namespace, not the module's.
7. `ruff check .`, the fast tier, the branch's CI run.

The codemod moves whole modules. Extractions (step 1) and function
splits (step 9) move names between modules; there, the codemod's dry run
for the source module lists the tests that patch it through the module
object, which are the ones to check by hand.
