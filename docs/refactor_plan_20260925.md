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
`tools/`. `tools/` keeps every current file name as a thin command-line
wrapper (argparse, then one call into the package), so every command in
the docs and the box scripts keeps working. `tools/analysis/` and
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
