# Documents

Start with `CLAUDE.md` (status, architecture, rules of work),
`BACKLOG.md` (the next actions in order) and `plan_20260904.md` below.
A dated document is a record of the day it names: its "Measured"
section holds the result, and a status line at its top says what
became of it when that changed later. `archive/` holds superseded plans,
leg records and runbooks (index in `archive/README.md`).

## Current references

| document | what it is |
|---|---|
| plan_20260904.md | the plan in force: engineering first (phase 1, closed), then search over turns (phase 2) |
| wesnoth_rules.md | the engine rules the simulator follows, each with its source citation |
| design_constants.md | where the derived numerical constants come from |
| box_specs.md | the current box shape, the worksheet that derives it, and every throughput measurement |
| checkpoint_naming.md | the checkpoint naming scheme and the lineage of named checkpoints |
| rust_port_plan.md | the Rust kernels: rules of the port, phases and status |
| turn_proposer_design_20260905.md | the design of the turn-level search pipeline that phase 2 builds (rows 1 to 6 without a pre-grader) |
| techniques.md | catalog of every learning technique the training system implements beyond bare REINFORCE, with its default and a code citation |
| corpus_v2_20260926.md | the corrected imitation corpus (version 2): what each correction changes, its measured reach, and the rebuild |
| refactor_plan_20260925.md | the refactor into one package per system: what is wrong now, the target layout, what a move must not break, the order of steps |
| refactor_inventory_20260925.md | the inventory the refactor plan rests on: the system map, dead code with evidence, structure problems (commit 90ff321) |

## Pre-registrations and their results

| document | result |
|---|---|
| turn_value_prereg_20260925.md | the turn-ranking value function: every judged grader FAILS (linear 0.274, head 0.269, rollout 0.422 against a bar of 0.7), barrier passed (2026-09-26) |
| unit_vocab_retrain_prereg_20260925.md | `obs8`'s recipe with every unit type on its own embedding row; waiting on the user (BACKLOG.md, "NEXT") |
| observation_retrain_prereg_20260924.md | `obs8`, the recipe on the current observation, beat `terrain` +73 +- 13 Elo and is the reference since 2026-09-25 |
| time_of_day_prereg_20260922.md | not run alone: batched into the observation retrain |
| turn_gap_ref_prereg_20260921.md | the turn-level gap under the reference is RICH (7 of 60), and no forward-only pre-grader passes (2026-09-23) |
| serve_batch_prereg_20260920.md | the serve batch cap 64 is the default; 3,000 leaves per second per 4090 met (2026-09-21) |
| terrain_multi_hot_prereg_20260919.md | the terrain set: +44 +- 12 Elo over `relset`; `terrain` was the reference 2026-09-20 to 2026-09-25 |
| composed_levers_prereg_20260919.md | the terrain arm and the end_turn offset add: +263 +- 16 over `relset` at `raw:t0` |
| endturn_rule_prereg_20260919.md | end_turn at the actor level passed (+193 Elo); the offset -1.5 scored +229 and became the reference decode (2026-09-20) |
| endturn_offset_sweep_prereg_20260919.md | the offset curve peaks between -1.5 and -2.5 |
| continuous_generation_20260918.md | continuous generation (`--stream`) failed its rule by a hair; it stays opt-in |
| turn_gap_prereg_20260904.md | the turn-level gap under the seed is sparse (3 to 4 of 60 confirmed, 2026-09-05) |

## Studies and reviews (dated records)

| document | what it is |
|---|---|
| raw_argmax_control_20260904.md | the finding that set the reference frame: argmax beats sampling +412, and 32-evaluation search loses to argmax |
| selfplay_redesign_20260904.md | the 20-agent panel's verdict on self-play designs; its XOD proposal was not built on `main` |
| gpu_forward_design_20260904.md | the GPU cost of one inference batch and the levers against it, priced at 1,270 tokens per leaf (the relevant-set basis runs about 320) |
| training_signal_panel_20260905.md | six proposers, three judges, seven kept proposals; test 1 (end_turn) ran 2026-09-19 |
| model_cost_study_20260905.md | plan step 1.4: tokens per leaf; the relevant-set basis adopted 2026-09-11 |
| value_head_study_20260907.md | the seed's value head against human outcomes, by game phase |
| data_contamination_20260908.md | the holdout contamination review, and the manifest split every tool now uses |
| literature_sparse_signal_20260921.md | literature survey: human data densifies the win-loss signal; its notes are in literature_notes/sparse_signal_20260921/ |
| scenario_build_plan_20260922.md | the scenario builder checked against the game; done 2026-09-23 |

## Environment variables the code reads

Each is optional; the default is the behaviour with the variable unset.

| variable | default | effect |
|---|---|---|
| `WESNOTH_RUST` | 1 | 0 forces the Python reach, legal-move and encoding paths (`tools/pathfind_sim.py`, `wesnoth_ai/encoder.py`) |
| `WESNOTH_RUST_OBSERVE` | 1 | 0 forces the Python observation and relevant-set rows (`wesnoth_ai/observe.py`) |
| `WESNOTH_RUST_COMBAT` | 1 | 0 forces the Python combat resolver (`wesnoth_ai/combat.py`) |
| `WESNOTH_RUST_CORE` | 0 | 1 makes the Rust-owned state (`GameCore`) the simulator's state of record (`tools/wesnoth_sim.py`) |
| `WESNOTH_STRICT_WML` | unset | set, an unmodelled WML construct raises instead of warning (`wesnoth_ai/rules/scenario_cfg.py`, `tools/scenario_events.py`, `wesnoth_ai/rules/wml_state.py`, `tools/neutral_ai.py`) |
| `WESNOTH_GAME_RECORD_DIR` | unset | the default of `sim_self_play --game-record-dir` (else `training/game_records`); the test suite points it at a temporary directory |
| `WESNOTH_EXE` | the Steam install's `wesnoth.exe` | the Wesnoth executable for the live bridge, the engine oracles and the template builder (`wesnoth_ai/constants.py`) |
| `WESNOTH_ENUM_REFERENCE` | unset | 1 uses the reference legal-action enumerator instead of the vectorized one (`wesnoth_ai/action_sampler.py`) |
| `WESNOTH_PRIOR_BIAS_END_TURN_MINI` | unset | a number added to the end_turn actor logit in mini-map games only (`wesnoth_ai/action_sampler.py`) |
| `WESNOTH_MINI_RANDOM_TOD` | unset | set, the fixed-time mini maps start at a random time of day (`wesnoth_ai/rules/scenario_pool.py`; `sim_self_play --mini-random-tod` sets it) |
| `WESNOTH_RUN_TAG` | the launch time | the run's provenance tag, which `sim_self_play` sets for every process it spawns (`tools/validation_exports.py`) |
| `WESNOTH_PROF` | 0 | 1 times the imitation trainer's stages and writes `<checkpoint>_prof.json` at each evaluation (`tools/supervised_train.py`) |
| `WESNOTH_GRAPHED_DUMP` | unset | a directory where the graphed inference server saves a batch it failed on (`tools/inference_seam.py`) |
| `ELO_MOVES_LEFT_UTILITY` | 0 | the moves-left utility of a searched eval player (`tools/elo_eval_game.py`) |
| `SIM_FORK_GUARD` | unset | 1 asserts that each search leaves the caller's live state unchanged, at two state hashes per search (`tools/mcts.py`) |
| `WAI_TORCH_THREADS` | 4 | torch threads of `sim_self_play` on CPU when `--torch-threads` is not given |
| `WAI_STREAM_GET_TIMEOUT` | 60 | seconds an imitation encode-worker read waits before checking for dead workers (`tools/supervised_train.py`) |

The box scripts read the Hugging Face settings (`HF_TOKEN`, `HF_REPO`,
`HF_PREFIX`, `HF_EXTRA_FILES` and the `HF_UPLOAD_*` limits of
`scripts/hf_upload_loop.py`) and Vast's `CONTAINER_ID` and
`CONTAINER_API_KEY`, with which a box stops its own instance.
`WESNOTH_SRC` and `WESNOTH_PATH` are Python constants, not
environment variables: the data tree's path (`wesnoth_ai/rules/scenarios.py`,
`wesnoth_ai/rules/scenario_cfg.py`) and the executable's path, which
`WESNOTH_EXE` overrides (`wesnoth_ai/constants.py`).
