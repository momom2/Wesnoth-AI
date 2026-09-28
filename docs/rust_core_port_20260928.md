# Finishing the Rust core, then retiring the Python one (2026-09-28)

User order 2026-09-28: finish the port of the game state to Rust, then retire
the Python implementation. The local Rust build works since the same day (a
Defender attack-surface rule blocked cargo's build scripts; an exclusion for
`C:\Users\amaur\.cargo-target` lifts it), so every step is built and tested on
the laptop as well as on CI.

## Where it stands

`GameCore` (`rust/wesnoth_core/src/core*.rs`, adapter `wesnoth_ai/game_core.py`)
applies init_side, end_turn, move, attack and recruit on its own records, keeps
each side's fog, and computes the observation and an encoding. It was compared
with the Python applier after every command of the whole corpus on 2026-09-12
(17,039 replays clean) and of 600 replays at phase 10 (2026-09-14), and has
changed since (phases 11 to 17) with CI tests only (docs/rust_port_plan.md
3b/4). Python still does:

| what | where | how the core reaches it |
|---|---|---|
| building a recruit (type stats, trait roll from the seed) | `replay_dataset._build_recruit_unit` | `CoreState._apply_recruit` builds the unit in Python and adds it |
| advancement (the choice, AMLA, the pick-advance override) | `replay_dataset._maybe_advance_unit` | `CoreState._advance` on a carrier object |
| plague corpses | `replay_dataset._build_plague_corpse` | `CoreState._spawn_corpse` |
| per-unit facts the records lack (feeding count, pick-advance list, defense overrides, scenario modifications) | `CoreState.unit_stash` | a Python dict beside the core |
| movement classes (costs and defenses per unit) | `CoreState._class_id` | computed in Python, registered in the core |
| scenario events (prestart, start, turn and side-turn events) | `tools/scenario_events.py` | any init_side or end_turn that can fire one goes through the Python applier and is reloaded; a terrain change rebuilds the core |
| recall, pick-advance commands | `replay_dataset._apply_command` | the same Python path |
| replay reconstruction (every imitation pair) | `replay_dataset._build_initial_gamestate`, `_apply_command` | not on the core at all |
| the encoding the network reads | `encoder.encode_raw` (Python over Rust kernels) | the core's own `encode_streams` lacks the terrain set, so it cannot serve `obs8` |

## End state

- One engine: the core applies every command and every rule effect (units,
  traits, advancement, plague, feeding, events' effects, terrain changes).
- One encoding and one observation, the core's; `encoder.encode_raw` and the
  Python observation go.
- `GameState` stays only as a read-only view built on demand for the tools that
  read a state (exporters, analysis), not after every command.
- The Python applier (`_apply_command` and its builders), the Python combat
  resolver and the Python reach and observation references are deleted. What
  checks the core afterwards is the engine itself: the corpus replays (their
  commands' legality and their recorded strike data), the scenario-init and
  hidden-unit oracles, and the Rust tests.

## Decisions for the user

1. **Scenario events.** Recommended: the WML event interpreter stays in Python
   and acts through core methods (place, modify or remove a unit, change a
   hex's terrain, add a time area, set a variable), with no view round trip and
   no core rebuild. Events concern a minority of maps and turns (61,331 of
   about 5.5M corpus commands took the Python path on 2026-09-12), so speed
   does not argue for Rust, and the interpreter is about 1,600 lines of WML
   handling.
   The alternative is a Rust interpreter.
2. **The Python applier as the last oracle.** Recommended: it stays until the
   full-corpus comparison (a CPU box, about $1) certifies the finished core,
   then goes. After that the core is checked only against the engine's records
   and oracles, which is what fidelity means anyway.

Refactor step 4a (moving the simulator's Python modules into
`wesnoth_ai/sim/`) is parked: the port deletes much of what it moves, and the
simulator package's layout comes out of the port. The branch
`refactor/step4a-sim` keeps the nine moves done so far.

## Steps

Each step lands with differential tests against the Python code it replaces,
run locally and on CI; the Python code it replaces stays until step 6.

1. Units in Rust: the per-unit facts of the stash become record fields (seven
   are set today: `_defense_table`, `_pickadvance`, `_feeding_count`,
   `_trait_order`, `_object_effects`, `_wml_role`, `_ai_guardian`); movement
   classes computed in Rust from the terrain and unit tables; the `[effect]`
   applier, since a trait is a list of effects and an advancement re-applies
   the unit's `[object]` effects; recruit construction with the trait roll;
   advancement (choice, AMLA, pick-advance); plague corpses; feeding. The core
   then needs no Python for any command, and the Python event interpreter
   applies its effects through the same Rust applier.
2. Events through core methods (decision 1), terrain changes included.
3. The core's encoding serves `obs8` (the terrain set) and becomes the only
   encoding; observation additions for the next retrain are built there.
4. Replay reconstruction and the simulator run on the core by default.
5. Certification: the full corpus through the core and the Python applier on a
   CPU box, field by field after every command; the scenario-init and
   hidden-unit oracles re-run on the core.
6. Retirement: the Python applier, its unit builders, the Python combat,
   reach and observation references and `encoder.encode_raw` are deleted; the
   tests that compared against them compare against replays and oracles.
