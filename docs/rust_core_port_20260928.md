# Finishing the Rust core, then retiring the Python one (2026-09-28)

User order 2026-09-28: finish the port of the game state to Rust, then retire
the Python implementation. The local Rust build works since the same day (a
Defender attack-surface rule blocked cargo's build scripts; an exclusion for
`C:\Users\amaur\.cargo-target` lifts it), so every step is built and tested on
the laptop as well as on CI.

## Where it stands

Steps 1 to 4 are done on `feature/rust-core-port` (phases 18 to 21). The core
(`rust/wesnoth_core/src`, adapter `wesnoth_ai/game_core.py`) reads the unit and
terrain databases, resolves every terrain fact and movement class from the
hexes' codes, builds units itself (recruits with their trait roll, plague
corpses, advancement with AMLA, pick-advance and re-applied traits and [object]
effects), keeps each unit's former underscore attributes in its record, runs
the scenario's events (setup, turn events, terrain changes, time areas), and
encodes the terrain set `obs8` reads. Every command is applied in Rust; replay
reconstruction and the simulator use the core unless `WESNOTH_RUST_CORE=0`.
Checked so far: differential tests against the Python code for every terrain
code, unit type, trait roll, [effect] form and 476 advancement cases, and
`tools/diff_core.py` over 134 imitation replays (up to four per scenario name,
36 names), clean after the setup and every command.

What Python still does, and step 6 removes:

| what | where |
|---|---|
| the oracle: the applier, the unit builders, the event interpreter | `replay_dataset._apply_command` and its builders, `tools/traits.py`, `tools/scenario_events.py` handlers |
| the policy's encoding, observation and legality mask, computed from a view | `encoder.encode_raw`, `observe.observe`, `action_sampler` over the Rust kernels |
| the exact advancement outcomes MCTS enumerates | `replay_dataset.enumerate_advancement_outcomes` |
| tools that replay a record on the Python applier | about 25 modules (`game_record`, `midgame_starts`, `value_corpus`, the diff and dump tools, benches) |


### Step 6 in detail

A view (`CoreState.to_state`) is a copy: editing it changes nothing in the
core, and the tests that did (ten files) now hand an edited view back
(`sim_test_helpers.commit_view`). For the encoding, the observation and the
legality mask to come from the core, a view must reach its core. The
simulator's view mirrors the live core (it is refreshed in place after every
command); a reconstruction view needs a snapshot (`core.fork()`, a clone of the
unit records), since the core moves on while a consumer may keep the view. A
state built by hand (tests, the live bridge) gets a core built from it
(`CoreState.from_state`, a few milliseconds). With every state reaching a core,
the Python encoding, observation and reach references and the applier go,
with the differential tests that compared against them; the core is then
checked against the engine's records and oracles only.
