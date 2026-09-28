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
(`sim_test_helpers.commit_view`). A view reaches its core through
`game_core.bind_view`: the simulator's view mirrors the live core (refreshed in
place after every command), a reconstruction view is bound to a snapshot
(`CoreState.fork`), and `encoder.encode_raw` encodes a bound view from its
core, the observation and the legality mask's reach rows included. With
`WESNOTH_CHECK_VIEWS` set (the test suite sets it) a bound view edited in
place is refused. A state built by hand (tests, the live bridge) gets a core
built from it (`CoreState.from_state`, a few milliseconds).

Rules Python still computes from a view, each to become a core method with a
differential test first, then to lose its Python version:

| rule | Python | production callers |
|---|---|---|
| the defender's weapon choice (the engine's rating) | `combat_outcomes.counter_weapon_choice`, `choose_counter_weapon` | the simulator's attack, the neutral AI |
| exact fight outcomes, advancement branches included | `combat_outcomes.enumerate_attack_outcomes`, `replay_dataset.enumerate_advancement_outcomes` | MCTS chance nodes, the neutral AI, the swap detector |
| the route of a move order and the hex to attack from | `pathfind_sim.unit_reach`, `route_to` (over `_terrain_arrays_for`) | `WesnothSim` order translation and `_find_attack_hex`, the neutral AI |
| the units a side sees | `visibility.units_visible_to` | the sampler's fallback, `material`, `gbc`, `turn_search` |

Then the deletions: the applier and its builders (the initial state's units
built by the core's `build_unit_fields`), `tools/traits.py`, the handlers of
`tools/scenario_events.py` (its parsing, `collect_events` and
`terrain_writes_applied` stay for the views), the Python combat resolver, the
Python reach and observation and `encoder.encode_raw`'s Python body; about 25
tools that replay a record move to `replay_dataset.record_core`. The
differential tests that compared against the deleted code go; the core is then
checked against the engine's records and oracles only.
