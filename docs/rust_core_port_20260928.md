# Finishing the Rust core, then retiring the Python one (2026-09-28)

User order 2026-09-28: finish the port of the game state to Rust, then retire
the Python implementation. The local Rust build works since the same day (a
Defender attack-surface rule blocked cargo's build scripts; an exclusion for
`C:\Users\amaur\.cargo-target` lifts it), so every step is built and tested on
the laptop as well as on CI.

## Where it stands

Steps 1 to 4 and the rule moves of step 6 are done on `feature/rust-core-port`
(phases 18 to 22). The core
(`rust/wesnoth_core/src`, adapter `wesnoth_ai/game_core.py`) reads the unit and
terrain databases, resolves every terrain fact and movement class from the
hexes' codes, builds units itself (recruits with their trait roll, plague
corpses, advancement with AMLA, pick-advance and re-applied traits and [object]
effects), keeps each unit's former underscore attributes in its record, runs
the scenario's events (setup, turn events, terrain changes, time areas), and
encodes the terrain set `obs8` reads. Every command is applied in Rust; replay
reconstruction and the simulator use the core unless `WESNOTH_RUST_CORE=0`, and
every rule asked of a view of it (fight outcomes, move routes, what a side
sees) is answered by it.
Checked so far: differential tests against the Python code for every terrain
code, unit type, trait roll, [effect] form and 476 advancement cases, and
`tools/diff_core.py` over 134 imitation replays (up to four per scenario name,
36 names), clean after the setup and every command.

What Python still does, and step 6 removes:

| what | where |
|---|---|
| the oracle: the applier, the unit builders, the event interpreter, the Python versions of the rules the core answers | `replay_dataset._apply_command` and its builders, `tools/traits.py`, `tools/scenario_events.py` handlers, `combat_outcomes`, `pathfind_sim`, `visibility` |
| the policy's encoding, observation and legality mask for a state no core stands behind | `encoder.encode_raw`, `observe.observe`, `action_sampler` over the Rust kernels |
| tools that replay a record on the Python applier | about 25 modules (`game_record`, `midgame_starts`, `value_corpus`, the swap detector's side-turn particles, the diff and dump tools, benches) |


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

Every rule the rest of the code asks of a state goes to the core for a bound
view since phase 22, each with a differential test against the Python version,
which answers any other state and stays the oracle until the retirement:

| rule | Python entry point | core | differential test |
|---|---|---|---|
| the defender's weapon choice (the engine's rating) | `combat_outcomes.counter_weapon_choice` | `counter_weapon_choice` | tests/test_rust_outcomes.py; `diff_core --outcomes` |
| an attack's exact outcomes, advancement branches included | `combat_outcomes.enumerate_attack_outcomes` | `attack_outcomes` | the same |
| a fight's statistics (the neutral AI's chance to hit) | `combat_outcomes.defender_chance_to_hit` | `fight_stats` | tests/test_rust_outcomes.py |
| the planner's context and a unit's single-turn reach (move routes, the hex an attack is made from) | `pathfind_sim.ReachContext.for_side`, `unit_reach` | `side_context`, `unit_reach` | tests/test_rust_moves.py |
| the units a side sees | `visibility.units_visible_to` | `visible_ids` | tests/test_rust_moves.py |
| an attack's children under a scripted hit-or-miss sequence | `swap_detector.enumerate_children_via_sim` | `apply_attack_scripted` | tests/test_rust_outcomes.py |

The outcome functions reuse the attack command's own fight setup, write-back
and advancement code, and agree to the last bit and in their order; on 61
corpus replays every attack's counter weapon, strike tables and distributions
equal the Python's (3,906 attacks, `tools/diff_core.py --outcomes`, which the
certification script runs over the whole corpus). The reach agrees in its
movement points, costs and predecessors per hex and in the iteration order of
its landable set, which decides ties between equally cheap attack hexes; the
visible units agree in the view's order. The Python enumeration now offers a
unit the advancements its pick-advance lists leave it, as the simulator does.
The Python planner still assembles the core's arrays into `UnitReach` and
picks routes and attack hexes from it (`route_to`, `_find_attack_hex`): data
handling, no rule.

Then the deletions: the applier and its builders (the initial state's units
built by the core's `build_unit_fields`), `tools/traits.py`, the handlers of
`tools/scenario_events.py` (its parsing, `collect_events` and
`terrain_writes_applied` stay for the views), the Python combat resolver, the
Python reach and observation, `encoder.encode_raw`'s Python body and the Python
fight outcomes (`tools/analysis/counter_weapon_census.py` reads their
internals); about 25 tools that replay a record move to
`replay_dataset.record_core`. The
differential tests that compared against the deleted code go; the core is then
checked against the engine's records and oracles only.
