# Finishing the Rust core, then retiring the Python one (2026-09-28)

User order 2026-09-28: finish the port of the game state to Rust, then retire
the Python implementation. The local Rust build works since the same day (a
Defender attack-surface rule blocked cargo's build scripts; an exclusion for
`C:\Users\amaur\.cargo-target` lifts it), so every step is built and tested on
the laptop as well as on CI.

## Where it stands

The port and the retirement are done (branch `refactor/retire-python-applier`).
The core (`rust/wesnoth_core/src`, adapter `wesnoth_ai/game_core.py`) reads the
unit and terrain databases, resolves every terrain fact and movement class from
the hexes' codes, builds units itself (the initial state's units through
`build_unit_fields`, recruits with their trait roll, plague corpses,
advancement with AMLA, pick-advance and re-applied traits and [object]
effects), runs the scenario's events (setup, turn events, terrain changes, time
areas), applies every command, and answers every rule asked of a state. It is
the state of record of the simulator and of replay reconstruction
(`replay_dataset.record_core`), with no other.

Before the retirement it was certified against the Python code: differential
tests for every terrain code, unit type, trait roll, [effect] form and 476
advancement cases, `tools/diff_core.py` over the whole corpus (14,376 of 14,376
replays, 2026-10-01, the core equal to the Python applier after every command),
and on 61 replays every attack's counter weapon, strike tables and outcome
distributions equal to the last bit (3,906 attacks). Those tests and tools
retired with the code they compared against (commit 07b2c91 has them). The core is checked against the engine's records and oracles: the
strict-sync combat fixture (tests/test_combat_seed_alignment.py),
`tools/diff_replay.py` over the replay corpus, the scenario-init and
hidden-units oracles, and the live mirror against the default AI.

## How Python reads the core

A view (`CoreState.to_state`) is a copy: editing it changes nothing in the
core, and a test that edits one hands it back (`sim_test_helpers.commit_view`)
or edits a copy. A view reaches its core through `game_core.bind_view`: the
simulator's view mirrors the live core (refreshed in place after every
command), and a reconstruction view is bound to a snapshot (`CoreState.fork`,
`game_core.view_of`). With `WESNOTH_CHECK_VIEWS` set (the test suite sets it) a
bound view edited in place is refused. A state built by hand or copied gets a
core built from it (`game_core.core_for`, a few milliseconds, reused while the
state is unchanged).

| what Python asks | entry point | core |
|---|---|---|
| the encoding | `encoder.encode_raw` | `encode_streams` |
| the observation, the relevant set, the acting units' landable rows | `observe.observe`, `visibility.relevant_hex_positions` | `observe` |
| the legality masks' move and attack rows | `action_sampler._build_legality_masks` | the observation's rows, `rows_from_reach` |
| the planner's context and a unit's single-turn reach | `pathfind_sim.ReachContext.for_side`, `unit_reach` | `side_context`, `unit_reach` |
| what a side sees | `visibility.units_visible_to`, `visible_hexes_for` | `visible_ids`, `seen_export` |
| the defender's weapon choice, an attack's exact outcomes, a fight's statistics | `combat_outcomes` | `counter_weapon_choice`, `attack_outcomes`, `fight_stats` |
| an attack under a scripted hit-or-miss sequence | `swap_detector` | `apply_attack_scripted` |

The Python planner assembles the core's arrays into `UnitReach` and picks
routes and attack hexes from it (`route_to`, `_find_attack_hex`): data
handling, no rule.

## What Python keeps

The scenario parsing (`tools/scenario_events.py`: `collect_events`, and
`terrain_writes_applied` for the views), the replay extraction, the scenario
builder, the unit-stats lookups the encoder's recruit rows read
(`replay_dataset._stats_for`), the time-of-day name of a turn, a side's income
for the scenario-init oracle (`replay_dataset.side_income`), the vacant castle
search of a recruit onto a hidden unit (`wesnoth_sim.nearest_vacant_castle`),
the movement costs the shaping reward's approach distances read
(`wesnoth_sim._move_cost_at_hex`), and the leadership bonus the swap detector's
screen reads (`abilities.leadership_bonus`).

Retired: the Python applier and its builders, `tools/traits.py`, the event
handlers, the combat resolver, the fight outcomes, the reach, the vision and
observation, `encoder.encode_raw`'s Python body, the `WESNOTH_RUST_CORE`,
`WESNOTH_RUST`, `WESNOTH_RUST_OBSERVE` and `WESNOTH_RUST_COMBAT` switches, and
the tools that compared against them or replayed a record on the applier
(`tools/diff_core.py`, `bench_core.py`, `diff_combat_strike.py`,
`diff_unit_counter.py`, `diff_move_final_hex.py`, `dump_unit_states.py`;
`tools/analysis/counter_weapon_census.py`, `hider_rule_sample.py`,
`vision_rule_census.py`, `observation_parity_census.py`), because the core
answers every rule they computed and was certified against them first. The
records those tools wrote stay where they are.
