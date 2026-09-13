# Rust hot-path port — plan (ruling 2026-08-30)

User ruling: port the HOT PATH to Rust; the ML stack, tooling,
launchers, and probes stay Python ("the brain stays Python").
Motivation: the 2026-08-28/29 profiling — training generation is
CPU-bound in per-leaf Python work (state fork, sim step, legal-move
enumeration, raw encoding); expected 2-5x games per dollar.

## Non-negotiables

1. **Bit-exactness is a property of the harness, not the code.**
   No Rust kernel replaces its Python original until it has passed
   (a) differential property tests against the Python reference
   over generated cases, and (b) a replay-corpus differential run
   (the same machinery that certified the Python sim). The Python
   implementation stays in-tree as the permanent diff oracle.
2. **Flag-gated adoption.** Every kernel lands default-OFF behind a
   flag; flips ON only after certification; a mismatch in
   production raises loudly (fork-guard precedent), never silently
   diverges.
3. **Integer determinism.** Combat math is integer arithmetic;
   anything float in an interface is forbidden unless the Python
   side is float too (then bit-compared via struct packing).

## Architecture

Crate `rust/wesnoth_core/` (PyO3 + maturin, abi3 wheels). Interface
style per phase:

- Phases 1-3: PURE-FUNCTION kernels over flat arrays (numpy in,
  numpy/ints out). No Python-object graph crossings on hot calls —
  the arrays the encoder already builds are the wire format.
- Phase 4 (end state): GameState OWNED by Rust; Python holds a
  handle; fork/step/encode/enumerate are Rust methods. This is
  what kills the deepcopy cost — Rust clone of a flat struct.

## Phases

1. **Reachability/pathfinding** — DONE 2026-08-30, with a measured
   lesson. Kernel certified BIT-EXACT (tests/test_rust_reach.py:
   scenario units + 60 fuzzed contexts, cost floats equal by ==),
   opt-in via WESNOTH_RUST=1. End-to-end speedup however is only
   1.29x (202us -> 157us/call): the Dijkstra is near-free in Rust
   but per-call Python packaging (ctx sets -> bytearrays, arrays ->
   UnitReach dicts) dominates — the marshaling trap this plan
   predicted, now measured. CONSEQUENCE: phase boundaries move to
   STATE granularity — the next port is the whole per-state
   legality enumeration (context build + all units' reach + move/
   attack target lists in one call over arrays), where marshaling
   amortizes over ~N units and the dict rebuild disappears into
   the mask builder. Default stays OFF until that call exists;
   the certified kernel is its verified core.
2a. **State-level enumeration** — DONE 2026-08-30 (the corrected
   phase-1 boundary): enumerate_moves computes every unit's
   move/attack row in one call. Certified by full mask-tensor
   differential (test_rust_enumerate.py, with an engagement
   counter so a gated run can't certify vacuously) + the mask/sim
   contract slice under WESNOTH_RUST=1 (60 tests). Measured:
   midgame mask build 3.28ms -> 0.73ms (4.5x); <2-eligible-unit
   states stay on Python (fixed overhead). **DEFAULT ON since
   2026-09-02** (user ruling: "verified, pure benefit"):
   pathfind_sim imports the wheel unless WESNOTH_RUST=0 and warns
   loudly on import failure; sim_self_play banners the live path;
   arm_vg_launch.sh / vast_onstart.sh build the wheel (rustup
   minimal + maturin, ~2 min) and abort the launch if it fails.

**Box + corpus certification (2026-08-30, user order):** wheel
builds on Linux (rustup minimal + maturin, ~2 min in box setup);
certification suites green on the box; and the FULL imitation
training corpus reconstructs 17,124/17,124 clean under
WESNOTH_RUST=1 (30-way sharded diff_replay, ~80s wall). A control
shard proved python/rust byte-identical on the same files. The
sweep's first run also caught the untracked-macros regression (see
BACKLOG 2026-08-30 correction) — the divergences it showed were
environmental, identical under both paths, and vanished once the
macros actually shipped. Records: eval_games/rust_corpus_cert/.

2b. **Raw encoding** (encoder.encode_raw, the loops): GameState ->
   RawEncoded arrays. Certify: byte-identical arrays.
   **Ported 2026-09-04** (`rust/wesnoth_core/src/encode.rs`,
   `encode_raw_streams`, one call per encode). Python keeps the slot
   orderings (`visible_units_in_slot_order`, `own_recruit_types`, the
   static hex cache), the vocab lookups (ids cross as arrays; Rust
   never owns the dicts), the fog and ownership predicates (dict and
   set lookups on Python objects, including the vision disc) and the
   Python-object fields of RawEncoded; it hands the facts over as
   flat int64/float64 arrays and Rust composes every numpy array
   (hex modifier/dynamic bits, unit stream, recruit stream, global
   features) with the reference builders' float order: each feature
   is the f64 expression Python evaluates, cast once to f32. The
   Python builders (`_python_*` in encoder.py) stay verbatim as the
   diff oracle. Selection is pathfind_sim's (`_RUST` importable and
   WESNOTH_RUST != 0); a pre-2b wheel lacks the function and takes
   the Python path.
   Certification: laptop differential test only
   (tests/test_rust_encode_raw.py: 3 scenario starts, 40 dummy-game
   midstates, fog on/off, recruit rejections, owner-map entries off
   the village terrain, a petrified unit, out-of-vocab names, both
   hex-stream modes; every array equal in dtype, shape and bytes,
   with engagement counters). Box measurement pending.
   Measured on the laptop (encode_raw over 200 dummy-game states,
   H ~860): 634 -> 391 us per encode with fog on (visible units mean
   4.5); on the 26 fog-off states with 12 visible units 112 -> 56 us.
   The fog-on residue is `visible_hexes_for` (the vision disc) and
   `units_visible_to`, which are outside this phase.
   Observation while certifying: sim-path hexes carry the village
   TERRAIN without the village MODIFIER (replay_dataset.py:740, on
   purpose), so on scenario_pool maps the static village bit lights
   only for owned villages; terrain ids still mark them. Unchanged by
   the port, recorded so the next reader does not rediscover it.
2c. **The observation** (2026-09-11, `rust/wesnoth_core/src/observe.rs`,
   `wesnoth_ai/observe.py`): one call per decision over a flat
   snapshot of the observable state (map geometry cached per hex
   container; per-unit arrays built in Python, O(units)) returns
   the vision disc, unit visibility with the hide-cover gate and
   adjacency discovery, the reach-context flags (occupied / ally /
   enemy / ZoC / inert) and the recruit row (castle-network BFS from
   the leader's keep, minus occupied and rejected hexes), all in MAP
   space. The encoder computes it once (`encode_raw`), stores it on
   RawEncoded/EncodedState (`observation`, arrays only, picklable),
   uses it for the disc and the visible units; the legality mask
   builder reads occupancy, the reach context and the recruit row
   from it and hands the flags straight to `enumerate_moves` (no
   two-unit bail). Relevant-set streams keep the Python path (the
   Rust rows do not serve subset streams). Why: the 2026-09-11
   shared-inference worker profile (docs/box_specs.md) put the
   Python that rebuilt these facts every decision at three quarters
   of the worker's own time; combat and the sim step were 1.5%.
   Certified: tests/test_rust_observe.py (disc, visibility, flags,
   recruit row and the whole legality mask against the Python
   originals on harvested states, both sides, fog on and off; the
   detached record pickles) plus the encoding byte-identity,
   enumeration, seam and visibility suites on a box (the laptop
   cannot execute freshly built binaries, so wheels build on a box:
   `pip install rust/wesnoth_core`). Switch: `WESNOTH_RUST_OBSERVE`
   (default on when the wheel carries it; 0 = Python path). Measured
   2026-09-11 (docs/box_specs.md "The observation kernel and the CPU
   budget"): a lone eval game 1.2x faster (15.7 -> 13.1 s, the same
   587 decisions), the crowded 40-game match unchanged, because that
   path is bound by the server's per-batch cycle and not by worker
   CPU (5.4 of 16 cores used). Kept on by default.
2d. **The relevant-set basis on the kernels** (2026-09-12, user order
   "complete the Rust port, including for eval and pool"; the
   reference player `relset` plays in that basis, which every kernel
   above declined). `observe(state, side, reach=True)` adds every
   acting unit's landable row (`wesnoth_core.reach_rows`, the
   Dijkstra half of `enumerate_moves`) and the relevant hex set: the
   union of those rows with the villages, the castles, the visible
   units' hexes and the leader's castle network (`observe_side` now
   returns the network). The encoder takes the subset from the
   cached full-board arrays through that mask (`_relevant_subset_static`;
   the same row-major order `relevant_hexes_in_slot_order` filters)
   and hands the mask builder the map-to-token index; the mask
   builder turns the observation's landable rows into move and
   attack rows in the subset's token space (`rows_from_reach`, the
   other half of `enumerate_moves`), so no decision in either basis
   runs the Python enumeration, the Python subset selection or the
   per-decision static-array build. Certification:
   tests/test_rust_relevant_set.py (the relevant set and every
   landable row against the Python originals on harvested states,
   both sides, fog on and off; the subset records byte-identical to
   the Python subset path; the subset masks equal to the Python mask
   path) plus the existing suites, on a box (`scripts/relset_rust_box.sh`),
   which also times the reference player against itself with the
   kernels off and on. CERTIFIED 2026-09-12 (52 tests on the box) and
   measured (docs/box_specs.md "The relevant-set basis on the Rust
   kernels"): a lone game of the reference player 27.0 -> 14.5 s for
   the same 614 decisions, the 40-game match 73-79 -> 57 s. Default
   on with the wheel.
3a. **Combat** (2026-09-12, `rust/wesnoth_core/src/combat.rs`): one
   attack resolved in Rust from the two snapshots as flat integers:
   std::mt19937 with the Knuth seeding (Wesnoth's `mt_rng`),
   `_compute_battle_stats`, the strike loop and the hit body of
   `wesnoth_ai/combat.py`, returning the outcome and the per-strike
   checkup records. `combat.resolve_attack` takes it behind
   `WESNOTH_RUST_COMBAT` (default off until certified; the Python
   body stays as `_resolve_attack_python`, the oracle). `random_int`
   covers the advancement draw. Certification: tests/test_rust_combat.py
   (3,000 fuzzed fights field for field including the strike records
   and the draw count; the `[mp_checkup]` fixture through the kernel)
   and the imitation corpus reconstructed with the kernel on and off,
   the divergence lists compared (`scripts/combat_rust_box.sh`).
   CERTIFIED 2026-09-12 on a box: the fuzz and fixture tests pass
   (the fixture's 29 attacks strike for strike against Wesnoth's
   records through the kernel), and 17,039 of 17,039 corpus replays
   reconstruct clean with the kernel on and off, the two sweeps
   identical (214 and 223 s of wall on 48 shards: combat is not
   where reconstruction spends its time). Default on since.
3b/4. **The Rust-owned state and its step kernels** (2026-09-12,
   user order "complete the port"; `rust/wesnoth_core/src/core*.rs`,
   `wesnoth_ai/game_core.py`, `tools/diff_core.py`). Phase 3b's
   kernels over flat arrays were skipped: with the state in Rust the
   commands apply on the state itself and nothing is marshaled.
   `GameCore` holds the units (a record per unit; the unit-type table
   for stats, weapons, abilities, advancements; the movement classes,
   one per unit type, slowed status and defense table: the
   pathfinder's cost arrays and the resolver's defense percentage on
   every hex), the sides, the turn scalars, the village owners, the
   uncovered hiders, the rejection sets, the advancement queue and
   the recorder's side channels; the map's static facts (geometry,
   castle and village flags, terrain healing and light, time areas,
   the encoder's slot order) are built once per hex set by Python and
   shared across forks behind `Arc`. `fork` is a clone.
   Commands in Rust: init_side (healing, the move refresh, income and
   upkeep), end_turn, move (`walk_move_path` with the hide cover,
   discovery by adjacency, the sight disc and the units a side sees,
   the village capture), attack (`build_attack_context` and
   `_to_combat_unit` over the records, the combat kernel of 3a, the
   outcome on both units, feeding, deaths, plague eligibility) and
   recruit (the unit from the Python builder with its trait roll, the
   gold and the uid counter in Rust). Python keeps what constructs
   units: recruits, advancements (`_maybe_advance_unit` on a carrier
   holding the unit and the advancement globals) and plague corpses
   (`_build_plague_corpse`), the scenario events (an init_side while
   the scenario still has an event that can fire runs on a Python
   view of the core, and a terrain morph rebuilds the core) and the
   rare pickadvance and recall commands (the same view path). The
   observation (`GameCore.observe`, the arrays of `observe.observe`)
   and the encoding (`GameCore.encode_streams`, every array of
   `encoder.encode_raw` in either basis) are methods over the core's
   own records; `CoreState.observe` / `CoreState.encode_raw` wrap
   them.
   `WesnothSim(use_core=True)` (default from `WESNOTH_RUST_CORE`,
   off until the timing below says on) keeps the core as the state
   of record: `sim.gs` is one Python view object refreshed in place
   after every command (built on first use for a fork), so every
   holder of the state, its map, its sides or its global info reads
   the current state; the mutating entry points are sim methods
   (`reject_recruit_hex`, `enable_uniform_advancement`, the
   recorder's clears). The action translation, the neutral side's
   turn and the policies still read the view.
   Certified 2026-09-12 on a box (records:
   `training/metrics/bench_pipeline/core_step_20260912/`):
   tools/diff_core.py replays the whole imitation corpus through the
   core and through the Python applier and compares the two states
   field by field after every command: 17,039 of 17,039 replays
   clean, 5,475,904 commands applied in Rust (3,023,243 moves,
   1,079,833 attacks, 467,264 recruits, 430,620 init_sides, 474,944
   end_turns) and 63,760 through the view (61,331 init_sides on maps
   with events, 2,429 pickadvances), 825 s of wall on 28 shards
   (`scripts/diff_core_box.sh`). The first full sweep found one
   defect, a unit-less view kept across a terrain morph that gave
   post-morph recruits pre-morph movement costs (2 Aethermaw
   replays); the round-trip of the first sample found the per-unit
   attributes the stash must carry. tests/test_game_core.py: the
   round trip, fork isolation and the state key, init_side and
   end_turn against the applier, the lawful bonus, three replays
   through the move and attack kernels, the observation equal to
   `observe` and the encoding byte-identical to `encode_raw` on
   harvested and replay states (both sides, fog on and off, both
   bases, the enemy-village gate), and twin simulators (the Python
   state of record against the core) playing identical games under
   a fighting driver and the dummy policy, extras included; the
   determinism and fork-isolation suites pass with it.
   Measured 2026-09-12 (docs/box_specs.md "Per call: the Rust-owned
   state against the Python state"): per call, fork 0.049 -> 0.019 ms,
   one move 0.361 -> 0.050 ms, an encode in the relevant-set basis
   0.546 -> 0.200 ms, which is plan step 1.2's acceptance met. On the
   EVAL path it buys nothing (`scripts/core_sim_box.sh`: a 40-game
   match 49 s against 49 s, a lone game 10 s against 11 s) because
   that path waits on the inference server for over four fifths of
   its wall, and the lone game pays about 5% for rebuilding the
   Python view after every command. Default off
   (`WESNOTH_RUST_CORE=1` to enable); the pool is where the per-call
   numbers can pay, measured as phase 1's exit.

## Build/dev

- Local Windows: rustc 1.96 msvc toolchain VERIFIED working
  (hello-world links); maturin via pip; `maturin develop` for the
  dev loop.
- Boxes (Linux): maturin build in the box setup; wheels are
  box-local (no cross-compilation needed — source ships in the
  tarball/clone and builds in ~1 min).
- Tests: pytest drives the differential tests (Rust called via the
  wheel); `cargo test` for Rust-internal invariants.

Rejected: full-codebase port (user agreed 2026-08-30) — the ML
stack is GPU-bound and ecosystem-locked; tooling is
iteration-speed-critical. Rejected: per-call Python-object
marshaling interfaces — conversion overhead would eat the gains
(hence flat arrays now, Rust-owned state later).
