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
   (default on when the wheel carries it; 0 = Python path). Timing
   pre-registered in BACKLOG (`scripts/eval_profile3_box.sh`).
3. **Combat + sim step** (wesnoth_sim combat resolution, healing,
   advancement, events glue): the [mp_checkup]-oracle-certified
   core. Full-corpus differential run required (the 24,796-replay
   sweep, on a box).
4. **GameState in Rust + cheap fork**: removes deepcopy from
   select_action and search forks; Python-side classes become
   views. Largest payoff, largest surgery — only after 1-3 are
   trusted.

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
