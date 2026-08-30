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
   states stay on Python (fixed overhead). Env-gated opt-in until
   box setups build the wheel (cargo+maturin in setup scripts).

2b. **Raw encoding** (encoder.encode_raw, the loops): GameState ->
   RawEncoded arrays. Certify: byte-identical arrays.
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
