# Box specs — derived, never decreed

**Standing rule (user, 2026-08-17): specs in this file are DERIVED
from profiling and complexity analysis, and MUST be re-derived
before renting a new box.** Past ad-hoc statements about box shapes
(including the user's own) are superseded by measurement: when a
pipeline change plausibly moves a bottleneck, re-profile first,
then rent. This file records the current derivation, its inputs,
and how to refresh each input — it is a worksheet, not a policy.

## How to re-derive (run these BEFORE renting)

1. **Forward cost + batching**: `tools/box_bench.py` on any
   available machine of the target class; on a live box, the
   telemetry columns `rollout_seconds` / `n_actions` give realized
   throughput. Open question Q9 (credit-assignment review): serial
   single-state forwards measured ~1.0 s single-thread on ladder
   maps vs ~20 ms through the pool server at 16-batch — quote
   NOTHING about forward cost without re-measuring on the target.
2. **Search cost model**: forwards per side-turn ≈
   `K × (1 + n_alt) × rounds_effective` for TCS (measured 113.6 at
   K=12.3, 30.5 at K=3.6, leg 3) + projection multiplies by
   `(1 + halfturns × actions/halfturn)` when on. Sim steps are
   ~2 ms and effectively free next to forwards.
3. **Memory per process**: a spawned torch worker (loader or
   actor) is ~1.5 GB RSS (measured: 40 loader workers ≈ 60 GB;
   96 GB box peaked at 76 GB). Actors under the pool: 57 processes
   fit a 96 GB box with the learner resident.
4. **VRAM**: trainer backward peak 12.6 GB measured (2026-07-18
   OOM incident + revision); inference server adds ~2-3 GB; the
   `gpu_mem_peak_mb` telemetry column is the live readout.
5. **Complexity shifts to watch for** (each invalidates a cached
   spec): TCS batched boundary evals (2026-08-17 — may move the
   bottleneck from the serial inference server back to CPU/sim,
   favoring MORE cores per GPU); projection on (multiplies
   forwards); net size changes (forward cost superlinear in
   tokens: ladder maps ~1050-1190 tokens, O(n²) attention).

## Current derived profiles (2026-08-17 — STALE the moment the
## batching change is benchmarked; re-run Q9 first)

### Training leg / measurement session box
- **GPU**: 24 GB class (RTX 4090). Derivation: 12.6 GB trainer
  backward peak + 15 GB reserve ruling + inference server; 12 GB
  cards fit inference-only work but not the learner.
- **vCPU**: ≥32, prefer 64. Derivation: pipeline is CPU-bound —
  actors ≈ cores−2 (57 on 64), loader pools 24-40 workers; the
  pre-batching serial server ceiling (~54 fwd/s) capped useful
  cores, so if Q9 shows batching lifted it, MORE cores become
  useful, not fewer.
- **RAM**: ≥64 GB, prefer 96. Derivation: input 3 above.
- **Disk**: 60 GB (dataset ~8 GB + checkpoints + workspace;
  proven twice).
- **Rental type**: on-demand at current ~$0.35/h. Derivation:
  escrow makes interruptible survivable (proven 3×), but the
  on-demand premium at this price is below the operational cost
  of restart churn. Revisit if 4090 on-demand exceeds ~$0.60/h.

### Elo-eval batch box
- **No GPU**. Derivation: eval games are CPU-bound (one process
  per game, ~2 threads each at mcts:32; measured 186% CPU per
  game process).
- **vCPU**: ~2 × concurrent games + 2. **RAM**: ~2 GB per
  concurrent game (measured ~1 GB RSS per game process + torch).
- Typical shape: 32-64 vCPU EPYC, $0.10-0.20/h. A GPU box that is
  already up and idle beats renting this (the tcs3 match ran
  CPU-mode on the leg box).

### RCA evals
- Laptop only — needs the real Wesnoth install. Not a rental.

## Operational facts (learned, box-management)

- **A stopped Vast instance is STORAGE, not reserved capacity**:
  its GPU can be rented from under it and stay occupied
  indefinitely (box 47853206, 2026-08-17). Never plan around
  restarting a specific stopped box; plan around the HF escrow
  (which is what actually made the leg-2-box pivot free).
- Storage on stopped boxes is ~$0.40/day per 60 GB — inventory
  against the escrow and destroy rather than accumulate.
- Host reliability varies wildly; `vast-box-ops-traps` memories
  catalog the create/relaunch failure modes.
