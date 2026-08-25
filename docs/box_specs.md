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

**Training-per-dollar derivation (2026-08-21, leg-5 rental).** The
pipeline is CPU-bound: leg 4 delivered ~8.5k decision-steps/hour
from 19 actors on 23 effective cores at $0.303/h => ~28k steps/$.
Steps/hour scales ~linearly with actor count (cores-2) until the
inference server saturates (batched boundary evals lifted that
ceiling; not yet re-measured). So rank 4090 offers by
cpu_cores_effective / dph_total, THEN sanity-check per-core class
(EPYC/Ryzen server cores ~ leg-4 baseline; old Xeon E5-v4 cores
measured ~2x slower on this workload -- discount their core count
accordingly). GPU tier stays 24 GB-class: the learner's 12.6 GB
backward peak rules out 12 GB cards, and a faster GPU than 4090
buys nothing while the CPUs are the bottleneck.
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
- **vCPU**: ~2 × concurrent games + 2. **RAM**: scales with the
  MODEL PAIR, not just the game count: ~2 GB per concurrent game
  for a 15M-vs-5M pairing, but 14 jobs of 15M-vs-15M page-thrashed
  a 30 GB box (2026-08-20: eight 40-min timeouts, then the
  min-free guard refused the next chunk at 1.8 GB/job). Budget
  ~2 GB per 15M model loaded per job + a few GB headroom, and let
  `run_elo_batch`'s memory guard set the ceiling on --jobs.
- Typical shape: 32-64 vCPU EPYC, $0.10-0.20/h. A GPU box that is
  already up and idle beats renting this (the tcs3 match ran
  CPU-mode on the leg box).

### RCA evals
- Laptop only — needs the real Wesnoth install. Not a rental.

## Operational facts (learned, box-management)

- **Always filter offers with `vms_enabled=false`** (2026-08-24/25:
  four consecutive VM-class hosts booted "running" but never
  delivered the ssh key — vast-cli issue #336, broken
  authorized_keys modes on KVM instances). A box that refuses the
  key after boot is destroyed and relocated, never debugged.

- **A stopped Vast instance is STORAGE, not reserved capacity**:
  its GPU can be rented from under it and stay occupied
  indefinitely (box 47853206, 2026-08-17). Never plan around
  restarting a specific stopped box; plan around the HF escrow
  (which is what actually made the leg-2-box pivot free).
- Storage on stopped boxes is ~$0.40/day per 60 GB — inventory
  against the escrow and destroy rather than accumulate.
- Host reliability varies wildly; `vast-box-ops-traps` memories
  catalog the create/relaunch failure modes.

## Amendments (2026-08-19, leg-4 launch lessons)

- **Offers list HOST cores, not your slice.** The leg-4 rental
  advertised 192 cores; the cgroup quota delivered 23 (pool sized
  itself to 19 actors). Filter/inspect `cpu_cores_effective` in
  offer queries, never `cpu_cores`, and treat the actor-pool's own
  startup line ("quota N cores") as the ground truth.
- **Per-core speed does not transfer between boxes.** Two estimates
  in one evening were wrong by 3-4x from quoting another box's
  measurements (image pull, fork-guard smoke: ~40 min on slow
  EPYC cores vs ~10 expected). The worksheet's numbers are
  per-box-class; re-measure or say "unknown on this box".
- **Fork-guard smoke duration**: budget 10-45 min depending on
  core speed (deep fingerprints x CPU forwards x one game, no
  parallelism). It logs at INFO now; silence no longer means
  anything.
- **vastai/pytorch:cuda-13.0.3-auto lacks sm_89 binaries** (RTX
  4090 runs via PTX JIT: slow first kernels). Cold pull ~10 GB,
  10-30 min on a slow link and silent during extraction. Worth
  evaluating a baked project image (backlogged).
- **Never hand-assemble bring-up.** Token file + bootstrap script +
  launcher IS the path; every manual-ssh deviation this launch
  (dataset extract, anchor env, double invocation) recreated a
  solved problem. The launcher now enforces a single-instance lock
  and a required-decisions preflight (unset no-default vars refuse
  the launch; decline explicitly with "none").
