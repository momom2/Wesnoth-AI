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

## Amendments (2026-09-04, raw-argmax control eval box)

Eval-box derivation for a 15M-vs-15M match, measured on box 49838860
(RTX 4090, 24 Ryzen 9 7900X3D cores at 23.04-core quota, 30 GB RAM,
$0.334/h, image `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`,
`--device cuda --jobs 10`, bf16 + compile defaults):

- **Use the GPU box for search matches; the CPU-only shape above is
  superseded for 15M-vs-15M.** MCTS-32 vs raw: 3.7 ms per forward,
  453 forwards per searched side-turn, median 146 s per game (max
  520), 40 games in 14 min. Raw vs raw: median 12 s per game, 40
  games in 2 min. The leg-5 note's CPU figure for the same pairing
  was ~3 min per TURN.
- Whole run (create, image pull, bring-up, both matches, pull, destroy):
  ~30 min, ~$0.17. Bring-up is `scripts/eval_box_setup.sh`-shaped:
  tarball of the working tree + `/workspace/.hf_token`, seed from HF,
  one GPU smoke game through `tools/elo_eval_game.py`.
- RAM: 10 concurrent 15M-vs-15M games fit in 30 GB with 17 GB free at
  start; VRAM stayed far below 24 GB.
- The `pytorch/pytorch:2.5.1-cuda12.4` image also lacks sm_89
  binaries (arch list ends at sm_86, sm_90); the 4090 ran via PTX JIT
  at the throughput above, so this is acceptable for evals.
- Offer selection: `vms_enabled=false`, CPU model EPYC or Ryzen (the
  cheapest offers that day were Xeon E5-2680 v4 hosts, the 3-4x-slower
  trap), cores >= 12, RAM >= 30 GB.

## Pipeline baseline (2026-09-04, `tools/bench_pipeline.py`, box 49861070)

The number every phase-1 lever is measured against. Box: RTX 4090,
24 EPYC 7K62 cores (23.04-core quota), 47 GB, $0.309/h, torch
2.5.1+cu124 (no sm_89 binaries), bf16 + compile, Python enumeration
path (the Rust wheel was not built on this box). Inputs: the 200
holdout-ladder positions of `configs/bench_states.json`. Record:
`training/metrics/bench_pipeline/baseline.{json,md,log}`.

Per-decision Python work, median ms: deepcopy 0.05, fork 0.05,
encode_raw 1.33, encode_from_raw 1.19, legality masks 2.20 (Python
path; the Rust path measured 4.5x faster on 2026-08-30), enumerate
priors 6.90, sim step 0.71, state_key 0.02. Sum 12.4 ms, dominated by
enumeration and masks. One forward at batch 1: 5.2 ms. A raw decision
therefore costs about 17.6 ms, 70% of it outside the network.

Forward throughput per process (samples/s): batch 1 gives 190 at
every size; batch 16 gives 600 at <= 714 tokens, 525 at <= 831, 417 at
<= 1018, 155 at <= 2200; batch 64 gives no more than batch 16. The
plateau is CPU-side (padding, per-sample output splitting), not the
GPU: 600 samples/s of ~35 GFLOP is about 12% of the card's bf16 peak.

End-to-end at `--jobs 10`, one process per game (checkpoint load and
compile per shape bucket inside the time):

| match | games | s/game median (max) | turns median | forwards per side | games/h | games/$ |
|---|---|---|---|---|---|---|
| raw:t0 vs raw:t0 | 20 | 34.6 (75) | 48.5 | 294 / 319 | 704 | 2,108 |
| mcts:32 vs raw:t0 | 20 | 160 (739) | 19 | 10,658 / 225 | 73 | 218 |

The argmax-vs-argmax games ran 48 turns median (outcomes were not
recorded in this run; the harness now records them): deterministic
self-play between identical weights may stall, which every play-out
cost model depends on. Measure the decisive rate before relying on
`raw:t0` self-play.

Phase-1 targets from these numbers: enumeration plus masks 9.1 ms to
under 1 ms (Rust, state-level); encoding 2.5 ms to under 0.3 ms;
batched forward from 600 to at least 3,000 samples/s (remove the
CPU-side ceiling: token-bucketed padding, pinned transfers, output
splitting off the hot path); persistent worker processes so a game
does not pay checkpoint load and compile.

## Phase-1 iterations (2026-09-04, box 49866047, same host family as the baseline)

Same harness, same 200 positions, Rust wheel built (`scripts/bench_box.sh`
builds it). Records: `training/metrics/bench_pipeline/{fwdbatch_v2,
fwdbatch_bf16,seam_fast,seam_sorted}.{json,md,log}`. One factor per
row; each row is measured against the previous one.

| change | component or number | before | after |
|---|---|---|---|
| Rust wheel present (the baseline ran the Python fallback) | legality masks, ms | 2.20 | 0.76 |
| vectorized enumeration (`enumerate_legal_actions_with_priors`) | enumerate priors incl. masks, ms | 6.90 | 2.53 |
| batched forward with launch count independent of B (`forward_padded`) | forward, samples/s, batch 16, <= 714 tokens | 602 | 679 |
| bf16 autocast on the batched path (it ran fp32 eager) | same | 679 | 1,584 (1,730 at batch 64; 1,170 at <= 1,018 tokens) |
| server-side priors protocol, first version | serve thread, leaves/s, batch 16, mixed sizes | 421 (logits protocol) | 320 |
| server fast path: padded streams straight from RawEncoded, GPU nonzero extraction | same | 320 | 393 |
| token-sorted batches in the seam benchmark (production pads at ~1.06x; the mixed-size benchmark padded every batch to the largest map) | serve thread, leaves/s, batch 16, ~660 tokens | 352 (logits, mixed) | 1,080 priors / 976 logits |
| wire bytes per leaf | | 60-87 KB | 9-15 KB |

Actor-side cost of the priors protocol: pack_masks 1.14 ms per leaf
(the mask build itself is 0.76 of it), unpack_compact 0.49 ms.

Readings:
- The batched forward's ceiling was precision and eagerness, not
  launches: the per-sample loop rewrite bought 13%, bf16 bought 2.3x.
  The single-sample path had bf16 and compile all along; the batched
  path had neither. Compile on the padded path is untested.
- A serve thread now runs at ~1,000 leaves/s for typical states, so
  two threads saturate the GPU's ~1,600/s (batch 16, bf16). Against
  the az legs' 300-370 leaves/s per box, that is 3-4x, to be confirmed
  end to end through the actor pool (`ActorPool.server_priors`).
- The remaining per-leaf costs are actor-side (encode_raw 1.35 ms,
  masks 0.76, pack 0.38, unpack 0.49, sim step 0.64) and the GPU
  itself. Next levers: compile the padded path; sort or bucket
  batches by length in the serve thread; Rust encode_raw; persistent
  eval worker processes (a 35 s raw game carries checkpoint load and
  compile).

