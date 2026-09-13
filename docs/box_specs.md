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

## Generation throughput through the actor pool (2026-09-04, box 49875606)

`tools/bench_pool.py`, one iteration as the az legs ran it (plain
PUCT, 32 evaluations, leaf batch 16, no Gumbel root, no tree reuse),
16 actors on an 18-core EPYC 7542 box ($0.326/h), one ladder game
each, max 30 turns, 25-minute cap. Records:
`training/metrics/bench_pipeline/pool_{off,on,on_bf16}.{json,log}`.

| server priors | bf16 on the server | leaves/s | games done / requested | iteration s | games/h | serve threads: infer / wait s |
|---|---|---|---|---|---|---|
| off | off (the az legs' configuration) | 141 (lower bound: truncated at the cap, idle tail counted) | 15 / 16 | 1,620 (cap) | 33 | 2,056 / 1,155 |
| on | off | 172 (lower bound, same reason) | 14 / 16 | 1,620 (cap) | 31 | 2,432 / 741 |
| on | on | 320 | 16 / 16 | 898 | 64 | 1,088 / 623 |
| on | on, 64 leaves coalesced per batch (was 16) | 364 | 16 / 16 | 747 | 77 | 1,106 / 335 |
| on | on, 64 coalesced, 14 actors instead of 19 | 315 | 16 / 16 | 886 | 65 | 1,109 / 599 |
| on | on (rerun of row 3, same seed, 45 min later) | 279 | 16 / 16 | 1,071 | 54 | 1,081 / 984 |
| on | on, server torch threads capped at 4 (default 64) | 358 | 16 / 16 | 790 | 73 | 985 / 521 |
| on | on, 28 actors, 28 games | 329 | 28 / 28 | ~1,560 | 65 | 1,671 / 1,231 |
| on | on, 38 actors | 0 (pids limit hit, see below) | 0 / 38 | - | - | - |

Tokens per leaf 1,150-1,300, padding ratio 1.11 (1.44 with 64-leaf
coalescing, which reached 30 leaves per batch on average), K median 10-12,
decisive 11-13 of the finished games. The az legs reported 300-370
leaves/s on a 24-core Ryzen box with 19 actors; this 18-core EPYC
box gives 141 in that configuration, so the same-box comparison is
the one that counts: up to 2.3x from the two committed changes (the
first two rows are lower bounds, so the true ratio is smaller).
Rows with 19 requested actors and 16 games ran 16 effective actors
(the pool gives surplus actors no game); the 14-actor row queued two
games behind the first finishers.

The box's cgroup CPU quota is 17.56 cores (`/sys/fs/cgroup/cpu.max`;
`nproc` reports the 128-thread host), shared by the actors and the
server. Inside the serve threads' inference stage the cost is about
4 ms per leaf in every row (3.8 at 16 leaves per batch, 4.0 at 30,
4.0 with 14 actors), against 0.9 ms per leaf for the same path in
the single-thread seam benchmark on an idle box. The rerun of the
reference row came out 13% lower than its first run (279 against
320; the host is shared), so the 64-leaf gain and the 14-actor loss
are inside run-to-run noise; the constant per-leaf cost is not.
A load snapshot 150 s into the rerun (`pool_on_bf16_rerun_top.txt`):
the server process at 206% CPU (both serve threads saturated), every
actor at about 50% (waiting on the server half the time), GPU
utilization 49%.

py-spy profile of the server process over a whole run (py-spy as
the parent process; attaching is refused in the container), 100 Hz,
idle samples included, records in
`training/metrics/bench_pipeline/server_profile/`. Each serve thread:
53% of samples waiting on the request queue (`select`, `Queue.get`),
12% on the first synchronous host-to-device copy inside
`batched_priors` (which is where the thread waits for the queued
forward to finish on the GPU), about 10% launching the transformer
forward, 6% building the padded streams, 4% the rest of the priors
extraction, 2% wire serialization. The main thread and the 15 queue
feeder threads are idle.

The same run restricted to GIL-holding samples (`prof_gil`): the GIL
was held for about a quarter of wall time in total (11% per serve
thread), so the two threads are not fighting over it. Of a serve
thread's GIL time, 54% is unpickling the actors' requests inside
`Queue.get` (about 8 ms per 16-leaf batch: a RawEncoded plus packed
masks per leaf, pickled as many small numpy objects), 10% the
forward's kernel launches, 4% the padded encode, 3% wire
serialization.

Reading: the server is not saturated, it is under-fed. Nineteen
actors each keep one 16-leaf request in flight and wait on it,
so both sides idle about half the time and the GPU with them.
Levers, in order: more requests in flight (more actors on the same
quota, or two in-flight leaf batches per actor), then the GPU
efficiency of the forward (compile or CUDA graphs against launch
gaps, length buckets against padding), then the per-leaf Python on
the server, starting with the request unpickling (one contiguous
buffer per request instead of many small arrays). Runs with 28 and
38 actors are the next measurements.

Result (same day): 28 actors gave 329 leaves/s, no better than 19;
each actor's cycle per 16-leaf request grew from ~0.95 s to ~1.36 s,
so the box's CPU quota is spent on the actors' own per-leaf work
(~25-50 ms of actor wall per leaf against ~5 ms of benchmarked
components). 38 actors hit the container's PID limit
(`/sys/fs/cgroup/pids.max` = 4,352 threads; each torch actor process
carries ~100 threads): the pool served nothing and sshd could not fork
for 25 minutes. The actor loop, not the server, is the next profile
(docs/gpu_forward_design_20260904.md section 2 derives the same
conclusion from the cycle arithmetic).

Reading: the serve threads were inferring for 60-75% of the
iteration in every row (2 threads sharing one process and one GIL:
encode, forward, priors, wire, all serialized by Python), so the
server is still the ceiling, not the actors (16 actors at ~5 ms of
Python per leaf could feed ~3,000 leaves/s). Next levers, in order:
serve from several processes (or take the per-batch Python off the
GIL), then compile the padded path, then length-bucketed batches.

## Evaluation throughput: persistent workers (2026-09-04, box 49875606)

`run_elo_batch.py`, seed (raw:t0) against itself, 20 games, 10
concurrent, max 200 turns, bf16 + compile, on the same box while it
was otherwise idle. Records:
`training/metrics/bench_pipeline/eval_workers/` (per-game timings in
`eval_timing.json`).

| mode | wall s for 20 games | mean game s | forward s per turn | games/h at 10 concurrent |
|---|---|---|---|---|
| one process per game | 408 | 171 | 1.86 | 176 |
| `--persistent-workers` | 97 | 32 | 0.24 | 742 |

The one-process mode pays checkpoint load, CUDA init and the compile
warmup in every game, and ten processes warming up together contend
for the CPU; the workers pay it once per process. At $0.33/h an
800-game gate costs about $0.36 of box time through the workers.
The first worker build shared one policy object between the two
sides of a same-spec match: side A's forward counter read 0 and 3 of
the 20 games ended differently from the one-process run. With the
cache keyed per side, the 6 slots covering those 3 games were
replayed twice through workers and once one-process: all four runs
(the original included) agree exactly on turns, outcomes and forward
counts. The argmax harness is deterministic run to run on this box
in both modes.

### Shared batched inference (built 2026-09-05, not yet timed on a box)

`--persistent-workers --shared-inference` adds one
`tools/eval_inference_server.py` process per distinct checkpoint
(both sides of a same-spec match share one). The server owns the
model on the GPU (bf16, the packed varlen trunk) and serves the
workers' forwards in batches coalesced over `--inference-window-ms`
(1.5) up to `--inference-max-batch` (default `--jobs`); the workers
keep the game loop, a RemoteEncoder with server-side priors and the
raw player. Raw players only (sims 0 with `--raw-temperature-a/-b`),
checkpoint specs only. Results record `shared_inference: true`,
`infer_bf16` and `infer_packed_trunk`; the outdir guards refuse to
mix them with per-process games. The server writes
`.inference_server_<k>.json` in the outdir (requests, batches, the
batch-size histogram, time idle / in the window / in the forward),
and the driver logs it at the end.

    python tools/run_elo_batch.py --label-a seed --spec-a CKPT \
        --label-b seed_ref --spec-b CKPT --games 40 --max-extra-games 0 \
        --seed-base 20000 --outdir eval_games/shared_inference_timing_40 \
        --mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0 \
        --device cuda --jobs 10 --persistent-workers --shared-inference

Expectation from the recorded figures, to be replaced by the
measurement: a decision costs the worker about 8-10 ms of Python
(encode, masks, packing, the sim step; the enumeration moves to the
server's priors kernel) plus the round trip, so ten workers ask for
roughly 600-700 forwards per second, about the server's saturated
rate (833 leaves/s with the packed trunk, plan 1.3). Mean batch 4-6,
wall time 2-2.5x lower than the per-process persistent workers (40
games at 10 concurrent: ~100 s against 215 s; 800 games ~25-30 min
against ~65). Under 15 minutes for 800 games needs the per-sample
cost cut of plan 1.4, not more workers. A second decision in flight
per worker would not help: the game loop is sequential (the next
state depends on the step's outcome), so the only overlap is two
games per worker, which saves worker memory, not GPU time.

Before a gate through shared inference is quoted, re-pin `raw:t0`
against itself once with the 20-game check above: batch composition
varies from run to run, so bf16 batched numerics are not
bit-identical to the per-process path and argmax can flip on
near-ties. On CPU (fp32, batch 1) the server's priors equal the
direct path's exactly (tests/test_eval_inference_server.py and the
seam tests).


## Raw player temperature (2026-09-05, box 49875606)

`run_elo_batch.py --persistent-workers --jobs 10`, 40 games per arm,
seed base 20000, side A = the seed at temperature T (joint-prior
sampling, `tools/raw_player.py`), side B = `raw:t0`, max 200 turns.
Records: `training/metrics/sweeps/raw_temperature_20260905/`.

| T of side A | W-L for A | games at the 200-turn cap | turns p50 / max | wall s |
|---|---|---|---|---|
| 0 (self-play of the reference) | 14-9 | 17 | 125 / 201 | 215 |
| 0.25 | 16-13 | 11 | 33 / 201 | 142 |
| 0.5 | 22-18 | 0 | 31 / 133 | 130 |
| 1 | 7-33 | 0 | 20 / 48 | 113 |

Reading: argmax against itself stalls in 17 of 40 games (both sides
repeat the same non-committal turns until the cap), so its decisive
rate from turn 1 is 23 of 40 and a self-play game costs 125+ turns at
the median. Temperature 0.5 removes the stalls and holds the argmax
player to 22-18 (score 0.55 +- 0.08, within the sample's resolution
of equal strength); temperature 1 is the ~400 Elo weaker sampler
(docs/raw_argmax_control_20260904.md). Consequences: playouts and
self-play generation that need decisive games should run at 0.5
until a larger match separates 0 from 0.5; the turn-gap playouts were
switched to 0.5 before that run (docs/turn_gap_prereg_20260904.md,
amendment).

## Elo catalog at raw:t0 (2026-09-05, box 49875606)

Four 40-game PURE matches of the seed (`2516k-b-294k-l4-0k`, the
15M imitation seed) against the checkpoints held locally, both sides
`raw:t0`, persistent workers, seed base 30000, collected into a
SEPARATE catalog (`training/metrics/elo_catalog_raw_t0.json`; the
committed catalog's edges are all `mcts:32` and the two estimands do
not mix). Records: `training/metrics/elo/raw_t0_20260905/`. Capped
games are excluded from the fit (PURE convention).

| opponent | seed W-L | capped | seed Elo (edge, +-1 SE) |
|---|---|---|---|
| 2291k (5M tier-a seed) | 16-15 | 9 | +9 +- 55 |
| 2516k (5M campaign end) | 28-6 | 6 | +215 +- 66 |
| 2516k-b-294k-l4-495k (leg-4 self-play product) | 16-14 | 10 | +17 +- 55 |
| 2516k-b-294k-tcs2-558k | 26-0 | 14 | +269 +- 72 |

Fit (reference 2291k = 0): seed +11 +- 61, l4-495k -12 +- 88,
2516k -246 +- 97, tcs2-558k -679 +- 256. Reading: judged at argmax,
the leg-4 self-play product is equal to its seed (the mcts:32 catalog
had it at -367), and the 5M 2291k checkpoint is equal to the 15M seed
(mcts:32: +223 for the seed). Both earlier gaps were properties of
the search-and-sampling procedure, not of the weights. Each edge is
40 games; none of these differences is resolved better than +-55.

## Whole-pool profile: actors and server together (2026-09-05, box 49875606)

py-spy as the parent of `bench_pool.py` with `--subprocesses`, 50 Hz,
idle samples included, 16 actors and 16 games, priors + bf16, packed
requests, server threads capped at 4 (throughput under sampling 200
leaves/s). Records: `training/metrics/bench_pipeline/pool_profile/`
(`prof_pool_all_summary.txt` per process and thread).

- Every actor's main thread: 95% of samples inside the reply receive
  (`multiprocessing/connection.py:395`); its own work (unpack, tree,
  masks, encode) is under 3%.
- Serve threads: waiting on the request queue 11-13%; the first
  host-to-device copy in `batched_priors` (where the thread waits for
  the queued forward on the GPU) 35%; padded embed 9%; forward
  launches ~10%; the rest of the priors extraction ~12%; request
  unpickling ~6%; wire 1%.

Reading, which corrects the section above: the server is the ceiling
and the actors' own per-leaf work is small. The earlier "53% waiting"
was an average over an iteration whose second half ran with most
actors already finished (median game finish at 40-55% of the wall);
in the fed phase the serve threads are busy and, inside them, the
GPU time per batch is the largest item. Consequences: (1) throughput
must be measured in the saturated window (the pool now logs the best
60-s rate, `saturated_leaves_per_s` in the bench records); (2) the GPU
levers of docs/gpu_forward_design_20260904.md apply now, in the
order given there (staged priors shipped, packed trunk, compiled
tensor-only forward, length buckets); (3) more serve threads or serve
processes overlap the serve thread's CPU work with the GPU wait.

## Model cost benchmark rows (2026-09-05, box 49875606)

`tools/bench_model_cost.py` on the 200 bench states, batch 16, bf16,
`forward_batch` (per-sample padding inside the call; its floor is
higher than the seam benchmark's `forward_streams` path). The ms
column is the median over the 12 batches and leaves/s is the rate it
implies (1000 x 16 / ms).

| row | tokens mean | ms per 16-leaf batch | leaves/s | GFLOP per leaf |
|---|---|---|---|---|
| full board | 893 | 34.7 | 461 | 35.1 |
| relevant set | 334 | 26.3 | 609 | 10.8 |
| hex stream cut to 300 | 327 | 25.9 | 617 | 10.6 |
| cut to 600 | 626 | 30.7 | 521 | 22.5 |
| cut to 900 | 813 | 34.4 | 465 | 31.1 |

Only about 9 ms of the 35 scale with the token count on this path;
the fixed ~26 ms is the per-sample padding and launch work that the
seam path avoids. The token dependence is what the study needed; the
absolute floor belongs to the server path's own levers.

## Pool runs with the saturated rate (2026-09-05, box 49875606)

From here every pool row carries the best 60-second window
(`saturated_leaves_per_s`, the fed rate) next to the iteration
average, which the tail dilutes (game finish p50 at 40-55% of the
wall). 16 actors, 16 games, priors + bf16, packed requests, server
torch threads 4, actor thread pools capped, Rust encode_raw.

| change | saturated leaves/s | iteration leaves/s | GPU ms per leaf (2 threads, over-counted) | infer s / wait s | games/h |
|---|---|---|---|---|---|
| staged priors (one H2D copy, device compaction, one sync) | 643 | 401 | 2.60 | 858 / 544 | 80 |
| same, 32 games, 2 serve threads (control for the row below) | 652 | 343 | 2.82 | 1,644 / 1,273 | - |
| same, 32 games, 4 serve threads | 587 | 339 | 5.14 | 3,048 / 2,778 | - |
| packed varlen trunk, 32 games, 2 serve threads | 833 | 489 | 2.05 | 1,303 / 857 | - |
| packed trunk + compiled layer loop, 32 games | 863 | 503 | 1.80 | - | - |
| packed trunk, 32 games (control for the rows below) | 831 | 466 | 1.90 | - | - |
| + packed embed | 851 | 481 | 1.75 | - | - |
| + length-aware coalescing | 833 | 562 | 1.76 | - | - |
| + packed embed + coalescing | 863 | 402 | 1.81 | - | - |

Reading: 643 leaves/s in the fed window is the design doc's estimate
of the fed ceiling with today's kernels (~670); the serve threads'
inference stage fell to ~3.0 ms per leaf (from 3.5-6.3 before the
staging) and the GPU is ~85% busy at saturation. Four serve threads are worse than two (the GPU is the shared
resource; more threads only interleave their kernels).

Packed varlen trunk (`wesnoth_ai/packed_trunk.py`), CUDA tests on the
box (`training/metrics/bench_pipeline/packed_trunk_tests.log`): fp32
packed vs padded agree to 1e-6 of scale; bf16 packed vs bf16 padded
differ by about 1e-2 of scale (the bf16 kernel noise, the same size
as bf16 padded vs fp32); no implicit sync. GPU ms per 16-leaf batch:

| lengths | tokens (padded) | padded trunk | packed trunk | ratio |
|---|---|---|---|---|
| near-homogeneous (pad 1.07) | 20,328 (21,664) | 18.2 | 10.8 | 1.69 |
| mixed (pad 1.44) | 24,372 (35,024) | 37.4 | 13.9 | 2.69 |

The flash varlen kernel replaces the masked mem-efficient kernel and
the padding. Compiled packed loop (one inductor graph, design section
13), CUDA tests on the box: warmup 6.9 s, no recompiles, parity within
bf16 noise; GPU ms per 16-leaf batch eager packed 10.74 -> compiled
9.79, a 0.95 ms gain at the pre-set kill threshold of 1 ms; in the
pool 863 against 833 leaves/s saturated (inside the run-to-run noise)
with GPU ms per leaf 2.05 -> 1.80, so the compile stays off by
default. In the pool the packed trunk lifts the saturated rate
from 652 to 833 leaves/s (2.05 GPU ms per leaf); the serve threads'
remaining CPU work (padded embed, priors, unpickling, wire) is now
the larger part of a batch, which the compiled loop and the length
buckets address next.

## Relevant-set zero-training probe (2026-09-05, box 49875606)

The seed encoded with the relevant hex subset (`--relevant-set-a`, no
retraining) against the seed full-board, both `raw:t0`, 40 games,
persistent workers, 219 s: 7-10 in decided games with 23 of 40 stalled
at the 200-turn cap (the reference's own self-play stalls 17 of 40).
The subset side's forward time per turn was 2.2x the full board's
(0.141 s against 0.064 s): the compiled single-sample path recompiles
for the varying token counts. Reading: the mode is not usable without
the retrain (docs/model_cost_study_20260905.md section 7), which is
therefore run; eval of subset-basis checkpoints should run eager or
through the shared inference server. Records:
`training/metrics/sweeps/relset_probe_20260905/`.

## Relevant-set two-arm retrain (2026-09-05, box 49875606)

docs/model_cost_study_20260905.md section 7. Two arms from the seed
weights, half an epoch (1.26M pairs) of the imitation recipe
(`configs/imitation.json`, lr 1e-4, batch 64, seed 20260905, same
file order): the control on the full board, the other in the
relevant-set basis. Each arm 3.9 h (90 pairs/s, 14 workers); the
matches `raw:t0` on both sides, persistent workers, 10 jobs, sides
alternated, stalled games (200-turn cap) excluded, replacements up to
400. Records: `training/metrics/elo/relset_arms/*.fit.json`,
`training/metrics/imitation_15m/relset_arms/` (holdout curves, train
logs, the seed's own read of the holdout).

| match | decisive (stalled) | side A score | Elo of A | wall |
|---|---|---|---|---|
| control vs seed | 800 (271) | 0.345 | -111 +- 13 | 27 min |
| relevant-set arm vs seed | 759 (441) | 0.449 | -35 +- 13 | 55 min |
| relevant-set arm vs control | 391 (209) | 0.537 | +26 +- 18 | 25 min |

Holdout (1,200 pairs, sample seed 0, legality-masked target CE): seed
1.341 +- 0.051 (total CE 2.838), control 1.264 (2.786), relevant-set
arm 1.264 (2.638, its own basis). Decisive games end by leader death
at a median of 24-28 turns, the normal argmax regime.

Readings. (1) The pre-registered control prediction (within +-30 Elo
of the seed) is refuted: half an epoch more of the seed's own recipe
costs 111 Elo at argmax while the holdout CE is equal or better.
Holdout imitation CE does not track argmax strength at this level;
no retrained checkpoint replaces the seed without its own match.
(2) The relevant-set arm passes its kill criterion (not below the
control by 2 SE; masked CE equal): +26 +- 18 directly and +76 +- 18
through the seed, with 4x fewer tokens per leaf. It still trails
the seed by 35, so it is not the reference. (3) Stalls: 25% of the
control's games and 37% of the relevant-set arm's against the seed.
Next (queued the same night, `chain33`, pre-registered in the study's
7b): the control recipe at lr 1e-5 and its 800-game match, to tell
whether continuation at a small rate keeps the seed's strength; the
distillation route (7c) otherwise.

## Training path cost (2026-09-05, box 49875606)

`tools/bench_train_step.py` on the 200 bench states (prior-drawn
visits), the loop's trainer configuration, medians of 3 steps.
Records: `training/metrics/bench_pipeline/train_step/`.

| precision | batch | fwd + bwd ms per experience | of which policy loss | step wall ms per experience | GPU peak MB |
|---|---|---|---|---|---|
| fp32 | 1 (the loop today) | 67.8 | 33.6 | 68.9 | 667 |
| fp32 | 16 | 57.1 | 32.3 | 57.5 | 6,200 |
| bf16 autocast | 1 | 72.1 | 33.8 | 72.2 | 567 |
| bf16 autocast | 16 | 45.0 | 32.7 | 45.5 | 3,795 |

Parity on one batch of 64 against fp32 batch 1: fp32 batch 16 exact
to 1e-4 in the gradient; bf16 loss within 3e-4, gradient cosine
0.9994, norm within 0.3%. Compiling the trainer's trunk gains under
2 ms per experience and breaks bf16 parity (cosine 0.96): not used.

Reading: the factored policy loss costs 32-34 ms per experience in
every row because it runs per experience in Python; it is the single
largest item of the training path, ahead of the backward. Implied
loop iteration at the loop's defaults (24 games, 9,912
forward+backward passes): 684 s of training path against 498 s of
generation at 833 leaves/s, i.e. the trainer is now the longer half.
Levers, in order: the policy loss over the whole batch (32 -> a few
ms), then batch 16 (exact parity, immediate), then bf16 autocast for
the forward and backward (parity within bf16 noise).

Same day, after the batched loss (`train_step_batched.md`, N 1024):

| precision | batch | forward | policy loss | backward | fwd + bwd | step wall ms per exp | train path s per iteration |
|---|---|---|---|---|---|---|---|
| fp32 | 1 | 5.6 | 28.9 | 21.2 | 59.5 | 60.0 | 596 |
| fp32 | 16 | 5.1 | 22.6 | 14.1 | 44.4 | 44.5 | 445 |
| bf16 | 1 | 7.1 | 29.1 | 27.2 | 67.5 | 68.3 | 676 |
| bf16 | 16 | 1.5 | 26.6 | 4.7 | 35.3 | 35.4 | 352 |

Parity unchanged (fp32 batch 16 exact; bf16 batch 16 loss 3e-4,
cosine 0.9994). The batched loss took its stage from 32 to 23-27 ms,
not to the predicted 2-3: its host side (mask rebuild and staging per
experience) dominates and is being profiled. The loop now trains at
batch 16 with bf16 autocast (`az_loop --train-bf16` default on).

## Evaluation through the shared inference server (2026-09-05)

`run_elo_batch --persistent-workers --shared-inference`, 40 games,
seed vs seed at argmax, 10 workers, the same slots as the T = 0 sweep
arm. Records: `training/metrics/bench_pipeline/eval_shared/`.

| mode | wall s for 40 games | mean batch | server busy |
|---|---|---|---|
| one process per game | 408 | 1 | - |
| persistent workers | 215 | 1 | - |
| persistent workers + shared inference | 145 | 3.7 | 72% |

The ten workers ask for ~195 forwards per second in total, so the
1.5 ms window fills 3.7 requests on average; the workers' own Python
per decision is the limit now, not the GPU. A 800-game gate is about
45 minutes this way. A re-pin of raw:t0 against itself through this
path is still required before its gates are quoted.

## Eval path, one factor at a time (2026-09-11, box 50568829, RTX 3090, 16 cores)

`scripts/eval_profile_box.sh`: the same 40-game raw:t0 match (seed
against itself, seed base 20000, no replacements) timed per mode on
one box, then a py-spy profile of one per-process worker. Records:
`training/metrics/bench_pipeline/eval_profile_20260911/`.

| mode | workers | wall s | server mean batch | server forward s | GPU ms per leaf |
|---|---|---|---|---|---|
| A persistent workers, Python enumeration (the match scripts before this date) | 10 | 155 | - | - | - |
| B A + shared inference | 10 | 121 | 4.3 | 95 of 116 | 3.6 |
| C B + the Rust core | 10 | 113 | 4.4 | 89 of 108 | 3.5 |
| D the Rust core alone | 10 | 154 | - | - | - |
| E C with 16 workers | 16 | 105 | 6.0 | 78 of 100 | 3.3 |
| F C with 20 workers | 20 | 94 | 7.7 | 70 of 90 | 3.1 |
| G C with 32 workers | 32 | 125 | 6.9 | 83 of 120 | 3.4 |

Reading: through the shared server the GPU is busy 78-82% of the
wall at a mean batch of 4-8, and a batch costs 17 ms at 4.4 leaves
and 24 ms at 7.7, so the per-batch overhead (kernel launches, the
priors kernel, the reply) is most of the forward time at these
batches; the pool reaches 1.2 ms per leaf at batch 16 on a 4090. The
lever is more decisions in flight per box, which raises the batch:
each worker's cycle is about 48 ms of which about 30 ms is its own
Python, so the workers sit idle most of the time and the box takes
twice as many as it has cores. The Rust core is worth 7% here
(masks and enumeration are a quarter of the worker's Python; the
worker waits on the server). The per-process worker profile
(inclusive shares of a game): the forward 27%, enumeration with
priors 33% (masks 11%, Rust rows 4%), encoding 13%, visibility 4%.
An 800-game match at 20 workers on this box is about 31 minutes,
about $0.10. Past the cores the workers contend and the batch stops
growing: 32 workers on 16 cores ran slower than 20 (125 s, batch
6.9); size the workers at about 1.25x the cores under shared
inference.
Next levers, in order: the compiled packed loop on the server (the
launch overhead at small batches), then the worker's encode and mask
Python, then tokens per leaf (plan 1.4).

### Round 2: the server's host levers and the batching knobs (2026-09-11, box 50582240, RTX 3090 Ti, 15.4-core quota)

`scripts/eval_profile2_box.sh` plus four hand runs; the same 40-game
match, shared inference, the Rust core, 20 workers unless stated.
Records: `training/metrics/bench_pipeline/eval_profile2_20260911/`.

| mode | wall s | server mean batch | server forward s of wall | ms per leaf |
|---|---|---|---|---|
| F0 packed embed off | 76 | 7.6 | 51 of 72 | 2.21 |
| H packed embed on (now the default) | 73 | 7.5 | 48 of 70 | 2.09 |
| I H + compiled packed loop | 139 | 5.3 | 76 of 126 | 2.64 |
| J H, window 3 ms | 78 | 7.7 | 48 of 74 | 2.08 |
| K H, window 5 ms | 83 | 8.1 | 49 of 80 | 2.12 |
| L H, 24 workers | 78 | 8.0 | 49 of 74 | 2.12 |
| M H, 24 workers, window 3 ms | 79 | 8.0 | 47 of 75 | 2.12 |

Reading: packed embed is worth about 5% and stays on. The compiled
packed loop is 12% cheaper per batch and a loss overall: under its
numerics the 40 games ran 25% more decisions, the batches shrank to
5.3 because a faster server drains the queue before requests
accumulate, and the wall per decision went from 3.3 ms to 4.8 ms;
it stays off (`--compile-packed` to opt in). The window and the
worker count do not move the batch past about 8 or the wall at all:
on this box the workers deliver about 314 decisions per second while
the server has 30% idle. Why is measured in "The observation kernel
and the CPU budget" below: the box uses about a third of its CPU
quota, and the wall is the per-batch cycle through one server. An
800-game match on this box is about 25 minutes, $0.15.

seed2 self-pin through this path (20 games, seed2 against itself,
seed base 50000, 20 workers, run twice): 18 of 20 games identical in
outcome, turns and forward counts; the two that differ are the
batched bf16 near-tie flips the 2026-09-05 note predicted. Outcomes
6-5 with 9 at the cap: seed2 stalls against itself at argmax as the
seed does. Records: `eval_profile_20260911/PIN*.{log,wall}`,
`pin_games.tgz`.

### The shared-inference worker under py-spy (2026-09-11, box 50585036, 4090)

`scripts/worker_profile_box.sh`: one raw:t0 game of seed2 against
itself through a hand-started inference server (max batch 4), the
worker sampled in parent mode. Records:
`training/metrics/bench_pipeline/worker_profile_20260911/`. Shares
of the worker's wall (1,640 samples), inclusive per function:

| item | share | note |
|---|---|---|
| waiting for the server's reply (`_recv`) | 62% | one game alone: the server's per-batch cost, not the worker's |
| `pack_masks`, of which `_build_legality_masks` | 10.6%, 9.6% | the Python around the Rust rows (`_rust_enumerate_rows` 4.4%): occupancy and reach-context dicts, per-hex flag loops |
| `enumerate_legal_actions_with_priors` = `unpack_compact` | 9.8% | building ~600 LegalActionPrior objects per decision to pick one; fixed 2026-09-11 (the raw player picks on the compact arrays: `compact_selection`) |
| visibility (`visible_hexes_for` 4.1, `units_visible_to` 1.5, `leader_castle_network` 1.6, dict builds 1.2) | 8.4% | sets of tuples rebuilt per decision |
| `encode_raw` (village entries 2.7, the vision disc 2.6, Rust streams 0.4) | 3.2% + | the Python predicates around the Rust arrays |
| the sim step (combat, rng) | 1.5% | not a target on this path |
| wire packing | 0.5% | |

Reading: after the compact fix the worker's own Python is about 28%
of a lone game's wall, and three quarters of it is the mask builder,
the visibility sets and the encoder predicates, all rebuilt from the
same state every decision. That is plan 1.2's next boundary: one
Rust call per decision over a flat snapshot of the observable state
(units, static hexes, owners, rejections) returning the vision disc,
the visible units, the reach-context flags and the move/attack rows
together. Combat and the sim step (phase 3 of the port plan) are not
where this path spends its time.

### The observation kernel and the CPU budget (2026-09-11, box 50593470, RTX 3090, 16-core quota)

`scripts/eval_profile3_box.sh` (round 2's 40-game match with the
kernel off and on, twice each, `WESNOTH_RUST_OBSERVE`),
`scripts/worker_profile_box.sh` (one lone game under py-spy per
kernel mode) and `scripts/eval_cpu_budget_box.sh` (the match once
per mode between two readings of the container's cgroup `cpu.stat`).
Records: `training/metrics/bench_pipeline/{eval_profile3,worker_profile_obs,eval_cpu}_20260911/`.

| measurement | kernel off | kernel on |
|---|---|---|
| 40-game match wall, s (two runs each) | 82, 79 | 94, 81 |
| 40-game match wall, s (the `cpu.stat` runs) | 79.5 | 79.3 |
| CPU consumed by the container over the match, s | 426 | 405 |
| quota periods throttled | 41 of ~795 | 36 of ~793 |
| one lone game (587 decisions, identical in both modes), s | 15.7 | 13.1 |
| lone game: the worker outside `_recv`, s | 6.0 | 4.8 |
| lone game: `visible_hexes_for` / `encode_raw` share of wall | 7.8% / 6.0% | 2.0% / 3.2% |
| server batches / mean batch over the match | 3017 / 7.4 | 2987-3863 / 6.2-7.6 |
| server GPU ms per batch | 14.5 | 13.9-15.0 |

Reading: the kernel is real on the worker (a lone game 1.2x faster,
5% less CPU over a match) and invisible on the match wall, so the
pre-registered kill (under 1.15x) applies: less worker Python is not
an eval-throughput lever. The reason is the CPU reading: the match
used 5.4 of the 16 cores on average and the quota throttled 5% of
its periods, so round 2's sentence "CPU quota saturated, 48 ms of
worker CPU per decision" was an inference and is wrong; the measured
CPU is at most 18 ms per decision (426 s over 23,086 decisions,
server and driver included). What bounds the path is the loop through
one server: each batch costs about 25 ms of wall (5.8 idle, 1.5
window, 14.5 GPU, 1-3 host and reply) and holds 7.5 of the 20
workers, which fits a two-cohort picture: the workers a batch just
answered are in their Python while the other half forms the next
batch, and the server idles until the first of them is back. The GPU
time per batch barely moves with its size (13.9 ms at 6.2, 15.0 at
7.6), so it is mostly a fixed launch cost; the GPU is busy 57% of the
wall. Levers left, in order: a second server process on the same GPU
(one cohort's forward under the other's host work; the pool has it,
`run_elo_batch` does not), a fixed-shape forward that cuts the launch
cost (CUDA graphs over bucketed lengths; the compiled loop lost to
recompiles), and fewer tokens per leaf (plan 1.4, the per-leaf part
of the GPU cost). The kernel stays on by default: certified, cheaper
per decision, and the self-play actors do the same work per leaf.

## The relevant-set twin of seed2 at one pass (2026-09-11, box 50585036, RTX 4090)

`scripts/seed2_relset_box.sh`: seed2's recipe from scratch in the
relevant-set basis (`--relevant-set-hexes`; the deduplicated corpus
with the manifest split, fog gate on, pre-encoded records, batch 64,
lr 1e-4 cosine over 4 epochs), stopped after one pass by user order
(cost), then the per-phase value evaluation and an 800-game PURE
raw:t0 match against seed2's own one-pass checkpoint
(`clean_seed_20260909/arm_epoch0.pt`), sides alternated, stalled
games replaced, each side served in its own basis by its own
inference server. Records:
`training/metrics/elo/seed2_relset_e1_20260911/`.

| match | decisive (stalled) | W-D-L for the twin | Elo of the twin | wall |
|---|---|---|---|---|
| seed2_relset_e1 vs seed2_e1 | 800 (623) | 464-0-336 | +56 +- 12 | 18 min |

Holdout probe at the end of the pass (1,200 pairs, sample seed 0):
CE 2.840, actor top-1 0.603, masked target CE 1.347, value AUC 0.752,
against seed2's one-pass 3.084 / 0.587 / 1.443 / 0.737. The value
head by phase (same-turn AUC, turns 1-5 to 31+): 0.65, 0.75, 0.80,
0.85, 0.82, 0.86 (`phase_seed2_relset.md`).

Caveat: the pass is not equal pairs. The twin's epoch trained on
2,825,379 pairs and seed2's on 2,491,171, with the same 16,650 files,
recipe and seed. The manifest's winner actions plus their end_turns
plus the expected value-only states give about 2.83M, so the twin's
count is the recipe's and seed2's run is 12% short; where its pairs
went is the pair census below. Training between probes ran 138
pairs/s against seed2's 107 (fewer tokens); the probe itself took
143 s against 26 s (the relevant-set encoding of the holdout pairs
through the simulator), 28% of the wall.

### Self-timings of the two one-pass checkpoints (same box, after the match)

40 games each against itself (seed base 20000, 20 workers, shared
inference, the Rust core): the relset checkpoint 54 s, seed2's 80 s.
Different games, so the server's own numbers are the comparison:
GPU 9.4 ms per batch at mean batch 7.2 (1.30 ms per leaf) for the
relevant-set basis against 10.2 ms at 6.1 (1.68 ms per leaf) for
the full board. Records: `seed2_relset_e1_20260911/self_*`.

### The imitation trainer timed before and after the batched flow (same box, idle GPU)

`scripts/train_profile_box.sh` (MODE=time): 30k pairs of the
relevant-set records from scratch, no probe, the arm's checkout
against the patched one (`scripts/ship_new_trainer.sh`). Records:
`training/metrics/bench_pipeline/train_profile_20260911b/`.

| trainer | pairs/s | wall for 30k pairs |
|---|---|---|
| per-pair finalize and loss (seed2's, the arm's) | 138 | 235 s |
| batched flow (`wesnoth_ai/imitation_loss.py`) | 257 | 135 s |

Under py-spy the new trainer's main thread spends 71% in the log
transfer after the step, that is waiting for the GPU (the forward and
backward of batch 64 in fp32 through the padded trunk), 7.5% in the
backward's launches, 4% reading records. The loop is now GPU-bound;
the levers left are the GPU's: TF32 matmuls or bf16 autocast for the
trunk (the eval path already runs bf16), then the packed trunk in
training mode.

### Pair census: the full-board trainer drops batches on out-of-memory (same box)

`scripts/pair_census_box.sh`: the same first 300 training files of
the seeded order, one epoch each, the arm's trainer with its logger at
DEBUG, the full-board records (pre-encoded here) against the
relevant-set ones. Records: `training/metrics/bench_pipeline/pair_census_20260911/`.

| corpus | pairs in the epoch | batch flushes failed |
|---|---|---|
| full board | 47,801 | 52, all "CUDA out of memory" |
| relevant set | 51,129 | 0 |

52 batches of 64 are the 3,328 missing pairs. The padded fp32
attention of batch 64 over up to ~860 hex tokens fills the 24 GB, the
flush raises, and the trainer dropped the batch with a DEBUG line and
no step. Over seed2's whole epochs that was 12% of the pairs
(2,491,171 trained of the recipe's ~2.83M), and not a random 12%:
the batches that do not fit are the games on the largest boards.
Since 2026-09-11 night the trainer splits such a batch in two and
accumulates (same gradient, nothing lost; `oom_splits` in the epoch
accounting) and any other flush failure is warned and counted.

## The relevant-set basis on the Rust kernels (2026-09-12, box 50710134, RTX A4000, 23-core quota)

`scripts/relset_rust_box.sh`: the phase-5 wheel built, the
certification suites (tests/test_rust_relevant_set.py plus the
observation, enumeration, encoding, reach, seam, priors, visibility
and compact-selection suites: 52 passed), then the reference player
`relset` against itself, 40 games, shared inference, 20 workers,
with the observation kernels off (`WESNOTH_RUST_OBSERVE=0`, the
Python subset selection, static arrays and mask enumeration) and
on, twice each, and one game under py-spy per mode. Records:
`training/metrics/bench_pipeline/relset_rust_20260912/`. Under
batched bf16 the games diverge at near-ties, so the walls compare
different games of the same 40 seeds; the lone game is the same
614 decisions in both modes.

| measurement | kernels off | kernels on |
|---|---|---|
| 40-game match wall, s (two runs) | 79, 73 | 57, 57 |
| one lone game, 614 decisions, s | 27.0 | 14.5 |
| worker outside the server wait (lone game) | 35% of wall | 16% of wall |
| server mean batch over the match | 5.6 | 7.3 |

Reading: in the reference player's basis the eval worker spent
about 44 ms of Python per decision before (the relevant-set
selection 9%, the mask enumeration 6%, the slot ordering and static
arrays 3% of a lone game's wall, on top of the full-board items) and
about 24 ms after; the crowded match runs 1.3x faster because the
faster workers form bigger batches on this box (5.6 to 7.3). What
remains in the worker is `_add_reach`'s per-unit Python (3%) and the
sim's own visibility calls (1.3%). The same kernels serve the pool's
actors, unmeasured there.

### bf16 training timed against fp32 (2026-09-12, box 50710134, RTX A4000 16 GB)

`scripts/train_bf16_box.sh`: the full-board corpus pre-encoded from
a fresh vocab, 30k pairs from scratch with the batched trainer at
batch 64, fp32 and under `--bf16` (autocast for the embeddings, the
trunk and the heads; fp32 weights, gradients and cross-entropies).
Records: `training/metrics/bench_pipeline/train_bf16_20260912/`.

| run | pairs/s | out-of-memory splits in 470 batches | loss at 28.8k pairs |
|---|---|---|---|
| fp32 | 27 | 173 | 8.55 |
| bf16 | 78 | 61 | 8.43 |

Reading: bf16 runs on CUDA without a NaN and with the same early
loss, and needs about a third of the memory splits at batch 64 on a
16 GB card. The speed ratio here is not the GPU's: on 16 GB the
full-board batch does not fit in either precision and both runs
pay for splitting and recomputation (a 24 GB card ran the fp32
trainer at 107 pairs/s on the old flow). The clean bf16 number and
its validation (holdout curve and match against an fp32 twin) belong
to the next training run on a 24 GB card.

## The Rust-owned state certified on the corpus (2026-09-12, box 50759583, RTX A4000, 30.7-core quota)

`scripts/diff_core_box.sh`: every replay of the imitation corpus
through `wesnoth_ai.game_core.CoreState` and through the Python
applier, the two states compared field by field after every command
(tools/diff_core.py), 28 shards. Records:
`training/metrics/bench_pipeline/core_step_20260912/`.

| sweep | replays clean | commands in Rust | through the Python view | wall |
|---|---|---|---|---|
| first (recruit still on the view) | 17,036 of 17,039 | 5,008,640 | 531,024 | 841 s |
| after the fix, recruit in Rust | 17,039 of 17,039 | 5,475,904 | 63,760 | 825 s |

The three flagged lines of the first sweep were two Aethermaw replays:
the unit-less view the unit builders read had been kept across the
turn-4 terrain morph, so recruits placed after it walked with
pre-morph movement costs. In the second sweep the view path carries
only the init_sides of maps whose scenario still has an event that
can fire (61,331) and the pickadvance commands (2,429).

### Per call: the Rust-owned state against the Python state (same box)

`tools/bench_core.py`, 8 replayed midgame states (median 20.5 units,
849 hexes), 40 calls each, median over states. These are plan step
1.2's acceptance numbers (step and fork under 0.1 ms, encode under
0.2 ms).

| operation | Python state | Rust-owned state | ratio |
|---|---|---|---|
| fork, ms | 0.049 | 0.019 | 2.6x |
| step (one move), ms | 0.361 | 0.050 | 7.3x |
| encode (relevant-set basis), ms | 0.546 | 0.200 | 2.7x |

The Python fork was already under the bar (its fast path aliases the
hexes and the unit objects); the step and the encode were not, and are
now. The encode number is the whole of `encoder.encode_raw` including
the observation.

### The simulator on the Rust-owned state, on the eval path (same box)

`scripts/core_sim_box.sh`: the reference player against itself, 40
games, shared inference, 20 workers, `WESNOTH_RUST_CORE` off and on,
twice each, then one lone game under py-spy per mode.

| measurement | core off | core on |
|---|---|---|
| 40-game match wall, s (two runs) | 49, 48 | 49, 49 |
| one lone game, s | 10 | 11 |
| py-spy: waiting on the inference server | 82% | 83% |

Reading: the Rust-owned state is not a lever on the eval path, and the
kill criterion of plan 1.2 (a speed claim needs the wall to move)
applies. The reason is the same one the observation kernel ran into
(see "The observation kernel and the CPU budget"): with the shared
server, a worker spends over four fifths of its wall waiting for a
forward, so removing worker Python moves nothing. The lone game is
about 5% SLOWER with the core because the Python view is rebuilt from
the core after every command (`to_state` 0.6% of the lone game's
samples, `unit_from_fields` 0.3%) while the policies still read that
view. `WESNOTH_RUST_CORE` therefore stays default off on this path;
the place the per-call numbers above can pay is the pool, where a
search forks and steps many times per served forward (measured under
"Phase 1's exit: the pool in the relevant-set basis").

## Phase 1's exit: the pool in the relevant-set basis (2026-09-12, box 50759583, RTX A4000, 30.7-core quota)

`scripts/phase1_exit_box.sh`, one `tools/bench_pool.py` iteration per
arm as the az legs ran it (plain PUCT, 32 evaluations, leaf batch 16,
19 actors, one ladder game each, max 30 turns, 25-minute cap). The
"az legs' configuration" arm turns off what phase 1 committed (server
priors, bf16 on the server, the packed varlen trunk and embed); the
others are today's defaults. Records:
`training/metrics/bench_pipeline/phase1_exit_20260912/`.

| arm | basis | state of record | saturated leaves/s | iteration leaves/s | games/h | games/$ | tokens/leaf | games done |
|---|---|---|---|---|---|---|---|---|
| az legs' configuration | full board | Python | 64.8 | 60.9 | 2.2 | 13.9 | 1,176 | 1 of 19 (at the cap) |
| today's defaults | full board | Python | 441.2 | 327.9 | 72.8 | 455 | 1,234 | 19 of 19 |
| today's defaults | relevant set | Python | 1,050.8 | 422.4 | 125.1 | 782 | 354 | 19 of 19 |
| today's defaults, repeat | relevant set | Python | 1,102.5 | 690.7 | 236.4 | 1,478 | 288 | 19 of 19 |
| today's defaults | relevant set | Rust core | 1,045.7 | 466.7 | 137.7 | 861 | 308 | 19 of 19 |
| today's defaults, repeat | relevant set | Rust core | 1,003.8 | 470.7 | 163.1 | 1,019 | 305 | 19 of 19 |

**Read the saturated column, not the iteration column.** The two
repeats of one configuration (rows 3 and 4, identical flags, identical
seed) differ by 1.64x in iteration leaves/s and 1.89x in games per
hour, because the actors under-feed the server and the games they
happen to play differ in length; the saturated rate, which is the
server's own throughput while it has work, repeats to within 5% across
all four relevant-set runs. Any pool claim below about 1.7x on the
iteration column is noise on this harness.

The baseline arm is ONE run (its repeat was stopped on the user's
order). Its saturated column is the safer of its two numbers: at that
configuration the server runs at 80% GPU and is the binding constraint,
so its rate is a machine property rather than a game-composition
artifact.

Phase 1's exit criterion was at least 10x more searched games per
dollar at a fixed search budget. Same box, same search budget, on the
stable measure: **64.8 -> ~1,050 saturated leaf evaluations per second,
16x**, of which 6.8x is the configuration (server priors, bf16, packed
trunk and embed) and 2.4x the relevant-set basis. End-to-end games per
dollar moved 13.9 -> 782-1,478, but the baseline's 13.9 is a lower
bound (it finished 1 of 19 games inside the cap, with the idle tail
counted), so the 16x is the number to quote.

What this does not say: the A4000 cannot judge plan 1.3's target of
3,000 leaves/s per 4090 for the 15M net. It says the target's
prerequisite held -- the relevant-set basis cut tokens per leaf from
about 1,200 to about 300 and the saturated rate rose 2.4x with it.
Confirming the 3,000 on a 4090 is a 30-minute run, about $0.25.

The Rust-owned state (`WESNOTH_RUST_CORE=1`) shows no effect here
either: its two runs (1,045.7 and 1,003.8 saturated) sit inside the
Python runs' band (1,050.8 and 1,102.5). Its per-call wins (see "Per
call: the Rust-owned state against the Python state") are real but
small against a pool whose ceiling is the server; default stays off.

## Actors buy in-flight leaves (2026-09-13, box 50869871, RTX 4090, 24 cores, 31 GB)

The pool's constraint was never actor CPU. An actor holds exactly ONE
inference request in flight and blocks on it: the in-repo whole-pool
profile (`training/metrics/bench_pipeline/pool_profile/prof_pool_all.txt`,
4.87M samples) puts 91.3% of actor main-thread samples in `_recv`, and
its own Python at 6.77 ms per leaf. Throughput is therefore
(leaves in flight) / (round-trip), and the actor count is the knob.

One factor, same search budget (plain PUCT, 32 evaluations, leaf batch
16, max 30 turns, one game per actor), `--server-priors --infer-bf16
--packed-trunk --packed-embed`:

| actors | iteration leaves/s | saturated leaves/s | games/h | games/$ |
|---|---|---|---|---|
| 19 | 665 | 1,470 | 219 | 655 |
| 32 | 914 | 1,495 | 285 | 852 |
| 48 | 1,006 | 1,524 | 294 | 877 |
| 64 | 1,116 | 1,565 | 346 | 1,032 |

64 actors is 1.68x the rate of 19 and 1.58x the games per dollar, and
the curve is still rising. The saturated column barely moves, which is
the point: the server's capacity was always there and the actors were
not filling it.

`az_loop --actors` defaulted to 8, then to 24 (this note said 32; the
code said 24, and since `--games-per-iter` also defaulted to 24 the
clamp made it 24 either way). It is now **0, meaning "as many as the
box and the iteration allow"**, because the reason for a low default
has been removed rather than worked around: a 2026-09-04 host hit a
container pids limit at 38 actors and produced ZERO leaves/s, so the
default was held under 38 on every box since, including the ones that
could take 64. `host_resources.max_actors` now READS that limit
(`pids.max` / `pids.current`, v2 and v1) and clamps to it, naming the
numbers in the log; with no readable limit it leaves the count alone
rather than guessing a cap. The pids controller counts THREADS, not
processes, which is why an actor count that looks modest can exhaust
it. Each run also logs the tasks an actor actually cost, so
`PIDS_PER_ACTOR_ESTIMATE` stops being a guess after the next box.

With actors auto-sized, **`--games-per-iter` is the binding knob**: it
caps the actor count, and the measured curve was still rising at 64.
Raising it is a TRAINING change (more games per gradient step), so it
needs its own decision, not a throughput tweak.

The 2026-09-04 reading that "28 actors gave no more than 19" has aged
out: the server saturated at 330 leaves/s then, so nothing could fill
it. A finding about a saturated resource expires when the resource
stops being saturated.

### Plan 1.3's target, on an actual 4090

The same box answers the target this project could not judge on an
A4000: **1,450-1,565 saturated leaf evaluations per second in the
relevant-set basis at about 320 tokens per leaf, against a target of
3,000** (docs/plan_20260904.md step 1.3). The target is not met and
docs/gpu_forward_design_20260904.md predicted exactly this: it prices
today's kernels at 1,300-1,800 and says 3,000 needs fewer tokens per
leaf or fp8. Per dollar this 4090 at $0.335/h gives 1,032 searched
games per dollar at 64 actors, against 782-1,478 for the A4000 at
$0.161/h -- the cheaper card is at least competitive, so pick a box by
measured leaves per dollar, not by the GPU's name.

## The eval path does not want more workers or more servers (2026-09-13, same box)

Five arms of 40 games, `relset` against itself at raw:t0 through the
shared server, one factor per arm. The wall is uninformative and the
server's own counters are not:

| arm | jobs | servers | wall s | requests | batches | mean batch |
|---|---|---|---|---|---|---|
| baseline | 20 | 1 | 42 | 19,863 | 2,533 | 7.84 |
| more workers | 40 | 1 | 46 | 18,809 | 2,239 | 8.40 |
| second server | 20 | 2 | 52 | 21,584 | 6,065 | 3.56 |
| both | 40 | 2 | 43 | 18,838 | 3,763 | 5.01 |
| baseline REPEAT | 20 | 1 | **74** | 20,981 | 4,628 | 4.53 |

**The baseline repeated at 74 s against its own 42 s, a 1.76x swing**,
so no arm here is resolvable: every difference is inside one
configuration's own variance. What IS stable and readable is the mean
batch, and it says why a second server cannot help: splitting the same
workers across two servers halves the batch (7.84 -> 3.56), and the
server's cost is mostly a fixed per-batch launch, so half the batch
means twice the batches. The backlog's standing expectation that a
second serve process per GPU buys 1.3-1.5x on eval is **not supported**;
`--inference-servers` is implemented and left at 1, worth using only
when the two sides run different checkpoints or a much larger worker
pool keeps batches full.

Consequence for method: an eval throughput claim needs repeats or the
server counters, never one 40-game wall. The same is true of the pool
benchmark (1.64x between repeats, "Phase 1's exit").

## bf16 on the imitation trainer, on a 24 GB card (2026-09-13, box 50869871, RTX 4090)

The clean number docs/box_specs.md "bf16 training timed" could not get
on a 16 GB card, where fp32 was crippled by memory splits (173 against
bf16's 61) and the ratio read 2.9x. Same seed, same 20,000 pairs, same
relevant-set basis, on-the-fly encoding, nothing else running:

| arm | wall | rate | loss at 19,200 pairs |
|---|---|---|---|
| fp32 | 181 s | 118.8 pairs/s | 7.553 (actor 2.498, value 0.700) |
| bf16 | 147 s | 148.2 pairs/s | 7.429 (actor 2.421, value 0.697) |

**1.25x, with the loss equivalent** -- not the 2.9x the memory-starved
card suggested. The number is a lower bound on the pre-encoded path,
where the encode does not dilute the GPU share. `--bf16` stays OFF by
default for the imitation trainer: this is 20,000 pairs of loss, and a
seed checkpoint's acceptance is its holdout curve and an 800-game
match, not a training-loss comparison. Flipping it is one flag.

## The actor's per-leaf Python, with the phase-8 wheel (2026-09-13, same box)

`tools/bench_leaf.py`, three replayed midgame states (19-37 units,
888-1,188 hexes), relevant-set basis, model on the GPU:

| component | ms per leaf |
|---|---|
| encode_raw | 0.36-0.50 |
| encode_from_raw (torch tensors) | 0.89-0.92 |
| enumerate_legal_actions_with_priors | 1.07-1.35 |
| sort + MCTSEdge over 662-677 legal actions | 0.24 |
| **total actor Python** | **2.8-3.2** |
| the forward it waits for, on the GPU | 2.1-2.3 |
| `CoreState.encode_raw` (the Rust-owned state) | 0.195-0.20 |

This is what settles the earlier puzzle: an actor's 27-58 ms per leaf
was never its own work. Its Python is about 3 ms and the rest is
waiting, which is why the Rust core moved nothing and why the actor
COUNT moved everything. It also says where actor-side effort would go
if it were ever worth spending: `encode_from_raw` is now the largest
single item, larger than the numpy/Rust `encode_raw` that feeds it.

### The batch's transfer to the device (same box)

`_embed_streams` sent one host-to-device copy per FIELD, each with its
own fresh pinned allocation. Coalescing to one buffer per dtype, timed
directly at batch 16 on the 4090: **0.897 -> 0.630 ms, 1.42x, and the
embeddings bit-identical**. Against a ~25 ms server cycle that is about
1% -- real, free, and much smaller than "most of the fixed host cost",
which is what it looked like before it was measured.

Rejected while here: embedding the unit and recruit streams in ONE
call instead of two (they share the embedding, so it is the same
arithmetic and saves five lookups plus a linear per batch, priced at
~3% of the cycle). Running the linear over N+M rows instead of N and
then M changes the matmul's tiling, so the embeddings stop being
bit-identical -- a numerics change on the path that produces every
strength verdict, for a few percent of a stage that is not the binding
constraint. The transfer coalescing above was taken precisely because
it moves bytes and changes no arithmetic.

## Verification of the 2026-09-13 changes (box 50878030, RTX 4090)

`scripts/verify_core_box.sh`, run where the wheel builds (the laptop
cannot execute freshly built binaries). Records:
`training/metrics/bench_pipeline/verify_20260913/`.

| check | result |
|---|---|
| the nine affected suites, Python state of record | 44 passed, 2 skipped |
| the same suites, `WESNOTH_RUST_CORE=1` | 44 passed, 2 skipped |
| `tools/diff_core.py` over 400 corpus replays | 400 clean, 0 divergences |

The sweep matters for one change in particular: the movement-class
cache is now keyed on the defense table's CONTENT rather than its
address, and a mistake there would show up as a wrong movement cost or
defense percentage on some unit, which is exactly what a field-by-field
comparison after every command catches.

## Hide cover certified after the root fix (2026-09-13, box 50882541, 28 cores)

`scripts/hide_cover_cert_box.sh`. Cover for ambush / concealment /
submerge now comes from the engine's own `[hides]` terrain globs
(`terrain_resolver.hides_cover`) instead of a hand-rolled overlay
table's defense keys. Records:
`training/metrics/bench_pipeline/hide_cover_20260913/`.

Cover moves in BOTH directions, over the Ladder pool's 20,726 playable
hexes: ambush +478, submerge +153, concealment +93 and **-302**. Every
lost hex is `^Gvs`, which is Farmland and not a village. Over all 114
tracked maps: +1,521, +332, +553/-640. So concealment NETS A LOSS.

| check | result |
|---|---|
| `tools/diff_replay.py`, every replay, 24 shards | **17,039 of 17,039 clean**, 233 s |
| `tools/diff_core.py`, 600 replays | 600 clean, 0 divergences |
| the nine affected suites, Python state of record | 72 passed |
| the same suites, `WESNOTH_RUST_CORE=1` | 72 passed |
| the fast tier, locally (after the review's fixes) | 1,050 passed, 38 skipped |

**What this establishes, and what it does not.** It establishes NO
REGRESSION: every recorded command stays legal under the new rule and
every post-state invariant holds. It does NOT establish that the new
rule is right -- MEASURED, by restoring the old rule behind a
monkeypatch and re-running:

| measurement | result |
|---|---|
| corpus replays that field a hider (300 sampled) | 164 (54.7%) |
| of 120 hider replays, reconstructing differently under the two rules | 4 (3.3%) |
| extra ambush stops under the new rule | 4 |
| moves whose LANDING HEX differs | **0** |
| `diff_replay` divergences on those 4, under either rule | **0** |

The replay format is a command STREAM with no recorded post-state, so
`diff_replay` never compares our position or movement points against
the replay directly; every check asks whether the NEXT recorded
command's preconditions hold in our state, plus four generic
invariants. That gives it an indirect grip on position (a drifted unit
trips `move:src_missing` or `attack:attacker_missing` later) and a
one-sided grip on movement points (it flags MP too LOW for a recorded
path, never too high). What it has no grip on at all is the stop
REASON of a truncated move and the uncovered-unit set -- and that is
exactly where this change lands: all four differing replays truncate
at the same hex, differing only in whether the mover was stopped by an
ambush and the hider revealed, which persists 10 to 30 commands before
converging.

So the sweep is not blind to every possible cover-rule change; it is
blind to the one we made. What establishes the rule is the engine's own
macro text, transcribed in docs/wesnoth_rules.md and pinned by
tests/test_hide_cover.py; tests/test_visibility.py now pins the
observation and move-truncation halves on `Gs^Fms`, a code the old
table missed, so those tests fail against the old rule.

`diff_core` is weaker still on this question: `game_core.map_static`
bakes its flags from the same `hides_cover` that `visibility` calls,
so the two sides cannot disagree about cover by construction.

Two parts of the blast radius the certification did not touch at all:
the encoder's unit tokens (through `units_visible_to`) and the
sampler's visible set. Both consume the changed predicate; neither has
a corpus check.

Provenance of the zero: the aggregate `clean == total` summed over the
shards is the measurement. The summariser's separate divergence
counter matched a string `diff_replay` never prints ("with
divergences" against its "with divergence(s)"), so that field was a
constant 0 whatever happened; it also printed 12 of 24 shard lines.
Both fixed, and the summariser now asserts `clean == total`.

Who was affected: the units that want the terrain. `ambush` (Wose,
Elder/Ancient Wose, Wose Shaman/Sapling, Elvish Ranger, Elvish
Avenger), which gained cover on 30.4% of the Ladder pool's
forest-overlay hexes; `concealment` (Fugitive), which gained villages
and lost farmland; and `submerge` (14 undead including the Skeleton
and Skeleton Archer), which was NOT unaffected -- it gained tropical
deep water, 153 hexes of Ruphus Isle. `nightstalk` has no terrain
condition and is untouched.

**Operational consequence: every pre-encoded corpus on HF is stale.**
Those records hold observations the sim no longer produces, and the
vocab, hex basis and fog gate they fingerprint are all unchanged, so
nothing would have caught it. `constants.OBSERVATION_EPOCH` is now
part of the fingerprint and of the policy-anchor cache's meta, and
both consumers refuse an older one by name. The next box run that
wants a pre-encoded corpus must re-encode into a fresh `--out`;
budget one pre-encoding pass (see "The imitation trainer timed").
Bump the epoch whenever a change alters what a player sees.

## Rejected: making `pos_to_hex` lazy (2026-09-13, measured locally)

`EncodedState.pos_to_hex` is built on every encode and consumed ONLY
by `action_sampler`'s legality masks and `gbc.py`. The imitation
trainer does not use it on its loss path (`supervised_train.py`: "NO
legality mask: the observed action in a human replay is legal by
construction"), only on the holdout probe, so it looked like free
waste. The 2026-09-11 py-spy profile put the dictcomp at 2.9% of the
trainer's samples.

Measured on a 756-hex board: the dictcomp is **0.194 ms of
`encode_from_raw`'s 2.200 ms, 8.8%** — so about 4% of the imitation
trainer, and NOTHING on the actor path, where an actor's own Python is
3 ms of a 27-58 ms cycle that is otherwise spent waiting on the
server.

**Rejected, because** a lazy mapping has to satisfy `.get`, `[]`,
`.items()` and `__eq__` across a dozen call sites in
`_build_legality_masks`, which runs on every actor decision. Risking
the hot legality path for 4% of one trainer is the wrong trade. Revisit
only if a FRESH profile of the imitation trainer shows the encode
dominating again; the 2026-09-11 profile predates both the batched
flow and the coalesced transfers and should not be re-cited.

## Serve thread host cost per 16-leaf batch (2026-09-05, box 49875606)

The serve stats now split the host milliseconds per batch (records
`training/metrics/bench_pipeline/pool_c_*.{json,log}`, packed trunk,
16 actors, 32 games):

| row | unpack | encode | forward launch | priors | GPU wait | reply | wire | total host | saturated leaves/s |
|---|---|---|---|---|---|---|---|---|---|
| control | 2.5 | 8.5 | 10.9 | 5.9 | 4.8 | 1.0 | 2.4 | ~36 | 831 |
| packed embed | 2.5 | 3.7 | 10.4 | 6.1 | 6.2 | 1.1 | 2.9 | ~33 | 851 |
| coalescing (length) | 2.5 | 8.0 | 10.4 | 5.5 | 4.3 | 0.9 | 2.5 | ~34 | 833 |
| both | 2.6 | 3.6 | 10.4 | 6.1 | 6.5 | 1.0 | 2.7 | ~33 | 863 |

Reading: with the GPU at 1.8 ms per leaf, two serve threads sharing
one GIL at ~33-36 ms of host work per batch are the ceiling (~830-860
leaves/s). Packed embed removes 4.9 ms (kept, on by default in
`az_loop`); coalescing only lowers the padding ratio (1.10 -> 1.07)
and delays requests (skipped 90-117k), no rate change (off). The
remaining host items: the forward's kernel launches (10.4 ms; the
compiled loop is the lever), the priors extraction (6 ms), request
unpickling (2.5 ms), wire (2.5 ms). A second serve process would
double the host budget; the GPU has room for ~1.5x more.

## Serve processes: how to run (built 2026-09-05, not yet timed on a box)

`ActorPool(serve_processes=N)` (tools/actor_pool.py, module docstring
"Serve processes") keeps the learner's in-process serve threads and
adds N-1 serve processes on the same device, each with its own copy
of the inference model at the learner's switches (bf16, packed trunk,
compiled packed loop, packed embed, server priors), its own request
queue and its own serve threads. Actor `a` asks server `a % N`
(server 0 is the learner process), so 16 actors and N = 2 give each
server 8 actors. Each serve process caps OMP/MKL/OpenBLAS and torch
at 4 threads (the caps go into the environment before the spawn,
since the child imports torch while unpickling its target; the
container's PID limit is 4,352 threads).

Weights: after every publication the loop calls `pool.sync_servers()`
between iterations; the learner's model and encoder state_dicts go
to each server as one `torch.save` byte string on its control queue
and the pool waits for the acks. `run_iteration` refuses to start
while a server's version (the learner's
`WesnothModel._weights_version`, bumped by every `load_state_dict`)
lags. Rejected: torch's CUDA IPC sharing of the state tensors,
because (torch 2.5.1 docs, "Sharing CUDA tensors") the sending
process must keep the original tensor alive as long as any receiver
holds it, a receiver killed by a signal never releases its handle,
and the pool's 'file_system' sharing strategy does not apply to CUDA
tensors; one copy of tens of MB per iteration of hundreds of seconds
buys nothing from the sharing and works the same on CPU and CUDA.

Failure: a dead serve process poisons its actors' reply queues (the
client raises instead of waiting) and the iteration aborts with
`ServeProcessDied`; `shutdown()` stops the serve processes; a serve
process whose learner is gone exits on its own. Stats: each server's
serve threads report the same dict (host ms split, leaf timeline),
`run_iteration` merges them, so `saturated_leaves_per_s` covers all
servers and `leaves_per_server` shows the split.

On the box (16 actors, 32 games, the packed-trunk configuration):

    python tools/bench_pool.py --checkpoint training/checkpoints/seed_imit_tierb_start.pt \
        --actors 16 --games 32 --sims 32 --leaf-batch 16 --max-batch 16 \
        --server-priors --infer-bf16 --packed-trunk --packed-embed \
        --serve-processes 2 --out pool_serve2.json

`bench_pool` first logs `serve-process parity on one leaf` (the value
and value-logit differences between the serve process and the learner
process on the same leaf: bf16 kernel noise, ~1e-2 of scale, is the
expected size; anything larger is a sync or switch mismatch) and
records `serve_processes`, `leaves_per_server` and `server_parity`.
Expected: the host budget doubles (two GILs), so if the host work was
the ceiling the saturated rate moves from ~850 toward the GPU's room
(~1.5x, 1,100-1,300 leaves/s) and `leaves_per_server` splits about
evenly; if it stays at ~850 with the GPU ms per leaf unchanged, the
device stream is the ceiling and the row is dead. Memory: a second
CUDA context plus the model copy, under 1 GB on the 4090.
