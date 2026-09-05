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
higher than the seam benchmark's `forward_streams` path).

| row | tokens mean | ms per 16-leaf batch | leaves/s | GFLOP per leaf |
|---|---|---|---|---|
| full board | 893 | 34.7 | 421 | 35.1 |
| relevant set | 334 | 26.3 | 610 | 10.8 |
| hex stream cut to 300 | 327 | 25.9 | 619 | 10.6 |
| cut to 600 | 626 | 30.7 | 515 | 22.5 |
| cut to 900 | 813 | 34.4 | 472 | 31.1 |

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
the padding; the pool row with the packed trunk follows.
