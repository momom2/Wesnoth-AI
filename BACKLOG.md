# BACKLOG

Live backlog for `docs/plan_20260904.md`. The pre-restart backlog
(1,055 lines of rulings and open items, 2026-05 to 2026-09-04) is
archived verbatim at `docs/archive/backlog_20260904.md`.

## NEXT ACTIONS (phase 1: engineering, in order)

1. **Benchmark harness** (plan 1.1): DONE 2026-09-04, baseline in
   `docs/box_specs.md` and `training/metrics/bench_pipeline/`. Per
   decision: 10.1 ms of Python (enumerate priors 6.9 including the
   2.2 ms mask build, encoding 2.5, sim step 0.7; an earlier quote of
   12.4 counted the masks twice) against 5.2 ms per forward; batched forwards plateau at ~600 samples/s per process
   from batch 16 (CPU-side ceiling). Raw game 35 s, searched game
   160 s at 10 jobs, one process per game. Open: run it on a second
   box shape with the Rust wheel built (`scripts/bench_box.sh` now
   builds it) to pin the reproducibility band.
2. **Per-decision Python work** (plan 1.2), ordered by the measured
   cost: enumeration 6.9 ms, masks 2.2 (Python path) / ~0.5 (Rust),
   encoding 2.5, sim step 0.7; fork and deepcopy are 0.05 ms already
   (the Rust plan's phase 4 "cheap fork" is not where the time is).
   - DONE 2026-09-04: vectorized enumeration
     (`enumerate_legal_actions_with_priors`, reference kept as
     `_enumerate_legal_actions_reference`, `WESNOTH_ENUM_REFERENCE=1`
     forces it; differential test `tests/test_enumerate_vectorized.py`):
     7.3 -> 3.8 ms on the laptop including the mask build; the
     combat-oracle damage computation in the mask builder is skipped
     when both oracle alphas are 0 (they are). The floor is now the
     construction of ~600-700 action objects per state; the
     struct-of-arrays output that removes it belongs to the server
     step below (arrays are what cross processes cheaply).
   - Measured on the box with the Rust wheel: masks 0.76 ms,
     enumeration incl. masks 2.53 ms (was 6.90).
   - DONE 2026-09-04: Rust `encode_raw` streams (rust_port_plan phase
     2b, `rust/wesnoth_core/src/encode.rs`, byte-identical on 124
     states in both hex modes, `tests/test_rust_encode_raw.py`):
     634 -> 391 us per encode on the laptop with fog on (the residue is
     the fog visibility computation), 112 -> 56 us fog off. Box
     measurement pending (wheel rebuild in the box setup).
   - MEASURED 2026-09-11 (docs/box_specs.md "The shared-inference
     worker under py-spy"; user order "do these two", 1.2 and 1.4):
     the eval worker's own Python per decision is the mask builder's
     Python around the Rust rows (9.6% of a lone game's wall), the
     unpacking of every legal action to pick one (9.8%), the
     visibility sets (8.4%) and the encoder's predicates (3%+); the
     sim step is 1.5%. Combat and step (phase 3) are NOT this path's
     cost; the plan's phase order changes accordingly.
   - DONE 2026-09-11: the raw player picks on the compact arrays
     behind a shared server (`RawPolicyPlayer(compact_selection)`,
     one action materialized; differential test).
   - NEXT (pre-registered): one Rust call per decision over a flat
     snapshot of the observable state returning the vision disc, the
     visible units, the reach-context flags and the move/attack rows
     (the mask builder, the visibility sets and the encoder's disc
     from one pass); Python keeps the snapshot build (O(units)) and
     the tensor assembly. Certification: differential tests against
     the Python originals on the harvested states plus fuzz, then
     the full-corpus reconstruction sweep on a box. Measurement: the
     40-game shared-inference timing on one box before and after
     (`scripts/eval_profile2_box.sh` mode H): prediction 1.3-1.5x
     on the worker-bound box, kill under 1.15x.
3. **Batched inference server** (plan 1.3). Measured 2026-09-04
   (docs/box_specs.md "Phase-1 iterations"): the batched forward ran
   fp32 eager (only the single-sample path had bf16 and compile);
   with bf16 it does 1,584 samples/s at batch 16 (was 602). The
   server-side priors protocol (`wesnoth_ai/server_priors.py`: actors
   ship packed masks, the server returns compact legal actions,
   `ActorPool.server_priors`, default ON since 2026-09-04) serves ~1,080 leaves/s
   per thread on token-sorted batches with 9-15 KB per leaf on the
   wire (was 60-87 KB). Certified: parity tests through seam and
   wire, pool smoke end to end.
   - MEASURED through the real pool (docs/box_specs.md "Generation
     throughput"): az-leg configuration 141 leaves/s -> priors on 172
     -> priors + bf16 320 leaves/s, 33 -> 64 games/h, same box. The
     serve threads share one GIL and were busy 60-75% of the time:
     the server is still the ceiling.
   - DEFAULTS FLIPPED (2026-09-04): `ActorPool(server_priors=True)`
     and the server's own bf16 switch (`InferenceServer(autocast_bf16)`,
     `ActorPool(infer_bf16)`); `az_loop.py --server-priors/--infer-bf16`
     default on (cuda), `--no-...` to opt out. The learner's in-process
     probes keep the model's precision.
   - Coalescing 64 leaves per server batch instead of 16: 364 leaves/s
     against 320 (`pool_mb64`), so per-batch Python is a minor cost.
   - PROFILED (docs/box_specs.md, server profile): the server is
     under-fed, not saturated. Serve threads wait on the queue 53% of
     the time, GPU at 49%, actors at ~50% CPU: 19 actors with one
     16-leaf request in flight each cannot fill the pipeline. The GIL
     is held ~25% of wall time; half of that was unpickling requests.
   - SHIPPED: packed requests (`wesnoth_ai/leaf_wire.py`, one buffer
     per request); server torch threads capped at 4 (358 leaves/s
     against 279-320 uncapped, edge of the ~13% run-to-run noise).
   - GPU-SIDE DESIGN (docs/gpu_forward_design_20260904.md): GPU time
     per 16-leaf batch at production sizes is ~24 ms (fed ceiling ~670
     leaves/s with today's kernels); the physical floor at 1,270
     tokens per leaf is ~2,950 leaves/s at 100% of the 4090's bf16
     peak, so the plan's 3,000 needs fewer tokens per leaf (plan 1.4),
     not better kernels. Ranked GPU levers: sync/copy consolidation
     in `batched_priors` (-8-12%), packed varlen trunk (-20-25%),
     inductor compile of a tensor-only `forward_streams` (-15-20% GPU,
     -50% CPU), length-bucketed coalescing (-8-10% at 16, -30% at 64);
     CUDA graphs never unless CPU is the limit again. All of it only
     pays once the server is fed.
   - SHIPPED 2026-09-05 (GPU design option 3): `batched_priors`
     stages all masks in one pinned copy, compacts on the device in
     the reference order, one sync per batch, and the reply carries
     value/cliffness/aux in the same transfer; serve threads log GPU
     ms per leaf. Bit-identical priors on harvested leaves; the CUDA
     tests pass on the box (CPU-vs-CUDA equality, no implicit sync).
     MEASURED: 643 leaves/s in the best 60-s window (401 over the
     iteration; 16 actors), the design's fed ceiling for today's
     kernels; the serve inference stage is ~3.0 ms per leaf.
   - SHIPPED 2026-09-05 (GPU design option 1, behind a switch):
     `wesnoth_ai/packed_trunk.py`, flash varlen attention on the
     packed sequence, `WesnothModel.infer_packed_trunk` (default off),
     `bench_pool --packed-trunk`. CPU tests pass; the CUDA parity,
     no-sync and timing tests plus a 32-game pool run are queued on
     the box. MEASURED on the box: parity as expected (bf16 noise
     ~1e-2 of scale, fp32 1e-6), no implicit sync, GPU ms per 16-leaf
     batch 18.2 -> 10.8 (homogeneous lengths) and 37.4 -> 13.9 (mixed);
     pool: 833 leaves/s saturated (from 652), 489 over the iteration.
     Four serve threads measured worse than two (587 vs 652): not a
     lever. NEXT: flip `infer_packed_trunk` on by default in the
     generation path once the compiled-loop row is in.
   - SHIPPED 2026-09-05 (GPU design option 2, behind a switch):
     the packed layer loop as one inductor graph (dynamic total
     length, attention as an opaque custom op, native bf16 weight copy
     refreshed on every load_state_dict), `WesnothModel.
     infer_compile_packed`, `bench_pool --compile-packed`. CPU parity
     3e-7, no recompile on new shapes. MEASURED on the box: warmup
     6.9 s, 0 recompiles, GPU 10.74 -> 9.79 ms per 16-leaf batch (at
     the 1 ms kill threshold); pool 863 vs 833 leaves/s saturated
     (inside noise), GPU 2.05 -> 1.80 ms per leaf. Kept off by default
     (`az_loop --compile-packed` to opt in). model.py is at 870 lines:
     split PaddedOutput and the stream helpers out in a later pass.
   - CORRECTED 2026-09-05 (whole-pool profile, docs/box_specs.md):
     the actors idle 95% of the time in the reply receive; the serve
     threads are busy ~88% with the GPU wait as the largest item. The
     "under-fed" reading came from iteration averages over a tail
     where most actors had finished. The server, and inside it the GPU
     time per batch, is the ceiling; the actors' own work is small.
     NEXT: measure in the saturated window (pool now logs the best
     60-s rate); apply the GPU levers in the design's order (staged
     priors shipped and queued for measurement, then the packed
     varlen trunk, the compiled tensor-only forward, length buckets);
     try 3-4 serve threads or serve processes to overlap CPU with the
     GPU wait. The PID limit (4,352) and thread caps stand.
   - MEASURED 2026-09-05 (serve host split, docs/box_specs.md): ~36 ms
     of host work per 16-leaf batch per serve thread is the ceiling
     (forward launches 10.9, padded encode 8.5, priors 5.9, unpack
     2.5, wire 2.4) with the GPU at 1.8 ms per leaf. Packed embed
     -4.9 ms (on by default); length coalescing no gain (off).
   - SHIPPED 2026-09-05: serve processes (`ActorPool(serve_processes=N)`,
     `bench_pool --serve-processes`): extra serving processes with
     their own model copy, weights pushed as one blob after each
     publication with a version check (CUDA IPC sharing rejected per
     the torch 2.5.1 constraints), stats merged. Pool row queued after
     the arms; expected toward ~1.5x if the host work is the ceiling.
     az_loop flag and its sync_servers() call after train_step still
     to add once the row is in.
   - MEASURED 2026-09-05 night: two serve processes 1,146 leaves/s
     saturated against 833 with two threads in one process (1.38x),
     exact parity on the leaf check; `az_loop --serve-processes 2` is
     the setting to use on a 4090 box.
   - SHIPPED 2026-09-06 (user order): the iteration's games are a
     shared ticket queue (`ActorPool._post_tickets`, actor_worker
     `_take_ticket`): each actor pulls the next game until the end
     marker, so the iteration's tail is one game long instead of one
     actor's share (the whole-pool profile had the median game
     finishing at 40% of the wall, the iteration average at ~55% of
     the saturated rate). A game's setup depends on (base seed, game
     index) only. Unit test on the queue contract; the slow pool and
     serve-process smokes pass through the real path. Not yet
     measured on a box (one pool row, ~$0.3): expected to bring the
     iteration average close to the saturated rate.
   - Persistent eval workers shipped (`run_elo_batch
     --persistent-workers`, tools/eval_workers.py): 20 seed-vs-seed
     games at 10 concurrent in 97 s against 408 s one-process
     (docs/box_specs.md, evaluation section). An 800-game gate is
     about 65 minutes of one 4090 box. The first build shared one
     policy object between the two sides (3 of 20 games diverged);
     fixed, and the replay agrees exactly with the one-process games
     across repeated runs: the argmax harness is deterministic.
4. **Defects** (plan 1.6): DONE 2026-09-04.
   - `tools/az_loop.py` `_probe` passes `--raw-temperature-a 0
     --raw-temperature-b 0` at sims 0 (every earlier pin compared the
     sampling player on both sides).
   - `tools/step_control.py`: `_clone_weights`/`publish_weights` cover
     the encoder too (flat dict, `model.`/`encoder.` prefixes); the
     restore test checks both encoders.
   - `tests/test_actor_pool_watchdog.py`: the stub carries
     `value_center` and `server_priors`; green.
   - Eval search procedure per player: `--gumbel-root-a/-b`
     (`elo_eval_game`, `run_elo_batch`; default on), procedure tag
     `puct:<sims>` for a plain PUCT root, recorded in the result JSON.
5. **Model cost study** (plan 1.4): SCOPED 2026-09-05
   (docs/model_cost_study_20260905.md, `tools/bench_model_cost.py`
   queued on the box). Hex tokens are 97% of a leaf's sequence and
   ~65% of them cannot be targeted by any legal action; the linears
   are 64% of the FLOPs, so attention sparsity caps at 1.56x and only
   the token count moves the ceiling. The existing relevant-set
   encoding (~415 tokens at production sizes, every legal action keeps
   its token) cuts FLOPs 4x; 2x2 hex pooling with an upsampling target
   head does the same with the action space untouched; a hex-conv
   trunk 7-13x but from scratch; hex-local attention 1.2x at best.
   Every token-cutting option lands on the ~6.6 ms CPU launch floor
   (~2,400 leaves/s) until the GPU design's floor removal ships.
   PROBED 2026-09-05: the seed in relevant-set mode without retraining
   plays 7-10 with 23 of 40 stalls (docs/box_specs.md), so the retrain
   is needed; authorized (user, 2026-09-05) and queued on the box.
   The two imitation arms
   from the seed weights, 0.5 epoch each (full-board control vs
   relevant set), legality-masked holdout CE at equal pairs, then
   800-game PURE matches of each vs `raw:t0` and 400 arm-vs-arm.
   Kill: relevant-set arm below the control by > 2 SE, or masked CE
   worse by > 0.05 nat; the pooling arm then replaces it.
   MEASURED 2026-09-05 night (docs/box_specs.md "Relevant-set two-arm
   retrain"): control -111 +- 13 vs the seed at argmax (800 decisive),
   relevant-set arm -35 +- 13 vs the seed and +26 +- 18 vs the control;
   masked holdout CE 1.264 for both arms against the seed's 1.341 on
   the same pairs. The encoding passes its kill; the recipe's
   continuation itself costs strength that the CE does not show.
   Queued: the lr 1e-5 control (study 7b, $1.7); designed: basis
   transfer by distillation from the seed (7c).
   **Eval at scale** (plan 1.5): 800 raw games in ~65 min / $0.36
   through persistent workers; with the shared inference server
   (2026-09-05, `--shared-inference`) 40 games take 145 s against
   215, mean batch 3.7: the workers' own Python per decision is the
   limit now. ~45 min per 800-game gate. Open: re-pin raw:t0 through
   the server before quoting gates through it.
7. **Training path** (measured 2026-09-05, docs/box_specs.md
   "Training path cost"): 69 ms per experience at the loop's fp32
   batch 1; the factored policy loss is 32-34 ms of it in every
   configuration (per-experience Python), the backward 17-29 ms.
   The training path is 684 s per default iteration against 498 s of
   generation. SHIPPED 2026-09-05: the policy loss over the whole
   batch (parity: loss 2e-7, gradient cosine 1.0000000), per-stage
   timings logged per iteration, az_loop at train_batch_size 16.
   MEASURED: the loss stage 32 -> 22.6 ms per experience (its host
   side dominates; being profiled), batch 16 + bf16 autocast 35.4 ms
   per experience (forward 1.5, backward 4.7), training path 352 s
   per iteration (from 684); parity cosine 0.9994. az_loop now
   defaults to batch 16 with bf16 training.
   - DIAGNOSED 2026-09-05: the loss stage's 22.6 ms is the host
     rebuild of each experience's legality masks plus
     `tools/pathfind_sim.py`'s 512-entry drop-all terrain caches
     thrashing over a learner-sized working set (the 200 bench states
     already exceed them). The trainer now consumes
     `MCTSExperience.masks` (the actor's PackedMasks, bit-packed
     staging, ~0.4 ms per experience on the laptop, bit-identical).
     SHIPPED 2026-09-05: the root's PackedMasks ride on the MCTSNode
     and ship with each MCTSExperience (+3-13% per experience); the
     learner takes the shipped path (pool smoke asserts it). The
     pool-sourced training-path confirmation is queued on the box
     (`bench_train_step --source pool`); expected ~23 ms per
     experience at bf16 batch 16, ~230 s per iteration. Open: the
     pathfind_sim 512-entry cache bounds for anything that still
     rebuilds masks over many maps (the bench-state source does).
6. **Review of the day's changes** (2026-09-04, 7 Opus finders + 3
   refuters per finding): 17 confirmed, 16 fixed the same day (static
   hex cache keyed on a freed address; timeout artifacts that aborted
   every cuda resume; `random` cached across worker games; refusal
   reasons lost in worker mode; per-player root flag recorded for
   arms that never read it; bench rates over truncated runs and over
   surplus actors; double-counted masks; reused bench outdirs; seam
   token and short-batch arithmetic; the priors protocol now refuses
   a nonzero combat-oracle anneal instead of ignoring it). Open:
   `RemoteEncoder.encode` packs masks for every encode, including
   value-only ones (turn search, probes); an `encode(want_priors)`
   switch when those paths return to use.
7b. **Second review** (2026-09-05 evening, 7 Opus finders + 3
   refuters per finding over the day's second half): 18 confirmed,
   all fixed the same evening (parked requests dropped when serving
   stopped, wedging actors; a serve process failing mid-iteration
   invisible until the end; the light state losing the hex basis;
   the imitation pair stream not seed-reproducible with workers; the
   arms script marking short matches done; the hex basis absent from
   eval provenance and guards; startup stderr lost; benchmark
   reporting inconsistencies; terminal candidates counted as
   playouts). Eight rejected. Two refuter calls failed on an API
   safeguard (absent votes). The queued box runs from the
   training-step confirmation on use the fixed tree.
7c. **Third review** (2026-09-05 night, 5 Opus finders + 2 refuters
   per finding over the evening's turn_gap work): 28 findings, 19
   confirmed by both refuters = 11 distinct defects, all fixed the same
   night (a sequential schedule on a position without a distinct
   alternative never played the base and crashed the run; a
   confirmation under the screen's seed reused the screen's playout
   salts unless an offset was passed; a failing worker initializer
   hung the pool forever; the server's outdir was not created; a
   replayed turn was not checked against the screen's realization; the
   proposed-candidate count ignored continue edits and replays; the
   flat-run equivalent counted terminal candidates; in the pre-grader
   script: equal weights regardless of playout count, terminal turns
   without a read, ties read as misses, an unreachable ALIVE verdict).
   Three split (the selected statistics under optional stopping are
   now labelled in the summary; a dropped alternative can still be the
   recorded best, left as is). Six rejected.

## Value head study (2026-09-06/07, user's plan; docs/value_head_study_20260907.md)

- MEASURED: same-turn AUC of the seed's head against human outcomes,
  by phase: 0.65 (turns 1-5), 0.77, 0.82, 0.84, 0.87, 0.93 (31+);
  material 0.56, 0.69, 0.77, 0.84, 0.85, 0.94. The head reads who is
  ahead and beats material in the early and middle game. Half an
  epoch more of the recipe left the pooled AUC unchanged.
- CORPUS: fog/shroud were not recorded; 18.9% of the games were
  fog-off while the encoder hid enemies on all of them. Recorded per
  side now (extractor, builder, `tools/annotate_corpus_fog.py`), the
  encoder's switch set from the two player sides; shroud counts as
  fog, fog-off-with-shroud (20 games) quarantined. Corpus annotated
  locally and on any box that stages the HF tarball.
- MEASURED 2026-09-08 (`training/metrics/value_head/arms_20260908/`):
  `value_head_plus_1` (one more iteration of the recipe on the
  fog-aware corpus, from the pre-encoded records) reads 0.66, 0.77,
  0.82, 0.85, 0.87, 0.94 by phase; Brier 0.230 in turns 1-5 against
  the seed's 0.248. On the 254 clean holdout games (no twin in the old
  corpus) the seed reads 0.62 and 0.75 in the first two buckets, the
  arm 0.63 and 0.76; the ordering against material is unchanged.
- MEASURED 2026-09-08 (plan step 4, HF `tier-b/value_head_arms_20260908/`):
  `value_head_plus_material` (material as an explicit input of the
  head, `--value-material`, same recipe) reads 0.66, 0.78, 0.83, 0.87,
  0.89, 0.94 by phase; paired against the seed +0.01 to +0.03 in every
  bucket, none significant (p 0.12 to 0.55), pooled AUC up in turns
  1-15 and 21-30, Brier down to turn 30; against plus_1 +0.01 to
  +0.025 from turn 6 (p 0.24 to 0.35), Brier slightly worse in turns
  6-15. Does not hurt, may help the middle game by one or two points;
  undecidable at 369 games. Box 50247106 destroyed; the two arm
  checkpoints and records are escrowed.
- MEASURED 2026-09-08: the seed's head with global feature 5 gated by
  fog reads 0.645, 0.766, 0.816, 0.838, 0.881, 0.932 by phase against
  0.647, 0.767, 0.818, 0.838, 0.874, 0.932 with the true count; the
  village lead alone reads 0.58 early (true) and 0.51 (seen). The
  god-view count carried nothing the head used.
- SIGNIFICANCE 2026-09-08 (`tools/analysis/value_head_compare.py`,
  paired by game): plus_1 against the seed, same-turn AUC difference
  +0.014 [-0.010, +0.038] in turns 1-5 and within +-0.03 of zero in
  every later bucket (p 0.24 to 0.99); Brier down in every bucket to
  turn 30 with the interval clear of zero. One more iteration
  sharpens the probabilities and does not change the ranking. The
  2026-09-05 control reads the same way.
- DONE 2026-09-10, the clean seed (`scripts/clean_seed_retrain_box.sh`,
  HF `tier-b/clean_seed_20260909/arm_epoch2.pt`, records in
  `training/metrics/value_head/clean_seed_20260909/`): the 15M
  architecture from scratch, the seed's imitation recipe, the
  deduplicated corpus with the manifest split and the fog gate on,
  cosine over 4 epochs, stopped after epoch 2 by user order when the
  probe flattened. Holdout CE 3.084 / 2.872 / 2.777 at the three
  epoch ends (the seed: 3.10), masked target CE 1.281 (seed 1.34),
  value AUC 0.750 (seed 0.63). Per phase the value head ranks like
  the seed's (same-turn AUC 0.64 / 0.75 / 0.81 / 0.85 by bucket,
  every paired difference within noise) and is better calibrated
  (Brier lower in every bucket to turn 20, intervals clear of zero).
  It replaces the seed as the reference for holdout numbers and for
  the value-head programme's frozen-trunk arm. Named seed2 by the
  user (2026-09-11, HF `tier-b/seed2.pt`). Cost $16 over three boxes
  (one offer expired, one uplink at 70 kB/s).
- MEASURED 2026-09-11 (`training/metrics/elo/seed2_vs_seed_20260911/`):
  seed2 against the seed, PURE raw:t0 both sides, sides alternated,
  ladder maps, 800 decisive games (438-362) with 660 more at the turn
  cap: **+33 +- 12 Elo for seed2**, material-sign diagnostic +23 +- 9.
  The first checkpoint to beat the seed in a match. seed2 is the new
  reference player once re-pinned against itself (the 20-game
  determinism check of docs/box_specs.md); until then matches quote
  both. Box 50568829 (3090, $0.18/h), about $0.40 for the match. The
  match ran per-process workers without --shared-inference; the eval
  profiling of plan 1.5 ran on the same box afterwards (below).
- MEASURED 2026-09-11, plan 1.5 (docs/box_specs.md "Eval path, one
  factor at a time"; `scripts/eval_profile_box.sh`; records in
  `training/metrics/bench_pipeline/eval_profile_20260911/`): the same
  40-game raw:t0 match on one 3090 box, 10 workers: per-process 155 s,
  shared inference 121 s, plus the Rust core 113 s, the Rust core
  alone 154 s; 16 workers 105 s, 20 workers 94 s, 32 workers 125 s.
  The server's GPU is busy 80% of the wall at a mean batch of 4 to 8,
  at 17-24 ms per batch, so small batches are the ceiling, not the
  workers' Python; workers idle two thirds of their cycle, so a box
  takes about 1.25x its cores in workers and no more. An 800-game
  match is about 31 minutes and $0.10 on a 3090 at 20 workers. All
  match scripts now run the shared server and build the Rust core at
  bring-up.
- MEASURED 2026-09-11, plan 1.5 round 2 (docs/box_specs.md "Round 2";
  `scripts/eval_profile2_box.sh`; records in
  `training/metrics/bench_pipeline/eval_profile2_20260911/`): the
  eval server now runs the pool's packed embed (5%, default on) and
  offers the compiled packed loop (`--compile-packed`), which is 12%
  cheaper per batch and a loss overall (139 s against 73: smaller
  batches, and the 40 games ran 25% more decisions under its
  numerics); it stays off. Window 3-5 ms and 24 workers change
  nothing: the batch stays about 8 and the box's CPU quota is
  saturated at about 314 decisions per second while the server idles
  30%. The path is balanced; plan 1.5 is closed for this round at
  about 25 minutes and $0.15 per 800-game match on a 3090 Ti. The next
  eval multiplier needs the per-decision Python in Rust (plan 1.2's
  remaining steps) and fewer tokens per leaf (1.4), both measured
  items, neither a quick win.
- RULING TO RECORD (2026-09-11): seed2 self-pinned through the shared
  path (20 games twice: 18 of 20 identical, the two others the
  predicted bf16 near-tie flips; 6-5 with 9 at the cap). seed2 is the
  reference player from here: `raw:t0` names seed2 unless a match
  says otherwise, and the seed stays as the second reference in
  matches that need the older scale.
- VERDICT 2026-09-09: the head cannot grade alternative turns. Its
  within-position error against 160-playout truth is 0.16 and it
  shrinks real gaps three to one, while the best of four sampled
  turns beats the played turn by about 0.05 (turn_gap pre-grader,
  12 positions). Who-is-ahead AUC does not measure this; every
  outcome-trained head shares the cause (one label per game, 17k
  games). Material grading dropped (user ruling).
- NEXT (user, 2026-09-09), the value-head programme:
  1. Auxiliary value targets from the existing corpus: material
     lost and killed over the next turn, villages held two turns
     on, turns to the end. One label per position instead of one
     per game, no new games needed (KataGo, Wu 2019).
  2. Weight the value loss and each auxiliary loss by a learned
     noise parameter (Kendall, Gal & Cipolla 2018). AlphaGo Zero's
     value weight on human-data-sized corpora is 0.01; ours is 1.0.
  3. A within-position benchmark: 100 self-play positions at
     temperature 0.5, four alternative turns each, 160 playouts per
     candidate (about $7). Every head is scored on it; it is
     regenerated when the player changes.
  4. Value network trained apart from the policy trunk (Phasic
     Policy Gradient, Cobbe 2021): the seed stays byte-identical.
     Two arms, frozen trunk features against from scratch.
  5. Scale: 100k self-play games at temperature 0.5 (about $35),
     with branching for replicate outcomes where the noise share of
     the loss is wanted exactly.
- CONTAMINATION REVIEW 2026-09-08 (docs/data_contamination_20260908.md):
  the seed's lineage trained on 108 of the 369 imitation-holdout games
  (twins in the old corpus) and its A3 value head on about 364 of
  their outcomes; global feature 5 was god-view under fog; 79 clusters
  of one match under two names, one straddling the split. Fixed:
  every tool splits by the manifest (`manifest_holdout_split`), the
  builder and `tools/dedup_corpus.py` keep one copy per match (corpus
  17,019 games, HF `tier-b/replays_dataset_imitation_dedup_20260908
  .tar.gz`), midgame starts and the human anchor skip holdout games,
  feature 5 is gated behind the checkpoint flag
  `fog_hides_enemy_villages`. Holdout numbers of the seed's lineage
  keep the caveat; matches do not.
- Trap fixed: `_load_policy` refused to fall back to a random init on
  a missing path (a two-hour study measured a random net).

## Phase 2 prerequisite (measure before designing)

- MEASURED 2026-09-05 (docs/turn_gap_prereg_20260904.md, run 1,
  $0.90): 60 holdout boundary positions, 4 sampled alternative
  turns each, 40 playouts at temperature 0.5. Fraction with gap
  >= 0.25: 12/60 = 0.20 +- 0.05 against a permutation null of 0.18
  (no information); unbiased split-half gain of the best
  alternative +0.049 +- 0.041 (1.2 SE). Inconclusive by the
  pre-registered rules; neither evidence nor kill fired.
- CONFIRMED 2026-09-05 ($0.84): of the 12 nominal big-gap positions,
  3 confirm out of sample (gains +0.96, +0.38, +0.28), mean +0.13 +-
  0.10, exactly as predicted. Large turn-level gaps exist but are
  sparse (~5% of positions); the average gain of the best of four
  sampled turns is ~+0.05 per turn. Price at this efficiency: about
  $0.6 per confirmed large-gap example (temperature-1 proposer,
  40 + 160 playouts). AUDIT (same day): 12 of 48 alternatives did
  not reproduce across runs (bf16 sampling from a seed), so the
  count is 3-4 of 60; the tool now records action lists for replay.
- DESIGN (docs/turn_proposer_design_20260905.md): a sequential screen
  halves the grading cost with the same hits ($0.58 -> $0.29 per
  confirmed example, no box time); shared inference and the Rust
  worker path take it to ~$0.15-0.20; $0.05 needs a pre-grader with
  within-position residual SD <= 0.2 or a better proposer. Two of the
  three confirmed gaps are base blunders (the base ended its turn
  early), one a find: a "do not end the turn yet" edit is the first
  deterministic proposer arm. Structural point for a ruling: a
  rollout-graded searcher cannot itself pass an 800-game gate
  ($50-300 per gate); it is an instrument that produces confirmed
  pairs to validate a cheap grader. NEXT (pre-registered in the
  design, ~$0.03): forward-only pre-graders (post-turn value, value
  after one argmax reply, material) against the 160-playout truth on
  freshly recorded candidates.
- SHIPPED 2026-09-05 (late): `turn_gap.py --shared-inference`: one
  inference server owns the model, the `--jobs` workers keep the sim
  and the raw player (the same remote base as the eval games; CPU
  parity test exact; provenance and the server's stats in the result
  file). Timing row queued on the box: the continue-edit run's first
  12 positions through 12 and 24 workers, per-position seconds against
  the per-process run of the same positions.
- SHIPPED 2026-09-05 (late): the design's grading schedules in
  `turn_gap.py`: sequential rounds with the drop/stop rules
  (`--rounds 10` for a screen, `--rounds 20 --stop-margin 0.10` for a
  confirmation; validated on the recorded outcomes at 35% fewer
  playouts for the same hits) and confirmation runs that replay the
  screen's recorded turns (`--confirm-from`). Not yet run on a box;
  the next screen uses them.

## Training-signal panel (2026-09-05, docs/training_signal_panel_20260905.md)

Six proposers, three judges, 21 proposals, 7 kept. Ranked: (1) end_turn
decided at the actor level, a decode rule with an end_turn-offset
attribution arm, nothing trained, ~$1.4, kill p <= 0.50 at 800
decisive (the audit's base blunders were early end_turns); (2) the
value-ranked whole-turn lookahead player, whose step A is the queued
pre-grader measurement ($0.03, modal outcome a kill) and step B a
$1.1 match only on a pass; (3) rating-weighted imitation (Bradley-Terry
over the corpus's player ids, fine-tune on the top quartile's winner
pairs, $2.6, waits for the relevant-set control arm's 800-game
number); (4) recruit type sampled from its marginal; (5) the blunder
harvest on on-policy boundaries; (6) an own-trunk boundary value net
from human outcomes; (7) a two-net confirmation rule for any win.
Rejected (14): everything that distils individually confirmed pairs
before a ruling, sigma_s readouts the 48-candidate set cannot resolve,
and three proposals that would train on the holdout games. Open
rulings: distilling confirmed pairs (R2); whether the relevant-set
retrain replaces the seed; the re-pin of raw:t0 through shared
inference.
- TEST 3 QUEUED (2026-09-05 evening, ~$2.4, last in the box queue):
  the corpus's player ratings are fitted (`tools/player_ratings.py`,
  records in training/metrics/player_ratings/): 142 regulars at 30+
  games, top quartile 36 at +147 Elo, the built-in AI at -177 over
  1,399 games; the winner subset is 3,520 games / 347k pairs (13.8%,
  inside the predicted 12-28%, passes the 250k kill). The arm trains
  from the seed on that subset with the control arm's recipe, then
  1,300 games vs the seed and 600 vs the control arm at argmax.

## Cheap measurements worth taking

- DONE 2026-09-05 (docs/box_specs.md "Raw player temperature"):
  temperature sweep 0 / 0.25 / 0.5 / 1 vs `raw:t0`, 40 games each.
  `raw:t0` vs itself: 14-9 with 17 stalls at the 200-turn cap (median
  125 turns); `raw:t0.5`: 22-18, no stalls, median 31 turns;
  `raw:t1`: 7-33. Open: an 800-game match 0.5 vs 0 to decide the
  reference's deployment temperature.
- DONE 2026-09-05: Elo catalog at `raw:t0`, four edges, separate
  file `training/metrics/elo_catalog_raw_t0.json` (docs/box_specs.md
  "Elo catalog at raw:t0"). At argmax the leg-4 self-play product
  equals the seed (+17 +- 55) and the 5M 2291k equals the 15M seed
  (+9 +- 55); the mcts:32 gaps (-367, +223) were procedure effects.
  Open: 800-game edges before any of these is quoted as a fact.

## Rulings (user, 2026-09-05)

- No optimizations conditioned on the current MCTS-like training
  algorithm (leaf reuse, adaptive sims, search batching): the
  training method may change. Throughput work stays on the generic
  path: tokens per leaf, the server's per-batch cost, evaluation
  cost, the training step.
- Evaluations and training legs bill real hours: scope every box
  test to the smallest informative version, train sparingly. The
  relevant-set retrain is authorized at my judgment; it runs only if
  the zero-training probe of the seed in that mode says it is needed.
- Temperature is a tool for decisive playouts, not an object of
  study; the seed will be retrained.
- Results and logs are generated on the run, never as atomic dumps
  at the end: every tool writes partial results as it goes so an
  ongoing process can be checked; a box job past ~1.5x its estimate
  gets inspected and cut.

## Ops notes that are still true

- Boxes: propose specs and cost, wait for a yes, `vms_enabled=false`,
  CPU model EPYC or Ryzen, destroy at the end (`yes | vastai destroy
  instance <id>`; the CLI prompts).
- Launch a detached job over ssh in one call and verify in another;
  the launching session hangs while the child runs.
- No compute on the laptop beyond sub-minute microbenchmarks.
