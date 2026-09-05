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
   - NEXT: the actor loop's untimed ~45 ms per leaf (see 3.), then
     combat and step (phase 3, full corpus sweep).
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
     the box. Turn on by default only after the parity tolerances and
     the GPU ms per batch are recorded.
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
   PROPOSAL (needs a yes: ~$4.3, one box-day): two imitation arms
   from the seed weights, 0.5 epoch each (full-board control vs
   relevant set), legality-masked holdout CE at equal pairs, then
   800-game PURE matches of each vs `raw:t0` and 400 arm-vs-arm.
   Kill: relevant-set arm below the control by > 2 SE, or masked CE
   worse by > 0.05 nat; the pooling arm then replaces it.
   **Eval at scale** (plan 1.5): 800 raw games in ~65 min / $0.36
   through persistent workers; the target (15 min, $0.25) needs the
   workers' forwards batched through one inference server instead of
   ten processes launching batch-1 forwards on one GPU (24 ms of
   forward per decision measured, 1.5 ms per sample when batched).
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

## Phase 2 prerequisite (measure before designing)

- MEASURED 2026-09-05 (docs/turn_gap_prereg_20260904.md, run 1,
  $0.90): 60 holdout boundary positions, 4 sampled alternative
  turns each, 40 playouts at temperature 0.5. Fraction with gap
  >= 0.25: 12/60 = 0.20 +- 0.05 against a permutation null of 0.18
  (no information); unbiased split-half gain of the best
  alternative +0.049 +- 0.041 (1.2 SE). Inconclusive by the
  pre-registered rules; neither evidence nor kill fired. NEXT: the
  pre-registered confirmation run on the 12 nominal big-gap
  positions with 160 fresh playouts each (~$0.75), queued on the
  box; it decides whether large gaps exist at this proposer quality.

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

## Ops notes that are still true

- Boxes: propose specs and cost, wait for a yes, `vms_enabled=false`,
  CPU model EPYC or Ryzen, destroy at the end (`yes | vastai destroy
  instance <id>`; the CLI prompts).
- Launch a detached job over ssh in one call and verify in another;
  the launching session hangs while the child runs.
- No compute on the laptop beyond sub-minute microbenchmarks.
