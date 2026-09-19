# BACKLOG

Live backlog for `docs/plan_20260904.md`. The pre-restart backlog
(1,055 lines of rulings and open items, 2026-05 to 2026-09-04) is
archived verbatim at `docs/archive/backlog_20260904.md`.

## Phase 1 status (CLOSED 2026-09-12)

The exit criterion is met and every step has its acceptance
(docs/plan_20260904.md 4; docs/box_specs.md "Phase 1's exit: the pool
in the relevant-set basis"). Same box, same search budget, the az
legs' configuration against today's defaults in the reference
player's basis: **64.8 -> about 1,050 saturated leaf evaluations per
second, 16x** (6.8x the configuration, 2.4x the relevant-set basis),
against a bar of 10x. Quoted on the server's saturated rate because
two repeats of one configuration differ by 1.64x end to end -- on
this harness, a pool claim under about 1.7x on the iteration column
is noise. 1.2 closed the same day with the Rust-owned state: 17,039
of 17,039 corpus replays clean after every command, and per call
fork 0.019 ms, step 0.050 ms, encode 0.200 ms (the bars were 0.1 /
0.1 / 0.2). `relset` is the reference player by user ruling
2026-09-11; its self-pin over 160 games reads -57 +- 37 Elo, no
asymmetry detected (`training/metrics/elo/relset_selfpin_20260912/`;
four arms replaying one 40-seed set on the shared luck stream, so the
standard error is optimistic).

Left open, neither blocking phase 2, both needing a box and a word
from the user first:
- plan 1.3's 3,000-leaves-per-second-per-4090 target: an A4000
  cannot judge it (30 min, $0.25 on a 4090). The prerequisite held:
  tokens per leaf about 1,200 -> about 300, saturated rate 2.4x.
- a tight self-pin of the reference player (800 games, 18 min,
  $0.20), which would replace the +- 37 Elo above with +- 12.

Measured 2026-09-13 (docs/box_specs.md "Actors buy in-flight leaves"
and the sections after it), on one 24-core 4090:
- **The pool's constraint was the actor COUNT, not actor CPU.** An
  actor blocks on the server for nine tenths of its cycle, so the
  count buys in-flight leaves: 19 -> 665, 32 -> 914, 48 -> 1,006,
  64 -> 1,116 leaves/s against a server saturating near 1,500.
  `az_loop --actors` was 8, then 32 (5af8065), then 24 (5868c35), and
  is now **0 = auto**: as many as --games-per-iter and
  the box allow, clamped by the pids limit host_resources now READS
  and (2026-09-14) by its available memory at ~500 MB per actor after
  6 GB for the learner, both estimates logged for calibration.
  64 measured best and the curve was still rising; the 2026-09-04 host
  that died at 38 is exactly what the read limit prevents, so the
  default no longer has to be conservative. **1.68x on generation,
  from a default.**
- Plan 1.3's 3,000-leaves-per-4090 target is NOT met: the real 4090
  reads 1,450-1,565 saturated at ~320 tokens per leaf, which LOOKS like
  docs/gpu_forward_design_20260904.md's 1,300-1,800 band but is NOT a
  confirmation of it: that band was computed at 1,270 tokens per
  leaf and this is measured at ~320, where the same doc's
  arithmetic gives a ceiling about 4x higher. They coincide because
  the binding cost turned out to be a fixed per-batch LAUNCH that
  does not scale with tokens -- a different mechanism than the one
  priced, so the GPU model still needs re-deriving.
- The eval path wants NEITHER more workers nor more servers. A second
  server halves the mean batch per server (6.9-8.2 -> 3.5-4.4,
  recorded 2026-09-14, docs/box_specs.md "The post-review box run") and
  the per-batch cost is mostly fixed, so it cannot win; the standing
  1.3-1.5x expectation is refuted. `--inference-servers` exists and
  stays at 1.
- bf16 on the imitation trainer, on a 24 GB card at last: 1.25x with
  an equivalent loss, not the 2.9x a memory-starved 16 GB card
  suggested. Still off by default; flipping it wants a holdout curve.
- An actor's own Python is ~3 ms per leaf (encode 1.4, enumerate 1.2,
  edges 0.24), so actor-side optimisation cannot move the pool.

Speed levers left, each a one-factor test on a run that is needed
anyway, ordered by what the measurements say is binding:
- **the largest generation lever left is the iteration's tail, and
  continuous generation removes it** (user order 2026-09-18, built
  the same day: `tools/actor_stream.py`, `az_loop --stream`,
  `bench_pool --stream`, docs/continuous_generation_20260918.md). An
  iteration ends with its longest game, and the iteration rate sits
  1.37-1.93x below the saturated rate on the three hosts measured. A
  stream keeps every actor in a game, the learner steps on windows of
  `--games-per-iter` completed games and publishes into the running
  servers under a gate; a game that lives through a publication is
  counted (straddle columns). NEXT: `scripts/stream_box.sh`, barrier
  against stream against form A (96 games on 48 actors, which the pool
  already runs), twice each on a single-tenant 4090, rule
  pre-registered in the script; then the learner-side question, which
  needs a learner that improves on the prior, run both ways. `--stream`
  is opt-in until then. MEASURED 2026-09-18 (docs/box_specs.md
  "Continuous generation against the barrier"): FAIL under the
  pre-registered rule by a hair in both pairs (1.302x with straddle
  0.69; 1.248x with straddle 0.70); steady windows 1.39-1.49x at
  straddle 0.92-0.93 with the server at its GPU roof; form A 1.10-1.32x.
  Stays opt-in.
- the inference server was LAUNCH-BOUND on the 2026-09-14 shared host
  (406 kernel launches per 16-leaf batch for 3 ms of device time; 20
  ms of host per pool batch against 12 ms of device). The graphed
  serve path (`--graphed-serve`, wesnoth_ai/graphed_serve.py) replays
  one CUDA graph per bucket and read eval infer 27.0 -> 12.7 ms per
  batch and pool saturated 1.30x there. Repeated 2026-09-18 on a
  single-tenant Core Ultra 9 285K host under a pre-registered rule
  (docs/box_specs.md "The graphed server on a quiet host"): pool
  saturated 1.05-1.07x, games per dollar 0.95-0.98x, the eval batch
  1.14-1.16x slower. The gain is the host's, not the model's: with 8
  ms of host per batch the device span already covers 95% of the
  server's infer time. **Both defaults stay OFF**; pass
  `--graphed-serve` on the pool when the first iteration's eager
  `host ms per batch` sum reads well above 12 ms, never on the eval
  path. On that host the pool is actor-bound at 48 actors (the server
  55-65% filled), which is the games-per-iteration lever above.
- the inference server is the ceiling on BOTH paths (eval: over four
  fifths of a worker's wall is spent waiting on it; pool: the server
  idles 40-60% of an iteration but its saturated rate is the roof).
  A second serve process per GPU is measured on BOTH paths
  (2026-09-14, docs/box_specs.md "The post-review box run") and moves
  neither. On the POOL it raises the saturated rate 1.45x and games
  per dollar 0.91x: an iteration of one game per actor ends with its
  longest game, one actor's serial chain of leaf batches that a
  second server barely shortens; the capacity is cashable only with
  more games than actors per
  iteration, i.e. the lever above. On EVAL splitting the same workers
  across two servers halves the mean batch per server (6.9-8.2 ->
  3.5-4.4, recorded 2026-09-14) and the cost is mostly a fixed
  per-batch launch, so it cannot win. The standing 1.3-1.5x
  expectation is refuted on both; `--serve-processes` stays at 1.
- the trainer is GPU-bound: bf16 autocast is built (`--bf16`) and
  timed only on a 16 GB card; TF32 for the trunk is untried.
- the az training path is NOT a lever: on the pool's own experiences
  (masks shipped, as production trains) the step costs 2.22 ms per
  experience, 0.10 in the policy loss, and the training path is 4% of
  an iteration (2026-09-14, docs/box_specs.md "The serve batch is
  launch-bound"). The 35.4 ms on record was the bench rebuilding masks
  its experiences did not carry.
- actor-side Python is NOT a lever, now settled: an actor's own
  Python is ~3 ms of a 27-58 ms per-leaf cycle and the rest is
  waiting on the server (docs/box_specs.md "The actor's per-leaf
  Python"). Spend on in-flight leaves or on the server, never on the
  actor's encode or mask.
- bugs fixed 2026-09-13 while measuring, each with a regression test:
  an out-of-memory inside `backward()` left partial gradients that the
  retry double-counted (tools/supervised_train.py); the search priced
  a REFUSED action by the material draw tiebreak, so a side that was
  ahead saw rejected actions as favourable draws (tools/mcts.py); a
  recruit bounce was written to the core's throwaway view and erased
  by the next command (tools/wesnoth_sim.py); the movement-class cache
  was keyed on a dict's ADDRESS (wesnoth_ai/game_core.py); and
  `--help` crashed on three entry points. Open, reported and not yet
  fixed: `turn_search`'s mover_mp0 boundary frame mutates Unit objects
  shared with the live game (tools/turn_search.py:440, non-default
  frame); the driver records infer_compile=False under
  `--shared-inference --compile-packed`, so a compiled outdir cannot
  be resumed (tools/run_elo_batch.py:703); mid-game [time_area]
  changes do not reach the core's baked map tables.
- a second hunt (2026-09-13, over the Rust kernels, the Elo accounting
  and the Wesnoth rule layer) reported eleven more, NONE fixed. In
  order of what they corrupt:
  * ~~hide cover decided by a hand-rolled overlay allow-list~~ FIXED
    2026-09-13 at the root and swept over the whole corpus. The
    sweep shows NO REGRESSION; it does not certify the new rule
    (it passes under the old one too -- see below). The
    engine matches the hex's terrain CODE (`*^F*`, `*^V*`, `Wo*^*`),
    not its defense class; `terrain_resolver.hides_cover` transcribes
    those globs and both the Python predicate and the Rust core's
    baked flags read it. Over the Ladder pool ambush gained 478 hexes,
    submerge 153 and concealment 93, while concealment LOST 302 to the
    farmland correction. 17,039 of 17,039 replays reconstruct clean
    (on the staging set the box used: the current corpus plus 20 games
    that have since left it -- docs/plan_20260904.md has the count)
    after the fix, which shows no regression and does not certify the
    new rule (docs/box_specs.md "Hide cover after the root fix: the
    corpus sweep, and what it does NOT certify"; docs/wesnoth_rules.md
    has the rule, `tools/analysis/hide_cover_census.py` the census).
  * **the Elo catalog sums repeat measurements of one pair as
    independent evidence** (tools/elo_catalog.py:360; the edge key is
    the games-dir name and nothing compares seeds). Both generators
    default to a fixed seed base, and at raw:t0 a rerun is
    deterministic, so a re-pin into a fresh outdir doubles n and
    shrinks the standard error on no new information. Already live in
    the committed catalog: two edges for `ref~old` pool to n=240.
  * basis, precision and batch estimands are dropped at the catalog
    boundary (tools/elo_collect.py:278): guarded three times inside a
    dir, lost between dirs, so a relevant-set edge and a full-board
    edge can be pooled into one fit with no warning. `value_center`
    and `ELO_MOVES_LEFT_UTILITY` change the searched player and reach
    no result field at all (tools/elo_eval_game.py:352).
  * every eval game shares one combat-luck stream
    (tools/wesnoth_sim.py:862): `_rng_requests` restarts at 0 per game
    with an empty salt, a correlation the standard error does not
    model.
  * ~~a reverse-plague corpse takes a different unit id when BOTH
    combatants die~~ **REFUTED** 2026-09-13. That state is
    unreachable: 1.18.4 floors drain damage so a strike cannot kill
    its striker, a target death ends the fight, and the hypothetical
    both-die path suppresses plague outright. Our two implementations
    mirror the floor and both loops break on the first death, and for
    the two reachable plague cases the appliers already agree on the
    corpse id. Written up with the verbatim quotes in
    docs/wesnoth_rules.md, "At most ONE combatant dies per fight".
  * ~~`rows_from_landable` does not bounds-check a token index~~
    FIXED 2026-09-13, and the severity was understated: an
    overflowing token stays inside the buffer for every unit but the
    last, so it writes into the NEXT unit's legality row SILENTLY;
    only the last unit's overflow panics. Not reachable today (both
    callers derive the token map and the row width from one object in
    one call) but the invariant spans three construction sites in two
    files. `__phase__` is 9; the test skips below it, so the fix
    needs a box build before it is trusted.
  * ~~the Rust combat bridge silently treats an out-of-range defender
    weapon as "no counter-attack"~~ FIXED 2026-09-13 on the BRIDGE,
    not the oracle: `-1`/`None` is Wesnoth's "no counter", an index
    past the end is a caller bug, and swallowing it resolves a
    different fight than the certified reference.
  * ~~`_modify_unit_action` mutates a fork-shared Unit in place~~
    FIXED 2026-09-13. It was the last offender in the file; it and
    `_object_action` now share one `_swap_unit` helper. The existing
    fork fingerprint guard was BLIND to it because it fired
    `_object_action` first, which had already swapped in a
    fork-private copy; reordered, it fails against the old code.
  * ~~`[effect] apply_to=new_ability` is silently dropped~~ FIXED
    2026-09-13, together with the deeper mistake under it (members
    keyed by TAG, not `id=`) -- see "Scenario [effect] members are
    named by id=" below.
  The full reports are in this session's workflow transcripts.

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
   - DONE 2026-09-11: the observation kernel (port plan 2c,
     `wesnoth_core.observe_side`, `wesnoth_ai/observe.py`): one Rust
     call per decision returns the vision disc, the visible units,
     the reach-context flags and the recruit row; the encoder and
     the mask builder read them. Certified (tests/test_rust_observe.py
     plus the encoding, enumeration and seam suites on a box); on by
     default (`WESNOTH_RUST_OBSERVE=0` for the Python path).
   - MEASURED 2026-09-11 (docs/box_specs.md "The observation kernel
     and the CPU budget"): a lone game 1.2x faster, the 40-game
     match unchanged (79-82 s off, 79-94 s on, 5% less CPU). The
     pre-registered kill (under 1.15x) applies: less worker Python
     is not an eval lever. The box was not CPU-bound (5.4 of 16
     cores, 5% of quota periods throttled; round 2's "quota
     saturated" reading was wrong). The match wall is the loop
     through one server: about 25 ms per batch, mostly a fixed GPU
     launch cost (14-15 ms at batch 6-8), 7.5 of 20 workers per
     batch. Levers left, CORRECTED 2026-09-13: the second server
     process is REFUTED on this path (it halves the mean batch and
     the cost is a fixed per-batch launch, see the entry above), so
     what remains is a fixed-shape forward (CUDA graphs over bucketed
     lengths) and fewer tokens per leaf (plan 1.4). On CUDA graphs,
     read docs/gpu_forward_design_20260904.md section 8 first: it
     ranks them LAST, but it prices 16-leaf POOL batches, not the
     6-8 batches this path runs, so the ranking does not settle it.
   - DONE 2026-09-12 (user order "complete the Rust port, including
     for eval and pool"): the relevant-set basis on the kernels
     (port plan 2d): the reach rows and the relevant set from
     `observe(reach=True)`, the subset from the cached full-board
     arrays, the mask rows through `rows_from_reach` in the subset's
     token space. Certified on a box (52 tests); the reference
     player's lone game 27.0 -> 14.5 s for the same decisions, its
     40-game match 73-79 -> 57 s (docs/box_specs.md "The relevant-set
     basis on the Rust kernels"). Combat (port plan 3a,
     `rust/wesnoth_core/src/combat.rs`) the same day: 3,000 fuzzed
     fights, the [mp_checkup] fixture and 17,039 corpus replays
     identical to the Python resolver; default on. Training under bf16 autocast is built
     (`--bf16`, fp32 weights; tests/test_imitation_flat_batch.py)
     and timed against fp32 on the box (`scripts/train_bf16_box.sh`);
     its validation is the next training run's holdout curve and
     match.
   - DONE 2026-09-12 (port plan 3b/4, user order "complete the
     port"): the Rust-owned state `GameCore` with init_side,
     end_turn, move, attack and recruit applied in Rust, the
     observation and the encoding as methods over its records, and
     `WesnothSim(use_core=True)` keeping it as the state of record
     behind one Python view refreshed in place. Certified on a box:
     17,039 of 17,039 corpus replays compare clean after every
     command against the Python applier (5.48M commands in Rust;
     `tools/diff_core.py`, `scripts/diff_core_box.sh`), plus the
     round-trip, observation, encoding byte-identity, twin-simulator,
     determinism and fork-isolation suites. Default off
     (`WESNOTH_RUST_CORE=1` to enable) until the eval-path timing
     (`scripts/core_sim_box.sh`) is read; Python still holds the
     unit builders (recruit, advancement, plague), the scenario
     events, the action translation and the policies' view.
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
   - **THIS NUMBER HAS NO RECORD (flagged 2026-09-13).** Every pool
     JSON in the repo carries `serve_processes: 1`, the commit that
     added this sentence added no metrics file, and 1,146 sits exactly
     inside the band docs/box_specs.md pre-registered as the
     EXPECTATION ("~1.5x, 1,100-1,300 leaves/s"). Four other places
     say the second serve process is still unmeasured, including
     box_specs' own section header "built 2026-09-05, not yet timed on
     a box". Treat it as a prediction written up as a result until a
     box reproduces it -- scripts/postreview_box.sh phase 2 does
     exactly that. The question may also be moot: a SINGLE server
     measured 1,470-1,565 saturated on 2026-09-13, well past the ~850
     ceiling that motivated a second one.
     Claimed 2026-09-05 night: two serve processes 1,146 leaves/s
     saturated against 833 with two threads in one process (1.38x),
     exact parity on the leaf check; `az_loop --serve-processes 2` is
     the setting to use on a 4090 box.
   - MEASURED 2026-09-14 (box 51006981, one factor, 48 actors and 48
     games, records `training/metrics/bench_pipeline/postreview_20260914/
     pool_bf16_p{1,2}.json` in the committed configuration and
     `pool_p{1,2}.json` in fp32): the second process raises the
     saturated rate 1.45x (bf16) and 1.15x (fp32) and games per
     dollar 0.91x and 1.00x. The median game finishes sooner
     (301 -> 197 s) and the iteration does not, because it ends with its
     longest game, played by one actor with one leaf batch in flight.
     Dead as a lever on its own (docs/box_specs.md "The post-review
     box run"); `--serve-processes` stays at 1.
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
   MEASURED 2026-09-11 (box 50585036, `scripts/seed2_relset_box.sh`,
   docs/box_specs.md "The relevant-set twin of seed2 at one pass"):
   seed2's twin from scratch in the relevant-set basis, one pass
   (user order, cost), beats seed2's own one-pass checkpoint +56 +- 12
   Elo (800 decisive raw:t0 games, 464-336, 623 more at the cap); its
   holdout probe reads better on every head (CE 2.84 vs 3.08, value
   AUC 0.75 vs 0.74). The pre-registered kill does not apply. Caveat:
   the twin's pass held 2,825,379 pairs against seed2's 2,491,171 on
   the same files, recipe and seed; the recipe's count is the twin's
   (manifest winner actions + end_turns + value states), so seed2's
   run was 12% short. MEASURED the same night (docs/box_specs.md
   "Pair census"): the full-board trainer at batch 64 on a 24 GB
   card hits CUDA out-of-memory on the largest boards' batches and
   dropped them with a DEBUG line (52 of ~750 batches over 300 files;
   the relevant-set records fit). seed2 therefore never trained on
   the biggest boards' games, and its lineage's eval numbers carry
   that; the trainer now splits such batches and accumulates
   (nothing lost). The +56 has two confounds in the twin's favor
   (13% more pairs, the large boards). The clean one-factor number
   needs seed2's one pass rerun with the fixed trainer at equal
   pairs (about 4 h, ~$3.5 with the batched flow); user's call.
   Self-timings through the shared server: 1.30 ms of GPU per leaf
   for the relevant-set basis against 1.68 for the full board at
   mean batch 6-7.
   Between probes the relset trainer runs 138 pairs/s against seed2's
   107 on the same recipe: fewer tokens do pay 1.3x in the training
   loop. Its holdout probe costs 143 s against seed2's 26 s (the
   relevant-set encoding of 1,200 holdout pairs through the
   simulator), 28% of its wall, which is why both runs read the same
   cumulative 100 pairs/s; the probe cache below removes that. Next
   imitation run: `--eval-every 250000`.
   BUILT 2026-09-11 evening (user: "if there is Python, there is room
   for optimization"): the imitation trainer's batched flow embeds
   the batch from one pinned host buffer (`encode_from_raw_embedded`,
   the server's path), runs `forward_embedded` once and scores every
   head over the PaddedOutput (`wesnoth_ai/imitation_loss.py`); one
   host-device synchronization per batch instead of about four
   blocking copies per pair. The holdout probe caches its sample
   after the first draw (`_evaluate(cache=)`), so a probe no longer
   reconstructs 150 games through the simulator. Differential test
   against the per-sample reference (values, fired heads, total,
   every gradient): tests/test_imitation_flat_batch.py. Found on the
   way: the batched flow backpropagates the UNWEIGHTED actor
   cross-entropy (the action-type weights only scale the per-pair
   CPU flow's total); seed and seed2 trained that way and the new
   flow keeps it. MEASURED the same night on the idle 4090
   (docs/box_specs.md "The imitation trainer timed"): 138 -> 257
   pairs/s on the relevant-set records (1.86x); the main thread now
   waits on the GPU 71% of its time. Next levers, one at a time with
   a holdout curve and a match as the check: TF32 matmuls, bf16
   autocast for the trunk, the packed trunk in training mode.
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
  about 25 minutes and $0.15 per 800-game match on a 3090 Ti.
  CORRECTED 2026-09-13: this paragraph's other two readings are dead.
  The box was NOT CPU-saturated -- "round 2's 'quota saturated' reading
  was wrong", 5.4 of 16 cores and 5% of quota periods throttled -- and
  the per-decision Python in Rust is NOT the next eval multiplier: a
  worker spends over four fifths of its wall waiting on the server, so
  removing worker Python moves nothing (the pre-registered kill under
  1.15x applied). What is left is fewer tokens per leaf (plan 1.4) and
  a fixed-shape forward.
- SUPERSEDED THE SAME NIGHT -- the reference player is `relset`, not seed2 (user ruling 2026-09-11 night; CLAUDE.md and the top of this file). Pointing an 800-game gate at seed2 produces a number that compares to nothing. Kept for the reasoning only:
  RULING TO RECORD (2026-09-11): seed2 self-pinned through the shared
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

## Scenario [effect] members are named by id= (2026-09-13, FIXED)

A hunt for siblings of the hide-cover bug -- a Wesnoth rule decided by
a hand-rolled enumeration with a silent wrong default -- found two
live ones, both in `tools/scenario_events._apply_effect_to_unit`, both
firing on **2p Silverhead Crossing**, one of the 21 Ladder maps and
351 of the 17,019 corpus games.

The root cause is one mistake: OUR model names an ability and a
weapon special by the engine's `id=` -- that is what unit_stats.json
scrapes, what combat reads and what the fog gate reads -- and the
effect handler read the TAG. Three specials share `[chance_to_hit]`
(`magical`, `marksman`, `deflect`) and every hide ability is `[hides]
id=<something>`, so the tag produced `chance_to_hit` and `hides`,
which no consumer knows: the effect was created and then did nothing.
(The ENGINE uses both keys -- numbers resolve by tag, identity by id;
docs/wesnoth_rules.md "An ability or weapon special has TWO keys" has
the citations.)

1. **`magical` never reached combat.** Silverhead's `prestart`
   `[object]` gives its side-3 Tentacle a ranged arcane 100-1 "evil
   eye" with `{WEAPON_SPECIAL_MAGICAL}`. `wesnoth_ai/combat.py` reads
   `"magical" in weapon.specials`, which SETS chance-to-hit to 70
   (`cumulative=no`; only `marksman` is a floor), so every player
   who attacked the Tentacle at range ate its counter at their OWN
   terrain's chance-to-hit (30% in forest, 60% on flat) instead of
   Wesnoth's flat 70%.
2. **`apply_to=new_ability` was not dispatched at all**, so the
   `{ABILITY_SUBMERGE}` the same `[object]` grants was dropped. The
   Tentacle stands on `Wo` at WML (1,1) and both playing sides run
   `fog=yes`, so Wesnoth hides it and we showed it, all game.

Fixed with `_effect_member_ids`, used by both specials sites and the
new `new_ability` / `remove_ability` branches; an `apply_to` outside
the modelled and cosmetic sets now warns once instead of vanishing,
which is how (2) survived. `tests/test_effect_ids.py` covers both,
end to end on the real scenario (6 tests). `OBSERVATION_EPOCH` is 3.

**Certification owed.** (1) changes COMBAT, so it needs the corpus
sweep that `scripts/hide_cover_cert_box.sh` runs, on a box. Budget:
the last full sweep was 233 s of wall on a 28-core box plus setup, so
well under an hour at roughly $0.30-0.50. Queue it with the next box
rental rather than renting for it alone.

Two further findings from the same hunt, NOT fixed:

- **`[set_specials]` without `mode=` REPLACES the weapon's specials;
  we append.** `src/units/attack_type.cpp:416-429`: `if(mode !=
  "append") { specials_.clear(); }`, with a deprecation warning when
  `mode=` is absent. Modelling replace needs a way to say "these are
  ALL the specials of this weapon": our `Attack.weapon_specials` is an
  additive overlay that `_to_combat_unit` unions with the scraped base
  (`tools/replay_dataset.py`), so a faithful fix is a contract change
  on the combat path, which carries bit-exact parity and would owe the
  corpus sweep. Zero live impact: the only `[set_specials]` in the
  shipped multiplayer data is Hornshark Island's `MODIFY_BOWMAN`,
  which is not in either pool, and a Bowman's bow has no base
  specials, so append and replace agree there. Cited in the code at
  the call site and in docs/wesnoth_rules.md.

- **The encoder's terrain one-hot is wrong on most forest hexes.** Not
  a rule divergence -- every Wesnoth rule routes through
  `terrain_resolver` / `_terrain_codes`, and the one rule-bearing
  member (`Terrain.VILLAGE`) is decided correctly. But
  `_first_terrain_id` (`wesnoth_ai/encoder.py`) tie-breaks a
  multi-member set with `next(iter(...))`, i.e. by enum ordinal, and
  `FLAT` (3) sorts before `FOREST` (4): `Gs^Fp` encodes as FLAT while
  `Hh^Fp` encodes as FOREST. **1,356 of 1,572 forest-overlay PLAYABLE
  Ladder hexes (86%)** are not labelled forest, on all 21 maps
  (`tools/analysis/hide_cover_census.py`, record
  `training/metrics/bench_pipeline/hide_cover_20260913/census.json`) --
  the same border-stripped basis as the hide-cover census, which is
  the right one here because `parse_map_data` strips the border ring
  and the encoder never sees it. (Border-inclusive the same counts
  read 1,499 of 1,759.) Separately, `replay_dataset._TERRAIN_BASE`
  defaults an unlisted base to FLAT, so 5,742 of 20,726 playable
  Ladder hexes (28%) take a default, some
  wrongly (`Xv` void, `Xos` wall, `Wot` deep water, `Hhd` dry hills,
  `Ai` ice, `Qlf` lava). Fixing it changes every observation the
  policy has ever been trained on, so it wants its own arm and an
  800-game match, not a drive-by.
- **Two more `[hides]` abilities are outside `_AMBUSH_ABILITIES`**:
  `burrow` (Horned Scarab) and `swamp_lurk` (the Swamp Lizard -- the FILE is
  Crocodile.cfg but `id=Swamp Lizard` and no unit type called
  Crocodile exists in 1.18.4;
  `wesnoth_src/data/core/units/monsters/Crocodile.cfg:143`, its
  `terrain=S*^*` at :153). It IS in the pinned scrape, the only
  carrier of the 356. Latent: neither carrier is in the pools or in any
  of the 17,019 corpus games.

Two `apply_to` values DO appear in the mini pool and look alarming;
both are fine, checked rather than assumed. `loyal` (9 occurrences
across the three enclave scenarios) sits inside a `[trait] id=loyal`,
and our loader records the TRAIT, which is what the upkeep math reads
(`replay_dataset`, init_side gold): those Tentacles come out
`traits=('loyal',)`. `movement_costs` (5 occurrences over THREE pool scenarios:
`2p_mini` and `2p_mini_edited` twice each,
`Modified_Tiny_Close_Relation` once) makes a guardian's castle cost 99 so it cannot
leave; those are side-3 units that never move in our sim anyway, and
the effect is not dispatched through `_apply_effect_to_unit` at
scenario build, so the new warning does not fire on them.

Ruled out with evidence, so nobody re-hunts them: terrain resolution
(354 distinct playable codes over the 21 Ladder maps, 365 over the
Ladder and mini pools together, 817 over all 114 tracked .map files;
every code resolves through terrain_db EXCEPT `Md^Xm`, `Mm^Xm`,
`Ms^Xm` and `_off^_usr`, which take `_get_underlying`'s
unknown-composite branch because it splits on `^` before checking
the whole string -- 1,734 Ladder hexes, values still right);
village/castle/keep detection (bidirectional against `terrain.cfg`);
the trait tables (divergent but dead -- `roll_traits` reads
`unit_stats.json`); ability-macro scraping (0 dropped macros across
the 158-type Default-Era closure: the six factions' recruit, leader
and random-leader pools give 78 seed types, closed over
`advances_to`); ZOC; init-side healing; the
unknown-unit fallback (94 distinct types in the corpus, all present);
`sight_radius_for`'s `max_moves` proxy (only 4 core types set
`vision=`, none reachable); and the remaining weapon-special gaps
(`absorb`, `plague_type`, `stun` -- none on a recruitable type).

## Stale-claim sweep: what is left after the corrections (2026-09-13)

Twenty-six claims that a LATER measurement in this repo refutes, or
that disagree between two places. The ones that would cost money or
correctness are fixed (see the commit). These are the rest, left
because they are small, need a judgement call, or need a box.

- **`_TERRAIN_BASE` / `_parse_hex_code` attribution.** `box_specs.md`
  credits `gpu_forward_design_20260904.md` with having "predicted
  exactly this" for the 4090's 1,450-1,565. The band it predicted
  (1,300-1,800) was computed at 1,270 tokens per leaf; the measurement
  is at ~320, where the same doc's arithmetic gives a ceiling about 4x
  higher. The numbers coincide because the binding cost turned out to
  be a fixed per-batch LAUNCH that does not scale with tokens -- a
  different mechanism than the one priced. Reading it as confirmation
  means nobody re-derives the GPU model, which is the thing phase 2's
  budget rests on.
- **`box_specs.md` presents a 20-game wall as a 40-game wall.** The
  table headed "wall s for 40 games" gives 408 for one-process-per-game;
  the run it cites (`eval_workers/eval_plain.log`) reads "20 pending",
  and a neighbouring section labels the same 408 as 20 games. The 145
  row IS 40 games. So the table understates shared inference against
  one-process by 2x (2.8x shown, ~5.6x real) and nobody knows what the
  harness change actually bought.
- **"the pool reaches 1.2 ms per leaf at batch 16 on a 4090"** is used
  to justify "the lever is more decisions in flight", but no pool
  record has 1.2 -- every recorded pool GPU-ms-per-leaf is 1.75-2.82.
  The nearest matching text is a DERIVATION in gpu_forward_design
  ("~1.0-1.2 ms per sample"). An estimate that hardened into a
  measurement.
- **"A second decision in flight per worker would not help"** is
  asserted, never measured, on the grounds that the game loop is
  sequential. In-flight decisions per box is precisely the lever that
  produced 1.68x on the pool. The 40-worker arm that WAS tried
  contends for cores, which two games per worker would not. Worth
  pricing rather than leaving as a settled negative.
- **Three token-ratio figures for one quantity**: the relevant-set
  basis is quoted as 2.7x, 4x, and 1,200 -> 300 fewer tokens. The 4x is
  the FLOP ratio wearing the token label; the token ratio is 2.7x on
  the bench states and 4x in the pool. Low stakes, but it is cited as a
  single fact.
- **Two BACKLOG "NEXT" items are already in the code**: `--packed-trunk`
  defaults True in az_loop, and both `--serve-processes` and the
  `sync_servers()` call after train_step exist.
- **Smaller code-vs-comment disagreements**, all verified: two
  profilers say the trainer's turn cap is 200 where the code says 100;
  `rewards.py` documents two defaults as non-zero that are 0.0;
  `eval_inference_server.py` attributes two numbers to a box_specs
  section containing neither; `eval_vs_builtin.py` computes every
  wall-clock estimate without dividing by its own `--parallel`;
  `wesnoth_rules.md` asserts Default Era uses 5 gold per village (no
  shipped value is 5), claims a `random_traits=no` closed set that has
  a fourth member, gives an AMLA sequence its own rounding rule
  contradicts, miscounts 2p maps, and cites `special-notes.cfg` for
  `charge` where the enforcing definition is `weapon_specials.cfg`;
  `design_constants.md` gives two values for one PUCT product without
  saying they are different operating points. Both files' tables of
  contents are stale (neither lists every section; the counts first
  published here were not checked and are withdrawn).

Checked and CLEAN, so nobody re-sweeps them: `raw_argmax_control`,
`turn_gap_prereg`, `data_contamination` (the one doc that keeps the
17,104 / 17,019 distinction straight), 11 of 14 entries in
`design_constants`, 18 of 19 spot-checked `wesnoth_src/data/` citations
in the rules catalog with every quantitative claim reproducing exactly,
`az_loop`'s quoted measurements, and the argparse/docstring numerics of
eighteen other tools.

## Mirror audit: paths that must agree, and what checks them (2026-09-13)

The shape: two implementations that must produce identical results,
with nothing asserting that they do. A mirror without an equality test
drifts, and here a drift corrupts either the simulator's fidelity or
the strength verdict.

FIXED in this pass: the two terrain tables (demonstrated drift, see the
commit), and the launcher's RUST banner answering importability instead
of capability. Also made visible: `_fallback_counter_weapon`
(`tools/combat_outcomes.py`) is a KNOWN-divergent v1 heuristic taken
when the strike DP overflows -- it picks a different retaliation weapon
than `choose_defender_weapon`, and it had no log line, no counter and
no test. It now counts itself and warns once.

Open, in the order they would bite:

- **What `diff_core`'s 17,039-replay sweep does NOT compare.** Worth
  knowing before citing it as certification. About 1.15% of commands
  are applied by literally the same function on both sides
  (`game_core._python_path` calls `replay_dataset._apply_command`),
  which takes every `pickadvance` and every `init_side` of any scenario
  carrying a `first_time_only=no` event -- so per-turn healing, poison,
  regeneration, MP refresh, income and ToD are self-compared there. The
  map, terrain, fog and mask are the SAME Python objects on both sides
  (`Map.__deepcopy__` aliases the hex set), so a terrain divergence is
  structurally unrepresentable and those fields are not compared
  anyway. `stash=False` means `_defense_table`, which drives combat
  defense, is never compared by the sweep. Unit construction,
  advancement and every static table are single-sourced Python. The
  observation and the encoding -- the two outputs the model consumes --
  are not in the sweep at all. Four compared fields are constants in
  this corpus, and `_rng_request_counter` has no writer anywhere, so
  `state_key` hashes a constant. `recall` appears in 0 of 4,000
  replays. In the TEST path, `diff_core(gz, every=5)` skips 4 of every
  5 commands.
- **`tools/mask_sim_fuzz.py` runs in no test.** `wesnoth_sim` states as
  present-tense fact: "fuzz-verified 2026-08-17: 0 live rejections in
  11,294 random mask-driven steps". The script is referenced by no
  test, no marker and no CI, and the mask has since gained the Rust
  batch enumeration, the observation kernel, `_rows_from_observation`
  and the relevant-set basis. Re-run 2026-09-13: still 0 rejects in 546
  steps with the Rust enumeration active -- so the contract holds, but
  the quoted number is a 2026-08 measurement wearing a guarantee's
  clothes. Either wire it into the suite or re-date the sentence.
- **The CUDA numerics tier never runs on a laptop.** `packed_trunk`'s
  flash path needs CUDA and fp16/bf16, so `test_packed_trunk.py`,
  `test_packed_embed.py` and `test_packed_compile.py` all exercise the
  REFERENCE attention, never the kernel that serves production.
  Flash-vs-padded, compiled-bf16, staged-priors device-vs-CPU and
  fp32-vs-bf16 training parity all live in `*_cuda.py` files that skip.
- **`diff_replay._castle_network_from` reimplements
  `visibility.leader_castle_network`** -- the declared SHARED contract
  -- inside the fidelity oracle itself, with different semantics and no
  equality test. Harmless today only because `diff_replay` checks
  occupancy first.
- **`tools/fog.py` is a complete second visibility implementation with
  ZERO importers** and its own ability-to-terrain table. Delete it or
  test it against `visibility`; the project's own rule is to prefer
  removing.
- Smaller: `test_encoder_batch.py` uses three snapshots from ONE replay
  on ONE map, so hex streams are all equal length and ragged padding is
  never covered, and nothing anchors `encode_from_raw_padded` /
  `encode_from_raw_embedded` (what the inference server runs) back to
  `encode_from_raw`. `test_batched_gumbel.py` asserts sim counts match,
  not the root action or the visit distribution, while
  `--mcts-batch-size` is a live flag. `test_rust_observe.py` has no
  engagement counter, so its hider and level-0-ZoC branches are
  unasserted, and its reach context is a THIRD hand-written
  transcription living in the test file.
- On a fresh wheel, `_rows_from_observation` bypasses the Python path's
  relevant-set invariant assertion: the Rust kernel silently DROPS a
  landable hex whose token is -1 where Python raises.

Verified genuinely well covered, so nobody re-hunts them: the
vectorized enumerator against its reference (extended to 9 real states
x 4 rejection-set variants x 4 bias configurations, 36 comparisons, no
disagreement); the Python masks against Rust `enumerate_moves`
(28 Rust-engaged builds with rejection sets injected -- branches no
existing test populates); server priors against local enumeration,
whose combat-oracle guard was PROVEN load-bearing by constructing the
disagreement it catches (priors differ by 2.8e-02 at a mismatched
decision step); `compact_action` round-trip over 1,276 actions; the
three pathfinder reach implementations; batched against reference
policy loss including every parameter gradient; `forward_batch` /
`forward_padded` against a single forward; the leaf wire; the replay
round-trip; Python against Rust `encode_raw_streams` on ten adversarial
states the harvest cannot produce; and hex distance (four copies) and
defense percent (two wrappers) compared exhaustively.

One correction to a code comment while there: `pack_masks` DOES ship
the combat-oracle bias arrays, and with a matched decision step the
server path reproduces the reference exactly. The guard is right, but
"server-side priors do not carry the combat-oracle anneal" is not why
-- what the actor cannot know is the caller's step.

## Lifecycle audit of the pool and the eval path (2026-09-13)

A sweep for OS resources created on a repeating path and not reliably
released, with the error, timeout and kill paths given equal weight --
these run for hours on rented boxes, so a leak bills. Two findings were
DEMONSTRATED with a script, not argued.

Being fixed in this pass (the pool):

1. **Actors have no parent-liveness check, so a killed learner orphans
   all of them.** `actor_worker`'s body is a blocking `ctrl_q.get()`
   with no deadline, and the actor inherits BOTH ends of its control
   queue at spawn, so the pipe never reaches EOF. `daemon=True` covers
   only a clean interpreter exit; on kill -9, an OOM-kill or a
   container-supervisor kill the actors survive for the rest of the
   rental, holding the cgroup PIDS budget and their RSS. A supervisor
   relaunch then starts a fresh pool on top of them -- and the pids
   controller is exactly what produced 0 leaves/s on 2026-09-04.
   `serve_worker` already polls `mp.parent_process().is_alive()`; the
   actors were never given the same guard.
2. **A straggler actor eats the next iteration's tickets and never
   reports done.** At the hard deadline the manager breaks with actors
   still outstanding and neither stops nor resynchronises them, then
   broadcasts PLAY for iteration N+1 to all of them. The straggler is
   still bound to iteration N, so it DROPS the new PLAY and then
   consumes and discards every ticket of N+1, end markers included
   (demonstrated: 7 tickets consumed, actor still blocked). Cost per
   event: an unbounded share of one iteration's games silently lost,
   other actors left outstanding, and the iteration running to the
   1800 s soft deadline with the GPU near-idle.
3. Serve threads leak on every UNNAMED error path of an iteration
   (`_stop_serving` is called at four named raise sites, not in the
   `finally`). 4. `ActorPool.shutdown()` never closes the ~2n+3
   queues -- each a pipe pair plus a feeder thread -- and calls
   `terminate()` with no `join()`. 5. A serve thread that dies OUTSIDE
   its `try` skips both the parked-request flush (the code that stops
   actors blocking forever) and its stats append, so throughput
   silently halves at the default `serve_threads=2`.

Still open (the eval path; the Elo files were being edited when the
audit landed):

- `WorkerPool` drops a killed worker WITHOUT closing it
  (`eval_workers.py:156` filters it out of the list and `close()` is
  never called), so its stderr file object stays open and its log stays
  on disk. One per timed-out game -- the leg-5 verdict saw 27 of 40
  games time out -- and the discarded log is the only record of why
  that worker died. DEMONSTRATED.
- `run_elo_batch.py:1163-1169` closes a child's stderr only `if
  _proc.poll() is None`, so a child that had already exited at
  teardown keeps its fd and leaves its log; `_peak_rss` is popped only
  on the normal-completion branch, so it grows by one per timed-out
  game.
- `az_loop` waits on its probe/profile children with no `timeout=` and
  no cleanup path, so a wedge hangs the leg and a kill of az_loop
  orphans a whole eval batch (with its own workers and servers) on the
  box. Also `gc.get_objects()` materialises a list referencing EVERY
  live object once per iteration, on a heap the surrounding comment
  says grows ~0.5M objects per iteration -- a full-heap scan on the
  learner's critical path, not a leak.
- `elo_eval_game._shared_client` replaces a broken client without
  closing it, so the server-side reader thread lives until GC.
- A game that raises inside a persistent eval worker leaves its
  `_pending` entries on the cached policy forever
  (`transformer_policy.py:298, 510-517`); `drop_pending` runs only on
  the normal game end.

Verified CLEAN, so nobody re-hunts them: no torch tensors cross the
pool's queues (numpy wire dicts and plain tuples), which closes the
2026-07-03 fd/`/dev/shm` leak class on that path; nothing in
`ActorPool` grows per iteration (every `last_*` field is rebound, not
appended); the eval inference server's lifecycle is sound (exits on
stdin EOF, which fires even on SIGKILL of the driver, and
`_shutdown_servers` is in the driver's `finally`); `turn_gap`'s pool
and server are both under context managers / `finally`;
`host_resources` reads every file under `with` and runs nvidia-smi
with a timeout; `game_core` creates no OS resources at all; and the
hot-path caches are all bounded (static hexes 64, Rust types 1024 with
the source pinned so `id()` keys cannot be recycled).

## Review of the day (2026-09-14, Fable)

What the audit of the 29 commits found and fixed is in the commit
messages (6e86474, a0f9979 and the one after); the standing
consequences: TF32 and fused AdamW are OPT-IN (`supervised_train
--tf32 --fused-adamw`, `az_loop --tf32`), each a one-factor change
with its own match; self-play games now carry a per-game combat-luck
salt (`pool:<label or seed>`), so a training run's games are
independent draws as eval games have been since 2026-09-13 (a
training-path numerics change; the verdict path is untouched); the
Rust core is at phase 10 (illuminated nightstalk, renamed cover
flags); the box run of 2026-09-14 (`scripts/postreview_box.sh`,
docs/box_specs.md "The post-review box run") certified both it and the
2026-09-13 changes on the corpus. The unrecorded readings of the day
are marked where they stand (docs/box_specs.md); the two that
mattered for a decision -- the two-server mean batch, the old-rule
hider sample -- were re-measured on that run with their records kept,
and the second serve process got its pool measurement: games per
dollar 0.91x, dead as a lever on its own.

## Open after the hide-cover review (2026-09-13)

Three independent adversarial reviewers checked the hide-cover root fix
and its certification. The CODE survived: the globs are transcribed
correctly (one reviewer ported the engine's `t_translation` matcher
from the 1.18.4 tag and diffed it over 19,738 codes and every cell of
the 114 tracked maps, 0 disagreements), the Rust core's baked flags are
read by nothing but the cover predicate, and no past Elo result is
invalidated. The WRITE-UP did not, and is corrected in
docs/box_specs.md and docs/wesnoth_rules.md. What stays open:

- **The certification is a no-regression test, not a proof of the
  rule.** Recorded 2026-09-14 (`tools/analysis/hider_rule_sample.py`,
  `training/metrics/bench_pipeline/postreview_20260914/hider_rule_sample.json`;
  the 2026-09-13 write-up's "164 of 300, 4 of 120" was the same
  measurement from a run with no record): of 300 sampled replays 150
  field a hider, 4 of those reconstruct differently (an ambush stop
  the old rule ran through, same landing hex, one more hider
  revealed), and `diff_replay` reports 0 divergences under the
  engine rule and 0 under the old one. The replay format carries no post-state,
  so every check asks whether the next recorded command's
  preconditions hold; nothing reads a move's stop REASON or the
  uncovered-unit set, which is the only state this change moves (all
  4 truncate at the same hex). A real check needs Wesnoth ground truth
  for a truncation, i.e. an `[mp_checkup]`-style oracle on a move an
  ambush stops -- the same shape as the combat parity we already have.
  `diff_core` cannot help: `game_core.map_static` bakes its flags from
  the same `hides_cover` that `visibility` calls, so the two sides
  cannot disagree by construction.
- **Post-fix matches are cross-build against pre-fix numbers.** The
  reference player's +56 +- 12 and seed2's +33 +- 12 were measured in
  a sim that hid units on a different set of hexes. They stay
  internally valid (both players in a match always ran the same
  predicate, and run_elo_batch alternates sides, so there was never a
  within-match asymmetry), but a new number must not be chained onto
  them without re-measuring. The relset self-pin is the natural place
  to re-establish the baseline.
- DONE 2026-09-14 (`replay_dataset.illuminated_lawful_bonus_at`, used by
  `visibility._hide_cover_active`, `build_attack_context` and the core's
  `hide_cover_active`; the combat arithmetic is unchanged, the nightstalk
  half is a behaviour change on a case no pool reaches). Was:
  **Nightstalk reads a ToD that omits unit illumination.** The engine
  evaluates `[hides]`'s `time_of_day=chaotic` on the ILLUMINATED ToD
  (`abilities.cpp`:447-451 sets `use_flat_tod` only for `illuminates`
  itself; `filter.cpp`:269-273 then calls
  `get_illuminated_time_of_day`), which adds terrain `light=` AND a
  scan of the hex plus its 6 neighbours for `illuminates` units
  (`tod_manager.cpp`:229, 237-262). `visibility._hide_cover_active`
  uses `_lawful_bonus_at`, which does the terrain half and skips the
  unit scan; the sim applies unit illumination only in combat
  (`abilities.illuminate_step` through `replay_dataset._apply_illum`).
  NOT reachable in the games we play: the illuminator must sit within
  one hex, only Mage of Light and Mermaid Diviner have the ability,
  and in the default era only Undead reaches nightstalk (Ghost ->
  Shadow) while only Loyalists and Rebels reach Mage of Light -- so
  the illuminator is always an ENEMY of the hider, and
  `_discovered_by_adjacency` reveals the unit anyway. It fires the
  moment a >2-side or team game, an allied illuminator, or an
  ability-granting event enters the pool. Root fix: one
  `illuminated_lawful_bonus_at()` = `_lawful_bonus_at` +
  `illuminate_step`'s bounded add, consumed by BOTH the hide predicate
  and combat, replacing combat's inline `_apply_illum`. That collapses
  two ToD readings into one; the arithmetic is unchanged, so the
  existing corpus sweep covers the combat half.
- **`burrow` is unmodelled** (the fifth `[hides]`,
  `abilities.cfg`:301-315, `terrain=*^F*,*^Qhhf,*^Qhuf,D*^*` plus a
  resting condition). Latent: no unit in the pinned 356-unit
  `unit_stats.json` carries it. Modelling it needs the "has not moved
  this turn" state the sim does not track.
- DONE 2026-09-14: the Rust core's three cover flags are named for
  what they are (`hides_ambush` / `hides_concealment` /
  `hides_submerge`, phase-10 wheel), and nightstalk's cover reads the
  illuminated time of day in both the Python predicate and the core
  (`replay_dataset.illuminated_lawful_bonus_at`, which combat also
  uses now: one reading of the time of day).

FIXED in the same pass: the observation half now has tests. The two
cover tests in tests/test_visibility.py used `Gg^Fp`, a code the OLD
defense-key table also covered, so they passed under the broken rule;
they now use `Gs^Fms`, which it missed, and a new
`test_encoder_omits_cover_hidden_enemy_tokens` asserts a
cover-hidden enemy loses its encoder token (the set the legality mask
reads) and gets it back once uncovered. The certification script counted divergences
with a regex matching a string `diff_replay` never prints, and listed
12 of 24 shards; it now parses the real string, prints every shard and
asserts `clean == total`. Pre-encoded corpora and the policy-anchor
cache now carry `constants.OBSERVATION_EPOCH` and refuse a cache built
under a different one -- the vocab, hex basis and fog gate all stay
identical when the sim's visibility rules move, so nothing else could
have caught a stale cache (tests/test_anchor_cache_gate.py).

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
