# BACKLOG sections closed by 2026-09-26 (verbatim)

Moved from BACKLOG.md on 2026-09-26 so that its front holds what is open.
Each section is as it stood on the day it moved.

## The spool self-play path is removed (2026-09-24, DONE, 0.5.8)

User ruling 2026-09-24. The spool workers (`--spool-workers`,
`tools/selfplay_worker.py`) were off by default since the 2026-08-10
topology ruling, and each looped until its control file changed,
whatever became of the learner. Removed with them: the VRAM-budgeted
device split and its demotion, `--prof` with `tools/prof_hooks.py` and
`prof_report.py`, `tools/profile_worker_split.py`, the spool ingest's
exit-6 basis tripwire, and the spool branch of `vast_onstart.sh`.
`box_bench.py` keeps the pool projection only.

## An encode worker whose trainer was killed exits (2026-09-24, FIXED, 0.5.7)

`supervised_train --workers N` spawns encode workers that waited on an
untimed `in_q.get()` and a bounded `out_q.put()` with no parent check,
holding both ends of both queues. When the trainer was killed (an OOM
kill, or `vast_onstart.sh`'s SL_MODE relaunch, whose pkill matches the
trainer's command line and not the workers') each worker waited
forever: shown on Windows here and on Linux in CI (run 36055626436) for
a worker waiting for a replay and one waiting for room on a full output
queue. `encode_worker.serve_files` now reads and writes in 2 s slices
and returns once the trainer is gone, and `_ParallelStream` starts its
workers through `start_child`, so the exit does not wait on unread
results (tests/test_orphan_exit.py). The per-file work, `encode_game`,
moved from preencode_corpus into encode_worker. New:
tests/test_parallel_stream_workers.py (slow tier), the first CI test
that spawns these workers, requires the in-process encoding through
two real workers with every encoder switch off its default.

## A pool child whose learner was killed exits (2026-09-24, FIXED, 0.5.6)

The 2026-09-13 orphan guard made an actor return once its learner was
killed, but its exit then waited for its queues' feeders to write out
what it had shipped, and nobody would read: the learner was dead, and
the pipes never break, since a spawned child holds their read ends
itself. An actor with more than a pipe's worth unread (one experience
is 9-142 KB) never exited. Measured before the fix: the actor returned
from its body and was still running 15 s later with 1 MiB shipped, on
Windows here and on Linux in CI (run 36051474864); with 1 KiB it exited
2.0 s after the kill. Actors and serve processes now start through
`tools/mp_teardown.start_child`, whose target cancels the exit's flush
on every queue the child was handed when its body ends with the parent
gone (tests/test_orphan_exit.py). A child whose body returns while
the parent lives (an actor on STOP) still writes everything out, since
the manager reads its queues during shutdown; if the parent dies before
that exit is done, a watch thread on the parent's sentinel ends the
process (0.5.9).

## shutdown() reads its children's output while they exit (2026-09-24, FIXED, 0.5.5)

A process that has put on an mp.Queue waits at exit until the queue's
feeder thread has written it all into the pipe (64 KiB on Linux, 8 KiB
on Windows), and one experience carries a whole game state (9 KB
pickled on a mini map, 47-142 KB on ladder maps). So once the loop
raised mid-iteration or mid-stream, shutdown() read nothing, every actor
with unread results ran out its 15 s join timeout and was terminated,
one after another (CI run 36022513684). The children now get one
deadline together while a daemon thread reads and discards the result
and server queues, logging any error or fatal report among them
(tools/mp_teardown.py). Measured on the laptop, three children each
holding 1 MiB unread, timeout 5 s: 15.66 s with all three terminated
before, 0.48 s with all three exiting on their own after.

## An ended iteration's leftovers stay out of the next session (2026-09-24, FIXED, 0.5.4)

An iteration that aborts (a serve process fails) or is abandoned at its
hard deadline leaves reports, a dead-server marker and tickets on the
queues its actors share, and the next session read them as its own:
test_serve_process's stream counted the aborted iteration's two games as
its first window (28 of 75 CI runs of the test failed that way). Per-game
reports and the marker now carry their session's tag, an ended
iteration clears its tickets, an actor keeps a later session's ticket
for that session's PLAY, and shutdown() stops open serving before
closing the queues.

## Vision follows the engine (2026-09-24, FIXED, 0.4.6)

A side sees its fog as the engine keeps it (docs/wesnoth_rules.md
"Vision and fog"; commits 0b7b3ea, 71f29c8, 73f4c1f). Open:
- **RESOLVED 2026-09-25: the reference learned from disc observations.**
  Its recipe retrained on the current observation (epoch 8, the time of
  day included) beat it +73 +- 13 Elo and is the reference `obs8`
  (docs/observation_retrain_prereg_20260924.md).
- **Not modelled:** `vision=` / `[vision_costs]` (four unit types, none
  in the default era; such a unit warns), jamming, shared vision, the
  delay-shroud preference some corpus players may have used, and
  sighted-move interrupts (exports carry `skip_sighted="all"`).

## The neutral side's turn ends through the applier (2026-09-24, FIXED, 0.5.1)

Side 3's end_turn was recorded and never applied (docs/wesnoth_rules.md
"End of a side's turn"). Open, from the engine reading, not observed in
a running game: a guardian with movement left probably gets a
"stay in place" move from the default AI's move-to-targets phase
(`ca_move_to_targets.cpp:269-277`), an unrecorded stop that zeroes its
movement (`unit.cpp:2784-2791`), so in live Wesnoth a guardian tentacle
likely shows 0 movement during the players' turns where replays and the
simulator show full movement. Healing is the same either way; the
encoder sees a different movement value.

## Every generated game is recorded (2026-09-24, SHIPPED, 0.5.0)

`tools/game_record.py`; records carry turn-start fingerprints, so a
rules change that makes a stored game rebuild differently is refused
instead of passing silently. Open:
- A mid-game record stores its corpus directory as the absolute path on
  the box; rebuilding elsewhere needs the corpus at that path.

## A revealed hider hides again at its turn start (2026-09-24, FIXED, 0.4.7)

Replay reconstruction never cleared STATE_UNCOVERED; the simulator did,
outside the command applier and on turn 1 too. One copy now, in
`_apply_command`'s init_side, gated like the engine's `turn() > 1`
(docs/wesnoth_rules.md "Hidden-unit visibility"). It changes what
reconstructed corpus positions show (804 of 66,873 decisions on 206
games), so the reference's imitation data is epoch 6 or older either
way; the retrain question of the vision entry above covers both.

## The network sees the time of day (2026-09-22; trained into `obs8` 2026-09-25)

Ran batched into the observation retrain: `obs8`, which sees it, beats
`terrain` +73 +- 13 Elo together with the vision, statue and re-hide
corrections; the time of day's own share is not measured. The record
below is the design as parked.

Built and tested on 2026-09-22, **not run**: the user's order is
"we're not retraining after every single bug", so the arm waits and is
batched with whatever else the scenario rework turns up.
`GLOBAL_FEAT_DIM` is 6 -> 8 (this turn's and next turn's lawful bonus
over `LAWFUL_BONUS_NORM` 25), mirrored in the Rust kernels with
`__phase__` 10 -> 11 (built and tested on CI 2026-09-23) and a gate
that refuses a stale wheel,
`OBSERVATION_EPOCH` 3 -> 4, seven tests in
`tests/test_time_of_day_features.py`, pre-registration in
docs/time_of_day_prereg_20260922.md.

The original finding, which stood until the time-of-day features
(`GLOBAL_FEAT_DIM` is 8 since 2026-09-22):

`encoder.GLOBAL_FEAT_DIM` is 6, and the six are turn number, side to
move, our gold, our income, our villages and theirs. No time of day,
no lawful bonus, anywhere in the global, unit or hex features. Combat
applies the bonus (`rust/wesnoth_core/src/combat.rs:136`,
`combat_modifier(alignment, lawful_bonus, fearless)`), so the network
plays a game where a lawful unit's damage swings by 50% between dawn
and midnight for reasons it cannot observe.

The turn number does not stand in for it. Two pool scenarios start at
second watch (Fallenstar Lake, Ruined Passage, `current_time=5`), four
minis roll `random_start_time=yes`, and Tombs of Kesorak and Elensefar
Courtyard carry `[time_area]` zones whose hexes run a different cycle
from the rest of the board. On those maps the same turn number means
different things, and inside a time area it means different things on
different hexes.

Shape of the work: a global feature for the current slot and its
lawful bonus, and a per-hex bonus so time areas are visible; then a
fresh arm on the reference's recipe and an 800-game match against the
reference, one factor, as the terrain-set arm was run
(docs/terrain_multi_hot_prereg_20260919.md is the template). It is a
checkpoint-flag change like `terrain_multi_hot`, so old checkpoints
keep observing what they observed.

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

Left open at the close, neither blocking phase 2, both needing a box
and a word from the user first (both resolved since):
- plan 1.3's 3,000-leaves-per-second-per-4090 target: MET
  2026-09-21 (user order). Every earlier reading (1,450-1,565 on
  2026-09-13, 2,126-2,269 and 2,634-2,914 on 2026-09-18) ran the
  pool under the az legs' 16-leaf serve batch cap, and "the roof is
  the GPU's execution time" was a reading of that cap: at 320 tokens
  per leaf 0.5 ms per leaf is 12% of the card's peak, the cost of
  400 small kernels per batch. With the cap at 64 (four requests
  coalesced) the same host that read 2,126-2,269 reads 3,222-3,237,
  1.34x in two interleaved pairs that repeat to 0.2%, games per
  dollar 1.09-1.17x; 96 adds nothing (the queue binds at 5 waiting
  requests); the graphed server at 64 falls back on 75% of batches
  past its 12,288-token bucket cap. 64 is the default in `az_loop`
  and `bench_pool` (docs/serve_batch_prereg_20260920.md,
  docs/box_specs.md "The serve batch cap"). The next generation
  lever is actors again, and a graphed-64 arm needs bigger buckets.
- a tight self-pin of the reference player (800 games, 18 min,
  $0.20), which would replace the +- 37 Elo above with +- 12.
  DONE 2026-09-19 as a rider of the end_turn box: 1,300 games,
  406-375 with 519 capped, p 0.520 +- 0.018 over 781 decisive, about
  +14 +- 13 Elo for side A, no asymmetry detected
  (docs/endturn_rule_prereg_20260919.md "Measured").

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
- SUPERSEDED 2026-09-21 (the target is met with the serve batch cap at
  64, above). Plan 1.3's 3,000-leaves-per-4090 target is NOT met: the
  real 4090 reads 1,450-1,565 saturated at ~320 tokens per leaf, which
  LOOKS like docs/gpu_forward_design_20260904.md's 1,300-1,800 band but
  is NOT a confirmation of it: that band was computed at 1,270 tokens
  per leaf and this is measured at ~320, where the same doc's
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
- the trainer is GPU-bound: bf16 autocast is built (`--bf16`, off by
  default) and timed on a 24 GB card at 1.25x with an equivalent loss
  (2026-09-13, above); TF32 for the trunk is untried (`--tf32`, opt-in
  since 2026-09-14).
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
  and the Wesnoth rule layer) reported eleven more, none fixed that
  day; each entry says what became of it. In order of what they
  corrupt:
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
  * FIXED 2026-09-13 (each edge records its (side, seed) game slots,
    and a collect that replays another edge's games is refused; the two
    `ref~old` edges turned out to carry disjoint seeds). Was:
    **the Elo catalog sums repeat measurements of one pair as
    independent evidence** (tools/elo_catalog.py:360; the edge key is
    the games-dir name and nothing compares seeds). Both generators
    default to a fixed seed base, and at raw:t0 a rerun is
    deterministic, so a re-pin into a fresh outdir doubles n and
    shrinks the standard error on no new information. Already live in
    the committed catalog: two edges for `ref~old` pool to n=240.
  * FIXED 2026-09-13 (the estimands travel on the edge as
    `protocol["estimands"]` and two different ones cannot pool into one
    fit). Was: basis, precision and batch estimands are dropped at the
    catalog boundary (tools/elo_collect.py:278): guarded three times inside a
    dir, lost between dirs, so a relevant-set edge and a full-board
    edge can be pooled into one fit with no warning. `value_center`
    and `ELO_MOVES_LEFT_UTILITY` change the searched player and reach
    no result field at all (tools/elo_eval_game.py:352).
  * FIXED 2026-09-13 (each eval game salts its own stream, recorded as
    `combat_stream=per_game`; self-play games since 2026-09-14). Was:
    every eval game shares one combat-luck stream
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

## Phase 1's actions, as recorded (phase 1 closed 2026-09-12)

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
     lever. DONE: `--packed-trunk` defaults on in az_loop (the
     compiled loop stays opt-in, its row below).
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
     DONE since: the saturated window is the pool's standing column;
     staged priors, the packed trunk and the packed embed shipped and
     are the defaults; four serve threads and a second serve process
     measured as no lever (docs/box_specs.md "The post-review box
     run"); the compiled forward stays opt-in; the PID limit is read
     from the cgroup, not guessed.
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
   the server before quoting gates through it (done: `relset`'s
   self-pins through the shared path, 2026-09-12 and 2026-09-19).
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

## Scenario building, from the ground up (2026-09-22, DONE 2026-09-23)

**docs/scenario_build_plan_20260922.md.** One scenario builder whose
output is checked against the game, with two automatic detectors
first: an exhaustive classification of the expanded WML that fails on
anything unprocessed, and a scenario-init oracle against real Wesnoth
that catches any discrepancy. User ruling: stop patching defects one
by one; make it correct from the ground up and detect the rest
automatically.

Work items, all done by 2026-09-23 (the plan's record says how): W0
diff our expansion against the game's as a standing test; W1 the
failing default; W2 repair and regenerate the templates; W3 the
classification manifest, bound to readers; W4 the engine oracle for
scenario init (`tools/scenario_init_oracle.py`); W5 one expansion
source for generation; W6 the assumptions that become reads. Our own
preprocessor is deferred behind a trigger.

Two independent reviews found seven factual errors between them, all
corrected in rev 3 and recorded there rather than dropped. The sharpest:
`{DEFAULT_SCHEDULE}` is not a recursion failure but an entry on
`_COSMETIC_MACROS` that deletes it, so rev 1's argument for writing our
own preprocessor was itself an instance of the bug class; the committed
templates are the preprocessor's output plus our builder's injections,
so a generation path reading them would read our own 70 as the
scenario's; and the quick-leader rule is already modelled at
tools/traits.py:275-285.

The second review also found a live defect that 716a1c3 CREATED and
be00037 half-fixed: `sim_to_replay`'s from-scratch export still wrote
PvPDefaults' village gold, so a mini game the sim now plays at 3
exported a save declaring 2. Fixed with a test the same day.

Measured while planning: the Wesnoth preprocessor costs 60-90 ms per
scenario, about 2 s for the pool, so speed is not a reason to
reimplement it; our expander deletes `{DEFAULT_SCHEDULE}` outright,
emitting zero `[time]` blocks where the game emits six, although every
macro body is in our cache; the committed templates are self-contained
(544 KB, map inlined for all 28).

## The scenario's economy is read from the scenario (2026-09-21, FIXED)

Found while sizing the corpus for the fine-tune arm, by asking why no
replay had ever diverged on gold or on levelling although 348 corpus
games ran a non-70 experience modifier and 444 a non-2 village gold.
The answer: the RECONSTRUCTION path reads both from the replay record
(`replay_extract` writes `village_income`, `village_support`,
`base_income`, `gold`, `experience_modifier` and `tod_start_index`;
`_build_initial_gamestate` reads them), so the faithfulness pipeline
was never at fault. The POOL path was: `build_scenario_gamestate`
hardcoded 2 gold per village, 1 free upkeep and a 70% experience
modifier, and patched them onto `global_info` AFTER the shared
builder had run. One rule, two copies -- the mirror class the
2026-09-14 audit named -- and the rule itself had been pinned in
docs/wesnoth_rules.md since May, when the same hardcode caused a
replay divergence on Den of Onis and was fixed on the replay side
only.

Scope, measured before the fix:
- **No ladder map moves, so no Elo moves.** None of the 21 whitelist
  scenarios declares an experience modifier; two (Clearing Gushes,
  The Walls of Pyrennis) declare `mp_village_gold=2`, which is the
  multiplayer default the pool already used. A test pins all 21.
- **Five of the seven mini scenarios declare `village_gold=3`**
  (2p_mini, 2p_mini_edited, Modified_Tiny_Close_Relation, both
  fallenstars), so every mini self-play game and every mini-based
  test paid a third less village income than its map specifies.
  Mini self-play games after this commit are not comparable with
  those before it; ladder matches are untouched.

The fix: `scenario_pool.scenario_economy` reads the per-side
`village_gold` / `village_support` and the scenario-level
`mp_village_gold` / `mp_village_support` (the game-creation spelling
mainline maps use), plus `experience_modifier` in the `"70%"` form the
add-on scenarios write; `build_scenario_gamestate` takes None for
each as "the scenario's value, else the multiplayer default" exactly
as `starting_gold` has since 2026-07-21, and hands them to
`_build_initial_gamestate` in the same dict fields a replay record
carries, so the two paths now share one code path instead of two
copies of the rule. `sim_self_play` stops mapping `PvPDefaults` onto
them for scenario games, under the same ruling; `PvPDefaults` still
governs the midgame-splice path, which has no scenario to read.
tests/test_scenario_economy.py: 9 tests, four of which fail with the
scenario read monkeypatched back to the old behaviour.

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
rental rather than renting for it alone. RAN 2026-09-14 on the
post-review box: 17,039 of 17,039 replays clean, the Tentacle's
`magical` and submerge included (docs/box_specs.md "The post-review box
run").

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
  **SHIPPED AS CODE 2026-09-19 (autonomous window), the arm
  pre-registered.** `terrain_resolver.terrain_members` derives each
  code's terrain SET from the database's aliases (a forested hill is
  HILLS and FOREST, a ford FLAT and SHALLOWWATER; an engine-defined
  full code such as `Mm^Xm` keeps its own list; `_off^_usr` and
  `^_fme` are impassable because no movetype prices them);
  `Hex.terrain_mask` carries it from the map parse, the scenario
  morphs and the live converter; behind the checkpoint flag
  `terrain_multi_hot` (on for a fresh network, absent = the one-class
  view, so `relset` observes bit for bit what it did) the encoder
  embeds the mask as a multi-hot over the same table
  (`GameStateEncoder.terrain_tokens`). The flag rides the pre-encoder
  fingerprint, the struct flags, the blueprint, the server hello and
  the pool's PLAY tuple, and eval records carry `terrain_a/terrain_b`
  as an estimand next to the basis
  (tests/test_terrain_multi_hot.py, tests/test_eval_terrain_provenance.py).
  Census on the mask: 0 of 1,572 forest-overlay hexes without the
  forest bit (`hide_cover_20260913/census_terrain_mask_20260919.json`).
  GameCore's own encode wrapper refuses the flag (its map bakes one id
  per hex); the default kernels take the mask from Python's static
  arrays and the Rust parity test is parametrized over it. The arm
  and its bars: docs/terrain_multi_hot_prereg_20260919.md,
  `scripts/terrain_arm_box.sh` (the reference's recipe with the flag,
  one pass, 800 decisive against `relset`, about $2.5-3.5).
  **RAN THE SAME EVENING AND PASSED: +44 +- 12 Elo over `relset`
  (800 decisive, 450-350, 339 of 1,139 capped), holdout proxies
  equal to the twin's** (docs/terrain_multi_hot_prereg_20260919.md
  "Measured"; checkpoint HF `tier-b/terrain_multi_hot_20260919/arm_epoch0.pt`).
  Under the pre-registered reading the arm is the candidate
  reference player, pending the user's ruling and a self-pin. **The
  composition RAN the same night** (docs/composed_levers_prereg_20260919.md):
  the arm under the -1.5 decode beats the reference under the same
  decode +26 +- 12 Elo (430-370), and the composed player beats
  today's reference +263 +- 16 (656-144, p 0.820, 4% capped). The
  arm's self-pin reads +2 +- 12 Elo (402-398 over 800 decisive, 22%
  capped against the reference's own 40%): no asymmetry, and the
  candidate is pinned.
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

