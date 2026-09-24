# CLAUDE.md — Wesnoth AI

## Project

A reinforcement-learning AI for *Battle for Wesnoth*, trained by self-play,
aiming to be (a) competitively strong against players and (b) readable
enough that humans can study its strategies. Customization is a first-class
goal: we need to easily incentivize unorthodox strategies, fine-tune on
human game logs, force specific openers, etc. — so code that gates behavior
behind config is preferred to code that gates behavior behind weights.

- **Language:** Python 3.11+
- **ML:** PyTorch
- **Game:** Wesnoth 1.18.x (Steam install on Windows)
- **Run:** training via `python tools/sim_self_play.py`; demos via
  `python tools/sim_demo_game.py`; live-Wesnoth setup checks via
  `python main.py --check-setup`
- **Test:** `pytest` (tests are Python-only; they exercise the WML
  parser and Lua-file generation with synthetic data — they do NOT spin
  up real Wesnoth). CI runs the full suite with a freshly built Rust
  wheel on every push (see Testing).

### Wesnoth data provenance (updated 2026-06-12)

`wesnoth_src/data/` is a WML-only copy (cfg/lua/map, no art) of the
LOCAL STEAM INSTALL's data tree, taken from **1.18.7**. The install
reports 1.18.8 (2026-09-23), and the files the simulator reads from it
(multiplayer factions, scenarios and maps, core macros and units, the
eras) are identical to this copy, compared that day. Refreshed via:

    robocopy "C:\Program Files (x86)\Steam\steamapps\common\wesnoth\data" wesnoth_src\data *.cfg *.lua *.map /S

It serves the RUNTIME readers (scenario_pool factions/eras,
scenario_events, map files). It is NOT a git checkout anymore and
has no `src/` tree; for engine-internals research, read the GitHub
**1.18.4 tag** directly (as docs/wesnoth_rules.md entries do).

**The sim's runtime WML subset IS tracked in git (since 2026-07-02):**
`data/multiplayer/{factions,scenarios,maps}`, `data/core/macros` (the
31 files the scenario expander reads), and the add-ons
`Mini_Maps_Collection`, `Seamless_Map_Picker` and `WL_Mappack` -- 276
files, counted 2026-09-23 -- so a bare `git clone` can run
self-play training on a GPU node with no Wesnoth install. The rest
of `wesnoth_src/` stays untracked. A robocopy refresh from Steam
will show these as git diffs; that's intentional (data drift
becomes visible in review instead of silent).

**`unit_stats.json` / `terrain_db.json` are pinned 1.18.4 scrapes
and are COMMITTED. Never re-scrape them from this wesnoth_src.**
Unit stats DRIFT between releases — e.g., in 1.19.x the Ghoul
gained a `[resistance] pierce=90` override that doesn't exist in
1.18.4 (where pierce inherits 70 from the gruefoot movement_type);
using drifted stats made combat reconstruction overdamage units.
The sim's bit-exact combat parity and the existing checkpoints
assume the 1.18.4 stats. Re-scraping requires fetching the 1.18.4
tag from GitHub first.

Most replays in `replays_raw/` are from 1.18.x clients; pin
accordingly. If a replay's `[scenario] version=` says something
other than 1.18.x, scrape from that version's tag instead.

## Current status (2026-09-04, entries through 2026-09-23)

**Read `docs/plan_20260904.md` first; `BACKLOG.md` holds the next
actions in order.** Superseded status blocks, plans, leg records and
mechanism specs are in `docs/archive/` (index in its README); the 78
quarantined training mechanisms are in `quarantine/INVENTORY.md`.

State of play:
- **The reference player (user ruling 2026-09-20) is `terrain` at
  `raw:t0+eo-1.5`:** the terrain-set arm (HF
  `tier-b/terrain_multi_hot_20260919/arm_epoch0.pt`, local
  `training/checkpoints/terrain.pt`; relevant-set basis, fog gate on,
  terrain set on) decoded with the end_turn logit offset -1.5, both
  pinned in `configs/reference_player.json`. It is +263 +- 16 Elo
  over the previous reference (docs/composed_levers_prereg_20260919.md)
  and self-pinned at +2 +- 12 at `raw:t0`. Before it (2026-09-11 to
  2026-09-20) the reference was `relset` at temperature 0: the
  relevant-set twin of seed2 at one pass, HF
  `tier-b/seed2_relset_20260911/arm_epoch0.pt`, local
  `training/checkpoints/relset.pt`, relevant-set basis, fog gate on;
  +56 +- 12 Elo over seed2's one-pass checkpoint (800 decisive
  games), a number that carries seed2's dropped large boards (see
  below). Its self-pin through the shared inference path RAN
  2026-09-12: -57 +- 37 Elo over 160 games, no asymmetry detected
  (`training/metrics/elo/relset_selfpin_20260912/`, the tally only:
  the game records stayed on the box). The +- 37 overstates what was
  measured: the 160 games are four arms replaying ONE set of 40 seeds
  on the pre-2026-09-13 shared luck stream, so the arms are not
  independent draws. **The tight self-pin ran 2026-09-19** (a rider
  of the end_turn box, per-game luck, the current hide-cover rule,
  the process-independent unit hash): 1,300 games, 406-375 with 519
  capped, p 0.520 +- 0.018 over 781 decisive, about +14 +- 13 Elo
  for side A, no asymmetry detected
  (docs/endturn_rule_prereg_20260919.md "Measured").
  Before it the reference was seed2 (+33 +- 12 over the original
  imitation seed), before that the seed itself
  (`tier-b/a3/seed_imit_tierb_start.pt`). Nothing produced by
  self-play has beaten any of them.
- Why (2026-09-04 review, docs/raw_argmax_control_20260904.md): the
  seed at argmax beats the seed sampling 37-3 (+412 ± 97), and the
  seed with Gumbel-MCTS-32 loses to the seed at argmax 13-27
  (−124 ± 58). Every earlier "search improves the seed" number used
  the sampling player as its reference. A 32-evaluation search over
  ~350 atomic actions looks 1-2 actions ahead, its targets are the
  prior or a truncation of it, and the value gradient owned 94-99% of
  every update through the shared trunk. No self-play leg ever had a
  teacher better than the prior.
- The 20-agent design panel (docs/selfplay_redesign_20260904.md)
  found no affordable outcome-graded teacher at this budget; the plan
  therefore starts with engineering (10-30x cheaper games, certified
  bit-exact) and then turn-level search built directly against
  `raw:t0` with 800-game gates.
- 2026-09-04/05 engineering days (details in BACKLOG.md and
  docs/box_specs.md): pool generation 141 -> 833 leaves/s saturated
  (server priors, bf16, staged priors, packed varlen trunk, packed
  embed; all on by default in az_loop); the serve threads' host work
  (~33 ms per 16-leaf batch) was the ceiling then; a second serve
  process is implemented and, measured 2026-09-14, buys no games per
  dollar (0.91x; an iteration ends with its longest game,
  docs/box_specs.md "The post-review box run"); training path
  684 -> 352 s per iteration (batched policy loss, batch 16, bf16) on
  a bench whose experiences carried no masks -- on production's, which
  ship the actor's masks since 2026-09-05, the step is 2.22 ms per
  experience and the training path 4% of an iteration (measured
  2026-09-14); eval 4.2x through persistent workers, 1.5x more through the
  shared inference server. Measured: raw:t0 against itself stalls
  (17 of 40 at the cap) while raw:t0.5 scores 22-18 with none; at
  argmax the leg-4 product equals the seed (+17 +- 55) and the 5M
  2291k equals the 15M seed (+9 +- 55). Phase-2 prerequisite
  (docs/turn_gap_prereg_20260904.md): large turn-level gaps exist
  but are sparse (3 of 60 positions confirmed out of sample, mean
  gain of the best sampled turn +0.05 per turn). Relevant-set
  two-arm retrain (docs/model_cost_study_20260905.md 7, measured
  2026-09-05 night): half an epoch more of the seed's own imitation
  recipe costs 111 +- 13 Elo at argmax with an equal holdout CE;
  the relevant-set arm is +26 +- 18 over that control and -35 +- 13
  against the seed. Holdout CE is not a strength proxy; every
  retrained checkpoint needs its own 800-game match. A lr 1e-5
  control is queued (7b); distillation from the seed into the new
  basis is designed (7c). Phase-2 tooling shipped the same night:
  turn_gap shared inference, sequential grading, replayed
  confirmations, the pre-grader analysis. 2026-09-07 (user's plan,
  docs/value_head_study_20260907.md): the seed's value head reads
  who is ahead in human games better than material in the early and
  middle game (same-turn AUC 0.65 -> 0.93 by phase against material
  0.56 -> 0.94); the corpus now records fog/shroud per side (18.9%
  of games were fog-off) and the encoder's fog switch follows it.
  2026-09-08 (docs/data_contamination_20260908.md): a contamination
  review found the seed's lineage trained on 108 of the 369
  imitation-holdout games and its value head on about 364 of their
  outcomes, and global feature 5 god-view under fog; every tool now
  splits by the corpus manifest, matches are deduplicated (17,019
  games) and feature 5 is gated by `fog_hides_enemy_villages`: on
  for every fresh network, a checkpoint's own setting on load, and
  pre-encoded caches refuse the other gate. Holdout CE of the seed's
  lineage carries that caveat; match results do not. One more
  iteration of the recipe (value_head_plus_1) sharpens the head's
  Brier in every phase and leaves its ranking unchanged (paired per
  game, tools/analysis/value_head_compare.py); the material input
  (value_head_plus_material) adds +0.01 to +0.03 same-turn AUC in
  every phase, within noise at 369 games. 2026-09-10: the clean seed
  (15M from scratch, the seed's recipe on the fixed corpus, two
  epochs, HF `tier-b/clean_seed_20260909/arm_epoch2.pt`) reads
  holdout CE 2.78 against the seed's 3.10 and a value head that ranks
  like the seed's and is better calibrated. 2026-09-11: seed2 beats
  the seed +33 +- 12 Elo (800 decisive raw:t0 games, 660 more at the
  cap), the first checkpoint to do so; it is the reference for holdout
  numbers, the frozen-trunk arm, and, after its self-pin, the
  reference player. The observation kernel (port plan 2c, on by
  default) makes a lone eval game 1.2x faster and leaves the 40-game
  match unchanged: that path is bound by the server's per-batch cycle
  (about 25 ms, mostly a fixed GPU launch cost, 7.5 of 20 workers per
  batch), not by worker CPU (docs/box_specs.md "The observation kernel
  and the CPU budget"). The relevant-set twin of seed2 (same recipe
  from scratch, 2.7x fewer tokens per leaf, one pass) beats seed2's
  own one-pass checkpoint +56 +- 12 Elo (800 decisive games), with
  13% more pairs in its pass (docs/box_specs.md "The relevant-set
  twin of seed2 at one pass"). Found the same night: the full-board
  imitation trainer at batch 64 on a 24 GB card dropped the largest
  boards' batches on CUDA out-of-memory with a DEBUG line, 12% of
  seed2's pairs; the trainer now splits and accumulates such batches
  and runs 1.86x faster (the batch embedded and scored at once,
  docs/box_specs.md "Pair census", "The imitation trainer timed").
  2026-09-12 (user order: complete the Rust port, then bf16 training):
  the relevant-set basis runs on the Rust kernels (reach rows, the
  relevant set, subset streams and masks; the reference player's
  lone game 27.0 -> 14.5 s) and combat resolves in Rust (fuzz,
  fixture and 17,039-replay sweep identical). Later that day the
  Rust-owned state (`GameCore`, `wesnoth_ai/game_core.py`) applies
  init_side, end_turn, move, attack and recruit itself and serves the
  observation and the encoding from its own records: 17,039 of 17,039
  corpus replays compare clean after every command against the
  Python applier (tools/diff_core.py); `WesnothSim(use_core=True)`
  runs on it behind one Python view (docs/rust_port_plan.md 3b/4).
  Timed the same day and left DEFAULT OFF: per call it is 2.6-7.3x
  cheaper (fork 0.019, step 0.050, encode 0.200 ms), but it moves
  neither the eval path (49 s against 49 s for a 40-game match,
  which waits on the inference server for four fifths of its wall)
  nor the pool (two runs just below the Python runs, within their 5%
  spread).
  **Phase 1 is CLOSED (2026-09-12).** Its exit criterion -- 10x more
  searched games per dollar at a fixed search budget -- is met at
  16x, measured same-box as 64.8 -> about 1,050 saturated leaf
  evaluations per second (6.8x the committed configuration, 2.4x the
  relevant-set basis), docs/box_specs.md "Phase 1's exit". The
  reference player's self-pin over 160 games reads -57 +- 37 Elo (no
  asymmetry detected; four replays of one 40-seed set, see above).
  Two optional confirmations need a box and a word: the
  3,000-per-4090 target of plan 1.3 (an A4000 cannot judge it) and a
  tight 800-game self-pin.
- 2026-09-13 (user order: keep optimizing throughput): **the pool's
  constraint was the actor COUNT.** An actor blocks on the inference
  server for nine tenths of its cycle -- its own Python is about 3 ms
  of a 27-58 ms per-leaf wall -- so the count buys in-flight leaves,
  not CPU: 19 -> 665, 32 -> 914, 48 -> 1,006, 64 -> 1,116 leaf
  evaluations per second against a server saturating near 1,500
  (docs/box_specs.md "Actors buy in-flight leaves"). `az_loop
  --actors` was 8, then 32, then 24 within the day; it is **0 = auto** since 2026-09-13 night
  (as many as `--games-per-iter`, the box's pids limit and, since
  2026-09-14, its memory allow: `host_resources.max_actors`). Also measured that day: plan 1.3's
  3,000-per-4090 target is NOT met on a real 4090 (1,450-1,565
  saturated; docs/gpu_forward_design_20260904.md's 1,300-1,800 band
  coincides numerically but priced 1,270 tokens per leaf against
  this run's ~320, so it is not a confirmation); the
  eval path wants neither more workers nor more servers (a second
  server halves the mean batch per server, 6.9-8.2 -> 3.5-4.4 recorded
  2026-09-14 with the walls overlapping, so it cannot win, refuting the
  standing 1.3-1.5x expectation); and bf16 on the imitation trainer
  is 1.25x with an equivalent loss on a 24 GB card, not the 2.9x a
  memory-starved 16 GB card suggested. Five bugs were found and fixed
  with tests (BACKLOG.md), the sharpest being an out-of-memory inside
  backward() whose retry double-counted the gradients the failed pass
  had already accumulated.
- 2026-09-13 night (user order: fix the hiding bug at the root): the
  sim decided terrain cover for `ambush` / `concealment` / `submerge`
  from a hand-rolled table's DEFENSE keys; the engine matches the
  hex's terrain CODE against the `[hides]` globs (`*^F*`, `*^V*`,
  `Wo*^*`). `terrain_resolver.hides_cover` now transcribes those
  globs and is the single source for the Python predicate and the
  Rust core's baked flags. Over the Ladder pool ambush gained 478
  hexes, submerge 153 (tropical deep water, all on Ruphus Isle) and
  concealment 93 while LOSING 302 to the farmland correction (`^Gvs`
  is Farmland, not a village). 17,039 of 17,039 replays reconstruct
  (the box's staging set; the current corpus is 17,019, a strict
  subset)
  clean. **That sweep is a no-regression test, not a proof of the
  rule** -- an adversarial review of three independent lenses showed
  it passes under the OLD rule too, since the replay format carries no
  post-state and nothing reads a move's stop reason or the
  uncovered-unit set, which is the only state this change moves
  (recorded 2026-09-14 by `tools/analysis/hider_rule_sample.py`:
  of 300 sampled replays 150 field a hider, 4 of those reconstruct
  differently -- an ambush stop the old rule ran through, same landing
  hex -- and `diff_replay` reports 0 divergences under the engine
  rule and 0 under the old one). The rule itself is established by the
  engine's macro text and pinned by tests/test_hide_cover.py;
  tests/test_visibility.py pins the observation and move-truncation
  halves on a code the old table missed. **Verified against real
  Wesnoth 2026-09-20** (user order): `tools/hidden_units_oracle.py`,
  54 of 54 scripted positions agree with the engine on what side 1
  sees and where a move stops (docs/wesnoth_rules.md "Verified against
  the engine"); the live bridge's collector leaked hiders on unfogged
  hexes and always reported "morning", both fixed the same day.
  **Consequence for numbers:** matches run after this change are
  CROSS-BUILD against every Elo measured before it. The old numbers
  stay internally valid (both players in a match always ran the same
  predicate) but nothing may be chained onto them without
  re-measuring; the reference player's self-pin is where to
  re-establish the baseline. Pre-encoded corpora and anchor caches
  now carry `constants.OBSERVATION_EPOCH` and refuse an older one.
  The review's open items are in BACKLOG.md "Open after the
  hide-cover review".
  A hunt for SIBLINGS of that bug (a Wesnoth rule decided by a
  hand-rolled enumeration with a silent wrong default) found two more,
  both live on 2p Silverhead Crossing -- a Ladder map, 351 corpus
  games. Its prestart `[object]` grants its Tentacle `{ABILITY_SUBMERGE}`
  and a `{WEAPON_SPECIAL_MAGICAL}` evil eye; we dropped BOTH, because
  `[effect]` members were keyed by their TAG (`hides`, `chance_to_hit`)
  instead of their `id=` (`submerge`, `magical`), and
  `apply_to=new_ability` had no branch at all. So the Tentacle was
  visible where Wesnoth hides it, and counter-attacked at the
  attacker's terrain chance-to-hit instead of the flat 70% that
  `magical` sets. Fixed at
  the root with tests (tests/test_effect_ids.py); an unmodelled
  `apply_to` now warns instead of vanishing. The combat half owes the
  corpus sweep on the next box (BACKLOG.md "Scenario [effect] members
  are named by id="). Two further findings there are NOT fixed: the
  encoder's terrain one-hot labels 1356 of the 1,572 forest-overlay
  Ladder hexes (86%) as something other than forest
  (`tools/analysis/hide_cover_census.py`; model input, no rule reads
  it, so it wants its own arm and an 800-game match), and `burrow` /
  `swamp_lurk` are unmodelled `[hides]` abilities whose carriers (the
  Horned Scarab, whose burrow the pinned scrape dropped; the Swamp
  Lizard) appear in neither pool.
  The same night closed the rest of the eleven-bug ledger and ran three
  more audits. **On a box 2026-09-14 (`scripts/postreview_box.sh`,
  docs/box_specs.md "The post-review box run"): the whole corpus
  reconstructs clean on the current tree (17,039 of 17,039, the
  Silverhead `magical` and submerge included), 600 replays clean
  through the phase-10 core, 117 tests on both states of record.**
  - **Elo accounting.** Eval games all replayed ONE combat-luck stream
    (`_next_seed` with no salt is a pure function of a counter that
    restarts per sim), so an 800-game match was 800 draws against one
    luck vector and the standard error assumed an independence it did
    not have. Now salted per game. Estimands (basis, precision, batch,
    shared inference, packed trunk, value_center, moves-left utility,
    and the luck regime) were guarded inside a games dir and compared
    nowhere BETWEEN dirs; they now travel on the edge. Each edge
    records its (side, seed) slots, so a deterministic rerun cannot be
    pooled as new evidence -- though the duplicate an audit alleged
    turned out not to exist, the two `ref~old` edges carry disjoint
    seeds. And the 2026-07 anchor chain was labelled `mcts:32` when
    both source records say "raw policy (no MCTS)"; corrected, ratings
    unchanged.
  - **Lifecycle.** Actors had no parent-liveness check, so a killed
    learner orphaned all of them -- they inherit both ends of their
    control queue, so the blocking read never sees EOF -- and orphans
    hold the cgroup pids budget, which is what produced 0 leaves/s on
    2026-09-04. At the hard deadline a straggler actor silently
    consumed and discarded every ticket of the NEXT iteration. Both
    demonstrated. `WorkerPool` dropped a killed worker without closing
    it, one fd and one stale log per timed-out game.
  - **Mirrors.** `state_converter` kept a second terrain table under a
    "keep in sync" comment and it had drifted: `Uu` read UNWALKABLE
    there and CAVE in the sim, eleven codes were missing. Single-sourced
    with a test. The launcher bannered "RUST (wesnoth_core)" off
    `rust_active()`, which is importability, not capability; measured,
    four of five kernels were on Python. `tools/kernel_status.py` now
    reports each kernel through production's own gate.
  - **Throughput.** `az_loop --actors` is 0 = auto, clamped by the pids
    limit `host_resources` now READS instead of guessing; the old
    conservative default cost throughput on every box that could take
    more. `auto_jobs` got the same treatment. Rejected and recorded: a
    lazy `pos_to_hex` (measured 4% of one trainer, not worth the hot
    legality path).
  - **Review 2026-09-14 (Fable).** The day's numbers with records
    match them to the digit; the eval sweep's server counters, the
    old-rule hider sample and "80% GPU" had no record and are marked
    so; the hex census and the hider sample now have tools
    (`tools/analysis/hide_cover_census.py`, `hider_rule_sample.py`)
    and records, and the eval counters were re-measured with their
    stats files kept. Fixed in
    code: the trainer's out-of-memory retry kept the failed
    attempt's activations alive; a resumed optimizer silently took
    the checkpoint's step kernel; TF32 and fused AdamW had landed as
    default recipe changes (both opt-in now); `tools/preencode_corpus.py`
    could not start as a script; the human anchor and the Elo reuse
    guards ignored the observation epoch; the coalesced embedding
    buffer left its fields unaligned; nightstalk's cover ignored
    unit illumination; preplaced `[unit][abilities]` were still keyed
    by tag; a refused search action was worth 0 rather than its
    parent's value; every SELF-PLAY game replayed one unsalted luck
    stream (the same defect eval lost on 2026-09-13) and now carries
    its own; an orphaned actor walked its queued tickets before
    exiting; the auto actor count is bound by memory too.
  - **Provenance.** Caches, and now checkpoints, carry
    `constants.OBSERVATION_EPOCH`; the local suite prints a banner when
    the Rust wheel is behind the source, because it is (phase 3 against
    9) and every Rust test was skipping silently.
- 2026-09-14 (user order: every ounce of performance): **the
  inference server was launch-bound**, and the trainer is not a lever.
  One 16-leaf serve batch launched 406 kernels for 3.0 ms of device
  time inside 8.6 ms of host wall; the trainer on production's
  experiences (masks shipped since 2026-09-05) costs 2.2 ms per
  experience, 4% of an iteration -- the 35 ms on record was a bench
  rebuilding masks its experiences lacked. `wesnoth_ai/graphed_serve.py`
  serves a priors batch from one CUDA graph per static bucket (the
  embed, the packed trunk over bf16 weights, the heads and the priors
  chain, the copies in and out), behind `--graphed-serve` (az_loop,
  bench_pool, run_elo_batch) and `--graphed` (eval_inference_server),
  default OFF until its box rows are the defaults' own: on the eval
  path the server's infer time per batch went 27.0 -> 12.7 ms and the
  40-game walls 1.2-1.4x (the workers' serial chain is the bound now);
  on the pool the saturated rate moved 1.30x, the iteration rate
  1.15x and games per dollar 1.04x (docs/box_specs.md "The
  serve batch is launch-bound"). Both compare a bf16 server to a
  bf16 server: same weights and math, priors within bf16 noise, so
  neither is a cross-build for Elo.
- 2026-09-18 (user authorized the repeat): **the graphed server's gain
  belongs to slow hosts; both defaults stay OFF.** On a single-tenant
  Core Ultra 9 285K box (24 cores, whole CPU, eager pool arms
  repeating within 3.5% on the saturated column) the pool pair read
  saturated 1.05x and 1.07x, iteration 0.99x and 1.03x, games per
  dollar 0.95x and 0.98x, and the eval batch 1.14-1.16x SLOWER (the
  coarse eval buckets cost 27-40% more device time and 30 captures
  2.5 s of a 22 s match), under a rule written before the box was
  rented (`scripts/graphed_default_box.sh`). The eager server's host
  cost per pool batch is 8 ms on that host against 20 ms on the
  2026-09-14 shared host, and the device span already covers 95% of
  its infer time, so the graphs have only the launch gaps left to
  remove. The flag stays for slow hosts: an eager `host ms per batch`
  sum well above 12 ms on the first iteration says the graphed pool
  server pays (docs/box_specs.md "The graphed server on a quiet host").
  The same host plays a 40-game raw:t0 match in 24-25 s and runs the
  eager pool 1.5x faster than the 2026-09-14 boxes.
- Rulings (2026-09-05): no optimizations conditioned on the MCTS
  loop; scope every box test, train sparingly; results are written
  on the run, never as atomic dumps; a box job past ~1.5x its
  estimate gets inspected and cut.
- 2026-09-18 (user order: build continuous generation): **the
  iteration's tail is the largest generation lever left, and
  `tools/actor_stream.py` removes it.** An iteration ends with its
  longest game (median 200 s, longest 300-430 s), so the server
  starves for its second half: the iteration rate sits 1.37x, 1.6x,
  1.84x and 1.93x below the saturated rate on the three hosts
  measured. A stream keeps the game queue topped up, the learner
  collects windows of `--games-per-iter` completed games and
  publishes weights into the running servers, every load under the
  server's `ServeGate` (inference_seam) so no batch forwards through
  half a state_dict; the serve processes sync while serving. A game
  that lives through a publication straddles it, and every window
  records the mean, the maximum and the share (az_history columns);
  with as many actors as games per window that is about one
  publication per game. `az_loop --stream` and `bench_pool --stream`
  are opt-in; the measurement against the barrier and against form
  A (more games than actors, which the pool already runs) is
  `scripts/stream_box.sh` with its rule pre-registered
  (docs/continuous_generation_20260918.md). Measured the same day on a
  single-tenant Ryzen 9 5950X box, two interleaved pairs: **FAIL under
  the rule by a hair in both pairs** (pair 1: 1.302x games per hour
  with a straddle mean of 0.69 over all windows, the first of which
  precedes any publication; pair 2: 1.248x with 0.70); the steady
  windows read 1.39-1.49x at a straddle of 0.92-0.93 with the server
  at its GPU roof, and form A 1.10-1.32x (docs/box_specs.md
  "Continuous generation against the barrier"). `--stream` stays
  opt-in. Whether a learner minds the straddling needs a learner that
  improves on the prior, run both ways.
- 2026-09-18 (a flaky test's root): **a unit's hash followed the
  process's hash seed, so every process enumerated a board's units
  in its own order.** `Unit.__hash__` hashed the id STRING, Python
  salts str hashes per process, and `gs.map.units` is a set: the
  encoder's token and vocabulary order, the sampler's action order
  and the sim all walked it, so the same seeded game diverged from
  its first decision between two processes (a fresh network even
  gave a unit type a different embedding row per process). Found
  because `tests/test_holdout_tripwire.py::test_holdout_stall_tripwire_exits_5`
  failed in one full-suite run and passed alone: its outcome
  depended on the pytest process's hash seed, not on test order
  (the exact suite prefix passed). Fixed at the root: the unit hash
  is a crc32 of the id, process-independent; `sim_self_play --seed`
  now also seeds torch, the global `random`, the policy's search
  generators (`MCTSPolicy(rng_seed=)`, threaded through the turn and
  plan policies) and the trainer's own subsampling generator (it
  drew from the global module); a rollout worker's game draws come
  from the iteration seed and the game index, not from which thread
  won the race for it. A seeded run now repeats byte for byte across
  processes and hash seeds. Consequence for old numbers: eval
  workers and pool actors are separate processes, so before this fix
  no game was reproducible across processes; the estimands and the
  standard errors are untouched (each game was still one draw), and
  no Elo needs re-measuring.
- 2026-09-19 (autonomous window): **the hex's terrain is its full set
  from the engine's aliases, behind a checkpoint flag.** The encoder's
  one-class view labelled 1,356 of the Ladder pool's 1,572 forest-
  overlay hexes as something other than forest; `Hex.terrain_mask`
  now carries each hex's terrain SET (`terrain_resolver.terrain_members`,
  from the database's movement and defense aliases) and
  `terrain_multi_hot` (on for a fresh network, absent for every
  earlier checkpoint, so `relset` observes exactly what it did) embeds
  it as a multi-hot over the terrain table. It travels with the fog
  gate everywhere (pre-encoder fingerprint, struct flags, hello, PLAY
  tuple) and eval records carry `terrain_a/terrain_b` as an estimand
  next to the basis. The arm (the reference's recipe with the flag,
  one pass, 800 decisive against `relset`) is pre-registered in
  docs/terrain_multi_hot_prereg_20260919.md and **RAN the same
  evening: +44 +- 12 Elo over `relset` (800 decisive, 450-350),
  holdout proxies equal to the twin's**; checkpoint
  `tier-b/terrain_multi_hot_20260919/arm_epoch0.pt`, the candidate
  reference pending the user's ruling; under the -1.5 decode it
  keeps +26 +- 12 over the reference under the same decode, and
  composed it is **+263 +- 16 over today's reference** (656-144,
  docs/composed_levers_prereg_20260919.md). The same day the panel's
  test 1 (end_turn at the actor level, docs/endturn_rule_prereg_20260919.md)
  ran on a box and **PASSED: p 0.752 +- 0.015 over 800 decisive games
  (602-198), about +193 Elo for a decode rule that trains nothing**,
  1.42x the decisions per side-turn and a capped fraction of 0.09
  against the reference's own 0.4; **the attribution arm, a plain
  end_turn logit offset of -1.5, beat it: p 0.789 +- 0.014, about
  +229 Elo**, so the lever is "act more" and the config scalar is the
  adopted form. The offset curve ran the same evening
  (docs/endturn_offset_sweep_prereg_20260919.md): -2.5 reads 0.801
  +- 0.014, -4 0.754, and acting while anything is legal (-99) 0.666,
  so the peak sits at -1.5 to -2.5 (a tie within 1 SE) at the corpus
  winners' rate of about 9-11 decisions per side-turn, and the
  end_turn head still marks turns worth passing. Whether
  `raw:t0+eo-1.5` becomes the reference decode is the user's ruling
  (BACKLOG.md "Training-signal panel").
- 2026-09-21 (user order): **the pool's serve batch cap was the
  binding constraint, and 64 is the default.** Every generation
  reading through 2026-09-18 ran the server under the az legs'
  16-leaf cap; the "GPU roof" of 0.46-0.54 ms per leaf was that cap
  (12% of the card's peak: 400 small kernels per 5,000-token batch).
  On a whole-CPU Ryzen 9 5950X 4090 host, two interleaved pairs
  (cap-16 arms repeating to 0.2%): 2,398/2,402 -> 3,223/3,222
  saturated leaves per second, 1.34x, games per dollar 1.17x and
  1.09x, batches of 39-40 leaves at 0.49 ms per leaf; the 96 cap adds
  nothing (the queue binds at 5 waiting requests) and the graphed
  server at 64 falls back on 75% of batches past its 12,288-token
  bucket cap (docs/serve_batch_prereg_20260920.md "Measured";
  docs/box_specs.md "The serve batch cap"). **Plan 1.3's 3,000 per
  4090 is MET.** The reference checkpoint ran through the pool for
  the first time (cap 16, 2,354 saturated, no error). The 48-minute
  box cost about $0.45.
- 2026-09-21 (Opus, branch `signal-levers`): **the scenario's
  economy is read from the scenario.** `build_scenario_gamestate`
  hardcoded 2 gold per village and a 70% experience modifier and
  patched them onto the state after the shared builder ran, while the
  replay path read both from the record -- one rule in two copies.
  No whitelist map declares either (two declare `mp_village_gold=2`,
  the value already used), so no Elo moves and a test pins all 21;
  five of the seven mini scenarios declare `village_gold=3`, so mini
  self-play had been paying a third less village income than its maps
  specify and mini games after this are not comparable with those
  before. The values now travel in the same dict fields a replay
  record carries (BACKLOG.md "The scenario's economy is read from the
  scenario"). `tools/analysis/corpus_census.py` reads era, layout,
  factions and host settings out of all 17,019 raw replay headers:
  the corpus is default-era play (29% declare `era_dunefolk`, which
  is the default era plus one faction, and no game fields a Dunefolk
  side), every scenario name resolves to one layout, all 23
  mainline-named maps are byte-identical to the shipped 1.18.7 maps,
  and the games split 11,457 whitelist / 582 mainline off-whitelist /
  4,936 mini / 44 custom.
- No box is rented (2026-09-21). Both 2026-09-19 rulings are taken
  (the reference is `terrain` at `raw:t0+eo-1.5`). Phase 2 is next:
  docs/plan_20260904.md 5, whose first measurement, the turn-level
  value gap against the current reference, is pre-registered
  (docs/turn_gap_ref_prereg_20260921.md; `tools/turn_gap.py
  --reference`, `scripts/turn_gap_ref_box.sh`, about $1.20) and waits
  for the user's word.
- 2026-09-22 (user order: finish the preprocessor rework before any
  retraining): **the scenario pipeline has detectors now, and the
  first thing they did was disagree with the game.** The record is
  docs/scenario_build_plan_20260922.md, whose work items W0-W3, W5 and
  W6 are done; W4 (the engine oracle for scenario init) is scoped but
  not built. The four detectors: our macro expansion against the
  game's own, per pool scenario (`tools/analysis/expansion_diff.py`,
  0.2 s, in the fast tier, and it FAILS under the old rule); every tag
  path and attribute the pool declares against a manifest that binds
  each MODELLED entry to a reader whose symbol is checked
  (`tools/analysis/scenario_surface.py`, 51 paths, 178 pairs, 0
  UNKNOWN); a dispatch that warns or raises instead of a silent no-op
  (`WESNOTH_STRICT_WML`); and each substitution's precondition checked
  where it is relied on. Found and fixed: `{DEFAULT_SCHEDULE}` sat on
  the cosmetic list, so our expansion emitted ZERO `[time]` blocks
  where the game's emits six (latent -- all 28 pool scenarios and all
  17,019 corpus games use exactly the default cycle, measured, so the
  hardcoded `TOD_DEFAULT_CYCLE` was right); macro names may contain
  `:` (`INTERNAL:SPECIAL_NOTES_*` all collapsed onto one name) and
  `#arg NAME ... #endarg` declares an optional named argument with a
  default (`{OVERLAY}` leaked unsubstituted), both now in
  docs/wesnoth_rules.md; the WML parser kept the `_ "` translatable
  marker; `random_start_time` has a THIRD form (a value list) that a
  yes/no coercion silently read as dawn; and `ai_special=guardian` was
  read by NOTHING while `tools/neutral_ai`'s combat-only substitution
  depended on it -- on Modified_Tiny_Close_Relation it is the only
  thing keeping the Tentacle still, and the docstring's stated reasons
  covered neither that map nor the real reason for two others.
  Generation now reads ONE rendering of a scenario (it read the
  committed template for the time of day through three private regexes
  while reading `load_scenario_wml` for everything else); the template
  builder imports again after six weeks (a dangling
  `DRILL_SCENARIO_IDS`) and all 29 templates regenerate BYTE-IDENTICAL
  from the current Steam install. The 28-scenario and 120-replay
  snapshots did not move: this block changed what we NOTICE, not what
  we build. **Owed: a `diff_replay` corpus sweep** (about 20 minutes
  and $0.20, with the old predicates monkeypatched back as the
  control), since the expander changes touch the reconstruction path
  too. The time-of-day encoder arm (docs/time_of_day_prereg_20260922.md,
  `GLOBAL_FEAT_DIM` 6 -> 8, `OBSERVATION_EPOCH` 3 -> 4) is built and
  tested but PARKED by the same user order, to be batched into one
  retrain with whatever else the rework turns up.
- 2026-09-23 (Opus 5.5; versions 0.1.0 -> 0.2.1): **a second model
  reviewed the 09-22 work before it was committed, and the manifest
  that claimed to bind every attribute to its reader did not.** Its
  test checked that a named reader EXISTED; 22 of 72 MODELLED entries
  named a function that never reads the attribute, and several
  (`random_traits`, `affect_self`, `cumulative`) are read by nothing.
  It also covered only the pool, not the corpus maps reconstruction
  loads, and keyed on raw nesting, so a `[unit]` inside `[switch]
  [case]` was invisible. Now: 31 scenarios, control flow folded out, a
  reader must name its attribute in its own source, and SUBSTITUTED
  rose from 8 to 25 -- seventeen behaviours claimed as read are
  visibly unread, each with the precondition that makes that safe.
  The review's sharpest find: under the old expander Hornshark
  Island's Mermaid Initiates (248 corpus games) carried an EMPTY
  `[heals]` block (the `INTERNAL:` macro collapse), and healed correctly
  only because the reader defaulted a missing value to 4, which equals
  `{ABILITY_HEALS}`; the engine's default is 0 (`heal.cpp:211`), and a
  heals+8 map would have healed half. The reader now reads the value.
  Also: preprocessor `#ifdef` / `#ifndef` / `#else` are evaluated as
  the engine does (0.2.1; nothing we build moved), and **versioning
  starts** (`wesnoth_ai.__version__`, see Code Style). The corpus
  sweep is NOT recommended as a standalone box: reconstruction reads
  only events and time areas from our expander, and diffing both, old
  against new, over all 31 corpus scenarios shows only display text
  and one heals block with identical sim abilities -- the sweep would
  pass both ways and certify nothing. The Rust change and the slow
  tier, which the laptop cannot run, passed on CI the same day (next
  entry).
- 2026-09-23 (0.2.2 -> 0.3.0): **every push is tested on GitHub, the
  Rust paths included.** Work happens on topic branches (Code Style,
  Branching), and `.github/workflows/tests.yml` runs on every push: it
  builds the Rust wheel from the pushed commit, refuses a wheel whose
  `__phase__` differs from the source's, lints, and runs the full
  suite, both tiers, on a 4-core Linux runner (about 13 minutes in
  all). Its first green run (1,322 passed, 49 skipped) built the
  phase-11 wheel, which carries the time-of-day features and had never
  been built, and every Rust test passes on it, `GameCore` and the
  combat and observation kernels included. The first run's nine
  failures held no fidelity bug. Two tools and one test reached around
  the Rust kernels (the strike verifier wrapped the Python resolver,
  the swap detector's scripted RNG had no `seed_int`, the relevant-set
  gap test shrank a set the kernel does not read); one test assumed 16
  host cores; and three SL tests ran on a corpus a clone lacks, where
  `train()` completed zero steps without complaint -- a missing or
  empty corpus now raises. The 49 skips need data outside git (33:
  replay, imitation and value corpora), CUDA (14) or a live Wesnoth
  (2). Also landed: the fast tier's 41/42 skip flicker was an unseeded
  turn search, now seeded (0.2.4).
- 2026-09-23 (0.4.0): **real Wesnoth builds every pool scenario's
  starting state, and ours agrees with it on every compared field.**
  `tools/scenario_init_oracle.py`, the last work item of
  docs/scenario_build_plan_20260922.md (W4), launches a multiplayer game
  per scenario with an AI config on side 1 whose Lua reports the whole
  board at side 1's first turn, builds the same game the way self-play
  does, and compares 26 fields: each side's economy, fog and recruits,
  every unit, village owners, terrain and lawful bonus per hex, and the
  time of day (record
  `training/metrics/fidelity/scenario_init_oracle_20260923.json`, 28 of
  28). It found two defects, both fixed. Three minis (`2p_mini`,
  `2p_mini_edited`, Modified_Tiny_Close_Relation) declare `fog=no` and
  self-play played them under fog, because the builder never read the
  declaration. The statues of Caves of the Basilisk, Sullas Ruins and
  Thousand Stings Garrison lacked the modifications that leave them 1 hp
  and no moves; that is encoder input only, since a petrified unit
  cannot be attacked in Wesnoth or in the simulator. A command-line
  start is not a lobby: it skips `configure_engine::write_parameters`,
  so villages pay 1 and experience runs at 100% unless the harness
  supplies the lobby's values, which it does (docs/wesnoth_rules.md).
  **Consequence for numbers:** mini self-play on those three maps is not
  comparable with earlier games, and the statue input changes on 3 of
  the 21 ladder maps eval draws from, so Elo measured from here is
  cross-build against earlier numbers. The oracle launches Wesnoth for
  about 50 s per scenario and runs only by hand, with the user's
  agreement; no test launches it (user ruling the same day).
- 2026-09-23 (0.4.1): **Phase 2's first measurement reads RICH.** The
  turn-level value gap under the reference (terrain at `raw:t0+eo-1.5`,
  docs/turn_gap_ref_prereg_20260921.md "Measured"): 7 of 60 holdout
  side-2 positions have a sampled alternative turn confirmed better by
  at least 0.25, 0.117 +- 0.041, over the bar of 6 (predicted 1 to 3).
  In 6 of the 7 the better turn takes more decisions (14.4 against
  10.3). By the rule the next factor is the pre-graded pipeline of
  docs/turn_proposer_design_20260905.md, starting with its section-6
  check of forward-only graders. Box 52267135, about $1.05.
- 2026-09-23 (0.4.2): **no forward-only pre-grader at this head.** The
  section-6 check, run on that measurement's recorded candidates with
  no box (docs/turn_gap_ref_prereg_20260921.md "Pre-grader check"): on
  the confirmation's 16 positions the value head's within-position
  residual is 0.323 +- 0.059 with 2 of 8 winning alternatives ranked
  below their base, and the HP margin's 0.322 with 3 of 8; the value
  head correlates 0.07 with the playout mean there. The turn-search
  pipeline proceeds without a pre-grader (the design's rows 1 to 6); a
  boundary value net is the prerequisite for row 7. The 2026-09-05 run
  on the seed's candidates, first written down the same day: value head
  0.156 with 2 of 8 below (a fail), HP margin 0.154 with 8 of 8 above.

Standing rules (full list in the plan): the reference player is
`terrain` at `raw:t0+eo-1.5` (user ruling 2026-09-20; one checkpoint
and one decode, both in `configs/reference_player.json`, which
`tools/reference_player.py --flags b` turns into run_elo_batch flags);
every strength claim is a PURE match against it with the standard
error stated; no teacher is distilled before it wins such a
match; one factor at a time, each with its own number and kill
criterion; proxies are crash barriers, never verdicts; compute on
rented boxes only, proposed with cost first.

Eval procedure: `tools/run_elo_batch.py ... --mcts-sims 0
--raw-temperature-a 0 --raw-temperature-b 0` for raw players (the
procedure tag is `raw:t0`; the legacy sampler is `raw` and never mixes
in one outdir); searched players carry `mcts:<sims>` (Gumbel root) or
`tcs:<sims>`. Run on a 4090 box with `--device cuda --jobs 10
--persistent-workers --shared-inference` (docs/box_specs.md). The last
two are `store_true` and default OFF; without them you get the slowest
mode in the repo, and every current match script passes them.

**Timing, corrected 2026-09-13, then corrected again the same night.**
The "40 raw games in 2 minutes" that stood here was match A of the
argmax control -- argmax against the SAMPLING player, 17 turns median
-- not `raw:t0` against `raw:t0`, which stalls: 17 of 40 games at the
cap, 125 turns median.

Current figure: **a 40-game `raw:t0` match through the recommended
path is 42-74 s** (2026-09-13, five arms of `relset` against itself,
docs/box_specs.md "The eval path does not want more workers or more
servers"). Quote the RANGE: the baseline arm repeated at 74 s against
its own 42 s, a 1.76x swing, so a single wall is not resolvable and an
800-game match is budgeted at about 18 minutes and $0.20. Those
figures are a box class: a single-tenant Core Ultra 9 285K host
(2026-09-18) played the same match in 24-25 s, its server at 4.6-4.9
ms per batch against 27 ms there.

Two older numbers are NOT comparable and should not be re-quoted: the
408 s one-process figure is a TWENTY-game wall (`eval_workers/
eval_plain.log` reads "20 pending"), and the 145 s shared-server
figure is 40 games but from 2026-09-05, on the seed basis, before
persistent workers were combined with the relevant-set basis, the Rust
kernels and the observation kernel. The "14 minutes for 40 searched
games" is the other row of the same superseded 2026-09-04 run, and it
was MCTS-32 against a RAW opponent, not a searched self-match.

## Architecture

### Two paths that share encoder + model + trainer

**Source layout (2026-07-23 reorg).** The core library modules listed
below now live in the **`wesnoth_ai/`** package (imported as
`from wesnoth_ai.X import ...`), not at the repo root. Scripts keep
their `tools/` prefix; the test suite moved to **`tests/`**
(`tests/conftest.py` bootstraps `sys.path`); `main.py` (the setup CLI)
stays at the root. So a bare name like `classes.py` below means
`wesnoth_ai/classes.py`.

**Production path: in-process simulator.**
- `tools/wesnoth_sim.py` — pure-Python game logic. Reuses the
  replay-reconstruction machinery from `tools/replay_dataset.py`
  (which is bit-exact against Wesnoth via `[mp_checkup]` oracle on
  combat); just swaps the data source from "WML command stream"
  to "policy queries."
- `tools/sim_self_play.py` — self-play training entry point. Drives
  N games per iteration through `WesnothSim`, calls `policy.observe`
  for shaping rewards, applies one gradient update per iteration via
  `policy.train_step`. `--mcts` flag swaps in `MCTSPolicy`.
- `tools/scenario_pool.py` / `tools/scenarios.py` — scenario
  randomization for training (Ladder Era 21-map whitelist, faction
  randomization with optional `--forced-faction` lock).
- `tools/mcts.py` / `tools/mcts_policy.py` — MCTS implementation
  and the MCTSPolicy adapter that wraps TransformerPolicy.

**Live-Wesnoth path (eval only).**
- `main.py` — setup / maintenance CLI (`--check-setup`,
  `--clean-games`). No longer drives training or `--display`.
- `wesnoth_ai/wesnoth_interface.py` — one Wesnoth process per eval game; state
  channel uses `std_print` → log-file tail (CA-blacklist bypass via
  custom Lua AI stage); actions written atomically as `action.lua`
  and read via `wesnoth.read_file`.
- `add-ons/wesnoth_ai/` — Lua side: `lua/state_collector.lua`,
  `lua/turn_stage.lua` (custom AI stage replacing default RCA so
  failed actions don't blacklist the CA), `lua/action_executor.lua`,
  `lua/json_encoder.lua`. `scenarios/training_scenario.cfg`.
- `tools/eval_vs_builtin.py` + `tools/eval_runner.py` — pits the
  trained model against Wesnoth's default RCA AI across a
  (map × matchup × side-swap) matrix.
- `tools/sim_demo_game.py` — headless one-game demo via the sim;
  exports a Wesnoth-loadable `.bz2` via
  `sim_to_replay.export_replay_from_scratch` (composes save WML
  from `wesnoth_src/` templates, no replays_raw/ dependency).

**Shared by both paths (all under `wesnoth_ai/`):**
- `wesnoth_ai/classes.py` — `GameState`, `Unit`, `Hex`, `Map`,
  `SideInfo`, `state_key`.
- `wesnoth_ai/encoder.py` — `GameState` → tensors (per-unit, per-hex,
  recruit phantom features, global features).
- `wesnoth_ai/model.py` — `WesnothModel` transformer with
  distributional C51 value head; emits `ModelOutput(actor_logits,
  type_logits, target_logits, weapon_logits, value, value_logits,
  cliffness, ...)`.
- `wesnoth_ai/action_sampler.py` — legal-action enumeration with
  priors; combat-oracle attack-bias on target logits.
- `wesnoth_ai/transformer_policy.py` — Policy adapter (`select_action`,
  `observe`, `train_step`, `save_checkpoint`, `finalize_game`).
- `wesnoth_ai/trainer.py` — REINFORCE + value baseline (`step`) and
  AlphaZero-style soft-target distillation (`step_mcts`); both
  use the categorical CE value loss against C51 atom projections.
- `wesnoth_ai/rewards.py` / `configs/reward_selfplay.json` — shaping
  reward (terminal ±1, gold/damage/village deltas, per-turn penalty,
  unit-type bonuses, turn-conditional bonuses).

### Coordinates

- **Wesnoth uses 1-indexed hex coordinates.**
- **Python uses 0-indexed hex coordinates everywhere internally.**
- Conversion happens in `wesnoth_ai/state_converter.py` (both
  directions). Keep it there; do not sprinkle `±1` around the codebase.

## Architecture Principles

### 2. Self-play is non-negotiable
The learning signal comes from self-play. Bootstrapping methods
(imitation from the built-in AI, human games) may be used to warm-start,
but the end state is self-play. Do not design architectures that
preclude it.

### 3. Config-driven, customizable
Rewards, openers, strategic biases, training curriculum — all must live
in data/config, not buried in network weights or scattered constants.
When you add a new behavior, ask: "could a modder flip this without
touching model code?"

### 4. The simulator must be perfectly faithful to Wesnoth
The simulator's combat math is verified against Wesnoth's own
`[mp_checkup]` oracle on strict-sync replays (731/731 strikes
matched). Any sim change that touches combat, healing, or
advancement must keep that parity. `tools/diff_replay.py` is the
regression check (runs the simulator over a corpus, compares
against the recorded WML command stream). New scenario events go in
`tools/scenario_events.py`; new abilities in `tools/abilities.py`;
both with citations to `wesnoth_src/` file:line.

Any mismatch between the simulator and Wesnoth (usually surfaced
by OOS errors when strict syncing sim-produced replays) is a
critical issue to be investigated and solved at the root.

When live Wesnoth is in the loop (display, eval), the same
narrow-waist principle applies: state crosses the bridge as one
well-defined serialization, actions as one schema, Lua side stays
dumb. But that path is no longer how training data is generated.

### 5. Failures are visible
Both paths log timeouts and stage-of-failure. The simulator returns
typed errors (e.g. `"recruit:insufficient_gold"`) that
`sim_self_play.py` surfaces in the per-game summary. The bridge
path (display, eval) wraps any Python wait in a finite timeout and
logs which stage timed out; Lua errors reach the Python log, not
silently die inside a `pcall`.

### 6. Legality mask = pure function of OBSERVABLE STATE
The action sampler's "legality mask" answers exactly one question:
**what can the policy validly attempt right now, given the
information it has?** It is a pure function of the observable state.

Observable state has two pieces:
1. **Visible game state** — what the encoder sees. Includes
   own-side fog hexes (the encoder retains them after the
   2026-04-28 fix), but NOT enemy units hidden in fog and NOT
   any god-view information from the simulator.
2. **Per-turn rejection history** — the set of hexes where a
   recruit attempt has bounced this turn, stashed on
   `gs.global_info._recruit_rejected_hexes`. Cleared at
   `init_side` (so each side starts a turn with a fresh slate).

What this resolves: the mask is NOT "what the engine will accept"
(which would require god-view fog truth, cheating) AND it is NOT
just "what the model wants to attempt" (which would let the model
infinite-loop on the same fog-hidden hex). It is "what the model
*can* validly attempt given everything it has observed so far,"
which gives the model the same information a human player has.

Concretely:

  - Fog castle hexes ARE legal recruit targets (the model can
    attempt them; like a human, it can't see what's there until
    it tries).
  - After a rejection, that hex becomes illegal AND a per-hex
    "recruit_rejected" bit appears in the encoder feature -- the
    mask consults the rejection set, the model sees the bit; both
    read the same state.
  - Next turn, rejection history clears. The hex is legal again
    (the enemy may have moved away).

Designs that violate this contract are bugs. In particular:

  - **God-view masking is forbidden.** Even though the simulator
    has fog truth, the mask must not consult it. The model must
    play with the same information a human would have.
  - **Rejection history is per-turn.** Persisting it across turns
    would model long-term knowledge that the human player doesn't
    have (the enemy could have moved).
  - **The mask must be a pure function of observable state.** Two
    decisions on the same observable state produce identical
    masks. Mutable state outside `gs.global_info` (e.g. cached
    per-call counters) is forbidden.

This contract applies to all action types -- attack, move,
recruit, end_turn -- not just recruits. The recruit-rejected
case is the most prominent example, but the rule generalizes:
if a future action category needs "we tried this and it
bounced" tracking, it lives on `gs.global_info`, clears at the
right turn boundary, and is mirrored in encoder features.

## Code Style

### File Size
Target ~600 lines. Split by responsibility when a file grows past that.

### Naming
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private members: `_leading_underscore`
- Lua: same conventions (Lua allows `snake_case` fine)

### Branching (adopted 2026-09-23)
GitHub flow: `main` holds only finished, checked work. Everything else
happens on a short-lived topic branch cut from `main`, merged back when
it is ready, then deleted.
- **Names:** `feature/<name>`, `fix/<name>`, `test/<name>`,
  `exp/<name>`; short, descriptive, hyphenated
  (`fix/heals-value-default`). One task, one branch.
- **Ready to merge** = `ruff check .` clean, the fast tier green
  locally, and the branch's latest CI run green (`gh run list --branch
  <branch>`; see Testing).
- **Merge with a merge commit** (`git merge --no-ff`), never squash:
  the commit messages are this project's lab notebook.
- **`exp/` branches differ.** An experiment's code merges only if it
  wins, but its pre-registration and its result land on `main` either
  way (a records-only merge or a direct commit): "rejected: X, because
  Y" is what stops X being proposed again.
- **Status entries** in this file and in BACKLOG.md are written on
  `main` when the work lands, never on the branch. Nearly every change
  touches them, so branch-side edits would conflict with each other.
- **Parallel agent sessions** each use their own branch in their own
  worktree, so no session edits another's files.
- **Pushing** a topic branch is free (user ruling 2026-09-23); pushing
  `main` asks.
- **Deleting a branch** that holds commits `main` does not have: tag
  its tip `archive/<name>` and push the tag first, so the work stays
  reachable.
- Rejected: Git Flow's `develop` / `release/` branches. They serve
  software shipped in numbered releases, and GitHub's own flow has
  neither (docs.github.com, "GitHub flow").

### Versioning (adopted 2026-09-23, revised the same day)
- `wesnoth_ai.__version__` in `wesnoth_ai/__init__.py` is the only
  place the version lives, and it numbers the states of `main`:
  **every change that lands on `main` bumps it once** -- M for a
  feature, P for anything else (fix, test, records) -- and N changes
  only on the user's explicit decision (0 until then).
- Bump it in the commit that lands the change on `main`: for a merge,
  `git merge --no-ff --no-commit <branch>`, bump, then commit. Topic
  branches never touch it, so two branches never claim one number.
  State the new version on the first line of that commit's body.
- History starts at 0.1.0 (the time-of-day encoder commit); nothing
  before it carries a number. The first day bumped per commit; merges
  to `main` are the unit from 0.2.2 on.

### Linting (adopted 2026-08-05)
- **Run `ruff check .` before committing** — config in `ruff.toml`
  (rules E/F/W; E501/E731 off; E402 allowed in tools//tests//scripts
  for the sys.path bootstrap pattern). The check must come back clean;
  `ruff check --fix` applies the safe autofixes.
- Intentional exceptions get a targeted `# noqa: <rule>` with a short
  reason, never a blanket file ignore.

### Imports
- No unused imports.
- No circular imports — if A needs data from B, pass it explicitly.
- `TYPE_CHECKING` blocks for type-only circular deps.

### Data patterns
- Structured data → `@dataclass`, not raw dicts.
- Action payloads on the wire are dicts (they cross the Lua boundary);
  internally prefer typed structures.
- Material/faction properties → LUTs in config, not `if`-chains.

### Encapsulation
- Python systems talk through explicit APIs on `GameManager` /
  `WesnothGame`, not by reaching into private attributes.
- Lua code never decides game logic; it just serializes state and
  executes actions the Python side chose.

## Testing

### Philosophy
Tests exist to catch regressions you'd otherwise only notice after
burning an overnight training run. Prefer few behavioral tests over
many line-coverage tests.

### What tests we have (and what they are NOT)
- `test_wml_parser.py`, `test_integration.py`, `test_lua_actions.py`
  are **Python unit tests with synthetic inputs**. Despite the name,
  `test_integration.py` does NOT launch Wesnoth. A real end-to-end
  test requires a live Wesnoth process — we don't have one yet.

### Guidelines
- Run `pytest` after changes. This runs the FAST tier (~2.5 min):
  tests marked `slow` (full-game / subprocess / threading e2e,
  >10s each — see pytest.ini) are excluded by default.
- **CI runs the FULL suite — `pytest -m ""` — on every push**
  (`.github/workflows/tests.yml`, GitHub Actions, about 10 minutes of
  tests). The slow tier holds the e2e regression guards (MCTS
  self-play smoke, concurrent train-step races, export validation);
  the fast tier alone does NOT cover them, and the laptop does not run
  them (they generate self-play games). A branch merges on a green
  run, and a training campaign launches from a commit that has one.
  CI has no GPU and no corpora: the CUDA tests and the tests that read
  replay, imitation or value data skip there.
- **Never run more than one pytest invocation at a time.** Each
  pytest spawns a Python process that imports torch + the model;
  parallel runs balloon memory (5+ GB per process) and a stuck
  test compounds the problem. Wait for each command to fully
  return (foreground) or be killed before launching the next. 
  If a test hangs, kill the process before starting another — 
  don't queue a second behind it. No "let me also kick off X 
  while we wait" patterns. (Lesson from 2026-05-10: four parallel 
  pytest jobs stuck on a hanging `_select_one` loop produced 
  multiple multi-GB zombie Python processes that locked up the 
  user's machine.)
- Use `constants.py` values in assertions (not hardcoded duplicates).
- **Every run lists its failed, errored and skipped tests** (pytest.ini
  `-rfEs`). A skip count that differs between two runs of one tree is a
  test that sometimes does not run; diff the two SKIPPED lists to name
  it.
- **A test that builds a search policy seeds it**: `MCTSPolicy(...,
  rng_seed=N)` / `TurnCommitPolicy(..., rng_seed=N)`, or `rng=` on a
  direct `mcts_search`. Unseeded, the search draws fresh OS entropy,
  so any assertion or skip that depends on what it chose varies from
  run to run -- the fast tier's 41/42 skip flicker was exactly that
  (2026-09-23).
- Never weaken a test without explicit user confirmation. A failing
  test is a signal — find the root cause first.
- **The Rust paths are tested on CI, not on the laptop.** CI builds
  the wheel from each pushed commit and asserts its `__phase__` equals
  the one `rust/wesnoth_core/src/lib.rs` declares, so every Rust test
  runs there; a green CI run certifies a Rust change, and a box is
  needed only for what CI cannot run (CUDA, throughput). The laptop's
  installed wheel is phase 3 and exports only `encode_raw_streams`,
  `enumerate_moves`, `unit_reach_arrays`, so locally
  `tests/test_game_core.py` skips in full and the other
  `tests/test_rust_*.py` files skip in part (the suite prints a banner
  saying so). Check `python -c "import wesnoth_core;
  print(wesnoth_core.__phase__)"` against lib.rs before believing any
  local core-on result.
  The wheel cannot be rebuilt here: `cargo check` in the project tree
  is refused on every crate whose build script must execute (measured
  2026-09-22 on `pyo3-build-config`, `proc-macro2` and `libc`, each
  "Accès refusé", os error 5). A test that reads the Rust SOURCE for a
  constant is the one local check available
  (tests/test_time_of_day_features.py does this for the feature
  widths).

## Working Style

- **High autonomy** on reversible local work (edits, tests, reads),
  including creating, switching, committing on and pushing topic
  branches (see Branching).
- **Ask before**: merging to `main` or committing to it directly,
  pushing `main`, force-pushing, deleting a branch whose work is not
  merged,
  deleting tracked files, making architectural changes (new IPC,
  replacing the model, etc.).
- **Give best effort**: production-quality code with edge cases handled,
  not sketches.
- **Research first** on non-trivial Wesnoth internals (WML, Lua API,
  scenario/add-on loading, `--plugin`, stdout behavior). The
  [Wesnoth wiki](https://wiki.wesnoth.org/) and
  [Lua API reference](https://wiki.wesnoth.org/LuaAPI) are authoritative.
- **Don't guess at Wesnoth WML attrs / engine semantics — check the
  source.** The wiki sometimes lags, has edge cases wrong, or omits
  attrs. **`wesnoth_src/` has NO `src/` tree** -- it is a WML-only
  copy of the local 1.18.7 install (see the provenance note above), so
  `grep wesnoth_src/src/` returns nothing and the 68 such citations in
  docs/wesnoth_rules.md are not locally verifiable (85 C++ citations
  in all, over 45 distinct files; 17 are already written bare). For C++ engine
  internals read the **1.18.4 tag on GitHub raw**, e.g.
  `https://raw.githubusercontent.com/wesnoth/wesnoth/1.18.4/src/units/unit.cpp`,
  and cite `src/...:line` as the rules catalog does. For WML and Lua,
  `wesnoth_src/data/` IS local and authoritative -- grep it first when
  you would otherwise hand-wave ("`income=` is probably an offset").
- **Don't assert without checking.** This applies to factual claims
  about Wesnoth (rules, mechanics, unit stats) AND to claims about
  our own code ("the function does X", "this list covers all cases",
  "the heuristic always fires correctly"). If you find yourself about
  to write a confident-sounding sentence in a commit message, code
  comment, or message to the user, pause: have you actually verified
  the claim with a grep / read / test? If not, either verify first OR
  hedge the language ("I believe X holds because Y; not yet verified").
  Especially: when listing a closed set ("the three races that get
  undrainable are undead, mechanical, elemental"), grep the source for
  the relevant marker and confirm the count matches before publishing.
- **`docs/wesnoth_rules.md` is the source-of-truth catalog.** Read
  it BEFORE researching a Wesnoth rule from scratch — most
  established rules are pinned there with verbatim source quotes.
  When you establish a new rule (or correct an existing one), add
  / edit the entry in `docs/wesnoth_rules.md`. Required: file:line
  citation, verbatim quote of the enforcing code, and (when the
  rule wasn't where you'd naively expect) a "why non-obvious" note.
  Grep recipes and a file map for common Wesnoth-source questions
  also live there. Treat the doc as a force multiplier: each rule
  added saves the next exploration session hours, and the doc
  prevents truth-drift across sessions.
- **Wesnoth rules can live in C++, Lua, OR WML — search all three.**
  Common gotcha: a rule we're hunting in `wesnoth_src/src/` is
  actually in `wesnoth_src/data/multiplayer/eras.lua` or a WML
  macro under `wesnoth_src/data/core/macros/`. After grepping `src/`,
  always also grep `data/multiplayer/`, `data/core/macros/`,
  `data/lua/`. Rules with a "post-pass" feel (applied after unit
  setup) often hide in `[event]name=prestart` Lua callbacks.
- **`changelog.md` is HISTORICAL — verify against current source.**
  Old changelog entries describe behavior at THAT version, which
  may have changed since. Cross-check any changelog quote against
  the live `wesnoth_src/` code path before treating it as authority.
- **`docs/design_constants.md` catalogues DERIVED numerical
  constants** (not arbitrary tuning knobs). Anything with a
  derivation — math, measurement, fixed external standard —
  belongs there rather than buried in a one-line code comment.
  When you find yourself writing "where does this 0.577 come
  from?" or "why exactly 51 atoms?", the rationale belongs in
  `docs/design_constants.md`; cite it from the code with a short
  comment + cross-reference. Pure tuning knobs (learning rate,
  c_puct, etc.) stay in `constants.py` with their own comment
  block — those aren't derived, they're picked, and the picking
  rationale (which may be "AlphaZero paper" or "experiment
  pending") stays nearby.
- **Magic-number principle, more generally:** if a number's
  origin isn't obvious from its name or its surrounding
  comment, the reader will eventually waste time deriving it
  again. Either rename it (`PRIOR_VAR_UNIFORM_M1_P1`), comment
  it inline (1-2 lines, no math), or — for anything more
  involved — write it up in `docs/design_constants.md` and
  cross-reference. Same principle applies to thresholds,
  bucket sizes, atom counts, normalizers, anything where
  someone could reasonably ask "why exactly that value?".
- **Prefer removing over adding.** This codebase is recovering from
  bloat. When a feature is load-bearing, we'll re-add it with evidence.
