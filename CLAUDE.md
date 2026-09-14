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
  up real Wesnoth)

### Wesnoth data provenance (updated 2026-06-12)

`wesnoth_src/data/` is a WML-only copy (cfg/lua/map, no art) of the
LOCAL STEAM INSTALL's data tree — currently **1.18.7** — refreshed
via:

    robocopy "C:\Program Files (x86)\Steam\steamapps\common\wesnoth\data" wesnoth_src\data *.cfg *.lua *.map /S

It serves the RUNTIME readers (scenario_pool factions/eras,
scenario_events, map files). It is NOT a git checkout anymore and
has no `src/` tree; for engine-internals research, read the GitHub
**1.18.4 tag** directly (as docs/wesnoth_rules.md entries do).

**The sim's runtime WML subset IS tracked in git (since 2026-07-02):**
`data/multiplayer/{factions,scenarios,maps}` + `data/add-ons/
Mini_Maps_Collection` (~840KB) — so a bare `git clone` can run
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

## Current status (2026-09-04, entries through 2026-09-13)

**Read `docs/plan_20260904.md` first; `BACKLOG.md` holds the next
actions in order.** Superseded status blocks, plans, leg records and
mechanism specs are in `docs/archive/` (index in its README); the 78
quarantined training mechanisms are in `quarantine/INVENTORY.md`.

State of play:
- The reference player (user ruling 2026-09-11 night) is `relset` at
  temperature 0: the relevant-set twin of seed2 at one pass, HF
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
  independent draws. A self-pin worth quoting is the 800-game one
  under per-game luck and the current hide-cover rule.
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
  (~33 ms per 16-leaf batch) is the ceiling now, a second serve
  process is implemented and queued for measurement; training path
  684 -> 352 s per iteration (batched policy loss, batch 16, bf16),
  the actor's packed masks shipped with each experience is the next
  cut; eval 4.2x through persistent workers, 1.5x more through the
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
  server halves the mean batch, so it cannot win, refuting the
  standing 1.3-1.5x expectation -- a reading off the server logs, not
  a recorded number, see docs/box_specs.md); and bf16 on the imitation trainer
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
  (measured: 4 of 120 hider replays reconstruct differently, 0
  divergences either way). The rule itself is established by the
  engine's macro text and pinned by tests/test_hide_cover.py;
  tests/test_visibility.py pins the observation and move-truncation
  halves on a code the old table missed.
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
  more audits. **Nothing below has been on a box; the corpus sweep and
  the second-serve-process measurement are queued in
  `scripts/postreview_box.sh`, about an hour on a 4090, roughly $0.35.**
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
    old-rule hider sample (164 of 300, 4 of 120) and "80% GPU" have
    no record and are marked so; the hex census now has a tool
    (`tools/analysis/hide_cover_census.py`) and a record. Fixed in
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
- Rulings (2026-09-05): no optimizations conditioned on the MCTS
  loop; scope every box test, train sparingly; results are written
  on the run, never as atomic dumps; a box job past ~1.5x its
  estimate gets inspected and cut.
- No box is rented (2026-09-13, after that day's three boxes). Phase 2
  is next: docs/plan_20260904.md 5, whose first measurement is the
  turn-gap pre-registration.

Standing rules (full list in the plan): the reference player is
`relset` at `raw:t0`; every strength claim is a PURE match against it with the
standard error stated; no teacher is distilled before it wins such a
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
800-game match is budgeted at about 18 minutes and $0.20.

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
- **Run the FULL suite — `pytest -m ""` (~11 min) — before
  committing sim/trainer/mask changes and before launching any
  training campaign.** The slow tier holds the e2e regression
  guards (MCTS self-play smoke, concurrent train-step races,
  export validation); the fast tier alone does NOT cover them.
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
- Never weaken a test without explicit user confirmation. A failing
  test is a signal — find the root cause first.
- **A green local run does NOT cover the Rust paths.** The laptop's
  installed `wesnoth_core` wheel is phase 3 and exports only
  `encode_raw_streams`, `enumerate_moves`, `unit_reach_arrays` — no
  `GameCore`, no `resolve_attack`, no `observe_side`. So
  `tests/test_game_core.py` skips in full and the other
  `tests/test_rust_*.py` files skip in part, silently. Check
  `python -c "import wesnoth_core; print(wesnoth_core.__phase__)"`
  against `rust/wesnoth_core/src/lib.rs` before believing any core-on
  result, and certify Rust changes on a box.
  The wheel cannot be rebuilt here, and the reason is narrower than
  "cargo does not work": `cargo check` in the project tree runs most
  build scripts fine and is refused on exactly one,
  `pyo3-build-config` ("Accès refusé", os error 5 — the compiled
  build-script binary is never executed). That looks like a security
  policy blocking that one binary. If it is ever whitelisted, the
  local gate gains the whole Rust path.

## Working Style

- **High autonomy** on reversible local work (edits, tests, reads).
- **Ask before**: committing, force-pushing, changing branches,
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
