# BACKLOG

Live backlog for `docs/plan_20260904.md`. The pre-restart backlog
(1,055 lines of rulings and open items, 2026-05 to 2026-09-04) is
archived verbatim at `docs/archive/backlog_20260904.md`.

## NEXT (2026-09-26)

**1. `obs8` is the reference** (user ruling 2026-09-25): +73 +- 13 Elo
over `terrain` (800 decisive games) (docs/observation_retrain_prereg_20260924.md). Every
number from here is measured against it.

**Done: the turn-ranking value function FAILS** (2026-09-26,
docs/turn_value_prereg_20260925.md "Measured"): on 199 human-game
positions a linear arm on `obs8`'s trunk features reads 0.274, a head arm
0.269 and a rollout read 3 half-turns ahead 0.422 (corrected
within-position correlation; bar 0.7, kill 0.5), with the crash barrier
passed. By its rule the next proposal is a fine-tuned copy of the trunk
on the better labels; the reported readings (the HP margin after the
turn 0.397, a rollout read 7 half-turns ahead 0.532) belong in that
design. The code stays on `exp/turn-value`.

**Waiting on the user: the unit-vocabulary retrain**
(docs/unit_vocab_retrain_prereg_20260925.md): `obs8`'s recipe with every
reachable unit type on its own embedding row, 800 decisive games against
`obs8`, about 4.5-6 box-hours. On the current code it also carries the
player-side correction (0.7.7: the enemy's faction and villages right in
the quarter of the corpus's decisions, 42% of its games, played on maps
with a third side), and its
match cannot separate the two corrections. It also carries 0.7.12 (each
plague corpse's variation on its base type's row). Whether the corpus
corrections (BACKLOG "The imitation corpus's labels") and any of the
observation gaps (BACKLOG "What the network observes against what a
player sees") go into the same retrain or their own is the user's call.

**2. Phase 2: turn search without a pre-grader.** The turn-level gap
under the reference is RICH (7 of 60 confirmed; caveat 2026-09-26: the
confirmation replayed each turn with the screen's in-turn dice, and two
of the seven, positions 15 and 57, carry luck in the alternative's
favour, +14 and +17 HP of the mover's: re-realize them under a second
turn salt, cents on a box, before building on RICH, which stands on 7
against a bar of 6) and neither forward-only
grader passes the section-6 check (both 2026-09-23,
docs/turn_gap_ref_prereg_20260921.md), so by the design's rules the
pipeline is built from rows 1 to 6 of
docs/turn_proposer_design_20260905.md. **Found 2026-09-26: every search
and playout runs on the true state under fog**
(docs/hidden_information_20260926.md), the turn-gap grading included, so
RICH carries a second caveat and a turn search needs a determinized root,
drawn from a belief model, before it meets the 800-game gate.

**Standing, taken whenever there is room (user, 2026-09-25):**

- **Signal telemetry in every trainer.** Recorded by the self-play
  learner (`az_loop`, since 2026-09-03), the imitation trainer (0.7.0),
  the value-head fit on cached features, the value pre-training and
  `sim_self_play` (on by default, its cost in `sig_seconds`; 0.7.3;
  `tools/signal_telemetry.py`), and since 0.8.11 the self-play learner's
  full reading, gradient and update space with the cross terms, every
  iteration on 128 kept experiences (`tools/az_signal.py`,
  `<workdir>/az_signal.jsonl`; 1.67 s a probe against 1.11 s for the
  train step, a tiny network on the laptop CPU). Still without it: the
  turn-value fitter's head arm (branch `exp/turn-value`) and the
  quarantined policy anchor's rehearsal steps. Open: `az_loop`'s norm
  telemetry draws its subsample from the loop's own generator
  (`tools/az_loop.py:752`), so switching it off would change the
  held-out split and every later iteration's seed.
- **Refactor for navigation, documentation and separation of
  systems.** Library modules live in `tools/` beside one-off scripts,
  and ten files are more than three times the 600-line target (counts in
  docs/refactor_inventory_20260925.md, taken 2026-09-25). Target: one package per system
  with a README each (what it does, entry points, invariants, tests),
  scripts as thin entry points whose documented command lines keep
  working, closed backlog sections archived. Plan:
  docs/refactor_plan_20260925.md. Done: step 0 (0.7.9: paths, source-
  reading tests, shared fixtures, the unpickler, the codemod) and step 1
  (0.8.1: the game loop, the eval players, the provenance helpers, the
  az trainer recipe, the benchmark states and the actor protocol out of
  their modules; two import cycles gone) and step 3 (0.8.8: the rules
  package, `wesnoth_ai/rules/`: the terrain table, the WML state reader,
  the scenario lists, the scenario builder and the scenario .cfg reader,
  with its README; `tools/scenario_events.py` keeps the event
  interpreter). Next: step 2, the deletions the user approves, and step
  4, the simulator with the `replay_dataset` split, one system per branch.

## Open after the 2026-09-25 night crawls

Five read-only crawls (documentation, the silent-failure class, box
operations, dead code and structure, the simulator and its Rust core)
ran while the turn-value box worked; what they found and was not fixed
the same night:

- **Decision: the Rust core (`GameCore`).** It is the state of record
  nowhere (`WESNOTH_RUST_CORE` defaults off and measured no gain on eval
  or the pool, 2026-09-12), yet ten commits since have had to change it,
  and nothing has compared it with the Python applier on replays since
  phase 10 (the corpus test skips on CI, the laptop's wheel is phase 3).
  Keep it: run `scripts/diff_core_box.sh --every 1` before anyone turns
  it on, and give CI replay coverage with a few committed game records of
  our own chosen for their engagements. Retire it: `core*.rs`,
  `game_core.py`, `diff_core.py`, `test_game_core.py` and the `use_core`
  branches, about 3,800 lines. Its encoder, observation and state key
  have no production caller either way, and its encoder cannot serve a
  checkpoint with the terrain set (obs8).
- Done in 0.7.16 (Rust phase 16): the dead mirror `_move_rejected_hexes`
  deleted in both languages; `rem_euclid` for a negative start slot; the
  core's village-count invariant; the Rust observation's zone of control
  follows the engine's rule; array-length errors name the array and the
  sizes; the reach and enumeration kernels each gated on the phase they
  need (`tools/kernel_status.py` reports them apart). No measurement moves.
- Done in 0.8.0: WL_Troll_Toll finds its scenario
  (`scenario_id_of_cfg` reads the scenario tag's own `id=`, macro
  definitions left out), so its 5 corpus games load their scenario WML;
  the corpus rebuild carries it.
- Done (fix/time-area-slots, Rust phase 17): a time area keeps its own
  slot under a random start (docs/wesnoth_rules.md "A time area keeps its
  own slot"); no corpus game or self-play game moves.

- **Every eval game has a Knalgan Alliance side**
  (`scenario_pool.FORCED_FACTION`; the in-process `sim_self_play` games
  too, `az_loop`'s actors not), so every Elo number since 2026-07-04 is
  Knalgan against the six factions. Decision: keep it, or lift it for
  eval.
- **`obs8` has no self-pin on record** (terrain has +2 +- 12 at `raw:t0`,
  800 of 1,029 games decisive). Decision:
  run one on the next box, or record that none is needed.
- **The value corpus has no game of the four eval maps with a third
  side** (`build_value_corpus` counted `[side]` blocks; fixed in 0.7.7):
  a rebuild adds about 1,500 games and needs a CPU box.
- **Mid-game exports on three-side maps:** `sim_to_replay.build_save_wml`
  loops over every SideInfo and raises "no leader record for side 3" on a
  replayed three-side game, so a validation export of a mid-game start
  drawn from such a game fails (only reachable with
  `--midgame-dataset replays_dataset_imitation`, or after the value
  corpus rebuild).
- **Box scripts not on the library (0.7.11):** 34 of the 36 `scripts/*_box.sh`
  are records of past runs and keep unbounded steps, no exit trap, stops
  that accept any 2xx and `ALL_DONE`-first uploads; port any of them to
  `scripts/box/boxlib.sh` (docs/box_runbook.md) before running it again.
  The unit-vocab retrain's script is ported, with a 30-minute stall
  window since the trainer logs its resume skip (0.7.15).
- **Legacy box path:** `scripts/box_stop_on_abort.py` (the quarantined
  campaign flow) puts the Vast ACCOUNT key on the box and logs requests
  that carry it. Retire it with the user's word.
  `scripts/observation_retrain_box.sh`'s stop could have written the
  instance key to `stop.log` on a network error (a record of a past run;
  the unit-vocab script's stop does not).
- **Structure:** `replay_dataset._apply_command` (783 lines) wants one
  function per command, named like the Rust core's methods, dispatched
  through a table that raises on an unknown kind (an unknown kind falls
  through silently today); the advancement carriers want one explicit
  context (the replay-side carrier ignores a game's pick-advance
  override). Refactor step 9 (docs/refactor_plan_20260925.md).

## Reproducibility (2026-09-26 crawl; fixed in 0.7.14: eval_sim and elo_ladder dice, the pretrain probe's order)

The match path holds: each game is a function of its slot, up to the
documented bf16 batch-composition noise, and the recorded 800-game seed
bases are disjoint except the documented offset-sweep overlap. Open:
- The turn-gap confirmation's in-turn dice (docs/turn_gap_ref_prereg_20260921.md
  status line): re-realize positions 15 and 57 under a second turn salt,
  and re-realize the turn per playout in any future confirmation.
- Relaunched `sim_self_play` legs replay the first segment's setups and
  dice (`random.Random(args.seed)` and the iteration counter restart; the
  salt ignores the run seed); quarantined legs only, `az_loop` is sound.
- Quarantined VG grounding: a state's rollouts fork one sim and roll the
  same dice while their variance is treated as independent;
  `TurnCommitPolicy._ground_rng` is identical in every actor. The
  quarantined human anchor seeds each game from a salted `str` hash.
- Done: searched eval players (MCTS, TCS, plan tournament) take their
  side's per-game seed in `elo_eval_game`, the plan tournament's own
  generator included. `elo_ladder` still builds its searched player once
  for the whole ladder, unseeded.
- `scripts/endturn_offset_sweep_box.sh` still steps its seed base by 100
  between 800-game matches (a record of a past run; port it to the box
  library before running it again). `endturn_readout.py` refuses the
  independent-arms SE for arms that share (side, seed) slots and says how
  many they share; a paired per-slot comparison is not built.

## The imitation corpus's labels (2026-09-26 crawl; each changes the corpus)

A crawl of the data path from raw replays to the trainer's pairs (every
label slot checked on 23,043 pairs of 85 games, 0 mismatches; the
committed guard's own run read 0 of 22,318 pairs of 84 games) found five
problems that only a re-extraction fixes; the retrain that would carry
them is the user's decision. The corpus-wide counts below are the
crawl's and were not recorded; docs/corpus_v2_20260926.md's samples
are.

**The corrected pipeline landed in 0.8.0** (docs/corpus_v2_20260926.md;
`CORPUS_VERSION` 2; today's corpus files and records load and label as
before, 6,850 pairs identical): a move the engine stopped early is
labelled with the clicked hex when it is a legal target (5.7% of move
labels; a move whose order simply ran past the turn keeps its stop,
which was right); a game is cut where a side's turn is played by the
other side's player after a surrender or a control change
(`tools/replay_control.py`, 6.4% of winner actions on a sample); winners
come from a committed labeller (`tools/replay_outcome.py`: leader death,
then the surrendering side loses; a surrender by the side more than 5%
ahead on material is abandoned, no labels); dedup on a 30-command
prefix; label-slot guards at pre-encoding; games where the AI played a
player side are left out at build (`quarantine_ai_player_sides`: 53 of
600 sampled games, 7.9% of winner actions, the AI the labelled winner in
12). **The rebuild is a box job:** the raw replays and the dispositions
ledger (0.23 GiB) must first go to HF (`tools/stage_raw_corpus.py`), then
`scripts/corpus_v2_rebuild_box.sh` runs on the box library, about 5-7
minutes on 16 cores; the pre-encoding and the retrain follow. The script
that wrote the dispositions ledger, like the old outcomes labeller, is in
no commit.
- **The value corpus's surrender winners are inverted:**
  `tools/build_value_corpus.py` read the [surrender] command's side number
  as 1-based when it is 0-based, so all 1,334 surrender games of
  `replays_dataset/value_corpus_index.jsonl` name the surrendering side as
  the winner (fixed in 0.8.0 with a test; the index is not rebuilt; its
  readers: mid-game starts, `value_finetune`, `probe_value_head`;
  `value_head_fit` reads the imitation corpus and is not affected).

- **A move's label is where the unit stopped, not the hex clicked.**
  `extract_replay` cuts each `[move]` at the checkup's `final_hex`, and
  the label takes the last hex: 192,575 of 3,017,162 player moves
  (12,194 games; 191,697 of them in fog games) stopped short of the
  clicked hex, 2.8 hexes on average. About 82% of them stopped on
  sighting an enemy (1,437 of 1,753 in the doc's 150-game sample); the
  rest ran past the turn's movement, where the stop is the right label. The simulator does not model
  sighting interrupts (docs/wesnoth_rules.md "Replay [move] playback:
  skip_sighted"), so in our games the chosen hex is where the unit goes;
  the label should be the clicked hex, the state still the stop.
- **Play after a surrender is trained as a game.** 2,669 games keep a
  player's actions after the other surrendered or left, when one player
  controls both sides: 94,655 winner-side commands (71,672 after a
  surrender message alone; 702 games with 20 or more; the corpus total
  was not counted, and in the doc's 300-game sample the cut removes
  3,015 of 47,199 winner actions). Cut each game at a
  player's first surrender, and at a leave once the side changes hands.
- **433 surrender games name as winner the side the server says
  surrendered** (16 holdout, 66,442 winner actions), and the script that
  wrote `training/logs/replay_outcomes.jsonl.gz` is in no commit, so its
  rule cannot be audited. Commit a labeller in which the surrendering
  side loses, and treat a surrender by the side ahead on material as
  abandoned (no labels). Leader-death winners check out (200 of 200).
- **Reloaded games escape the dedup:** 9 clusters (19 games) share their
  first 30 or more commands and then diverge (the key hashes 200); none
  straddles the holdout. Cluster on a 30-command prefix, keep the longest.
- **Model input:** village gold is never observed (other than 2 in
  3,210 of 17,019 games by the crawl's count; the host's `mp_village_gold`
  alone is 3, 4, 5 or 8 in 167, training/metrics/corpus_census.json, and
  map settings such as 2p_mini_edited's village_gold=3 carry the rest);
  global feature 3, "income", holds `base_income`.


## Hidden information in the search (2026-09-26 crawl)

docs/hidden_information_20260926.md. The mask and the encoder respect
fog; MCTS, the turn-commit search, the plan tournament, the turn-gap
playouts and the turn-value playout reads run on the true state. Open:
- **Decision: the belief model** a determinized root draws from (last
  seen hexes advanced by reach, uniform over reachable fogged hexes, or
  learned from the corpus), and PIMC (one search per sampled world)
  against information-set MCTS (one world per simulation). Phase 2's
  turn search needs it before its gate.
- **Decision: principle 6's scope.** CLAUDE.md bars god view in the
  mask only; extending it to search and playouts that choose or grade
  actions (a training-time critic stays allowed) is the user's wording.
- Tag searched procedures on fogged games (for example `mcts:32+godview`)
  until the root is determinized, so their numbers are not read as fair
  strength.
- Re-grade the seven confirmed turn-gap pairs from sampled worlds once a
  belief model exists (with the second-salt re-realization of 15 and 57).
- `Observation.detached()` (`wesnoth_ai/observe.py:202-208`) keeps rows
  for hidden units; no consumer reads them. Drop them.

## What the network observes against what a player sees (2026-09-26 crawl)

docs/observation_parity_20260926.md compares `obs8`'s observation with
the 1.18.4 interface and counts each difference over 70 Ladder corpus
games (26,789 decisions). Every item changes the network's input: a
checkpoint flag, a retrain and a match each. Which of them ride with the
unit-vocabulary retrain, and which get their own arm, is the user's call.
In the order the counts suggest:
- **A unit's own weapons, resistances and traits** (161,919 of 455,569
  unit observations have weapons other than their type's; 3,254 of 5,331
  attacks involve one): about 79 unit columns, or 4 (damage and strikes
  per range) plus trait bits as a minimal form.
- **Poisoned and slowed** (4,812 decisions): 2 unit columns.
- **The fog overlay** (a "seen" hex flag) and **memory of enemies seen**
  (at 6,975 of 25,667 fog decisions an enemy seen this turn or last is
  hidden now): the flag is small; the memory is a per-side sighting record
  through the simulator, reconstruction, the Rust core and game records.
- **The relevant set's reach**: 30,648 of 142,848 hexes from which a
  visible enemy could hit an own unit next turn have no token; adding own
  units' neighbours costs 4% more hexes, the enemies' reach 52% (a
  separate, unrecorded count; the doc's Provenance section).
- **Mushroom grove and reef** have no class of their own (cave, shallow
  water): two terrain classes.
- **The time of day at a unit's hex** (668 of 1,488 decisions on
  Elensefar Courtyard in an unrecorded 4-game count; in the recorded
  census, 146 of 26,789 decisions over all maps): one per-hex column.
- **The enemy's economy with fog off** and **water villages under fog**:
  small, batch with village gold.
- **Information a player lacks:** the enemy's faction from turn 1 when the
  opponent chose Random (96 of 188 sampled player sides, unrecorded; eval unaffected);
  scenery shown on fogged hexes (recorded, not changed).

## Open after the 2026-09-25 audits

Found by five audits (rules, observation, training, evaluation, tests
and hygiene) and not fixed in 0.6.1-0.6.7.

- **Decision: a recruit onto an occupied castle hex.** The engine places
  the recruit on another castle hex and spends the gold
  (`actions/create.cpp:419-421`: an occupied location is treated as none
  given); the simulator refuses at no cost, as CLAUDE.md principle 6
  assumes. Rare (an enemy hidden inside the recruiter's own castle).
- **Decision: deletions.** About 20 fast-tier test files test quarantined
  mechanisms (`test_rewards`, `test_plan_tournament`, `test_swap_detector`,
  `test_holdout_tripwire`, the gbc and vg tests, ...); tests that restate
  the code (`test_boundary_telemetry.py:25-50`,
  `test_distributional_value.py:69-118`, `test_action_type_head.py:281-324`),
  read source text instead of behaviour, or pin defaults; dead modules
  (`wesnoth_ai/policy.py`, `wesnoth_ai/profiling.py`,
  `tools/replay_builder.py`) and about 20 one-shot probe scripts with no
  user. Each needs the user's word. Extended by the 2026-09-25 inventory
  (docs/refactor_inventory_20260925.md, section b, with the evidence for
  each): 48 entry points that nothing imports, runs or documents (10,111
  lines: probes whose results live in archived docs, corpus-rebuild tools,
  fidelity oracles worth keeping for the next fidelity bug, legacy loops
  and dashboards); `benchmarks/` (3 files named nowhere); 23 functions and
  6 methods with no caller; argparse flags no caller passes; unread
  constants and configs (`configs/replay_map_whitelist.txt`,
  `map_whitelist_1v1.json`, `vendored_addon_ids.txt`); about 29 more test
  files that test only quarantined or dead code. Moving or deleting
  quarantined code also needs a ruling on quarantine/README.md's "the code
  stays where it is".
- **A lever to measure: the attack hex.** For an attack on a unit the
  attacker is not next to, the simulator picks the hex by route cost
  (`wesnoth_sim.py:1016`), in effect the nearest; ranking by the
  attacker's defense there is a decode change, measured by a match.
- **Model input:** recruit options code lawful and neutral the other way
  round from units on the board (`encoder._alignment_value`); fixing it
  changes the observation, so it waits for a retrain behind a flag.
- **Engine rules, latent or rare:** exact counter-weapon ties (the
  engine's summation order) and the berserk prediction's 99% cutoff;
  teleport in reach and vision (0 of 207,184 corpus moves); scenario
  ability names by `id=` against the scrape's macro names; `apply_to=
  hitpoints` forms; `[modify_side] income=` as an offset; a recruit on a
  castle-village capturing it; an out-of-range weapon index replaced by 0
  without a warning; a `[time_area]` added after the Rust core is built;
  the order of turn events.
- **Live-Wesnoth observation** (eval against the built-in AI only): the
  converted state has no `_fog_cleared`, our own fogged villages lose
  their owner bit, and the time-of-day start offset is not set.
- **Training:** `az_loop` trains on at most 4,000 experiences per step
  without saying so; `--stream` publishes trial weights of the line
  search and actors accumulate boundary pairs; `value_pretrain
  --freeze-trunk` accumulates encoder gradients; loaders drop games or
  pairs at DEBUG (the replay size filter, the holdout probe and the value
  loaders report theirs since 0.7.2); the anchor caches encode the full board whatever the
  basis.
- **Evaluation:** the committed catalog mixes `raw` and `mcts:32` edges in
  one fit; the catalog's duplicate-game key depends on which label is A;
  a game's 2,000-action cap is recorded nowhere.
- **Host:** on a slow shared host the eval-style inference server is
  launch-bound (turn-value box: 25 requests per batch, 34.6 ms infer, the
  GPU 6% busy); the graphed server is the lever there
  (docs/box_specs.md "The graphed server on a quiet host").
- The box scripts of past retrains import the seeding function 0.6.7
  removed; they are records, and fail at the vocabulary step if rerun.

## Open after CI landed (2026-09-23)

- **The corpus tests never run on CI.** 33 of its 49 skips need
  replay, imitation or value data that is not in git, among them the
  SL trainer, pathfind parity and relevant-set imitation tests. A few
  extracted games committed as a fixture would run them. The repo is
  public and the corpus is other players' games, so which games (if
  any) may be committed is the user's call; the AI-vs-AI fixture
  `tests/fixtures/strict_sync_hamlets_t9.bz2` is the precedent.
- **`rust/wesnoth_core/src/encode.rs` holds six `#[test]` functions
  that nothing runs.** CI builds the wheel but does not run `cargo
  test`. With pyo3's `extension-module` feature unconditional in
  Cargo.toml the test binary probably fails to link against libpython
  (not verified; nothing here builds the crate). A `cargo test` step
  on CI settles it.

## Closed sections

The sections closed between 2026-09-12 and 2026-09-24 (fixes that
landed, phase 1's record, the time of day trained into `obs8`, the
scenario builder, the review of 2026-09-14) are in
docs/archive/backlog_closed_20260926.md, verbatim.

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

- MEASURED 2026-09-23 (docs/turn_gap_ref_prereg_20260921.md "Measured"):
  RICH, 7 of 60 confirmed, against the prediction of 1 to 3. The
  measurement against the CURRENT reference, as pre-registered: Run 1 measured the seed,
  and two of its three confirmed gaps were the base ending its turn
  early, which the reference's -1.5 offset now removes from the base.
  `tools/turn_gap.py --reference` runs the config's checkpoint under
  its decode (base turn, sampled alternatives, both playout sides;
  provenance carries the tags), the screen and the confirmation take
  the design's sequential schedule, and
  `tools/analysis/turn_gap_verdict.py` reads KILL (under 3 of 60
  confirmed) / SPARSE / RICH (6 or more). Predicted: 1 to 3 confirmed,
  so the plan's kill fires or sits on its boundary, which sends phase
  2 to the value function on human boundary states first.
  `scripts/turn_gap_ref_box.sh` runs it end to end.
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
retrain replaces the seed (ruled 2026-09-11: `relset` became the
reference); the re-pin of raw:t0 through shared inference (done
2026-09-12 and 2026-09-19).
- TEST 1 SHIPPED AS CODE (2026-09-19, autonomous window): the
  actor-level end_turn rule and the end_turn logit offset are decode
  options of the raw player (`tools/raw_player.py`, exact on the
  compact arrays and on the enumerated list), reach a match through
  `run_elo_batch --raw-end-turn-a/-b` and `--raw-end-turn-offset-a/-b`,
  and carry procedure tags (`raw:t0+endm`, `raw:t0+eo<x>`). The run
  is pre-registered against the reference player in
  docs/endturn_rule_prereg_20260919.md; `scripts/endturn_rule_box.sh`
  runs the screens, the 800-decisive match and the conditional
  attribution arm and writes the verdict (`tools/analysis/
  endturn_readout.py`). About one box-hour, $0.30-0.60 on a
  single-tenant 4090 host. Approved 2026-09-19 and RUN the same day
  (a 32-core slice of an EPYC 7B13 4090 host, $0.75/h, after three
  rentals that never came up or refused ssh and one that had no C
  linker: the script now falls back to conda's compiler and every
  failure path uploads ALL_DONE). The riders passed first: 59 Rust-
  path tests with the phase-10 wheel, `diff_core` 600 of 600 clean.
  The screen read p 0.806 +- 0.066 over 36 decisive (29-7, 4 of 40
  capped), decisions per side-turn 8.07 against 5.59 (1.44x; kill 1
  passed). **The 800-decisive match PASSED: p 0.752 +- 0.015 (602-198,
  75 capped of 875), about +193 Elo, decisions per side-turn 8.55
  against 6.03 (1.42x), capped fraction 0.09 against the reference's
  own 0.4** (docs/endturn_rule_prereg_20260919.md "Measured"). **The
  attribution arm, the end_turn logit offset -1.5, reads p 0.789 +-
  0.014 (631-169, 63 capped of 863), about +229 Elo, 1.57x the
  decisions per side-turn**: 1.7 SE of the difference above the rule,
  which the pre-registered reading calls "differs, rule-specific" (the
  commit that recorded the result rewrote the readout to read it as
  "act more"). The offset scores at least as well as the rule and is
  the simpler lever. Open for the user (ruled 2026-09-20: `raw:t0+eo-1.5`
  is the reference decode): whether `raw:t0+eo-1.5` (or a larger
  offset: the curve is still rising at -1.5, not pre-registered)
  becomes the reference DECODE (the reference checkpoint is
  unchanged), which re-pins every strength claim's opponent; and
  whether the offset helps a searched player. The offset curve's
  next points (-2.5, -4, and -99 = act while anything is legal, 800
  decisive each, seed bases 46000-46200) are pre-registered in
  docs/endturn_offset_sweep_prereg_20260919.md and RUNNING the same
  evening (`scripts/endturn_offset_sweep_box.sh`, about $0.50). The
  corpus's own rate, read before that result
  (`tools/analysis/decisions_per_side_turn.py`, 3,000 games): 7.50
  decisions per human side-turn, the winner's 9.60 and the loser's
  6.00; the reference at argmax plays 6.0 (the loser's rate) and with
  the offset -1.5 9.25 (the winner's). **The sweep RAN the same
  evening: -2.5 reads p 0.801 +- 0.014 (641-159, 1% capped), -4
  0.754 +- 0.015, -99 (act while anything is legal) 0.666 +- 0.017;
  the curve peaks between -1.5 and -2.5 (a tie within 1 SE, resolved
  toward the smaller offset by the pre-registered reading) and the
  end_turn head carries information the argmax needs** (docs/
  endturn_offset_sweep_prereg_20260919.md "Measured"). Proposed to the
  user: `raw:t0+eo-1.5` as the reference decode, -2.5 its equal with
  fewer capped games (ruled 2026-09-20: -1.5).
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

## What the imitation corpus actually contains (2026-09-21)

`tools/analysis/corpus_census.py` reads era, map layout, board size,
factions and the host's rule settings out of all 17,019 raw replay
headers (record: `training/metrics/corpus_census.json`). It settles
four questions that the manifest cannot, and each had been assumed:

- **The corpus is default-era play.** 12,157 games declare
  `era_default` and 4,862 `era_dunefolk`, but that era is
  `{ERA_DEFAULT}` plus one faction file
  (`wesnoth_src/data/multiplayer/eras.cfg:21`), and the 34,038 sides
  field only the six default factions: Loyalists 6,192, Rebels 6,077,
  Undead 5,991, Northerners 5,532, Knalgan Alliance 5,292, Drakes
  4,954. No Dunefolk side survived the corpus filter.
- **One scenario name, one board.** Each of the 36 names resolves to
  exactly one layout hash, so `2p_mini_edited` is not a map picker in
  this window, and all 23 mainline-named maps are byte-identical to
  the shipped 1.18.7 map files. There is no ladder-variant layout
  here, and "ladder map" is the wrong name for what the whitelist
  selects: 21 mainline maps.
- **Packs.** 11,457 games on the whitelist the evaluation pool plays,
  582 on mainline maps outside it (Cynsaun Battlefield 334, the
  deepest board in the corpus at 358 decisions, and Hornshark Island
  248), 4,936 mini, 44 custom.
- **Non-default rule sets cluster in the mini pack.** 782 of the
  11,457 whitelist games changed a host setting (779 turned "use map
  settings" off, 325 the experience modifier, 190 the village gold,
  184 to a random time-of-day start) against 2,113 of the 4,936 mini
  games, 2,093 of them a random time-of-day start, which the
  evaluation pool never runs. Filtering the whitelist pack to default
  settings leaves 10,675 games.

Corrected in the catalog on this evidence: docs/wesnoth_rules.md had
"Default Era uses **5** gold per village (not the historic 1)".
`mp_village_gold` is 2 in 16,712 of the 17,019 games and 5 in 26, and
every mainline 2p map that declares it declares 1 or 2.

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
- DONE 2026-09-24 (73f4c1f): **`tools/fog.py`, a complete second
  visibility implementation with ZERO importers** and its own
  ability-to-terrain table, is removed.
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

## Open after the hide-cover review (2026-09-13)

Three independent adversarial reviewers checked the hide-cover root fix
and its certification. The CODE survived: the globs are transcribed
correctly (one reviewer ported the engine's `t_translation` matcher
from the 1.18.4 tag and diffed it over 19,738 codes and every cell of
the 114 tracked maps, 0 disagreements), the Rust core's baked flags are
read by nothing but the cover predicate, and no past Elo result is
invalidated. The WRITE-UP did not, and is corrected in
docs/box_specs.md and docs/wesnoth_rules.md. What stays open:

- DONE 2026-09-20: **the hide-cover rules are verified against the
  engine itself.** `tools/hidden_units_oracle.py` drives real Wesnoth
  on a scripted board (the `ai_oracle` test scenario, terrain, units,
  fog and time of day from a setup file at prestart) and compares the
  engine's `[filter_vision]` verdicts, its chosen route, the landing
  hex, the movement left and the visible set after each move with the
  simulator's walk of the same route: 54 of 54 cases agree
  (`training/metrics/fidelity/hidden_units_oracle_20260920.json`;
  docs/wesnoth_rules.md "Verified against the engine"). The oracle
  found two bugs in the LIVE bridge on the way: the Lua collector
  reported every unit on an unfogged hex, hiders included (now
  `[filter_vision]`), and its time of day was always "morning" (it
  read a field that does not exist; now `wesnoth.schedule.get_time_of_day`).
  The next probes to add when a rule changes: `burrow` and
  `swamp_lurk` (no carrier in either pool), and the terrain codes of a
  real Ladder map in place of the grass board.
- The certification below was, until then, a no-regression test, not
  a proof of the rule. Recorded 2026-09-14 (`tools/analysis/hider_rule_sample.py`,
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
  to re-establish the baseline (done 2026-09-19: the tight self-pin,
  "Phase 1 status" above).
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

## Rulings (user, 2026-09-05; reference player 2026-09-20 and 2026-09-25)

- 2026-09-25: the reference player is `obs8` at `raw:t0+eo-1.5`
  (terrain's recipe retrained on the observation of epoch 8; +73 +- 13
  Elo over `terrain`, docs/observation_retrain_prereg_20260924.md),
  adopted on the condition that it beat the previous reference.
- 2026-09-20: the reference player is `terrain` at `raw:t0+eo-1.5`
  (the terrain-set arm decoded with the end_turn logit offset -1.5;
  `configs/reference_player.json`, `tools/reference_player.py`).
  "Moving the bar up is a good thing." Every number measured against
  `relset` at `raw:t0` stays valid as a number against that player;
  nothing is chained across the two references.

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
  CPU model EPYC or Ryzen, destroy at the end. `scripts/rent_box.py`
  does it through Vast's SDK (the `vastai` CLI binary is blocked on the
  laptop): `search` lists offers without VM hosts, `create OFFER_ID
  --onstart <script>` starts the image with an onstart that fetches the
  named script from HF `tier-b/staging/` and runs it detached,
  `status` and `start` follow an instance, `destroy` ends it. The
  retrain scripts since 2026-09-24 (`observation_retrain_box.sh`,
  `unit_vocab_retrain_box.sh`) leave ALL_DONE on HF and stop their own
  instance on every exit (`stop_self`).
- No compute on the laptop beyond sub-minute microbenchmarks.
