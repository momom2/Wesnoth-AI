# BACKLOG

Live backlog for `docs/plan_20260904.md`. The pre-restart backlog
(1,055 lines of rulings and open items, 2026-05 to 2026-09-04) is
archived verbatim at `docs/archive/backlog_20260904.md`.

## NEXT (2026-10-08)

**0. The self-play program** (lead's decision 2026-10-08,
docs/selfplay_program_20261008.md): policy iteration with a one-step
look-ahead over the prior's top actions, exact combat outcomes as chance
nodes, and Muesli's clipped target; each round's tilted player gated
against `parity3` before it is distilled. Its evaluator, a critic learned
from games or short rollouts scored by material, is chosen by
measurement. **Step 1 is pre-registered** in that document and reviewed:
six critics trained on records already on HF (a size curve inside the
engine matches, an observation form, the human corpus, a small critic),
judged by paired tests against the static HP margin on the turn-value
benchmark, about $1.3-1.9 of one box. Its code is being built on
`feature/critic-step1` (from `exp/value-policy-iteration`); the box needs
the user's word. The CPU turn planner's proposal
(docs/selfplay_algorithm_design_20261007.md) is parked as step 1's kill
branch; a policy-gradient leg is parked on its cost.

**Runs awaiting the user's approval** (user, 2026-10-08: budget 500€,
about $540; no box is rented until a run is approved). In the order the
lead recommends; each states what it decides.

| run | ready | cost | decides |
|---|---|---|---|
| Q1. Step 1, the critics (`scripts/critic_step1_box.sh` on `exp/value-policy-iteration`) | yes | about 3.6 h, $1.8 expected, $2.9 at its 6 h switch | the evaluator: Pass -> Q3, Data-limited -> Q4, Kill -> Q5 |
| Q2. `parity3`'s baselines: the self-pin, 64 against 0 slots, 64 against 16 slots, 800 decisive each | yes (`scripts/parity3_baselines_box.sh`, docs/parity3_baselines_prereg_20261008.md) | about 1.5 h, $0.7 | the noise floor of every later match, and the memory's cost and size in play |
| Q7. The look-ahead player with the material evaluator against `parity3` (docs/lookahead_material_gate_prereg_20261009.md) | yes (`scripts/lookahead_gate_box.sh` on `exp/value-policy-iteration`, `LOOKAHEAD_ARMS=material:configs/lookahead_material_gate.json:103000`) | about 45 min, $0.3-0.5; can share Q2's rental | whether exact one-step look-ahead with plain material is already a teacher, and the baseline every critic gate is read against |
| Q3. Step 2's gate: the look-ahead player with the critic against `parity3`, with the material evaluator as its control | being built (`feature/lookahead-player`); after Q1 Pass | priced by the build | whether the operator is a teacher (plan rule 2) |
| Q4. 20,000 `parity3` self-play games and their critic | after Q1 Data-limited | about $3.5-5 | whether more games make the critic rank |
| Q5. The rollout evaluator in the same player: gain test, then gate | after Q1 Kill | about $5-15 | whether rollouts make the operator a teacher |
| Q6. A KL-anchored actor-critic leg from `parity3`, 50,000 games | a candidate, not built | about $30-60 | direct improvement by outcomes, the literature's most reliable route from an imitation seed; it trains on every decision, so it is bottlenecked on the network, against the ruling of 2026-10-06 |

The rest of the budget is held for scaling whichever operator passes its
gate: rounds of games, critic and distillation.

**1. `parity3` is the reference** (user ruling 2026-10-04): +82 +- 13
Elo over `parity2` (800 decisive games,
docs/imitation_anneal_prereg_20261003.md "Measured"), played with its
memory at 64 slots. Every number from here is measured against it.
**Next (user order 2026-10-04): self-play.** **User ruling 2026-10-05:
self-play keeps the memory;** its size is open to discussion, its
existence is not. It reaches every place that plays or trains the
network since 0.14.0 (docs/memory_everywhere_20261005.md). **User
ruling 2026-10-06: no self-play training on the current algorithm.**
Search-distilled self-play as `az_loop` runs it would need far more
compute than the project has; before any training launches, a new
training algorithm must handle the reward's sparsity, multi-step turns
whose plans are conditional on the dice (aleatoric outcomes inside the
turn), and the simulator's speed without being bottlenecked on the
network. That algorithm is item 0's program. The one size measurement
of the memory is `parity2`'s: 16 and 64 slots within noise in play.
Parked meanwhile:

- **Imitation's remaining gains.** The anneal rule's last estimate was
  0.061 of holdout loss a further epoch at the peak rate; a further run
  starts from `tier-b/imitation_anneal_20261003/hold2.pt`, the weights
  before the lowering.
- **The memory's cost in play.** It was 37 +- 12 Elo for `parity2` and is
  not measured for `parity3`. The investigation (branch
  `exp/memory-in-play`, now tag `archive/exp-memory-in-play`: the match records' tempo, a counterfactual reader
  and a box run the user declined) found no bug and no train/play
  mismatch: the player computes what the trainer computed, to the bit
  (tests/test_match_memory.py). The likely cause, unproven, is the
  player's own history in its memory, which training never showed it.
- **The items the audit deferred** until after the retrain:
  docs/parity_memory_audit_20260929.md "After the retrain".
- **Live games against the default AI: done (0.13.0,
  `tools/live_vs_rca.py`).** Its board check found the appliers not
  resting side 1's starting units on the game's first side turn, nor
  petrified units: fixed in 0.13.1 (records format 4). Since 0.17.0 its
  games run on the patched Wesnoth build (`tools/wesnoth_build`), which
  plays a hosted game's 70% experience (scenario-init oracle, 2026-10-07);
  the legacy eval path it superseded is deleted (0.15.5). Open: the first
  live game since then. It needs the add-on junction to point at a
  checkout that has the live stage; the primary checkout is back on
  `main` since 2026-10-08, which has it.

**Done: the turn-ranking value function FAILS** (2026-09-26,
docs/turn_value_prereg_20260925.md "Measured"): on 199 human-game
positions a linear arm on `obs8`'s trunk features reads 0.274, a head arm
0.269 and a rollout read 3 half-turns ahead 0.422 (corrected
within-position correlation; bar 0.7, kill 0.5), with the crash barrier
passed. By its rule the next proposal is a fine-tuned copy of the trunk
on the better labels; the reported readings (the HP margin after the
turn 0.397, a rollout read 7 half-turns ahead 0.532) belong in that
design. The code stays on `exp/turn-value`.

**Done inside the parity-memory retrain (2026-10-02):** the unit
vocabulary (docs/unit_vocab_retrain_prereg_20260925.md: every reachable
unit type on its own row), the corpus corrections (`CORPUS_VERSION` 5)
and the observation gaps the design corrects
(docs/parity_memory_design_20260929.md); one match measured them
together.

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
drawn from a belief model, before it meets the 800-game gate. MCTS,
the turn search and the turn-gap tool carry the memory since 0.14.0.

**3. The Rust core replaces the Python one** (user order 2026-09-28;
docs/rust_core_port_20260928.md). Done (0.9.0): units, events, the
encoding and every rule asked of a position are the core's, and it is
the state of record of the simulator, reconstruction and the pipeline
tools. Certified 2026-10-01 over the whole corpus (14,376 of 14,376
replays, clean again on phase 29;
training/metrics/fidelity/core_certify_20261001/). **Retired 0.18.0
(2026-10-09):** the Python applier, its builders, the Python rules and the
`WESNOTH_RUST*` switches are gone (+2,724 / -17,369 lines); the tools that
replay a record run on the core, and those that compared against or
debugged with the applier are deleted (the lead's decisions, tool by tool,
in the merge commit). An equivalence review played the same seeded games,
encodings, masks, record walks and pre-encodings on both sides of the
change and found them byte-identical, except the full-board masks after a
terrain change, where the old path offered moves onto friendly-occupied
hexes (Aethermaw; no checkpoint since `relset` uses the full board).
Open, small: six Rust functions Python no longer calls
(`encode_raw_streams`, `unit_reach_arrays`, `enumerate_moves`,
`observe_side`, `reach_rows`, the combat `resolve_attack`) go with the next
phase bump; 13 box scripts of past runs set the retired switches or call
deleted tools and are records only. Refactor step 4a is parked (tag
`archive/refactor-step4a-sim`).

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

- Decided 2026-09-28 (user order): the Rust core replaces the Python
  one (item 3 above). `scripts/diff_core_box.sh` is superseded by
  `scripts/core_certify_box.sh`; CI still has no replay coverage (a few
  committed game records chosen for their engagements would give it).
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

- Done (user ruling 2026-09-28): no faction is forced on eval games;
  both sides draw uniformly from the six default-era factions. Every Elo
  number from 2026-07-04 to 2026-09-27 had a Knalgan side, and a match
  under the uniform draw does not pool with them.
- **`obs8` has no self-pin on record** (terrain has +2 +- 12 at `raw:t0`,
  800 of 1,029 games decisive). Approved 2026-09-28: one rides on the
  next box.
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
- Done (user ruling 2026-09-28): `scripts/box_stop_on_abort.py` is
  retired; its escrow is `scripts/abort_escrow.py`, the launch flows stop
  boxes with the per-box key, and no provisioning places the account key
  on a box.
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
- Done with the applier's retirement (0.18.0): `Observation.detached()`
  and its rows for hidden units are gone.

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

- Done (user ruling 2026-09-28): a recruit ordered onto an occupied
  hex goes to the vacant castle hex nearest the leader, gold spent, as in
  the engine; the ordered hex is still rejected for the turn
  (docs/wesnoth_rules.md "A recruit onto an occupied hex").
- **Deletions (user review 2026-09-28, one item at a time).** Done: the
  42 files the user approved (the old policy registry, profiling,
  replay_builder, benchmarks/, four signal_profiler drivers, the
  July-September probes, smokes, dashboards and corpus one-shots). Kept:
  `download_replays` (corpus growth) and `eval_daily` (to rework, below).
  Not yet reviewed: the functions and methods with no caller, the unused
  argparse flags, dataclass fields, constants and configs, and the tests of
  dead code (docs/refactor_inventory_20260925.md section b); the review
  stopped at the file level. Quarantined code may be
  moved or archived (user ruling 2026-09-28).
- **Fidelity tools to audit before use** (user ruling 2026-09-28):
  `diff_move_final_hex`, `diff_unit_counter`, `dump_unit_states`,
  `make_strict_replay` and `check_mask_coverage` predate the September
  simulator fidelity fixes; each is to be validated before its next use.
- **`eval_daily` becomes an on-demand eval against Wesnoth's RCA AI**
  (user 2026-09-28), and the latest checkpoint is to be played against the
  RCA AI some time (live Wesnoth: a box, or the laptop with the user's word).
- **The corpus's candidate list has no committed builder:**
  `build_imitation_dataset` reads `training/logs/replay_dispositions.jsonl.gz`
  (36,309 raw replays classed by era and mods on 2026-08-07), and the script
  that wrote it is in no commit, so a replay downloaded later cannot enter
  the corpus. Commit a builder that reproduces the ledger's classes on the
  existing pool.
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
  without a warning; the order of turn events.
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

Moved 2026-10-08 to docs/archive/backlog_closed_20261008.md, verbatim:
the value head study, phase 2's prerequisite, the training-signal
panel, the cheap measurements, the corpus census, the stale-claim
sweep, the mirror audit, the lifecycle audit and the hide-cover
review. Items still open inside them: the mirror audit's notes on
what `diff_core` does not compare, `tools/mask_sim_fuzz.py` in no
test, the CUDA numerics tier that never runs on CI, and
`diff_replay._castle_network_from` duplicating the castle network;
the lifecycle audit's eval-path leaks (`WorkerPool` dropping a killed
worker unclosed, `run_elo_batch` stderr handles, `az_loop`'s untimed
probe children); `burrow` and `swamp_lurk` unmodelled.

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
