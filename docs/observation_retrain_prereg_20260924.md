# Pre-registration: the reference's recipe retrained on the current observation (2026-09-24)

Written before the box is rented, on the template of
docs/terrain_multi_hot_prereg_20260919.md. Approved in principle by the
user on 2026-09-24 ("Retraining the imitation seed: Approved"), batched
as the user asked on 2026-09-22 ("we're not retraining after every
single bug").

## Question

The reference player, `terrain` at `raw:t0+eo-1.5`
(`configs/reference_player.json`), was trained on 2026-09-19 at
`OBSERVATION_EPOCH` 3. Since then the observation of the imitation
corpus changed four times (`wesnoth_ai/constants.py`):

| epoch | change | what the reference now sees that it never trained on |
|---|---|---|
| 4 | this turn's and next turn's lawful bonus as global features | nothing: it loads through a zero pad and ignores them |
| 5 | statues carry their scenario modifications (1 hp, no moves) | 1-hp statues on 3 of the 21 ladder maps |
| 6 | vision as the engine keeps it, not a disc of radius max moves | a different set of seen hexes at 30,814 of 31,137 decisions of 92 fogged corpus games; 19,439 enemy units shown that the disc hid and 3,341 hidden that it showed, of 309,711 (`training/metrics/fidelity/vision_rule_census_20260924.json`) |
| 7 | a revealed hider hides again at its side's turn start in reconstruction | 804 of 66,873 decisions of 206 corpus games |

Epoch 8 (the neutral side's end of turn) leaves reconstructed corpora
unchanged. So the reference plays every match on observations drawn
from a distribution it did not learn, and cannot see the time of day
at all. Does the same recipe, trained on the current observation, play
stronger than the reference?

The arm changes the observation in two ways at once, by the user's
order to batch retrains: the corrections (epochs 5 to 7) and the new
time-of-day input (epoch 4). The match cannot attribute a result to
either; an attribution arm would cost another run and is proposed only
if the result is surprising.

## Estimand

- Arm: the reference's recipe from scratch on the current code
  (`main` at 0.5.1, `OBSERVATION_EPOCH` 8), `scripts/terrain_arm_box.sh`
  to the letter otherwise: the relevant-set basis, `--terrain-multi-hot`,
  `configs/imitation.json`, batch 64, lr 1e-4, cosine over 4 epochs
  stopped after one pass, the deduplicated corpus with the manifest
  split, fog gate on, pre-encoded records, run seed 20260909, arch
  384/8/12/1536. `scripts/observation_retrain_box.sh` runs it.
- Opponent: the reference player as configured, served through its own
  inference server in its own encoding.
- Match: PURE, both sides at the reference decode (`raw:t0` with the
  end_turn logit offset -1.5), sides alternated, Ladder maps, the
  recommended path (persistent workers, shared inference, 20 workers on
  a 4090), max 200 turns, seed base 64000 (disjoint from every earlier
  match), decisive results bought to 800 (`--games 800
  --max-extra-games 1500`).
- Headline: the arm's decisive-game score p with its standard error
  (0.018 at 800) and the Elo it implies. Secondary: the capped fraction
  per side, the same read with capped games scored 0.5, the holdout
  probe at the end of the pass (CE, actor top-1, masked target CE,
  value AUC; proxies, never verdicts), the per-phase value AUC, and p on
  the two maps whose turn number misleads about the time of day
  (Fallenstar Lake, Ruined Passage; descriptive, too few games for a
  verdict).

Held fixed: the reference checkpoint and decode, bf16 packed serving on
cuda, combat-oracle alphas 0, one result directory for the pair.

## Bars

- **Crash barrier, before the pre-encoding**: the time-of-day, terrain
  set, Rust encode parity, core, vision and Rust observation tests
  (`tests/test_time_of_day_features.py`, `test_terrain_multi_hot.py`,
  `test_rust_encode_raw.py`, `test_game_core.py`, `test_vision.py`,
  `test_rust_observe.py`, both tiers) run on the box with the wheel
  built from the staged source, whose phase must equal the one
  `rust/wesnoth_core/src/lib.rs` declares. A failure stops the run
  before anything trains.
- **Crash barrier, the pass**: it trains within 1% of the terrain arm's
  2,826,147 pairs. Outside it the recipe drifted and the match is a
  different comparison.
- **Barrier, not a verdict**: holdout CE more than 0.05 nat worse than
  the reference's on the same pairs (2.837) says the recipe broke
  rather than the observation failing; investigate before reading the
  match.
- **Kill**: p <= 0.50 at 800 decisive. The reference keeps its place;
  the corrections and the time of day are not worth a retrain at this
  recipe.
- **Pass**: p >= 0.535 (428 of 800), about +24 Elo and more than 1.9
  standard errors. The arm becomes the candidate reference, pending the
  user's ruling and a self-pin.
- **Inconclusive**: in between; recorded, and the reference keeps its
  place.

## Prediction, before the run

p = 0.56 (range 0.50 to 0.63), about +40 Elo; P(pass) about 0.6.
Reasoning: the vision correction moves about 7% of the enemy units the
reference sees in fogged games (mostly units shown that it never saw
in training), which a one-pass imitation product may handle poorly
out of its distribution, and the time-of-day arm was predicted at 0.55
alone (docs/time_of_day_prereg_20260922.md). Against that, a network
reading its units' positions may be robust to which fogged enemies it
sees, and the reference already plays well enough to hold +263 over
its predecessor under the old observation. The holdout CE moves by
less than 0.03 either way; decisions per side-turn move by less than
5%, since this is an observation change, not a decode change.

## Consequences for numbers

If the arm becomes the reference, every later strength claim is
measured against it, and nothing is chained across the two references.
Matches run on the current code are cross-build against every match
before epoch 6 regardless of this run (CLAUDE.md, 2026-09-24).

## Cost

As the terrain arm ran on 2026-09-19 (instance 51597775, 4.5
box-hours, about $3.5): bring-up and wheel 5-10 minutes, the tests a
few minutes, pre-encoding 17,019 games at 30 workers about 20 minutes,
one pass of about 2.83M pairs about 3.5 hours at 221 pairs/s, the
per-phase evaluation 5 minutes, the 800-decisive match 15-30 minutes
(cut at 60). About 4.5 box-hours, $2.5-3.6 at $0.55-0.80 per hour, on a
single-tenant host (`vms_enabled=false`) with an RTX 4090, at least 32
effective cores, 64 GB of memory and 60 GB of disk. Every exit, clean
or not, leaves ALL_DONE on HF.

## Measured (2026-09-24/25, instance 52448275: a 64-core EPYC 7B13 host with an RTX 4090, $0.67/h)

Records: `training/metrics/bench_pipeline/observation_retrain_20260924/`
(the match's fit and log, the training log and holdout curve, the
per-phase value table, the box's logs). The arm's checkpoint is HF
`tier-b/observation_retrain_20260924/arm_epoch0.pt`. The 800 games'
result files and whole-game records stayed on the stopped instance's
disk.

| match | seed base | games (capped) | decisive | Elo of the arm | p | wall |
|---|---|---|---|---|---|---|
| obs_e1 vs terrain, PURE, both at `raw:t0+eo-1.5`, current observation against the reference's | 64000 | 800 (0) | 800 | +73 +- 13 | 0.604 +- 0.017 (483-317) | 491 s |

**PASS** under the pre-registered bar (p >= 0.535; predicted 0.56, range
0.50 to 0.63): the reference's recipe trained on the current observation
beats the reference by about 73 Elo. Under the pre-registered reading
the arm is the candidate reference, pending the user's ruling and a
self-pin. The result cannot say how much of it is the time of day and
how much the corrected vision, statues and hiders (batched by the
user's order); no attribution arm is proposed unless the user wants one.

Crash barriers: the tests before the pre-encoding passed (41 passed, 1
skipped: `wesnoth_src/data/core/units` is not staged); the pass trained
2,826,147 pairs, exactly the terrain arm's count, over 16,650 files
with no file error, flush failure or out-of-memory split. Holdout
proxies at the end of the pass, against the terrain arm's: CE 2.822
against 2.837, actor top-1 0.595 against 0.584, masked target CE 1.342
against 1.350, value AUC 0.732 against 0.743; per-phase same-turn value
AUC 0.656, 0.725, 0.792, 0.840, 0.787, 0.864 against 0.645, 0.733,
0.783, 0.833, 0.792, 0.864. As in every earlier arm, the proxies do not
see the match. No game of the 800 reached the 200-turn cap. Not
measured: the two maps whose turn number misleads about the time of day
and the decisions per side-turn, which need the game files on the
stopped instance.

**The run was cut and resumed.** At 1.79M pairs Vast stopped the
instance because the account's credit ran out (balance -$0.19 against
its -$0.01 threshold). The trainer's `--resume` restarted an epoch in a
new order, which would have broken the one-pass recipe; the trainer now
continues a cut pass (`fix/exact-resume`, tools/supervised_train.py
`PassPosition`, tested to bit-identical weights on a cut-and-resumed
small run). The box's checkpoint predated that change, so it resumed
through the first-epoch path: the same file order from the seed, the
1,792,000 pairs already trained read again and skipped in 2,936 s,
training continued at step 28,000, and the pass ended at the same pair
count as the terrain arm. Two things differ from an uncut run: the
dropout draws (probability 1e-4) after the cut, and the holdout
evaluation points after it. The restarted box re-staged its code, and
at the end stopped itself through Vast's API (`stop_self`, 00:20:51 UTC).

Cost: 3.9 box-hours for the first run and 2.5 for the resumed one, 6.4
in all, about $4.3 at $0.67/h, against the pre-registered $2.5-3.6: a
slower pass than the terrain arm's (153-160 pairs/s against 221 on the
same host class), the 49-minute skip and the restart's bring-up.

