# Pre-registration: the reference's recipe with every unit type on its own row (2026-09-25)

Written before the box is rented, on the template of
docs/observation_retrain_prereg_20260924.md. The defect was found by two
independent audits on 2026-09-25 and verified the same day.

## Question

The reference player, `obs8` at `raw:t0+eo-1.5`
(`configs/reference_player.json`), has a vocabulary of 356 unit-type
names and a type embedding of 200 rows. Every name whose id reaches 199
is clamped onto the last row, so 157 names share one embedding: 76 of the
190 unit types that can take part in our games, among them 6 of the 7
Northerner recruits, 4 of the 7 Undead recruits, and Spearman, Thief,
Poacher, Wose, Merman Hunter and both Saurians. A Northerners position
encodes its recruits as `[199, 199, 199, 199, 199, 199, 153]`. The unit
features carry hit points, moves, experience, cost and alignment but no
attacks, resistances or abilities, so the network has no input that
tells an Orcish Assassin's poison from a Troll Whelp's regeneration. The
same holds for `terrain`, `relset` and seed2's lineage: every network
trained from scratch since the vocabulary was seeded from all of
`unit_stats.json` (2026-09-09). Does the same recipe, with every
reachable unit type on its own row, play stronger than `obs8`?

The arm carries a second, small correction by the user's standing order
to batch retrains: the imitation pairs no longer include the neutral
side's commands (its AI's, about 0.9% of the commands of a 1/13 corpus
sample, almost all on the mini maps), which the pre-encoded path trained
as winner targets for side 2 and the serial path as loser states. The
match cannot separate the two; the first is expected to dominate.

## Estimand

- Arm: `obs8`'s recipe from scratch on `main` as staged at rental (0.7.0
  or later; the run records its stage and version)
  (`tools/unit_vocab.py`: a fresh vocabulary of the 190 reachable unit
  types in name order, none on the overflow row; player-side imitation
  pairs only), `scripts/observation_retrain_box.sh` to the letter
  otherwise: the relevant-set basis, `--terrain-multi-hot`,
  `configs/imitation.json`, batch 64, lr 1e-4, cosine over 4 epochs
  stopped after one pass, the deduplicated corpus with the manifest
  split, fog gate on, pre-encoded records, run seed 20260909, arch
  384/8/12/1536, `OBSERVATION_EPOCH` 9 or later. `scripts/unit_vocab_retrain_box.sh`
  runs it. The code also carries the engine rules landed with it
  (0.6.6): the corpus reconstructs 16 games with their declared zero
  village economy and keeps a levelling unit's trait movement, and both
  players of the match choose counter weapons by the engine's level-up
  rule.
- Opponent: `obs8` as configured, served through its own inference server
  in its own encoding (its 356-name vocabulary and its clamp).
- Match: PURE, both sides at the reference decode (`raw:t0` with the
  end_turn logit offset -1.5), sides alternated, Ladder maps with a
  Knalgan side in every game as every earlier match, the recommended path
  (persistent workers, shared inference, 20 workers on a 4090), max 200
  turns, seed base 70000 (disjoint from every earlier match), decisive
  results bought to 800 (`--games 800 --max-extra-games 1500`).
- Headline: the arm's decisive-game score p with its standard error
  (0.018 at 800) and the Elo it implies. Secondary: p by the faction of
  the arm's side (Northerners and Undead lose the most rows), the capped
  fraction, the holdout probe at the end of the pass (proxies, never
  verdicts), the per-phase value AUC.
- Recorded, not read for the verdict: the trainer's signal telemetry
  (`arm_signal.jsonl`, tools/signal_telemetry.py `ImitationSignal`):
  every 25,000 pairs, each loss term's share of the encoder's, the
  trunk's and the heads' gradient and of AdamW's update, and the
  steps' gradient norms, from a probe built to leave the training
  unchanged (bit-identical on CPU in tests; training on CUDA is not
  bit-reproducible in any case) whose cost each row records (estimated
  0.5-0.7% of the pass); and the stage timing (`arm_prof.json`). The
  box's stage is rebuilt from `main` at rental: the stage the script
  names by default (`stage_20260925u`) predates the telemetry.

Held fixed: `obs8` and its decode, bf16 packed serving on cuda,
combat-oracle alphas 0, one result directory for the pair.

## Bars

- **Crash barrier, before the pre-encoding**: the tests of the unit
  vocabulary, the imitation pairs, the time of day, the terrain set,
  Rust encode parity, the core, vision and Rust observation run on the
  box with the wheel built from the staged source. The fresh vocabulary
  must hold 190 unit types, none on the overflow row. A failure stops
  the run before anything trains.
- **Crash barrier, the pass**: it trains between 2,770,000 and 2,826,147
  pairs: `obs8`'s count less the neutral side's pairs (about 0.9% of the
  commands, fewer of the trained pairs). Outside that the recipe drifted
  and the match is a different comparison.
- **Barrier, not a verdict**: holdout CE more than 0.05 nat worse than
  `obs8`'s on the same pairs (2.822) says the recipe broke; investigate
  before reading the match.
- **The match is read only complete**: 800 decisive games, or the run
  says how many and the result is recorded as cut, not read.
- **Kill**: p <= 0.50 at 800 decisive. `obs8` keeps its place.
- **Pass**: p >= 0.535 (428 of 800), about +24 Elo and more than 1.9
  standard errors. The arm becomes the candidate reference, pending the
  user's ruling.
- **Inconclusive**: in between; recorded, and `obs8` keeps its place.

## Prediction, before the run

p = 0.57 (range 0.51 to 0.65), about +50 Elo; P(pass) about 0.65.
Reasoning: a third of the ladder recruits and most of two factions'
line-ups become distinguishable, and 2.8M pairs are enough to learn a
type's row (the other 114 reachable types learned theirs in the same
pass). Against that, the network already separates many of these types
by their hit points, moves and cost, which differ between most recruits
of one faction, and every match game has a Knalgan side, whose recruits
lose only two rows. p by faction should move most for the arm playing
Northerners or Undead. The holdout CE moves by less than 0.03.

## Consequences for numbers

If the arm becomes the reference, every later strength claim is measured
against it, and nothing is chained across the two references.

## Cost

As `obs8`'s retrain ran on 2026-09-24/25 (6.4 box-hours with a cut and
a resume, $4.3; an uncut run is about 4.5 box-hours): bring-up and wheel
5-10 minutes, the tests a few minutes, pre-encoding about 20 minutes,
one pass of about 2.8M pairs 3-5 hours at 153-221 pairs/s, the per-phase
evaluation 5 minutes, the 800-decisive match 15-30 minutes (cut at 60).
About 4.5 box-hours, $2.5-3.6 at $0.55-0.80 per hour, on a single-tenant
host with an RTX 4090, at least 32 effective cores, 64 GB of memory and
60 GB of disk. The balance is checked before renting. Every exit, clean
or not, leaves ALL_DONE on HF and stops the instance.
