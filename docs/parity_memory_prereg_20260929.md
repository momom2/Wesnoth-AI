# Pre-registration: the parity-memory retrain (2026-09-29)

Written before any box for it is rented. The design is
docs/parity_memory_design_20260929.md; the user's rulings behind it are
listed there.

## Question

`obs8`, the reference, sees one type row per unit and none of the unit's
own weapons, traits or statuses; it misses the fog overlay, two terrain
classes and the time of day per hex; it knows the enemy's faction from
turn 1 even when a player could not; and it has no memory of what it saw.
76 of its 190 unit types share one embedding row. Does one network trained
with all of that corrected, with a learned memory of 64 slots, play
stronger than `obs8`? And how much of its strength does the memory carry,
measured on the same checkpoint at 64, 16 and 0 slots?

## Audit gate

The run is rented only when all of these hold:

1. The Rust core's certification over the whole corpus
   (`scripts/core_certify_box.sh`) is clean: the core, which builds every
   training position, equals the Python applier after every command.
2. The four independent reviews of the data path started 2026-09-29 (the
   labels; what the network observes against a player; training against
   play; capacities and silent failures) have each finding fixed in this
   retrain or recorded as accepted by the user.
3. A targeted audit (user request, 2026-09-29) of the parts most at risk
   for the coming boxes, before any box, the certification's included:
   box operations (the scripts, the box library, renting and staging,
   after three faults on the day); whether the certification can fail
   when the core is wrong; the binding of Python views to the Rust core,
   behind every encoding and rule the pipeline asks; and the match harness
   that gives the verdict. Each finding fixed or recorded as accepted.
4. A second review, of this retrain's own code, finds the same.
5. CI is green on the commit that is staged.

## Estimand

- **Arm:** the design's recipe from scratch, one pass, run seed 20260929,
  arch 384/8/12/1536, `observation_parity` and the relevant set version 2,
  the fog gate and the terrain set on, memory 64 slots trained at nested
  sizes, `OBSERVATION_EPOCH` 11, on the corpus rebuilt at version 3.
  `scripts/parity_memory_box.sh` runs it; the run records its stage and
  code version.
- **Matches:** PURE, both sides at the reference decode (`raw:t0` with the
  end_turn logit offset -1.5), sides alternated, the Ladder maps with both
  factions drawn uniformly from the six default-era factions and assigned
  openly, the recommended path (persistent workers, shared inference, one
  server per checkpoint), max 200 turns, decisive results bought to 800
  (`--games 800 --max-extra-games 1500`), seed bases disjoint from every
  earlier match:
  1. the arm at 64 slots against `obs8` (seed base 80000): the verdict;
  2. the arm at 64 slots against the arm at 0 (81000): the memory's share;
  3. the arm at 16 slots against the arm at 0 (82000): the curve's shape;
  4. `obs8` against itself (83000): the self-pin approved on 2026-09-28.
- **Headline:** match 1's decisive-game score p with its standard error
  (0.018 at 800) and the Elo it implies. Secondary: matches 2 to 4 with
  theirs; p by the arm's faction; the capped fraction; the holdout probe at
  0, 16 and 64 slots (proxies, never verdicts); the per-phase value AUC.
- **Recorded, not read for a verdict:** the signal telemetry, the memory
  write's gradient norms, the stage timing, the manifest's counts (games,
  decisions, value states, sighting tokens, posterior errors, hidden
  units with no token).

Held fixed: `obs8` and its decode, bf16 packed serving on CUDA, one
result directory per match.

## Bars

- **Crash barrier, before the corpus:** the tests of the core, the
  encoding, the pre-encoding, the sequence trainer and memory serving pass
  on the box with the wheel built from the stage. The fresh vocabulary
  holds 190 unit types, none on the overflow row.
- **Crash barrier, the corpus:** the rebuild's dispositions account for
  every raw replay, and fewer than 1% of games fail to build.
- **Crash barrier, the pre-encoding:** fewer than 0.5% of games skipped;
  posterior errors (a seen set no faction can field) below 0.1% of
  decisions.
- **Crash barrier, the memory, after 500,000 positions:** on the holdout,
  the belief loss at 64 slots is below the belief loss at 0 slots, paired
  over game-sides, by more than two standard errors. If not, the memory is
  not remembering: the run stops and is inspected.
- **Crash barrier, the pass:** it trains exactly the pre-encoded positions.
- **Barrier, not a verdict:** holdout policy CE at 0 slots more than 0.05
  nat worse than `obs8`'s on the same holdout decisions says the recipe
  broke; investigate before reading the matches.
- **Matches are read only complete:** 800 decisive games, or the run says
  how many and the result is recorded as cut, not read.
- **Kill (match 1):** p <= 0.50. `obs8` keeps its place.
- **Pass (match 1):** p >= 0.535 (428 of 800), about +24 Elo and more than
  1.9 standard errors. The arm at 64 slots becomes the candidate
  reference, pending the user's ruling.
- **Inconclusive:** in between; recorded, and `obs8` keeps its place.
- **The memory (match 2):** p >= 0.535 says the memory adds strength;
  p <= 0.465 says it costs strength; in between, no effect was resolved.

## Prediction, before the run

- Match 1: p = 0.60 (range 0.52 to 0.68), about +70 Elo; P(pass) about
  0.75. The unit-vocabulary fix alone was predicted at +50
  (docs/unit_vocab_retrain_prereg_20260925.md); a unit's own weapons and
  statuses and the corrected labels add to it; the lower recipe risk is the
  batch change (512 positions a step against 64).
- Match 2: p = 0.54 (range 0.47 to 0.62). Most hidden units a player tracks
  were seen during the last enemy turn, which the sighting tokens give the
  arm at 0 slots too.
- Match 3: p = 0.52 (range 0.47 to 0.58).
- Match 4: p within two standard errors of 0.50.
- The belief loss at 64 slots beats the last-seen baseline at the end of
  the pass.

## Consequences for numbers

If the arm becomes the reference, every later strength claim is measured
against it at the slot count the ruling names, and nothing is chained
across references. Matches of the arm are cross-build against every
earlier number (a new observation epoch).

## Cost

On a single-tenant host with an RTX 4090, at least 32 effective cores,
64 GB of memory and 120 GB of disk:

| step | estimate |
|---|---|
| bring-up, the wheel, the tests | 0.5 h |
| the raw corpus and the version-3 rebuild | 0.3 h |
| the pre-encoding (about 5.6 million decisions, both sides) | 0.8-1.0 h |
| the pass | 11-14 h |
| five holdout probes | 0.4 h |
| four matches | 1.5-2 h |
| **in all** | **14.5-18 box-hours, about $7-11 at $0.50-0.60 an hour** |

The pass is the bulk: `obs8` trained 2.8 million pairs in about 4 hours,
and this one trains about twice as many positions (every decision of both
sides, since each side's memory must see all of its decisions), each about
1.2 times longer (memory and relevant-set tokens) and recomputed once in
the backward pass (about 1.3 times). The balance is checked before
renting; every exit, clean or not, leaves `ALL_DONE` on HF and stops the
instance. The raw corpus goes to HF first (`tools/stage_raw_corpus.py`,
0.23 GiB).
