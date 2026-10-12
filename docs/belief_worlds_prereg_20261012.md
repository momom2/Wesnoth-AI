# Pre-registration: the rollout re-grade in belief worlds (2026-10-12)

Written by the project lead before any code or box for it. The step the
re-grade's Fail pre-registered (docs/rollout_regrade_prereg_20261012.md
"Measured"): a belief over the hidden units, with the rollouts re-read on
worlds drawn from it.

## Question

Rollouts on the true state select better turns than the base turn (+0.049
+- 0.017, `obs8`'s rollouts); the same rollouts in the mover's observed
world, every unit it cannot see removed, do not (-0.002 +- 0.018, `own2`'s).
How much of that gap does a belief a fair player can hold recover, and what
part of the hidden units carries it: that they exist, or where they stand?

## What is fixed from the re-grade

The benchmark (199 positions, 966 candidate turns, the truth of
`training/metrics/turn_value_20260925/bench_truth.npz`, luck-adjusted basis
primary), the candidates' rebuild and its checks, the rollouts (8 per
candidate, both sides `own2` at 0 slots at `raw:t0.5+eo-1.5`, to the fourth
player turn start after the candidate), the reads and the statistics
(`wesnoth_ai/turn_bench_stats.py`). A rollout's dice come from the
re-grade's salt for that candidate and rollout in every arm; a placement
draws from its own stream.

## The worlds

Every world is built from the state right after the candidate's end_turn
(the opponent's turn begun). "The estimates" are the observed world's rules
for what the mover cannot see besides units (`wesnoth_ai/lookahead_world.py`):
villages owned as the mover's display shows them, and the opponent's gold
and base income set to the mover's, except that the mover's gold is read at
the start of its turn, before the candidate, so every candidate of a
position shares it. The re-grade read it after the candidate, so a
candidate that recruited more also lowered the opponent's assumed gold.

| arm | the opponent's units the mover cannot see | the rest |
|---|---|---|
| T, true | where they are | true |
| TU, true units | where they are | the estimates |
| RB, roster at belief hexes | the same units, placed by the belief head | the estimates |
| RU, roster at uniform hexes | the same units, placed uniformly | the estimates |
| LS, last seen | each one the mover saw since its last end_turn, at the hex where it last saw it; the others removed | the estimates |
| O2, none | removed | the estimates |

Units of the scenario's other sides that the mover cannot see are removed
in every arm but T, as in the observed world. The opponent's fog is
recomputed from its units of the world and its sighting record emptied, as
in the observed world, in every arm but T.

- **Placement (RB, RU).** Each rollout draws its own placement. A hex
  qualifies for a unit when the mover does not see it, it holds no unit and
  it is not impassable for that unit. The opponent's leader, when hidden,
  goes first to its side's starting position when that hex qualifies (a
  player knows where the enemy leader starts); otherwise it is placed as the
  others. RB reads the belief head (`own2` at 0 slots, at the candidate's
  end_turn decision; its memory adds little to it: holdout belief loss
  0.01582 at 0 slots, 0.01555 at 64, the last-seen baseline 0.0490) as what
  it was trained to be, each hex token's own probability that a hidden unit
  stands there: every token hex is occupied with its probability, drawn
  independently; the hidden units still unplaced, in a random order, take
  the occupied hexes that qualify for them, in a random order, and those
  left over take RU's rule over the hexes without a token, counted. RU
  draws each unit's hex uniformly over every hex of the map that
  qualifies.
- **Last seen (LS).** The units of the mover's sighting record at its
  end_turn decision, each with its own record (the mover saw it, its type
  and traits with it) and the hit points it was seen with, at the hex of the
  sighting when that hex is one the mover does not see and holds no unit;
  otherwise left out, counted. The opponent's leader, when hidden and not
  placed so, goes to its side's starting position when that hex qualifies,
  as in RB and RU.

## Statistics

Per arm, the primary grader, the HP margin at read 2: its selection gain
over the base turn on the luck-adjusted truth. Between arms, paired
differences of those gains, bootstrapped over positions:

- TU - O2: what knowing the hidden units is worth to these rollouts (the
  ceiling a belief can recover);
- T - TU: what the estimates of gold and villages cost;
- RB - RU: whether the belief head knows where the units are;
- RU - O2: whether knowing what exists is enough;
- LS - O2: what the record of last sightings recovers.

Reported, not judged: the other three reads per arm, each arm's agreement
with TU's pick, the raw basis, the placements that fell back or left a unit
out, and the positions where the opponent's leader is hidden.

## Predictions (the lead's)

TU - O2 +0.040 (+0.010 to +0.070); T - TU 0.000 (-0.020 to +0.020);
RB - RU +0.015; RU - O2 +0.010; LS - O2 +0.010; O2's own selection gain
+0.005.

## Readings

- **Unresolved ceiling:** TU - O2 not above 0 by two paired standard
  errors. The hidden units' worth to these rollouts is not resolved on this
  benchmark; nothing is built on the decomposition, and it is recorded.
- Otherwise, each of these that holds:
  - **Last seen suffices:** LS - O2 above 0 by two standard errors. The
    sighting record is belief enough for rollouts; step B (a rollout player
    gated against `own1`) is priced on last-seen worlds.
  - **The belief head knows where:** RB - RU above 0 by two standard
    errors. A fair roster sampler (how many units and of which types, from
    the observation) is built next and re-graded the same way.
  - **What exists, not where:** RU - O2 above 0 by two standard errors and
    RB - RU within two. The belief to build is a roster model.
  - **None of the three:** the rollouts need the units' exact hexes; a
    search under fog is capped at these depths, and the program's next
    design goes to the user with the tactical engine.

## Cost

Six arms of about 7,700 rollouts at the re-grade's rate (540 s an arm on
28 workers): about 55 minutes of rollouts and 15 of bring-up and tests,
about $0.5 at $0.37-0.46 an hour. BOX_MAX_H 2.5.

## Decision record

- Revised 2026-10-12, before any run, on reading the implementation: hex
  tokens cover only the part of the map near the mover's units (426 of 984
  hexes on the first fixture position), so drawing every hidden unit's hex
  in proportion to the belief over token hexes put units that stand far
  away next to the mover's; the head's probabilities are per hex, so each
  token hex is now occupied with its own probability and the units left over
  go to the hexes without a token. A hidden leader placed anywhere on the
  map could not recruit; it now starts from its side's starting position in
  every arm that places units by an estimate.

## Measured (2026-10-12)

Box 55588563 (RTX 4090, EPYC 7742, $0.371/h), stage
`tier-b/staging/stage_20261012_belief.tar.gz` (commit b3d4e77 of
`exp/belief-worlds`, CI green), 17:15 to 18:22 UTC, about $0.45; records in
`training/metrics/belief_worlds_20261012/` (`belief_readout.md`,
`belief_readout.json`, the six arms' summaries), the rollouts on HF
`tier-b/belief_worlds_20261012/`. Every arm graded all 966 candidates
(7,728 rollouts an arm, 575 to 685 s each), no failure.

**Reading: Unresolved ceiling.** TU - O2 is +0.015 +- 0.015 for the
primary grader (predicted +0.040): what knowing the hidden units is worth
to these rollouts is not resolved on this benchmark, and nothing is built
on the decomposition.

The primary grader, the HP margin at read 2, its selection gain over the
base turn on the luck-adjusted truth: T +0.033 +- 0.016, TU +0.013 +-
0.017, RB +0.035 +- 0.018, RU +0.021 +- 0.018, LS +0.011 +- 0.017, O2 -0.002
+- 0.018 (the re-grade's observed world, its gold read after the
candidate: -0.002 +- 0.018). Paired: T - TU +0.021 +- 0.010, RB - RU +0.014
+- 0.013, RU - O2 +0.024 +- 0.016, LS - O2 +0.013 +- 0.015.

Reported, not judged: the value head at read 2 resolves the ceiling, TU -
O2 +0.037 +- 0.015 (T +0.050, TU +0.044, RU +0.038, RB +0.024, LS +0.022,
O2 +0.008), and there RU - O2 is +0.030 +- 0.016 and RB - RU -0.014 +-
0.014. Each arm's agreement with TU's pick: 0.61 to 0.62 for T, RB, RU and
LS, 0.46 for O2. The opponent's leader was hidden at 135 of the 198
positions; in RB and RU it went to its start in 4,288 of the 4,368
rollouts where it was hidden. RB's occupancy draw left 8,563 of 22,824 units
to the hexes without a token.

**What it says.**
- With today's rollout player the true-state ceiling is itself small,
  +0.033 +- 0.016. On 198 positions, the paired standard errors are
  0.010-0.016, so no decomposition of it resolves.
- The gold the re-grade read after the candidate did not cause its Fail:
  O2 and the observed world read the same.
- At point estimates, every world that holds the opponent's units ranks
  above the world without them, under all four graders.
- Where the value head resolves the gap, placing the true roster uniformly
  recovers most of it and the belief head's hexes add nothing: what exists
  matters more than where it stands. That is a reading of an unjudged
  grader, recorded so that a future belief model starts from the roster.
- A rollout teacher's whole upside on this benchmark, even seeing through
  fog, is a few hundredths of a game result per turn chosen among five.
