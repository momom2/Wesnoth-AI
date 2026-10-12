# Pre-registration: the memory trained on its own trajectories, round 1 (2026-10-11)

Written by the project lead before any code or box for it. It follows the
root-causing of the memory's cost (docs/memory_in_play_parity3_prereg_20261009.md
"Measured"): `parity3` at 64 slots loses to itself at 0 slots by 105.5 +-
12.8 Elo; a stronger end_turn offset recovers 65 of it and no more; the
memory changes a quarter of the decisions on human and own trajectories
alike, predicting humans better and playing worse. The memory was only ever
trained to predict a human from that human's history, never on its own play.
The user rules that the memory stays and that its cost is a flaw to fix
(2026-10-09).

## Question

Does training the memory on the memory player's own trajectories, toward
its own memoryless reading there, remove the memory's cost in play?

## The method

DAgger (Ross, Gordon and Bagnell, AISTATS 2011): the learner is trained on
the states its own play reaches, labelled by an expert. Here:

- **Own trajectories:** every game-side `parity3` played at 64 slots without
  a per-turn reset, from the recorded matches on HF: the baselines' self-pin
  (both sides) and its 64-slot sides against 16 and 0 slots
  (`tier-b/parity3_baselines_20261008/`), and the 64-slot sides of A1 and A3
  (`tier-b/memory_in_play_parity3_20261009/`): about 5,500 game-sides,
  about 1.5 million decisions.
- **Their targets:** the same network's distribution over the legal actions
  at 0 slots, raw (before the end_turn offset), computed once by a frozen
  copy of `parity3`, kept as its top 16 actions renormalised. The student's
  memory is carried along the recorded trajectory as in play; the policy
  loss at an own position is the cross-entropy of the student's raw
  distribution (memory carried) to that target. A soft raw target keeps the
  decode consistent: student and teacher both receive the -1.5 offset at
  play, so a label taken after the offset would apply it twice, and a hard
  label would not teach the margins the offset acts on.
- **Human trajectories:** the corpus at `CORPUS_VERSION` 5 under `parity3`'s
  own recipe (policy on the winners' decisions, value and belief on both
  sides, the manifest's holdout split), so the memory keeps what it predicts
  of humans.
- **Mixing and schedule:** equal numbers of own and human positions per
  update, every own position carrying its soft policy loss at weight 1 (the
  human recipe trains the policy on the winners' positions only, so own
  policy rows are about twice the human ones per update); a fine-tune from `parity3` (HF
  `tier-b/imitation_anneal_20261003/lower.pt`) through the sequence trainer,
  one pass over about 1 million own and 1 million human positions, the
  learning rate at a tenth of the recipe's peak, falling linearly to 0 over
  the pass's second half. The value loss on own positions uses their games'
  outcomes; the belief loss their true states.

What it can and cannot show: the target on own trajectories is the
memoryless reading, so the best this round can do is make the memory
harmless (the memory player plays as the 0-slot player does), not useful. A
teacher better than the memoryless player, from self-play, is what the next
round would need; this round builds the half of that pipeline that trains
the memory on its own games.

## Measurements

Two matches, PURE, both sides at `raw:t0+eo-1.5`, 800 decisive games each,
the Ladder pool with uniform factions, against `parity3` at 0 slots:

- **M1:** the tuned checkpoint at 64 slots (the memory's cost after tuning);
- **M2:** the tuned checkpoint at 0 slots (whether tuning moved the base
  policy).

Crash barriers, before the matches, all read on this run's records:
`parity3`'s and the student's holdout imitation loss at 0 slots within
0.05 of each other (the base policy intact); the student's at 64 slots no
higher than `parity3`'s at 0 slots plus 0.05 (the memory may lose what it
predicted of humans, which is the expected price, but not break the
policy); on held-out own trajectories the student's agreement with its
target, read at the decode, above `parity3`'s at 64 slots (0.749 in turns 6
and later); and the replay of the own trajectories reproducing at least
95% of the recorded decisions in turns 6 and later (the evidence that the
student's memory is carried as the player's was).

## Predictions (the lead's)

M1: -10 Elo (-35 to +15). M2: 0 Elo (-25 to +25). Agreement on held-out own
trajectories: 0.90 (0.82 to 0.96).

## Readings

- **Fixed:** M1 within one standard error of 0 or above (about -12 or
  better), and M2 within two standard errors of 0. Training on its own
  trajectories removes the memory's cost; the tuned checkpoint at 64 slots
  is proposed as the reference (it would beat today's reference by about
  the 105 Elo the memory cost); the next round replaces the memoryless
  target by a better teacher.
- **Partly:** M1 between one and four standard errors below 0 (about -12
  to -50). The direction holds and one round is not enough; a second round
  on the tuned player's own games, priced then.
- **Not fixed:** M1 more than four standard errors below 0 (below about
  -50). Training on own trajectories toward the memoryless reading does not
  remove most of the cost at this scale; the flaw sits where this round does
  not reach (the architecture of the write, or what the memory reads), and
  the next step is a design review, not another run.
- **M2 outside two standard errors of 0** qualifies every reading: the
  tuning moved the base policy, and M1 is read against M2 as well as against
  0.

## Cost

The corpus rebuild and pre-encoding as in `parity3`'s retrain (about 25
minutes); encoding the own trajectories and their targets (a GPU pass of
about 1.5 million forwards, about 10 minutes); the pass, about 2 million
positions at 4.24 ms each (`parity3`'s hold pass 1), about 2.4 hours; two
matches, about 35 minutes. About 3.5 box-hours, $1.5-2.2 at $0.42-0.63/h.

## Decision record

- Revised 2026-10-11, before any run, after the code review: the barrier
  on the 64-slot holdout loss was within 0.05 of `parity3`'s at 64 slots;
  a student that stops leaning on its memory loses up to the memory's gain
  on human labels (`parity2`: 0.08 between 64 and 0 slots), so that barrier
  would have stopped the run on the outcome it looks for. It now bounds the
  64-slot loss by the 0-slot one. The own loss's weight (1 per position),
  left unstated, is stated; the replay barrier is added.

## Measured (2026-10-11)

Box 55404472 (RTX 4090, EPYC 7702, $0.463/h), stage
`tier-b/staging/stage_20261011_own.tar.gz` (commit d7bc4cd of
`exp/memory-own-trajectories`), 14:49 to 19:16 UTC, about $2.1; records in
`training/metrics/memory_own_trajectories_20261011/`, the tuned checkpoint
and the games on HF `tier-b/memory_own_trajectories_20261011/`.

**Data.** 5,472 of 5,475 own game-sides kept (3 dropped: their decision
count did not match the match's forwards), 1,402,979 positions, 556
move-and-attack pairs merged; the replay reproduced 98.7% of the recorded
decisions in turns 6 and later (barrier 95%). The targets pass took 59
minutes against an estimated 25 (390 positions a second), the training
pass 2,000,384 positions at about 217 a second.

**Barriers, all passed.** Holdout imitation loss, `parity3` against the
student: 2.791 against 2.812 at 0 slots, 2.688 against 2.709 at 64 slots
(the memory's gain on human labels kept, about 0.10). Agreement with the
target at the decode on 277 held-out own game-sides (65,841 decisions in
turns 6 and later): the student 0.862, `parity3` at 64 slots 0.672.

**Matches** against `parity3` at 0 slots, 800 decisive games each, none
capped:

| match | Elo of the tuned checkpoint | its decisions per side-turn (opponent) |
|---|---|---|
| M1, at 64 slots | -4.3 +- 12.3 | 11.03 (10.97) |
| M2, at 0 slots | **+37.4 +- 12.3** | 11.18 (10.90) |

**Reading: Fixed, qualified by M2.** M1 lies within one standard error of
0: the memory player no longer loses to `parity3` without its memory (it
lost by 105.5), it keeps the memoryless player's tempo (9.8 decisions per
side-turn before, 11.0 now) and its games no longer stall. M2 lies three
standard errors above 0, outside the pre-registered two: the tuning made
the network itself stronger, so read against its own memoryless reading
the memory still costs about 42 Elo (M1 minus M2), down from 105. The
tuned checkpoint acts 2.6% more than `parity3` at 0 slots, too little to
carry +37 Elo near the optimal tempo; the gain is in the choice of
actions, and it came from training on the network's own trajectories.
Both readings are first measurements and are confirmed on fresh seeds
before anything is built on them.
