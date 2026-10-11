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
  update; a fine-tune from `parity3` (HF
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

Crash barriers, before the matches: the holdout imitation loss at 64 and at
0 slots within 0.05 of `parity3`'s (2.688 at 64 slots), and on held-out own
trajectories the student's agreement with its target's argmax above
`parity3`'s at 64 slots (0.749 in turns 6 and later).

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
