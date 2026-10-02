# Pre-registration: why the memory costs strength in play (2026-10-02)

Written before any box for it is rented. The reference player is `parity2`
at 64 slots (user ruling 2026-10-02); the user asked why its memory
underperforms.

## Question

In the second pass's matches (docs/parity_memory_prereg_20260929.md
"Measured, pass 2") the checkpoint at 64 slots loses to itself at 0 slots
by 37 +- 12 Elo, and at 16 slots by 30 +- 12, although the memory lowers
the holdout cross-entropy at every probe (2.852 at 64 slots against 2.935
at 0 at the last). What does the memory change in the player's decisions,
and does a decode change recover the loss?

## What the match records show

Read before this pre-registration, from the pass-2 match records (HF
`tier-b/parity_memory_pass2_20261002/games_*.tar.gz`) with
`tools/analysis/turn_tempo.py`; records in
`training/metrics/memory_in_play_20261002/`. An action is a move, an
attack or a recruit; the end_turn is not counted.

| | 64 slots | 0 slots |
|---|---|---|
| actions per side-turn, turn 3 | 7.40 | 6.73 |
| turn 4 | 7.94 | 7.39 |
| turn 6 | 8.73 | 8.94 |
| turn 9 | 9.66 | 10.38 |
| turn 12 | 9.87 | 10.49 |
| turns 6-15, mean (7,553 and 7,561 side-turns) | 9.63 | 10.22 |
| turns 6-15, chance the turn ends after 2 to 7 actions, given it reached them | 0.052 to 0.102 | 0.034 to 0.090 |
| units that never moved, at the end of turn 3 | 0.91 | 1.16 |
| at the end of turn 9 | 4.05 | 3.47 |
| at the end of turns 11-15 | 4.67 | 4.22 |
| units, end of turn 9 / turns 11-15 | 11.66 / 12.36 | 11.60 / 12.55 |
| gold left at the end of turn 2 | 17.6 | 20.1 |

The census counts decided games only. From 8 to 17 actions the two
players end their turns at the same rate (within 0.02). Units standing
next to an opposing unit without attacking are at most 0.04 per turn for
both, with no consistent difference. The 16-slot player shows the same direction
(turns 6-15: 9.82 actions against 10.39).

So the memory player is busier in the first five turns and lazier from
turn 6 on: it ends more turns after only a few actions and leaves about
half a unit more unmoved per turn, with the same army.

For scale, 2,000 human games sampled from the laptop's corpus copy: in
turns 6-15 the winners take 13.21 actions per side-turn and end a turn
after 2 to 7 actions at a rate of 0.012 to 0.043; the losers take 10.37,
at 0.035 to 0.085. `obs8` takes 7.91 against the memory player and 8.54
against itself.

## Hypotheses

- **T (tempo):** the memory raises the end_turn choice in mid-game
  positions, either by raising the end_turn prior or by flattening the
  other actions' priors (the joint argmax ends the turn when the end_turn
  prior, after the -1.5 offset, exceeds the best single action), and the
  offset, tuned on players without memory, no longer compensates. The
  loss lies in turns not played out.
- **Q (choice):** the memory changes which actions are chosen, beyond the
  end of the turn, in ways that are worse in play. One candidate mechanism:
  its state on the player's own trajectories differs from the states it
  learnt on human ones, so it reads the player's own moves as evidence of
  what a human making them would know or intend (the copycat and
  self-delusion problems of history-conditioned imitation: Wen et al.,
  "Fighting Copycat Agents in Behavioral Cloning from Observation
  Histories", NeurIPS 2020; Ortega et al., "Shaking the foundations:
  delusions in sequence models for interaction and control", 2021).

Both can hold.

## Measurements (one box)

1. `tools/analysis/memory_counterfactual.py`, the reference checkpoint at
   the reference decode (offset -1.5), every decision read with the memory
   carried at 64 slots and at 0 slots:
   - **own:** the 64-slot player's decisions in all 824 games of the
     pass-2 match against 0 slots (the trajectories the memory player
     produced);
   - **other:** the 0-slot player's decisions in the same games
     (trajectories the memory did not produce, the memory carried along
     them);
   - **human:** both sides' decisions in the corpus's holdout games
     (version 5, the training positions, the memory carried per side).
2. Two matches, PURE, 800 decisive games each, the reference's procedure
   but for the offset, the Ladder maps with factions drawn uniformly and
   assigned openly:
   - **M1:** 64 slots at offset -2.0 against 0 slots at offset -2.0
     (seed base 88000);
   - **M2:** 0 slots at offset -2.0 against 0 slots at offset -1.5
     (seed base 89000).

Readings of 1, per source and turn bucket (1-5, 6-10, 11-15, 16+):
agreement (the two choose the same action); among disagreements, the
share where only the 64-slot player ends the turn, where only the 0-slot
player does, and where both act differently; mean end_turn prior and mean
best other prior under each; on **own**, the reproduction rate (the
64-slot choice equals the recorded command); on **human**, the recorded
label's prior under each, by label kind.

## Reading

- **The tool's check:** reproduction on **own** of at least 0.95 (the
  match ran bf16 through the shared server, the tool runs bf16 one state
  at a time; near-ties can fall either way). Below it, nothing of 1 is
  read until the difference is explained.
- **T** is supported when, on **own** in turns 6 and later, the
  disagreements where only the 64-slot player ends the turn outnumber the
  reverse at least two to one and make up at least a third of all
  disagreements, and M1's deficit is smaller than -37 by at least two
  standard errors of the difference (M1 above -3 Elo); partly, when M1
  lies between -25 and -3.
- **Q** is supported when, on **own** in turns 6 and later, both-act
  disagreements make up at least two thirds of all disagreements, and M1
  stays within one standard error of -37 (below -25).
- **The memory's own trajectories:** the 64-slot-only turn-end share and
  the mean end_turn prior difference (64 slots minus 0 slots), at the same
  turn buckets, on **own** against **human**, with standard errors between
  games: a difference of two standard errors says the memory reacts to the
  player's own trajectories differently from human ones; **own** and
  **other** together against **human** says model trajectories in general.
- M2 says whether -2.0 serves the checkpoint without memory at all; it
  sets the context of M1, not a decode ruling.

## Predictions

- Reproduction on **own**: 0.98.
- Agreement on **own**, turns 6 and later: 0.80 (0.70 to 0.88).
- Share of disagreements on **own**, turns 6 and later, where only the
  64-slot player ends the turn: 0.25 (0.10 to 0.45); only the 0-slot
  player: 0.10.
- End_turn prior, 64 slots minus 0 slots, on **own** in turns 6-15:
  +0.01 (-0.01 to +0.03); on **human** at the same turns: +0.00.
- M1: -15 Elo (-40 to +5). M2: +10 Elo (-15 to +35).

## Consequences

- T: a decode for the reference at 64 slots, from an offset curve of the
  memory player, is proposed to the user; the offset is a property of a
  player, not of a checkpoint.
- Q: the remedies are in training, a design decision for the user;
  candidates are noise or dropout on the history (Wen et al. 2020), a
  memory restricted to what the opponent did (the sightings) rather than
  the player's own moves, or the memory trained on self-play trajectories.
- The reference stays `parity2` at 64 slots until the user rules
  otherwise.

## Cost

One box of the pass-2 class (RTX 4090, 32 cores): about 15 minutes of
setup and corpus build, about 35 for the two matches (run first), then 20
to 50 for the readings (16 processes sharing the card one state at a
time, the least certain figure); 70 to 100 minutes, $0.50 to $0.75 at the
pass-2 box's rate ($0.42 an hour). `BOX_MAX_H` 4.
