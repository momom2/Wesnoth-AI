# Pre-registration: why `parity3`'s memory costs it strength in play (2026-10-09)

Written before any box for it is rented. The reference player is `parity3`
at 64 slots at `raw:t0+eo-1.5` (configs/reference_player.json). The user
rules that the memory stays and that its cost in play is a design flaw to
root-cause. The 2026-10-02 investigation of `parity2`'s smaller cost (tag
`archive/exp-memory-in-play`, docs/memory_in_play_prereg_20261002.md on
that tag) found that the player computes what the trainer computed, decision
by decision (tests/test_match_memory.py).

## Question

`parity3` at 0 slots beats `parity3` at 64 slots by 105.5 +- 12.8 Elo (810
games, 800 decisive; 16 slots play as 64: 0.0 +- 12.3), although the memory
lowers the holdout policy loss (2.688 at 64 slots against 2.791 at 0;
docs/parity3_baselines_prereg_20261008.md "Measured",
docs/imitation_anneal_prereg_20261003.md "Measured"). What does the memory
change in the player's decisions, and where in the game does the loss arise?

## What the records show

Read from the 64-against-0 match records before this pre-registration
(docs/parity3_baselines_prereg_20261008.md "Measured"): the memory side
makes 9.6 decisions per side-turn (median; mean 9.8) against 11.0 (mean
11.1), and a memory player against itself reached the 200-turn cap in 220
of 1,020 games, against 10 of 810 when one side has no memory. For
`parity2` the 2026-10-02 census found the memory player busier in turns 1-5
and ending more turns after 2 to 7 actions from turn 6 on, with the same
army. The end_turn logit offset -1.5, tuned on players without a memory, is
the largest lever on record (about +229 Elo for `relset`).

## Hypotheses

- **T (tempo):** the memory raises the end_turn choice in mid-game
  positions, by raising the end_turn prior or by flattening the other
  actions' priors, and the offset -1.5 no longer compensates. The loss lies
  in turns not played out, and a stronger offset recovers it.
- **Q (choice):** the memory changes which actions are chosen within a
  turn, for the worse, consistent with the copycat and self-delusion
  problems of history-conditioned imitation (Wen et al., "Fighting Copycat
  Agents in Behavioral Cloning from Observation Histories", NeurIPS 2020;
  Ortega et al., "Shaking the foundations: delusions in sequence models for
  interaction and control", 2021). No decode recovers it.
- **L (carried):** the harm is carried across side-turns: the memory reads
  the player's own earlier play as evidence about what kind of player it
  imitates, and drifts. A memory reset at each of the side's turns
  recovers it.
- **W (the policy never acted from a losing side's memory):** the sequence
  trainer carries the memory along both game-sides and trains value and
  belief on both, but the policy on the winners' decisions only (in
  `wesnoth_ai/sequence_loss.py` a losing side's positions carry policy
  weight 0; configs/imitation.json `policy_winners_only` for the
  one-position trainer). The trunk can write "this side is losing" into
  the memory, and the policy was never trained to act from such a state; in
  play it arises whenever the player falls behind.

More than one can hold.

## Measurements (one box, `scripts/memory_in_play_parity3_box.sh`)

### Matches

Three matches against `parity3` at 0 slots at the reference decode
(`parity3_slots0`, offset -1.5), PURE, both sides raw players at argmax,
sides alternated, the Ladder maps with fog and both factions drawn
uniformly, max 200 turns, 800 decisive games each (at most 1,500
replacements), 20 persistent workers and one shared inference server, every
game recorded whole:

| arm | side A | procedure | seed base |
|---|---|---|---|
| A1 | `parity3` at 64 slots, offset -2.5 | `raw:t0+eo-2.5` | 104000 |
| A2 | `parity3` at 64 slots, its memory reset at each of its turns, offset -1.5 | `raw:t0+eo-1.5+mr` | 105000 |
| A3 | `parity3` at 64 slots, offset -3.5 | `raw:t0+eo-3.5` | 106000 |

The per-turn reset (`--memory-reset-a`, tools/raw_player.py): at the side's
first decision of each game turn the memory is the learned initial memory,
and it is carried from decision to decision within the turn only; a refused
first decision is decided again from the initial memory. The procedure tag
`+mr` keeps it out of any outdir of the plain decode, and every result
records `memory_reset_a/_b`. With the option off a seeded game is the game
the code before it played (one game compared command by command and by its
final digest, 2026-10-09; tests/test_memory_reset.py). The reset shows the
network an initial memory at mid-game positions, which training showed it at
a game-side's first decision only; A2 can therefore fail for that reason
while L holds.

Seed base S plays seeds S to S+799, and slot i's replacements S + i + k x
1,000,000. Scanned 2026-10-09 over the 110 branches and tags of the
repository: the highest seed bases named are 100000 (the baselines, which
also played 101000 and 102000) and 103000 (the material look-ahead gate),
and none of the 1,641 committed game result files carries a seed from
104000 to 106999 (the largest are 60023 and replacements from 1,020,006).

Each arm's Elo against the 0-slot player is read beside the plain decode's
-105.5 +- 12.8 (seed base 101000, the same checkpoint, opponent, horizon
and map pool), never pooled with it.

### Readings

`tools/analysis/memory_counterfactual.py`, `parity3` at the reference decode
(offset -1.5), each decision read with the memory carried at 64 slots (m)
and at 0 slots (z), bf16 on the GPU, one state at a time:

- **own:** the 64-slot player's decisions in the 810 games of the
  64-against-0 baselines match (HF
  `tier-b/parity3_baselines_20261008/games_slots64_vs_slots0.tar.gz`), the
  trajectories the memory player produced;
- **other:** the 0-slot player's decisions in the same games, the memory
  carried along trajectories it did not produce;
- **human:** both sides' decisions in the 312 holdout games of the corpus
  at `CORPUS_VERSION` 5, rebuilt from the raw replays as for `parity3`'s
  training (the sequence trainer's positions, timeouts included).

Per source and turn bucket (all, 1-5, 6-10, 11-15, 16+, and 6 and later):
agreement (the two choose the same action); among disagreements, the share
where only the 64-slot player ends the turn, where only the 0-slot player
does, and where both act differently; the mean end_turn prior and the mean
best other prior under each, and the end_turn prior difference (m minus z);
on **own** and **other**, the reproduction rate (the reading that played
chooses the recorded command); on **human**, the recorded label's prior
under each and its log-prior gain (log prior(m) minus log prior(z), per
game; a label not among the legal actions is counted and left out), by
label kind. Every agreement, disagreement and prior reading is also given by
the mover's standing and by the side's outcome:

- **Standing:** the mover's HP margin at the decision, (its units' HP minus
  the opponent's) divided by the HP of both player sides' units on the board,
  read from the state of record (units the mover does not see included);
  **behind** below -0.10, **ahead** above +0.10, **level** otherwise.
- **Outcome:** **won** or **lost** by the game's result; a game without a
  winner (capped) is read apart.

On **human** the label's prior and its gain are also given for winning and
losing sides separately, and the gain's difference, winning side minus
losing side, paired by game. Means of a game-level quantity carry the
standard error between games.

`tools/analysis/turn_tempo.py` reads the actions per side-turn by turn and
the end-of-turn hazard over turns 6 to 15 for each match and for the
baselines records. `tools/analysis/memory_counterfactual_readout.py` applies
the rules below to the rows and the three fits.

## Reading

- **The crash barrier:** the tests of the match path, the reset and the
  reader pass on the box's own wheel. A match is read only with 800 decisive
  games, else recorded as cut.
- **The tool's check:** reproduction on **own** of at least 0.95 (the match
  ran bf16 batched through the shared server, the reader runs bf16 one state
  at a time; near-ties can fall either way). Below it no row reading is read
  until the difference is explained. On one record, on the CPU in fp32, the
  reader reproduced the 64-slot player's first 40 decisions.
- Each arm **recovers** when its Elo against the 0-slot player is within one
  standard error of 0 or above, **nears** when within two but not one, and
  **trails** otherwise.
  - **T** is supported if A1 or A3 recovers.
  - **L** is supported if A2 recovers and A1 and A3 do not.
  - **Q** (within-turn choices) is supported if no arm comes within two
    standard errors of 0.
  - **Partial T:** no arm recovers, and A1 or A3 nears. **Partial L:** no
    arm recovers, and A2 alone nears. When A2 recovers as well as A1 or A3,
    the reading is T with A2 named beside it.
- **W** is supported when both hold with two standard errors between games:
  on **own** in turns 6 and later, the agreement when ahead minus the
  agreement when behind is at least 0.05 and exceeds two standard errors
  (the two per-game means' errors combined); and on **human** over all
  turns, the label's log-prior gain on winning sides exceeds 0 by two
  standard errors, and the winning sides' gain exceeds the losing sides' by
  two standard errors (paired by game). The own reading uses turns 6 and
  later because in the first turns the margin follows the order of
  recruitment: side 2 plays its first turn against a side 1 already
  recruited.
- The rows say where a supported hypothesis acts: the turn buckets, the
  disagreement types and the end_turn prior difference on **own** against
  **human** (a difference of two standard errors says the memory reacts to
  the player's own trajectories differently from human ones; **own** and
  **other** together against **human** says model trajectories in general).

## Predictions (the lead's, before any run)

Reproduction on own: at least 0.97. A1 (64 slots at -2.5): -50 Elo against
0 slots (-90 to -10). A2 (per-turn reset): -30 Elo (-70 to +10). A3 (64
slots at -3.5): -60 Elo (-110 to -10). On own trajectories in turns 6 and
later, the share of disagreements where only the 64-slot player ends the
turn: 0.30 (0.15 to 0.50); the mean end_turn prior, 64 slots minus 0 slots:
+0.02 (0.00 to +0.05); on human trajectories at the same turns: +0.00 (-0.01
to +0.01).

Under W, on own trajectories the agreement between 64 and 0 slots is lower
when the mover is behind than when it is ahead by at least 0.05, and on human
holdout games the memory's gain in the recorded label's prior is positive on
winning sides and smaller or negative on losing sides.

## Consequences

A decode that recovers the loss would not by itself answer the user's point
that a memory should not make play worse: it locates the flaw, and the
remedy belongs in training.

- **T:** the memory player's offset curve (more offsets around the best of
  A1 and A3), and why the memory ends turns earlier: the end_turn prior
  difference by turn, standing and outcome on **own** against **human**.
- **L:** a training remedy for cross-turn self-conditioning, designed after
  the readings (for example, the memory trained on trajectories of the
  player's own play, or the policy's history restricted to what the
  opponent did).
- **Q:** a training remedy within the turn (noise or dropout on the
  history, Wen et al. 2020), designed after the readings.
- **W:** a training remedy that conditions the policy on both sides'
  decisions (for example the losing side's decisions as labels at a reduced
  weight, or the policy trained on both sides with the outcome as an input
  set to "win" in play), designed after the readings.
- **Partial T or partial L:** the corresponding next step, with the arm's
  number named as the evidence it rests on.
- The reference stays `parity3` at 64 slots until the user rules otherwise.

## Cost

An RTX 4090 with at least 32 usable cores. Bring-up and tests about 10
minutes; each match 13 to 18 minutes (the baselines box: 64 slots against 0,
810 games in 750 s; a memory player against itself, 1,005 to 1,093 s); the
raw replays and the corpus about 15 minutes (10 to build on 32 cores); the
readings 20 to 50 minutes (the 2026-10-02 estimate for sources of this size,
never measured, the least certain figure: **own** holds 179,448 recorded
commands, **other** 207,081, **human** 312 games, at two forwards a
decision; the CPU check above
took 0.58 s a decision on 4 threads); tempo, readout and the final upload
about 5. About 1.5 to 2.2 box-hours, $0.62-1.40 at $0.42-0.63 an hour; 2.8
hours and $1.20-1.80 if the matches run 1.8x slower, as identical repeats
have. `BOX_MAX_H` 4. Records in HF `tier-b/memory_in_play_parity3_20261009`.
