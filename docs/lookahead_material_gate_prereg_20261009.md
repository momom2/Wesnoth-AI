# Pre-registration: the look-ahead player with material against `parity3` (2026-10-09)

Written before any box for it is rented: run Q7 of BACKLOG.md's queue. The
player is step 3 of docs/selfplay_program_20261008.md with its cheapest
evaluator; its code is on `exp/value-policy-iteration`
(`tools/lookahead_player.py`, built on `feature/lookahead-player`), the box
script `scripts/lookahead_gate_box.sh` on the same branch.

## Question

Does `parity3`, choosing at each decision among its eight most probable
actions by exact one-step look-ahead with the HP margin as the value, beat
`parity3` playing its own argmax?

## Why now

- The program's operator needs an evaluator; step 1 (run Q1) chooses between
  a learned critic and rollouts. Material is the evaluator every critic is
  judged against, and its gate costs cents, so its number is the baseline the
  critic's gate is read against.
- It runs the whole gate pipeline (the observed-world determinization, the
  exact outcome states, the per-side match flags, the telemetry) in 944 real
  games before any critic's gate spends more.
- The exact-outcome studies of 2026-09 tested dominance only: an attack that
  beats the prior's on every count, 0.98 per 1,000 decisions (tag
  `archive/exp-xod-dominance`). The tilted choice also takes trades, which
  nothing measured.

## The arm

Side A: `parity3` at 64 slots through the look-ahead player with
`configs/lookahead_material_gate.json`: the material evaluator (tanh of the
mover's HP margin over the two player sides divided by `hp_scale`), k 8, c 1,
sigma 0.1, every decision kind, the observed world (the deciding side's
unseen enemy units removed, the opponent's gold and base income set to its
own). The configuration is the player's default, untuned. Side B: `parity3`
at 64 slots, raw, at the reference decode. PURE, 800 decisive games, sides
alternated, the Ladder pool with fog and uniform factions, seed base 103000
(disjoint from every earlier match), every game recorded whole.

The null control is the tested one: at c 0 the player plays the prior's
argmax decision for decision (tests/test_lookahead_player.py), so no match
is spent on it.

## Predictions (the lead's, before the run)

p between 0.44 and 0.52: a value read right after one action sees the
material a fight trades and none of the position it leaves, and the prior
already chooses its attacks well. The operator changes 3-10% of decisions,
most of them attacks and end_turn.

## Readings

- **Teacher:** p at 0.535 or above (about +25 Elo, two standard errors).
  Confirmed by 800 decisive games on a disjoint seed base before anything
  depends on it; then it is a teacher by plan rule 2 at almost no cost, and
  its distillation is pre-registered next.
- **Neutral:** p between 0.465 and 0.535. Material neither helps nor harms
  at this tilt; a critic's gate must beat this arm's p.
- **Harm:** p at 0.465 or below. A myopic evaluator at c 1 costs strength;
  the critic's gate keeps c but its reading is compared with this arm's p,
  and the telemetry's flips by kind show which decisions the evaluator gets
  wrong.

Reported in every case: the flip rate by kind and the kind played instead,
evaluator states and seconds per decision, failed expansions by reason, the
capped share.

## Cost

Bring-up and the look-ahead and match tests about 15 minutes; 50 decisions
timed on the box; the match about 23 minutes (the 944-game raw match of
2026-10-04 took 929 s at 20 workers, and the look-ahead adds about 35 ms of
worker CPU per decision, 268 decisions per game side). About 45 minutes,
$0.32-0.47 at $0.42-0.63 an hour. It can share a rental with run Q2.

## Measured (2026-10-09)

Box 54996224 (RTX 4090, EPYC 7B13, $0.432/h), stage
`tier-b/staging/stage_20261009_q7.tar.gz`, records in
`training/metrics/lookahead_material_gate_20261009/` (games on HF
`tier-b/lookahead_material_gate_20261009/games_material.tar.gz`). 991 games
in 1,569 s, 800 decisive: the look-ahead side won 390 and lost 410, 191
games reached the turn cap (19%).

**Neutral: p 0.4875 +- 0.018, -8.7 +- 12.3 Elo** for the look-ahead player
against `parity3` (`elo_collect`, PURE). Inside the predicted 0.44-0.52.

The operator ran at 99% of decisions and changed 4.1% of them (predicted
3-10%), but not where predicted: of 10,632 changed decisions, 5,817 became a
recruit (2,573 a move, 2,219 an attack, 23 end_turn), taken from moves
(5,171), end_turns (3,102), attacks (1,287) and recruits (1,072). A recruit
puts its hit points on the board at once, so the HP margin read right after
it rewards recruiting regardless of the position; the material evaluator
almost never prefers ending the turn. The measurement says a one-step value
must see past the action's immediate material to tilt the prior usefully,
which is what a critic is for.

Cost per decision on the box: 63 ms (the prior 26, the expansions 32, the
evaluator 4), 14.7 states per decision; 692 attack expansions over the
512-sequence cap kept their prior score (71 games). About $0.32 for the
box.
