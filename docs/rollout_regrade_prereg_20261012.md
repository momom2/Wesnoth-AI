# Pre-registration: rollouts from the mover's own view, re-grading the turn benchmark (2026-10-12)

Written by the project lead before any code or box for it. Step A of the
rollout evaluator (Q5 of BACKLOG.md), the pre-registered step after step 1's
Kill (docs/selfplay_program_20261008.md "Step 1, measured"), on the
recommendation of the review that withdrew the RL rounds
(docs/selfplay_rl_rounds_prereg_20261012.md, decision record).

## Question

On the turn-value benchmark, does choosing a turn by short rollouts played
from the mover's own view (no unit the mover cannot see) beat the base turn?
The same rollouts on the true state did: the HP margin read two half-turns
on gained +0.049 +- 0.017 over the base turn on the luck-adjusted truth
(`obs8`'s rollouts, 8 per candidate; training/metrics/turn_value_20260925/
reads_by_depth_20261007.txt), the only evaluator on record with a resolved
gain. A teacher used in play may not see through fog; this measures what is
left once it does not.

## Design

- **Benchmark:** the 199 positions and 966 candidate turns of the turn-value
  validation (side 2's turn starts in holdout human games; `obs8`'s turn,
  three turns at temperature 1, a "continue" edit), their states as step 1
  rebuilt them (HF `tier-b/critic_step1_20261009/bench.tar.gz`), their truth
  the 20 last playouts of each candidate on the luck-adjusted basis
  (`training/metrics/turn_value_20260925/bench_truth.npz`).
- **The world:** each candidate's state right after its end_turn, seen by
  the mover: every unit of another side the mover does not see removed, the
  opponent's gold and base income set to the mover's (the look-ahead
  player's observed world, `wesnoth_ai/lookahead_world.py`). The opponent's
  hidden units are missing from every candidate of a position alike.
- **Rollouts:** 8 per candidate, both sides `own2` at 0 slots at
  `raw:t0.5+eo-1.5` (the truth's continuation was `obs8` at 0.5), fresh
  dice for each, played to the fourth player turn start after the candidate
  or the game's end.
- **Reads:** at the second and the fourth player turn start after the
  candidate (read 2, read 4), the mover's HP margin over the two player sides
  and `own2`'s value head at 0 slots signed to the mover, each averaged over
  the 8 rollouts; a rollout that ended takes the result.
- **Statistics:** step 1's (`wesnoth_ai/turn_bench_stats.py`): the
  selection gain over the base turn and the corrected within-position
  correlation, paired bootstraps over positions, the luck-adjusted basis
  primary and the raw basis beside it. Also reported: how often each grader
  picks the "continue" edit, and the selection gain over the positions where
  it does not.
- **Primary grader:** the HP margin at read 2 (the true-state reading with
  the resolved gain). The other three are reported, not judged.

In the same box, one match: `own1` at 64 slots at `raw:t0+eo-2.5` against
`own1` at `raw:t0+eo-1.5`, 800 decisive games, seed base 120000 (the end_turn
offset was last swept on `terrain`, a player without memory; tempo is the
largest lever on record).

## Predictions (the lead's)

Primary grader's selection gain over the base turn: +0.030 (+0.005 to
+0.055); the share of its picks that are the "continue" edit: 0.30. The
value head at read 2: +0.015. The offset match: +10 Elo (-15 to +35).

## Readings

- **Pass:** the primary grader's selection gain over the base turn above 0
  by two paired standard errors. Rollouts from the mover's own view carry a
  teacher's signal; step B (a rollout player gated against `own1` over 800
  decisive games, its in-turn dice handled by deciding one action at a time)
  is pre-registered next.
- **Pass, tempo only:** a pass whose gain is not above 0 by two standard
  errors once the "continue" picks are removed. The signal is "act more";
  the offset match says whether a decode gets it, and step B waits.
- **Fail:** otherwise. The rollout signal does not survive the fog at this
  depth and width; no rollout teacher is built, and the next evaluator is
  a belief over the hidden units (docs/hidden_information_20260926.md).
- **The offset match** at +25 or more moves the reference decode to -2.5
  (the reference's checkpoint unchanged).

## Cost

About 966 x 8 rollouts of about four half-turns, 300,000 decisions through
the shared inference server (550 to 3,200 a second): 2 to 10 minutes; the
match about 15 minutes; bring-up and tests about 15. About 45 minutes,
$0.20-0.30. BOX_MAX_H 2.
