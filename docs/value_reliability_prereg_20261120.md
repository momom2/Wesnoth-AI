# Pre-registration: how reliable the value estimate is (2026-11-20)

**Status:** registered 2026-11-20, before the logs are read for it.
Results below.

## Why

The simple loop weights every decision of a game-side by `z − v_t`: the
side's result less the value estimate logged at that decision. A weight
that credits a decision with what the value estimate did soon after it
(generalized advantage estimation with λ under 1) is the standard way to
cut that weight's noise, and it leans entirely on the value estimate.

The user, 2026-11-20: "Before we try lowering lambda, we need to measure
the reliability of the value estimate. So far, I'm not aware of any
evidence that it's much good."

What is known already, and no more: the pass logs the value head's
squared error against the result at about 0.52 to 0.55 over the run, and
it has not fallen (`progress.jsonl`).

## What is read

No game is played and no network is run. The reading is over the
decision logs of the simple loop's own games
(`wesnoth_runs\simple_loop\round_NNNN\games_*`): at each decision the
side, the value head's expectation `v`, and the game's result.

- **The games:** rounds 177 to 208, the current network's game-sides
  only (both sides of a self game, the current side of a past-self game),
  about 7,400 game-sides. Rounds 1 to 32 are read the same way beside
  them, to see whether 200 rounds changed anything.
- **A game-side's progress** is a decision's index over the side's count
  of decisions; a third is a third of that.
- It runs on the CPU at Idle priority, on 4 threads at most.

## What is measured

1. **Calibration.** The slope and intercept of the result on `v` (least
   squares), over all decisions and in each third of the game-side; and
   the mean result in ten equal-count bins of `v`.
2. **How much it knows, and when.** One less the mean of `(z − v)²` over
   the mean of `z²`, over all decisions and in each tenth of the
   game-side: the share of the result's spread the estimate accounts for.
3. **Whether it moves like a value.** Between a side's consecutive
   decisions the estimate changes by `d_t = v_{t+1} − v_t`, the last
   change being `z − v_T`. For the value of the policy that played, a
   change cannot be predicted from what came before. Read: the
   correlation of `d_t` with `d_{t−1}`, and of `d_t` with `v_t`.
4. **What λ would do to the weight.** For λ of 1, 0.98, 0.95, 0.9 and
   0.8, the weight `sum over l of λ^l d_{t+l}`: its mean square over
   today's (λ = 1), and its correlation with today's.
5. **Where the weight sits** (for the question of which games to spend a
   gradient on). Each game-side's `sum over t of (z − v_t)²`: the share
   held by the top tenth, quarter and half of the game-sides, and one
   plus the squared coefficient of variation of its square root.

## The numbers that decide

Whether a run at λ under 1 is worth registering. All three must hold on
rounds 177 to 208:

- **Calibrated:** the slope of the result on `v` is between 0.8 and 1.2
  in each third of the game-side.
- **Moves like a value:** the correlation of consecutive changes is
  within ±0.1.
- **Worth it:** at λ = 0.95 the weight's mean square is half of today's
  or less.

If the first or the second fails, the value estimate is what needs work
first, whatever λ is; the reading says which way it fails. If only the
third fails, λ under 1 is not worth a run at this value estimate.

These thresholds are the assistant's, set before the reading. They test
necessary conditions. None of them shows that a λ under 1 trains better:
only games can, in a run of its own.

## What it cannot show

- The logged estimates are the playing network's, in fp16, at the moment
  of play. A value head trained longer or differently is another
  question.
- A correlation of changes near zero does not prove the estimate is the
  policy's true value. A wrong estimate that is merely noisy passes it.
- The bias a λ under 1 adds to the gradient is not measurable from logs.

## Results

Read 2026-11-20 15:08 to 15:13 with
`training/metrics/value_reliability_20261120/read_value.py` (its outputs
are beside it). The script was first tried on rounds 207 and 208 alone, to
check that it ran.

**The verdict: by the registered rule the value estimate does not pass,
narrowly, so no run at λ under 1 is registered.** It fails on calibration
in the first third of the game. The second number sits on its threshold.
The third passes by a wide margin.

| The number | Rounds 177 to 208 | The rule | |
|---|---|---|---|
| Slope of the result on `v`, first third | 0.771 ± 0.030 | 0.8 to 1.2 | fails, by one standard error |
| second third | 1.089 ± 0.012 | 0.8 to 1.2 | holds |
| last third | 1.043 ± 0.004 | 0.8 to 1.2 | holds |
| Correlation of consecutive changes | −0.100 ± 0.001 | within ±0.1 | on the threshold (−0.0999) |
| Weight's mean square at λ = 0.95 over today's | 0.096 ± 0.001 | 0.5 or less | holds |

Rounds 177 to 208 are 7,360 game-sides and 1,728,584 decisions (3,693
wins, 3,667 losses, no game without a winner). The standard errors come
from resampling whole games 400 times.

1. **Calibration.** Over all decisions the slope is 1.026 and the
   intercept −0.083.
   - At the ends the estimate is exact: where it reads −0.96 the result
     averages −0.98, and where it reads +0.97, +0.97.
   - In the middle it is optimistic by about 0.1: it reads −0.02 where
     the result averages −0.12, +0.15 where it averages +0.03, +0.34
     where +0.22, +0.61 where +0.49. Both sides of a game think
     themselves a little ahead.
   - In the first third it is overconfident: the result moves 0.77 for
     each 1 the estimate moves.
2. **How much it knows, and when.** It accounts for 45.3% of the result's
   spread over all decisions (mean squared error 0.547). By tenth of the
   game-side: 2%, 5%, 13%, 26%, 40%, 53%, 65%, 77%, 85%, 89%. In the
   first fifth of a game it knows next to nothing, which is the game's
   nature as much as the estimate's.
3. **Whether it moves like a value.** A change is followed by a change of
   the opposite sign a little more often than chance: the correlation of
   consecutive changes is −0.100. That is what an estimate with some
   reading-to-reading jitter does. With independent jitter the figure
   puts it at about a fifth of the change's mean square (0.0103 a
   decision). The changes are also slightly against the level (−0.049):
   a high reading tends to come down, the optimism again. The last change
   of a side, from its last reading to the result, has a mean square of
   0.123.
4. **What λ would do to the weight.** Its mean square over today's, and
   its correlation with today's weight: 0.19 and 0.74 at λ = 0.98; 0.096
   and 0.53 at 0.95; 0.060 and 0.39 at 0.9; 0.039 and 0.28 at 0.8.
5. **Where the weight sits.**
   - Among game-sides it is spread evenly. The top tenth hold 35.5% of
     it, the top quarter 59.5%, the top half 81.4%, and one game-side in
     a thousand holds under a tenth of the mean. One plus the squared
     coefficient of variation of its root is 1.22: choosing which
     game-sides to pass by their weight could gain about a fifth, no
     more.
   - Within a game-side it sits early. From the tenths above, the first
     three tenths of a side's decisions hold 51% of the weight's mean
     square and the first half 76%: the decisions where the estimate
     knows least are weighted most, by a result they did least to
     decide.

**Rounds 1 to 32 read the same** (7,360 game-sides, 1,815,523
decisions): slopes 0.757 ± 0.028, 1.068 ± 0.010 and 1.037 ± 0.004; a
correlation of consecutive changes of −0.081 ± 0.001; 46.3% of the
result's spread accounted for; 0.096 at λ = 0.95. Two hundred rounds of
training have not changed the value estimate's quality in either
direction.

**Why it has not changed: the value head is held by the learning rate**
(read 2026-11-21 from the checkpoints; not registered). Between the start
network and round 208's:

| Weights | Count | Change over the weights' norm | Root mean square change of a weight |
|---|---|---|---|
| The value head | 167,526 | 1.66% | 1.89e-3 |
| The encoder (the trunk) | 14,195,712 | 0.33% | 1.55e-4 |
| Every weight | 15,304,046 | 0.39% | 2.51e-4 |

- Adam at a learning rate of 1e-5 can move a weight by at most 1e-5 a
  step, so by 2.08e-3 in 208 steps, and it moves it that far only if its
  gradient has the same sign every round.
- The value head's weights moved by 91% of that limit. Its gradient has
  pointed the same way for the whole run, and the learning rate is what
  stopped it going further.
- The other weights moved by 7 to 10% of the limit, as a gradient that
  is mostly noise from round to round does.
- So the value head is trained in name only: 208 steps have changed it by
  1.7% and its readings not at all. A learning rate sized for the
  policy's noisy gradient is far too small for the value head's steady
  one.

**What this says, and what it does not.**
- The estimate is evidence-worthy in the second half of a game and weak
  in the opening, where it is also overconfident. It is not "not much
  good": it is exact at the ends and within about 0.1 in between.
- Today's weight spends half its power on the opening third, where the
  result is nearly unpredictable. That is the noise a λ under 1 would
  remove, and the size of it (a tenth of the mean square at 0.95) is
  why the question is worth settling.
- A λ under 1 would also carry the estimate's opening overconfidence and
  its optimism into the weight. By the registered rule that comes first.
- None of this shows what trains better. Only games can.

## Follow-up: why the estimate is optimistic and overconfident (2026-11-21)

The user, 2026-11-21: "I don't understand how the value estimate can be
too optimistic. There's probably some mismatch between what the network
sees and what it thinks it is evaluating [...] The value estimate also
shouldn't be overconfident [...] Understanding the value estimate and
fixing any issue there is our top priority."

This is a search for a cause, not a run with a deciding number. The
candidate causes and what each predicts were written here before the
logs were read for them (2026-11-21 22:05).

A head trained on the result at every decision, and read on decisions of
the same kind, cannot stay biased on average: the bias is itself the
training signal. So a steady bias means one of two things: the head is
read on something other than what it was fitted on, or it is fitted too
slowly to follow what it is read on.

- **C1, the side's own turn.** A side decides only during its own turn.
  As it moves and attacks, the board improves for it, and the enemy's
  reply is not on the board yet. *Predicts:* the estimate less the result
  grows from a turn's first decision to its last, and the estimate falls
  back between a side's last decision of one turn and its first of the
  next.
- **C2, hidden enemies.** The estimate reads the visible board as the
  whole board. *Predicts:* the estimate less the result grows with the
  enemy strength the deciding side cannot see, and is near zero when it
  sees everything.
- **C3, another opponent.** The head was fitted against other play than
  it now meets (a weaker opponent, another decode). *Predicts:* the same
  optimism at every point of a turn and whatever is hidden.
- **C4, the winner decides more often.** The side that wins has more
  units and so more decisions. *Predicts:* the mean result over decisions
  is above zero. That would explain a mean estimate above zero, not an
  estimate above the result within a bin.
- **C5, a prior on the matchup** (for the overconfidence early). The
  estimate at a game's first decision spreads more than the map, the
  factions and the side decide the result. *Predicts:* at the first
  decision, a slope of the result on the estimate well under 1.

Why it matters beyond the reading itself: a weight with λ under 1 pays a
decision for raising the estimate soon after. If the estimate is high
where the enemy is unseen (C2) or late in the side's own turn (C1), such
a weight would pay the policy for hiding the enemy from itself or for
acting late, not for winning.
