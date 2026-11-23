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

### What was found (2026-11-21, rounds 177 to 208 unless said)

Scripts and outputs: `training/metrics/value_reliability_20261120/`
(`where_optimism.py`, `hidden_enemy.py`, `hidden_crosstabs.py`,
`lambda_clock.py`). The replays' position tables are in
`wesnoth_runs\value_reliability_20261120\`. Standard errors take games as
the units.

**Where the estimate comes from.** By the project's own records the value
head was fitted on the results of human games during imitation
(`configs/imitation.json`, `value_from_outcome_weight`), and the
policy-iteration rounds read it frozen
(`docs/training_design_20261019/current_design.md`: "the frozen value
head's reading"). The simple loop is the first training to move it on
this network's own games, and Adam moves a weight by the learning rate a
step at most. So what is read is still largely an estimate of who wins
when humans play on, applied to this network's sampled play. That is how
it can be biased under a regimen that would not make it so.

**C1, the side's own turn: no in these rounds, yes in most others**
(corrected 2026-11-23: "A correction to C1" below). The estimate less the result is +0.075
at a turn's first decision, +0.086, +0.088, +0.078 and +0.059 through its
four quarters, and +0.079 at the end-of-turn decision (each ± 0.003). The
estimate moves +0.0001 ± 0.0001 a decision within a turn and −0.0016 ±
0.0011 across the enemy's turn.

**C4, the winner decides more often: true, and not the cause.** The
winner makes 56.2% of the decisions, so the mean result over decisions
is +0.125. The estimate's mean is +0.203: the excess of +0.078 ± 0.003
is what needs a cause.

**C3, other play: yes, and it is the larger part.** The error follows the
enemy's faction. At a turn's first decision (682 replayed games, 22,736
positions, +0.064 ± 0.007 over all):

| Against | Estimate less result |
|---|---|
| Undead | +0.198 ± 0.045 |
| Northerners | +0.165 ± 0.045 |
| Loyalists | +0.029 ± 0.043 |
| Rebels | +0.010 ± 0.046 |
| Drakes | −0.005 ± 0.044 |
| Knalgan | −0.008 ± 0.046 |

- Against Undead or Northerners it is too high whatever the side sees:
  +0.106 ± 0.031 with no enemy unit hidden, +0.192 ± 0.035 with one or
  two, +0.221 ± 0.035 with three to five. Where it reads between −0.4 and
  +0.4 against them, the side's result averages −0.21 to −0.35.
- Playing Undead it is too low (−0.108), playing Drakes too high
  (+0.189). In the cells of the table of own faction by enemy faction
  (about 19 games each, so rough) the two sides of a matchup err in
  opposite directions: Drakes against Undead +0.62, Undead against Drakes
  −0.43.
- Under this network's sampled play the factions are far from even. The
  mean result over a faction's decisions is +0.36 for Northerners, +0.32
  for Loyalists, +0.17 for Rebels, +0.06 for Undead, −0.10 for Drakes and
  −0.15 for Knalgan.
- The estimate's opinions on matchups are not this play's. The human
  games it was fitted on are the likely source; their results by faction
  were not read for this.

**C2, hidden enemies: partly, and not as the early bug.** From turn 5 on a
side cannot see 2.0 of the enemy's 9.4 units on average: 1.1 that it saw
at an earlier decision of its own, 0.9 that it never saw (819 replayed
games, 27,672 positions). What a unit is worth, by least squares from
turn 5 on:

| A unit | To the estimate | To the result | Estimate less result |
|---|---|---|---|
| An own unit | +0.088 ± 0.002 | +0.100 ± 0.003 | −0.012 ± 0.002 |
| An enemy unit in sight | −0.084 ± 0.002 | −0.095 ± 0.003 | +0.011 ± 0.002 |
| An enemy unit seen before, now out of sight | −0.079 ± 0.004 | −0.104 ± 0.006 | +0.025 ± 0.004 |
| An enemy unit never seen | −0.128 ± 0.007 | −0.122 ± 0.010 | −0.006 ± 0.008 |

- The estimate does not take the visible board for the whole board. It
  counts an enemy unit it has lost sight of almost as one in sight (0.079
  against 0.084), and it allows in full for units it has never seen.
- Its error is elsewhere: an enemy unit that has gone out of sight turns
  out worth more than one in sight (0.104 against 0.095), and the
  estimate does not follow. It carries about three quarters of such a
  unit's worth. With 1.1 of them on average that is about +0.03 of the
  optimism.
- Held at the same reading, the gap is large where several are out of
  sight: reading between −0.4 and +0.4 with three or more units seen
  before and now hidden, and none never seen, the estimate is too high by
  +0.31 ± 0.06. Positions picked by their reading overstate a per-unit
  error, and the table above is the fairer measure.
- The estimate also counts every unit, on either side, at about 88% of
  what it proves worth. With even armies that cancels.

**C5, a prior on the matchup: yes, and it is shrinking.** At a game's
first decision the result moves 0.66 ± 0.08 for each 1 the estimate
moves, and the estimate's spread there is 0.147. From the side's second
turn the slope is 1 within its error (1.08 ± 0.11, 0.94 ± 0.08, 1.02 ±
0.03 and 1.10 ± 0.02 at the second, third, fifth and eighth turns). The
overconfidence of the first third is the opening's. Over the run the
opening slope went from 0.41 (rounds 1 to 32) to 0.61 (rounds 283 to
314) and the spread from 0.171 to 0.128.

**Over the run.** The estimate less the result over all decisions was
+0.034 in rounds 1 to 32, +0.053, +0.063, +0.078 (rounds 177 to 208),
then +0.049 and +0.045 (rounds 283 to 314). By own faction it is steady:
Knalgan +0.10 to +0.16 and Northerners −0.004 to −0.04 in every block.

**A correction to "trained in name only" above.** The value head does
learn: the opening prior has shrunk by a quarter. It learns at the pace
Adam allows a steady gradient, the learning rate a step, and the play it
evaluates moves meanwhile.

**After each kind of decision** the estimate moves, on average, by −0.0087
± 0.0002 after an attack, +0.0026 ± 0.0001 after a move and +0.0069 ±
0.0003 after a recruit. By the estimate, the sampled policy's attacks
cost it and its recruits gain. Whether that is the play or the estimate
is not separable from the logs.

**For a weight with λ under 1.**
- The mean square of the weight over today's, fading by the decision:
  0.19 at 0.98, 0.096 at 0.95, 0.060 at 0.9. Fading once a turn of the
  side (12.7 decisions): 0.56 at 0.95, 0.38 at 0.9, 0.23 at 0.8, 0.17 at
  0.7, 0.13 at 0.5.
- Today's weight has 51% of its mean square in the first three tenths of
  a side's decisions; fading by the turn at 0.8 leaves 34% there.
- The enemy's turn moves the estimate by 0.24 (root mean square), one own
  decision by about 0.08, and a side's last reading is 0.35 from its
  result.
- The faction error is fixed for a game, so it leaves the estimate's
  changes alone and enters only through the last step to the result,
  which a λ under 1 fades for early decisions. The error on units out of
  sight changes the estimate little when a unit leaves sight (0.084 to
  0.079).

## Follow-up: a weight with λ = 0.99 (2026-11-23)

The user, 2026-11-23, to a one-off refit of the value head on the loop's
own games: "No. How about we continue with λ = 0.99?" These readings
were taken for that discussion. They were not registered and decide
nothing by themselves. No λ under 1 has run.

Scripts and outputs: `lambda_clock.py` (`clock_177_208.json`,
`clock_393_424.json`) and `where_optimism.py` (`where_*.json`) in
`training/metrics/value_reliability_20261120/`. Where two figures are
given they are rounds 177 to 208, then rounds 393 to 424.

**The weight.** For one side's decisions in play order, v the logged
estimates and z the side's result, standing in for the estimate after
its last decision:

    a_t = sum over k >= 0 of λ^k (v_{t+k+1} − v_{t+k})
        = −v_t + (1 − λ) sum over k >= 1 of λ^(k−1) v_{t+k} + λ^(its later decisions) z

At λ = 1 it is today's z − v_t. Under 1 it is a blend of the later
estimates and the result, less the present estimate. It is generalized
advantage estimation with no discount and the result as the only reward.

**What it does to the weight.**

| λ, fading by the decision | Mean square over today's | Correlation with today's | The result's share of the blend | Mean square over today's in a game-side's first tenth, last tenth |
|---|---|---|---|---|
| 0.995 | 0.49, 0.47 | 0.95, 0.94 | 55%, 53% | 0.35 and 0.90; 0.33 and 0.89 |
| 0.99 | 0.32, 0.30 | 0.87, 0.86 | 36%, 34% | 0.19 and 0.82; 0.18 and 0.80 |
| 0.98 | 0.19, 0.18 | 0.74, 0.72 | 21%, 19% | 0.09 and 0.69; 0.09 and 0.67 |
| 0.95 | 0.096, 0.092 | 0.53, 0.52 | 8.5%, 7.9% | 0.03 and 0.46; 0.03 and 0.45 |

A game-side has 235 and 252 decisions, a turn 12.7 and 13.0. So 0.99 a
decision is 0.88 a turn, and a change of the estimate counts for half
after 69 decisions, a little over five turns.

**Where a λ under 1 could mislead.** The λ weight less today's is

    −sum over k >= 1 of (1 − λ^k) (v_{t+k+1} − v_{t+k}):

the later changes of the estimate, which λ leaves out in part. Were the
estimate exact, those changes would average zero after any decision and
the two weights would agree on average. Where they can be foreseen from
what was decided, a λ under 1 pays for the decision and not for the
result. Read by kind of decision, each decision less the mean of its
turn, so that when in a game a kind is taken does not count:

| The λ weight less today's, within a turn | Attack | Move | Recruit | End of turn |
|---|---|---|---|---|
| 0.995 | +0.00001, −0.00006 | +0.00016, +0.00006 | +0.00009, +0.00011 | −0.00135, −0.00044 |
| 0.99 | −0.00002, −0.00012 | +0.00023, +0.00009 | +0.00016, +0.00018 | −0.00185, −0.00061 |
| 0.98 | −0.00011, −0.00020 | +0.00028, +0.00014 | +0.00027, +0.00024 | −0.00210, −0.00076 |
| 0.95 | −0.00037, −0.00036 | +0.00032, +0.00025 | +0.00041, +0.00018 | −0.00182, −0.00109 |

- The standard errors are 0.00013 at most.
- Today's weight read the same way: attack −0.0054 and −0.0131, move
  +0.0002 and +0.0033, recruit +0.0043 and +0.0044, end of turn +0.0104
  and +0.0082.
- So at 0.99 the weights of attacks, moves and recruits move against one
  another by 0.0003 at most. Today's weight sets attacks and recruits
  0.010 and 0.018 apart.
- The end-of-turn decision is always its turn's last, so its column
  carries the place in the turn too. Two neighbouring decisions' columns
  differ by (1 − λ) times the later one's weight, and the weight's mean
  is under zero by the optimism.
- This reads the kind of decision only. An error that depends on which
  hex or which target is not read by it, and the logs cannot read it.

**A correction to C1.** "C1: no" above was read on rounds 177 to 208
alone. Over every window read:

| Rounds | The estimate's rise over a side's turn | Its change across the enemy's turn | Estimate less result: all decisions | A turn's first decision | The end-of-turn decision |
|---|---|---|---|---|---|
| 1 to 32 | +0.011 | −0.010 | +0.034 | +0.044 | +0.057 |
| 65 to 96 | +0.002 | −0.002 | +0.053 | +0.064 | +0.069 |
| 129 to 160 | +0.018 | −0.019 | +0.063 | +0.061 | +0.081 |
| 177 to 208 | +0.002 | −0.002 | +0.078 | +0.075 | +0.079 |
| 241 to 272 | +0.034 | −0.035 | +0.049 | +0.026 | +0.062 |
| 283 to 314 | +0.040 | −0.042 | +0.045 | +0.014 | +0.057 |
| 393 to 424 | +0.024 | −0.024 | +0.027 | +0.005 | +0.030 |

- The changes are ± 0.001, the differences ± 0.003.
- So C1 holds in five of the seven windows: the estimate rises over a
  side's own turn and the enemy's turn takes it back, by up to 0.04. The
  window the candidates were read on was one of the two without it.
- In the latest window all of the optimism is of this kind. The estimate
  is right at a turn's first decision (+0.005 ± 0.003) and +0.030 too high
  at its last.
- The optimism over all decisions has fallen to +0.027 ± 0.003, a third
  of what it was in rounds 177 to 208.
- After each kind of decision the estimate now moves by −0.0184 ± 0.0002
  (attack), +0.0090 ± 0.0001 (move) and +0.0065 ± 0.0004 (recruit).
- The opening slope reads 0.54 ± 0.09 with a spread of 0.131: no change
  from rounds 283 to 314 (0.61 ± 0.09, 0.128).

**What the rise within a turn means for λ.** A rise that the enemy's turn
takes back cancels in today's weight. Fading by the decision, a rise k
decisions before the turn's end keeps 1 − λ^k of itself: from the middle
of a turn about 6% at 0.99 and 28% at 0.95, so 0.0015 and 0.007 of a rise
of 0.024. Fading once a turn of the side keeps none of it, since a
turn's own changes and the enemy's reply all count in full. So at 0.99
the clock matters little, and a shorter horizon would be the case for
ticking by the turn.
