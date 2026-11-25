# A weight with λ = 0.99: pre-registration (2026-11-23)

**Status: the reading closes at round 544, with five matches; the
registered verdict cannot be read. λ stays at 0.99.** Round 465 was the
first at λ = 0.99 (2026-11-23 23:58:47), and it checked out (Results).
The simple loop (`docs/simple_loop_prereg_20261117.md`) ran at λ = 1 to
its match at round 464. The deciding number was to be read after the
match at round 704. From round 545 the network's inputs and the legal
actions change, on the user's order of 2026-11-24
(`docs/input_facts_prereg_20261125.md`, departure 1 below), so no later
match is a match of λ alone. Kill criterion 2 was read at rounds 528 and
544 and is not met (Results).

The user proposed it on 2026-11-23 ("How about we continue with
λ = 0.99?"), was shown the readings taken on the logged games
(`docs/value_reliability_prereg_20261120.md`, "Follow-up: a weight with
λ = 0.99") and answered "Alright, do it." This document states what
changes, what is measured, the number that decides and the kill criteria.
The result lands here whatever it says. It is departure 9 of the simple
loop's registration.

## What changes

One thing: the weight each decision's log probability gets in the pass
(`tools/pg_game_pass.py`, `lambda_weights`).

- **Until the switch** it is the side's result less the value estimate
  logged at the decision: z − v_t.
- **From the switch** every later change of the estimate counts for 0.99
  of the one before it:

      a_t = sum over k >= 0 of 0.99^k (v_{t+k+1} − v_{t+k}),

  where the estimate after a side's last decision is its result z.
- **Said as a comparison:** the present estimate is compared with a blend
  of the estimates logged later in the game and the result, where today
  it is compared with the result alone. The estimate k decisions later
  takes 1% of what the ones before it left, and the result takes the
  rest.
- **It is the standard form:** generalised advantage estimation (Schulman
  et al. 2016) with no discount and the result as the only reward. λ = 1
  is today's weight.
- **It fades by the decision,** not by the turn. At 13 decisions a turn,
  0.99 a decision is 0.88 a turn, and a change of the estimate counts for
  half after 69 decisions, a little over five turns.
- A refused decision is in its side's chain like any other. The last 64
  rounds hold none.

## What does not change

- The value head trains on the result at every decision, as before. Only
  the policy's weight leans on the later estimates.
- The learning rate (1e-5, one for every parameter) and the optimizer,
  whose state carries over the switch.
- The games, the pool of past selves, the matches and their seeds, the
  loop's kill criteria, and `config.json`.

## What is known before the switch

Read on two windows of the loop's logged games, rounds 177 to 208 and
rounds 393 to 424 (the value document's follow-up holds the tables):

- **The weight is less noisy.** Its mean square is 0.32 and 0.30 of
  today's, and 0.19 and 0.18 in a game-side's first tenth.
- **The result stays in it.** It is 36% and 34% of the blend on average.
  The λ weight's correlation with today's is 0.87 and 0.86.
- **By kind of decision it changes little.** Within a turn, the weights
  of attacks, moves and recruits move against one another by 0.0003 at
  most. Today's weight sets attacks and recruits 0.010 and 0.018 apart.
  That check sees the kind of decision only, not the hex or the target.
- **Why not lower.** In five of seven windows the estimate rises over a
  side's own turn and the enemy's turn takes it back, by up to 0.04.
  Fading by the decision lets about 6% of such a rise into the weight at
  0.99 and 28% at 0.95.
- **The estimate itself** is optimistic by +0.027 ± 0.003 over all
  decisions in rounds 393 to 424, down from +0.078 in rounds 177 to 208.

The loop's matches at λ = 1, to round 448 (the candidate samples, the
reference `own2` plays its pinned decode, 800 decisive games each):

- The fourteen matches of rounds 240 to 448 average +152.8 ± 3.6 Elo.
  A line through them rises 0.03 ± 0.06 Elo a round: no rise is read
  within them.
- They scatter by 7.3 Elo around their mean, where each one's standard
  error is 13.5. Every match plays the same seeds, which would explain
  it: two matches then differ by less than their standard errors say.
- A line through the twenty-one matches from round 128 rises 0.16 ± 0.03
  Elo a round. That rise came as a step between rounds 224 and 240.

## What is measured

1. **The matches,** unchanged: after every 16th round, 800 decisive games
   from the same seeds, the candidate sampling at temperature 1.
2. **Every round,** in `grad.json` and `progress.jsonl`: the weights' mean
   square and what it would be at λ = 1; the gradient's norm over each
   group of parameters (the value head's group holds the cross-entropy's
   gradient, the others the policy's); the update's norm; the value
   head's cross-entropy.
3. **The value estimate again,** on rounds 497 to 528 with the value
   document's script (`where_optimism.py`): its optimism and its rise
   over a side's turn.

Only the matches decide. The rest are logged and never a verdict.

## The number that decides

Read after the match at round 704: the fifteen matches at λ = 0.99
(rounds 480 to 704) against the fifteen before the switch (rounds 240 to
464, the block since the level stepped up). About 3.3 days after the
switch.

D is the later block's mean less the earlier's, with the standard error
of that difference: about 4.9 Elo if each match's stays near 13.5.

- **D under −2 standard errors: rejected.** "λ = 0.99 by the decision, at
  1e-5, lost strength in 240 rounds." The default is then to go back to
  λ = 1, as a numbered departure; what follows is the user's to decide.
- **Otherwise λ = 0.99 stays.** D above +2 standard errors is recorded as
  "the loop gained under λ = 0.99", and D within ±2 as "no change read in
  240 rounds".

**What the number cannot say.** It compares before with after on one
machine: no run at λ = 1 plays beside it. So a gain would say that the
loop gained under λ = 0.99, not that λ made it gain. The earlier block's
own line (0.03 ± 0.06 Elo a round), carried over the 240 rounds between
the two blocks' centres, allows +7 ± 13 Elo with no change at all.
Telling the two apart takes a second run at λ = 1 from round 464's
network. That is rented compute and is not proposed here.

So the registered question is whether λ = 0.99 loses strength. It was
chosen on principle (the standard form, a weight a third as noisy, the
kinds of decision left alone), and it stays unless the games say it
costs.

## Kill criteria

1. **The loop's own,** unchanged (`docs/simple_loop_prereg_20261117.md`,
   "Kill criteria"): two matches in a row more than 3 standard errors
   below the run's first, a failed exact check, a gradient that is not
   finite, a display-driver reset, a round past 1.5 times its estimate,
   the user's word or a STOP file. The runner enforces the first four.
2. **An early loss.** From the fourth match at λ = 0.99 on (round 528):
   the mean of the λ matches so far reads more than 3 standard errors
   below the mean of rounds 240 to 464 (23 Elo below at four matches, 18
   at eight). λ then goes back to 1 at the next round's start, as a
   numbered departure, and the user is told. The runner does not read
   this one: it is read by whoever records the match.
3. **The first round at λ = 0.99 does not check out:** its replay of the
   games is outside the exact check's 1e-4 (the runner stops there), or
   its weights' mean square is not between 0.2 and 0.45 of what it is at
   λ = 1, where the logged games read 0.30 to 0.32. The second would mean
   the pass does not compute the weight that was measured: the run goes
   back to λ = 1 until it is explained.

## What to expect

Stated before the switch so that the first rounds can be read against
it. None of it is a verdict.

| Reading | At λ = 1 (rounds 388 to 451, medians) | Expected at λ = 0.99 | Why |
|---|---|---|---|
| The weights' mean square over what it is at λ = 1 | 1 | about 0.3 | read at 0.32 and 0.30 on the logged games |
| The policy's part of the gradient's norm | read on the rounds between the merge and the switch | about 0.55 of it | if that part is mostly noise and the noise follows the weight: √0.3 |
| The value head's part | read on the same rounds | unchanged | its loss does not change |
| The whole gradient's norm | 69.9 | about 55 | the value head holds 43% of the squared gradient Adam has averaged (read at round 451) |
| The update's norm | 0.0095 | about 0.6 of it within some twenty rounds, then back toward 0.0095 over about 1,000 | below; the value head's weights are about 6% of the update's square and keep their step |
| The value head's cross-entropy | 0.41 | unchanged | its loss does not change |

**Corrected before the switch** (2026-11-23 23:47, while the match at
round 464 played): the whole gradient's norm is expected at about 44, not
about 55.
- The table took the value head's share of the squared gradient from
  Adam's average: 43%. That average weighs every round since the run's
  first between 0.64 and 1 (0.999^451 = 0.64), so it reads the share over
  the whole run. The rounds before the switch do not match it.
- The eleven rounds between the merge and the switch (454 to 464, at
  λ = 1, the first whose passes record the parts) read:

  | Reading | Rounds 454 to 464: median (range) | Expected at λ = 0.99 |
  |---|---|---|
  | The policy's part of the gradient's norm | 57.0 (55.4 to 83.0) | about 31: 0.55 of it |
  | The value head's part | 30.4 (11.8 to 46.0) | unchanged |
  | The whole gradient's norm | 68.2 (61.0 to 86.0) | about 44 |
  | The value head's share of the squared gradient | 23% (4% to 39%) | about half |
  | The weights' mean square | 0.526 (0.461 to 0.580) | about 0.3 of what the same round reads at λ = 1 |

- The update's norm keeps its expectation. It was worked from the value
  head's share of the update's square (6%), not of the gradient's.
- Nothing that decides or kills changes.

**What Adam's memory does to the pace.** Adam divides each weight's step
by the root of its squared gradient averaged over about the last 1,000
rounds (β2 = 0.999: an old gradient's share of that average halves in 693
rounds, 9.5 days at 73 rounds a day).

- **At first** the average still remembers λ = 1's gradients. A round
  then moves the policy as far along what the rounds agree on as it does
  now, with about half the random motion.
- **As the average forgets,** every step grows, by up to 1.8 times: the
  move along what the rounds agree on with it, the random motion back to
  where it is now.
- So if λ = 0.99 gives a faster pace, it is expected late in the 240
  rounds or after them, not at once. A higher learning rate would bring
  it forward. That is a separate change with its own number, not made
  here.

## The switch

- **The code:** `feature/lambda-weight`. `tools/pg_game_pass.py --lam`
  and `scripts/simple_loop_local.py --lam`, both 1 by default. Like
  `--lr`, `--lam` may change between two starts of a run; a round's line
  in `progress.jsonl` carries the λ its pass recorded.
- **Until the switch** the run's passes use the merged code at λ = 1: the
  weight is the same, and each round's `grad.json` gains the readings
  above. Those rounds are the λ = 1 side of the comparison for the
  gradient's parts.
- **How:** a STOP file written while the match at round 464 plays ends
  the run after it. The launch's command starts again with `--lam 0.99`
  added.
- **The check:** kill criterion 3, on round 465.
- **What happened** (2026-11-23): the STOP file was written at 23:45:59,
  25 seconds into the match at round 464. The match read at 23:58:14 and
  the runner stopped before round 465. The launch's command started again
  at 23:58:47 with `--lam 0.99`; its first log line reads "lambda 0.99".
  The GPU idled for 33 seconds.

## Results

Filled in as the run goes.

**The block before the switch.** The match at round 464 read +148.0 ±
13.4 (561 wins of 800). The fifteen matches of rounds 240 to 464 average
+152.5 ± 3.5 Elo. They scatter by 7.2 around their mean, and a line
through them rises 0.02 ± 0.05 Elo a round.
- Kill criterion 2 in numbers: the first four matches at λ = 0.99
  averaging under +129.7, or the first eight under +134.7.
- Both take each match's standard error near 13.5. The deciding
  number's standard error stays about 4.9.

**Round 465, the first at λ = 0.99, checks out** (2026-11-24 00:15:11;
kill criterion 3).

| Reading | Rounds 454 to 464, λ = 1: median (range) | Expected | Round 465 |
|---|---|---|---|
| The replay of the games (second-order KL, limit 1e-4) | 7.1e-6 (6.4e-6 to 7.5e-6) | inside the limit | 7.0e-6 |
| The weights' mean square over the same round's at λ = 1 | 1 | about 0.3; outside 0.2 to 0.45 kills | 0.328 (0.1735 over 0.5293) |
| The policy's part of the gradient's norm | 57.0 (55.4 to 83.0) | about 31 | 31.9 |
| The value head's part | 30.4 (11.8 to 46.0) | unchanged | 20.8 |
| The whole gradient's norm | 68.2 (61.0 to 86.0) | about 44 | 38.1 |
| The update's norm | 0.0090 (0.0088 to 0.0093) | about 0.6 of it within some twenty rounds | 0.0086 |
| The value head's cross-entropy | 0.39 (0.35 to 0.44) | unchanged | 0.39 |

- It is one round. The weight's ratio and the replay are the check; the
  rest is logged against the expectations and is no verdict.
- The policy's part is 0.56 of its median at λ = 1. A gradient that is
  all noise following the weight would read √0.328 = 0.57.
- The update's norm is 0.93 of round 464's. Adam's step follows a running
  average of the gradients in which the newest counts for a tenth
  (β1 = 0.9), so the fall is expected over some twenty rounds.
- The round holds 230 game-sides, 52,905 decisions and no refused
  decision. Its pass took 806 s.

**Matches at λ = 0.99** (800 decisive games each, as before):

| After round | Elo | Wins of 800 | Read |
|---|---|---|---|
| 480 | +139.8 ± 13.3 | 553 | 2026-11-24 04:50 |
| 496 | +178.1 ± 13.9 | 589 | 2026-11-24 09:36 |
| 512 | +164.9 ± 13.7 | 577 | 2026-11-24 14:39 |
| 528 | +175.8 ± 13.9 | 587 | 2026-11-24 19:21 |
| 544 | +179.2 ± 13.9 | 590 | 2026-11-25 00:12 |

**To round 500** (2026-11-24 10:48). Nothing here decides.
- The two matches average +158.9 ± 9.6: +6.4 ± 10.2 against the fifteen
  before the switch, 0.6 standard errors. Round 496's is the highest
  reading of the run. Two matches say nothing yet.
- The thirty-six rounds at λ = 0.99 (465 to 500) read, as medians against
  the eleven rounds before the switch:

  | Reading | Rounds 454 to 464, λ = 1 | Expected | Rounds 465 to 500 |
  |---|---|---|---|
  | The weights' mean square over the same round's at λ = 1 | 1 | about 0.3 | 0.310 (0.278 to 0.343) |
  | The policy's part of the gradient's norm | 57.0 (55.4 to 83.0) | about 31 | 33.3 (27.6 to 43.7) |
  | The value head's part | 30.4 (11.8 to 46.0) | unchanged | 29.5 (12.1 to 88.9) |
  | The whole gradient's norm | 68.2 | about 44 | 45.1 |
  | The update's norm | 0.0090 | about 0.6 of it within some twenty rounds | 0.0058 over rounds 481 to 500: 0.64 of it |
  | The value head's cross-entropy | 0.39 | unchanged | 0.39 |

- The update's norm fell from 0.0092 at round 464 to 0.0060 by round 474
  and has stayed between 0.0056 and 0.0060 since.
- The value head's part reads above the eleven rounds' highest (46.0) in
  seven rounds, all from round 484 on: 484, 485, 486, 489, 491, 495 and
  499, at 46.8 to 88.9. Its median is 26.4 over rounds 465 to 480 and
  34.2 over rounds 481 to 500. The value head's cross-entropy is 0.39 and
  0.41 over the same two stretches, and it read 0.41 over rounds 388 to
  451. Eleven rounds are a short range to compare with, and the passes
  before round 454 did not record the parts. No warning is logged. The
  value estimate's re-read on rounds 497 to 528 is where this is looked
  at.
- Every replay is inside the exact check (7.7e-6 at most), and no round
  holds a refused decision. A round takes 178 s of play and 872 s of
  pass.

**Kill criterion 2 at round 528: not met** (read 2026-11-24 20:17, the
first reading the criterion allows).
- The four matches at λ = 0.99 average +164.7 ± 6.8 Elo. That is +12.2 ±
  7.7 against the fifteen before the switch (+152.5 ± 3.5): 1.6 standard
  errors above, where the criterion asks for 3 below.
- With these four matches' own standard errors the criterion's line is
  +129.5 (+129.7 was stated with each match's at 13.5).
- So the run goes on at λ = 0.99, unchanged. The criterion is read again
  with each match.
- The four scatter by 17.5 Elo around their mean, and the fifteen before
  the switch by 7.2. Four matches are few to read a scatter from.
- Nothing here decides, and +1.6 standard errors is no gain read. The
  deciding number is the fifteen matches to round 704.

**Rounds 497 to 528**, as medians (range). Nothing here decides.

| Reading | Rounds 454 to 464, λ = 1 | Expected | Rounds 497 to 528 |
|---|---|---|---|
| The weights' mean square over the same round's at λ = 1 | 1 | about 0.3 | 0.317 (0.270 to 0.345) |
| The policy's part of the gradient's norm | 57.0 (55.4 to 83.0) | about 31 | 32.4 (28.7 to 37.8) |
| The value head's part | 30.4 (11.8 to 46.0) | unchanged | 33.6 (14.3 to 88.9) |
| The whole gradient's norm | 68.2 | about 44 | 47.1 |
| The update's norm | 0.0090 | about 0.6 of it | 0.0058 (0.0056 to 0.0062): 0.64 of it |
| The value head's cross-entropy | 0.39 | unchanged | 0.39 (0.34 to 0.46) |

- The value head's part, sixteen rounds at a time from round 465: 26.4,
  36.6, 32.8 and 33.6, against 30.4 over the eleven rounds before the
  switch. It reads above those rounds' highest (46.0) in six of rounds
  481 to 496 and in six of rounds 497 to 528 (499, 502, 503, 515, 520 and
  524).
- The value head's cross-entropy over the same four stretches: 0.39,
  0.42, 0.39 and 0.39. So the head's fit is where it was before the
  switch.
- Every replay from round 465 to round 528 is inside the exact check
  (7.7e-6 at most, limit 1e-4), and no warning is logged. A round takes
  183 s of play and 860 s of pass.

**The value estimate on rounds 497 to 528** (measurement 3, 2026-11-24;
`where_optimism.py`, whose outputs `where_433_464.json`,
`where_465_496.json` and `where_497_528.json` are in
`training/metrics/value_reliability_20261120/`). The two windows before
the registered one were read too, so that the last window at λ = 1 stands
beside the two at λ = 0.99. The first row is the latest window the value
document holds.

| Rounds | λ | The estimate's rise over a side's turn | Its change across the enemy's turn | Estimate less result: all decisions | In self games alone | A turn's first decision | The end-of-turn decision | The opening slope |
|---|---|---|---|---|---|---|---|---|
| 393 to 424 | 1 | +0.024 | −0.024 | +0.027 | +0.026 | +0.005 | +0.030 | 0.54 |
| 433 to 464 | 1 | +0.006 | −0.006 | +0.030 | +0.037 | +0.018 | +0.026 | 0.63 |
| 465 to 496 | 0.99 | +0.014 | −0.013 | +0.017 | +0.028 | −0.001 | +0.014 | 0.57 |
| 497 to 528 | 0.99 | +0.019 | −0.019 | +0.032 | +0.036 | +0.012 | +0.033 | 0.73 |

- The changes are ± 0.001, the differences ± 0.003 (± 0.001 in self games
  alone) and the slopes ± 0.09.
- **No change is read at the switch.** Over all decisions the estimate is
  optimistic by +0.032 ± 0.003, where the last window at λ = 1 read +0.030
  ± 0.003. The four windows run from +0.017 to +0.032: the reading moves
  by about 0.01 from one window to the next under either λ, which is more
  than one window's standard error allows.
- **The rise over a side's own turn** is +0.019, between the two windows
  before the switch (+0.024 and +0.006), and the enemy's turn still takes
  it back. At 0.99 about 6% of such a rise enters the weight: 0.001.
- **The opening slope** reads 0.73 ± 0.09, the highest of the four: 0.8
  standard errors above the last window at λ = 1 and 1.5 above rounds 393
  to 424. Its spread stays at 0.128. No change is read.
- **Two patterns older than the switch stay.**
  - Early in a game the estimate is too low. At the first decision of a
    side's third turn it reads 0.111 under the result (± 0.004). The same
    reading was −0.037 in rounds 241 to 272, −0.075 in rounds 283 to 314,
    then −0.096, −0.095 and −0.122 in the three windows before this one.
  - By the side's own faction it is too high for Knalgans (+0.12), Drakes
    (+0.10) and Loyalists (+0.09) and too low for Northerners (−0.09),
    each ± 0.02, as in the earlier windows.
- None of it is a verdict: the matches decide.

**Departure 1: the reading closes at round 544** (written 2026-11-25
00:30, before round 545).
- **Why:** on 2026-11-24 the user approved the new inputs, ordered the
  attack from a distance removed, and said: "Continue training after the
  network has gotten its improved visibility." Both take effect at round
  545 (`docs/input_facts_prereg_20261125.md`). A match after that round
  reads λ = 0.99 together with them.
- **What is read instead:** the five matches of rounds 480 to 544. They
  average +167.6 ± 6.1 Elo (scatter 16.5). That is +15.1 ± 7.1 against
  the fifteen before the switch (+152.5 ± 3.5): 2.1 standard errors
  above.
- **Kill criterion 2 at round 544: not met.** Its line is +131.3 with
  these five matches' standard errors.
- **What it says:** in 80 rounds at λ = 0.99 the loop did not lose
  strength, and it read 15 Elo higher. As registered, that is not shown
  to come from λ: no run at λ = 1 played beside it, and five matches are
  a third of the fifteen the verdict asked for.
- **What follows:** λ stays at 0.99. The user said "Continue training";
  nothing in the readings asks for a return to λ = 1. The five matches
  are the baseline of the next registration.
