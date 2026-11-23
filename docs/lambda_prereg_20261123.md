# A weight with λ = 0.99: pre-registration (2026-11-23)

**Status: registered, not switched yet.** The simple loop
(`docs/simple_loop_prereg_20261117.md`) runs at λ = 1 until its match at
round 464 has been read. Round 465 is planned as the first at λ = 0.99.

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

## Results

Filled in as the run goes.
