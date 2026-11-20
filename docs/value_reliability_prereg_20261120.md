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

Not read yet.
