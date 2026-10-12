# Pre-registration: self-play RL in rounds, the advantage-tilted target (2026-10-12)

Written by the project lead before any code or box for it.

## Where it starts

Training on its own trajectories toward its memoryless reading
(docs/memory_own_trajectories_prereg_20261011.md, docs/own_confirm_prereg_20261011.md,
docs/memory_own_round2_prereg_20261011.md) fixed the memory and gave the
reference `own1` (+88.6 +- 12.7 over `parity3`); a second round left the
memoryless policy where it was (-6.1 +- 12.3), and within `own2` the memory
no longer costs strength (+13.0 +- 12.3). That target cannot teach the policy
anything it does not already do: a better target must carry information the
policy lacks. The one information source every self-play method shares is
the game's result.

## The method

Each iteration k, starting from `own2`:

1. **Games.** `own_k` at 64 slots against itself, under the reference decode
   with a temperature of 0.5 (the raw player's sampling on top of the
   end_turn offset), N games recorded whole. At 0.5 the player keeps its
   strength (`relset` at 0.5 against 0: 22-18 over 40 games, CLAUDE.md
   archive) and plays alternatives to its argmax, so the results say
   something about actions it would not otherwise take.
2. **Advantages.** Along each game-side, the value head of `own_k` with its
   memory carried at every decision; the reward is the result (+1, -1) at the
   side's last decision, a capped game bootstrapping from the value at its
   last decision (Pardo et al. 2018); the advantage of each decision is
   GAE with lambda 0.95 and gamma 1 over the side's own decisions (Schulman
   et al. 2016). sigma is the standard deviation of the advantages over the
   iteration's positions.
3. **Targets.** At each own position, `own_k`'s raw memoryless distribution
   (the top 16, renormalised), the played action's mass multiplied by
   exp(clip(A / sigma, -1, 1)) and the whole renormalised (the played action
   added with its own prior when outside the 16). The tilt moves the target
   by at most tanh(1/2) = 0.46 in total variation (Muesli, theorem 4.1), and
   in expectation toward the actions that won more than the value expected.
4. **Training.** The pipeline of rounds 1 and 2, from `own_k`: equal numbers
   of own and human positions per update, own policy weight 1, the value
   head trained on own outcomes as before, a pass of M own and M human
   positions, the learning rate 2.8e-5 falling linearly to 0 over the pass's
   second half. The result is `own_{k+1}`.

Three iterations in one box (`own3a`, `own3b`, `own3c`), each on its
predecessor's games only. N and M are set from the measured rates so an
iteration takes about 3 box-hours: N about 1,200 games (about 2,400
game-sides, 650,000 positions), M 600,000.

Why this and not a search teacher: no evaluator a fair player can use has
ranked candidate turns better than material (docs/selfplay_program_20261008.md
"Step 1, measured"), and the material look-ahead played level with the
policy; the result of the game needs no evaluator, only a baseline, and any
baseline leaves the advantage's expectation unbiased.

## Measurements

After the third iteration, PURE, both sides at `raw:t0+eo-1.5` (argmax, the
temperature is for generation only), 800 decisive games each:

| match | side A | side B | seed base |
|---|---|---|---|
| G1 | `own3c`, 64 slots | `own2`, 64 slots | 117000 |
| G2 | `own3c`, 64 slots | `own1`, 64 slots (the reference) | 118000 |

and after the first iteration, read as a learning-curve point, not a
verdict: G0, `own3a` against `own2`, both at 64 slots (seed base 119000).

Crash barriers per iteration: the holdout imitation loss at 0 and 64 slots
within 0.05 of the iteration's start; the generation match's games all
recorded (none failed); the replay at least 95% at temperature 0 on the
held-out sides (the replay reads the argmax, so at temperature 0.5 it is
read as the share of decisions that were the argmax, reported, not a
barrier); the mean |A / sigma| and the share of positions whose played
action is not the prior's argmax reported.

## Predictions (the lead's)

G1 +30 Elo (-10 to +70). G2 +40 (0 to +85). G0 +10 (-15 to +35). The share
of played actions that are not the argmax at temperature 0.5: 0.25 (0.15 to
0.40).

## Readings

- **RL works:** G1 at +25 or more. Self-play from the game's result improves
  the policy at this scale; the rounds are the program's engine, scaled by
  games per iteration and iterations, each block gated against the
  reference.
- **Too weak at this scale:** G1 within two standard errors of 0. The
  outcome signal from about 7,000 games over three iterations does not move
  the policy measurably; the next step is a teacher with more information
  per position (rollouts or a search), not more of these rounds unpriced.
- **Harmful:** G1 at -25 or below; the cause is found before anything else
  is trained.
- **G2 at +25 or more:** `own3c` becomes the reference.

## Cost

About 3 box-hours an iteration from the measured rates (a generation match
of 1,200 games at temperature 0.5 about 20 minutes; targets and advantages
at 390 positions a second about 30 minutes; the pass of 1.2 million
positions at 217 a second about 90 minutes; the corpus once), three
iterations and three matches: about 10 box-hours, $4.2-6.3 at
$0.42-0.63/h. BOX_MAX_H 15.
