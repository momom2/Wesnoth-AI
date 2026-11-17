# The simple loop: pre-registration (2026-11-17)

**Status: registered before the launch.** The user ruled on ten training
mechanisms between 2026-11-12 and 2026-11-17
(`docs/training_dynamics_theory_20261111.md`, "The user's decisions"),
was shown the build's defaults, and answered "Launch training."
(2026-11-17). This document states what runs, what is measured, the
numbers that decide and the kill criteria. The result lands here whatever
it says.

The code is on `feature/simple-loop` (not merged; the commit the run
launches from is named under "The launch").

## What runs

One round is one optimizer step, on the desktop's RX 6600
(`scripts/simple_loop_local.py`).

1. **Play.** The current network plays 128 games at temperature 1 with no
   end_turn offset, decisions logged: 102 against itself and 26 against
   one past self, drawn uniformly from a pool. The pool is the start
   network and every checkpoint a match has read. Maps and factions are
   drawn as in every match (the 21-map Ladder pool, fog, no forced
   faction). A game without a winner is kept.
2. **Pass** (`tools/pg_game_pass.py`). Over every game-side the current
   network played (230 of the 256), the gradient of

       sum_t (z − v_t) log π(a_t | h_t)  −  sum_t CE(value head, z)

   - z is the side's result: +1, −1, or 0 without a winner.
   - v_t is the value estimate logged when the decision was taken. Nothing
     is fitted, and no dice-luck or map term enters.
   - The memory's gradient is not cut: a group of game-sides runs from
     each side's first decision to its last as one graph.
   - The value head is trained on z at every decision, on the trunk's
     output detached, so only the value head (and the material
     projection) receive that gradient.
   - The gradient goes to every parameter.
3. **Step** (`tools/pg_adam_step.py`). One Adam step: β1 0.9, β2 0.999,
   ε 1e-8, no weight decay, no clipping, no schedule, one learning rate
   for every parameter. The optimizer's state is carried from round to
   round.

**The start network** is round 22's (`rounds_local\round_22\pi.pt`), the
strongest held: +114.7 ± 6.5 Elo over `own2` at the reference's decode
(departure 49 of `docs/stage_c_rounds_local_prereg_20261030.md`).

**The learning rate is 1e-5.** It is the one number expected to be tuned,
and a change is a numbered departure below.

- Where it comes from: the low end of the usual range for fine-tuning a
  transformer with Adam (a convention, not a measurement here).
- How it compares with the one step read in games. The match test's
  `full8` candidate moved every parameter by 5.888e-4 times stage C's
  per-weight-normalised direction d, whose norm is 32.69
  (`trunk_step_match_20261111\candidates\candidates.json`): a move of
  norm 0.0192, read at +52.5 ± 6.2 Elo at T=1.
  - d is the pass's mean gradient over the root of its mean squared
    window gradient, times its windows over twice its games (14,967 over
    40,000, the factor `pg_step.second_moment` leaves free). So a
    parameter's signal-to-noise ratio per window is d/0.374.
  - A round's 230 game-sides hold about 56,000 decisions, about 110 of
    those windows (508 decisions each), so its batch's ratio is √110
    times that: 28 d.
  - Where noise leads, Adam moves a parameter by the learning rate times
    its batch's ratio. A round's move along the signal is then 28 × lr ×
    |d| = 916 lr: 0.0092 at 1e-5, **0.48 of `full8`'s move**. They are
    equal at 2.1e-5.
  - Its noise part: Adam's averaged gradient (β1 0.9) leaves about 0.23
    lr on each of the 13.0 million parameters that have a gradient, a
    norm of about 830 lr, about as long as the signal's.
- What this does not cover: d was read at `own2`, under the labels of the
  time and the 16-decision cut; the whole-game gradient's noise is not
  read; and Adam's first step moves every parameter by the full learning
  rate (a norm of 0.036, 1.9 times `full8`'s, mostly noise).

## What is measured

**The reading is a match** (`scripts/match_local.py --sampled-candidate`),
before the first round, after round 6 (an early look at the learning
rate) and after every 16th round (about four a day at the measured 23
minutes a round):

- the network samples at temperature 1 with no end_turn offset, as it
  trains; the reference `own2` plays its pinned decode (`raw:t1` against
  `raw:t0+eo-1.5`);
- 800 decisive games from seed 900,000, the same seeds every time, fp16
  serving, fitted on decisive games (`tools/elo_collect.py`).

The reading starts below zero: a sampling player is weaker than a
top-choice one (415 Elo for `own2`, about 140 by round 15,
`docs/status.md`). Its change over the run is what counts.

**It is not an adoption claim.** It mixes the weights and the decode. A
candidate for the reference needs its own PURE match at the reference's
decode (CLAUDE.md, "Strength claims").

**The number that decides.** After the match at round 160 (ten matches
past the first, not counting round 6's; about 20,500 games and 2.6 days):

- **the loop trains** if the mean of the matches at rounds 128, 144 and
  160 is above the first match by more than 3 standard errors of that
  difference;
- otherwise the result is "rejected: at 1e-5 the simple loop did not gain
  in 160 rounds", and what follows is the user's to decide.

The run goes on past round 160 either way, until a kill criterion or the
user's word.

For scale, the rounds it replaces gained about +0.008 in score for each
5,000-game round at argmax (`docs/status.md`).

**Logged every round, never a verdict:** the gradient's norm and the
update's, the value head's cross-entropy and squared error, the replay's
readings, the pass's rate and peak memory, the loss scale's overflows.

## Kill criteria

1. **Two matches in a row read more than 3 standard errors below the
   first** (the difference's standard error: the two matches' in
   quadrature). The run stops and reports.
2. **An exact check fails:** the pass did not train every logged decision
   of its game-sides, or its replay of the games is not the law that
   played (half the mean squared difference of the log probabilities
   above 1e-4).
3. **A gradient is not finite.** It cannot be applied; the step refuses.
4. **A display-driver reset** during a GPU tool (the card drives the
   display).
5. **A round past 1.5 times its estimate** is inspected (the 2026-09-05
   ruling); the tools' own cuts are 45 min for a round's games, 90 for
   its pass and 120 for a match.
6. The user's word, or a STOP file in the run's directory.

**The blow-up warning stops nothing** (user, 2026-11-17: "With a warning
logged if we detect blowup, in which case, we'll revisit it."). A line
goes to `warnings.log` when a round's gradient norm is more than 10 times
the median of the rounds before it (from the sixth round on), or when a
group's gradient needed a loss scale under 1.

## Measured before the launch

The whole-game pass was run on round 22's games (played by round 21's
network, `rounds_local\round_22\play\games_s00`) before anything else was
built on it.

**A first probe, 16 games** (32 game-sides, 7,878 decisions, groups of at
most 2,048 decisions):

- peak allocated memory 1,000 MB; no overflow at the loss scale 1,024
  over chains of up to 488 steps; every gradient finite;
- a time step costs about 180 to 200 ms whether it holds 4 rows or 8, so
  the rate follows the rows a step holds: 21.5 decisions a second at 3.9
  rows a step, 41.7 at 8.3;
- the replay's mean difference read 1.1e-4 and its second-order reading
  5.8e-6 (round 22's pass over 2.45 million decisions: 6.7e-6 and
  5.7e-6). The mean's sampling error at this size is of the limit's
  order, which is departure 1 below.

**The round-sized measurement, 128 games** (256 game-sides, 66,036
decisions, 10 groups of at most 32 game-sides and 8,192 decisions;
`wesnoth_runs\simple_loop_fit\full_8192\grad.json`):

- **It fits:** 2,796 MB allocated at the peak. The card held about 5.6 GB
  in all while the largest group ran, of its 8.
- **The rate is 60.2 decisions a second:** 1,102 s for the pass, against
  85 a second under the 16-decision cut. The longest game-sides are the
  slow ones: the first group (13 game-sides of up to 995 decisions) ran
  at 32 a second, the groups of 32 game-sides at 75 to 84.
- **No overflow** at the loss scale 1,024, and every gradient finite.
- **The replay:** the mean difference −7.8e-6 and the second-order
  reading 5.8e-6, against a limit of 1e-4.
- **The value head before any training:** cross-entropy 0.394, squared
  error against the result 0.529.
- **The gradient's norm** is 52.3 for the 256 game-sides and was 153.6
  for 32: it falls as one over the root of the game-sides, as noise
  does. The longest group's norm per game-side is 586, against 118 to
  291 for the others: 5% of the game-sides carry about 23% of the
  groups' squared norms. No group is 10 times the others.

**So a round takes about 23 minutes** (3 for its games, 18.5 for its
pass, the rest for the step and the tools' starts; the games' and the
step's parts are estimates until the first rounds are timed): about 62
rounds and 8,000 games a day, where the round's size was chosen at an
estimated 17 minutes. The size stays 128, the user's number to revise.

## Departures from the defaults stated to the user

These were settled while building, before the launch.

1. **The replay identity is held on its second-order reading** (half the
   mean squared difference of the log probabilities, at most 1e-4), not
   on the mean difference. Over one round's decisions the mean's sampling
   error is of the limit's order: it read 1.1e-4 on 7,878 decisions whose
   second-order reading was 5.8e-6. The mean is still logged.
2. **No precision floor is read in a round.** The default said "read once
   per run". Round 22's stands (4.0e-6 under float16 serving), and the
   limit is not lifted by it.
3. **An early match after round 6.** The learning rate is the least
   grounded number, and a first reading after 6 rounds costs one match.
4. **A group of the pass holds at most 8,192 decisions** as well as 32
   game-sides: the bound on the card's memory.
5. **The value head trains on every decision, a game without a winner at
   z = 0,** as that game's policy weight is 0 − v_t.
6. **Both players of a past-self game sample at temperature 1.** Only the
   current network's side is trained.
7. **A match every 16 rounds, not 24.** The default said "every 24 steps
   (about four times a day)". At the measured 23 minutes a round, 16
   keeps the four a day, and the kill criterion's two matches are about 6
   hours apart.

## The launch

- **The code:** `feature/simple-loop` at 23e4dff8, not merged into
  `main` (that is the user's call). The run executes from the detached
  worktree `..\wesnoth_ai-run-simple` at that commit.
- **In place of CI** (nothing can be pushed: no GitHub credentials are
  stored on the desktop): the full suite on the desktop at 23e4dff8, both
  tiers (`pytest -m ""`): 2,262 passed, 139 skipped, none failed, in
  2,103.97 s. Ruff is clean.
- **A two-round smoke of the real tools** (`wesnoth_runs\simple_loop_smoke`:
  16 games a round, matches of 16 games): every tool ran in order. Round
  2's pass replayed the games the stepped network played (second-order
  reading 4.9e-6). The first Adam step's update norm read 0.0357, the
  0.036 worked out above.
- **The command,** from the run worktree, started detached through WMI:

      set HIP_VISIBLE_DEVICES=1
      <.venv>\python.exe -u scripts\simple_loop_local.py --out C:\Users\momom2\wesnoth_runs\simple_loop
          --start C:\Users\momom2\wesnoth_runs\rounds_local\round_22\pi.pt --lr 1e-5 --early-match 6
          --python C:\Users\momom2\venvs\wesnoth_rocm\Scripts\python.exe --tools-python <.venv>\python.exe
          --keep-awake

  The same command again resumes the run from its markers. Its console
  goes to `wesnoth_runs\simple_loop.console.log`.
- **To stop it:** a file named STOP in `wesnoth_runs\simple_loop`.
- **Started 2026-11-17 16:33:49** (the launching `cmd.exe` is pid
  27620). The match at step 0 was playing two minutes later, 43 games
  in.

## Results

Filled in as the run goes.
