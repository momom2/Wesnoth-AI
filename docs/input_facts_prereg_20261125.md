# The new inputs and the end of the attack from a distance: pre-registration (2026-11-25)

**Status: switched. Round 545 is the first on the new inputs** (2026-11-25
01:36, from the branch `feature/input-facts` at 2a523b56). **Not merged
into `main`: that is the user's to decide.** The simple loop (`docs/simple_loop_prereg_20261117.md`) stopped
at the round boundary after its match at round 544 (2026-11-25 00:12:28).
This is its departure 10.

The user's words, 2026-11-24:
- "Alright, remove attack from a distance. For the rest approved." (on
  `docs/input_facts_design_20261124.md`)
- "Continue training after the network has gotten its improved
  visibility. Hopefully, that will fix the wrong weapon choice issues. No
  special learning rate, I need to go to sleep and can't discuss them
  now; maybe we'll reconsider tomorrow if the night's training is clearly
  flawed or stalled."
- "Also, make sure the network has ToD visibility like a human: globally
  and per hex."

This document states what changes, what is measured, the numbers that
decide and the kill criteria. The results land here whatever they say.

## What changes

Two things, at one round boundary.

1. **A unit attacks only an enemy on a hex that touches its own.** The
   legal actions no longer hold an attack on an enemy the unit could
   reach that turn, and the simulator no longer moves a unit before its
   attack (`rust/wesnoth_core/src/lib.rs`, `rows_from_landable`;
   `tools/wesnoth_sim.py`). This holds for every player: the network,
   its past selves and the reference.
2. **The network reads 232 more numbers of each unit** (the unit row of
   version 2, `wesnoth_ai/encoder.py`):
   - its level, its chance to be hit where it stands and on each of the
     16 kinds of terrain, and its movement costs (34 numbers, for every
     unit it sees and every recruit option);
   - for each of its own units that can still attack, each enemy on a hex
     that touches it and each of the unit's weapons: the damage of a hit,
     the number of strikes and the chance to hit, for the weapon and for
     the enemy's answer, and which of the enemy's weapons answers (198
     numbers).

   The new numbers go through a projection of their own
   (`unit_feat_proj2`), which starts at zero. The network at the switch is
   therefore exactly round 544's network.

With them, and not factors of play: the run plays from the branch's
worktree (`C:\Users\momom2\wesnoth_worktrees\input-facts`) on the core of
phase 40; the observation epoch is 12 and a game record's format is 6.

## What does not change

- **The learning rate:** 1e-5, one for every parameter, with Adam as
  before ("No special learning rate"). Adam's state carries over. It
  meets the new projection as a parameter it has never stepped, which is
  its ordinary treatment of one.
- **λ = 0.99,** the games, the pool of past selves, the matches and their
  seeds, `config.json`.
- **The reference:** own2 at its pinned decode, reading the 109 columns
  it was trained on. A past self of the pool reads its own rows too.
- **How the weapon is scored.** It is still scored from the attacker's
  token alone, whichever enemy is struck (the design's Part 3). That
  change is at the output and has not been put to the user. With one
  enemy next to the attacker (1,888 of 2,196 measured attacks by a
  two-weapon unit) the attacker's token now holds that fight's numbers
  for each weapon. With several (308) it holds every fight's numbers, but
  the weapon's score still cannot depend on which enemy was chosen.

## The time of day: already in the input

Checked on the user's request. Nothing had to be added.

| What a player sees | Where the network reads it | Test |
|---|---|---|
| The time of day now, of the six | Two numbers of the global token: this turn's lawful bonus and the next turn's. The pair tells the six apart (dawn is 0 then +25, dusk 0 then −25, and so on). | `tests/test_time_of_day_features.py` |
| The time of day on a hex (a cave's permanent night, a lit hex, a hex beside a unit that illuminates) | A number of each hex token: the hex's lawful bonus less the board's, as the interface shows it. | `tests/test_parity_observation.py`, `test_statuses_time_of_day_and_leadership` and `test_a_hexs_time_of_day_is_the_interfaces` |
| What it does to a unit | A number of each unit's row: the bonus its alignment gives it on its hex, now. | the same first test |

Both are in the network's input since 2026-09-22 (the global pair) and
2026-09-30 (the hex and the unit). The new fight numbers also carry it:
a hit's damage is after the time of day.

## What is known before the switch

- **The reference does not use the attack from a distance.** At its
  pinned decode it ordered 0 in 1,660 attacks (16 games against itself,
  seeds 20261202 to 20261217, on the CPU). Round 513's network ordered 8
  in 2,637. So the removal leaves the yardstick's play as it was.
- **The new numbers are the fight's own.** `tests/test_unit_rows2.py`, on
  the branch:
  - on hand-built positions, the level, the defences and the movement
    costs equal `unit_stats.json`, and a fight's line equals the engine's
    arithmetic worked by hand (a Spearman on a Skeleton at night: 2 a
    hit, three strikes at 60%, answered by 9 a hit);
  - on played games, every line equals the core's fight statistics, and
    every attack then played struck as its line said, strike by strike
    from the simulator's record of the fight;
  - no attack is offered on a hex that does not touch the attacker's;
  - a checkpoint given the new projection gives its source's output to
    the last bit, and the loop's tools play it, pass it (the exact check
    holds) and step it.
- **The night cannot show a gain.** Adam moves a weight by at most the
  learning rate a round: 1e-5. About 25 rounds fit before the morning, so
  a new weight is at most 2.5e-4 by then. The weights that read the
  present columns average 0.07. The night can show that nothing broke,
  and how fast the new weights move.

## What is measured

1. **The switch itself.** Round 544's network at the switch plays the
   loop's match twice: the loop's own match before the switch (+179.2 ±
   13.9, 590 of 800), and the same match after it, by the widened network
   on the new code. D0 is the second less the first. The two networks
   compute the same function, and the reference orders no attack from a
   distance, so D0 measures the removal for round 544's network, which
   is expected near 0. Its standard error is about 19.7.
2. **Every round,** as before: the exact check (the pass reproduces the
   probabilities the players logged, under 1e-4), the gradient's and the
   update's norms, the value head's cross-entropy.
3. **The new weights,** at each match: their root mean square and
   largest size (`unit_feat_proj2`), beside the old projection's.
4. **The matches,** unchanged: after every 16th round, 800 decisive
   games.

Only the matches decide. The rest are crash barriers.

## The numbers that decide

**The baseline** is the five matches at λ = 0.99 before the switch (rounds
480 to 544): +139.8, +178.1, +164.9, +175.8 and +179.2, mean +167.6 ± 6.1
(scatter 16.5).

**After the match at round 672,** the eighth after the switch (about
2026-11-26 16:00): D is the mean of the eight matches of rounds 560 to 672
less the baseline, with the standard error of the difference (about 7.8).

- **D under −2 standard errors: "the loop lost strength after the
  switch".** The user is told; what follows is theirs to decide. The
  default is to keep the run going meanwhile: the inputs were decided on
  principle.
- **Otherwise:** D above +2 is recorded as "the loop gained after the
  switch", D within ±2 as "no change read in 128 rounds".

**What the numbers cannot say.** No run without the new inputs plays
beside this one, so a gain would not be shown to come from them. Whether
the new numbers are used is read from the weights (measurement 3) and,
later, from the attacks themselves (the audit's counts of beaten weapons
and attacks into resistance, `training/metrics/input_audit_20261124/`).

## Kill criteria

1. **The switch changes the player:** D0 is outside ±3 standard errors
   (about 59 Elo). Round 545 does not start on the new code; the run goes
   back to `main` and the old core from round 544's own files, kept
   beside the widened ones, until it is explained.
2. **A round does not check out:** its exact check is over 1e-4, or the
   pass or the step refuses. The runner stops there by itself. If it is
   not explained, the run goes back as in 1.
3. **"Clearly flawed"** (the user's words): the match at round 560 reads
   more than 3 standard errors below the baseline (below about +122), or
   the mean of the matches at 560 and 576 does (below about +131). The
   run stops and the user is told.
4. **The loop's own,** unchanged: two matches in a row more than 3
   standard errors below the run's first, a gradient that is not finite,
   a display-driver reset, a round past 1.5 times its estimate, the
   user's word or a STOP file.

"Stalled" is not a kill criterion: by the arithmetic above the new weights
cannot matter to play in one night. What the night shows of their pace is
reported for the discussion the user set for the morning.

## The switch

1. The run stops at the round boundary (a STOP file placed during the
   match at round 544: done, 2026-11-25 00:12:28).
2. The branch's tests pass on the desktop and in CI.
3. Both venvs take the wheel built from the branch (phase 40).
4. `round_0544\pi.pt` and `adam.pt` are copied to `pi.rows1.pt` and
   `adam.rows1.pt`, then replaced by their widened forms
   (`tools/widen_unit_rows.py`); `pool.json` takes the widened file's
   hash for round 544.
5. A one-round smoke of the real tools on the GPU, in a directory of its
   own, from the widened checkpoint.
6. The match of measurement 1.
7. The run resumes from the worktree, with the launch's command and
   `--lam 0.99`.

## Results

**Before the switch, on the GPU, from the branch's wheel read through an
overlay folder (nothing installed yet)** (2026-11-25 00:24 to 00:47, in
`C:\Users\momom2\wesnoth_runs\rows2_smoke`):

- **One round of the real tools from the widened network** (16 games):
  every tool ran in order, and the pass replayed the games within the
  exact check (6.6e-6 against the limit of 1e-4; the run's rounds read
  5e-6 to 8e-6).
- **The server and the pass read the new columns alike.** A copy of the
  widened network had its new projection drawn at random at the old
  projection's scale, so that the new columns move its play. It played a
  round on the GPU and its pass replayed it within the exact check
  (8.0e-6). On 100 of its decisions, the probability its player logged
  is the one recomputed on the CPU with the new projection (mean
  difference 0.0015 in log probability) and not the one without it
  (0.22). This check exists because a zero projection hides a path that
  leaves the new columns out.
- **Adam's state carries to the real network:** a step from round 544's
  state on the widened checkpoint is step 545 for the old weights and
  step 1 for the new projection, whose weights each move by 1e-5 at most.

**Measurement 1, the switch match: kill criterion 1 is not met.**

| Round 544's network | Elo | Wins of 800 | Read |
|---|---|---|---|
| before: the loop's match, on `main` | +179.2 ± 13.9 | 590 | 2026-11-25 00:12 |
| after: widened, on the branch's code | +164.9 ± 13.7 | 577 | 2026-11-25 00:46 |

D0 = −14.3 ± 19.5 Elo, 0.7 standard errors, inside the ±3 the criterion
allows. The two matches play the same seeds but not the same games: the
candidate samples, and a game diverges at the first decision where an
attack from a distance was legal. Nothing here says the removal cost or
gained strength.

**The branch's gate** (2026-11-25 01:34), at 2a523b56:
- The full suite on the desktop, both tiers, against the branch's wheel:
  2,284 passed, 139 skipped, none failed, in 809 s. Ruff is clean.
- CI is green on the same commit (run 45517312709). The run before it,
  on the branch's first commit, was red on twelve tests, all of them
  fixtures: games of a tiny random network or of a random legal player,
  which used to fight through attacks ordered from a distance. They were
  re-seeded, with no assertion or threshold changed (commit 2a523b56).
  One of them now stands on a favourable seed (`BACKLOG.md`).

**The switch** (2026-11-25 01:34):
- Both venvs hold the wheel of phase 40, built from the branch's source
  (its source hash is the worktree's).
- `round_0544\pi.pt` is the widened checkpoint (SHA-256 `d735b652…`);
  the source is kept as `pi.rows1.pt` (`bcff00f8…`), and the optimizer
  state as `adam.rows1.pt`. Checked tensor by tensor: the source's
  weights are untouched, the new projection (384 by 232) is all zero,
  and the checkpoint holds the same tensors as the network that played
  the switch match. The optimizer state names the widened checkpoint and
  holds no moments for the new projection.
- `pool.json` holds the widened file's hash for round 544, with the
  source's beside it.
- The run's records carry `code_version` 0.35.6, the version `main` had
  when the branch was cut: a topic branch does not touch the version.
  The commit each tool ran from is in the runner's log.
- The run resumed at 01:35:16 from the worktree, with the launch's
  command and `--lam 0.99`; its tools log `code 2a523b56`.
- To go back (kill criteria 1 and 2): stop the run, copy `pi.rows1.pt`
  and `adam.rows1.pt` over `pi.pt` and `adam.pt` of round 544, restore
  `pool.json`'s hash, rebuild the wheel from `main` into both venvs,
  delete the rounds after 544 and their markers, and start the launch's
  command from the main checkout.

**Round 545, the first on the new inputs, checks out** (2026-11-25
01:52:36). Kill criterion 2 is not met.
- **The exact check:** 7.2e-6 against the limit of 1e-4, where the run's
  rounds read 5e-6 to 8e-6.
- **The round:** 230 game-sides, 57,560 decisions, no warning. Its games
  took 183 s and its pass 853 s (182 s and 848 s in round 544): the new
  columns cost no time that shows.
- **The two kinds of network played each other.** The round's past self
  was round 112's network, which reads the 109 columns.
- **The gradient's norm** is 64.7, of which the value head's part is 54.8
  (48.2 in round 544; 14 to 89 over rounds 497 to 528).
- **The update's norm** is 0.0069, where rounds 543 and 544 read 0.0061.
  The difference is the new projection's first step: 97% of its 89,088
  weights moved, each by 1e-5, which is 0.0029 in norm. Adam's first step
  of a weight is the learning rate whatever the gradient's size, so this
  says only that the gradient reaches them. The 3% that did not move
  read columns that were 0 in every position of the round.
- **The new weights' size** after one round: 9.8e-6 root mean square, in
  every block of columns (level, defences, movement costs, fights). The
  old projection's is 0.104.

What the next rounds can show is whether the new weights keep moving the
same way (1e-5 a round) or turn back and forth: read at each match with
`training/metrics/input_facts_20261125/new_weights.py`.

**The matches after the switch** (baseline +167.6 ± 6.1):

| After round | Elo | Wins of 800 | Read |
|---|---|---|---|
| 560 | +157.4 ± 13.6 | 570 | 2026-11-25 06:29 |

**The match at round 560: kill criterion 3's first half is not met.** It
reads 10.2 under the baseline, 0.7 of the difference's standard error
(14.9), where the criterion asks for under about +122. For its second
half (the mean of 560 and 576 under about +131), the match at round 576
would have to read under about +105.

**Rounds 545 to 560, the first sixteen on the new inputs** (01:35 to
06:17, 17.6 minutes a round, as before the switch). Kill criterion 2 is
not met in any of them.
- **The exact check** reads 6.8e-6 to 7.7e-6 in the sixteen rounds, against
  the limit of 1e-4. Every gradient is finite, and the console log holds
  no warning.
- **The gradient's norm** runs from 34 to 73, **the update's** from
  0.0069 in round 545 down to 0.0060 to 0.0063 from round 550 on (0.0061
  before the switch), and **the value head's cross-entropy** from 0.353
  to 0.455 (0.34 to 0.46 over rounds 497 to 528).

**Measurement 3, the new weights at round 560**
(`training/metrics/input_facts_20261125/new_weights_round_0560.json`):

| Columns | Root mean square | Largest |
|---|---|---|
| level | 4.3e-5 | 1.08e-4 |
| defence where it stands | 4.0e-5 | 1.07e-4 |
| defence by terrain | 3.9e-5 | 1.31e-4 |
| movement costs by terrain | 3.9e-5 | 1.25e-4 |
| fights | 4.7e-5 | 1.43e-4 |
| all 232 | 4.6e-5 | 1.43e-4 |

- **Against the old projection:** its root mean square is 0.1043 at
  round 544 and at round 560. The new weights are about 2,300 times
  smaller.
- **How they move.** A weight whose gradient kept its sign for sixteen
  rounds would stand at 1.6e-4, and the largest is at 1.43e-4, so some
  weights are pushed one way round after round. The average weight
  stands 3.7e-5 from zero, 0.23 of the learning rate a round. For
  comparison, a weight stepping the full rate in a direction drawn afresh
  each round would stand 3.2e-5 from zero on average. So for most of the
  new weights the gradient does not keep its sign from round to round.
- **98% of them have moved** (97% after round 545).
- **What it says of the pace.** The whole projection grew from 9.8e-6
  after one round to 4.6e-5 after sixteen: 4.7 times for 16 times the
  rounds, close to the square root (4). At that law it reaches a
  hundredth of the old projection's size (0.001) after about 7,500
  rounds, three months of this run; the few weights that march steadily
  get there in 100 rounds. Nothing here says what size the new weights
  need before they change play. This is the reading for the discussion
  the user set: the night was not flawed, and by the one learning rate
  the new inputs cannot yet count.
