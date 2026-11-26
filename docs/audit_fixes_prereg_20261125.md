# Three fixes from the blind-spot audit, and a fourth: pre-registration (2026-11-25)

**Status: switched. Round 609 is the first under the four changes**
(2026-11-25 22:49, from the branch `feature/advancement-choice` at
bec71da6; departure 1 below). **The GPU checks passed, and the switch
match reads +22.0 ± 19.5 against the loop's own match of the same network:
kill criterion 1 is not met** (Results). **The first two matches from
the switch read +161.7 ± 13.6 (round 624) and +212.0 ± 14.6 (round 640),
+186.9 ± 10.0 together; rounds 609 to 640 check out.** Six matches are
left before the registered verdict (round 736).
Both branches are in `main` since 2026-11-26, merged on the user's word
("Commit, merge and push."): `feature/audit-fixes` at 0.37.0 and
`feature/advancement-choice` at 0.38.0. The user answered the question of
the switch's time with "Otherwise, approved", on a message that
recommended that evening; that reading was put back to them. It is
departure 11 of the simple loop (`docs/simple_loop_prereg_20261117.md`).

**The gate** (2026-11-25 15:17), at e2107544:
- The full suite on the desktop, both tiers, against the branch's wheel:
  2,301 passed, 139 skipped, none failed, in 959 s. Ruff is clean.
- CI is green on the same commit (run 45629475998).
- The run before it failed 7 tests in four places: two faults of the new
  code (an output built by hand has no pointer fields; the simulator set
  the stop rule after its core was built), fixed, and two fixtures of
  played games that change with the rule, re-recorded or re-sized with no
  assertion changed (commit e2107544 says how).
- The tool that gives a checkpoint the pointer was run on a copy of round
  588's checkpoint and optimizer state: it names the 3 new parameters
  after the model's others and re-indexes the encoder's moments, checked
  against the network it builds.

The user, 2026-11-25, on `docs/blindspot_audit_20261125.md`:
- "Gap 3 absolutely needs to be fixed."
- "Stopping moves when an enemy comes into view: Do it."
- "The network shouldn't be told the enemy's faction from the first
  decision, only from the first moment it sees an enemy unit (more
  precisely, it should receive information about which subset of factions
  the enemy might be, starting at all 6, then being cut if not 1 each
  time it sees a new unit [...])."

## What changes at the switch

1. **A move stops where it brings an enemy into view.** A player's move
   under fog stops on the first hex where a stop is harmless once a step
   has shown an enemy, and the unit keeps the movement it has left. This
   is what Wesnoth does for a player (`src/actions/move.cpp:548-566`,
   `576-587`, `955-957`; `src/actions/vision.cpp:281-290`, 1.18.4;
   `docs/wesnoth_rules.md` has the quotes). It holds for every player of
   a game: the network, its past selves, the reference.
2. **The enemy's faction is known as a player knows it.** A game set up
   by a random draw is a game of two Random picks. Each side's faction
   input starts at the era's six factions and is cut by each unit type
   the side sees: to two for a Mage (Loyalists or Rebels), to one for
   most. The input itself is unchanged (`wesnoth_ai/faction_posterior.py`);
   the games used to be set up as two open picks. It holds for every
   player of a game.
3. **The weapon's score reads the target.** The trained network gains a
   pointer's term, the form the target's own score has: one query per
   weapon slot from the attacker's token against one key from the target
   hex's token (`wesnoth_ai/model.py`, `weapon_target`). The key's map
   starts at zero, so the network at the switch scores as before to the
   last bit. The reference and the pool's past selves keep their own
   scores.

In the code: core phase 41, rules version 41, observation epoch 13, game
record format 7, checkpoint key `weapon_target`.

### Departure 1 (written 2026-11-25 17:30, before the switch): a fourth change joins it, if its gate passes

The user approved the audit's proposal for gap 2 the same day ("Otherwise,
approved."), and it was built on `feature/advancement-choice`, a branch cut
from `feature/audit-fixes`. **If that branch is green in CI before the
switch, the run switches to it and takes the four changes at once.** One
switch breaks the match series once, where two would break it twice. If it
is not green in time, the switch is to the three and this change waits for
a later match.

4. **The advancement is the side's choice on its own turn.** A unit that
   levels up on its own side's turn with several types to become waits
   after the fight. The side's recruit rows then hold those types, placed
   on the unit's hex, and nothing else is legal; the network picks one
   with the head that picks a recruit's type. On the enemy's turn the
   unit takes the first type listed, as Wesnoth 1.18 gives a player
   (`docs/wesnoth_rules.md`, "Who chooses an advancement, and what an
   unasked unit becomes"). It holds for every player of a game. Until now
   every such unit took the first type: 0.65 a game are now choices.

In the code it adds: core phase 42, rules version 42, observation epoch
14, game record format 8. No new head, token kind or column.

What is known of it before the run:
- The network has never scored such rows as recruits: they are types no
  side recruits. What it picks at first is whatever its recruit head gives
  them, so the first advancements are not chosen well. They are learned
  like any other decision of a game.
- Tested on a tiny network through the inference server and the pass, in
  games at 10% experience: the choices are logged, the records rebuild,
  and the pass reproduces every logged probability
  (`tests/test_advancement_choice.py`).
- The first full run of the suite on the branch found a match loop that
  asked again without end (the check for a recruit bouncing off an
  occupied hex took the choice for one); fixed and pinned before the push.

Measured in addition, at each match: how many advancements the round's
games held as choices, and the share that took a type other than the
first.

**Its gate passed** (2026-11-25 17:31), at bec71da6: CI is green on both
tiers (run 45652084128) and ruff is clean. The run before it, on a5ae4eb0,
failed 5 tests of 2,332, all of an advancement nobody is asked about: a
game under the earlier self-play's uniform draw now asks no side, a
scripted fight leaves no unit waiting, and one export test's simulator
plays without the asking (commit bec71da6; no assertion changed). The full
suite was not run again on the desktop after those fixes: the earlier run
there had slowed the training (`docs/input_facts_prereg_20261125.md`,
round 590). **So the switch is to `feature/advancement-choice`, with the
four changes.**

## What is known before the run

- **The matches do not chain across the switch.** Both players of a match
  are told less of the enemy's faction and both have their moves stopped,
  so the scale against the reference moves. How far is not known: the
  switch match reads it for the network at the switch.
- **The first two changes are not neutral at the switch.** The network has
  played every round with the enemy's faction certain from the first
  decision. It read uncertain factions in imitation only.
- **The third starts at zero.** A network given the pointer plays its
  source's games command for command
  (`tests/test_weapon_target.py`). Adam meets the two maps as parameters
  it has never stepped: the keys move by the learning rate at most at
  their first step, the queries only once the keys have moved.
- **Zero or random for the new weights** (the user asked on 2026-11-25:
  "might as well randomize the new parameters instead of setting them to
  0. If they're to 0, they will receive very little gradient and take a
  lot of time to train, won't they?").
  - A weight's gradient does not depend on the weight's own value: it is
    the input the weight multiplies times what reaches its output. Read
    from Adam's state at round 588
    (`training/metrics/input_facts_20261125/gradient_sizes.py`): the
    gradient per weight is 0.0053 (root mean square) at the projection
    that started at zero at round 544, 0.0098 at the old unit projection
    and 0.017 over the whole network. About half the old projection's,
    because the columns it reads are smaller and more often zero.
  - Adam moves a weight by about the learning rate whatever its
    gradient's size, so even that half does not slow it. The new
    projection moved as far in its first 32 rounds as the old one did in
    the loop's first 32 (`docs/input_facts_prereg_20261125.md`, "The
    yardstick").
  - The one place a zero does hold a gradient back is a product of two
    new maps: the weapon pointer's queries get none while its keys are
    zero. That is one step: the keys move at Adam's first step, and the
    queries from the second.
  - Random weights would move at the same pace, a learning rate a round.
    What they change is the start: the network would play with noise in
    every unit's token and in every weapon choice from the first game,
    and the switch match could no longer tell that noise from the rule
    changes.
  - What would make the new weights count sooner is a larger step for
    them, which is the one-learning-rate question the user set aside on
    2026-11-24.
- **Not checked against the engine by a replay:** the sighting stop is
  written from the engine's source and tested on hand-built boards. The
  replay export does not replay under fog, so a strict-sync replay cannot
  check it yet.

## What is measured

1. **The switch itself.** The network of the last round before the switch
   plays the loop's match twice: the loop's own match on the old code,
   and the same match after the switch, with the pointer at zero. D0 is
   the second less the first. It is not expected at 0: it is what the two
   rule changes do to this network against the reference.
2. **Every round,** as before: the exact check (under 1e-4), the
   gradient's and the update's norms, the value head's cross-entropy.
3. **The pointer's weights,** at each match: the keys' and the queries'
   root mean square.
4. **The matches,** unchanged in form: after every 16th round, 800
   decisive games.
5. **The attacks themselves,** once, on the network 128 rounds after the
   switch: the input audit's counts on 24 games
   (`training/metrics/input_audit_20261124/`), which read 1.0% of attacks
   by a two-weapon unit with a better weapon left unused, and 3.6% of
   attacks into a resistance of 40% or more, on round 513's network.

Only the matches decide. The rest are crash barriers and readings.

## The numbers that decide

The three changes were decided on principle; no number here undoes them.
What the numbers answer is whether the loop still trains after them.

- **After the eighth match from the switch:** the mean of matches 5 to 8
  less the mean of matches 1 to 4, with its standard error (about 9.8).
  Under −2 standard errors: "the loop loses strength under the new
  rules", and the user is told. Otherwise it is recorded as it reads.

## Kill criteria

1. **The switch costs more than can be explained:** D0 under −60 (about 3
   standard errors of the difference). The run does not resume. Each rule
   change is then switched on alone in a match of its own, to read which
   costs what, and the games are read for a fault before anything else.
2. **A round does not check out:** its exact check is over 1e-4, or the
   pass or the step refuses. The runner stops there by itself.
3. **A difference from Wesnoth is found in the sighting stop:** it is
   fixed before the run goes on (CLAUDE.md, Principle 4).
4. **The loop's own,** unchanged.

## What it does to the reading in progress

The new inputs' registration (`docs/input_facts_prereg_20261125.md`) asks
for the eight matches of rounds 560 to 672. A switch before round 672
closes it early, as the new inputs closed the lambda reading at round 544.

## The switch

1. The run stops at a round boundary (a STOP file placed during a match).
2. The branch's tests pass on the desktop and in CI.
3. Both venvs take the wheel built from the branch (phase 41).
4. The last round's `pi.pt` and `adam.pt` are kept beside their new forms
   and given the pointer (`tools/add_weapon_pointer.py`); `pool.json`
   takes the new file's hash.
5. On the GPU, in a directory of its own: one round of the real tools
   from the new checkpoint (the exact check holds); the same with the
   pointer's keys far from zero, with the logged probabilities compared
   against a recomputation with and without the pointer; Adam's state
   carried.
6. The match of measurement 1.
7. The run resumes from the branch's worktree with the launch's command.

## Results

**The loop's last match on the old code** (round 608, 2026-11-25 22:20):
+161.7 ± 13.6 (574 of 800). A STOP file placed during that match stopped
the run at the round's boundary, at 22:20:10.

**The GPU checks, from the branch's wheel read through an overlay folder
(nothing installed yet)** (22:20 to 22:49, in
`C:\Users\momom2\wesnoth_runs\audit_fixes_smoke`; the scripts are in
`C:\Users\momom2\wesnoth_runs\tools\audit_fixes_switch`):

- **One round of the real tools from round 608's network with the pointer
  at zero** (16 games, 4 of them against a past self): every tool ran in
  order, and the pass replayed the games within the exact check (6.9e-6
  against the limit of 1e-4; the run's rounds 593 to 608 read 7.0e-6 to
  8.0e-6).
- **The advancement is chosen in those games.** The smoke run's 48 games
  (its round and its two 16-game matches) hold 42 choices, 0.88 a game,
  and 28 of them (67%) took a type other than the first.
- **The server and the pass compute the pointer's term alike.** A copy of
  the network had its keys drawn from a normal law of scale 0.5, so that
  the term moves its weapon choices. It played a round on the GPU and its
  pass replayed it within the exact check (6.4e-6). On 180 of its
  attacks, the probability its player logged is the one recomputed on the
  CPU with the pointer (mean difference 0.0007 in log probability) and
  not the one without it (0.32). On 691 other decisions the two
  recomputations are the same (0.0020 from the log). This check exists
  because keys at zero hide a path that leaves the term out.
- **Adam's state carries.** The smoke round cannot show it: a run's first
  round steps from a fresh state (its update's norm reads 0.036). So a
  step was made apart with the real tools (`carried_step.cmd`): the pass
  over the smoke round's 4 games against a past self, then
  `tools/pg_adam_step.py` from round 608's optimizer state with the
  pointer's three parameters named in. It is step 609 for the old weights
  and step 1 for the keys: each of the keys' 24,576 weights moved by
  1e-5 and no query moved, as "What is known before the run" says. The
  update's norm is 0.0157, on a gradient of norm 248.6 from 4 games (a
  round of the run: 0.0068 on a gradient of 38 to 73).

**Measurement 1, the switch match: kill criterion 1 is not met.**

| Round 608's network | Elo | Wins of 800 | Read |
|---|---|---|---|
| before: the loop's match, on the old code (2a523b56) | +161.7 ± 13.6 | 574 | 2026-11-25 22:20 |
| after: with the pointer at zero, on the branch's code (bec71da6) | +183.7 ± 14.0 | 594 | 2026-11-25 22:47 |

D0 = +22.0 ± 19.5 Elo, 1.1 standard errors, where the criterion asks for
under −60. It does not say the new rules favour the network: the two
matches play the same seeds but not the same games, and the difference is
within what two matches of one player give. In the 800 games after, 561
advancements were chosen (0.70 a game, by either player), and 311 of them
(55%) took a type other than the first
(`training/metrics/audit_fixes_20261125/advance_choices_switch_match.json`).

**The switch** (2026-11-25 22:49):
- Both venvs hold the wheel of phase 42, built from the branch's source
  (`tools/kernel_status.py` reads "current" in both).
- `round_0608\pi.pt` is the checkpoint with the pointer (SHA-256
  `fdfcd59a…`); the source is kept as `pi.nopointer.pt` (`9618cc13…`) and
  the optimizer state as `adam.nopointer.pt`. Checked tensor by tensor:
  the source's weights are untouched, the keys (64 by 384) are all zero,
  and the checkpoint holds the same tensors as the network that played
  the switch match. The optimizer state names the new checkpoint, holds
  the source's moments under their new indices and none for the pointer.
- `pool.json` holds the new file's hash for round 608, with the source's
  beside it.
- The run's records carry `code_version` 0.36.2, the version `main` had
  when the branch was cut. The commit each tool ran from is in the
  runner's log.
- The run resumed at 22:49:57 from the worktree
  `C:\Users\momom2\wesnoth_worktrees\advancement`, with the launch's
  command and `--lam 0.99`; its tools log `code bec71da6`. It stood idle
  for 30 minutes.
- To go back (kill criterion 2): stop the run, copy `pi.nopointer.pt` and
  `adam.nopointer.pt` over `pi.pt` and `adam.pt` of round 608, restore
  `pool.json`'s hash, rebuild the wheel of phase 40 into both venvs from
  the worktree `input-facts`, delete the rounds after 608 and their
  markers, and start the launch's command from that worktree.

**Round 609, the first under the four changes, checks out** (2026-11-25
23:09:51). Kill criterion 2 is not met.
- **The exact check:** 7.3e-6 against the limit of 1e-4.
- **The round:** 230 game-sides, 62,540 decisions, no warning. Its games
  took 200 s and its pass 991 s (192 s and 933 s in round 608).
- **The games keep their shape**
  (`training/metrics/audit_fixes_20261125/round_games_605_to_609.json`):
  every game has a winner, the self games last 20.2 turns on average
  (18.5 to 20.9 in rounds 605 to 608), and no side's turn was ended for
  it or left unplayed.
- **The kinds of network played each other.** The round's past self was
  round 0's network, which reads the 109 columns and has no pointer. The
  current network won 22 of the 26 games.
- **Advancements chosen:** 125 in the 128 games (0.98 a game), 67 of them
  (54%) a type other than the first.
- **The value head reads the new games worse.** Its cross-entropy is
  0.511, where rounds 545 to 608 read 0.328 to 0.476 (0.398 on average
  over rounds 593 to 608). Its part of the gradient's norm is 103.9 of
  110.7, against 13.9 of 37.9 in round 608. It is the kind of thing "the
  first two changes are not neutral at the switch" announced: the head
  has never met an enemy of unknown faction in self-play. It is one
  round; whether the reading comes back is read in the next ones.
- **The pointer's first step is the one registered**
  (`pointer_weights_round_0609.json`): every one of the keys' 24,576
  weights moved, by 1e-5, and no query moved. The update's norm is
  0.0074 (0.0068 on average over rounds 593 to 608), of which the keys'
  step is 0.0016.

**Round 610** (23:34:00): the value head's cross-entropy is back at 0.385,
inside the range of the rounds before the switch, and its part of the
gradient's norm is 29.2 of 45.7. So round 609's reading is not a new
level; one Adam step cannot have mended a head, so it was that round's
games. The exact check reads 8.0e-6. The pass took 1,236 s, under the 1.5
times that asks for an inspection: its first group ran at 16.6 decisions
a second where round 609's ran at 45.6, cause not established.

### The matches from the switch

Each is the loop's own match: the round's network sampling at
temperature 1 against the reference at its pinned decode, 800 decisive
games from seed 900000, fp16 serving, on the branch's code. They are read
against each other, never against the matches before round 609.

| Match | Round | Elo | Wins of 800 | Read |
|---|---|---|---|---|
| 1 | 624 | +161.7 ± 13.6 | 574 | 2026-11-26 04:10 |
| 2 | 640 | +212.0 ± 14.6 | 618 | 2026-11-26 09:29 |

**Match 2 (round 640) is the highest of the run's 42 matches, and it is
one match.** Nothing decides before round 736.

- **The two matches together read +186.9 ± 10.0,** which is the switch
  match's reading of round 608's network (+183.7 ± 14.0): +3.2 ± 17.2.
  Read that way, the first 32 rounds on the new rules show no change.
- **Match 2 against match 1 is +50.3 ± 20.0,** 2.5 standard errors with
  the games taken as independent. Seed by seed
  (`training/metrics/audit_fixes_20261125/match_pairs_0624_0640.json`)
  the two share the outcome on 602 games; of the 198 others, round 640's
  network won 121 and round 624's 77. The outcomes of a seed correlate
  0.36 here, which puts the difference's standard error at about 16.
- **It is not taken as a gain of 50 Elo in sixteen rounds.** The loop
  made about 130 Elo in its first 544 rounds, and sixteen steps cannot
  be expected to add 50. The plainer reading is one match on the low side
  and one on the high side of the same strength: the nine matches of
  rounds 480 to 608 ranged from +139.8 to +179.2 around +168. The next
  matches say which.
- **The match's games are ordinary:** 800 decided by a leader's death,
  no turn ended for a side or left unplayed, round 640's own checkpoint
  (`fd046e90…`) on the same code, seeds and reference as match 1.

**Match 1 (round 624): nothing decides yet, and no kill criterion is
met.** Against the switch match (round 608's network on the same code,
+183.7 ± 14.0) it is −22.0 ± 19.5, 1.1 standard errors; one match is not
a reading, and the registered one is after round 736.

- **The count is the one of round 608's match on the old code (574), by
  coincidence, not by a copy**
  (`training/metrics/audit_fixes_20261125/match_pairs_0608_0624.json`).
  Seed by seed, the two matches have the same map, sides and factions on
  all 800 games, a different candidate on all 800 (round 624's network,
  `c1b40984…`, at observation epoch 14, against round 608's, `9618cc13…`,
  at epoch 12), the same outcome on 554 and the same number of turns
  on 54.
- **Two matches on the same seeds are not independent.** A seed fixes the
  map, the sides and the factions, so the two outcomes of a seed
  correlate: 0.24 between rounds 608 and 624, 0.29 between the switch
  match and round 624 (`match_pairs_switch_0624.json`). The standard
  error of a difference between two matches is then about 0.85 of the
  figure for independent games (19.5 becomes about 17), and the same
  factor holds for the registered difference of two means of four. The
  registered figure (about 9.8) stands as written; it is the wider one,
  so the criterion as registered asks for a larger drop (19.6 Elo) than
  the corrected one would (about 17).

**Rounds 609 to 624 check out** (measurement 2;
`round_checks_593_to_624.json`, from `round_checks.py`). They took 5 h 07
min, 19.2 minutes a round.

| | Rounds 593 to 608 | Round 609 | Rounds 610 to 624 | Rounds 625 to 640 |
|---|---|---|---|---|
| Exact check (limit 1e-4) | 7.4e-6 | 7.3e-6 | 7.6e-6 (7.2e-6 to 8.0e-6) | 7.8e-6 (7.4e-6 to 8.3e-6) |
| Gradient's norm | 50.0 | 110.7 | 50.9 (39.0 to 83.3) | 46.7 (39.9 to 55.2) |
| of which the value head | 35.2 | 103.9 | 34.7 (10.2 to 73.7) | 28.9 (16.8 to 40.1) |
| Update's norm | 0.0068 | 0.0074 | 0.0077 | 0.0078 |
| Value head's cross-entropy | 0.398 ± 0.008 | 0.511 | 0.406 ± 0.008 (0.351 to 0.468) | 0.403 ± 0.004 (0.371 to 0.432) |
| Decisions in the pass | 58,180 | 62,540 | 59,780 | 61,580 |
| Pass, seconds | 912 | 981 | 943 | 935 |

The last column was added with match 2 (`round_checks_609_to_640.json`);
what it shows is under "Rounds 625 to 640" below.

- **The value head reads the new games as it read the old ones.** Its
  cross-entropy over rounds 610 to 624 is 0.406 ± 0.008 against 0.398 ±
  0.008 over the sixteen rounds before the switch; round 609's 0.511
  stands alone. Its part of the gradient is unchanged on average.
- **The update is 14% larger than before the switch,** on a gradient of
  the same norm. By group, on average over rounds 610 to 624 against
  rounds 593 to 608:
  - the weapon head, 0.0016 against 0.0007: the pointer's new weights,
    which Adam moves by about the learning rate each while they are new;
  - the value head, 0.0041 against 0.0032, with its gradient's norm
    unchanged. Its step was already growing before the switch (0.0026
    over rounds 545 to 560, 0.0028 over 577 to 592). With Adam, a larger
    step on an unchanged gradient means gradients that agree more from
    round to round, or recent ones larger than the long average on the
    weights concerned (rounds 609 and 620 read 103.9 and 73.7). Which of
    the two is not established;
  - the encoder, 0.0013 against 0.0010, on a gradient of 4.0 against
    3.2 (it read 0.0014 over rounds 545 to 560, after the new inputs);
  - the trunk, 0.0060 against 0.0057.
- **The pass's time follows the decisions:** 3% longer for 3% more
  decisions (the table's seconds are the sum of the pass's groups, some
  10 s under the console's figure). Two passes ran over 1,200 s on the
  console (rounds 610 and 616), none over 1.5 times the usual.
- **No warning, no refusal:** the console holds none since the relaunch.

**The pointer's weights at round 624** (measurement 3;
`pointer_weights_round_0624.json`): the keys' root mean square is 4.4e-5
after sixteen steps, every one of the 24,576 away from zero; the queries
have moved 4.2e-5 (weights) and 5.5e-5 (bias) from the switch's. That is
the loop's ordinary pace: the new inputs' projection stood at 4.6e-5
sixteen rounds after its own switch, and the old unit projection at
4.3e-5 after the loop's first sixteen rounds.

**Advancements chosen:** 95 in round 624's 128 games (0.74 a game), 48 of
them (51%) a type other than the first
(`advance_choices_round_0624.json`); 544 in the match's 800 games (0.68 a
game, by either player), 302 of them (56%) another type than the first
(`advance_choices_match_0624.json`; the switch match read 561 and 55%).

**Rounds 625 to 640 check out** (2026-11-26 04:10 to 09:16, 19.2 minutes
a round; the table above).

- **The exact check holds** (7.4e-6 to 8.3e-6), every gradient is finite,
  and the console holds no warning.
- **The value head:** its cross-entropy is 0.403 ± 0.004, as before the
  switch (0.398 ± 0.008), and steadier from round to round (0.371 to
  0.432). Its part of the gradient is lower, 28.9 against 35, with no
  round over 40.1 where six of the sixteen rounds 593 to 608 and six of
  rounds 609 to 624 were. Its step stays at 0.0041, the level it took
  after the switch.
- **The update stays 16% over the level before the switch** (0.0078
  against 0.0068). The weapon head's step is coming down as the pointer's
  weights age (0.0012, from 0.0016 over rounds 610 to 624 and 0.0007
  before the switch); the trunk's is at 0.0063 (0.0057 before).
- **The pointer's weights at round 640**
  (`pointer_weights_round_0640.json`): the keys' root mean square is
  5.7e-5 (4.4e-5 at round 624), and the queries have moved 5.7e-5
  (weights) and 6.6e-5 (bias) from the switch's.
- **Advancements chosen:** 95 in round 640's 128 games (0.74 a game), 59
  of them (62%) a type other than the first
  (`advance_choices_round_0640.json`); 548 in the match's 800 games (0.69
  a game), 327 of them (60%) another type than the first
  (`advance_choices_match_0640.json`).
