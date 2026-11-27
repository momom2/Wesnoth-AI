# Three fixes from the blind-spot audit, and a fourth: pre-registration (2026-11-25)

**Status: switched. Round 609 is the first under the four changes**
(2026-11-25 22:49, from the branch `feature/advancement-choice` at
bec71da6; departure 1 below). **The GPU checks passed, and the switch
match reads +22.0 ± 19.5 against the loop's own match of the same network:
kill criterion 1 is not met** (Results). **The first four matches from
the switch read +161.7 ± 13.6 (round 624), +212.0 ± 14.6 (round 640),
+207.1 ± 14.5 (round 656) and +199.9 ± 14.4 (round 672): +195.2 ± 7.1
together, the first half of the registered comparison. The fifth, sixth
and seventh read +220.8 ± 14.8 (round 688), +218.3 ± 14.8 (round 704)
and +209.6 ± 14.6 (round 720). Rounds 609 to 720 check out.** One match
is left before the registered verdict (round 736).
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
| 3 | 656 | +207.1 ± 14.5 | 614 | 2026-11-26 14:59 |
| 4 | 672 | +199.9 ± 14.4 | 608 | 2026-11-26 20:26 |
| 5 | 688 | +220.8 ± 14.8 | 625 | 2026-11-27 02:00 |
| 6 | 704 | +218.3 ± 14.8 | 623 | 2026-11-27 07:41 |
| 7 | 720 | +209.6 ± 14.6 | 616 | 2026-11-27 13:11 |

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

**Match 3 (round 656): +207.1 ± 14.5 (614 of 800, 2026-11-26 14:59). It
stands with match 2, not between the first two.** Nothing decides before
round 736.

- **The three together read +193.6 ± 8.2,** which is +9.9 ± 16.2 over
  the switch match's reading of round 608's network (+183.7 ± 14.0). By
  that comparison no change is read yet.
- **Matches 2 and 3 agree** (−4.9 ± 20.6 between them) and average
  +209.6 ± 10.3, +25.9 ± 17.4 over the switch match: 1.5 standard
  errors. Before round 640 no match of the run had read over +184 (the
  loop's own highest was +179.2, the switch match +183.7); two in a row
  now read over +200.
- **What was written at match 2 is neither confirmed nor refuted.** "One
  match low and one high around the same strength" put the next one near
  +187; it read 20 higher, 1.1 standard errors of that difference. The
  other reading, that round 624's match was the low one and the level
  has risen, fits as well. Match 3 against match 1 is +45.4 ± 19.9;
  seed by seed they share 576 outcomes, and of the 224 others round
  656's network won 132 and round 624's 92
  (`match_pairs_0624_0656.json`; `match_pairs_0640_0656.json` holds the
  pair of rounds 640 and 656: 614 outcomes shared, correlation 0.34).
- **The match's games are ordinary:** 800 decided by a leader's death,
  no turn ended for a side or left unplayed, round 656's own checkpoint
  (`e552c020…`) on the same code, seeds and reference.

**Match 4 (round 672): +199.9 ± 14.4 (608 of 800, 2026-11-26 20:26). The
first half of the registered comparison is complete: matches 1 to 4
average +195.2 ± 7.1.** Nothing decides before round 736.

- **What the verdict will ask:** the mean of matches 5 to 8 less +195.2.
  Under −19.6 (2 standard errors as registered) is "the loop loses
  strength under the new rules".
- **Three matches in a row read about +200 or over** (+212.0, +207.1,
  +199.9): +206.4 ± 8.4 together. That is +22.7 ± 16.3 over the switch
  match's reading of round 608's network (+183.7 ± 14.0), 1.4 standard
  errors: it leans to a gain since the switch and does not establish
  one. All four matches read +11.5 ± 15.7 over the switch match.
- **The switch match is what limits that comparison.** It is one match
  of 800 games, and its standard error (14.0) is larger than that of the
  three matches together (8.4). More matches of the run will not narrow
  it; only more games of round 608's network on the new rules would.
  None is registered.
- **Match 4 against match 3 is −7.2 ± 20.4.** Seed by seed they share
  586 outcomes, correlation 0.26 (`match_pairs_0656_0672.json`).
- **The match's games are ordinary:** 800 decided by a leader's death,
  no turn ended for a side or left unplayed, round 672's own checkpoint
  (`f39934f4…`) on the same code, seeds and reference.

**Match 5 (round 688): +220.8 ± 14.8 (625 of 800, 2026-11-27 02:00), the
highest of the run's 45 matches. It is the first of the four the verdict
sets against +195.2 ± 7.1.** Nothing decides before round 736.

- **Toward the verdict:** match 5 alone is +25.6 ± 16.5 over the mean of
  matches 1 to 4. For "the loop loses strength" (under −19.6), matches 6
  to 8 would have to average under +160.5; none of the five has read
  under +161.7.
- **Since the switch** (`match_line_to_0688.json`, from `match_line.py`,
  the matches taken as independent):
  - the five matches average +200.3 ± 6.4, +16.6 ± 15.4 over the switch
    match's reading of round 608's network (+183.7 ± 14.0);
  - the four after the first average +210.0 ± 7.3, +26.3 ± 15.8 over the
    switch match, 1.7 standard errors;
  - a line through the switch match and the five rises 0.54 ± 0.21 Elo a
    round (2.5 standard errors); through the five alone, 0.69 ± 0.28.
    For scale, the loop made about 0.24 Elo a round over its first 544.
- **What that says:** the evidence of a gain since the switch is
  stronger than at match 4 and still short of 3 standard errors on any
  of these readings. The line's slope leans on match 1, the lowest; it
  was not registered, and neither was any of these comparisons.
- **Match 5 against match 4 is +20.8 ± 20.7.** Seed by seed they share
  613 outcomes, correlation 0.34 (`match_pairs_0672_0688.json`).
- **The match's games are ordinary:** 800 decided by a leader's death,
  no turn ended for a side or left unplayed, round 688's own checkpoint
  (`07c92639…`) on the same code, seeds and reference. They are a little
  shorter than the earlier matches' (17.7 turns on average, against 18.1
  to 18.8 in the four before).

**Match 6 (round 704): +218.3 ± 14.8 (623 of 800, 2026-11-27 07:41), the
second highest of the run's 46 matches, after match 5.** Nothing decides
before round 736.

- **Toward the verdict:** matches 5 and 6 average +219.5. For "the loop
  loses strength" (matches 5 to 8 more than 19.6 under +195.2), matches
  7 and 8 would have to average under +131.7; the lowest match since the
  switch read +161.7.
- **Since the switch** (`match_line_to_0704.json`, the matches taken as
  independent):
  - the six matches average +203.3 ± 5.9, +19.6 ± 15.2 over the switch
    match's reading of round 608's network (+183.7 ± 14.0);
  - the five after the first average +211.6 ± 6.5, +27.9 ± 15.5 over the
    switch match, 1.8 standard errors;
  - a line through the switch match and the six rises 0.48 ± 0.17 Elo a
    round (2.8 standard errors); through the six alone, 0.56 ± 0.21.
- **What that says:** each reading is a little firmer than at match 5
  and none reaches 3 standard errors. The comparison with the switch
  match stays limited by that one match's standard error (14.0); the
  line is the reading that the run's own matches can sharpen, and it
  leans on match 1.
- **Match 6 against match 5 is −2.5 ± 20.9.** Seed by seed they share
  612 outcomes, correlation 0.32 (`match_pairs_0688_0704.json`).
- **The match's games are ordinary:** 800 decided by a leader's death,
  no turn ended for a side or left unplayed, round 704's own checkpoint
  (`328e4030…`) on the same code, seeds and reference; 18.0 turns on
  average.

**Match 7 (round 720): +209.6 ± 14.6 (616 of 800, 2026-11-27 13:11).**
One match is left before the verdict.

- **Toward the verdict:** matches 5 to 7 average +216.2. "The loop loses
  strength" would now need match 8 under +53.8; with match 8 at the
  level of the last six, the registered difference would be about +20.
- **Since the switch** (`match_line_to_0720.json`, the matches taken as
  independent):
  - the seven matches average +204.2 ± 5.5, +20.5 ± 15.0 over the switch
    match's reading of round 608's network (+183.7 ± 14.0);
  - the six after the first average +211.3 ± 6.0, +27.6 ± 15.2 over the
    switch match, 1.8 standard errors;
  - a line through the switch match and the seven rises 0.37 ± 0.14 Elo
    a round (2.7 standard errors); through the seven alone, 0.40 ± 0.17.
- **What that says: a level since round 640, not a rise.** The six
  matches from round 640 sit between +199.9 and +220.8, and a line
  through them alone is flat: +0.07 ± 0.22 Elo a round
  (`match_line_0640_to_0720.json`). The slope of the longer lines is the
  step between the two early readings (the switch match and match 1) and
  the rest. Those two average +172.7 ± 9.8 and the six since +211.3 ±
  6.0, a difference of +38.6 ± 11.5; that split was chosen after seeing
  the matches, so its 3.4 standard errors overstate the evidence. What
  can be said: if there was a gain, it came in the first 32 rounds on
  the new rules, and the 80 rounds since show none.
- **Match 7 against match 6 is −8.7 ± 20.8.** Seed by seed they share
  605 outcomes, correlation 0.30 (`match_pairs_0704_0720.json`).
- **The match's games are ordinary:** 800 decided by a leader's death,
  no turn ended for a side or left unplayed, round 720's own checkpoint
  (`ade44719…`) on the same code, seeds and reference; 18.7 turns on
  average.

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

**Rounds 641 to 656 check out** (2026-11-26 09:29 to 14:46, 19.8 minutes
a round; `round_checks_609_to_656.json`).

| | Rounds 593 to 608 | Rounds 625 to 640 | Rounds 641 to 656 |
|---|---|---|---|
| Exact check (limit 1e-4) | 7.4e-6 | 7.8e-6 | 7.9e-6 (7.5e-6 to 9.0e-6) |
| Gradient's norm | 50.0 | 46.7 | 56.2 (39.8 to 86.7) |
| of which the value head | 35.2 | 28.9 | 42.1 (15.4 to 78.3) |
| Update's norm | 0.0068 | 0.0078 | 0.0076 |
| of which the value head | 0.0032 | 0.0041 | 0.0037 |
| Value head's cross-entropy | 0.398 ± 0.008 | 0.403 ± 0.004 | 0.397 ± 0.007 (0.332 to 0.442) |
| Decisions in the pass | 58,180 | 61,580 | 60,960 |
| Pass, seconds | 912 | 935 | 977 |

- **The exact check holds,** every gradient is finite, and the console
  holds no warning. Its largest reading, 9.0e-6, is the highest since the
  switch and eleven times under the limit.
- **The value head reads its games as before the switch** (0.397 ±
  0.007). Its part of the gradient is back up (42.1, with six rounds
  over 40.1 as in the rounds before round 625), so the lower level of
  rounds 625 to 640 was sixteen quiet rounds and not a change. Its step
  is at 0.0037, between the level before the switch (0.0032) and the one
  after it (0.0041).
- **The pass:** one ran 1,301 s (round 646, on 69,376 decisions, the
  most of any round since the switch), under 1.5 times the usual.
- **The pointer's weights at round 656**
  (`pointer_weights_round_0656.json`): the keys' root mean square is
  6.4e-5 (5.7e-5 at round 640), and the queries have moved 7.2e-5
  (weights) and 8.2e-5 (bias) from the switch's.
- **Advancements chosen:** 118 in round 656's 128 games (0.92 a game),
  58 of them (49%) a type other than the first
  (`advance_choices_round_0656.json`); 564 in the match's 800 games (0.71
  a game), 328 of them (58%) another type than the first
  (`advance_choices_match_0656.json`).

**Rounds 657 to 672 check out** (2026-11-26 14:59 to 20:13, 19.6 minutes
a round; `round_checks_609_to_672.json`).

| | Rounds 593 to 608 | Rounds 641 to 656 | Rounds 657 to 672 |
|---|---|---|---|
| Exact check (limit 1e-4) | 7.4e-6 | 7.9e-6 | 7.9e-6 (7.5e-6 to 8.4e-6) |
| Gradient's norm | 50.0 | 56.2 | 51.0 (39.4 to 63.4) |
| of which the value head | 35.2 | 42.1 | 34.4 (15.6 to 54.0) |
| of which the weapon head | 0.46 | 0.62 | 0.64 |
| Update's norm | 0.0068 | 0.0076 | 0.0077 |
| of which the value head | 0.0032 | 0.0037 | 0.0035 |
| of which the weapon head | 0.0007 | 0.0013 | 0.0018 |
| Value head's cross-entropy | 0.398 ± 0.008 | 0.397 ± 0.007 | 0.399 ± 0.007 (0.347 to 0.450) |
| Decisions in the pass | 58,180 | 60,960 | 60,600 |
| Pass, seconds | 912 | 977 | 961 |

- **The exact check holds,** every gradient is finite, and the console
  holds no warning.
- **The value head reads its games as before the switch** (0.399 ±
  0.007), and its step is coming back down (0.0035, from 0.0041 after
  the switch and 0.0032 before it).
- **The weapon head's group is the one that moves more.** Its gradient
  has grown since the switch (0.46 before and through round 624, then
  0.52, 0.62, 0.64), and its step with it (0.0018 over these rounds,
  0.0017 over rounds 657 to 664 and 0.0018 over 665 to 672, from 0.0013
  over the sixteen rounds before). The group holds the pointer.
- **The pointer's weights at round 672**
  (`pointer_weights_round_0672.json`): the keys' root mean square is
  8.2e-5 (6.4e-5 at round 656, 5.7e-5 at round 640), and the queries
  have moved 8.9e-5 (weights) and 1.0e-4 (bias) from the switch's. The
  keys grew more over these sixteen rounds (1.7e-5) than over the
  sixteen before (0.8e-5). Whether the weapon's choice now follows its
  target is not read from weights: the audit's counts at round 736 will
  say (measurement 5).
- **Advancements chosen:** 91 in round 672's 128 games (0.71 a game), 57
  of them (63%) a type other than the first
  (`advance_choices_round_0672.json`); 526 in the match's 800 games (0.66
  a game), 305 of them (58%) another type than the first
  (`advance_choices_match_0672.json`).

**Rounds 673 to 688 check out** (2026-11-26 20:26 to 2026-11-27 01:48,
20.1 minutes a round; `round_checks_609_to_688.json`).

| | Rounds 593 to 608 | Rounds 657 to 672 | Rounds 673 to 688 |
|---|---|---|---|
| Exact check (limit 1e-4) | 7.4e-6 | 7.9e-6 | 8.1e-6 (7.5e-6 to 8.5e-6) |
| Gradient's norm | 50.0 | 51.0 | 50.3 (38.9 to 87.2) |
| of which the value head | 35.2 | 34.4 | 33.0 (15.7 to 81.1) |
| of which the weapon head | 0.46 | 0.64 | 0.54 |
| Update's norm | 0.0068 | 0.0077 | 0.0078 (0.0075 to 0.0082) |
| of which the value head | 0.0032 | 0.0035 | 0.0037 |
| of which the weapon head | 0.0007 | 0.0018 | 0.0014 |
| Value head's cross-entropy | 0.398 ± 0.008 | 0.399 ± 0.007 | 0.391 ± 0.007 (0.322 to 0.421) |
| Decisions in the pass | 58,180 | 60,600 | 61,230 |
| Pass, seconds | 912 | 961 | 992 |

- **The exact check holds,** every gradient is finite, and the console
  holds no warning.
- **The value head reads its games as before the switch** (0.391 ±
  0.007). One round stands out, round 687: its part of the gradient is
  81.1 and the round's cross-entropy 0.322, the lowest since the switch.
  The update's norm of rounds 686 to 688 is the highest since the switch
  (0.0082), of which the value head's part is 0.0043.
- **The weapon head's group came back down** (gradient 0.54, step
  0.0014): what rounds 657 to 672 showed was not the start of a rise.
- **The pass is 3% slower for 1% more decisions.** Four passes of the
  evening ran over 1,050 s on the console (rounds 679 to 681 and 685;
  the longest 1,257 s), none over 1.5 times the usual. Cause not
  established.
- **The pointer's weights at round 688**
  (`pointer_weights_round_0688.json`): the keys' root mean square is
  9.9e-5 (8.2e-5 at round 672), and the queries have moved 1.0e-4
  (weights) and 1.1e-4 (bias) from the switch's.
- **Advancements chosen, fewer than before:** 74 in round 688's 128
  games (0.58 a game), 42 of them (57%) a type other than the first
  (`advance_choices_round_0688.json`); 464 in the match's 800 games (0.58
  a game, where the four matches before read 0.66 to 0.71), 280 of them
  (60%) another type than the first (`advance_choices_match_0688.json`).

**Rounds 689 to 704 check out, and the update keeps growing**
(2026-11-27 02:00 to 07:28, 20.4 minutes a round;
`round_checks_609_to_704.json`).

| | Rounds 593 to 608 | Rounds 673 to 688 | Rounds 689 to 704 |
|---|---|---|---|
| Exact check (limit 1e-4) | 7.4e-6 | 8.1e-6 | 8.5e-6 (8.0e-6 to 9.3e-6) |
| Gradient's norm | 50.0 | 50.3 | 50.9 (42.0 to 67.4) |
| of which the value head | 35.2 | 33.0 | 31.4 (14.8 to 55.1) |
| Update's norm | 0.0068 | 0.0078 | 0.0089 (0.0081 to 0.0094) |
| of which the value head | 0.0032 | 0.0037 | 0.0049 (0.0043 to 0.0053) |
| of which the trunk | 0.0057 | 0.0065 | 0.0070 |
| of which the memory's write | 0.0010 | 0.0010 | 0.0013 |
| Value head's cross-entropy | 0.398 ± 0.008 | 0.391 ± 0.007 | 0.400 ± 0.005 (0.354 to 0.436) |
| Decisions in the pass | 58,180 | 61,230 | 63,420 |
| Pass, seconds | 912 | 992 | 1,004 |

- **The exact check holds,** every gradient is finite, and the console
  holds no warning. The check's reading has crept up since the switch
  (7.6e-6, 7.8e-6, 7.9e-6, 7.9e-6, 8.1e-6, 8.5e-6 over the six spans of
  sixteen rounds); it is eleven times under the limit.
- **The update's norm has grown by 14% in sixteen rounds, on a gradient
  that has not.** It read 0.0076 at round 681 and 0.0092 at round 704;
  rounds 697 to 704 are the highest of the run since at least round 545. Most of it is the
  value head, whose step went from 0.0034 at round 681 to 0.0052 at
  rounds 701 to 703 while its gradient's norm fell; the trunk and the
  memory's write grew less.
  - With Adam, a step that grows on an unchanged gradient means the
    gradients of successive rounds agree more: the value head is being
    pushed the same way round after round. Toward what is not
    established.
  - The head's cross-entropy on its games has not moved (0.400 ± 0.005).
  - It is no kill criterion. It is reported to the user because the
    value estimate is their first priority.
- **The rounds are longer:** 63,420 decisions a round, 9% over the level
  before the switch, and the pass with them.
- **The pointer's weights at round 704**
  (`pointer_weights_round_0704.json`): the keys' root mean square is
  1.07e-4 (9.9e-5 at round 688), and the queries have moved 1.09e-4
  (weights) and 1.25e-4 (bias) from the switch's.
- **Advancements chosen:** 102 in round 704's 128 games (0.80 a game),
  46 of them (45%) a type other than the first
  (`advance_choices_round_0704.json`); 539 in the match's 800 games (0.67
  a game), 302 of them (56%) another type than the first
  (`advance_choices_match_0704.json`). Match 5's lower count (0.58 a
  game) did not repeat.

**Rounds 705 to 720 check out; the update has stopped growing**
(2026-11-27 07:41 to 12:58, 19.8 minutes a round;
`round_checks_609_to_720.json`).

| | Rounds 593 to 608 | Rounds 689 to 704 | Rounds 705 to 720 |
|---|---|---|---|
| Exact check (limit 1e-4) | 7.4e-6 | 8.5e-6 | 8.8e-6 (8.2e-6 to 9.4e-6) |
| Gradient's norm | 50.0 | 50.9 | 54.4 (45.8 to 62.9) |
| of which the value head | 35.2 | 31.4 | 36.6 (20.4 to 47.6) |
| Update's norm | 0.0068 | 0.0089 | 0.0092 (0.0089 to 0.0094) |
| of which the value head | 0.0032 | 0.0049 | 0.0049 (0.0043 to 0.0052) |
| of which the trunk | 0.0057 | 0.0070 | 0.0074 |
| of which the weapon head | 0.0007 | 0.0013 | 0.0018 |
| Value head's cross-entropy | 0.398 ± 0.008 | 0.400 ± 0.005 | 0.394 ± 0.005 (0.352 to 0.431) |
| Decisions in the pass | 58,180 | 63,420 | 61,470 |
| Pass, seconds | 912 | 1,004 | 972 |

- **The exact check holds,** every gradient is finite, and the console
  holds no warning. The check's reading goes on creeping up (8.8e-6
  here, 7.6e-6 over the first sixteen rounds after the switch); its
  largest reading is ten times under the limit.
- **The update's norm has levelled off** at 0.0092, 36% over the level
  before the switch: 0.0094 over rounds 705 to 708, 0.0090 over rounds
  717 to 720. The value head's step stays near 0.0050 through round 716
  and reads 0.0045 over rounds 717 to 720. Its cross-entropy has not
  moved.
- **The pointer's weights at round 720**
  (`pointer_weights_round_0720.json`): the keys' root mean square is
  1.04e-4 (1.07e-4 at round 704: no longer growing), and the queries
  have moved 1.32e-4 (weights) and 1.58e-4 (bias) from the switch's. The
  weapon head's step is back at 0.0018.
- **Advancements chosen:** 96 in round 720's 128 games (0.75 a game), 42
  of them (44%) a type other than the first
  (`advance_choices_round_0720.json`); 588 in the match's 800 games (0.74
  a game), 358 of them (61%) another type than the first
  (`advance_choices_match_0720.json`).

### The value estimate since the switch (read 2026-11-27, not registered)

Read because the value head's step grew (above). It is the reading of
`docs/value_reliability_prereg_20261120.md`, "What was found":
`training/metrics/value_reliability_20261120/where_optimism.py` on the
decision logs of 32 rounds, no game played and no network run. v is the
estimate at a decision and z the game's result for the side that
decides, both from −1 to +1; v − z over many decisions is the estimate's
optimism. The windows before the switch are that document's.

| v − z | Rounds 433 to 464 | 497 to 528 | 609 to 640 | 641 to 672 | 673 to 704 |
|---|---|---|---|---|---|
| All decisions (± 0.003) | +0.030 | +0.032 | +0.017 | +0.032 | +0.013 |
| Self games (± 0.001) | +0.037 | +0.036 | +0.025 | +0.035 | +0.021 |
| Against a past self (± 0.022) | −0.025 | +0.002 | −0.046 | +0.012 | −0.051 |
| First decision of own turn 2 (± 0.004) | −0.013 | −0.022 | −0.063 | −0.055 | −0.086 |
| First decision of own turn 3 | −0.095 | −0.111 | −0.155 | −0.142 | −0.171 |
| First decision of own turn 4 | −0.043 | −0.061 | −0.110 | −0.086 | −0.128 |
| First decision of own turn 5 | −0.025 | −0.037 | −0.090 | −0.061 | −0.095 |
| First decision of own turn 10 | +0.048 | +0.044 | +0.020 | +0.034 | +0.013 |
| The end-turn decision (± 0.003) | +0.026 | +0.033 | +0.030 | +0.046 | +0.034 |

- **The estimate's overall optimism has not grown since the switch.** It
  reads +0.013 to +0.032, against +0.030 and +0.032 in the two windows
  before.
- **At the start of its early turns the estimate is pessimistic, and
  more so since the switch** (the second half is corrected below: it was
  falling long before the switch). At the first decision of own turn 3 it
  reads 0.17 under the results in the last window, against 0.10 and 0.11
  before the switch; the same holds at turns 2, 4 and 5, each about 0.05
  lower than before. It did not come back over the 96 rounds: the last
  window is the lowest of the three.
- **A candidate cause, not established (refuted the same day, below):**
  since the switch a side does
  not know the enemy's faction until it sees an enemy unit, which early
  turns rarely do, and the value document found that the estimate
  follows the enemy's faction. Telling this cause from another needs the
  early decisions split by whether an enemy has been seen, which this
  tool does not do.
- **Whether this is what the head's growing step is working on is not
  established either.** The early pessimism is larger in the last window
  than in the one before, so if the head is being pushed against it, it
  has not shown yet.
- The outputs are `where_609_640.json`, `where_641_672.json` and
  `where_673_704.json` in `training/metrics/value_reliability_20261120/`.

#### Corrected the same day: the pessimism is the run's, not the switch's, and not the faction's

Two things written above are wrong, and both were told to the user that
morning: that the early pessimism grew at the switch, and the candidate
cause. Two readings show it, neither registered.

**1. It grew through the whole run.** The table above set the rounds
after the switch against windows that ended 80 and 144 rounds before it.
The two windows just before the switch were then read
(`where_545_576.json`, `where_577_608.json`), and with the value
document's earlier windows they give, at the first decision of a side's
own turn (v − z, each ± 0.004):

| Rounds | Turn 2 | Turn 3 | Turn 4 | Turn 5 |
|---|---|---|---|---|
| 1 to 32 | +0.043 | +0.053 | +0.049 | +0.016 |
| 177 to 208 | +0.057 | +0.025 | +0.062 | +0.074 |
| 241 to 272 | +0.032 | −0.037 | −0.004 | −0.012 |
| 283 to 314 | +0.010 | −0.075 | −0.036 | −0.035 |
| 393 to 424 | −0.002 | −0.096 | −0.063 | −0.049 |
| 465 to 496 | −0.029 | −0.122 | −0.074 | −0.053 |
| 497 to 528 | −0.022 | −0.111 | −0.061 | −0.037 |
| 545 to 576 | −0.027 | −0.117 | −0.068 | −0.051 |
| 577 to 608 | −0.043 | −0.141 | −0.094 | −0.072 |
| 609 to 640, after the switch | −0.063 | −0.155 | −0.110 | −0.090 |
| 641 to 672 | −0.055 | −0.142 | −0.086 | −0.061 |
| 673 to 704 | −0.086 | −0.171 | −0.128 | −0.095 |

At the third turn the estimate was optimistic until about round 200 and
pessimistic from round 250; the turns around it followed, and all have
gone on falling since. The last window before the switch already read −0.141 at turn 3.
The three windows after it read −0.155, −0.142 and −0.171: a
continuation, with no step that the switch can be given.

**2. It does not follow what the side knows of the enemy's faction**
(`early_faction.py`, written for this; a third of the games of rounds
673 to 704 and of rounds 577 to 608 replayed through the simulator, 1,365
games each, no network run; `early_faction_673_704.json`,
`early_faction_577_608.json`). At the first decision of each own turn,
the positions are split by whether the enemy unit types the side has
seen name the enemy's faction. That is what the faction input knows
since the switch; before it the side was told the faction outright, and
the same split is the control.

| v − z at the turn's first decision | Share named | All | Types name the faction | They do not |
|---|---|---|---|---|
| **After the switch, rounds 673 to 704** | | | | |
| Own turn 2 | 8% | −0.085 ± 0.007 | −0.313 ± 0.047 | −0.066 ± 0.008 |
| Own turn 3 | 55% | −0.170 ± 0.007 | −0.205 ± 0.018 | −0.128 ± 0.022 |
| Own turn 4 | 95% | −0.126 ± 0.007 | −0.120 ± 0.008 | −0.254 ± 0.083 |
| Own turn 5 | 100% | −0.095 ± 0.007 | −0.096 ± 0.008 | (1 position) |
| **Before the switch, rounds 577 to 608** | | | | |
| Own turn 2 | 8% | −0.052 ± 0.007 | −0.166 ± 0.045 | −0.043 ± 0.008 |
| Own turn 3 | 54% | −0.149 ± 0.007 | −0.171 ± 0.018 | −0.123 ± 0.020 |
| Own turn 4 | 95% | −0.101 ± 0.007 | −0.101 ± 0.009 | −0.107 ± 0.085 |
| Own turn 5 | 100% | −0.084 ± 0.007 | −0.084 ± 0.007 | (none) |

- **The candidate cause is refuted.** By its fourth turn a side can name
  the enemy's faction in 95% of positions, and by its fifth in all of
  them; the pessimism is there all the same (−0.126 and −0.095). I had
  written that early turns rarely see an enemy unit. That was a guess,
  and it is wrong from the third turn on.
- **Where the two groups can be told apart, the estimate is lower where
  the faction is named, not where it is unknown** (turn 2: −0.313
  against −0.066; turn 3: −0.205 against −0.128). A side that names the
  faction that early has seen an enemy unit early. So the estimate reads
  an early sighting as worse than it turns out. That holds before the
  switch too (−0.166 against −0.043 at turn 2), on 185 positions.
- **After against before, the same turns read 0.01 to 0.03 lower**
  (−0.033 ± 0.010 at turn 2, −0.021 at turn 3, −0.025 at turn 4, −0.011
  at turn 5): the size of one window's step in the run-long fall above.

**What stands after the correction.**
- The estimate is too low at the start of a side's second to eighth
  turns, by up to 0.17 at the third, and it has been falling there for
  some 450 rounds. It is the opening of the game: by the ninth turn the
  gap is back near zero.
- Its cause is not established. It is not the switch, and not the
  enemy's faction being unknown.
- Why it may matter for training (my reading, not measured): the weight
  of a decision is built from the result and the estimates that follow
  the decision (λ = 0.99). An estimate that dips over the opening and
  comes back by the ninth turn adds the same push to every decision made
  at a given point of the opening, whatever its merit: against the
  decisions made before the bottom of the dip, for those made after it.
  What that costs in play is not measured.
- Whether the value head's larger step of rounds 689 to 704 has to do
  with it is not established.

**`main` holds the run's code since 2026-11-26** (0.37.0 and 0.38.0, on
the user's word), and CI is green on it at 0.38.1 (14657e7e, both tiers).
The run still executes from the branch's worktree.
