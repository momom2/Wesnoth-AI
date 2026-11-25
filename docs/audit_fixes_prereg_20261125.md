# Three fixes from the blind-spot audit: pre-registration (2026-11-25)

**Status: built on the branch `feature/audit-fixes` (a897c7f3, c782e361),
where its own 17 tests pass; the full suite, CI and the GPU checks of
"The switch" are still to go. Not in the run. When the run switches to it
is the user's to say.** It would be departure 11 of the simple loop
(`docs/simple_loop_prereg_20261117.md`).

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

None yet.
