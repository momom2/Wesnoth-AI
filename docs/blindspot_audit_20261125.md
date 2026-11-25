# Blind spots: what a player sees or can order and the network cannot (audit of 2026-11-25)

**Status: the audit is done and nothing is changed. What to close is the
user's to decide.**

The user, 2026-11-25: "Please send Opus 5.5 agents to do a comprehensive
audit to make sure it doesn't have anymore blindspots."

The standard is the user's ruling of 2026-09-29: "For each item of the
present state that a player sees, the network sees it too." Its reverse is
CLAUDE.md's Principle 6: the network sees nothing a player cannot.

## Conclusion

- **No input holds a wrong number.** Every column was read against the
  code that fills it and against Wesnoth 1.18.4; the fights' numbers
  follow the engine's own computation step by step.
- **Nothing is truncated or masked away** between the input and the
  decision: no token limit is reached, every unit type, trait, ability and
  weapon special of the default era has its place, and no unit has more
  weapons than the row has slots.
- **Three large gaps remain** between a player and the network:
  1. about two thirds of the board has no hex token, so its terrain is
     not shown;
  2. the network cannot choose what a unit becomes when it levels up: it
     always gets the first type listed;
  3. the weapon is scored without the enemy it strikes (known since
     2026-11-24, the input design's Part 3).
- **Four of medium weight,** and a list of small ones, follow below.
- **One suspicion was cleared by a run:** the half-precision serving
  carries the new unit columns' effect in full (1.02 ± 0.02 of it).
- **One convention needs the user's ruling:** the network is told the
  enemy's faction from its first decision.

## How it was done

Seven agents (Opus 5.5), one part each, read-only since the training run
held the machine: units; the map; the game's state beyond the board; time
and memory; the outputs; the truth of the existing columns; the path from
the tensor to the decision. Their brief and their reports are in
`training/metrics/blindspot_audit_20261125/reports/`. Each report also
lists what it checked and found right.

An agent's finding is a claim. Each one below was then checked in this
session: by reading the code it names, and where a count settles it, by a
run on the simple loop's own games (40 games of round 576, two a map,
unless said otherwise; scripts and outputs in
`training/metrics/blindspot_audit_20261125/`). Claims that did not hold
are listed at the end.

## The large gaps

### 1. About two thirds of the board has no hex token

- **A player sees** the whole map: every hex's terrain, also under fog.
- **The network gets** a token only for the hexes of the "relevant set"
  (`rust/wesnoth_core/src/core_observe.rs:138-159`,
  `core_parity.rs:428-456`): villages, castles and keeps, hexes holding a
  unit it sees, the hexes its own units can reach this turn, the six
  hexes around each of its units, and the hexes it does not see within 6
  of one of its units. Every other hex is absent: its terrain, whether
  the side sees it, its time of day.
- **Measured** (`hex_tokens_round_0576.json`, 4,397 decisions):

  | Per decision | Hexes | Share of the board |
  |---|---|---|
  | The board | 1,030 | |
  | Hex tokens | 364 | 35% |
  | Hexes the side sees now | 602 | 58% |
  | of which without a token | 297 | 49% of the seen |

  By map the share with a token runs from 20% (Aethermaw, whose board
  counts its off-map hexes) to 50% (Thousand Stings Garrison). It is 27%
  in a game's first five turns.
- **What is lost:** the ground beyond one move (a river, a mountain line,
  the way to the enemy's keep), the terrain an enemy two moves away
  stands to cross, and "I look there and it is empty" beyond 6 hexes of
  its units.
- **Why it is so.** The set was introduced on 2026-09-11 for speed: the
  full board was 2.67 times the tokens, and the subset ran 2.8 to 4 times
  faster (`docs/model_cost_study_20260905.md`). The network trained on it
  read +56 ± 12 Elo over its full-board twin after one pass of imitation
  (`relset`). The September audit listed the rest as its gap 4
  (`docs/observation_parity_20260926.md`), and version 2 of the set
  closed a small part of it (+9% tokens). The remainder is not on the
  list of differences the user accepted
  (`docs/parity_memory_design_20260929.md`, "Known unfaithfulness").
- **What closing it costs.** Every hex the side sees: 661 tokens, 1.8
  times today's. The whole board: 1,030, 2.8 times. The network's cost
  grows faster than the token count. A network trained without those
  tokens does not read them unchanged: this is not a change that can
  start at zero like the new unit columns.
- **Nothing leaks:** every rule of the set reads the side's own view.

### 2. The network cannot choose what a unit becomes when it levels up

- **A player chooses.** When a unit with several advancements levels up
  on its own side's turn, Wesnoth asks its owner
  (`src/actions/advancement.cpp:205-207`, 1.18.4). On the enemy's turn it
  takes the first type listed (`:232-237`).
- **The network has no such order,** and the simulator always takes the
  first type (`rust/wesnoth_core/src/core_units.rs:100-110`): the
  simple loop's games and its matches never switch on the uniform draw
  that the earlier self-play used. Every record carries
  `uniform_advancement: false`.
- **So in every game** a Spearman becomes a Swordsman (never a Pikeman or
  a Javelineer), a Mage a White Mage (never a Red Mage), an Elvish
  Fighter a Captain, an Elvish Archer a Ranger, a Troll Whelp a Troll, a
  Dark Sorcerer the first of its two.
- **Measured** (`audit_counts_round_0576.json`, 40 games): 111
  advancements, 42 of them from a type with a choice, and the first
  option was taken in all 42. 26 of the 42 came on the unit's own side's
  turn, where a player would have chosen: 0.65 a game.
- **Known in part:** `docs/status.md` lists "the advancement" among what
  is decided beside the network. New here: it is always the first
  option, in training as in matches, and two docstrings say otherwise
  (`tools/wesnoth_sim.py:29-31`, `:737-740`).

### 3. The weapon is scored without the enemy it strikes

Known since 2026-11-24 (`docs/input_facts_design_20261124.md`, Part 3,
not built and not yet put to the user). The audit confirms it is the only
pair of choices a player makes together and the network makes apart: the
target reads which unit acts, a recruit's hex reads which type is
recruited, and ending the turn reads the whole board.

## The gaps of medium weight

### 4. A move does not stop when an enemy comes into view

- **In Wesnoth,** a player's move into fog stops at the hex where a new
  enemy is sighted; the unit keeps its movement and the player decides
  again (`src/actions/move.cpp:560-566`, `:955-957`).
- **In the simulator** a move runs to its end; it is cut short only by a
  hidden unit in its way (`core_move.rs`). The replays it exports say
  `skip_sighted="all"`, so Wesnoth replays them without a difference.
  The rule is written in `docs/wesnoth_rules.md:645-656`.
- **Measured:** of 13,349 moves, 964 (7.2%) brought an enemy into view
  that the side did not see before, 895 of them moves of two steps or
  more: 22 a game. That is an upper bound on the moves that would have
  stopped early, since it counts enemies sighted at the last step and
  ambushers uncovered.
- **Why it matters:** a scout sent eight hexes into fog walks on past the
  army it uncovers at the second. The network can order one short move
  after another instead, and nothing teaches it to.

### 5. The reach the interface draws is not in the input

- **A player sees,** on selecting any unit, every hex it can reach this
  turn; for an enemy, with its full movement
  (`src/mouse_events.cpp:370-388`). "Show enemy moves" draws all the
  enemies' reach at once (`src/menu_events.cpp:466-503`).
- **The network gets** its own units' reach as the legal moves offered
  once the unit is chosen, and as which hexes have a token. It gets no
  reach of the enemy, and with gap 1 it often lacks the hexes to work it
  out. Version 2 of the set added the six hexes around each own unit,
  where an enemy must stand to strike it. It did not add the hexes the
  enemies can reach: September's count put those at 52% more hex tokens
  (`docs/observation_parity_20260926.md`, gap 4).

### 6. The enemy's faction is known from the first decision

- **The network's input** holds the chances of each faction for the
  enemy. In every game of the loop and every match it is certain from
  the first decision: 80 of 80 game-sides measured. The games' sides are
  built as if each player had picked a faction openly
  (`wesnoth_ai/rules/scenario_pool.py:800-823`), which is the convention
  written in `docs/parity_memory_design_20260929.md`.
- **A player** knows the enemy's faction from the start only when it was
  picked openly. Against a Random pick, which draws uniformly as these
  games do, a player learns it from the units the enemy shows.
- **To rule on:** which of the two the games stand for. Under Random,
  this is a leak, and the first recruits are chosen knowing what the
  enemy will bring.

### 7. Anything seen before the enemy's last turn rests on the learned memory

- **The record of units seen and lost to fog** is emptied at each of the
  side's own end of turn (`core_sight.rs:112-118`). Older sightings, the
  side's own losses, and who struck whom during the enemy's turn reach
  the network only through its 64 memory slots.
- **What the memory holds is not measured on the current network.** On
  `own2` it was worth +13 ± 12 Elo and held one pattern following the
  game's phase (`docs/status.md`). The loss that taught it to track
  hidden units in imitation is not part of the loop's loss.
- **A unit seen and lost keeps little:** its type, hit points and last
  hex. Not that it was the enemy's leader, nor its experience, traits or
  poison (`core_sight.rs:31-39`).

## A suspicion that a run cleared: the half-precision serving

- **The suspicion** (tensor report). Games and matches are served in
  half precision (fp16), which keeps about three decimal digits. The new
  unit columns' weights are still small (6.2e-5 at round 576), so what
  they add to a unit's token is about one rounding step of that token.
  The serving keeps the sum exactly in a float32 stream, but every matrix
  product reads it rounded. The new numbers might then reach play only
  as rounding noise.
- **The run** (`new_term_reach_round_0576.json`): 2,206 decisions of
  eight game-sides played by round 576's network. For each chosen
  action, its log probability was recomputed on the CPU in full
  precision with the new columns' weights and with those weights at
  zero. The difference is what the new columns do to the policy; the
  probability the player logged while serving in fp16 was then set
  against it.

  | | |
  |---|---|
  | What the new columns change in a chosen action's log probability | 0.0015 on average, 0.0035 root mean square, 0.058 at most |
  | Served against the full-precision network with the new columns | 0.0035 root mean square (the rounding's own noise) |
  | Served against the network without them | 0.0050 |
  | Share of the new columns' effect found in the served policy | 1.02 ± 0.02 |

- **So the serving carries the new columns in full.** The rounding adds
  noise of the same size as their present effect and takes nothing away
  on average. After 32 rounds they move the policy a little, and that
  little is played.
- **What stays true:** the loop's exact check (a pass must reproduce the
  logged probabilities within 0.014 root mean square) could not by itself
  tell a serving that left the new columns out, as long as their effect
  is under that size. This run is the check that it does not.
- **Not checked** (it needs the GPU): whether the gradient pass, which
  also builds its input tokens in fp16, loses the smallest of the new
  weights there. The pass's gradient does not depend on it, and the
  weights grow as recorded.

## The small ones

| Gap | Where | Weight |
|---|---|---|
| The hex where a recruit bounced is not shown. The column is written 0 for every network on the present observation (`core_encode.rs:370-381`), while the legal-move mask reads it. CLAUDE.md's Principle 6 says the encoder shows it. | 3 of 96,718 recruits in rounds 561 to 576, all on Arcanclave Citadel, none refused | rare; the principle's text and the code disagree |
| Attack or move is chosen from the unit's own token, before and without the target's scores (`wesnoth_ai/model.py:409`) | every unit decision | a valid factoring; costs learning, not reach |
| A mounted unit's or a gryphon's chance to be hit on mixed terrain (forest on hills, mushrooms on grass) does not follow from the per-terrain columns: a cap is stored as a plain value (`terrain.rs:123`) | Horseman line, Cavalryman line, Gryphon Rider line | exact where the unit stands; elsewhere learnt by type |
| The fights' six directions are tied to hex tokens only through learned positions; the model has no notion of distance or of which hexes touch | every decision | structural; the network plays moves and attacks through it today |
| Recruit rows carry no trait, while the recruit dialog shows a type's fixed traits (undead, feral) | Undead recruits, Vampire Bat | small |
| The oasis heals and cures and looks like plain ground | Ruined Passage, 4 hexes | small |
| Own upkeep and own unit count have no column (derivable from the units); the starting gold (75 to 175 by map) has none | every game | small |
| Aethermaw's terrain change on turns 4 to 6, announced in its objectives, shows only once it happens; Tombs of Kesorak's shifted time areas have no next-turn value per hex | 2 maps | small |
| The statistics dialog (losses, kills by type, damage against expectation) has no input | every game | history, not present state |
| The "has attacked" column is also filled for enemy units; no report shows it | every game | a leak by the letter: a player watched the attack |
| A weapon's accuracy has no column | Elvish Champion's sword | negligible; the fights' columns include it |
| An own unit's "invisible" status has no column | ambush, nightstalk, concealment, submerge | derivable |
| Orders the network cannot give: delaying the fog's update, undo, a route other than the default | every game | little or none |
| Hidden units count in a visible unit's leadership and in a hex's lighting | rare | as the interface itself shows; an unrecorded exception to Principle 6 |

## Checked and found right

The reports' coverage lists hold each item with its file and line. In
short:
- **Units:** type, level, alignment, cost, hit points, experience,
  movement, the 13 traits, 13 abilities and 10 weapon specials, poison
  and slow, resistances, up to 3 weapons, leader. All 158 unit types the
  six factions can field and the 11 walking-corpse forms have their own
  id.
- **Map:** all 333 terrain codes on the 21 maps get a terrain set, and
  two codes with one set never differ in cost or defence for any unit
  (but on hexes no unit can enter). Villages and their owner under fog
  as the game draws them; keeps and castles; fog; each hex's time of
  day. Shroud is refused for these games. The largest map (47 by 46) is
  inside every limit.
- **Game state:** turn, side, gold, income, villages, the time of day
  now and next, village gold and support. The enemy's gold and income
  are hidden under fog, as in the status table. The enemy's village
  count under fog is only what the side sees: the flag that hides the
  rest (`fog_hides_enemy_villages`) is on in round 576's checkpoint and
  in the reference's, read from both.
- **Outputs:** the masks match Wesnoth for moves into fog, partial
  moves, zones of control and skirmishers, a leader recruiting from
  another keep, a castle partly held, every weapon, no move after an
  attack, and ending the turn at any moment. Nothing beside the network
  alters the trained player's choice but the legality mask.
- **Memory:** the same in play, in the gradient pass and in matches;
  carried through the whole game; no memory or sighting crosses sides or
  games.
- **Tensor to decision:** every path builds unit tokens through one
  function, so none leaves the new columns out; padding and masks line
  up; the past self that reads 109 columns gets rows of its own.

## Claims that did not hold as written

- **"Self-play draws the advancement at random and matches take the
  first"** (units report). The simple loop's self-play runs through the
  match's own path: both take the first (gap 2).
- **"About half the board is absent"** (map report, inferred from
  September's figures). Measured: 65%.
- **"The fp16 serving path has no test with loud weights"** (tensor
  report). No automated test, but the check was run by hand on the GPU
  before the switch and is recorded
  (`docs/input_facts_prereg_20261125.md`, Results).
- **"The new columns reach play only partly through the fp16 serving"**
  (tensor report, from the arithmetic). Measured: in full (above).
- **"The enemy's village count leaks unless a flag is on"** (state and
  truth reports, which could not read the flag). It is on.
- **"A recruit moved or refused by a hidden enemy is lost entirely"**
  (memory report). True as written, and it happened 3 times in 96,718
  recruits.

## Beside the audit

- **The fights' columns are thinly tested** (truth report). The tests on
  played games compare the columns with the same functions that fill
  them. The hand-computed cases cover night, resistance, range, slow,
  leadership and fog. Marksman, magical, charge, backstab, steadfast,
  swarm, drain, first strike, berserk and a slowed defender are read as
  right in the code and pinned by no test of the columns. The combat
  itself is pinned strike by strike against Wesnoth
  (`tests/test_combat_seed_alignment.py`).
- **Stale comments** to correct: `wesnoth_ai/action_sampler.py:36-37`
  (attack from a distance), `:41` (ending the turn is always legal);
  `tools/wesnoth_sim.py:29-31`, `:737-740` (the uniform draw),
  `:1276-1278` (moves are not limited to touching hexes);
  `wesnoth_ai/encoder.py:198-204` (an old list of global columns),
  `:310-312` (the lighting counts every unit); `wesnoth_ai/constants.py:106-109`
  (values above the divisors do occur, and nothing is clipped);
  `wesnoth_ai/rules/scenario_pool.py:672-677`.
- **Runs the reports propose and this audit did not make,** since they
  need the GPU the training holds: the current network with its memory
  against itself without (gap 7), and the memory reset each turn.

## What is the user's to decide

One at a time, in the order this audit would take them:
1. **The weapon reading its target** (gap 3): the design's Part 3.
2. **The advancement as the network's choice** (gap 2): a new kind of
   decision, about 0.65 a game.
3. **Moves that stop on a sighting** (gap 4): a change to the simulator
   that brings it to what Wesnoth does for a player.
4. **The enemy's faction** (gap 6): openly picked, or Random.
5. **The board** (gap 1), and with it the enemy's reach (gap 5): the
   largest gap and the costliest to close.
6. **The bounced recruit's hex:** make the code match Principle 6, or
   the principle's text match the code.
