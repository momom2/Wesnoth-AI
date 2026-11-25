# Level, defence and the fight's numbers in the network's input: design (2026-11-24)

**Status: approved on 2026-11-24 and built, but for Part 3. The run takes
it from round 545** (`docs/input_facts_prereg_20261125.md`).
- The user, on the version below: "Alright, remove attack from a
  distance. For the rest approved." And on the training of the new
  weights: "No special learning rate [...] maybe we'll reconsider
  tomorrow if the night's training is clearly flawed or stalled."
- **Built as written:** Part 1, Part 2, and the removal of the attack
  from a distance (decision 2), on the branch `feature/input-facts`.
- **Built otherwise than written, in one place.** The new columns are not
  appended to `unit_feat_proj.weight`. They are read by a projection of
  their own, `unit_feat_proj2`, at zero. Zero columns appended to the old
  weight and to Adam's averages of it would have moved two to four times
  the learning rate for some hundred rounds, because Adam corrects its
  averages by a parameter's step count. A new parameter gets Adam's
  ordinary step, which is what "no special learning rate" asks. The
  widened network is then also the old one to the last bit.
- **Not built:** Part 3 (the weapon's score reading the target, decision
  3). It changes the output and has not been put to the user.
- **The time of day,** which the user asked about the same evening
  ("make sure the network has ToD visibility like a human: globally and
  per hex"), was already in the input; the registration lists where.

The history of this document:
- The user ruled on 2026-11-24, after watching a game of the simple loop's
  network: "for sure the network needs that information".
- The user read the first version the same day, did not approve it, and
  ruled two things:
  - **No token per possible attack:** "the number of possible attacks is
    astronomically high". The fight's numbers are columns of the
    attacker's row.
  - **No outcome of a fight is worked out for the network:** "If you mean
    the modal or expected outcome, then no, that's for the network to
    learn." The first version's four such columns are removed (Part 2).
- The first version's wording was unclear to the user: "six neighbouring
  hexes holding an enemy", "the dialog", "the predicted outcome". This
  version says each in plain terms.
- How the new weights are trained was not decided then: "We'll discuss how
  to train it". The last section holds what was known and the options.
- It then waited for the user: the design as a whole, and two points of
  it ("Decisions this design raises").

## In short

A player is shown a unit's level, its defence on any hex, and, before
every attack, the damage, strikes and chance to hit of both sides. The
network is shown none of these. This design adds them as columns of the
unit's row:

1. **What the sidebar shows:** the unit's level, its chance to be hit
   where it stands and on each kind of terrain, and its movement costs.
2. **The numbers of a fight the unit can start now.** In Wesnoth a unit
   attacks only an enemy standing on a hex next to its own, whichever
   weapon it uses. When a player orders the attack, a window opens to
   choose the weapon: "the attack dialog" below. For each weapon it lists
   the damage per hit, the strikes and the chance to hit, of both sides.
   The network gets those numbers for each of our units that can attack,
   each enemy next to it, and each of its weapons.
3. **The weapon reads the target:** the score of a target and of a weapon
   gets the dialog's line for that pair. Today the weapon is scored from
   the attacker's row alone.

- Parts 1 and 2 are input only: columns of the row that the network's
  input layer turns into the unit's token. Part 3 is a change at the
  output.
- There is no new kind of token and no change to the trunk.
- Every new weight starts at zero, so the widened network plays exactly
  the game it plays today on the day the columns arrive.
- Nothing hand-set decides anything (principle 3): each column is a number
  the interface shows a player.

## Why

### The ruling in force
On 2026-09-29 the user ruled: "For each item of the present state that a
player sees, the network sees it too"
(`docs/parity_memory_design_20260929.md`, ruling 1). The gap list of
2026-09-26 (`docs/observation_parity_20260926.md`) named two rows that the
retrain did not close:
- "Unit type, level, abilities, advancements: through the type's embedding
  row only". The retrain added the abilities.
- "Defense and movement cost per terrain: no; the type row with the terrain
  token".

The attack dialog was never on that list.

### What was found on 2026-11-24
The user's observations on a game of round 503's network: it places
level-0 units as if they held a zone of control, it sometimes attacks
with a damage type the target resists, its recruiting and its use of
experience look weak. The audit that followed
(`training/metrics/input_audit_20261124/`):

- **The simulator's rule is right.** A level-0 unit holds no zone
  (`rust/wesnoth_core/src/core_observe.rs:67`, `core_move.rs:172`,
  `tests/test_zone_of_control.py`). Over 11,622 positions of 24 fresh
  games, 1,236 of them with a level-0 enemy in view, the core's zone flags
  equal "the neighbours of each visible enemy of level 1 or more" every
  time.
- **The rows hold what they claim.** On a real position, 18 unit rows
  decode back to the database's resistances, weapon types and ranges,
  hit points and cost. Every weapon special, ability and trait of the 158
  unit types a default-era game can hold has a column, and none has more
  than three attacks.
- **The rows do not hold:** the level, any defence, the movement costs,
  what the unit advances to.
- **Before an attack the network sees no number of the fight,** and the
  weapon is scored without the target (`wesnoth_ai/model.py:1174`: the
  weapon head reads the attacker's row; the module's docstring says "Not
  conditioned on target yet").
- **Its choices agree with the fight's numbers in all but a few percent
  of attacks** (24 games of round 513's network, sampling as the loop
  does; 2,637 attacks):

  | Reading | Count |
  |---|---|
  | Its most likely weapon is beaten by another | 22 of 2,196 attacks by a unit with two weapons (1.0%) |
  | An attack into a resistance of 40% or more | 96 (3.6%); another weapon would deal 1.5 times as much in 20, another adjacent target would take 1.5 times as much in 5 |
  | Expected to take 1.5 times what it deals, without a likely kill | 133 (5.0%); 108 of them its most likely action |
  | Several adjacent targets | 308 of the 2,196; the best weapon differs between them in 76 |

  "Beaten" is: another weapon deals 25% more expected damage and takes no
  more, or deals as much (within 10%) and takes 3 hit points less.
  Expected damage is chance to hit × damage × strikes, without the cut-off
  at a death.
- **The game the user watched** (round 503's network, Drakes against
  Undead): 22 of the Drakes' 86 attacks went into a resistance of 40% or
  more, none of the Undead's 42. On turn 17 three Drake Fighters struck
  the Skeleton at 18,10 with the war blade (it takes 60% of blade and
  answers 8×3 at 70%) where fire breath deals about as much and it cannot
  answer. On turn 7 the Drake Clasher at 29,7 used its spear on the
  Skeleton at 30,6 (40% of pierce) where its war talon would deal half as
  much again.

So the measured cost at the moment of the attack is a few percent of
attacks. What the missing level and defence cost in moves and recruits
was not measured. The ruling decides on principle: a player has these
numbers, so the network gets them.

## What a player is shown

Wesnoth 1.18.4, GitHub tag, read 2026-11-24.

| What | Where | Source |
|---|---|---|
| The unit's level | sidebar | `src/reports.cpp:277-291` |
| Its experience and the cap | sidebar | `src/reports.cpp:530-545` (in the row today) |
| What it can advance to, as icons | sidebar | `src/reports.cpp:557-566` |
| Its defence on the hex the mouse points at, else where it stands | sidebar | `src/reports.cpp:577-627` |
| Its movement cost on each terrain | sidebar tooltip | `src/reports.cpp:671-752` |
| Per weapon: damage per hit, strikes and chance to hit of both sides, and the weapon that answers | attack dialog | `src/gui/dialogs/unit_attack.cpp:84-200` |
| The exact chance of each ending of the fight: the hit points each unit is left with, a death (not given to the network: the user's ruling, Part 2) | the dialog's "Damage Calculations" button | `src/gui/dialogs/unit_attack.cpp:57-67` |
| The same dialog from the hex a move-and-attack would strike from, before the unit moves | attack dialog | `src/mouse_events.cpp:926-946` |

`reports.cpp:619-627`, the defence a player reads:
```cpp
	const map_location& mouseover_hex = rc.screen().mouseover_hex();
	const map_location& displayed_unit_hex = rc.screen().displayed_unit_hex();
	const map_location& hex = (mouseover_hex.valid() && !viewing_team.shrouded(mouseover_hex)) ? mouseover_hex : displayed_unit_hex;
	return unit_defense(rc, u, hex);
```
`unit_attack.cpp:184-188`, one weapon's line:
```cpp
		attacker_stats << "<b>" << attw_name << "</b>" << "\n"
			<< attw_type << attw_type_second << "\n"
			<< attacker.damage << font::weapon_numbers_sep << attacker.num_blows
			<< attw_specials << "\n"
			<< font::span_color(a_cth_color) << attacker.chance_to_hit << "%</span>";
```
`mouse_events.cpp:936-945`, the dialog before a move-and-attack:
```cpp
					// block where we temporary move the unit
					{
						temporary_unit_mover temp_mover(pc_.get_units(), src, attack_from, itor->move_left, true);
						choice = show_attack_dialog(attack_from, clicked_u->get_location());
					}

					if(choice < 0) {
						// user hit cancel, don't start move+attack
						return;
					}
```

## Part 1: what the sidebar shows

Added to every unit's row, ours and the enemy's (a player reads them for
any unit it sees). 34 columns.

| Columns | Value |
|---|---|
| 1 | the level / 4 (the highest level of the 158 types is 4) |
| 1 | the chance to be hit where it stands / 100: the number the fight reads (`core_attack.rs:168-174`, `defense_at`) |
| 16 | the chance to be hit on each of the 16 terrain classes / 100, from the unit's own table (`core.rs:339-346`, `class_table`: the feral trait's village value included) |
| 16 | one over the movement cost on each terrain class, 0 where it cannot go, as the unit pays them now (doubled when slowed) |

- **The level settles the zone of control** (level 1 or more), the upkeep
  and the experience a fight gives the enemy (its level per fight, 8 times
  its level per kill, 4 for a level-0 unit).
- **The table gives the defence on a hex the unit could move to.** The
  network pairs it with the hex's terrain classes, as a player pairs the
  sidebar with the hex it points at. On a hex of two classes (a forested
  hill) the game takes the better or the worse of the two by the terrain's
  rule; the sidebar's tooltip lists them the same way.
- **A recruit option** takes the level and the two tables from its type.
  The column "where it stands" is 0.

## Part 2: what the attack dialog shows

**Whom a unit can attack.** Every attack in Wesnoth, with a melee or a
ranged weapon, is made on an enemy standing on one of the six hexes that
touch the attacker's hex. "Ranged" only decides which of the enemy's
weapons can answer. So at any moment a unit has at most six possible
targets. In the 24 measured games, 1,888 of the 2,196 attacks by a
two-weapon unit had one enemy next to the attacker and 308 had several.

**The columns.** The row of each of our units with an attack left gets
one slot for each of the six directions (north, north-east, south-east,
south, south-west, north-west: `observe.rs:26-34`). A slot is filled when
an enemy the unit may attack stands on that hex, and is zero otherwise.
A filled slot holds 11 columns for each of the unit's three weapon slots:
198 columns in all.

| Columns | Value |
|---|---|
| 1 | this weapon can strike this enemy |
| 3 | our damage per hit / 40, strikes / 6, chance to hit / 100 |
| 1 | the enemy answers |
| 3 | the answer's damage per hit / 40, strikes / 6, chance to hit / 100 |
| 3 | which of the enemy's weapon slots answers (one-hot) |

- **The numbers are the fight's own.** The core's `fight_inputs`
  (`core_attack.rs:227-256`) and the answer the engine picks
  (`outcomes.rs:498-536`, `counter_weapon`) are what the attack command
  runs, so the columns carry resistance, time of day, leadership, slow,
  backstab, charge, marksman, magical and steadfast as the fight applies
  them.
- **Everything else is zero:** an enemy's row, a unit that has attacked, a
  direction with no enemy the unit may attack, a recruit option.
- **The enemy's weapon specials** (poison, slow, drain) are in the enemy's
  own row; the one-hot says which weapon they belong to.
- **No outcome is worked out for the network** (the user, 2026-11-24).
  The dice decide a fight. The game can show a player the exact chance of
  each ending (the "Damage Calculations" window), and the first version
  gave the network four numbers from it: each side's chance to die and
  the average hit points each is left with. The user: "that's for the
  network to learn." What stays is what one strike does and how often it
  lands, for both sides.
- **The cost in time is measured before the columns are switched on.**
  They are arithmetic on numbers the core already holds.

The unit's row grows from 109 columns to 109 + 34 + 198 = 341.

## Part 3: the weapon reads the target

Today (`wesnoth_ai/action_sampler.py:126-133`):

    P(unit u attacks hex h with weapon w)
        = P(u) × P(attack | u) × P(h | u, attack) × P(w | u)

The last factor does not see h: one unit's weapon is scored the same
whichever enemy it strikes. That holds even with Part 2's columns in the
row, because the weapon's score is read from the attacker's token alone.
This part is not input: it changes how the output is computed. It is put
to the user as its own item (decision 3 below).

As first written, two terms are added, each a learned weighting of Part
2's 11 numbers, starting at zero:

- the score of weapon w for target h gains `weapon_line · line(u, h, w)`;
- the score of target h for an attack gains
  `target_line · [line(u, h, 1), line(u, h, 2), line(u, h, 3)]`.

`line(u, h, w)` is read from u's own row: h is one of the six hexes next
to u. For a target that is not next to u (the attack from a distance,
below) the line is zero and the scores are today's.

## What changes in the code

- **The core** (`rust/wesnoth_core/src/core_parity.rs`): `unit_extra` and
  `recruit_extra` write the new columns, `parity_layout` names them, and
  `__phase__` goes up so the adapter refuses an older wheel. Both venvs
  hold the wheel, so the run stops for the rebuild.
- **The encoder** (`wesnoth_ai/encoder.py`): the widths, behind a new
  checkpoint key `unit_row_version` (1: today's 109 columns; 2: this
  design's 341). `OBSERVATION_EPOCH` goes to 12.
- **The run's checkpoint is widened once,** at the switch: zeros for the
  new columns of `unit_feat_proj.weight` and for their averages in Adam's
  state. `pad_legacy_encoder_state` and `repair_optimizer_state_shapes`
  (`wesnoth_ai/encoder.py`) do this for the layers that grew before; the
  first gains this layer. The loop's step (`tools/pg_adam_step.py`) loads
  Adam's state as saved, so the widening is a step of its own.
- **The network** (`wesnoth_ai/model.py`), if Part 3 stays as first
  written: two vectors, `weapon_line` (11) and `target_line` (33), zero at
  the start; the heads add the two terms.
- **The readers of the weapon's score** take it per target:
  `wesnoth_ai/action_sampler.py`, `wesnoth_ai/server_priors.py`, and the
  loop's pass (`tools/pg_game_pass.py`), which replays a decision's
  probability.
- **The reference and the pool's past selves** keep their own observation:
  a checkpoint without the key reads 109 columns.
- **A test** holds the claim "the same game on the day the columns
  arrive": on recorded positions, a checkpoint loaded into the widened
  network gives the logits it gave before, bit for bit.

In the run it is a numbered departure of
`docs/simple_loop_prereg_20261117.md`, with its own registration: what is
measured, the number that decides, the kill criteria (principle 1). That
registration is written once the training is decided.

## Not in this design

- **What a unit advances to.** The sidebar shows it as icons. It stays in
  the type's learned row: it is fixed per type.
- **The traits a recruit option always has** (undead, fearless, feral).
  They are fixed per type too. Setting the existing trait columns on a
  recruit option would change the present network's play at once.
- **The dialog from a hex the unit has not moved to.** A player sees it
  before a move-and-attack. In 2,629 of 2,637 attacks the network's unit
  already stood next to its target, and Part 1 gives it the ingredients (its defence
  by terrain, the enemy's chance to be hit where it stands, resistances,
  time of day). The numbers themselves arrive once it stands next to the
  enemy.
- **Where an enemy can move** (the reach a player sees by clicking it).
- **A mark on hexes inside a zone of control.** The level gives it, and
  the interface shows no such mark either.

## Decisions this design raises

1. **Where the fight's numbers go: decided, columns of the attacker's
   row** (the user, 2026-11-24). The other form was one token per possible
   attack (attacker, target, weapon), which the network would pick
   directly. Rejected by the user: "the number of possible attacks is
   astronomically high (combinatorial explosion)". It was not the
   recommended form either: a new kind of token changes what every other
   token attends to, so the network would not start as the same player.
2. **The attack from a distance.** An attack is one decision: a unit, an
   enemy, a weapon. The legal actions offer two kinds:
   - on an enemy next to the unit: the fight happens where the unit
     stands;
   - on an enemy the unit is not next to but can reach this turn: the
     simulator first moves the unit to a hex next to that enemy, then
     runs the fight. It picks that hex by a fixed rule
     (`tools/wesnoth_sim.py:271-319`, the cheapest route). The network
     does not choose it.

   That rule decides play beside the network, against principle 3. A
   player ordering the same thing chooses the hex with the mouse
   (`mouse_events.cpp:691-760`). In 24 games the network's attacks were
   of the first kind 2,629 times of 2,637: the unit already stood next to
   its target. It got there by a move decided earlier or the enemy came
   to it; that split was not counted.
   - Recommended: remove the second kind from the legal actions.
3. **How the weapon comes to read the target** (Part 3; not yet put to
   the user). Two forms, both starting at zero:
   - as first written: the weapon's and the target's scores read Part 2's
     numbers directly, beside the network;
   - the weapon's score reads the target's token as well as the
     attacker's. Everything the network knows of the fight then comes
     through its input, and nothing is wired from an input column to an
     output.

## Training the new weights: open

To be discussed with the user. What is known:

- **A zero start changes nothing by itself.** The widened network is the
  same function until the new weights move.
- **At the loop's rate they would barely move.** With Adam at 1e-5 a
  weight moves at most about 1e-5 a round. The loop's whole update has a
  norm of 0.0058 a round over 15.6 million weights, 1.5e-6 a weight on
  average (rounds 481 to 500). The weights that read the unit's present
  columns average 0.07 in size. Reaching that takes 7,000 rounds at the
  fastest and about 48,000 at the average pace; the run plays about 80 a
  day.
- **Imitating its own games teaches the new weights nothing.** The widened
  network already reproduces its own choices exactly.

Options, none chosen:

| Option | What it touches |
|---|---|
| Raise the one learning rate for the whole network for a stretch | keeps one rate; changes the whole run's dynamics |
| A larger rate for the new weights until they reach the others' size | the one-rate ruling of 2026-11-21, for a stretch |
| Fit the new weights alone, outside the loop, on recorded games | close in kind to the one-off refit declined on 2026-11-23 |
| Retrain from the imitation stage with the new columns, as the parity retrain did | imitation is parked (2026-10-04); gives up the loop's rounds |
