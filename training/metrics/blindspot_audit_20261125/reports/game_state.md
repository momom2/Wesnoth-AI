# Blind-spot audit, part "the game beyond the board" (2026-11-25)

Scope: the status bar, the status table (both tabs), the statistics
dialog, the scenario's settings and objectives, the turn limit, sides and
factions, and the 21 Ladder maps' own rules, compared with the global token
and the other inputs that carry game-level facts.

Sources read. In the repository: `wesnoth_ai/encoder.py`,
`rust/wesnoth_core/src/core_encode.rs`, `encode.rs`, `core_parity.rs`,
`core_observe.rs`, `core_move.rs`, `events.rs`, `wesnoth_ai/game_core.py`,
`wesnoth_ai/faction_posterior.py`, `wesnoth_ai/classes.py`,
`wesnoth_ai/rules/scenario_pool.py`, `wesnoth_ai/constants.py`,
`tools/wesnoth_sim.py`, `tools/run_elo_batch.py`, `tools/elo_eval_game.py`,
`scripts/simple_loop_local.py`, the 21 scenario cfgs under
`wesnoth_src/data/multiplayer/scenarios/`, and two map files. From Wesnoth
1.18.4 (GitHub raw, copies kept in `audit_20261125/src_gs/`):
`src/reports.cpp`, `src/gui/dialogs/game_stats.cpp`,
`src/gui/dialogs/statistics_dialog.cpp`, `src/team.cpp`, `src/team.hpp`,
`src/display_context.cpp`, `src/units/unit.cpp`, `src/menu_events.cpp`,
`src/hotkey/hotkey_handler.cpp`. I also read the game records of the live
run's rounds 540 to 589 (5,760 games) for how games end.

The network in question reads the parity observation with the unit row of
version 2 (`encoder.py:316-326`, 15 global columns), so the global row is:

| col | content | built at |
|---|---|---|
| 0 | turn / 60 | `encode.rs:165`, `core_encode.rs:442` |
| 1 | the encoded side, -1 (side 1) or +1 (side 2) | `encode.rs:166` |
| 2 | own gold / 500 | `core_encode.rs:425` |
| 3 | own base income / 50 (2 + the [side] income offset) | `core_encode.rs:426` |
| 4 | own villages / 30 | `core_encode.rs:427` |
| 5 | enemy villages / 30: the true count, or under the checkpoint flag `fog_hides_enemy_villages` with fog on, the enemy villages on hexes the side sees | `core_encode.rs:428-433` |
| 6, 7 | the board's lawful bonus this turn and next / 25 | `core_encode.rs:444-445` |
| 8, 9 | village gold / 8, village support / 4 | `core_parity.rs:410-411` |
| 10 | own net income as the status table computes it / 50 | `core_parity.rs:385-398, 412` |
| 11 | fog on | `core_parity.rs:413` |
| 12-14 | enemy gold, net income, upkeep: only with fog off, else 0 | `core_parity.rs:416-421` |

plus `our_faction_embed` and the enemy-faction posterior
(`encoder.py:1255-1268`, `game_core.py:949, 956-965`).

---

## 1. Findings, most serious first

Nothing in this part is a large blind spot. The economy, the turn, the
side, the time of day and the game's settings are all present or follow
from a fixed per-pool value. What follows are a convention worth the
owner's ruling (F1), one leak that hangs on a checkpoint flag I could not
read (F2), and smaller gaps.

### F1. The enemy's faction is known from turn 1 in every self-play and match game: the "openly chosen" convention

- **What a player knows.** In 1.18 MP, a faction chosen openly in the lobby
  is shown to everyone in the staging screen. A faction chosen as Random
  is not shown. Once the game runs, no game dialog names a side's faction:
  the status table's two tabs show the leader image (only if the leader is
  visible, or the side is "known", `game_stats.cpp:101-111`), team name,
  gold and so on (`:138-165`), and start gold, base income, village gold,
  village support, fog and shroud (`:177-202`). No faction column. So a
  Random opponent's faction is learnt from the units it shows.
- **What the network gets.** The parity input carries a posterior over the
  enemy's faction (`faction_posterior.py:97-118`). It is one-hot on the
  true faction whenever the enemy's side has `chose_random` false
  (`faction_posterior.py:89, 107`). The self-play and match games build
  their sides in `scenario_pool.build_scenario_gamestate`
  (`scenario_pool.py:800-823`) without a `chose_random` field, so
  `SideInfo.chose_random` keeps its default False (`classes.py:397`, read
  at `game_core.py:511`). In every self-play and match game, then, the
  network knows the enemy's faction exactly from its first decision. The
  harness draws the factions uniformly (`scenario_pool.py:438-439`), which
  is what a Random pick does.
- **Is it a leak?** It is consistent with the convention the project wrote
  down: "in eval the harness assigns factions openly"
  (`docs/parity_memory_design_20260929.md:114-116`), and the exported
  replays say `chose_random="no"` (`tools/sim_to_replay.py:956`). Under that
  convention a human in the same seat also knows the faction (from the
  lobby). Under the other reading, where the game is the ladder's usual
  Random-against-Random, it is a leak: the network sees the enemy's
  faction and so its recruit list before any enemy unit is seen.
  Leaders follow the Random rule (`scenario_pool.py:441-442`, drawn from
  `random_leader_pool`), and the leader's type is not in any game-level
  input. That matches the Random-leader case.
- **Why it matters.** The opening recruit is chosen against the faction:
  for example, Drakes against a known Undead (who will bring Dark Adepts,
  cold and arcane) or against a known Loyalists. Under a Random lobby a
  player recruits blind on turn 1 and reacts on turns 3 to 5.
- **Confidence.** Verified by reading. That the convention is the one
  intended is the owner's call.
- **Run to confirm.** On one fresh game from `random_setup` plus
  `build_scenario_gamestate`, encode side 1's first decision and print
  `raw.their_faction_probs`: expect a single 1.0 at the true faction's row.

### F2. Enemy village count under fog: a leak unless the checkpoint's `fog_hides_enemy_villages` is on

- **What a player sees.** With fog on, the status table fills an enemy's
  gold, villages, units, upkeep and income only if the side "knows about"
  it (`game_stats.cpp:90, 139-165`), and under fog an enemy is never known
  (`team.cpp:704-720`, "We don't know about enemies"). The status bar's
  villages report is the viewing side's own count (`reports.cpp:1465-1482`).
  So a player knows only the enemy flags on hexes it sees now (the map
  draws an enemy flag only on an unfogged hex, `docs/wesnoth_rules.md`
  under "Enemy side statistics under fog or shroud").
- **What the network gets.** Global column 5 is the enemy's true
  village count unless the checkpoint flag is set and fog is on
  (`core_encode.rs:428-433`). Every game of the loop is fogged:
  `random_setup`'s default category is "ladder" and `fogless` is False
  (`scenario_pool.py:380, 425`), and every Ladder cfg declares `fog=yes`.
  So column 5 is honest only if the running network was built with
  `fog_hides_enemy_villages=True`. The parity design builds its encoder
  that way (`docs/parity_memory_design_20260929.md:293`), and every parity
  test helper sets it (`tests/helpers/parity_games.py:86`,
  `tests/helpers/pg_games.py:76`). I could not read the flag out of the
  live checkpoint without torch.
- **Why it matters.** With the true count the network would know, for
  example, that the enemy has taken 9 villages behind its fog line where it
  sees only 5, and so roughly how much its income has grown: information
  no player has.
- **Confidence.** The code path is verified by reading. That the live
  network has the flag on is inferred.
- **Run to confirm.** `peek_checkpoint_arch(<round_0584/pi.pt>)["fog_hides_enemy_villages"]`
  and the same for `training/checkpoints/own2.pt`: expect True for both.

### F3. Own upkeep is not given; the status bar shows it on every turn

- **What a player sees.** The default theme's upkeep report reads
  "expenses (upkeep)": the gold deducted, and the total levels of the
  side's units (`reports.cpp:1489-1497`; `team_data` at
  `display_context.cpp:171-177`; leaders pay none, `unit.cpp:1746-1754`).
  The status table's own row shows the units, the upkeep and the net income
  (`game_stats.cpp:156-164`).
- **What the network gets.** The net income (column 10) and its parts:
  base income (3), villages (4), village gold (8), support (9). So the
  expenses can be recovered (base + villages × village gold − net). The
  total upkeep, the report's second number, cannot be recovered from the
  global row when the villages cover it all (expenses 0). It must be summed
  from the level column (109) over the side's non-leader unit tokens.
  Column 14 gives the **enemy's** upkeep (fog off only, so never in these
  games), but there is no own-upkeep column. The same goes for the unit count
  (`reports.cpp:1484-1487`): no column, only the count of own tokens.
  Searched: `core_parity.rs:401-423` and `core_encode.rs:441-446` are the
  whole global row.
- **Why it matters.** Small. Whether one more recruit costs income now (the
  villages' support is used up) is the question it answers, and the network
  can form it by attention over its own units. The asymmetry with column 14
  is odd.
- **Confidence.** Verified by reading.
- **Run to confirm.** None needed. A probe would be to regress
  `side_economy(side).0` on the trunk's global-token output.

### F4. Starting gold varies by map (75 to 175) and is shown in the status table's "Scenario Settings" tab; the network has no column for it

- **What a player sees.** The status table's second tab lists each side's
  start gold, base income, village gold, village support, fog and shroud,
  for every side, enemies included, with no fog gate
  (`game_stats.cpp:172-204`).
- **The pool's values.** `gold=75` on Aethermaw (`2p_Aethermaw.cfg:20,32`),
  175 on Arcanclave Citadel (`:23,36`), 150 on Thousand Stings Garrison
  (`:59,71`), 125 on Elensefar Courtyard (`:18,30`) and Tombs of Kesorak
  (`:102,114`), and 100 (stated, or the default at `scenario_pool.py:690-693`)
  on the other 16. The two players are equal on every map. Base income is 2
  for both everywhere (`scenario_pool.py:806, 817`; the only `income=` is
  on Thousand Stings' statue side 3, which the parse skips at line 657).
  Village gold is 2 and support 1 everywhere (`wml_state.py:49-50`; the
  only economy key set in the pool is `mp_village_gold=2` on Clearing Gushes
  and Walls of Pyrennis), and they are columns 8 and 9 anyway.
- **What the network gets.** Its own current gold (column 2), which equals
  the start gold only at its first decision. After that the start gold, and
  with it the enemy's start gold, comes from memory or from recognising the
  map from its hex tokens.
- **Why it matters.** Small: it bounds the enemy's army (at 175 gold the
  enemy fields about 50% more than at 100 by turn 2). It is fully
  determined by the map, which the network sees.
- **Confidence.** Verified by reading.
- **Run to confirm.** None needed.

### F5. Scripted future changes announced in the objectives are not in the input (Aethermaw), and per-hex next-turn time of day is not in the input (Tombs of Kesorak)

- **Aethermaw.** The objectives tell both players: "Beginning on turn 4, the
  Great Mage Sulla will begin to unite the two halves of the battlefield.
  The process will be complete by the end of turn 6"
  (`2p_Aethermaw.cfg`, prestart `[objectives]` notes). The events change
  terrain on side 1 and side 2's turns 4, 5 and 6 (`[terrain]` to `Chw`,
  `Wwf`, `Chw^Xo`). The core applies `[terrain]` actions
  (`events.rs:407-468`), so the network sees each change once it has
  happened, in the hex tokens. That it is coming, and when, is not an
  input. It is a fixed function of (map, turn), so the network can learn it
  from experience but cannot read it.
- **Tombs of Kesorak.** Three `[time_area]`s: four mausoleum hexes run the
  schedule one step darker, four campfire hexes are always "morning" (+25),
  and hex (19,12) is always first watch (−25) (`2p_Tombs_of_Kesorak.cfg:19-96`).
  The objectives note says so. The network has each hex's *current*
  bonus less the board's (`core_parity.rs:364-375`, `area_lawful_bonus` on
  unseen hexes) and the board's *next* bonus (`core_encode.rs:445`), but no
  per-hex next bonus. A player hovering a hex gets the area's full schedule
  and current step (`reports.cpp:1259-1281`). Elensefar Courtyard's area
  (`2p_Elensefar_Courtyard.cfg:79-118`, "Underground" illuminated) is a
  single fixed step, so for it current = next.
- **Why it matters.** Small. Example: on Kesorak, a chaotic unit holding a
  mausoleum hex at the board's dusk is already at night there; whether it
  stays so next turn is not given, though it is learnable per hex.
- **Confidence.** Verified by reading.
- **Run to confirm.** None needed.

### F6. The statistics dialog (own side's history) has no counterpart

- **What a player can look up.** The Statistics hotkey shows the viewing
  side's own record (`hotkey_handler.cpp:87-89` passes
  `viewing_team()+1`; `menu_events.cpp:140-143`): recruits, recalls,
  advancements, losses and kills, by unit type and with their gold cost,
  plus damage inflicted and taken against expectation
  (`statistics_dialog.cpp:109-128, 368-371`). The kills list is the
  enemy's losses by type.
- **What the network gets.** No aggregate. It has whatever its memory
  kept. This is history rather than present state, so under the owner's
  standard ("the present state") it is at most marginal. I list it because
  "enemy gold lost so far" is a common human estimate of the enemy's
  remaining army under fog.
- **Confidence.** Verified by reading (no input carries it: the global row
  is complete above).
- **Run to confirm.** None.

### F7. A hidden horizon: the 2,000-actions-per-side cap is invisible to the network, but it never binds now

- **What a player sees.** The turn report shows "turn/limit" when the game
  has a limit, with a tooltip saying the game ends past it
  (`reports.cpp:1428-1440`). No Ladder cfg sets `turns=` (searched
  `turns\s*=` over the 21 cfgs: no match).
- **What the games do.** They end at turn 200 (`run_elo_batch.py:595`
  default; the loop passes no `--max-turns`, `simple_loop_local.py:137-142`;
  game records say `"max_turns": 200`), as a draw that trains at z = 0. They
  also end at 2,000 actions by one side (`wesnoth_sim.py:349, 1230-1235`).
  The turn number is column 0 and the turn cap is constant, so turns
  remaining follow from column 0 (no blind spot). The action count is in
  no input.
- **Observed.** In rounds 540 to 589 (5,760 games) every game ended by a
  leader's death. Lengths were 3 to 71 turns (median 18, 99th percentile
  42), so neither cap binds. A record for the owner, not a present problem.
- **Confidence.** Verified by reading and by the game records.

---

## 2. Coverage: checked and found present and right

Status bar (`reports.cpp`):
- turn: column 0 (`core_encode.rs:442`, `encode.rs:165`); turn limit: none in the scenarios, constant 200 in the harness, so derivable (F7).
- gold (own; the whiteboard's planned spending does not apply): column 2 (`core_encode.rs:425`).
- villages, own "owned/total": own count column 4; the total is derivable because every village hex is always in the hex stream, relevant set or not (`core_observe.rs:139-142`) and carries the village bit.
- num_units: no column; count of own unit tokens (F3).
- upkeep "expenses (upkeep)": expenses derivable from columns 3, 4, 8, 9, 10; total upkeep only by summing levels (F3).
- income (net): column 10, same formula as `team_data` (`core_parity.rs:385-398` against `display_context.cpp:171-177`, `team.hpp:179-191`, `unit.cpp:1746-1754`: leaders free, loyal free, expenses = max(0, upkeep − villages × support)).
- time_of_day (board): columns 6 and 7. The six default steps give six distinct (this, next) pairs: dawn (0,+25), morning (+25,+25), afternoon (+25,0), dusk (0,−25), first watch (−25,−25), second watch (−25,0). So the step, and the whole cycle after it, is identified.
- tod_stats per hex (schedule position): current per hex in hex column 4 (`core_parity.rs:364-375`); per-hex next missing only on Kesorak's shifted area (F5).
- side_playing: the network is asked only on its own turn; column 1 says which seat it plays (`encode.rs:166`).
- observers, clock, battery, countdown: not game state (no timer in the sim).

Status table, "Current Status" tab (`game_stats.cpp:80-167`):
- own gold, villages, units, upkeep, income: columns 2, 4, 10; units and upkeep only by aggregation (F3).
- enemy under fog (every game here): gold, villages, units, upkeep and income hidden; columns 12-14 zero under fog (`core_parity.rs:416`). Column 5 depends on the flag (F2).
- enemy leader image and name: only if the leader is visible (`:101-111`); the network has the leader as a unit token when visible; no global leader field. Consistent.
- team names: cosmetic (north/south etc.).

Status table, "Scenario Settings" tab (`game_stats.cpp:172-204`):
- start gold: no column (F4).
- base income: column 3 (own); the enemy's is equal on every map.
- village gold, village support: columns 8, 9; equal for both sides.
- fog: column 11 (always 1 here); shroud: none of the 21 maps declares it, and the build refuses it (`scenario_pool.py:663-667`).

Scenario settings not in the table:
- experience modifier: 70 on all 21 (no cfg sets it; `wml_state.py:51`); shown per unit through max XP (unit column 4) and on recruit rows (`encoder.py:296-297`).
- time-of-day schedule and start: `random_start_time="no"` on all 21; DEFAULT_SCHEDULE on 19, DEFAULT_SCHEDULE_SECOND_WATCH (starting at second watch) on Fallenstar Lake and Ruined Passage. The start is read in `scenario_pool.py:484-527`, and columns 6 and 7 honour the offset (`core_encode.rs:434-445`).
- victory: no Ladder cfg overrides the defaults (defeat the enemy leader). The two maps with `[objectives]` (Aethermaw, Tombs) state "Defeat the enemy leader". The sim ends on a leader's death (`wesnoth_sim.py:1219-1240`). Every recent game ended that way.
- which side is first or second, and the frame: column 1; the encoding is from the mover's side ("ours"/"theirs", `encoder.py:12-15`; `core_encode.rs:397`).
- own faction: `our_faction_embed` (`encoder.py:1258`); own recruit list: recruit rows of the side's own list (`core_encode.rs:545-557`).
- enemy faction and its recruit list: the faction posterior (F1); the recruit list is implied by the faction.
- enemy leader type: no global field; it appears when seen, as for a Random leader (`scenario_pool.py:441-442`).

Per-map rules (21 maps; counts of `[objectives]`/`[event]`/`[time_area]`/extra sides/pre-owned `[village]`/placed units per cfg):
- Aethermaw: objectives notes; 10 prestart and turn events; terrain changes on turns 4-6 applied by `events.rs` and seen once they happen (F5). The cairn hexes are `Chw^Xo` (map read): "may not move into, may recruit to" is a map/mask matter (see below). Items and halos are decoration.
- Arcanclave Citadel: 175 gold; 2 pre-owned `[village]`s; no events.
- Caves of the Basilisk: side 3 of petrified statues, side code 2 (`core_move.rs:32-35`); 1 pre-owned village.
- Clearing Gushes: `mp_village_gold=2` (the default); side 2 owns (2,10) at the start.
- Den of Onis, Hamlets, Hellhole, Howling Ghost Badlands, Ruphus Isle, Swamp of Dread, The Freelands, Weldyn Channel: one pre-owned `[village]` each, decorative images only, no events.
- Elensefar Courtyard: 125 gold; a prestart `[time_area]` (Underground, a single step) seen per hex in hex column 4; items decorative.
- Fallenstar Lake, Ruined Passage: second-watch start (columns 6 and 7).
- Sablestone Delta: nothing special.
- Silverhead Crossing: side 3, an armed "Nani the Shapeshifter" (a Tentacle of the Deep given a 300-damage strike, an arcane magical "evil eye", submerge and +475 HP) at (1,1); its side ends its turn at once (`side 3 turn` event); a null controller, so the sim skips it (`scenario_pool.py:628-640`). See "For other parts" on its side code.
- Sulla's Ruins: side 3 statues (petrified), side code 2; 1 pre-owned village.
- Thousand Stings Garrison: 150 gold; side 3 statues; side 2 owns (16,3).
- Tombs of Kesorak: 125 gold; three time areas (F5); objectives notes; 1 pre-owned village.
- Walls of Pyrennis: `mp_village_gold=2`; side 2 owns (5,3); a decorative `terrain_graphics`.

No global column holds a value a player cannot know, apart from the conditional column 5 (F2) and the faction posterior under the Random reading (F1). Columns 12-14 are zero in every game.

## 3. To check by a run

1. `fog_hides_enemy_villages` in the live network and in `own2.pt`: `peek_checkpoint_arch(path)["fog_hides_enemy_villages"]` should be True (F2).
2. The enemy-faction posterior at side 1's first decision of one fresh pool game: one-hot on the truth (F1).
3. Optional, for F3: on 1,000 logged decisions, compare `side_economy(side)` upkeep with the expenses implied by columns 3, 4, 8, 9 and 10, to count how often the total upkeep is not recoverable from the global row (expenses 0 with upkeep > 0).

## 4. Could not determine

- The lobby's turn limit and timer in real Ladder games, and whether a desktop or live-bridge game against a human would ever show "turn/limit". The scenarios set none, and the harness uses 200.
- Whether the owner means the self-play game to be "factions chosen openly" (as documented) or "Random" (as the brief's "drawn at random" might suggest). F1 hangs on it.

## 5. For other parts

- Map: on Aethermaw, the hexes turned to `Chw` on turn 4, e.g. (13,13), (14,13), (15,14), read as `_off^_usr` in `2p_Aethermaw.map` (WML coordinates, border included). If that reading is right, the board gains hexes during play: check that the hex stream and the static hex arrays include them after the change (inferred, not checked).
- Map/actions: Aethermaw's `Chw^Xo` cairns: recruiting onto them must be legal and moving onto them not.
- Units: Silverhead's armed side-3 Tentacle is not scenery (`core_move.rs:34`: it has attacks), so it gets side code 1, the opponent's code (`core_encode.rs:397`). The network cannot tell it from an enemy player's unit except by type.
- Memory: the enemy leader's type once seen and then lost in fog; start gold after turn 1 (F4).
- Code comment: `scenario_pool.py:672-677` says Thousand Stings' `income=-2` overpaid "both sides"; that `income=` is on side 3 (`2p_Thousand_Stings_Garrison.cfg:86-87`), which the parse skips (line 657). The behaviour is right, but the comment is wrong.
