# Audit part: units (sidebar, unit description, recruit dialog, unit list)

Auditor's sources. Wesnoth C++ at the 1.18.4 tag, fetched to the scratchpad
(`.../scratchpad/wsrc/`): `src/reports.cpp`, `src/units/unit.cpp`,
`src/actions/advancement.cpp`, `src/gui/dialogs/unit_recruit.cpp`,
`src/gui/dialogs/unit_list.cpp`, `src/gui/widgets/unit_preview_pane.cpp`,
`data/themes/default.cfg`. The default theme's sidebar shows the reports
`unit_status, unit_image, unit_name, unit_side, unit_type, unit_level,
unit_alignment, unit_abilities, unit_race, unit_traits, unit_vision, unit_hp,
unit_xp, unit_moves, unit_defense, unit_weapons` (default.cfg:287-503).

Repository paths below are relative to
`C:\Users\momom2\Desktop\perso\projects\wesnoth_ai`. The unit row is the
341-column version 2 row (encoder.py:183-196, 276-373; filled by
rust/wesnoth_core/src/encode.rs:108-151, core_parity.rs:171-317,
core_encode.rs:383-503).

Reachable set used for the vocabulary checks: the recruits and leaders of the
six `*-default.cfg` factions (Dunefolk skipped, scenario_pool.py:212-236),
closed under `advances_to` in `unit_stats.json`: 158 base types, plus the
Walking Corpse / Soulless variations of 11 races (bat, drake, dwarf, goblin,
gryphon, mounted, saurian, swimmer, troll, wolf, wose).

---

## 1. Findings

### F1 (most serious). The advancement choice cannot be ordered; self-play draws it at random and matches always take the first option

- **What the player can order.** When a unit with more than one advancement
  reaches its experience cap on its own side's turn, Wesnoth asks the human
  owner with the advancement dialog (src/actions/advancement.cpp:205-207:
  `if(... t.is_local_human() && ... (is_current_side || !is_mp)) res =
  advance_unit_dialog(loc_);`). Off-turn (a defender advancing in the enemy's
  turn) it takes option 0 (:232-236), unless the Pick Advance modification
  is on, in which case the player pre-sets it (the core replays that command:
  core_units.rs:284-315, `apply_pickadvance`). 20 reachable types have two or
  more options, e.g. Spearman (Swordsman / Pikeman / Javelineer), Mage (White
  / Red), Elvish Fighter (Captain / Hero), Elvish Archer (Ranger / Marksman),
  Elvish Shaman (Druid / Sorceress), Dark Sorcerer (Lich / Necromancer),
  Ghost (Wraith / Shadow), Skeleton (Revenant / Deathblade), Troll Whelp,
  Drake Burner, Drake Clasher, Saurian Augur, Horseman, Wolf Rider, Goblin
  Spearman, Trapper, Red Mage, Merman Hunter, Merman Warrior, Knight
  (computed from unit_stats.json `advances_to`).
- **What the network gets.** No action. The action space per unit is move and
  attack (core_masks.rs:29 `UNIT_ACTION_TYPES = 2`), plus recruit and end
  turn tokens; nothing names an advancement target. The core decides
  (core_units.rs:100-110): a queued choice (replays only), else a uniform
  draw when `advance_uniform` is set, else option 0.
  - Self-play sets uniform (tools/selfplay_game.py:531 and :578,
    `sim.enable_uniform_advancement()`; actor_worker.py:280 runs games through
    `_play_one_game_safe`).
  - Match games do not (tools/elo_eval_game.py:1148-1156 builds the sim
    without it; no `enable_uniform_advancement` in that file), so every
    on-turn advancement in a match takes option 0: every Spearman becomes a
    Swordsman, every Mage a White Mage, every Dark Sorcerer a Lich.
- **Why it matters.** A human picks the advancement for the position: a Mage
  becomes a White Mage when the army needs healing, a Red Mage against
  undead-free armies needing damage; a Ghost becomes a Shadow to assassinate
  or a Wraith to hold. The network cannot make this choice, and it is
  trained on random choices but evaluated on fixed first choices, so its
  value estimates around levelling units are trained on a different
  distribution from the one it is measured on.
- **Confidence:** verified by reading (the action space, the core's choice,
  the self-play and match setup). That the simple loop's games pass through
  `selfplay_game.py` is inferred from actor_worker.py:280 and :459.
- **Run to confirm:** see "To check by a run" R1.
- Note: this is an output (an order), so it may also appear in the orders
  auditor's report.

### F2. Mixed-terrain chance to be hit away from the unit's hex is not given, and "cap" values are not marked

- **What the player sees.** Hovering any hex shows the selected unit's
  defence there, mixed terrains resolved, with the "maximum/minimum" alias
  lines in the tooltip (reports.cpp:577-627, `unit_defense` at the mouseover
  hex, `underlying_def_terrain` loop :594-610; selected-unit variant
  :628-641).
- **What the network gets.** Its chance to be hit where it stands (column
  110, `defense_at`, core_parity.rs:261-268) and one value per pure terrain
  class (columns 111-126, `class_def`, built per archetype at
  core.rs:382-385 through `terrain::def_pct`, terrain.rs:192-196). For any
  other hex it must combine the classes of the hex's terrain set itself.
  The value a movetype gives as a cap (negative in WML, e.g. mounted
  `forest=-70`, mounted `fungus=-80`, flyers `fungus=-70`; unit_stats.json)
  comes through as its absolute value, with nothing marking it as a cap.
- **Why it matters.** A Cavalryman on forested hills (Hh^Fp) has 30 percent
  defence (forest caps it), not the 40 percent of hills that the usual
  "best of the aliases" reading of its two columns (hills 0.60, forest 0.70)
  would give; a Gryphon Rider on a mushroom-grove hex likewise. Where to park
  cavalry and gryphons is a frequent decision.
- **Partial cover:** the type embedding identifies the movetype, so the rule
  is learnable per type, and the fight columns (known) give the exact numbers
  for the unit's current hex. Not given explicitly.
- **Confidence:** inferred (the absence is verified by reading; that the
  network does not learn it is not established). Overlaps with the terrain
  auditor's part.
- **Run to confirm:** R4.

### F3. Recruit rows carry no traits, while the recruit dialog shows the type's must-have traits

- **What the player sees.** The recruit dialog's preview pane lists the
  type's must-have traits under "Traits" (unit_preview_pane.cpp:368-392:
  every trait with `availability == "musthave"` is listed). In the default
  era: undead on the Undead faction's undead-race recruits (Skeleton,
  Skeleton Archer, Ghost, Ghoul, Walking Corpse; not the human Dark Adept),
  feral on the Vampire Bat (unit_stats.json `traits.musthave`; race
  musthaves at units.rs:134-144).
- **What the network gets.** The recruit row writes weapons, resistances and
  abilities only; columns 79-91 (traits) stay 0 (core_parity.rs:171-178,
  `recruit_extra`; the comment at encoder.py:296-299 states "traits (rolled
  when recruited) ... 0").
- **Why it matters.** Small: undead means unpoisonable / undrainable /
  unplagueable, which decides whether poison or drain is worth anything
  against those recruits. The type
  embedding on the recruit token identifies the type, so the information is
  learnable, and the same unit on the board does carry the bit, so the row
  for "the same unit" differs between its recruit token and its board token.
- **Confidence:** verified by reading.
- **Run to confirm:** R2.

### F4. Weapon accuracy and parry have no column

- **What the player sees.** The sidebar's weapon lines print a weapon's
  accuracy/parry ("accuracy +10%") and the tooltip gives it
  (reports.cpp:957-970). One reachable weapon has one: the Elvish Champion's
  sword, `accuracy=10` (1.18.4 data/core/units/elves/Champion.cfg:21-29;
  unit_stats.json; no reachable weapon has a parry).
- **What the network gets.** The weapon slot is present, damage, strikes,
  ranged, damage type, specials (encoder.py:279-286, core_parity.rs:144-155);
  no accuracy. The fight columns' chance to hit (+3, +7) includes it for the
  six touching hexes (core_parity.rs:196-216, numbers from `fight_line`).
- **Why it matters.** Very little: one level-3 type, identified by its
  embedding; the only gap is judging a distant Champion's threat from the
  row.
- **Confidence:** verified by reading.

### F5. An own unit's "invisible" status is not an explicit column

- **What the player sees.** An "invisible" status icon on the unit
  (reports.cpp:341-344, `u->invisible(displayed_unit_hex)`), which is false
  once the unit is uncovered (unit.cpp:2596-2631) or would be discovered by an
  adjacent enemy.
- **What the network gets.** No status column for it (statuses are only
  poisoned 105 and slowed 106; core_parity.rs:249-250). It can be derived:
  the hide ability bits (ambush, nightstalk, concealment, submerge; columns
  92-104), the hex's terrain set and time of day (hex tokens), has_attacked
  (column 8; an own unit is uncovered at decision time only through its own
  attack this turn, since `uncovered` is cleared at its side's init,
  core_step.rs:266-271, and set by attacking, core_attack.rs:390) and the
  adjacent enemy tokens.
- **Why it matters.** Low: whether a Ranger in forest or a Shadow at night is
  hidden from the enemy now. Derivable, but the network must learn the hide
  filters' terrain/time rules.
- **Confidence:** inferred (derivability reasoned from the code; not tested).

### F6 (possible leak, by the letter). An enemy unit's has_attacked column

- **What the player sees.** Nothing: no sidebar report shows an enemy unit's
  attacks left (reports.cpp: `unit_moves` shows movement only, :671-752; no
  attacks-left report among the default theme's, default.cfg:287-503).
- **What the network gets.** Column 8 `has_attacked` for every visible unit,
  enemies included (core_encode.rs:398, encode.rs:128). The core refreshes it
  only at the owner's init side (core_step.rs:250-254), so during the
  network's turn it says whether that enemy attacked in its last turn.
- **Why it matters.** Small. In a two-player game every enemy attack hits one
  of the player's units, which sees it, so a player who watched the turn
  knows it; the column hands it over as present state. It is a leak only by
  the letter of Principle 6 (information not on screen), not god view.
- **Confidence:** verified by reading (the column and its refresh); "a player
  always saw the attack" is inferred (Silverhead Crossing's side-3 Tentacle,
  scenarios/2p_Silverhead_Crossing.cfg:50-54, could in principle be attacked
  in fog).

---

## 2. Coverage (present and right)

Unit identity and class
- Type: vocab embedding by type name (encoder.py:735; core_encode.rs:392-396).
  The live vocab (round_0583 `pi.pt`, read from the pickle opcodes) has 190
  ids, 0-189, plus 47 Walking Corpse / Soulless / Giant Scorpion variation
  names mapped onto their base id. All 158 reachable base types and all 11
  reachable corpse variations are in it; none overflows (cap 200, overflow
  row 199, encoder.py:58, 1633). Variations share the base WC/Soulless row on
  purpose; their differences (moves, movetype, resistances) are in the row's
  numbers, since `type_rec` is the variation's (units.rs:411, core_parity.rs:245).
- Race: not a column; fixed by type (embedding). Race matters through traits
  (pool) and plague variation, both visible otherwise.
- Level: column 109 / LEVEL_NORM 4 (core_parity.rs:261-268, `unit_level`);
  reachable levels 0-4.
- Alignment: one-hot columns 9-12 (encode.rs:129); the time of day's effect
  at the unit's hex with fearless: column 107 (core_parity.rs:324-329), as
  the sidebar's `unit_alignment` shows it (reports.cpp:377-397). Reachable
  alignments: lawful, chaotic, neutral (no liminal).
- Leader / can recruit: column 7 is_leader (encode.rs:127). Ladder leaders all
  have canrecruit.
- Cost: column 6 (encode.rs:126); upkeep follows from level, is_leader and
  loyal (no loyal unit in the pool: grep of the 21 ladder cfgs for `loyal`
  found none).
- Side: side code embedding (core_encode.rs:397): ours, theirs, scenery
  (petrified or attackless non-player units, core_move.rs:32-35).

Hit points, experience, advancement
- HP and max HP: columns 0-1 (encode.rs:120-121), the unit's own after
  traits and AMLA (units.rs:213-261).
- XP and XP to advance: columns 4-5, the cap scaled by the game's experience
  modifier and the intelligent/dim traits (units.rs:30-32, 232-235); the
  sidebar's `unit_xp` (reports.cpp:530-545) shows the same.
- Advancement targets (unit description; sidebar report not in the default
  theme): fixed per type, so carried by the embedding. AMLA count (`unit_amla`,
  reports.cpp:293-303, also not in the default theme): its effect is in max HP.
  The choice: F1.

Movement, vision, attacks this turn
- Moves left and max: columns 2-3 (encode.rs:122-123), the sidebar's
  `unit_moves` numerator/denominator (reports.cpp:739-751).
- Movement cost per terrain class (the `unit_moves` tooltip,
  reports.cpp:686-736): columns 127-142, slowed table while slowed
  (core_parity.rs:265, 184-194).
- Vision: no column, and none needed: the sidebar shows vision only when it
  differs from movement or there is jamming (reports.cpp:643-659); no
  reachable type differs (the core clears fog with `max_moves`,
  core_shroud.rs:61; the scraped `vision` field is read nowhere: grep for
  `vision` in wesnoth_ai/*.py, tools/wesnoth_sim.py, db.rs, core_fog.rs).
- Attacked this turn: column 8; fight columns zero once it has attacked
  (core_parity.rs:284-287).
- Rested (+2 heal): not shown by any sidebar report (none of
  reports.cpp:161-760 reads `resting_`) and not encoded; no finding under the
  standard (the player only remembers it; the network has memory).

Defence
- Chance to be hit where it stands: column 110 (core_parity.rs:264), the
  sidebar's `unit_defense` (reports.cpp:577-627). Per terrain class: columns
  111-126. Mixed terrain at other hexes: F2.

Traits (13 columns 79-91, core_parity.rs:26-29, 247)
- strong, quick, intelligent, resilient, healthy, dextrous, weak, slow, dim,
  fearless, undead, feral, elemental: every trait a reachable unit can carry
  (race pools, units.rs:122-144; the 4-move leader's quick, units.rs:170-174;
  `traits` blocks of unit_stats.json). loyal, mechanical and aged have no
  column and cannot appear in these games. The row's HP, moves, XP cap and
  damage are the unit's own after traits (units.rs:213-261,
  core_parity.rs:236-243); the resistances are the type's, and no default-era
  trait changes them.

Abilities (13 columns 92-104, core_parity.rs:31-34, 248)
- leadership, skirmisher, regenerate, submerge, cures (which in 1.18 includes
  heals +8: data/core/macros/abilities.cfg:78-86), ambush, heals_4, steadfast,
  feeding, illuminates, nightstalk, concealment, teleport. The reachable set's
  abilities are exactly these 13 (computed from unit_stats.json); heals_8 and
  swamp_lurk occur only on Dunefolk / non-era types. Which unit has which: per
  row. Their effects: leadership received column 108; illumination in the hex
  time of day (core_parity.rs:349-378); steadfast and the rest inside the
  fight numbers.

Statuses
- poisoned 105, slowed 106 (core_parity.rs:249-250); petrified as the scenery
  side code; invisible: F5; unhealable and invulnerable: no default-era source.

Resistances
- Six columns 73-78, (100 - damage percent)/100, the type's table
  (core_parity.rs:157-162, 245). The sidebar's att/def split
  (reports.cpp:483-511) differs only under steadfast, whose bit is present and
  whose effect is inside the fight numbers.

Attacks (3 slots of 20 columns, columns 13-72)
- Present, damage per strike after traits and objects, strikes, ranged,
  damage type one-hot, specials multi-hot (core_parity.rs:144-155, 236-243).
  No reachable type has more than 3 attacks (23 types have 3; computed), so no
  weapon is dropped. Damage norm 40 and strikes norm 6 are the reachable
  maxima (computed). The 10 specials (magical, poison, slow, marksman,
  firststrike, drains, backstab, charge, plague, berserk) are exactly the
  reachable set's; swarm, absorb, stun, deflect occur only on non-era types.
  The sidebar's damage with time of day, leadership and slow
  (reports.cpp:788-845) is the base damage with columns 106-108.
  Accuracy: F4. Specials' active/inactive state: inside the fight lines.

Enemy units in view
- The same row as an own unit (core_encode.rs:390-409, 478-489), which is what
  the sidebar shows for any visible unit (`get_visible_unit` in every report
  generator). Fight columns are zero for enemy rows (known). Enemy moves left:
  shown by the sidebar too (reports.cpp:739). has_attacked: F6. Hidden enemies
  are absent (`obs.view.visible`, core_encode.rs:315).

Recruit list
- One row per own recruit option (core_encode.rs:447-458, 559-580): the
  dialog's type (embedding), cost (column 6), and the preview pane's level,
  race, alignment, HP, XP, MP, movement costs, resistances, abilities and
  attacks (unit_preview_pane.cpp:276-410 against core_parity.rs:171-178,
  270-279, 331-340). Recruitability (greyed rows, unit_recruit.cpp:150-183) is
  the legality mask plus gold. Must-have traits: F3.
- The enemy's recruit list has no rows; see "For other parts" (faction).

Unit list dialog (own units)
- Type, name, moves, HP, level, XP, traits (unit_list.cpp:100-140): all in
  the rows.

Vocabulary limits
- Types 200 (190 used); abilities 13, specials 10, traits 13. An unknown
  ability, special or trait is dropped silently (`name_bits`,
  core_parity.rs:138-142); an unknown type takes the overflow row with a
  logged warning (encoder.py:919-991, game_core.py:967-971). No default-era
  type, ability, special or trait falls outside.

---

## 3. To check by a run

- **R1 (F1).** Play a few match games with `tools/elo_eval_game.py` and a few
  self-play games; from the game records list every advancement event with
  two or more options (`last_advance_events` / the record's advancement
  entries) and its index. Expected: matches all index 0 on the side to move;
  self-play spread uniformly. Also confirm the simple loop's game path:
  print whether `sim._advance_uniform` (or the core's global
  `advance_uniform`) is set in a simple-loop worker game.
- **R2 (F3).** `encode_streams` on a state with an Undead side to move,
  `unit_row_version=2`: recruit_feats columns 79-91 are all 0 for Skeleton,
  Ghost and Vampire Bat; after recruiting a Skeleton its board row has column
  89 (undead) = 1.
- **R3 (F6).** After an enemy attack in its turn, encode the next decision of
  the other side: that enemy's row has column 8 = 1.
- **R4 (F2).** For a Cavalryman on an Hh^Fp hex, compare column 110 (0.70)
  with columns 111-126's hills and forest entries (0.60, 0.70) to show the cap
  is not readable from the row alone.

## 4. Could not determine

- Whether the owner's target games (human ladder play) use the Pick Advance
  modification; it decides whether off-turn advancements are also a player's
  order.
- Whether the network has in fact learned the per-type mixed-terrain rule
  (F2) and the hiding rule (F5); only a probe could say.

## 5. For other parts

- Faction / enemy recruit list: the enemy's recruit options reach the network
  only through `their_faction_probs` / `their_faction_embed`
  (encoder.py:615, 1123-1125, 1261-1267); the global or faction auditor
  should check it against what the status table shows.
- `unit_stats.json` records `vision` 4 for Walking Corpse and Soulless
  variations whose movement is 3, 5 or 6 (the scraper inherits the parent's
  vision, tools/scrape_unit_stats.py:898-901, where Wesnoth defaults vision to
  the variation's own movement). The core does not read it (it uses
  `max_moves`, core_shroud.rs:61), so it is harmless today; anything that
  starts reading `vision` would be wrong for those variations.
- Mixed-terrain defence (F2) overlaps the terrain/hex auditor's part.
