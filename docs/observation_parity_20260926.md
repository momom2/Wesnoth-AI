# What the network observes against what a player sees (2026-09-26)

A read-only crawl compared `obs8`'s observation with what the Wesnoth 1.18.4
interface shows a player, then counted how often each difference occurs in
corpus games. Every gap below changes the network's input, so each fix is a
checkpoint flag, a retrain and an 800-game match; which to run is the
user's decision (BACKLOG "What the network observes against what a player
sees").

**What was compared.** `obs8`'s flags, read from its checkpoint: the
relevant-set basis, the terrain multi-hot and the fog gate are on. Its
encoder has 13 unit features, 8 global features, 14 terrain classes and 3
side codes. The player's view is taken from the 1.18.4 source
(`src/reports.cpp`, `src/display.cpp`, `src/game_stats.cpp`,
`src/units/unit.cpp`).

**The sample.** 70 Ladder-map corpus games (67 with fog), the first games
of a seeded shuffle of the corpus manifest: 26,789 player decisions,
25,667 of them under fog, replayed with the Python applier. The relevant-set
counts use every third decision (8,907). Record:
`training/metrics/observation_parity_20260926/census_total.json`, from
`tools/analysis/observation_parity_census.py` (its docstring gives the
command line that reproduces each slice).

## The parity table

| What a player sees | In the tokens? | Where |
|---|---|---|
| Unit type, level, abilities, advancements | through the type's embedding row only | `unit_type_embed`; 76 of the 190 types share a row in `obs8` (CLAUDE.md, 2026-09-25) |
| Position, HP, XP, moves left (the enemy's too), attack used, leader, cost, alignment | yes | `_unit_features`, `encoder.py:1922`; recruit options code alignment the other way round (BACKLOG) |
| Traits (`reports.cpp:305-333`) | no | quick, resilient, intelligent, dim and slow show through max moves, HP or XP; strong, dextrous, weak, fearless and healthy show nowhere |
| Weapons after traits, time of day, leadership and slow; strikes, damage type, range, specials (`reports.cpp:777-990`) | no | the type row only |
| Resistances (`reports.cpp:484-514`); defense and movement cost per terrain (`:619-760`) | no | the type row with the terrain token |
| Poisoned, slowed (`reports.cpp:335-365`) | no | |
| Petrified | yes | side code 2 |
| Time of day at the unit's hex: time areas, lit terrain, illumination (`reports.cpp:100-112`, `:377-397`) | the board's value only | global features 6 and 7 |
| Terrain of every hex | partly | 14-class multi-hot; mushroom grove is filed as cave and reef as shallow water (`terrain_resolver.py:109-117`); under the relevant set 2,180,053 of 7,958,785 hex observations have a token |
| Villages, castles, keeps | yes | a water village (`Ww^Vm`) loses its village mark under fog |
| Village flags (`display.cpp:339-356`: no enemy flag on a fogged hex) | yes, the same rule | dynamic flags 1 and 2 |
| Fog overlay (which hexes are seen now) | no | except non-own villages, through modifier column 0 |
| Own gold, own villages and the total | yes | global features 2 and 4 |
| Own net income; village gold and support (`game_stats.cpp:170-201`) | no | village gold is a BACKLOG item |
| Enemy row of the status table under fog: the name only (`game_stats.cpp:139-168`) | the same as a player | global feature 5 counts enemy villages on seen hexes |
| Enemy gold, units, upkeep and net income with fog off | no | |
| Enemy faction | yes, also where a player cannot know it | `their_faction_embed` |
| Turn, side, time of day now and next turn | yes | global features 0-1 and 6-7 |
| Turn limit, recall list | not used on the Ladder | no turn limit in the corpus; recall lists are empty in 1v1 |
| What was seen earlier (units that went into fog, the enemy's turn as it was watched) | no | the observation is one frame |

## Gaps, most likely to matter first

1. **A unit's own combat numbers.** The unit features carry no weapons,
   resistances or defenses (known); the new finding is the per-unit part,
   which no type row can hold. Weapons differ from the unit's type on
   161,919 of 455,569 visible-unit observations (strong 146,430, dextrous
   15,868, and the goblins' weak). At 25,209 of 26,789 decisions at least
   one visible unit is affected, and 3,254 of 5,331 attacks have an
   affected attacker or defender: one more damage per strike moves kill
   thresholds (an Elvish Archer at 6-4 instead of 5-4 deals 25% more).
   Fearless (drawn on the Heavy Infantryman and Troll lines) is in 601
   attacks; healthy (dwarves) is invisible too.
   *Fix:* a checkpoint flag with, per unit, 3 weapon slots of presence,
   damage, strikes, range, 6 damage types and about 10 default-era
   specials (57 values), 6 resistances, about 14 trait bits and the 2
   status bits of gap 2: `UNIT_FEAT_DIM` 13 to about 92. Recruit options
   take the type's base values. A minimal form is damage and strikes per
   range (4 values) plus the trait bits. Old checkpoints need
   `unit_feat_proj.weight` zero-padded in `pad_legacy_encoder_state` (it
   pads `global_proj` and `dynamic_flag_proj` today, not this layer), and
   the Rust `encode.rs` rows (`UNIT_STAT_COLS`, `unit_feature_row`) change
   under their bit-exact test; `OBSERVATION_EPOCH` bumps.

2. **Poisoned and slowed.** 4,812 of 26,789 decisions have a visible
   poisoned or slowed unit (own poisoned at 2,094, enemy poisoned at
   2,416); 335 of 5,331 attacks involve one, 47 a slowed one. Slow halves
   damage and doubles movement costs; poison costs 8 HP a turn until cured,
   which decides where the unit must go. *Fix:* 2 columns, inside gap 1's
   block or first under their own flag.

3. **Where unseen enemies can be.** Under fog 533,351 of 1,205,990 board
   hexes are fogged at a decision; within 6 hexes of own units 16.9 hexes a
   decision are fogged, 1,278 of 25,297 of them have a token, and no token
   says a hex is fogged. At 6,975 of 25,667 fog decisions an enemy the side
   saw this turn or last is now hidden (12,422 of 62,323 hidden-enemy
   observations; a lower bound, since a player also watches the enemy's
   turn). The enemy leader is hidden at 13,408 of 25,667.
   *Fix, the overlay:* a "seen" hex flag from `observation.seen`
   (`NUM_HEX_DYNAMIC_FLAGS` 4; the padding covers that layer). *Fix, the
   memory:* a per-side sighting record on `global_info` (id, type, hex, HP,
   turn) kept by the simulator, replay reconstruction, the Rust core, game
   records and `state_digest`; each remembered enemy becomes a token with a
   fourth side code and a "turns since seen" feature, never an actor or a
   target.

4. **Terrain outside the relevant set.** The set keeps 245 of 894 hexes a
   decision. Without a token: 165,080 of 301,955 hexes next to visible
   enemies, 30,648 of 142,848 hexes from which a visible enemy could hit
   an own unit next turn (3.4 a decision), and a neighbour of 33,497 of
   89,656 own units. The basis measured as a net gain (the relevant-set
   twin, +56 +- 12 Elo at one pass), so widening it is a cheap arm, not a
   verdict: own units' neighbours cost 10 more hexes a decision (+4%), both
   sides' neighbours 25 (+10%), the enemies' full reach 130 (+52%).
   *Fix:* a relevant-set version flag (slot indices change) in
   `relevant_hex_positions` and `observe._add_reach`.

5. **Terrain classes that share a row.** Mushroom groves (`Tb^Tf`, 181
   pool hexes on 18 of the 21 maps) and caves (`Uu`, 82 hexes on 5 maps)
   are one class; reef (26 hexes on 7 maps) reads as shallow water. The
   Elvish Archer has 50% defense and 2 MP on mushrooms against 30% and 3 MP
   in caves; the Dwarvish Fighter 40% against 50%; the Merman Hunter 70% on
   reef against 60% in shallow water. 10,536 of 455,569 unit observations
   stand on mushrooms; 335 of 5,331 attacks have a fighter on a merged
   terrain. *Fix:* FUNGUS and REEF classes (`NUM_TERRAINS` 16, two new
   `terrain_embed` rows, which the padding needs a clause for), mapped from
   `Tt` and `Wrt` behind a flag that reaches the `Hex.terrain_mask`
   builder.

6. **The time of day at a unit's hex.** Only 146 of 26,789 decisions have
   a visible unit whose hex reads another time of day than the board, but
   they concentrate: 668 of 1,488 decisions on Elensefar Courtyard (the lit
   cave area), 131 of 1,055 on Tombs of Kesorak, 0 of 990 on Thousand
   Stings Garrison (4 games each). *Fix:* one per-hex value, the hex's
   lawful bonus minus the board's, as a dynamic column.

7. **The enemy's economy with fog off.** The status table shows the
   enemy's gold, units, upkeep and net income when fog is off: 1,122 of
   26,789 sampled decisions, none in eval (fog is always on there).
   *Fix:* 3 global values, zero under fog; batch with the village-gold
   item.

8. **Water villages under fog.** On 4 hexes (Ruphus Isle, Elensefar
   Courtyard) a non-own `Ww^Vm` village reads as plain shallow water,
   because modifier column 0 is written only when the owner is visible
   (`encoder.py:1671-1677`). *Fix:* carry the static village bit, under
   gap 5's flag.

## Information a player does not have

1. **The enemy's faction when the opponent chose Random.** Under fog the
   status table shows an unseen enemy's leader as unknown, and no display
   of the resolved faction was found (not checked in a live game). 96 of
   188 player sides in 100 sampled Ladder games chose Random (raw replays
   carry `chose_random="yes"`). Under fog a side first sees an enemy unit
   on turn 2 in 28 of 134 side-games, turn 3 in 66, turn 4 in 34 and later
   in 6, while the network has the faction from turn 1, so its first
   recruits in imitation are paired with a fact the player did not have.
   Eval is unaffected: the harness assigns the factions, which a lobby
   would show. *Fix:* the reserved "" faction row until the first sighting
   when the opponent chose Random; it needs `chose_random` in the corpus
   records (the extractor drops it), a sighting latch on `global_info` and
   a flag.
2. **Scenery under fog.** `visibility.py:472-478` and `observe.rs` always
   show scenery, where the engine hides every non-own unit on a fogged hex
   (`unit.cpp:2645-2676`); 6,303 of 25,667 fog decisions have a statue on
   a fogged hex. Statues stand on the same hexes in every game of a map, so
   a player who knows the map knows them; recorded, not changed.

**Checked, no leak:** fogged units and hiders (visibility, the relevant set
and the reach contexts use only visible units), the enemy's village count
(gate on in `obs8`), the enemy's gold and recruit list (never encoded), the
dice (seeds live only in the commands; recruit options carry base stats),
next turn's time of day (a public schedule), unit ids (used only to break
sort ties, and none occur).

## Provenance

The census figures (every count of decisions, units, attacks and hexes in
gaps 1 to 4 and 6's overall 146, the scenery count, the first sightings)
are in `census_total.json`. Five groups of figures come from the crawl's
one-off scripts, printed during the crawl and not recorded as files: the
fogged hexes (gap 3), the cost of widening the relevant set (gap 4), the
terrain counts per map (gap 5), the per-map time-of-day counts (gap 6, 4
games per map) and the Random-faction count (100 games).
