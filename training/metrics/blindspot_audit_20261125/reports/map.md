# Blind-spot audit, part: the map (2026-11-25)

Read only. The settings of the live network were read from the pickle bytes of
`C:\Users\momom2\wesnoth_runs\simple_loop\round_0583\pi.pt` (zipfile +
byte search, no unpickling, no torch): `relevant_set_hexes` True,
`relevant_set_version` 2, `terrain_multi_hot` True, `observation_parity` True,
`fog_hides_enemy_villages` True, `unit_row_version` 2, `memory_slots` 64.
Everything below is about that configuration.

Wesnoth C++ was read at the 1.18.4 tag (downloaded to
`scratchpad\audit_20261125\wsrc_map\`: `display.cpp`, `game_display.cpp`,
`reports.cpp`, `menu_events.cpp`, `mouse_events.cpp`). The terrain analysis
was done by a stdlib port of `rust/wesnoth_core/src/terrain.rs` over the
pinned `terrain_db.json` and every unit type's table in `unit_stats.json`
(`scratchpad\audit_20261125\terrain_groups.py`, output `groups.txt`; the map
codes from `map_codes.py`, output `codes.txt`).

## 1. Findings

### F1. Most of the board has no hex token: the network does not see the terrain, fog state or time of day of about 55-60% of the map (serious)

- **What the player sees.** The whole map, every hex: its terrain (also
  under fog; shroud is off), the fog overlay, the village flags, and on
  mouse-over the terrain, position and time-of-day reports
  (`reports.cpp:1545-1621` terrain_info, `:1623-1660` terrain,
  `:1672-1705` position, `:100-112` + `:1334` time of day). It also knows
  the map's name and edges.
- **What the network gets.** Hex tokens only for the "relevant set"
  (`rust/wesnoth_core/src/core_encode.rs:301-305`, filtered by
  `obs.relevant`). The set is (`core_observe.rs:138-159`): village hexes,
  castle and keep hexes, the leader's castle network, hexes holding a
  visible unit, the leader's hex, and every hex an own unit with moves left
  can land on this turn; version 2 adds the six neighbours of every own unit
  and every unseen hex within 6 hexes of an own unit
  (`core_parity.rs:428-456`). Every other hex has no token, so its terrain,
  its "seen now" column (hex dynamic col 3), its time-of-day column (col 4)
  and its existence are absent. There is no map-name or map-size input
  (global row: `encoder.py:198-231`, `:316-323`).
- **Measured size.** Version 1 left out 62% of the board's terrain on
  average, up to 84% on the largest maps (`docs/model_cost_study_20260905.md:157-164`,
  which names the cost: "the ground between the armies beyond one move,
  chokepoints and rivers two turns away, the enemy's approach lanes ...
  What the trunk can no longer compute is the enemy's reach over terrain it
  does not see"). Version 2 adds "about 9% more hex tokens"
  (`docs/parity_memory_design_20260929.md:145-149`). So roughly half or more
  of the board is still absent (inferred from those two numbers; not
  re-measured).
- **Consequences, each a separate player-visible item lost:**
  - terrain beyond one move: a river, a mountain line or a forest belt two
    turns away; the enemy's approach lanes and their defence;
  - the enemy's reach (see F2): it cannot be computed over hexes that have
    no token;
  - fog vs seen-and-empty beyond 6 hexes of own units: a scout with vision
    7-9 (Gryphon Rider, Elvish Scout/Outrider, Cavalryman, Wolf Rider, Bat)
    sees empty hexes past 6 that never get a token, so "I looked there and
    nothing is there" is not in the input (within 6 the absence of a token
    implies "seen", since unseen hexes within 6 are added);
  - the map's edge and off-board holes (`_off^_usr`, 1,514 hexes over 7
    maps) away from own units: a missing token is ambiguous between "off the
    board" and "not relevant" (next to an own unit it is not, since all six
    neighbours are included);
  - per-hex time of day of time areas far from own units (Tombs of Kesorak's
    mausoleums and campfires, Elensefar Courtyard's lit cave areas);
  - the map's identity, from which a player knows all of the above.
- **Why it matters.** Planning a march, choosing where to hold a line, or
  judging what the enemy can hit next turn all read terrain beyond this
  turn's reach. Example: on The Freelands an army two moves from a river
  ford cannot see the ford or the water on either side of it, so it cannot
  prefer arriving on the far bank.
- The network could in principle memorise the 21 maps from the tokens it
  does get (absolute position embeddings + villages and castles, which are
  always present), but that is a learned workaround, not the input.
- No god-view leak in the set: every criterion reads the side's view
  (`view.occupied`, `view.seen`, own units' landable rows computed against
  visible units only, `core_observe.rs:136-137`).
- **Confidence:** verified by reading (the rule and the live flags); the
  fraction is inferred from the two cited measurements.
- **How a run would confirm it:** see "To check by a run", item 1.

### F2. No reach is in the input: neither each own unit's reach nor the enemy's (medium-serious, compounded by F1)

- **What the player sees.**
  - Selecting or hovering an own unit draws its reach: every hex it can
    reach is lit, the rest shaded "unreachable"
    (`game_display.cpp:278-284`, `highlight_reach` `:538-551`); along a
    hovered route it shows defence % and turns to reach, and marks ZoC stops
    and village captures (`game_display.cpp:388-432`, `misc/zoc.png` at
    `:418-419`).
  - Hovering an ENEMY unit shows where it can go with its full movement
    (`mouse_events.cpp:370-388`: `unit_movement_resetter` when the unit is
    not of the side to move, then `pathfind::paths(...)` and
    `highlight_reach`).
  - "Show enemy moves" and "best possible enemy moves" (ignore units)
    overlay the union of every visible enemy's full-move reach, with a count
    of how many enemies reach each hex (`menu_events.cpp:466-503`,
    `game_display.cpp:459-465`).
- **What the network gets.** No column for any of these. The hex dynamic
  row is recruit-rejected (written 0 under parity), village ours, village
  theirs, seen, time of day (`encoder.py:158-173`, `:308-315`;
  `core_encode.rs:370-381`, `core_parity.rs:349-378`). The model takes no
  mask as input (`wesnoth_ai/model.py`: only the key-padding mask, lines
  599-616). Own reach appears only as the legality mask applied to the move
  target after the actor is chosen (`action_sampler.py:26-35`), and
  indirectly as which hexes have tokens (the union over all own units, not
  per unit). Enemy reach is nowhere; the network would have to run a
  shortest-path over hex tokens with the enemy's movement-cost columns
  (unit row 127-142) and ZoC, through attention, and over hexes that are
  often absent (F1).
- **ZoC hexes.** The interface draws no global ZoC overlay; it marks ZoC
  only on a route and folds it into the reach map. The network can derive
  ZoC from visible enemy positions and their level (unit row col 109,
  `encoder.py:335`), so ZoC itself is derivable; what is missing is its
  effect on reach (above).
- **Why it matters.** Whether a hex is safe next turn is read off the enemy
  reach overlay by every competent player. Example: placing a wounded unit
  one hex beyond a Drake's reach versus one hex inside it.
- The owner's rulings forbid a token per possible attack and worked-out
  fight outcomes; a per-hex reach column is neither, so it is not ruled out
  by them (my reading).
- **Confidence:** verified by reading.
- **How a run would confirm it:** item 2 below.

### F3. The oasis heals (and cures) but its hex token looks like plain flat or shallow water (minor; one map)

- **What the player sees.** The oasis overlay `^Do` (terrain id `oasis`,
  `heals=8`, `terrain_db.json` entry `^Do`); the terrain report names it
  (`reports.cpp:1623-1660`, `get_terrain_string`). The sim heals and cures
  on it (`rust/wesnoth_core/src/terrain.rs:206-211` `terrain_heals`, used at
  `core.rs:460`; `docs/wesnoth_rules.md:960`, `:1140`, `:1154`).
- **What the network gets.** The terrain mask of `Gs^Do`, `Rd^Do` is FLAT
  and of `Ww^Do`, `Wwt^Do` is SHALLOW (the overlay's alias is `_bas`, so it
  adds no class; `groups.txt` FLAT and SHALLOW groups). The static village
  bit is not set (it needs a `V` in the overlay, `terrain.rs:370-372`, or
  the capturable modifier). No other column carries healing. So an oasis
  token is identical to a plain one.
- **Where.** Only Ruined Passage among the 21 maps (one hex of each of the
  four codes; `grep -l "\^Do"` over the 21 maps).
- **Why it matters.** A player parks a poisoned or wounded unit on the
  oasis; the network cannot know it heals until it learns that map's
  coordinates.
- **Confidence:** verified by reading.
- **How a run would confirm it:** item 3 below.

## 2. Coverage (checked and present)

Terrain
- Terrain set per hex as a 16-class multi-hot over Wesnoth's movetype
  classes (castle, cave, deep, flat, forest, frozen, hills, impassable,
  mountains, sand, shallow, swamp, unwalkable, village, fungus, reef):
  `terrain.rs:243-262`, `:287-337`; `core_encode.rs:342-350`;
  `encoder.py:826-834`. Every one of the 333 codes on the 21 maps gets a
  non-empty mask (none falls back to the one-class view; `groups.txt` line 2).
- Two codes with the same mask never differ in movement cost or defence for
  any of the 89 distinct unit terrain tables (356 types), except the
  IMPASSABLE+MOUNTAINS group (`Md^Xm`, `Mm^Xm`, `Ms^Xm` vs `Mm^Xo`), where
  only the defence differs and every unit's movement cost is 99 on all four,
  so no unit can stand there: harmless (`groups.txt:417-423`, checked with
  `mvt` = 99 for all tables).
- Mixed terrains: forest, fungus and rubble overlays are worst-of for
  movement and best-of for defence (`-,_bas,-,Ft` / `_bas,+,Ft`); bridges,
  fords and castle mixes are best-of for both; Ha, Ms, Hd, Uh are worst-of
  movement, best-of defence (`groups.txt`). The mask does not say which
  rule; but on these maps each mask has one rule, so with the unit row's
  per-class defence and movement columns (`encoder.py:337-340`) the exact
  values are a function of the input. The value where a unit stands is
  given directly (col 110). The per-hex number a player sees on mouse-over
  (`reports.cpp:1693-1702`) for an empty hex must be combined by the
  network (implicit, not a blind spot).
- Ambush cover (forest overlays): always with the FOREST class
  (`terrain.rs:235` vs `groups.txt`). Concealment (villages): village bit or
  class. Submerge (deep water): DEEP class.
- Terrain light (braziers `^Ecf`, `^Ebn`; lava `Ql`, `Qlf`, `Mv`): differs
  inside the CASTLE, FLAT and UNWALKABLE groups, but is carried by the hex's
  time-of-day column, which includes lit terrain (`core_parity.rs:367-375`).
- Terrain changed by events (Aethermaw turns 4-6, `2p_Aethermaw.cfg:323-392`)
  updates the mask, keep, castle and village arrays
  (`rust/wesnoth_core/src/events.rs:437-451`). The objectives' note
  announcing it (`2p_Aethermaw.cfg:317-319`) is not input, but turn number
  is (global col 0).
- Aethermaw cairns (`^Xo` on castles, "may be recruited to and move from"):
  mask CASTLE+IMPASSABLE(+...) and castle/keep bits.

Villages
- Village hex: static village bit on every village hex under parity
  (`core_encode.rs:351-353`), including the water village `Ww^Vm`, whose
  terrain mask is SHALLOW only.
- Owner, own side: `village_ours` always, fog or not
  (`core_encode.rs:365-368`, `encode.rs:288-289`); the map draws an own flag
  on a fogged village (`display.cpp:339-356`: the flag shows if not fogged
  or not an enemy's). Matches.
- Owner, enemy: `village_theirs` only on a hex seen now
  (`encode.rs:290-291`); the map and terrain_info report hide an enemy flag
  under fog (`display.cpp:346`, `reports.cpp:1595`). Matches; no leak.
- Neutral: both 0. A fogged enemy village reads neutral, as drawn.
- Every village has a token (villages are in the relevant set,
  `core_observe.rs:140`).
- Starting ownership from `[side][village]` (Tombs of Kesorak side 2):
  `wesnoth_ai/rules/scenario_pool.py:681`, `:848-867`.
- Healing 8 on villages: implied by the village bit (all village codes heal
  8, `groups.txt`).

Castles and keeps
- Keep bit and castle bit (static modifiers 1, 2; `core_encode.rs:338-339`),
  plus the CASTLE terrain class; every castle and keep hex has a token
  (`core_observe.rs:140`).
- Which castle hexes connect to the leader's keep: no explicit column; all
  castle tokens are present with positions, so connectivity is derivable;
  the recruit target mask is the exact network (`observe.rs:168-200`).
- Whose keep: not shown by the game either; the leader token's position is
  present.

Fog and shroud
- Fog on: global col 11 (`encoder.py:321`). Seen-now per token: hex dynamic
  col 3, 1 everywhere with fog off (`core_parity.rs:368-369`).
- Terrain shown under fog: the terrain stream does not depend on fog
  (`core_encode.rs:337-350`). Matches the game (fog shows terrain).
- Shroud: refused at scenario build (`scenario_pool.py:663-668`); none of the
  21 maps' sides is played with shroud.
- Enemy units in fog are not tokens (`observe.rs:111-131`); scenery
  (statues) visible in fog is an accepted ruling (`observe.rs:113-116`).

Time of day per hex
- Hex dynamic col 4, the hex's lawful bonus minus the board's, as the
  interface's `get_visible_time_of_day_at` (`reports.cpp:100-112`):
  fogged hex its time area only, seen hex illuminated
  (`core_parity.rs:344-378`). Tombs of Kesorak's time areas and Elensefar's
  are time areas (`2p_Tombs_of_Kesorak.cfg:19-87`). Illumination counts
  hidden units too; the comment cites `tod_manager.cpp:221-262` for the
  interface doing the same (inferred: I did not fetch tod_manager.cpp).

Geometry
- Positions: hex and unit tokens add the same learned `pos_x_embed` +
  `pos_y_embed` (`encoder.py:731-732`, `:1047-1072`), so a unit token and
  the token of its hex share their position term. Recruit tokens sit at the
  leader's position (`core_encode.rs:250-251`, `:411-415`).
- Adjacency and distance: not given (no relative-position bias in the model;
  grep for distance/adjacency/relative/rope in `model.py`,
  `packed_trunk.py`, `transformer_policy.py`, `action_sampler.py` finds
  none). They are a function of the absolute positions, with the even/odd
  column shift learnable because x has one embedding row per value. Inferred
  learnable; not a blind spot by the standard (the player also reads
  adjacency off positions).
- The six-direction order of the fight columns (N, NE, SE, S, SW, NW,
  `encoder.py:341-345`) is `m.nbrs[h*6 + k]` (`core_parity.rs:304-305`),
  the order of `tools/abilities.py:41-53` (even x: N (x,y-1), NE (x+1,y-1),
  SE (x+1,y), S, SW (x-1,y), NW (x-1,y-1)) and of `observe.rs:26-34`. The
  enemy in that direction always has a unit token, so direction d maps to a
  token through the unit's x parity, which its position embedding carries.
- Map size: the largest Ladder map is Arcanclave Citadel, 47x46 playable
  (2,162 hexes); `MAX_MAP_SIZE` 128 (`encoder.py:48`), positions clamp at
  127 (`core_encode.rs:33-36`, `:340-341`): no clamp is reached. No cap on
  hex tokens: a batch past the graphed server's `max_len` 2048 or its
  buckets goes to the eager path, not truncated
  (`wesnoth_ai/graphed_serve.py:75-79`, `:405-420`).

Reports
- `position` report (`reports.cpp:1672-1705`): the hex coordinate (position
  embedding) and the selected unit's defence and cost there (derivable, see
  mixed terrains).
- `terrain` / `terrain_info` reports: terrain name and icons (the class
  set) and village owner unless fogged (above).
- Attack indicator (`game_display.cpp:302-311`): only from the attacker's own
  hex now (no attack from a distance); the fight columns carry each
  direction.

## 3. To check by a run

1. Token share of the board, live settings. Over a few hundred recorded
   decisions of recent rounds, call `GameCore.encode_streams(...,
   relevant_set=True, relevant_set_version=2, observation_parity=True,
   terrain_multi_hot=True, unit_row_version=2)` and report `len(hex_xs) /
   map hexes` (mean, by map, by turn), and the share of hexes the side SEES
   now that have no token. Expect roughly 40-50% of the board present.
2. Enemy reach without tokens. For the same decisions, compute each visible
   enemy's full-move reach (`GameCore.unit_reach(x, y, max_moves)` from the
   enemy's side view) and report the share of reachable hexes that have no
   token in the mover's encoding.
3. Oasis. On Ruined Passage, encode a state and compare the hex rows of the
   four `^Do` hexes with a plain `Gs`, `Rd`, `Ww`, `Wwt` hex: terrain mask,
   modifiers and dynamic columns should be identical apart from position
   and time of day.

## 4. Could not determine

- Whether the network in fact learns the per-map terrain it is not shown
  (memorising the 21 maps). Only a probe could say (e.g. a linear read-out
  of an absent hex's terrain class from the trunk).
- `tod_manager.cpp:221-262` (illumination by hidden units on seen hexes):
  taken from the code's own citation, not re-read.
- That Wesnoth's internal 0-based neighbour rule matches
  `tools/abilities.py:41-53`: not re-read in `map_location`; combat parity
  tests would have caught a mismatch, so inferred correct.

## 5. For other parts

- Actions: a player can choose a route (waypoints, or which of two equal
  paths), which decides what is revealed, which ambusher is hit and which
  village is passed; the network orders only a destination
  (`game_display.cpp:388-432` route marks). Not checked here.
- Game state: `fog_hides_enemy_villages` makes global col 5 count only seen
  enemy villages; whether the status table shows the enemy's true village
  count under fog belongs to that part.
- Game state: the scenario objectives' notes (Aethermaw's turn-4 change,
  Tombs of Kesorak's time shifts) are text a player reads; not in the input.
