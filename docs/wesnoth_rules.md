# Wesnoth 1.18.4 rules — verified source citations

This document collects the Wesnoth engine rules our simulator must
honor, each pinned to a verbatim source quote with file:line. It
exists because re-deriving these from scratch every time costs
hours, and because the rules often live in non-obvious places
(C++, Lua, WML macros, schema defaults all interact).

## How to use this document

**Before researching a Wesnoth rule, read the relevant section here
first.** If the rule is documented, cite it; don't re-derive.

**When you establish a new rule, add an entry here.** Required:

- One-line statement of the rule
- File path + line number where it's enforced: C++ as
  `src/<path>:<line>` at the 1.18.4 tag on GitHub (`wesnoth_src/` holds
  the data tree only), WML and Lua as `wesnoth_src/data/<path>:<line>`.
  Older entries cite C++ as `wesnoth_src/src/...`, from when the tree
  was a full checkout; read those paths at the tag.
- Verbatim quote of the smallest snippet that proves it (with code fence)
- A "why this is non-obvious" note when the answer wasn't where you'd
  expect (e.g. lives in Lua not C++, or contradicts a stale changelog)

**When you discover a previous entry is wrong, EDIT it.** Don't add
a contradicting entry; that's how truth-drift starts. Note the date
and the corrected source citation.

**Quote source verbatim, not paraphrased.** Future-you needs to grep
the quote to find the file again. Paraphrases drift; quotes don't.

---

## Table of contents

- [Movement](#movement)
- [Rounding rules](#rounding-rules)
- [Combat](#combat)
- [Units, traits, leaders](#units-traits-leaders)
- [Villages](#villages)
- [Recruit and recall](#recruit-and-recall)
- [Replay structure](#replay-structure)
- [Scenario events](#scenario-events)
- [Common pitfalls](#common-pitfalls)
- [File map (where to look first)](#file-map-where-to-look-first)
- [Search recipes](#search-recipes)
- [Verification protocol](#verification-protocol)
- [Combat-outcome prediction (the in-game damage calculator)](#combat-outcome-prediction-the-in-game-damage-calculator)
- [Turn-1 init_side: WRONG, superseded -- see the init_side entry above](#turn-1-init_side-wrong-superseded----see-the-init_side-entry-above)
- [\[capture_village\] = set_owner per matched hex](#capture_village--set_owner-per-matched-hex)
- [\[modify_unit\] moves= writes CURRENT MP, not max](#modify_unit-moves-writes-current-mp-not-max)
- [Enemy side statistics under fog or shroud (added 2026-09-08)](#enemy-side-statistics-under-fog-or-shroud-added-2026-09-08)
- [The preprocessor's macro grammar: `:` in names, `#arg` defaults (added 2026-09-22)](#the-preprocessors-macro-grammar--in-names-arg-defaults-added-2026-09-22)
- [`random_start_time` has three forms, not two (added 2026-09-22)](#random_start_time-has-three-forms-not-two-added-2026-09-22)
- [Preprocessor conditionals, and what a multiplayer game defines (added 2026-09-23)](#preprocessor-conditionals-and-what-a-multiplayer-game-defines-added-2026-09-23)
- [Vision and fog: what a side sees, and when it is recomputed (added 2026-09-24)](#vision-and-fog-what-a-side-sees-and-when-it-is-recomputed-added-2026-09-24)

---

## Movement

### Terrain resolver: scrape terrain.cfg, walk the alias graph

Our runtime resolver (`tools/terrain_resolver.py`, fed by
`tools/scrape_terrain.py` → `terrain_db.json`) IS the
implementation of the rules in this section. Use it for any
movement / defense lookup. Don't add hand-rolled overlay tables
to `_move_cost_at_hex` / `_terrain_keys_at` — those existed pre-
2026-05-02 and silently mispriced ~75% of overlay codes.

**Scraper gotchas** (lessons from writing `scrape_terrain.py`):

  - `_bas` is a LITERAL token in alias values, not a translation
    marker. Don't strip leading `_` from structural fields like
    `aliasof=`, `mvt_alias=`, `def_alias=`, `default_base=`. Only
    strip from user-facing `name=` / `description=` / etc.
  - Lines like `string=Gt       # wmllint: ignore` carry trailing
    `# ...` comments. Strip these before parsing the value, or
    `Gt` becomes `Gt       # wmllint: ignore` and abstract terrain
    lookups break (which silently fails every alias resolution
    since most overlays alias to abstract types like Gt, Ut, Vt).
  - terrain.cfg has 280+ entries. The abstract types (Gt, Vt, Rt,
    Ut, Wst, Wdt, Mt, Ht, Ft, At, St, Xt, ...) have `hidden=yes`
    and serve as the terminal-id hooks for the unit's
    movement_costs table.

### `mvt_alias` resolution: default MIN, `MINUS` marker means MAX

When a terrain has multiple underlying types (e.g. `^Vhs` =
swamp village = `aliasof=_bas, Vt`), Wesnoth picks the BEST (lowest)
movement cost across the alias list. A leading `-` marker (encoded
as `t_translation::MINUS`) flips the rule to pick the WORST (highest)
cost.

`wesnoth_src/src/movetype.cpp:336-368`:
```cpp
// This is an alias; select the best of all underlying terrains.
bool prefer_high = params_.high_is_good;
int result = params_.default_value;
if ( underlying.front() == t_translation::MINUS )
    // Use the other value as the initial value.
    result =  result == params_.max_value ? params_.min_value :
                                            params_.max_value;

// Loop through all underlying terrains.
t_translation::ter_list::const_iterator i;
for ( i = underlying.begin(); i != underlying.end(); ++i )
{
    if ( *i == t_translation::PLUS ) {
        // Prefer what is good.
        prefer_high = params_.high_is_good;
    }
    else if ( *i == t_translation::MINUS ) {
        // Prefer what is bad.
        prefer_high = !params_.high_is_good;
    }
    else {
        // Test the underlying terrain's value against the best so far.
        const int num = value(*i, fallback, recurse_count + 1);

        if ( ( prefer_high  &&  num > result)  ||
             (!prefer_high  &&  num < result) )
            result = num;
    }
}
```

For movement, `high_is_good=false` (low cost is good). So default
prefers LOW (MIN). A `MINUS` marker flips `prefer_high` to true →
MAX. Examples from `data/core/terrain.cfg`:

- `^Fp` (forest): `mvt_alias=-,_bas,Ft` → MAX (forest is the harder cost)
- `^Vhs` (village): `aliasof=_bas, Vt` (no `mvt_alias`, no marker) → MIN
- `Wwf` (Ford): `aliasof=Gt, Wst` → MIN of (grass, shallow_water)
- `^Bw/` (wooden bridge): `aliasof=_bas, Gt` → MIN (bridge over water → flat)
- `^Xo` (impassable wall): aliases drop base entirely → impassable

**Why non-obvious**: our pre-2026-04-30 sim used MAX everywhere,
which silently overpriced every village hex. The Lua/WML code
doesn't make the rule explicit; you have to read `movetype.cpp` to
see the marker semantics.

### Single-turn move limit: `total_path_cost ≤ current_moves`

`wesnoth_src/src/actions/move.cpp:756-769` (`unit_mover::plot_turn`):
```cpp
remaining_moves -= move_it_->movement_cost(map[*end]);
if ( remaining_moves < 0 ) {
        break;
}

// We can enter this hex. Record the cost.
moves_left_.push_back(remaining_moves);
```

A unit's recorded path stops where remaining MP would go negative.
**No "free first hex" rule** — if the unit has 4 MP and the next
hex costs 5, the move stops, the unit doesn't enter.

### Path-occupancy rule: friendly units passable, enemies block

`wesnoth_src/src/pathfind/pathfind.cpp:777-786`:
```cpp
if (other_unit)
{
    if (teams_[unit_.side() - 1].is_enemy(other_unit->side()))
        return getNoPathValue();
    else
        // This value will be used with the defense_subcost (see below)
        // The 1 here means: consider occupied hex as a -1% defense
        // (less important than 10% defense because friends may move)
        other_unit_subcost = 1;
}
```

A unit can move THROUGH a friendly unit's hex (with a tiny defense-
preference subcost), but cannot move through an enemy's hex.
The unit cannot END its move on any occupied hex (friend or foe).

### Default route selection: cost model, not explicit tie-breaks

When a player clicks a destination, the route comes from
`mouse_handler::get_route` (`wesnoth_src/src/mouse_events.cpp`):
```cpp
const pathfind::shortest_path_calculator calc(*un, team, board.teams(), board.map());
...
route = pathfind::a_star_search(
    un->get_location(), go_to, 10000.0, calc, board.map().w(), board.map().h(), &allowed_teleports);
```
The viewing team is the MOVING player's team; the `ignore_unit /
ignore_defense / see_all` flags default to false. All path
preference lives in `shortest_path_calculator::cost`
(`wesnoth_src/src/pathfind/pathfind.cpp:742-820`, 1.18.4):

1. Shrouded hex (`viewing_team_.shrouded(loc)`, line 748) →
   `getNoPathValue()`. **Shroud blocks pathing; FOG does not** —
   fogged hexes are pathable, and units hidden in them are simply
   not seen (next point).
2. Occupancy via `resources::gameboard->get_visible_unit(loc,
   viewing_team_, see_all_)` — **only units visible to the moving
   player block or penalize**. Visible enemy → `getNoPathValue()`
   (line 780); visible friendly → `other_unit_subcost = 1`
   (line 785, pass-through with a tiny avoidance preference).
3. ZoC (line 797-806): entering a ZoC hex costs ALL remaining MP
   (`move_cost += remaining_movement`) instead of the terrain
   cost — UNLESS entering it consumes exactly the remaining MP
   anyway (`remaining_movement != terrain_cost` guard), or the
   unit is a skirmisher. So equal-MP routes AVOID ZoC hexes, and
   a ZoC hex is effectively terminal for the turn.
   `enemy_zoc` (pathfind.cpp:134-140) also uses
   `get_visible_unit` — **only visible enemies exert ZoC for
   planning** — and checks `u->emits_zoc()` (petrified/lvl-0
   emit none).
4. Tie-breaks (line 815-820): `return move_cost +
   (defense_subcost + other_unit_subcost) / 10000.0` where
   `defense_subcost = unit_.defense_modifier(terrain)` (chance
   to be hit, 0-100). Among equal-MP routes Wesnoth prefers
   (a) better defense terrain along the way, (b) hexes without
   allied units, at 1e-4 weight so they never outweigh 1 MP.

**Residual exact ties are unspecified.** `a_star_search`'s node
comparison is `t < o.t` only (`astarsearch.cpp:113`), neighbors
expand via `get_adjacent_tiles` iterated in reverse
(`astarsearch.cpp:187-194`); ties fall to heap order. The route is
computed CLIENT-SIDE and the chosen path is recorded in the replay
verbatim — playback validates path legality (`plot_turn`), not
that it matches any canonical choice. So a reimplementation is
Wesnoth-faithful iff it minimizes the same cost function
(MP + defense/ally subcosts); residual-tie choices are free.

**Why non-obvious:** "the default path" sounds like a tie-break
rule but is really a cost model; and the visibility handling
(fog ≠ shroud, hidden units ignored, ZoC from visible enemies
only) means route planning is a pure function of the moving
player's OBSERVABLE state — exactly our legality-mask contract.

### ZoC and incapacitation

**Rule (corrected 2026-09-26):** a unit holds a zone of control when its
level is 1 or more and it is not petrified, whatever its attacks; it
holds it against a mover's side when it is that side's enemy and
visible to it.

`src/units/unit.hpp:1352-1356` (1.18.4):
```cpp
	/** Tests whether the unit has a zone-of-control, considering @ref incapacitated. */
	bool emits_zoc() const
	{
		return emit_zoc_  && !incapacitated();
	}
```
(this entry used to quote the function as `get_emit_zoc`, which is the
raw flag's getter just below it). `emit_zoc_` is the unit type's
`zoc=`, `src/units/types.cpp:215`:
```cpp
	zoc_ = get_cfg()["zoc"].to_bool(level_ > 0);
```
copied onto the unit in `unit::advance_to` (`src/units/unit.cpp:991`,
`emit_zoc_ = new_type.has_zoc();`); a `[unit] zoc=` (`:507-508`) or an
`[effect] apply_to=zoc` (`:2270-2274`) overrides it, and nothing in the
default era, the pool scenarios or the corpus scenarios sets either
(grep of wesnoth_src/data, 2026-09-26). The mover's side enters the
test in `enemy_zoc` (`src/pathfind/pathfind.cpp:134-146`):
```cpp
		const unit *u = resources::gameboard->get_visible_unit(adj, viewing_team, see_all);
		if ( u  &&  current_team.is_enemy(u->side())  &&  u->emits_zoc() )
			return true;
```

Petrified (`STATE_PETRIFIED`) → `incapacitated()` is true →
emits no ZoC. Also has `attacks_left() = 0` and `movement_left() = 0`
(unit.hpp:998 and 1299).

Sim: `tools/pathfind_sim.emits_zoc` is the one predicate; the planner
(`ReachContext.for_side`), the walker (`walk_move_path`), the legality
mask's reach context (`action_sampler`) and the observation's unit flags
(`wesnoth_ai/observe.py`) ask it. The Rust observation kernel
(`observe.rs`, from phase 16) reads those flags for every visible enemy,
scenery included, and the Rust core's walk (`core_move.rs`) applies the
same rule. Every non-own side counts as an enemy.

**Why non-obvious:** our "scenery" class (`visibility.is_scenery_unit`:
petrified, or attackless on a side past 2) reads like "inert", and the
planner skipped scenery for ZoC while the walker did not, so on a board
with an attackless level-1 side-3 unit the mask offered moves that the
walk cut short at the unit's zone (to 2026-09-26; the Rust observation
kernel likewise to phase 15). No pool or corpus board has such a unit:
every scenery unit there is a petrified statue.

### controller=null sides get no turn, ever

`wesnoth_src/src/playsingle_controller.cpp:198-210`
(`skip_empty_sides`):
```cpp
for (; side_num != max; ++side_num) {
    int side_num_mod = modulo(side_num, sides, 1);
    if(!gamestate().board_.get_team(side_num_mod).is_empty()) {
        return { side_num_mod, side_num_mod != side_num };
    }
}
```
The turn loop advances past every `team::is_empty()` side
(controller=null parses to an empty team): no `[init_side]`, no
actions, no events for that side -- its units simply exist on the
map (occupying hexes, exerting ZoC, defending when attacked).

**Why non-obvious:** a null side can own ARMED, non-petrified units
-- Silverhead Crossing's side-3 "Shapeshifter" is a live Tentacle
of the Deep (the scenario even carries a belt-and-braces
`[event] name="side 3 turn" -> [end_turn]`). Treating "armed
neutral" as "gets a turn" (our mini-map machinery's rule) emitted
sim turns + exported [init_side] side_number=3 commands the engine
never expects; playback's turn counter drifted, dragging ToD with
it -- surfacing as damage-verification overrides once combat
crossed a lawful-bonus phase boundary (the 2026-07-19 19-vs-12
Wose strike: afternoon vs night). The controller attribute, not
the armament, decides who acts: minis' tentacle sides are
controller=ai and legitimately fight.

**The converse holds too: a controller!=null side keeps its turn
with ZERO living units.** `is_empty()` is the ONLY skip in the
turn loop above -- there is no "no units left" branch. An ai side
whose units all died still gets `[init_side]` + (no-op AI) +
`[end_turn]` recorded every round to game end. Empirically
confirmed 2026-07-21: exported tentacle-mini replays that dropped
side 3 after extinction fail playback at exactly the first omitted
turn ("received a synced [command] from side 1. Expacted was a
[command] from side 3" + "found corrupt movement in replay").
Both directions of the same census rule: WHO ACTS = declared
controller attribute, never the live unit roster.

### Side order within a turn

The next side to play is the next side number whose team is not
empty; wrapping past the last side starts a new turn. At each side's
turn start, `src/play_controller.cpp:465` (`do_init_side`):
```cpp
gamestate_->next_player_number_ = gamestate_->player_number_ + 1;
```
and at its end, `src/playsingle_controller.cpp:251` and `:259`
(`finish_side_turn`):
```cpp
int next_player_number_temp = gamestate_->next_player_number_;
auto [next_player_number, new_turn]  = skip_empty_sides(next_player_number_temp);
```
`skip_empty_sides` (quoted in the entry above) walks forward modulo
the number of teams to the first team that is not `is_empty()`, and
reports a new turn when it wraps. `src/team.hpp:247`:
```cpp
bool is_empty() const { return info_.controller == side_controller::type::none; }
```
and `src/side_controller.hpp:21` names that controller in WML:
```cpp
static constexpr const char* const none = "null";
```
On a 2p map with a third side, each turn is therefore sides 1, 2,
then 3 when its controller is not null, and side 1 opens the next
turn. The simulator plays sides 1 and 2 by policy and an acting third
side by `tools/neutral_ai.py`, inside side 2's end_turn step
(`WesnothSim._next_player_side`).

**Why non-obvious:** who acts is a property of each side's
controller, and our states do not carry controllers. A scenario build
records the census from the scenario's [side] blocks
(`_null_controller_sides`, `_neutral_actor_sides`); an extracted replay
record keeps no controller but keeps a SideInfo for every declared
side, statues included. Counting SideInfos therefore gave a replayed
Caves of the Basilisk position continued in the simulator a side-3
turn every round: the init_side and end_turn of a side the engine
never plays (its units' turn-start healing and end-of-turn refresh,
its `side 3 turn` events), with side 3 as the side to move after
side 2. Replay reconstruction reads the census from the record's
init_side commands (`replay_dataset.extra_side_turns`). Over the
imitation corpus (2026-09-25), 7,118 of 17,019 games declare a third
side; its init_side appears in every game of the six tentacle minis
(`controller=ai`: 2p_mini, 2p_mini_edited,
Modified_Tiny_Close_Relation, enclave_micro_isar,
enclave_mini_fallenstar_1v1, enclave_small_fallenstar_1v1) and in no
game of Caves of the Basilisk, Silverhead Crossing, Sullas Ruins,
Thousand Stings Garrison or WL_Troll_Toll (`controller=null`).

### Hide cover is a terrain-CODE filter, not a defense class

`wesnoth_src/data/core/macros/abilities.cfg:280-382` — each `[hides]`
carries a `[filter_location]` that matches the hex's terrain CODE:

```
#define ABILITY_AMBUSH
    [hides]
        id=ambush
        # ^Qhh* and ^Qhu* are the bluff and glutch terrains ...
        [filter]
            [filter_location]
                terrain=*^F*,*^Qhhf,*^Qhuf
            [/filter_location]
        [/filter]
    [/hides]
```

and likewise `terrain=*^V*` (concealment), `terrain=Wo*^*` (submerge),
`time_of_day=chaotic` (nightstalk, no terrain condition at all).

**Why non-obvious.** The natural reading of "hides in forest" is "hexes
a unit defends on as forest", and this project implemented it that way
for months: `visibility._hide_cover_active` asked
`replay_dataset._terrain_keys_at`, whose `_OVERLAY_DEFENSE_KEYS` is a
hand-rolled allow-list. The engine never consults the defense class.
Any overlay missing from the list fell through to plain flat, so the
ability was SILENTLY INACTIVE there. Two denominators, both exact,
and every published version of this figure is one of them:

| set | no ambush cover | no concealment |
|---|---|---|
| Wesnoth's 66 core multiplayer maps, border included | 1,016 of 5,374 = 18.9% | 383 of 1,511 = 25.3% |
| the 21 Ladder maps, playable hexes only | 478 of 1,572 = 30.4% | 93 of 336 = 27.7% |

(`Gs^Fms`, `Hh^Fms`, `Gs^Ftd`, `Re^Fms` for ambush; `Gg^Ve`,
`Gs^Vht`, `Aa^Vha` for concealment.) The Ladder pool is the set
that decides agent behaviour; the 66-map set is what the first
write-up of this fix counted, which is why "19% and 25%" and
"30.4% and 27.7%" are both correct and neither was reproducible
until now. The affected units are the
ones that want that terrain: Woses and Elvish Rangers/Avengers
(ambush), the Fugitive (concealment), and the Undead line (submerge,
which was missing tropical deep water; see below).

The glob is matched by `terrain_resolver.hides_cover`, and it is the
single source for both the Python predicate and the cover flags
`game_core.map_static` bakes for the Rust-owned state
(tests/test_hide_cover.py).

Cover moves in BOTH directions. Counted over playable hexes (the
1-hex border stripped) with `parse_terrain_codes`, old rule against
new:

| ability | Ladder pool (21 maps, 20,726 hexes) | all tracked maps (114, 96,133 hexes) |
|---|---|---|
| ambush | +478 / -0 | +1,521 / -0 |
| concealment | +93 / **-302** | +553 / **-640** |
| submerge | +153 / -0 | +332 / -0 |

Every lost hex, on every map, is farmland. The two corrections behind
the table:

- **`^Gvs` is Farmland, not a village** (`terrain.cfg`:399-405,
  `aliasof=_bas`, i.e. an embellishment that changes nothing). The
  table listed it as `["village"]`, so concealment applied on open
  farmland: 302 playable hexes of the Ladder pool, 640 of all tracked
  maps. `*^V*` does not match it. The blast radius stops at cover --
  `_terrain_keys_at`'s other callers are `wesnoth_sim._move_cost_at_hex`
  and `pathfind_sim.defense_pct_at`, both only as a fallback when the
  hex has no terrain code (pathfind also when `def_pct` raises), and
  `replay_dataset._resolve_combat`, which passes an explicit
  `defense_pct` from `terrain_resolver.def_pct` alongside it. The
  encoder reads its village bit from `_parse_hex_code`, which `Gvs`
  fails.
- **`Wot` is deep_water_tropical** (`terrain.cfg`:44-47) and was
  missing from the base list, so submerge did not apply there: 153
  playable hexes of the Ladder pool, all of them on Ruphus Isle, and
  332 of all tracked maps. `Wo*^*` matches any `Wo` base.

**`burrow` and `swamp_lurk` are the two `[hides]` we do not model.**
`abilities.cfg`:301-315 gives it `terrain=*^F*,*^Qhhf,*^Qhuf,D*^*`
(forest or SAND) plus a resting condition, and `hides_cover` returns
False for it. Latent, not live: the pinned `unit_stats.json` scrape
DROPPED burrow -- "Horned Scarab" is among its 356 units with
`abilities: []` (its scrape's ABILITY_MACROS had no BURROW entry) --
and `swamp_lurk` (`Crocodile.cfg`, `terrain=S*^*`) rides the Swamp
Lizard, which the scrape does carry; both units appear only in
`2p_Isle_of_Mists.lua`, in neither pool. Adding a burrowing unit
needs the glob AND the "has not moved this turn" state, which the sim
does not track. Nightstalk's `time_of_day=chaotic` is evaluated on
the ILLUMINATED time of day (abilities.cpp:447-450 runs the [hides]
filter with use_flat_tod=false; filter.cpp:268-273 then calls
get_illuminated_time_of_day), so an [illuminates] unit on or next to
the hex lifts the cover: `replay_dataset.illuminated_lawful_bonus_at`
is the one reading the hide predicate and combat share.

**Verified against the engine (2026-09-20).** `tools/hidden_units_oracle.py`
builds scripted positions on a 14x14 grass board in real Wesnoth (the
`ai_oracle` test scenario sets terrain, units, fog and time of day at
prestart from a setup file), asks the engine which units side 1 sees
(`[filter_vision]`, src/units/filter.cpp: fogged OR an enemy hidden by
its hides ability), orders a move through the AI stage and reads the
route the engine chose, the landing hex, the movement left and the
visible set afterwards; then walks the same route through
`replay_dataset._apply_command` and compares. 54 of 54 cases agree
(record `training/metrics/fidelity/hidden_units_oracle_20260920.json`):
ambush on eleven forest codes and none on six non-forest codes
(farmland `^Gvs`, a village, embellishments, plain hills); concealment
on nine village codes and none on farmland, forest or plain; submerge
on `Wo`, `Wot`, `Wog` and a bridge over deep water (`Wo^Bsb|`) and none
on shallow water, fords, reefs or swamp; nightstalk at first and second
watch and not at dawn, morning or dusk, lifted by an ally's
`illuminates` on the adjacent hex and not from two hexes away; the
ambush stop on the first hex adjacent to the hider with movement
zeroed and the hider revealed; hides without fog; the adjacency reveal;
a visible enemy's zone of control routing a non-skirmisher around; two
moves in one turn. This is the ground truth the 17,039-replay sweep
could not give (it passes under the old rule too).

### An ability or weapon special has TWO keys: its tag and its `id=`

The engine uses both, for different jobs, and confusing them is how a
granted ability can exist and do nothing.

**Numbers resolve by TAG.** `src/actions/attack.cpp:173`:

```cpp
cth = weapon->composite_value(weapon->get_specials_and_abilities("chance_to_hit"), cth);
```

and `src/units/abilities.cpp:933-941` gathers them with
`specials_.child_range(special)`, a tag lookup. So `[chance_to_hit]
id=magical` and `[chance_to_hit] id=marksman` are collected by the SAME
query and combined through their `value=` / `cumulative=`. Hidden
status is the same shape, `src/units/unit.cpp:2620-2622`:

```cpp
// Test hidden status
static const std::string hides("hides");
bool is_inv = get_ability_bool(hides, loc);
```

so every hide ability answers to the tag `hides`, and its
`[filter_location]` decides which one actually fires.

**Identity resolves by `id=`.** Dedup on `apply_to=new_ability`
(`src/units/unit.cpp:2275-2285` appends only children whose id the unit
lacks, via `has_ability_by_id`, `:1414-1423`), removal on
`apply_to=remove_ability` (`:2286-2294` → `remove_ability_by_id`,
`:1425-1436`, which erases every matching child whatever its tag), and
named lookup (`has_special` matches tag OR id,
`src/units/abilities.cpp:807-814`).

**Why non-obvious, and what it means for us.** Our model flattens
"which rule fires" into the NAME: `unit_stats.json` scrapes abilities
and specials as ids, combat asks `"magical" in weapon.specials`, and
the fog gate asks `"submerge" in unit.abilities`. So an `[effect]`'s
children must be read by `id=` — the engine's ids are unique where its
tags are not. `tools/scenario_events._effect_member_ids` does that,
falling back to the tag for a block with no `id=` (which the engine
keeps working by tag but makes invisible to every id-keyed operation:
a blank attribute never compares equal to a non-empty string,
`src/config_attribute_value.cpp:422-427`).

Until 2026-09-13 we read the tag, so 2p Silverhead Crossing's
`prestart` `[object]` gave its Tentacle an ability called `hides` and a
special called `chance_to_hit`, neither of which anything consumes. The
unit was visible where Wesnoth submerges it and its counter-attack lost
the flat 70% `magical` SETS (`cumulative=no` replaces the
defender's terrain defence in BOTH directions; only `marksman` is a
floor), on a Ladder map and 351 corpus games
(tests/test_effect_ids.py).

**`[set_specials] mode=` defaults to REPLACE.**
`src/units/attack_type.cpp:416-429`:

```cpp
if(mode != "append") {
    specials_.clear();
}
```

with a deprecation warning when `mode=` is absent. Anything that is not
exactly `"append"` wipes the weapon's existing specials first, and
append does no de-duplication at all. We model append. No scenario in
either pool uses `[set_specials]` except Hornshark Island, whose bow
has no base specials, so the two agree there; the gap is recorded in
BACKLOG.md.

### Hidden-unit visibility: live adjacency + persistent UNCOVERED

`wesnoth_src/src/units/unit.cpp:2596-2637` (`unit::invisible`):
```cpp
if(get_state(STATE_UNCOVERED)) {
    return false;
}
...
bool is_inv = get_ability_bool(hides, loc);
if(is_inv){
    is_inv = (resources::gameboard ? !resources::gameboard->would_be_discovered(loc, side_,see_all) : true);
}
```
`wesnoth_src/src/display_context.cpp:29-49` (`would_be_discovered`):
a hider is discovered iff an ADJACENT tile holds an enemy (of the
hider) that is not incapacitated and is itself visible.

So a `hides`-ability unit is invisible even on a fog-visible hex,
EXCEPT:
1. **Live adjacency** — while an enemy unit stands directly
   adjacent. This does NOT persist: move the adjacent unit away and
   the hider re-hides.
2. **STATE_UNCOVERED** — persistent until the hider's own side's
   turn start (`unit::new_turn`, unit.cpp:1277), from turn 2 on:
   `unit::new_turn` runs from `game_board::new_turn`, which
   `do_init_side` calls only `if(turn() > 1)` (play_controller.cpp:
   488-490), so a hider revealed on turn 1 stays revealed until its
   side's turn-2 start. Set in exactly two
   places: `reveal_ambusher` when an ambush/blocked-move reveal
   fires (move.cpp:870), and the hider itself attacking
   (attack.cpp:1378, unconditional on the attacker). Until 2026-09-24
   the simulator reset it outside the command applier, at every
   init_side including turn 1, and replay reconstruction never reset
   it; `_apply_command`'s init_side does it now for both.

**Why non-obvious:** "I saw it earlier this turn so it stays
visible" is NOT engine behavior for mere adjacency — only
ambush-trigger/blocked reveals and self-attacks persist. Also note
ambush *eligibility* during a move uses the PRE-MOVE invisibility
cache (unit.cpp:2613-2618; cleared in post_move via
`clear_status_caches`), so a mover is ambushed by a hider it was
about to discover by stepping adjacent.

### Move resolution: blocked vs ambush vs ZoC vs capture

`wesnoth_src/src/actions/move.cpp`:
- `plot_turn` (726-780): the turn's route prefix stops at MP
  exhaustion or after entering a (visible-)enemy ZoC hex; the
  expected end BACKTRACKS off hexes holding a visible unit
  (776-780). No village stop — passing THROUGH a village mid-path
  neither stops the unit nor captures it.
- `check_for_ambushers` (422-440): entering a hex ADJACENT to a
  hidden (`invisible(loc)`) enemy → `ambushed_`, stop AT that hex.
- `check_for_obstructing_unit` (449-...): an (invisible) unit ON
  the next path hex → `blocked()`, stop BEFORE it.
- `post_move` (1019-1055): `if (ambushed_ || blocked())
  reveal_ambushers()`; MP zeroed ONLY for
  `ambushed_ || final_loc == zoc_stop_` — a BLOCKED unit keeps its
  remaining MP; village capture on the FINAL hex zeroes MP
  (1046-1053).

**Why non-obvious:** blocked ≠ ambush. Both reveal the hidden unit
(STATE_UNCOVERED), but ambush zeroes MP while blocked leaves the
remainder spendable. And hidden units exert NO ZoC while hidden
(`plot_turn` checks `enemy_zoc` with the current team's visibility,
765-768) — they stop movers via ambush/block, never via ZoC.

### Replay [move] playback: skip_sighted

`wesnoth_src/src/synced_commands.cpp:305-314` (move handler):
```cpp
if(child["skip_sighted"] == "all") { skip_sighted = true; }
else if(child["skip_sighted"] == "only_ally") { skip_ally_sighted = true; }
```
Playback re-checks sighted-interrupts UNLESS the [move] WML carries
`skip_sighted="all"`. Blocked/ambush truncation applies regardless.
Our exports always emit `skip_sighted="all"` (the sim doesn't model
sighting interrupts, so exports must not re-check them —
sim_to_replay._wml_for_command).

### Replay [move] checkup: final_hex and stopped_early

`src/actions/move.cpp:1178-1182` (1.18.4, `move_unit_internal`):
```cpp
config cn {
    "stopped_early", mover.stopped_early(),
    "final_hex_x", mover.final_hex().wml_x(),
    "final_hex_y", mover.final_hex().wml_y(),
};
```
and `move.cpp:221`:
```cpp
bool stopped_early() const  { return expected_end_ != real_end_; }
```
`expected_end_` is the end of this turn's part of the ordered route
(`plot_turn`, 726-780: movement points, entering a visible enemy's zone
of control, the backtrack off a hex holding a visible unit);
`real_end_` is where the unit stopped. A [move] whose `final_hex`
differs from the last hex of its `x=`/`y=` path therefore comes in two
kinds: `stopped_early=no`, an order longer than this turn (the unit went
as far as the order planned for the turn), and `stopped_early=yes`, a
unit cut short of that (a sighted enemy, an ambush, a blocked hex).
On 150 corpus games (2026-09-26), 305 of the 1,753 stopped moves were the
first kind; `replay_dataset.move_label_hex` labels the second kind with
the clicked hex (the path's last hex, kept as the move's order by
`replay_extract`).

**Why non-obvious:** the path is the player's order, not the unit's
route; the unit's route ends at `final_hex`.

### Surrender and control changes in a server replay

The server records each change of a side's controller as a chat line
from "server" and never as a [change_controller] the replay keeps
(`src/server/wesnothd/game.cpp`, 1.18.4):
```cpp
send_and_record_server_message(player_name + " takes control of side " + side + ".");   // 590
send_and_record_server_message(user->info().name()
    + (disconnect ? " has disconnected." : " has left the game."), player);           // 1492-1493
change_controller(side_index, owner_, username(owner_));                               // 1517
send_and_record_server_message(username(user) + " has surrendered.");                  // 1054
```
A leaver's sides go to the host (1517), announced as "takes control".
A surrendering player's first side goes to the host, or to the next
side's player when the surrenderer is the host (1034-1052), and that
"takes control" line comes BEFORE "has surrendered.". The [surrender]
command the replay keeps carries the client's viewing team, 0-based:
`pmc->surrender(display::get_singleton()->viewing_team());`
(`src/quit_confirmation.cpp:78`), `std::size_t viewing_team() const {
return currentTeam_; }` (`src/display.hpp:121`), accepted only when
`sides_[side_number] == user` (game.cpp:927-935). So `side_number=0`
is side 1 surrendering. `tools/replay_control.py` reads these lines.

**Why non-obvious:** read as a side number, `side_number=1` makes side 1
the loser when side 2 surrendered; `tools/build_value_corpus.py` read it
so, and named the surrendering side as the winner of every surrender game
of the 2026-07 value corpus (fixed 2026-09-26). And by the time "X has surrendered."
appears, X no longer holds the side they surrendered.

### Recruit `place_recruit` zeroes MP and attacks

`wesnoth_src/src/actions/create.cpp:626-631`:
```cpp
if (full_movement) {
    u->set_movement(u->total_movement(), true);
} else {
    u->set_movement(0, true);
    u->set_attacks(0);
}
```

`full_movement` is false for the normal recruit path. Recruits have
0 MP AND 0 attacks on their spawn turn. Both reset on next side's
init_side via the standard `unit::new_turn` path.

---

## Rounding rules

Wesnoth has TWO different integer-rounding rules that look similar
but apply to different domains. Mixing them up has burned multiple
sessions.

### `apply_modifier` / `div100rounded`: round-half-AWAY-from-zero (+50 bias)

Used for: HP percent modifications (`apply_to=hitpoints
increase_total=±N%`), max_experience percent (`apply_to=max_experience
increase=±N%`), and any other `[effect]` percent that goes through
`utils::apply_modifier`.

`wesnoth_src/src/serialization/string_utils.cpp:395-408`:
```cpp
int apply_modifier( const int number, const std::string &amount, const int minimum ) {
    int value = 0;
    try {
        value = std::stoi(amount);
    } catch(const std::invalid_argument&) {}
    if(amount[amount.size()-1] == '%') {
        value = div100rounded(number * value);
    }
    value += number;
    if (( minimum > 0 ) && ( value < minimum ))
        value = minimum;
    return value;
}
```

`wesnoth_src/src/utils/math.hpp:38-41`:
```cpp
/** Guarantees portable results for division by 100; round half up, to the nearest integer. */
constexpr int div100rounded(int num) {
    return (num < 0) ? -(((-num) + 50) / 100) : (num + 50) / 100;
}
```

Net effect: round half AWAY from zero. Examples:
- `div100rounded(140) = (140+50)/100 = 1` → `+1`
- `div100rounded(-165) = -(165+50)/100 = -2` → `-2`
- `div100rounded(-140) = -(140+50)/100 = -1` → `-1`

Critical for trait-order-dependent HP. Resilient-then-Quick Dark
Adept: `28 → 33 → 31` (33 + div100rounded(33×-5) = 33-2 = 31).
Quick-then-Resilient: `28 → 27 → 32` (28 + div100rounded(28×-5) =
27, then +5 = 32). The order matters because `apply_modifier` works
on the CURRENT value, not the base.

Our port: `tools/traits.py:apply_traits_to_unit` uses running
`max_hp` (not original `u.max_hp`) and `(raw + 50) // 100` (not
Python's `int()` which truncates toward zero). Same fix applied to
`max_xp` for the intelligent / dim trait paths.

### `round_damage`: round half TOWARD base (the "50 rule")

Used for: combat damage after multiplicative modifiers (ToD bonus,
leadership, resistance). Different from `div100rounded` because
"round toward the base" means increases round DOWN, decreases round
UP — the rounded value is always closer to (or equal to) the
unmodified base.

`wesnoth_src/src/utils/math.hpp:75-84`:
```cpp
/**
 *  round (base_damage * bonus / divisor) to the closest integer,
 *  but up or down towards base_damage
 */
constexpr int round_damage(int base_damage, int bonus, int divisor) {
    if (base_damage==0) return 0;
    const int rounding = divisor / 2 - (bonus < divisor || divisor==1 ? 0 : 1);
    return std::max<int>(1, (base_damage * bonus + rounding) / divisor);
}
```

Mechanics:
- `bonus < divisor` (multiplier < 1, i.e. damage decreases) →
  `rounding = divisor/2`. Effect: round half UP — toward base.
- `bonus >= divisor` and `divisor != 1` (multiplier ≥ 1, damage
  increases) → `rounding = divisor/2 - 1`. Effect: round half DOWN
  — toward base.

Worked example (Strong Fencer hitting Dwarvish Steelclad at day):
- Base attack damage: 4. Strong adds +1 (ADDITIVE, applied to the
  weapon's modified_damage). Effective base = 5.
- Damage multiplier: 100 + 25 (ToD lawful_bonus = +25 at day) = 125.
- Resistance: Steelclad pierce-resist 30% → multiplier ×= 70.
  Combined: bonus = 125 × 70 = 8750, divisor = 10000.
- 8750 < 10000 → rounding = 5000.
- `(5 × 8750 + 5000) / 10000 = 48750 / 10000 = 4` damage.

Stacking rules:
- Additive damage modifiers (strong's +1, dextrous's +1) stack INTO
  the base via the trait's `[effect] apply_to=attack increase=...`
  applied to the weapon. They modify `weapon->modified_damage()`
  before round_damage runs.
- Multiplicative modifiers (ToD `lawful_bonus`, leadership, resistance)
  stack INTO `damage_multiplier`. ToD/leadership ADD percent points
  to a base of 100. Resistance MULTIPLIES the running multiplier
  (so multiple resistances compound; matches `attack.cpp:199`'s
  `damage_multiplier *= opp.damage_from(...)`).

Order from `wesnoth_src/src/actions/attack.cpp:182-203`:
```cpp
int base_damage = weapon->modified_damage();
int damage_multiplier = 100;
damage_multiplier += combat_modifier(...);
int leader_bonus = under_leadership(...);
if(leader_bonus != 0) {
    damage_multiplier += leader_bonus;
}
damage_multiplier *= opp.damage_from(*weapon, !attacking, opp_loc, opp_weapon);
damage = round_damage(base_damage, damage_multiplier, 10000);
```

Our port: `combat.py:177-192` (`round_damage`) and `combat.py:347-373`
(multiplier assembly). Bit-exact.

---

## Combat

### Attack must be from an adjacent hex

`wesnoth_src/src/synced_commands.cpp:152-230` (synced [attack] handler):
checks both source and destination units exist, then calls
`attack_unit_and_advance(src, dst, ...)`. Adjacency is enforced
by `battle_context` (battle context disables out-of-range weapons,
where for melee `max_range()=1`):

`wesnoth_src/src/actions/attack.cpp:148-152`:
```cpp
{
    const int distance = distance_between(u_loc, opp_loc);
    const bool out_of_range = distance > weapon->max_range() || distance < weapon->min_range();
    disable = weapon->has_special("disable") || out_of_range;
}
```

If source isn't adjacent to dest for a melee weapon, the engine
DOES NOT error — it disables the weapon. The attack proceeds with
zero strikes, consumes no synced RNG, and any `[random_seed]`
follow-up emitted by our exporter dangles → "found dependent
command in replay while is_synced=false" on the next outer-loop
iteration. **Implication for our sim**: never emit a bare `[attack]`
where source and destination aren't neighbors. If the policy picks
a non-adjacent target, plan a `[move]` first.

### Charge: doubles damage on BOTH sides, but ONLY if the unit with charge is the attacker

Wesnoth's `[charge]` weapon special doubles damage for both attacker
and defender during a single attack — but the bonus fires ONLY when
the unit possessing charge is the ATTACKER (initiates the attack).
On a counter-attack the charge unit gets no bonus and neither does
the opponent.

`wesnoth_src/data/core/macros/special-notes.cfg` plus the engine's
`[specials]` filter walker. Empirically the rule cashes out as:

- Horseman ATTACKS Dark Sorcerer (Horseman has charge):
  Horseman 18 dmg/strike (9×2), DS counter 8 dmg/strike (4×2).
- Skeleton ATTACKS Horseman (defender has charge but skeleton doesn't):
  Skeleton 7 dmg/strike, Horseman counter 9 dmg/strike. NO doubling.
- Horseman ATTACKS Horseman (both have charge):
  Both 18 dmg/strike. Charge fires once because the ATTACKER carries it.

Pre-2026-05-02 our combat doubled whenever the unit's own weapon had
charge (regardless of attacker/defender role). That over-counted
defender's counter-attack on a charge defender (Horseman defended
gives 18 instead of 9) and under-counted attacker's strike when the
unit didn't have charge but the opponent did (Skeleton attacking
Horseman: we'd double Horseman's counter, not Skeleton's strikes).

Fix in `combat.py:_compute_battle_stats`:

```python
self_charges = "charge" in weapon.specials
opp_charges = (opp_weapon is not None
               and "charge" in opp_weapon.specials)
charge_doubled = ((self_charges and is_attacker)
                  or (opp_charges and not is_attacker))
if charge_doubled:
    base_damage *= 2
```

### Petrified units are untargetable for attack (UI level)

`wesnoth_src/src/mouse_events.cpp:753`:
```cpp
target_eligible &= !target_unit->incapacitated();
```

The AI action API refuses it too, `src/ai/actions.cpp:208-212`
(1.18.4):
```cpp
	if(defender->incapacitated()) {
		LOG_AI_ACTIONS << "attempt to attack unit that is petrified";
		set_error(E_INCAPACITATED_DEFENDER);
		return;
	}
```
Only the synced replay handler (`src/synced_commands.cpp:152`, the
`attack` command) has no such check, so a recorded [attack] on a statue
would resolve; neither a player nor an AI can issue one. Our legality
mask marks petrified units inert (`action_sampler`, the `occupancy`
table) and `WesnothSim.step` refuses such an attack as well; both are
pinned by tests (`test_scenery_never_an_attack_target_nor_actor`,
`test_sim_gate_rejects_statue_attack_and_counts`).

### Healing: main sources take the MAX; rest adds on top

`src/actions/heal.cpp` (1.18.4 tag, verified 2026-07-12):
```cpp
inline bool update_healing(int & healing, int & harming, int value)
{
    if ( value > healing ) {
        healing = value;
```
Village/oasis (terrain `heals=8`), `regenerate`, and adjacent
healers do NOT sum: the single best source wins. Observable
consequence: two adjacent heals+4 healers give +4, not +8.

Rest healing (+2) is ADDED on top, before the poison branch:
```cpp
if ( patient.resting() || patient.is_healthy() )
    healing += game_config::rest_heal_amount;
```
**Why non-obvious**: `is_healthy()` (the `healthy` trait) bypasses
the `resting` state entirely — a healthy unit rest-heals even on a
turn after it moved, attacked, or WAS attacked (fighting only
clears `resting`, which healthy units don't need). Also: DEFENDING
breaks resting — `attack.cpp` calls `set_resting(false)` on both
combatants.

### Resting lifecycle: set after healing, cleared by MP deficit

`src/play_controller.cpp` `do_init_side` (1.18.4, verified
2026-08-07): after `calculate_healing(...)`:
```cpp
// Set resting now after the healing has been done.
for(unit& patient : resources::gameboard->units()) {
    if(patient.side() == current_side()) {
        patient.set_resting(true);
    }
}
```
then the `turn refresh` WML event fires (`pump().fire("turn_refresh")`
— space and underscore are interchangeable in event names: scenario
WML writes `name=turn refresh` and it matches).

`src/units/unit.cpp:1280-1292` `unit::end_turn` (via
`game_board::end_turn(side)`, `src/game_board.cpp:79-86`, current
side's units only; 1.18.4, line numbers re-read 2026-09-24):
```cpp
set_state(STATE_SLOWED,false);
if((movement_ != total_movement()) && !(get_state(STATE_NOT_MOVED))) {
    resting_ = false;
}
```
**Why non-obvious**: resting is not "didn't move or fight" — it is
"ended the turn at FULL MP". Any MP drain counts as activity, even
one the unit didn't choose. Mini_Maps_Collection exploits this: a
repeating `turn refresh` event `{MODIFY_UNIT (role=monster) moves 0}`
(`enclave_micro_isar.cfg:86-91`) pins tentacles at 0 MP right after
each refresh, so `end_turn` clears `resting` every round and they
heal regen-only +8, never +10 (user-verified viewer frames, Micro
Isar 38859: turn-4 heal 15→23; our former +2 rest produced a 1-HP
survivor whose ZoC forked the whole game). Sim port:
`tools/replay_dataset.py` end_turn handler + `turn refresh` firing
at the end of init_side; MODIFY_UNIT expands to `[modify_unit]` in
`tools/scenario_events.py::_load_core_macros`.

### End of a side's turn: the same for every controller, AI included

A side's turn ends through one path whatever controls it (1.18.4,
read 2026-09-24). `play_controller::play_side`
(`src/play_controller.cpp:1279-1308`) runs `maybe_do_init_side()`,
then `play_side_impl()`, whose AI branch is
(`src/playsingle_controller.cpp:497-498`)
```cpp
} else if(current_team().is_local_ai() || (current_team().is_local_human() && current_team().is_droid())) {
    play_ai_turn();
```
and `play_ai_turn` finishes with `require_end_turn();` (`:676-678`).
`play_side` then calls `sync_end_turn()`, which records `[end_turn]`
and sets the `TURN_ENDED` phase (`:752-774`). Back in `play_some`
(`:227-229`):
```cpp
if (!is_regular_game_end() && gamestate().in_phase(game_data::TURN_ENDED)) {
    finish_side_turn();
}
```
and `finish_side_turn` (`:243-254`) calls `finish_side_turn_events()`
(`src/play_controller.cpp:571-595`), in order:
```cpp
gamestate().board_.end_turn(current_side());
...
// Clear shroud, in case units had been slowed for the turn.
actions::clear_shroud(current_side());

pump().fire("side_turn_end");
pump().fire("side_" + side_num + "_turn_end");
pump().fire("side_turn_" + turn_num + "_end");
pump().fire("side_" + side_num + "_turn_" + turn_num + "_end");
// This is where we refog, after all of a side's events are done.
actions::recalculate_fog(current_side());
check_victory();
```
`game_board::end_turn` (`src/game_board.cpp:79-86`) calls
`unit::end_turn` on each of that side's units
(`src/units/unit.cpp:1280-1292`):
```cpp
expire_modifications("turn end");

set_state(STATE_SLOWED,false);
if((movement_ != total_movement()) && !(get_state(STATE_NOT_MOVED))) {
    resting_ = false;
}

set_state(STATE_NOT_MOVED,false);
// Clear interrupted move
set_interrupted_move(map_location());
```

**Why non-obvious**: an AI side's turn looks like it needs no ending
-- nobody clicks "end turn" -- so a port that drives a neutral side
with its own loop can record the `[end_turn]` and skip the effects.
The effects are real on the mini maps: a tentacle slowed by a player
stays slowed until ITS side's turn ends, and a tentacle pinned at 0 MP
by `turn refresh` must lose `resting` there, or it rest-heals +2 on
top of regeneration (previous entry). `unit::end_turn` is the only
turn-boundary code that clears `STATE_SLOWED` (the other clear is
`unit::new_scenario`, `src/units/unit.cpp:1303`), so the units of a
controller=null side, which never ends a turn, keep a slow across
turns: Silverhead Crossing's Tentacle, once slowed, stays slowed unless
it advances (an AMLA clears the slow, entry "AMLA grants +3 max_hp").
The simulator opens and closes the neutral side's turn through the same
appliers as a player's (`WesnothSim._play_neutral_turn`); the default
AI's attacks in between are `tools/neutral_ai.py`.

What the applier models of this list: slow expiry, the `resting`
rule and the refog (`tools/replay_dataset.py` end_turn handler, Rust
`apply_end_turn`). Not modelled, with the reason each is safe today:
- `STATE_NOT_MOVED`. Only `unit::remove_movement_ai` sets it, on a unit
  at full movement (`src/units/unit.cpp:2784-2791`), and the default AI
  calls that through a stop-unit action that is not recorded
  (`src/ai/actions.cpp:912-913`). Its move-to-targets phase hands a
  guardian with movement left a move from its hex to its hex
  (`src/ai/default/ca_move_to_targets.cpp:269-277`), which becomes that
  stop (`src/ai/actions.cpp:496-507`). So the engine's guardian ends its
  turn at 0 MP with `resting` kept, where ours ends it at full MP with
  `resting` kept: the same healing, but the engine's guardian shows 0 MP
  until its next turn. Replay playback replays the recorded commands
  instead of running the AI, so a replay, and our reconstruction of it,
  shows full MP as the simulator does. (Read from the source on
  2026-09-24, not observed in a running game.)
- Interrupted moves: nothing we interpret sets them.
- `expire_modifications("turn end")`: no pool scenario creates a
  `duration=turn end` object. The one core macro that does,
  `FORCE_CHANCE_TO_HIT` in `data/core/macros/utils.cfg`, is used by no
  multiplayer scenario or tracked add-on, and the two scenarios that
  create one in Lua, 2p_Dark_Forecast and 2p_Isle_of_Mists, are outside
  the pool.
- The `side turn end` events: none in `data/multiplayer`,
  `data/core/macros` or the three tracked add-ons (grep, 2026-09-24).

### Healing vs poison

`heal.cpp` (same function, rest already added before the branch):
```cpp
if ( !patient.get_state(unit::STATE_POISONED) ) {
    healing += heal_amount(side, patient, healers);
} else {
    curing = poison_progress(side, patient, healers);
    if ( curing == POISON_NORMAL && patient.side() == side )
        healing -= game_config::poison_amount;
}
```
  - CURE (village/oasis/regen/adjacent `cures`): poison cleared
    INSTEAD of healing — no main heal that turn. Rest still
    applies (added before the branch): a resting unit is cured
    AND heals +2.
  - SLOW / counteract (adjacent heals+4/+8 WITHOUT `cures`): no
    main heal, no poison damage — even heals+4 cancels the full
    8 poison. Rest still applies (net +2 if resting).
  - NORMAL: -8 on the patient's own turn. With rest: net -6,
    ONE combined clamp at the end (floor 1 HP — poison cannot
    kill; no intermediate clamping between rest and poison).
  - A healer with BOTH `cures` and `heals+x` cures and does NOT
    heal that turn.

Our sim's port lives in `tools/replay_dataset.py` (init_side
healing loop) and matches every branch above; oasis (`^Do`,
`heals=8`, not a village, cures poison like one) resolves via
`tools/terrain_resolver.terrain_heals` mirroring
`terrain.cpp:230`'s `max(base.heals_, overlay.heals_)`.

### Default (RCA) AI combat rating

`src/ai/default/attack.cpp` `attack_analysis::rating` (1.18.4,
~lines 298-345; fetched verbatim 2026-07-14):
```cpp
if(leader_threat) { aggression = 1.0; }
if(uses_leader)   { aggression = ai_obj.get_leader_aggression(); }
double value = chance_to_kill*target_value - avg_losses*(1.0-aggression);
if(terrain_quality > alternative_terrain_quality) {
    const double exposure_mod = uses_leader ? 2.0 : ai_obj.get_caution();
    const double exposure = exposure_mod*resources_used*
        (terrain_quality - alternative_terrain_quality)*vulnerability
        /std::max<double>(0.01,support);
    value -= exposure*(1.0-aggression);
}
value += ((target_starting_damage/3 + avg_damage_inflicted)
          - (1.0-aggression)*avg_damage_taken)/10.0;
if(!is_surrounded || (support != 0 && avg_damage_taken != 0)) {
    if(vulnerability > 50.0 && vulnerability > support*2.0
       && chance_to_kill < 0.02 && aggression < 0.75
       && !attack_close(target)) { return -1.0; }
}
if(!leader_threat && vulnerability*terrain_quality > 0.0 && support != 0) {
    value *= support/(vulnerability*terrain_quality);
}
value /= ((resources_used/2) + (resources_used/2)*terrain_quality);
if(leader_threat) { value *= 5.0; }
```
Inputs (analyze()): target_value = cost*(1+xp/max_xp);
resources_used = attacker cost (same scaling); terrain_quality =
cost-weighted defender-CTH-vs-attacker (x0.5 on village);
target_starting_damage = max_hp - hp. The combat CA
(src/ai/default/ca.cpp) executes the best-rated analysis only while
rating > 0.0 -- "no efficient fight -> idle".

**Why non-obvious**: `leader_threat` does NOT mean "the target is
a leader" -- analyze() sets it when the target stands adjacent to a
leader of the AI's OWN side (defending my leader justifies
sacrifices: aggression=1.0 and x5). A no-leader monster side
(Mini_Maps tentacles) therefore NEVER triggers it (independent
review 2026-07-14 corrected an inverted first reading that briefly
lived in this entry). Our stationary port:
tools/neutral_ai.py (exposure term exactly 0 for immobile
attackers; power_projection gates approximated as
support=vulnerability=0 pending a port -- documented there).

### The defender's counter weapon is chosen on a prediction that counts level-ups

**Rule (added 2026-09-25):** `choose_defender_weapon` simulates every
candidate counter with `combatant::fight` and keeps the one
`better_combat` prefers, comparing death probabilities, then
`average_hp()`. That simulation predicts level-ups: a unit is scored
at FULL HP in every outcome it survives when the fight's XP alone
reaches its `max_experience`, and in the outcomes where it kills when
only a kill's XP does. Death probabilities are unchanged.

GitHub 1.18.4 tag, fetched 2026-09-25. The default argument,
`src/attack_prediction.hpp:38`:
```cpp
	void fight(combatant &opponent, bool levelup_considered=true);
```
`battle_context::simulate` (`src/actions/attack.cpp:376`, called by
`choose_defender_weapon` at :643) passes none:
```cpp
		attacker_combatant_->fight(*defender_combatant_);
```
On the exact matrix, `complex_fight` (`src/attack_prediction.cpp:2189-2200`):
```cpp
	if(levelup_considered) {
		if(stats.experience + game_config::combat_xp(opp_stats.level) >= stats.max_experience) {
			m->forced_levelup_a();
		} else if(stats.experience + game_config::kill_xp(opp_stats.level) >= stats.max_experience) {
			m->conditional_levelup_a();
		}
```
and the same for B. `forced_levelup_a` moves every surviving cell to
row `a_max_hp_`; `conditional_levelup_a` moves only column 0, B dead
(:1157-1199; `merge_col` skips row 0, :793-802). A fight in which
neither side slows, drains, petrifies or berserks, neither starts
slowed, and each strikes at most once goes to `one_strike_fight`
instead (`do_fight`, :2223-2229). It uses the vector forms (:2038-2048),
and the kill case there is an approximation that treats the unit's HP
as independent of the kill (`conditional_levelup`, :1733-1752):
```cpp
	double scalefactor = 0;
	const double chance_to_survive = 1 - hp_dist.front();
	if(chance_to_survive > DBL_MIN) {
		scalefactor = 1 - kill_prob / chance_to_survive;
	}
```
followed by scaling every surviving entry and
`hp_dist.back() += kill_prob;`. `no_death_fight` applies only the
forced form (:1959-1965), where nobody can die.

**Why non-obvious:** the prediction values a levelling unit's damage
at nothing, so it changes which counter the defender picks. Corpus
game `0223d226cc2f` (replays_dataset), command 333: a Skeleton at
26/27 XP attacks a Dwarvish Fighter. The fight's 1 XP levels it, both
counters leave it at 34 in the prediction, they tie, and
`better_combat`'s last tie-break (`them_a.average_hp() <
them_b.average_hp()`, attack.cpp:514) keeps the first candidate: the
axe, which the engine recorded. Scored at its real HP, the hammer's
10x2 against the Skeleton's impact weakness wins. Over 149 long games
of the imitation corpus, our choice matches the recorded counter in
141 of the 143 attacks where a level-up was possible, 138 without the
rule; the 4 disagreements left among 737 recorded choices are exact
ties in expected damage, where the engine's choice rests on
floating-point residue our strike DP does not reproduce
(`training/metrics/fidelity/counter_weapon_census_20260925.json`,
`tools/analysis/counter_weapon_census.py`).

Ours: `tools/combat_outcomes._levelup_average_hp` and
`_is_one_strike_fight`, applied in `_engine_marginals`. Tests:
`tests/test_counter_weapon.py::test_an_attacker_the_fight_levels_is_scored_at_full_hp`,
`test_levelup_scoring_follows_the_engine_fight_paths`.

### Attacking a petrified defender

`wesnoth_src/src/actions/attack.cpp` and `unit.hpp:1352-1355`:
the petrified defender has `attacks_left()=0` (no counter-attack)
and the attacker's strikes proceed normally. Our combat resolver
must skip the counter-attack when defender is petrified (we already
do this in `tools/replay_dataset.py`).

### At most ONE combatant dies per fight, and the plague corpse rises after the victim is erased

A fight can never end with both units dead. Only the target of a
strike takes damage, and a death ends the fight
(`attack.cpp:1161-1163`, `if(dies) { ... return false; }`). The one
code path that could kill the striker — negative drain — is floored
so it cannot:

`wesnoth_src/src/actions/attack.cpp:1039-1040`:
```cpp
		// if drain is negative, don't allow drain to kill the attacker
		drains_damage = std::max<int>(drains_damage, 1 - attacker.get_unit().hitpoints());
```

so `take_hit(-drains_damage)` always leaves at least 1 HP and the
`attacker_dies` branch never fires, `attack.cpp:1143-1159`:
```cpp
	bool attacker_dies = false;

	if(drains_damage > 0) {
		attacker.get_unit().heal(drains_damage);
	} else if(drains_damage < 0) {
		attacker_dies = attacker.get_unit().take_hit(-drains_damage);
	}

	if(dies) {
		unit_killed(attacker, defender, attacker_stats, defender_stats, false);
		update_fog = true;
	}

	if(attacker_dies) {
		unit_killed(defender, attacker, defender_stats, attacker_stats, true);
		(attacker_turn ? update_att_fog_ : update_def_fog_) = true;
	}
```

Even if it did, that call passes `drain_killed=true`, which gates the
plague spawn off, `attack.cpp:1283-1287`:
```cpp
	units_.erase(defender.loc_);
	resources::whiteboard->on_kill_unit();

	// Plague units make new units on the target hex.
	if(attacker.valid() && attacker_stats->plagues && !drain_killed) {
```

Note the ORDER in that same quote: the dead unit is erased from the
unit map BEFORE the corpse is created, and the corpse takes a fresh
id from the monotonic `id_manager` (`unit::create`), not from the
surviving units. `attacker.valid()` is what makes the plague
direction-agnostic: on a counter-kill the roles are swapped at the
call site (`unit_killed(defender, attacker, ...)`), so the defender's
plague weapon raises a corpse for the DEFENDER's side on the
attacker's hex.

**Why non-obvious**: `unit_killed`'s parameters are named
`attacker`/`defender` but hold the STRIKER and its TARGET for the
strike that landed the kill, so the reverse-plague case reads as the
forward one. And the `attacker_dies` branch looks like a live
both-die path until you trace the floor 100 lines earlier.

Our mirrors: `wesnoth_ai/combat.py::_perform_hit_body` (`heal =
max(heal, 1 - striker.hp)`) and `rust/wesnoth_core/src/combat.rs`;
both fight loops break on the first death, so `resolve_attack` cannot
return `attacker_alive=False` together with `defender_alive=False`.
The two appliers (`replay_dataset.py::_apply_command` "attack" and
`wesnoth_ai/game_core.py::_apply_attack`) therefore agree on the
corpse's id even though the core erases both combatants up front
while Python erases them one branch at a time.

### Chance-to-hit formula uses `accuracy` and `parry`

`wesnoth_src/src/actions/attack.cpp:168-169`:
```cpp
signed int cth = opp.defense_modifier(...) + weapon->accuracy()
    - (opp_weapon ? opp_weapon->parry() : 0);
```
clamped to [0, 100]; THEN the `marksman` (60) / `magical` (70)
specials apply as floors via `weapon->composite_value(...)`.

`accuracy=N` and `parry=N` are NUMERIC attrs on the `[attack]`
block, distinct from the named `[specials]` macros. Default era
1.18.4 uses them on exactly one weapon — Elvish Champion sword
(`accuracy=10`). That makes them easy to miss when scraping;
when missed, the Champion's 5-strike sword effectively misses 10%
more often than reality.

**Why non-obvious**: every other CTH adjustment in default era
goes through a `[specials]` macro (`marksman`, `magical`, `feeding`,
`charge`, etc.), so it's natural to assume the scrape only needs
to track the [specials] children. The numeric `accuracy` / `parry`
attrs sit directly on `[attack]` and require their own scrape path.

`tools/scrape_unit_stats.py::extract_attacks` reads them; see
`combat.py::_compute_battle_stats` for the CTH application
(adds before the marksman/magical floor). Test:
`test_combat_rules.py::test_accuracy_adds_to_cth`.

### Illuminate is a terrain-light modifier, not an ally aura

`wesnoth_src/src/tod_manager.cpp:237-262` (in
`get_illuminated_time_of_day`):
```cpp
std::array<map_location, 7> locs;
locs[0] = loc;
get_adjacent_tiles(loc, locs.data() + 1);
for(std::size_t i = 0; i < locs.size(); ++i) {
    const auto itor = units.find(locs[i]);
    if(itor != units.end() && !itor->incapacitated()) {
        unit_ability_list illum = itor->get_abilities("illuminates");
        if(!illum.empty()) { ... record this value ... }
    }
}
```
The 7-hex scan visits every unit in `loc` + the 6 adjacent hexes
**without filtering by side**. An enemy Mage of Light at the
attacked hex still illuminates the attacker, lifting the attacker's
lawful_bonus from −25 (first_watch) to 0 (dusk-equivalent).

**Why non-obvious**: the WML description ("any units adjacent to
this unit will fight as if it were dusk when it is night") doesn't
mention the side filter, but it's intuitive to think of illuminate
as buffing your team. The engine treats it as an environmental
ToD shift instead — adjacent enemies retaliating into the
illuminated hex feel the same boost.

Witnessed in `2p__Hamlets_Turn_20_(41655)` cmd[731]: Merman
Netcaster (lawful) striking a side-2 Mage of Light at first_watch.
Without the area-illumination credit, the Netcaster's club did
5/hit (lawful −25%) instead of 7/hit, leaving the MoL at 4 hp
where Wesnoth had it dead. Cascade: cmd[760] attacker_missing.

`tools/abilities.py::illuminate_step` no longer filters by side.
Test: `test_combat_rules.py::test_illuminate_lights_enemy_too`.

### Petrified/incapacitated units project NO adjacent abilities

A petrified (or otherwise `incapacitated()`) unit is skipped when a
neighbor gathers adjacency abilities. `wesnoth_src/src/units/abilities.cpp`
(`get_abilities_weapons` / the `affect_adjacent` loop, ~line 150 in the
1.18.4 tag):
```cpp
const unit_map::const_iterator it = units.find(adjacent[i]);
if(it == units.end() || it->incapacitated())
    continue;
// ... only here are it's abilities considered for the neighbor ...
```
`incapacitated()` is `get_state(STATE_PETRIFIED)` (see the ZoC entry
above, `unit.hpp`). Illumination is the same rule via a different code
path (`tod_manager.cpp:443`, quoted above). So a petrified **source**
unit contributes no `leadership`, `heals`, `cures`, or `illuminates`
to its neighbors (and `is_backstab_active` already excludes a petrified
flanker). Note this is the SOURCE-side rule; the RECIPIENT-side rule
(a petrified patient receives no healing — `heal.cpp` gates on
`patient.incapacitated()`) is separate.

**Why non-obvious**: `heal.cpp` itself has no source-side check — the
filtering is delegated to `unit::get_abilities`, so grepping the
healing code alone misses it. Verified 2026-07-01 against the 1.18.4
tag while fixing a review finding: all four `tools/abilities.py`
scanners (`illuminate_step`, `leadership_bonus`, `healer_heal_amount`,
`adjacent_curer`) now skip `"petrified" in source.statuses`.

### AMLA grants +3 max_hp, +20% max_experience, AND clears poisoned/slowed

`wesnoth_src/data/core/macros/amla.cfg:4-30` (`AMLA_DEFAULT`):
```
[advancement]
    strict_amla=yes
    max_times=100
    [effect]
        apply_to=hitpoints
        increase_total=3
        heal_full=yes
    [/effect]
    [effect]
        apply_to=max_experience
        increase=20%
    [/effect]
    [effect] apply_to=status remove=poisoned [/effect]
    [effect] apply_to=status remove=slowed [/effect]
[/advancement]
```

The `+20%` on `max_experience` COMPOUNDS across AMLAs — each AMLA
fires after the unit reaches the now-larger threshold, so the
sequence is `36 → 43 → 52 → 63 → ...` for a Sharpshooter base 150
under `experience_modifier=30` and intelligent trait. The 20%
modifier rounds half-up via `apply_modifier` (`+50` bias before
`/100`): `int(36*20+50)//100 = 7` so `36 → 43`.

**Why non-obvious**: skipping the `max_experience` increase looks
harmless — the unit just AMLAs more often than it should. But
each extra AMLA full-heals the unit via `heal_full=yes`, so combats
that should have killed the unit instead leave it healthy at full
hp. We hit this in `2p__Weldyn_Channel_Turn_26_(53914)`: a
Sharpshooter that should have died to an Arch Mage's fire was
saved by a spurious 2nd AMLA mid-combat that healed 40→60 hp.

`tools/replay_dataset.py::_maybe_advance_unit` (AMLA branch).
Tests: `test_combat_rules.py::test_amla_increases_max_exp_20pct`,
`test_amla_clears_slowed`.

### Walking Corpse variations preserve parent unit's `[resistance]` overrides

In Wesnoth, when a `[unit_type]` has a child `[variation] inherit=yes`
that switches `movement_type=...`, the variation effectively
re-applies the parent unit's own `[resistance]` / `[defense]` /
`[movement_costs]` overrides on top of the new movetype's defaults.

Example: Walking Corpse base unit declares `[resistance] arcane=140`.
The mounted variation switches to `movement_type=mounted` (which has
arcane=90 by default). The mounted variant inherits the parent's
arcane=140 override despite the movetype switch — Wesnoth does this
implicitly because `inherit=yes` copies the parent cfg into the
variation cfg before the engine resolves resistances.

**Why non-obvious**: an early read of `extract_variations` looks
correct — pull the new movetype's defaults, layer the variation's
own overrides. But the parent's overrides are silently dropped
unless re-applied. Plague-spawn cascades: a Cavalryman killed by
plague becomes Walking Corpse:mounted; if our sim has its arcane
at 90 instead of 140, every faerie-fire / lightbeam attack on
that mounted corpse does ~64% the damage Wesnoth actually deals.

`tools/scrape_unit_stats.py::extract_variations` re-extracts the
parent unit's explicit [resistance]/[defense]/[movement_costs]
blocks and layers them on top of the new movetype's defaults.
Variation-level overrides still win on top of those.
Tests: `test_combat_rules.py::test_wc_mounted_preserves_arcane_140`,
`test_wc_scorpion_variation_overrides_win`.

---

## Units, traits, leaders

### LEADERS DO NOT GET RANDOM TRAITS in default-era multiplayer

`wesnoth_src/src/units/unit.cpp:880-883` (inside `generate_traits`,
the random-fill loop):
```cpp
// For leaders, only traits with availability "any" are considered.
if(!must_have_only && (!can_recruit() || avl == "any")) {
    candidate_traits.push_back(&t);
}
```

Translation: the candidate-trait pool is filtered by
`!can_recruit() || avl == "any"`. For leaders (`can_recruit()=true`,
so `!can_recruit()=false`), a trait must have `availability="any"`
to be eligible. **No trait in `wesnoth_src/data/core/macros/traits.cfg`
has `availability="any"`** (`musthave` traits exist; `availability=`
is absent on the random-pool traits). So the candidate pool for
leaders is empty after the filter, the empty-candidates check at
unit.cpp:886-888 fires:

```cpp
// No traits available anymore? Break
if(candidate_traits.empty()) {
    break;
}
```

and leaders end up with only their must-have traits (`undead` for
Lich/Dark Sorcerer, etc.). Random-pool traits (`strong`, `quick`,
`intelligent`, `resilient`, `healthy`) are all skipped for leaders.

**Why non-obvious**:
- `unit::create(temp_cfg, true, vcfg)` at `unit_creator.cpp:187`
  passes `use_traits=true`, which drives `generate_traits(false)`
  (NOT must-have-only) — superficially looking like it'd run the
  full random pool for leaders. The `!can_recruit()` filter on
  unit.cpp:881 is what actually empties the pool.
- The `wesnoth_src/changelog.md:12321` (Version 1.3.7, ~2007) entry
  says "Leaders can't get random traits yet, because it breaks MP".
  That's an OBSOLETE 1.3.x note. The current mechanism (the
  `!can_recruit() || avl == "any"` filter) was added later and is
  what enforces the rule in 1.18.4.

### `quick_4mp_leaders` post-pass adds `quick` to base-4 leaders

`wesnoth_src/data/core/macros/multiplayer.cfg`:
```
#define QUICK_4MP_LEADERS
    # This makes all leaders with 4 MP receive the quick trait, except ones with
    # unit.variables.dont_make_me_quick=yes (boolean)

    [event]
        name=prestart
        [lua]
            code = << wesnoth.require("multiplayer/eras.lua").quick_4mp_leaders(...) >>
            [args]
                {TRAIT_QUICK}
            [/args]
        [/lua]
    [/event]
#enddef
```

`wesnoth_src/data/multiplayer/eras.lua`:
```lua
res.quick_4mp_leaders = function(args)
    local make_4mp_leaders_quick = wml.variables["make_4mp_leaders_quick"]
    if make_4mp_leaders_quick == nil then
        make_4mp_leaders_quick = wesnoth.scenario.mp_settings and (wesnoth.scenario.mp_settings.mp_campaign == "")
    end
    if not make_4mp_leaders_quick then
        return
    end

    local trait_quick = args[1][2]
    for i, unit in ipairs(wesnoth.units.find_on_map { canrecruit = true, T.filter_wml { max_moves = 4 } }) do
        if not unit.variables.dont_make_me_quick then
            unit:add_modification("trait", trait_quick )
            unit.moves = unit.max_moves
            unit.hitpoints = unit.max_hitpoints
        end
    end
end
```

`ERA_DEFAULT` (and `ERA_HEROES`) in
`wesnoth_src/data/core/macros/multiplayer.cfg` always include
`{QUICK_4MP_LEADERS}`. So in default-era multiplayer, every
`canrecruit=true` unit with `max_moves=4` after must-have traits
gets the `quick` trait added at prestart, then has its
`moves` and `hitpoints` reset to max.

**Filter detail**: `max_moves=4` is exact equality, not `≤4`. In
practice no default-era leader type has base MP < 4, so equivalent
in 1.18.4 — but match the source if a future era introduces a
3-MP leader type.

### `ignore_race_traits=yes` clears the entire trait pool

`wesnoth_src/src/units/types.cpp:337-363`:
```cpp
for(const config& t : traits) {
    possible_traits_.add_child("trait", t);
}

if(race_ != &unit_race::null_race) {
    if(!race_->uses_global_traits()) {
        possible_traits_.clear();
    }

    if(cfg["ignore_race_traits"].to_bool()) {
        possible_traits_.clear();
    } else {
        for(const config& t : race_->additional_traits()) {
            if(alignment_ != unit_alignments::type::neutral || t["id"] != "fearless")
                possible_traits_.add_child("trait", t);
        }
    }
    ...
}

// Insert any traits that are just for this unit type
for(const config& trait : cfg.child_range("trait")) {
    possible_traits_.add_child("trait", trait);
}
```

Order:
1. Pool starts with the global pool (strong/quick/intelligent/resilient).
2. If race opts out (`uses_global_traits=false`): clear.
3. If unit-type has `ignore_race_traits=yes`: **clear everything**;
   else add race's `additional_traits` (musthaves like undead/
   mechanical, plus race-specific like dextrous for elves; with the
   `fearless` skip for neutral-alignment units).
4. Always: add the unit-type's own `[trait]` children.

Three 1.18.4 unit types use `ignore_race_traits=yes`: **Dark Adept**
(pool = quick/intelligent/resilient — NO strong), **Black Horse**
(pool varies by gender — see gender-specific traits caveat below),
**Bay Horse / Dark Horse**.

**Gender-specific [trait] children**: a `[unit_type]` may declare
traits inside `[male]` / `[female]` blocks (e.g. Black Horse's
`[male] {TRAIT_STRONG}` / `[female] {TRAIT_FEARLESS}`). Wesnoth
applies these per gender; our scraper currently merges both into
a single pool. Not relevant for default-era PvP (no leader / recruit
pulls a gendered trait), but flag if it ever hits the diff_replay
oracle.

### Trait `availability` semantics

`wesnoth_src/src/units/unit.cpp:766-797` (must-have phase) +
unit.cpp:855-893 (random-fill phase):

- `availability="musthave"` → applied to every unit of the type,
  unconditionally. Examples: `undead` on Skeleton, `mechanical`
  on Walking Corpse, `feral` on certain villagers.
- `availability="any"` → eligible for leader random pool. **No
  default-era trait uses this.**
- (no `availability=` attribute) → eligible for non-leader random
  pool only.

### init_side: healing and MP-refresh/income sit behind DIFFERENT gates

`wesnoth_src/src/play_controller.cpp:484-507` (1.18.4, do_init_side):
```cpp
	// Healing/income happen if it's not the first turn of processing,
	// or if we are loading a game.
	if(turn() > 1) {
		gamestate().board_.new_turn(current_side());
		current_team().new_turn();
		...spend_gold(expense)...
	}

	if(do_healing()) {
		calculate_healing(current_side(), !is_skipping_replay());
	}

	// Do healing on every side turn except the very first side turn.
	// (1.14 and earlier did healing whenever turn >= 2.)
	set_do_healing(true);
```
- **MP/attack refresh + income/upkeep**: only when `turn() > 1`.
  Turn-1 inits get NEITHER — this is what lets start events that
  modify turn-1 MP (Marshy Fill's leader shave) survive.
- **Healing** (rest, village, regeneration, healers, poison ticks):
  gated by `do_healing()`, which is false ONLY for the game's very
  first side-init and true from the second onward — INCLUDING the
  later sides of turn 1.

**Why non-obvious**: the comment above the `turn() > 1` block says
"Healing/income" but healing was split out in 1.16+ (the in-code
1.14 note). Folding healing into the turn gate skipped the Micro
Isar tentacles' turn-1 regeneration (+8 at side 3's first init,
user-observed in the viewer): our sim's tentacle entered a turn-2
fight at 11 HP where the engine's stood at 19, died vs survived on
identical rolls, and the fork surfaced as attack:defender_missing
two commands later (2026-08-06 fidelity sweep, 4 files).

### Plan Unit Advance (pickadvance): picks NARROW advances_to; [choose] indexes the narrowed list

`data/modifications/pick_advance/main.lua` (MAINLINE, shipped with
Wesnoth; runs as modification id `plan_unit_advance`):

- A pick replaces the unit's advancement options via an object
  effect (main.lua:46-56):
```lua
	unit:add_modification("object", {
		pickadvance = true,
		take_only_once = false,
		T.effect {
			apply_to = "new_advancement",
			replace = true,
			types = array
		}
	})
```
- The pick itself is a synced dialog (`wesnoth.sync.evaluate_single`,
  main.lua:128) recorded in the replay as TWO commands:
```
	[command] from_side=N [fire_event]
		raise="menu item pickadvance"
		[source] x= y= [/source]
	[/fire_event] [/command]
	[command] dependent=yes from_side=N [input]
		ignore=no
		is_unit_override=yes|no  unit_override="Type1,Type2"
		is_game_override=yes|no  game_override="..."
	[/input] [/command]
```
- **Consequence for reconstruction**: a subsequent advancement's
  `[choose] value=N` indexes the NARROWED list, not the unit-type's
  vanilla `advances_to`. With a single-type pick every recorded
  choose collapses to `value=0`.
- **Forced-choice mode** (`pickadvance_force_choice`, defaulted ON
  for maps without recruiting and settable per game): the dialog
  fires DIRECTLY inside the recruit event (main.lua:166-175), so the
  dependent `[input]` follows the recruit's `[random_seed]` with **NO
  `[fire_event]` marker at all** -- the pick target is the freshly
  recruited unit. An extractor keyed only on the menu-item
  fire_event silently drops every forced-mode pick (Tombs 113893:
  the engine honored a Lancer pick, 9 MP, while the un-narrowed
  reconstruction advanced Knight, 8 MP -- surfaced as
  move:mp_insufficient at turn 16, 2026-08-06 sweep 2).
- game_override semantics (main.lua:142-148): sets a per-side WML
  variable read by units of the type initialized LATER; current
  same-side same-type units get `set_advances(dialog.unit_override)`
  (the UNIT list -- a mod quirk). Freshly-advanced units re-init for
  their new type ("post advance", main.lua:231), clearing the old
  narrowing.
- **"Initialized LATER" includes NEW RECRUITS**: initialize_unit runs
  on the "recruit" event (main.lua:231 event list), so a unit
  recruited after a game_override inherits the narrowing. Missing
  this made a post-override Wolf Rider advance as Goblin Knight
  (vanilla index 0) where the engine made the overridden Goblin
  Pillager -- surfaced 35 commands later as `attack:weapon_oob` on
  the Pillager's third weapon (net), Hellhole 21368; engine playback
  of the file is clean end-to-end (harness-verified 2026-08-07).

**Why non-obvious**: the pick leaves NO trace on the advancement
command itself. Our index-into-vanilla-list resolution silently
advanced an Elvish Fighter to Captain (index 0) where the engine
made the picked HERO; the +1-damage difference surfaced 200 commands
later as a ZoC-blocked move (`move:src_missing`, CotB replay 74713,
root-caused 2026-08-06 from the user's engine-viewer HP/XP ledger).
Handled in `replay_extract.py` (compact `pickadvance` command) +
`replay_dataset._apply_command` / `_advance_unit_once`.

### Recruit NAME GENERATION also uses the synced MP RNG (named races only)

`wesnoth_src/src/utils/markov_generator.cpp:59-69` (1.18.4, comment
abridged):
```cpp
	// This name generator is called by mp_connect.cpp and
	// [get_random() getting called a different number of times ...
	//  causes a problem since the random state is no longer in sync ...]
	// To avoid that problem we call get_random()
	std::vector<int> random(max_len);
	...
		random[j] = randomness::generator->next_random();
```
and `unit.cpp:689` runs `generate_name()` during `unit::init` for
every freshly created unit (recruits included).

If the unit's RACE has a name source (a `{*_NAMES}` macro in
`data/core/units.cfg` — drake, dwarf, elf, goblin, gryphon, human,
dunefolk, lizard, merman, naga, ogre, orc, troll, wolf, wose; every
such macro defines BOTH genders), the generator draws a fixed
max_len batch of `next_random()` from the SYNCED RNG. Nameless
races (undead, mechanical, bats, monster, falcon, horse, raven) get
the base `name_generator` whose `generate()` returns `""` with zero
draws.

**Why non-obvious**: RNG-consumption prediction based on
gender+traits alone is correct for every nameless-race unit
(Skeleton, Ghoul — the 2026-07-06 calibration) and coincidentally
correct for every unit that also draws traits — it breaks ONLY on
named-race units with zero random-trait draws. In default-era 2p
that's essentially the WOSE (num_traits=0). Found by the validation
pipeline's very first batch (2026-07-15): a Ruined Passage midgame
export hit "found [recruit] command in replay expecting a user
choice" exactly at a human Wose recruit. `_NAMED_RACES` in
`tools/sim_to_replay.py` pins the race list (derived from the
1.18.4 tag — don't re-derive from the local 1.18.7 wesnoth_src).

### Recruit traits use the synced MP RNG

`wesnoth_src/src/units/unit.cpp:890`:
```cpp
int num = randomness::generator->get_random_int(0,candidate_traits.size()-1);
```

The active RNG when this runs depends on the surrounding context.
For starting units (including leaders) in scenario init, it's the
synced gamedata RNG (see `game_state.cpp:188-200`):
```cpp
{
    //sync traits of start units and the random start time.
    randomness::set_random_determinstic deterministic(gamedata_.rng());
    ...
    for(team_builder& tb : team_builders) {
        tb.build_team_stage_two();
    }
    ...
```

For recruits during play, the RNG is keyed by the per-recruit
`[random_seed]` block in the replay; trait order: gender call iff
multi-gender, then per-trait `get_random_int(0, len-1)` in the
candidate pool.

### A unit type's experience need is `(base * modifier + 50) / 100`, at least 1

`src/units/types.cpp:577-589`:

```cpp
int unit_type::experience_needed(bool with_acceleration) const
{
	if(with_acceleration) {
		int exp = (experience_needed_ * experience_modifier + 50) / 100;
		if(exp < 1) {
			exp = 1;
		}

		return exp;
	}

	return experience_needed_;
}
```

`unit_stats.json` stores the BASE value (Swordsman 80,
`wesnoth_src/data/core/units/humans/Loyalist_Swordsman.cfg:11`). Lua's
`wesnoth.unit_types[t].max_experience` returns the accelerated one
(`src/scripting/lua_unit_type.cpp:58`, `ut.experience_needed()`), so a type
whose base is exactly 100, such as the Dwarvish Berserker, reads the
modifier the game applies: `(100 * m + 50) / 100 = m`.
`add-ons/wesnoth_ai/lua/init_oracle.lua` reports it that way.

### `not_living` is a name for three statuses, not a status of its own

`src/units/unit.cpp:1334-1337`:

```cpp
	// Backwards compatibility for not_living. Don't remove before 1.12
	if(all_states.count("undrainable") && all_states.count("unpoisonable") && all_states.count("unplagueable")) {
		all_states.insert("not_living");
	}
```

`get_state("not_living")` returns the conjunction of the three
(1349-1355) and setting it sets all three (1400-1405). A unit's saved or
Lua-reported statuses therefore list `not_living` beside its parts; our
units carry the three parts only.

### Advancement keeps the movement left, clamped only after traits are re-applied

**Rule (added 2026-09-25):** a unit that advances keeps the movement
points it had, capped at the new type's total movement WITH its traits
and objects: a quick Spearman at 6/6 becomes a quick Swordsman at 6/6.

`get_advanced_unit` (`src/actions/advancement.cpp:319-322`, 1.18.4)
clones the unit and calls `advance_to`, then `heal_fully`; nothing
after touches movement. `unit::advance_to` (`src/units/unit.cpp:921`)
snapshots first:
```cpp
	auto ss = stats_storage_resetter(*this, true);
```
It then resets `max_movement_ = new_type.movement();` (:987), re-applies
the traits and objects with `apply_modifications();` (:1021), and only
then restores the snapshot with `ss();` (:1026), whose clamping branch
is (:195-196):
```cpp
			if(clamp) {
				u.set_movement(std::min(u.total_movement(), moves));
```

**Why non-obvious:** at :987 the maximum is the bare type's, so a
clamp taken there, before quick adds its point back, loses the point.
It shows only on a unit that levels while defending with more movement
left than its new type's bare total, and only until its side's next
turn start refreshes movement: attacking spends all movement before
the advance.

Ours: `tools/replay_dataset._advance_unit_once`. Test:
`tests/test_sim_advance.py::test_a_quick_defender_keeps_its_extra_move_through_advancement`.

---

## Villages

### Pre-owned villages from `[side]/[village]` children

`wesnoth_src/src/team.cpp:208-217` (in `team::team(const config&)`):
```cpp
const config::const_child_itors& villages = cfg.child_range("village");
for(const config& v : villages) {
    map_location loc(v["x"].to_int() - 1, v["y"].to_int() - 1);
    villages_.insert(loc);
}
```

A `[side]` block can carry `[village] x=N y=N [/village]` children that
declare which villages the side owns at scenario start. These count
toward the side's village list from turn 1, contributing income at
init_side(turn>1) AND support against upkeep.

In the 1.18.4 default-era 2p maps, only some scenarios use this:
Clearing Gushes bundles a village with side 2's keep, Arcanclave
Citadel pre-credits both sides. Without honoring these, our
reconstructor under-counts side income for the entire game and
cascades into "recruit:insufficient_gold" divergences mid-game.

Our handling:
- `tools/replay_extract.py` walks `[side]/[village]` children at
  extraction time, populates `gs.villages_owned`, serializes to
  `starting_villages` in the JSON.
- `tools/replay_dataset.py:_build_initial_gamestate` reads
  `starting_villages`, marks each hex with the VILLAGE modifier,
  bumps `nb_villages_controlled` on the appropriate side, and
  populates `_village_owner` so subsequent move-time captures
  don't double-credit ownership.

### Same-side village revisit: NO ownership change

`wesnoth_src/src/actions/move.cpp` and friends (the `try_actual_movement`
path) call `actions::get_village(loc, side, ...)` which checks
`village_owner == side` and returns early without touching the
team's `villages_` list. Matches `game_board.cpp` `village_owner`.

Practical effect: a leader walking back onto its own village does
NOT lose that village's count. Our pre-2026-05-02 `_capture_village`
in `tools/replay_dataset.py` decremented unconditionally and only
guarded the increment on `prev != capturing_side`, leaving a -1 net
each revisit. Symptom: side-1 village count drops mid-turn after a
leader return-to-keep, income underpays for several turns.

Our fix: short-circuit `_capture_village` when `prev_owner ==
capturing_side` — no count change, just keep the modifier on the hex.

## Recruit and recall

### Recruit cost is deducted by `team::spend_gold`, no clamp at 0

`wesnoth_src/src/team.hpp` (`spend_gold`):
```cpp
void spend_gold(const int amount) {
    info_.gold -= amount;
}
```

No clamp. Wesnoth allows negative gold from upkeep drain. Recruits
that would push gold below zero are rejected upstream
(`find_recruit_location` and the synced `[recruit]` handler), not
in `spend_gold`. **Don't `max(0, ...)` clamp gold in our sim** —
band-aid that hides upstream gate bugs.

### Income / upkeep applied at side's `init_side`, only for `turn() > 1`

`wesnoth_src/src/play_controller.cpp` `do_init_side` runs once per
side-turn. Income = base_income + villages_owned * village_gold.
Upkeep = sum(unit.level for non-loyal, non-leader units of side).
Net = +income − max(0, upkeep − support).

Leaders (`canrecruit=true`) and `loyal`-trait units don't contribute
to upkeep. Verified at `wesnoth_src/src/units/unit.cpp:1746-1751`
(`unit::upkeep` short-circuits on `can_recruit()`).

### `village_gold` / `village_support` are PER-SIDE attributes, set by host

`src/team.cpp:236-244` (1.18.4; corrected 2026-09-25, this entry cited
235-243 and quoted only the support half):
```cpp
	income_per_village = cfg["village_gold"].to_int(game_config::village_income);
	recall_cost = cfg["recall_cost"].to_int(game_config::recall_cost);

	const std::string& village_support = cfg["village_support"];
	if(village_support.empty()) {
		support_per_village = game_config::village_support;
	} else {
		support_per_village = lexical_cast_default<int>(village_support, game_config::village_support);
	}
```

**A declared 0 is kept.** Both defaults apply only to a value that is
not there: `village_support` tests `empty()`, and `to_int` returns its
argument only for a blank, boolean, translatable or unparsable value,
an integer coming back as itself (`src/config_attribute_value.cpp:277-283`):
```cpp
	T operator()(const utils::monostate&) const { return def_; }
	T operator()(bool)                 const { return def_; }
	T operator()(int i)                const { return static_cast<T>(i); }
```
So `village_gold=0` pays nothing per village and `village_support=0`
supports no upkeep. 16 of the corpus's 17,019 games declare one of the
two: 7 the gold (the 7 raw headers with `mp_village_gold` 0 in
`training/metrics/corpus_census.json`), 10 the support, one both, read
from every record's `starting_sides` on 2026-09-25. Until then both appliers read a 0 as
"not set" and paid the multiplayer default (`village_gold or 2` in
Python, `!= 0` in the Rust core), which gave those games more gold
than the engine did; no recorded recruit could fail on it, so the
replay sweep could not see it. Ours: `tools/wml_state.village_economy`
(the default only for None) and `apply_init_side` in
`rust/wesnoth_core/src/core_step.rs`. Tests:
`tests/test_scenario_economy.py::test_a_declared_zero_village_economy_is_paid_as_zero`,
`tests/test_game_core.py::test_init_side_pays_a_declared_zero_village_economy`.

`village_gold=` is on the `[side]` block, not global. In MP, the
host's game-options dialog sets it identically across all sides, but
`replay_extract.py` MUST capture it per-side -- our
`SideState.village_income` field reads `[side] village_gold` directly.
The
host can also customize it. NOT reading the per-side value and
defaulting to a hardcoded 2 was the cause of a multi-replay diff
divergence (commit 2026-05-03, Den of Onis #14f7a7c1a17f).

Same for `village_support=` (default 1, mainline always 1 but capture
it for completeness — `SideState.village_support`).

**A scenario writes it either of two ways, and both must be read**
(2026-09-21). The per-side `[side] village_gold=` above is the
runtime form and the one the Mini Maps Collection uses: five of its
seven 1v1 scenarios ask for 3. Mainline maps instead declare the
game-creation setting `mp_village_gold=` on the scenario, which
multiplayer setup copies onto every side --
`wesnoth_src/data/multiplayer/scenarios/2p_Clearing_Gushes.cfg:15`
and `2p_The_Walls_of_Pyrennis.cfg:15` (both 2),
`2p_Cynsaun_Battlefield.cfg:14` (2), `2p_Dark_Forecast.cfg:16` and
`2p_Isle_of_Mists.cfg:19` (both 1). `tools/scenario_pool.
scenario_economy` reads both, the per-side form winning.

**The 1v1 multiplayer default is 2, not 5.** An earlier revision of
this entry read "Default Era uses **5** gold per village (not the
historic 1)"; nothing supports it. Measured over the corpus's raw
replay headers (`tools/analysis/corpus_census.py`, 17,019 games):
`mp_village_gold` is 2 in 16,712, absent in 113, and 5 in 26. Every
mainline 2p map that declares the setting declares 1 or 2. The engine's
own constant is `village_income=1` (`wesnoth_src/data/game_config.cfg:17`);
the multiplayer 2 is what a hosted game writes into the sides (next
entry).

### A command-line `--multiplayer` start skips the lobby's parameter writes

A hosted game runs `configure_engine::write_parameters`
(`src/game_initialization/configure_engine.cpp:168-208`, 1.18.4), which
writes the host's settings into the scenario it is about to start:

```cpp
	scenario["experience_modifier"] = params.xp_modifier;
	scenario["turns"] = params.num_turns;

	for(config& side : scenario.child_range("side")) {
		if(!params.use_map_settings) {
			...
		} else {
			if(side["fog"].empty()) {
				side["fog"] = params.fog_game;
			}
			...
			if(side["village_gold"].empty()) {
				side["village_gold"] = params.village_gold;
			}
```

`mp::start_local_game_commandline`
(`src/game_initialization/multiplayer.cpp:692-794`) sets the same
parameters but builds no `configure_engine`, so none of them reaches the
scenario. A side that declares no village gold then pays the engine's
base rate, `income_per_village = cfg["village_gold"].to_int(game_config::village_income);`
(`src/team.cpp:236`), which is 1, and units need their BASE experience,
because the level keeps `level["experience_modifier"].to_int(100)`
(`src/play_controller.cpp:160`). A side declaring no fog plays without
it: `fog_.set_enabled(cfg["fog"].to_bool());` (`src/team.cpp:358`).

**Why non-obvious:** both starts share `connect_engine`, and Lua's
`wesnoth.scenario.mp_settings` reports village gold 2 and experience
modifier 70 in both, because those are the parameters, not what the
scenario received. Measured 2026-09-23 on Caves of the Basilisk: every
side paid 1 per village and a Swordsman leader needed 80 experience,
its base. `tools/scenario_init_oracle.py` writes the lobby's side values
back with `--parm`, choosing the sides from Wesnoth's own preprocessing
of the scenario, and reads the applied experience modifier with the
probe in the next section's experience entry.

---

## Replay structure

### `[command] dependent="yes"` follow-ups consume synced RNG, not vice versa

The replay engine's `is_synced` flag goes true when a synced command
starts (recruit, attack, move-with-rng, etc.) and goes false when
the command's handler returns. A `[command] dependent="yes"` block
is consumed BY the synced handler when it calls `get_random_int`
(`get_user_choice`-style).

Failure mode: emit a `[command] dependent="yes" [random_seed]` block
after a synced command that DOESN'T actually call `get_random_int`
in its handler (e.g. a Skeleton recruit that has no random traits
to roll). The next outer-loop iteration sees the dep block while
`is_synced=false` → "found dependent command in replay while
is_synced=false". See `tools/sim_to_replay.py:_command_consumes_synced_rng`
for the gate logic.

### Order around `[attack]`

`wesnoth_src/src/actions/attack.cpp:1556-1573` (`attack_unit_and_advance`):
```cpp
attack_unit(attacker, defender, attack_with, defend_with, update_display);

unit_map::const_iterator atku = resources::gameboard->units().find(attacker);
if(atku != resources::gameboard->units().end()) {
    advance_unit_at(advance_unit_params(attacker));
}

unit_map::const_iterator defu = resources::gameboard->units().find(defender);
if(defu != resources::gameboard->units().end()) {
    advance_unit_at(advance_unit_params(defender));
}
```

WML order in our exported replay must be: `[attack]` →
`[command dependent="yes"][random_seed]` → optional attacker
`[choose]` (if attacker advanced) → optional defender `[choose]`
(if defender advanced).

---

## Scenario events

### `current_time` overrides `random_start_time`; schedule macros set it

`wesnoth_src/src/tod_manager.cpp:51-55` (1.18.4, constructor):
```cpp
// ? : operator doesn't work in this case.
if(scenario_cfg["current_time"].to_int(-17403) == -17403) {
    random_tod_ = scenario_cfg["random_start_time"];
} else {
    random_tod_ = false;
}
```
and `:66-67`:
```cpp
currentTime_ = fix_time_index(times_.size(), scenario_cfg["current_time"].to_int(0));
```

If the scenario cfg carries a `current_time=N` attribute, the
turn-1 ToD slot is N and `random_start_time` is IGNORED entirely —
`resolve_random()` (called at scenario init under the synced
gamedata RNG, see the game_state.cpp quote in "Recruit traits use
the synced MP RNG") no-ops without consuming a draw. Only when
`current_time` is ABSENT does `random_start_time=yes` draw
`r.next_random()` from the synced RNG.

**Why non-obvious**: (1) the schedule macros hide the attribute —
`{DEFAULT_SCHEDULE_SECOND_WATCH}` (Fallenstar Lake, Ruined Passage)
expands to the six default `[time]` blocks PLUS `current_time=5`,
so those maps start at second watch even though their .cfg shows no
`current_time` textually. (2) The Mini_Maps pool has
`random_start_time=yes`: a sim-exported replay whose [scenario]
lacks `current_time` makes playback re-draw the slot from the
save's seed and every ToD-sensitive strike diverges (witnessed
2026-07-15: tentacle retaliation 4 sim-dawn vs 3 engine-day).
`build_save_wml` therefore always pins `current_time=<the slot the
sim played>`; `scenario_pool._scenario_tod_start` reads the slot
from the expanded template for fresh builds. Pinned by
`test_sim_to_replay_from_scratch.py::test_exported_save_pins_tod_start_slot`.

**The declared slot is wrapped, never taken raw (added 2026-09-26).**
`src/tod_manager.cpp:521-528`:
```cpp
int tod_manager::fix_time_index(int number_of_times, int time)
{
	if(number_of_times == 0) {
		return 0;
	}

	return modulo(time, number_of_times);
}
```
with `modulo` from `src/utils/math.hpp:62-74`, which adds `mod` to a
negative remainder: on the six-slot schedule `current_time=-1` starts
at second watch and `current_time=7` at morning. `wml_state.read_tod`
applies `wml_state.fix_time_index` where the slot is read, and
`_build_initial_gamestate` wraps a record's `tod_start_index` the same
way; the cycle indices (`_tod_cycle_index`, `_lawful_bonus_at`, and
`rem_euclid` in the Rust core's `core_step.rs` from phase 16) wrap
whatever slot they are given. Before, the raw value went through: the default-cycle index
clamped a negative one to dawn, the time-area index wrapped it, and the
Rust core panicked on it. No pool scenario or corpus record holds one
out of range (measured 2026-09-26).

### Turn events: which names fire, how often, in what order (added 2026-09-26)

**Rule:** at a side's turn start the engine fires `turn N` and
`new turn` only if no side has started this turn yet, then `side turn`,
`side S turn`, `side turn N`, `side S turn N`; after the refresh,
income and healing, `turn refresh`, `side S turn refresh`,
`turn N refresh`, `side S turn N refresh`. At a side's turn end it
fires `side turn end`, `side S turn end`, `side turn N end`,
`side S turn N end`, and after the last side's turn of a turn,
`turn end` and `turn N end`.

`src/play_controller.cpp:472-482` (1.18.4, `do_init_side`):
```cpp
		// We might have skipped some sides because they were empty so it is not enough to check for side_num==1
		if(!gamestate().tod_manager_.has_turn_event_fired()) {
			pump().fire("turn_" + turn_num);
			pump().fire("new_turn");
			gamestate().tod_manager_.turn_event_fired();
		}

		pump().fire("side_turn");
		pump().fire("side_" + side_num + "_turn");
		pump().fire("side_turn_" + turn_num);
		pump().fire("side_" + side_num + "_turn_" + turn_num);
```
then `:519-522`, after `board_.new_turn`, income, healing and resting:
```cpp
		pump().fire("turn_refresh");
		pump().fire("side_" + side_num + "_turn_refresh");
		pump().fire("turn_" + turn_num + "_refresh");
		pump().fire("side_" + side_num + "_turn_" + turn_num + "_refresh");
```
`finish_side_turn_events` (`:585-588`), after `board_.end_turn(side)`
and before the refog:
```cpp
		pump().fire("side_turn_end");
		pump().fire("side_" + side_num + "_turn_end");
		pump().fire("side_turn_" + turn_num + "_end");
		pump().fire("side_" + side_num + "_turn_" + turn_num + "_end");
```
and `finish_turn` (`:597-604`), which `finish_side_turn` calls when the
next side to play wraps to a new turn (`src/playsingle_controller.cpp:259-262`):
```cpp
	pump().fire("turn_end");
	pump().fire("turn_" + turn_num + "_end");
```
The latch is reset by `tod_manager::next_turn` (`src/tod_manager.cpp:574-579`,
`has_turn_event_fired_ = false;`). An event answers to every name in its
comma-separated `name=` list, each trimmed and with internal spaces made
underscores, case kept (`event_handler::names`,
`src/game_events/handlers.cpp:64-88`; `event_handlers::standardize_name`,
`src/game_events/manager_impl.cpp:65-76`).

Sim: `tools/scenario_events.py` names the four sequences
(`side_turn_event_names`, `turn_refresh_event_names`,
`side_turn_end_event_names`, `turn_end_event_names`), and the applier
fires them from `_apply_command`. It opens a turn at side 1's
init_side, where it counts turns (side 1 always opens a turn), so that
init_side fires the previous turn's two end names first, then `turn N`
and `new turn`. The Rust core runs no events: `CoreState.apply_command`
sends an init_side or end_turn to the Python applier when one of the
names it fires has an event that can still fire
(`init_side_event_names`). Not modelled: the `side_number` and
`turn_number` WML variables the engine sets for these events, and
event filters (`[filter_side]`, `[filter_condition]`).

**Why non-obvious:** "new turn" reads like "a side's new turn". The
applier fired `side S turn N`, `turn N`, `new turn` and `side turn` at
every side's init_side (to 2026-09-26): a repeating `new turn` or
`turn N` event ran once per side, the other side, refresh and end forms
never ran, and the order differed. No pool or corpus scenario defines
an event on a name whose behaviour changed: their turn events are
Aethermaw's `side 1|2 turn 4|5|6` (first-time-only terrain morphs, same
moment), the enclave minis' `turn 1` (first-time-only, same moment) and
`turn refresh` (unchanged), and Silverhead Crossing's repeating
`side 3 turn`, whose side is `controller=null` and never takes a turn.

### Pre-placed units via `[switch] variable=pN_faction [case]` (Hornshark Island)

Most MP maps spawn only leaders and let players recruit. **Hornshark
Island is the prominent exception**: each side gets 4-7 named-or-
anonymous pre-placed units chosen by the side's faction.

`wesnoth_src/data/multiplayer/scenarios/2p_Hornshark_Island.cfg:65-499`:
```
[event] name=prestart
    [lua]
        code= << for i, side in ipairs(wesnoth.sides.find({})) do
                    wml.variables["p" .. tostring(i) .. "_faction"] = side.faction
                 end >>
    [/lua]
    [fire_event] name=place_units [/fire_event]
[/event]

[event] name=place_units
    [switch] variable=p1_faction
        [case] value=Drakes
            [unit] side=1 type=Young Ogre x,y=24,4 ... [/unit]
            [unit] side=1 type=Drake Fighter x,y=1,1 ... [/unit]
            ...
        [/case]
        [case] value=Loyalists ... [/case]
        ...
        [else] ... [/else]   # fallback: monster grab-bag
    [/switch]
    [switch] variable=p2_faction ... [/switch]
[/event]
```

Implementation cost in our reconstructor:

  1. **`[set_variable] name=X value=Y`** — store on
     `gs.global_info._wml_variables` dict.
  2. **`[lua]` code= …** — we don't run a Lua interpreter; we just
     pre-populate `pN_faction` from `gs.sides[i].faction` directly
     in `_setup_scenario_events`. Robust to parser truncation of
     the multi-line `<<...>>` literal. Other [lua] blocks no-op.
  3. **`[fire_event] name=Y`** — find the named [event] in
     `gs.global_info._scenario_events`, run its actions (honor
     `first_time_only`).
  4. **`[switch] variable=X [case] value=V`** — match `X`'s value
     against each case's `value=` (comma-separated allowed),
     fall through to `[else]` if no case matches.
  5. **`[unit] side=N type=T x,y=X,Y [modifications]{TRAIT_…}…[/modifications]`** —
     spawn fresh unit. Honor `variation=` by composite-key lookup
     (e.g. `Soulless:saurian` for Hornshark Undead's named hero
     "Rzrrt the Dauntless"). Apply `[trait]id=…` children from
     `[modifications]` via `apply_traits_to_unit`. Walking-Corpse-
     family variations (saurian/dwarf/...) get the variation's
     movement_type and defenses, not the base humanoid's.

After the prestart event chain runs, side N has all its faction-
specific Hornshark units in addition to its leader.

**Why non-obvious**: most pre-placed-unit conventions in Wesnoth
use `[side]/[unit]` direct children (which Wesnoth's snapshot
materializes into `starting_units` for us). Hornshark instead uses
runtime event-driven spawning, which `replay_extract` cannot
materialize at extract time -- it has to be re-fired by our
`scenario_events.py` interpreter at `_setup_scenario_events` time.

### AMLA also emits `[choose] value=N`, must pop the queue

`wesnoth_src/src/actions/advancement.cpp:296`:
```cpp
config selected = mp_sync::get_user_choice("choose",
    unit_advancement_choice(params.loc_, ...), side_for);
```

`get_user_choice` is called regardless of how many advancement
options exist. For AMLA (After-Maximum-Level-Advancement), the
unit has only one (default) "advancement" — the +3 max_hp full-
heal — but Wesnoth STILL goes through the choose machinery.
The replay records `[choose] value=0`.

If our reconstructor doesn't pop the queue on AMLA, that stale
value=0 stays around and gets consumed by the NEXT unit's REAL
advancement, picking advances_to[0] instead of the choice the
replay actually recorded. **Found via Goblin Pillager misadvance**:
replay `1b43dd9087ae` cmd[1071]: Troll Rocklobber AMLA, value=0
left in queue. cmd[1084]: Wolf Rider's slot[8]=[1] pushed →
queue=[0,1]. Pop → advance to advances_to[0] = Goblin Knight
instead of advances_to[1] = Goblin Pillager. cmd[1098]: replay
expects weapon idx 1 (Pillager has 3 attacks), but Goblin Knight
has only 1 attack → `weapon_oob`.

Fix: pop one entry from `_advance_choices` in `_maybe_advance_unit`'s
AMLA branch, even though we don't use the value (AMLA has no
advances_to to index into).

### Move-path ambush truncation: replay records FULL planned path

`wesnoth_src/src/actions/move.cpp` (`move_unit_internal`): when a
unit's planned path crosses a hex held by an enemy that the moving
side couldn't see (fog), the engine STOPS the unit at the hex
BEFORE the enemy and zeros remaining MP. The replay records the
FULL planned path as `[move] x="..." y="..."`, but the engine
truncated it during play.

Reconstruction implications:

  - `_apply_command` for "move" must walk the path step-by-step
    and stop at the first enemy-occupied hex. If it just teleports
    to the final hex, two units may overlap (the stationary enemy
    and the truncated mover at a clear hex).
  - The truncation point must itself be empty. If a friendly unit
    is on the truncation hex (path was planned through friendlies,
    which is legal per pathfind.cpp:779-786), back off to the
    previous hex.
  - Set `current_moves = 0` on truncation -- **superseded**: only an AMBUSH zeroes MP; a move blocked by a unit KEEPS its remaining MP (see the blocked-versus-ambush entry earlier in this file and `tools/replay_dataset.py`'s truncation handling) -- a fog-ambushed unit
    has no MP left.

  - `diff_replay`'s precondition check should NOT flag mid-path
    enemies as a divergence -- in real Wesnoth this is a
    legitimate fog ambush. Without per-side fog tracking we can't
    distinguish "fog ambush" from "stale state cascade", so we
    accept all and let the truncation mirror Wesnoth's behavior.
    Cascades surface later as `final_occupied` / `src_missing`.

For non-strict-sync replays the `[mp_checkup]` block carries no
per-step truncation data, so this ambush rule must run at apply
time, not at extract time.

### `[checkup]` verification: empty blocks record, filled blocks verify

Replay verification is driven by the `[checkup]` child of each
synced `[command]`. `synced_checkup::local_checkup`
(src/synced_checkup.cpp, 1.18.4) compares only when stored data
exists; otherwise it APPENDS and returns true:

```cpp
if(buffer_.child_count("result") > pos_)
{
    real_data = buffer_.mandatory_child("result",pos_);
    pos_++;
    return real_data == expected_data;
}
else
{
    assert(buffer_.child_count("result") == pos_);
    buffer_.add_child("result", expected_data);
    pos_++;
    return true;
}
```

So an exported save with EMPTY `[checkup]` blocks plays back
without any verification (recorder mode), while one carrying
`[result]` children makes playback hard-compare every entry. The
buffer binds to the **last real command's plain `[checkup]` child**
(`set_scontext_synced::generate_checkup`,
src/synced_context.cpp: `resources::recorder->
get_last_real_command().child_or_add(tagname)` with tagname
`"checkup"`; end-of-turn contexts use the numbered variant, e.g.
`[checkup1]`).

**Why non-obvious:** the same call site switches to
`mp_debug_checkup` when `resources::classification->oos_debug` is
set — THAT variant writes the per-result dependent
`[command][mp_checkup]` blocks seen in strict-sync replays (our
731/731 oracle's data source). The inline-`[checkup]` form and the
dependent-`[mp_checkup]` form carry the SAME payloads in different
wire shapes; which one a save uses is decided by the
`[game_classification]` `oos_debug` flag, not by preferences at
playback time.

### Attack checkup payload: two `[result]`s per strike, nominal damage

`attack::perform_hit` (src/actions/attack.cpp, 1.18.4) runs TWO
`local_checkup` calls per strike attempt — misses included:

```cpp
int damage = 0;
if(hits) {
    damage = attacker.damage_;
    resources::gamedata->get_variable("damage_inflicted") = damage;
}

const config local_results {"chance", attacker.cth_, "hits", hits,
                            "damage", damage};
```

then after applying the strike:

```cpp
equals_replay = checkup_instance->local_checkup(config{"dies", dies},
                                                 replay_results);
```

`damage` is the NOMINAL post-modifier per-strike value
(`attacker.damage_`), NOT clamped by the target's remaining hp —
the clamp happens later (`damage_done = std::min<int>(
defender.get_unit().hitpoints(), attacker.damage_)`). Bools
serialize as `yes`/`no`. On mismatch the engine logs an errbuf
line starting `SYNC:` and overrides its calculation with the data
source's result (so playback visually follows the recording while
flagging the divergence).

Verified empirically against tests/fixtures/strict_sync_hamlets_t9
.bz2: every strike shows `{chance, damage, hits}` then `{dies}`,
e.g. `chance=40 damage=0 hits=no` / `dies=no` for a miss.

Sim-side: `combat.resolve_attack` records exactly these payloads
(`CombatResult.checkup_strikes`); `sim_to_replay` emits them into
exported `[attack]` commands so any manual playback verifies the
sim's combat bit-for-bit. `tools/playback_verdict.py` scans the
session log for the failure markers afterward.

### Petrified scenery via `random_traits=no` in `[side]`

The mainline 2p maps with petrified statues (Caves of the Basilisk,
Sullas Ruins, Thousand Stings Garrison) carry `random_traits=no` on
their statue side blocks. This is the only place `random_traits=no`
appears in 1.18.4 mainline — it's NOT used to disable random traits
on player leaders. Player leaders get filtered via the
`!can_recruit()` check above.

---

## Common pitfalls

### Pitfall 1: `changelog.md` is HISTORICAL — verify against current src

Old changelog entries describe behavior at THAT version, which may
have been changed since. The 1.3.7 entry "Leaders can't get random
traits yet, because it breaks MP" misled an exploration session
because the rule's mechanism changed (and the rule itself stayed
true via the `!can_recruit()` filter). When you find a relevant
changelog entry, always cross-check with the current source code
before quoting it as authority.

### Pitfall 2: Wesnoth wiki sometimes lags

The wiki documents intent; the source documents behavior. They
disagree on edge cases. When they conflict, source wins. Cite
file:line, not wiki URLs, in our code comments.

### Pitfall 3: rules can live in C++, Lua, OR WML — search all three

Common gotcha: a rule we're hunting in `src/` is actually in
`data/multiplayer/eras.lua` (e.g. `quick_4mp_leaders`) or in a WML
macro (`data/core/macros/multiplayer.cfg`). Workflow:

1. Grep `wesnoth_src/src/` for the C++ side of the rule.
2. Grep `wesnoth_src/data/` for the WML/Lua side.
3. If a rule has a "post-pass" feel (applied after unit setup), check
   for `[event]name=prestart` and Lua callbacks.

### Pitfall 4: terrain-key tables don't include all overlays

Our `_DEFENSE_KEYS_FOR_CODE` and `_OVERLAY_DEFENSE_KEYS` tables in
`tools/replay_dataset.py` are hand-rolled and miss overlays. When
you find a terrain-cost mismatch, check the actual terrain.cfg
entry (mvt_alias, aliasof, default_base) before adding ad-hoc
patches.

### Pitfall 5: replay [scenario] block doesn't carry leader traits

Real replay files describe leaders as `type=` attributes on `[side]`,
NOT as `[unit]` children. The leader's actual traits aren't preserved
in the file — Wesnoth re-rolls them at scenario start using the
recorded `[random_seed]` blocks. So you can't directly read traits
from a replay's [scenario] block. To check what traits a real
leader had, look at the rolled-trait outcome through gameplay
(e.g. observed MP suggests `quick` was rolled).

### Pitfall 6: 78% of `replays_raw/` carry mods that change game rules

A scan of `replays_raw/` (2026-05-03, ~207k replays) found 161,669
files (77.8%) with at least one `[modification] addon_id="..."`
block. Most-common mods that change combat / XP / recruit math:

  - `plan_unit_advance` (126k) — UI-then-gameplay: lets players
    pre-pick advancement options that fire even when the unit
    levels up off-turn. **Set-aside**, not deletable: 1.18-stock
    feature, eventually addressable.
  - `Rav_XP_Mod` (66k), `XP_Modification` (11k), `XP_Bank_Mod` (8k) —
    modify XP requirements / banking. Combat XP math diverges.
  - `RandomRecruits` (26k) — randomizes recruit list. Recruit
    decisions diverge.
  - `Ageless_Era`, `Ladder_Era`, `War_of_Legends`, `Reign_of_the_Lords`,
    `LotI_Era` — alternate eras. Different units, factions,
    abilities. Cannot reconstruct without scraping their unit DB.
  - `Biased_RNG_in_MP` (772) — smooths combat hit/miss to the
    expected value. With same seed, vanilla MTRng gives different
    per-strike hits, so combat trajectories diverge from turn 1.
    **Most poisoned cases of post-combat-bit-exact divergence.**

**Truly cosmetic** (safe to keep):
  - `Color_Modification`, `Rav_Color_Mod` — player team color only.
  - `Bloody_Mod_PSR` — blood splash overlay.

**Triage tool**: `tools/purge_mod_replays.py`. Three buckets:
keep / set_aside / purge. Run with `--apply` to actually delete +
move + reconcile `index.jsonl`. Dry-run by default.

The mod blind-spot does NOT affect self-play training — that runs
vanilla→vanilla. It only hurts our ability to score against real-game
corpora (diff_replay clean rate). After purging, the corpus is 22%
of original size but uniformly vanilla.

---

## File map (where to look first)

The Wesnoth source tree is large. Here's where each kind of question
tends to land. The C++ paths below are written `wesnoth_src/src/...`,
from when the tree was a full checkout; read them at the 1.18.4 tag on
GitHub (`src/actions/attack.cpp` for `wesnoth_src/src/actions/attack.cpp`).
The WML and Lua paths are local.

### Combat / damage / hits / rolls

- `wesnoth_src/src/actions/attack.cpp` — battle_context, attack_unit,
  attack_unit_and_advance, damage rolls
- `wesnoth_src/src/synced_commands.cpp` — replay [attack] handler
- `wesnoth_src/src/random_synced.cpp` — synced RNG that drives rolls

### Movement / pathfinding

- `wesnoth_src/src/actions/move.cpp` — unit_mover, plot_turn,
  try_actual_movement (ZoC stop, ambush stop)
- `wesnoth_src/src/pathfind/pathfind.cpp` — pathfinder (multi-turn);
  enemy = block, friend = pass with subcost
- `wesnoth_src/src/movetype.cpp` — movement_cost / defense_modifier
  resolution from terrain alias lists (MIN/MAX rule)

### Units / traits / advancement

- `wesnoth_src/src/units/unit.cpp` — init, advance_to, generate_traits,
  upkeep
- `wesnoth_src/src/units/unit.hpp` — incapacitated, get_emit_zoc,
  state queries (poisoned, slowed, petrified)
- `wesnoth_src/src/units/types.cpp` — unit_type config + traits()

### Recruit / recall / leaders

- `wesnoth_src/src/actions/create.cpp` — find_recruit_location,
  place_recruit, recruit_unit
- `wesnoth_src/src/actions/unit_creator.cpp` — add_unit (used by
  team_builder for both leader placement and [unit] tag spawning)
- `wesnoth_src/src/teambuilder.cpp` — team_builder stages 1/2/3,
  handle_leader, place_units
- `wesnoth_src/src/team.cpp` — team::attributes (the side cfg
  attributes that get stripped from leader cfg)
- `wesnoth_src/src/team.hpp` — spend_gold, recall_list

### Game / scenario / lobby setup

- `wesnoth_src/src/game_state.cpp` — set_random_determinstic block,
  team_builder driving
- `wesnoth_src/src/play_controller.cpp` — do_init_side (income/
  upkeep/MP reset)
- `wesnoth_src/src/game_initialization/connect_engine.cpp` —
  side_engine::new_config builds the side cfg from lobby choices
- `wesnoth_src/src/game_initialization/flg_manager.cpp` —
  faction/leader/gender resolution from era choices

### Era / multiplayer rules

- `wesnoth_src/data/multiplayer/eras.lua` — quick_4mp_leaders,
  turns_over_advantage
- `wesnoth_src/data/core/macros/multiplayer.cfg` — QUICK_4MP_LEADERS,
  TURNS_OVER_ADVANTAGE, ERA_DEFAULT, ERA_HEROES
- `wesnoth_src/data/multiplayer/factions/*-default.cfg` — per-faction
  leader pool, recruit list, terrain_liked

### Terrain

- `wesnoth_src/data/core/terrain.cfg` — every terrain code's `aliasof`,
  `mvt_alias`, `default_base`
- `wesnoth_src/data/core/terrain-graphics.cfg` — visual only
- `wesnoth_src/src/terrain/terrain.cpp` — alias parsing
- `wesnoth_src/src/terrain/translation.cpp` — `MINUS`/`PLUS` markers,
  string ↔ ter_list conversion

### Traits

- `wesnoth_src/data/core/macros/traits.cfg` — every trait definition
  (TRAIT_QUICK, TRAIT_STRONG, etc.)

---

## Search recipes

When hunting for a rule, these grep patterns hit faster than ad-hoc
exploration. All assume `wesnoth_src/` is the root.

### "Is rule X enforced in C++ or somewhere else?"

```bash
grep -rln "<keyword>" wesnoth_src/src/ wesnoth_src/data/multiplayer/ \
  wesnoth_src/data/core/macros/ wesnoth_src/data/lua/
```

If matches in `data/multiplayer/eras.lua` or `data/core/macros/`,
the rule is at least partly in WML/Lua. Don't stop at the C++ search.

### "Where does [command-tag] get handled at replay time?"

```bash
grep -n "SYNCED_COMMAND_HANDLER_FUNCTION" wesnoth_src/src/synced_commands.cpp
```

Each handler is registered with `SYNCED_COMMAND_HANDLER_FUNCTION(name,
child, ...)`. That's where the replay-execution rules for
`[recruit]`, `[move]`, `[attack]`, `[recall]`, etc. live.

### "What's the default for WML attribute X?"

```bash
grep -rn '"X"' wesnoth_src/src/ | grep "to_bool\|to_int\|str()"
```

Look for the `cfg["X"].to_bool(true)` / `to_int(0)` / etc. calls —
the argument is the default.

`wesnoth_src/data/schema/units/single.cfg` also has `DEFAULT_KEY`
entries that document the schema-level default.

### "Which `[multiplayer]` macros wrap default-era rules?"

```bash
grep -B 5 -A 30 "^#define ERA_DEFAULT$" wesnoth_src/data/core/macros/multiplayer.cfg
```

That macro lists every faction file plus rule-injecting macros
(`QUICK_4MP_LEADERS`, `TURNS_OVER_ADVANTAGE`).

### "Where are [side] cfg attributes stripped/added?"

```bash
grep -n "team::attributes" wesnoth_src/src/team.cpp
grep -n "stored.remove_attribute\|new_config\(" wesnoth_src/src/teambuilder.cpp \
  wesnoth_src/src/game_initialization/connect_engine.cpp
```

### "Is X applied at scenario start?"

```bash
grep -rn 'name="prestart"\|name=prestart' wesnoth_src/data/multiplayer/ \
  wesnoth_src/data/core/macros/
```

prestart events fire AFTER team_builder and BEFORE turn-1 play.
That's where post-init unit modifications (like `quick_4mp_leaders`)
typically live.

---

### Keeps are defined by `recruit_from=yes`, NOT by the terrain's name or its `K` prefix

`wesnoth_src/data/core/terrain.cfg` — the two mechanics are separate keys:

```
[terrain_type]                     [terrain_type]
    id=orcish_fort                     id=orcish_keep
    name= _ "Castle"                   name= _ "Keep"
    string=Co                          string=Ko
    aliasof=Ct                         aliasof=Ct
    recruit_onto=yes                   recruit_from=yes
[/terrain_type]                        recruit_onto=yes
                                   [/terrain_type]
```

- **`recruit_from=yes`** = a leader standing here MAY recruit (this is
  what "keep" means mechanically).
- **`recruit_onto=yes`** = recruited units may APPEAR here (castle).
- **Every keep also has `recruit_onto=yes`**, so keeps are valid recruit
  TARGETS as well as recruit sources.

**Why non-obvious:** the display `name=` of `Co` is literally `"Castle"`
and of `Ko` is `"Keep"`, but a name is not a mechanic — and neither block
contains a `castle=`/`keep=` key, so grepping for those finds nothing.

**Audited 2026-07-31 across all 284 `[terrain_type]` blocks:** our sim's
substring test (`"K" in base or "K" in overlay` → KEEP) reproduces
`recruit_from=yes` with **zero false positives and zero false negatives**.
The 24 engine keeps are `Kd Kdr Ke Kea Ker Ket Kf Kfa Kfr Kh Kha Khr Khs
Khw Km Kme Ko Koa Kte Kud Kv Kva Kvr ^Kov` — note **`^Kov` is an OVERLAY**,
so any keep test that only inspects the base code is wrong (same class as
the `^Xo` overlay bug, `a21030c`).

### Leaders are placed on the map's start marker — the engine does NOT snap them to a keep

When a `[side]` has no `x=`/`y=`, the leader goes to the map's
starting-position marker, full stop. `src/game_state.cpp`
`place_sides_in_preferred_locations()` assigns start positions
(`board_.map().set_starting_position(i->side, i->pos)`) with **no
relocation and no keep-seeking**; the wiki states the leader "is placed on
the tile represented by this number according to the map's starting
positions". `find_vacant_tile` exists only for OCCUPIED hexes, not
non-keep ones.

**Consequence (verified 2026-07-31): 5 of 29 shipped 2p maps start a
leader OFF a keep** — `2p_Tombs_of_Kesorak` both sides on `Co`,
`2p_Hornshark_Island` both sides on `Ce`, `2p_Sablestone_Delta` side 1 on
`Gs^Ft` (grass+forest, not even a castle; side 2 starts on `Kud`). Those
leaders cannot recruit until they walk to a keep — **this is correct
Wesnoth behaviour, not a sim defect.**

**Why non-obvious:** a turn-1-start snapshot shows those sides with ZERO
legal recruit actions, which reads like a mask bug. It is not: recruiting
costs no MP, so the leader may move onto a keep and recruit **in the same
turn**, and the mask recomputes `leader_on_keep` from the leader's current
position on every decision. Measure recruit availability across the turn,
never from the turn-start state alone.

---

## Verification protocol

When establishing a new rule, follow this protocol so the entry
holds up:

1. **Find the enforcing code path.** Don't stop at "the wiki says X"
   or "the changelog mentions X". Locate the actual `if (...) ...`
   or filter in source.

2. **Verify with a controlled test.** Pick a real replay (or
   construct a synthetic case) where the rule must apply. Check
   that our sim's behavior matches Wesnoth's recorded outcome.
   `tools/diff_replay.py` is the standing oracle for this.

3. **Quote the smallest snippet that proves it.** Don't paste 50
   lines if 5 will do. Future-you needs to grep the quote, not
   skim a wall.

4. **Cite file:line.** Wesnoth source moves between releases;
   pin to the version: C++ as `src/<path>:<line>` at the 1.18.4 tag
   on GitHub, WML and Lua as `wesnoth_src/data/<path>:<line>`
   (`wesnoth_src/` is a copy of the local 1.18.7 install's data tree;
   CLAUDE.md, "Wesnoth data provenance").

5. **Note non-obvious paths.** If the rule lives in Lua but the
   superficial search would land on C++, write that down.

6. **Update entries when we discover errors.** Don't add a
   contradicting entry. Find the earlier one and edit it, with
   a brief note on what changed and when.

---

## Combat-outcome prediction (the in-game damage calculator)

**Rule:** Wesnoth's attack-prediction oracle computes EXACT joint
HP distributions by sparse dynamic programming, with four
approximations: (a) probabilities within 1e-9 of 0/1 are snapped,
(b) extra berserk rounds stop once >= 99% of probability mass has a
dead combatant, (c) above a complexity threshold of 50,000 it
abandons exactness for Monte-Carlo simulation with 5,000 sampled
fights, (d) in a fight of at most one strike a side, a level-up that
needs a kill is scored as if the unit's HP were independent of the
kill (Combat, "The defender's counter weapon is chosen on a
prediction that counts level-ups", added 2026-09-25). (Researched
2026-06-12 from the GitHub 1.18.4 tag; local wesnoth_src/src/ was
lost in the machine move.)

- `src/attack_prediction.cpp` — `prob_matrix`: a sparse 2D matrix
  of (A_hp, B_hp) probabilities across FOUR planes
  (NEITHER_SLOWED / A_SLOWED / B_SLOWED / BOTH_SLOWED). Each strike
  shifts probability mass between cells (`shift_cols`/`shift_rows`
  take `drain_constant`/`drain_percent`; slow procs move mass
  between planes).

- Berserk round loop, `complex_fight()`:

```cpp
} while(--rounds && pm->dead_prob() < 0.99);
```

- Tiny-probability snapping, `round_prob_if_close_to_sure()`:

```cpp
if(prob < 1.0e-9) {
    prob = 0.0;
} else if(prob > 1.0 - 1.0e-9) {
    prob = 1.0;
}
```

- Complexity switch (`src/attack_prediction.hpp`):

```cpp
static const unsigned int MONTE_CARLO_SIMULATION_THRESHOLD = 50000u;
```

  with `fight_complexity()` =

```cpp
return num_slices * opp_num_slices * ((stats.slows || opp_stats.is_slowed) ? 2 : 1)
       * ((opp_stats.slows || stats.is_slowed) ? 2 : 1) * stats.max_hp * opp_stats.max_hp;
```

  (`num_slices` = swarm HP-threshold slices; 1 without swarm.) The
  Monte-Carlo path (`monte_carlo_combat_matrix`) uses
  `static const unsigned int NUM_ITERATIONS = 5000u;`.

- Output per combatant: `hp_dist` (full HP distribution) plus
  `untouched` / `poisoned` / `slowed` marginals
  (`src/attack_prediction.hpp`, `struct combatant`).

**Why non-obvious:** the calculator is commonly assumed to be
fully exact; it is exact ONLY below the complexity threshold, and
berserk fights are truncated by the 99%-dead-mass rule — the
residual <1% "both still alive after N rounds" mass is simply left
in place, which is the "ignores extremely low probability
outcomes" behavior visible in-game.

## Turn-1 init_side: WRONG, superseded -- see the init_side entry above

**Do not port from this section.** It read `turn() > 1` as gating the
WHOLE per-side refresh, so it claimed turn 1 refreshes nothing at all
for every side. Healing is gated separately, by `do_healing()`, which
is false only for the game's VERY FIRST side-init and true from the
second onward -- including the later sides of turn 1.

The correct rule, with the verbatim source, is
"init_side: healing and MP-refresh/income sit behind DIFFERENT gates"
earlier in this file. Porting from here instead re-introduces the bug
that entry was written to fix: Micro Isar's tentacle regenerated to
11 HP in the sim against the engine's 19.

Kept as a decision record rather than deleted, because the wrong
reading is the natural one and someone will derive it again.

## [capture_village] = set_owner per matched hex

`data/lua/wml-tags.lua:444-461` (1.18.4):

```lua
function wml_actions.capture_village(cfg)
	local side = cfg.side
	...
	local locs = wesnoth.map.find(cfg)
	for i, loc in ipairs(locs) do
		wesnoth.map.set_owner(loc[1], loc[2], side, fire_event)
	end
end
```

Absent `side=`, ownership is cleared (village becomes neutral).
Used at prestart by add-on maps (WL Cold War / Summer Frosts) for
asymmetric starting villages.

**`set_owner` moves the village between the sides' village SETS, and
a side's village count is that set's size (corrected 2026-09-26).**
`src/scripting/game_lua_kernel.cpp:1142-1193` (1.18.4,
`intf_set_village_owner`), abridged:
```cpp
	map_location loc = luaW_checklocation(L, 1);
	if(!board().map().is_village(loc)) {
		return 0;
	}

	const int old_side_num = board().village_owner(loc);
	const int new_side_num = lua_isnoneornil(L, 2) ? 0 : luaL_checkinteger(L, 2);
	...
	if(old_side_num == new_side_num) {
		return 0;
	}
	...
	// The new side was valid, but already defeated. Do nothing.
	if(new_side && board().team_is_defeated(*new_side)) {
		return 0;
	}
	...
	if(old_side) {
		old_side->lose_village(loc);
	}

	// If the new side was valid, re-assign the village.
	if(new_side) {
		new_side->get_village(loc, old_side_num, (luaW_toboolean(L, 3) ? &gamedata() : nullptr));
	}
```
and `team::get_village` / `team::lose_village`
(`src/team.cpp:437-468`) insert into and erase from `villages_`. A
location that is not a village is skipped, side 0 or none leaves the
village to nobody, a side counted as defeated (by default: no leader
left) gets nothing, and `fire_event=yes` fires `capture` events.

Sim: `_capture_village_action` (tools/scenario_events.py) hands each
matched village through `replay_dataset.set_village_owner`, the
transfer a move's capture uses, so the owner map and each side's
`nb_villages_controlled` (which income reads) move together;
`WesnothSim._assert_invariants` (e) checks they agree. It reads `side`,
`x`, `y` and `terrain`; any other filter key, and `fire_event`, is
reported as unmodelled; the defeated-side no-op is not modelled.

**Why non-obvious:** the Lua tag looks like an owner assignment, and
the first handler wrote only `_village_owner`. Our count is a second
record of the same fact, so the handler left WL Cold War starting at
counts 1 and 1 with 2 and 3 villages owned, and WL Summer Frosts at 1
and 1 with 1 and 2: every turn paid (and supported upkeep for) one
village too few on Cold War's side 1, two on its side 2, and one on
Summer Frosts' side 2. Neither map is in the pools or the corpus.

## [modify_unit] moves= writes CURRENT MP, not max

`data/lua/wml/modify_unit.lua:14-17,41` (1.18.4): every scalar
attribute passes verbatim into the stored unit WML
(`wml.variables[unit_path.key] = value`) and the unit is re-created
via [unstore_unit]. In unit WML, `moves` is the remaining-MP-this-
turn attribute; `max_moves` is the separate stat. So `[modify_unit]
moves=N` changes only the current turn's movement allowance. Sim:
`_MODIFY_UNIT_SCALARS` maps moves->current_moves only.

## Enemy side statistics under fog or shroud (added 2026-09-08)

**Rule:** a player never sees an enemy side's gold, village count, unit
count, upkeep or income while fog or shroud is on. The status table
fills those columns only for sides the viewing team "knows about",
and an enemy is never known under fog/shroud, whether or not its units
have been seen.

**Source (1.18.4):** `src/gui/dialogs/game_stats.cpp:90`
`const bool known = viewing_team_.knows_about_team(team.side() - 1);`
and `:139` `if(known || see_all) {` around the gold / villages
(`:148` `std::to_string(team.villages().size())`) / units / upkeep /
income columns; `src/team.cpp:704-716`:

    bool team::knows_about_team(std::size_t index) const
    {
        ...
        // If we aren't using shroud or fog, then we know about everyone
        if(!uses_shroud() && !uses_fog()) { return true; }
        // We don't know about enemies
        if(is_enemy(index + 1)) { return false; }

**Why non-obvious:** the encoder's global feature 5 was the enemy's
true village count on every path (found by the 2026-09-08 contamination
review); `wesnoth_ai/visibility.enemy_villages_visible_to` counts the
enemy villages on hexes the mover sees instead, behind the
checkpoint flag `fog_hides_enemy_villages` (the seed was trained with
the true count). With fog off every side's statistics are visible, so
the count is legitimate there.

## The preprocessor's macro grammar: `:` in names, `#arg` defaults (added 2026-09-22)

Two features of Wesnoth's preprocessor that a naive `{NAME args}`
reader gets wrong, both found by diffing our expansion against the
engine's own (`tools/analysis/expansion_diff.py`).

**A macro name may contain `:`.** The core macros use it as a
namespace separator for definitions that are not meant to be called
from scenarios:

**Source (1.18.7 data):** `data/core/macros/abilities.cfg:4`

    #define INTERNAL:ABILITY_HEALS_NO_NOTES

and `data/core/macros/special-notes.cfg:8`
`#define INTERNAL:SPECIAL_NOTES_SPIRIT`. They are invoked by name
including the colon, e.g. `abilities.cfg:373`
`special_note={INTERNAL:SPECIAL_NOTES_SUBMERGE}`.

**Why non-obvious:** a `\w+` name class stops at the colon, so every
such definition lands under the single name `INTERNAL` — each one
overwriting the last — and every invocation expands to nothing while
looking like an ordinary unknown macro.

**A macro may declare optional NAMED arguments with defaults.**

**Source (1.18.7 data):** `data/core/macros/traits.cfg:4-7`

    #define TRAIT_LOYAL
    #arg OVERLAY
        "misc/loyal-icon.png"
    #endarg

used at `:23` as `add={OVERLAY}` and overridden by the caller at `:28`
`{TRAIT_LOYAL OVERLAY="misc/hero-icon.png"}`.

**Why non-obvious:** the block sits INSIDE the macro body, between
`#define` and `#enddef`, so a reader that takes parameters only from
the `#define` line never learns the name exists. Worse, a comment
stripper that drops every line starting with `#` removes the `#arg`
and `#endarg` markers but leaves the default value behind as a bare
stray line in the body, and `{OVERLAY}` is never substituted.

Both are read by `tools/scenario_events.py`
(`_MACRO_DEFINE_RE`, `_MACRO_INVOKE_RE`, `_split_optional_args`).

## `random_start_time` has three forms, not two (added 2026-09-22)

`random_start_time=` accepts `yes`/`no` AND a value list such as
`"2,4"`, meaning "draw one of these slots".

**Source (1.18.4):** `src/tod_manager.cpp` `resolve_random()`, whose
boolean branch draws once modulo the schedule length while the list
branch draws over the listed values.

**Why non-obvious:** a reader that coerces the attribute with a
yes/no parser folds the list form onto `False`, which reads as "no
random start" and silently begins the game at dawn. Our reconstruction
path did exactly this: the guard meant to drop such replays sat inside
the branch only a plain `yes` could enter. No corpus replay uses the
list form, so nothing had diverged; `tools/wml_state.wml_bool_or_none`
now distinguishes the third form and the caller drops it.

## Preprocessor conditionals, and what a multiplayer game defines (added 2026-09-23)

`#ifdef SYM` keeps its branch when SYM is defined, `#ifndef` negates,
`#else` flips the current branch, `#endif` closes it, and they nest.

**Source (1.18.4):** `src/serialization/preprocessor.cpp:1322-1331`

    } else if(command == "ifdef" || command == "ifndef") {
        const bool negate = command[2] == 'n';
        ...
        bool found = parent_.defines_->count(symbol) != 0;
        conditional_skip(negate ? found : !found);

and `:1377-1400` for `#else` (`Unexpected #else` outside a branch) and
`#endif` (`Unexpected #endif`).

**Why non-obvious:** `defines_` is ONE map holding both preprocessor
symbols and every macro defined with `#define` so far. So `#ifdef
SOME_MACRO` is a test for a macro's existence and turns true at the
line that defines it, and a `#define` inside a dead branch defines
nothing.

**What is defined in a multiplayer game:** `MULTIPLAYER` (the template
builder preprocesses with `--preprocess-defines=MULTIPLAYER`), plus the
scenario's own `define=` attribute, which the game adds when it LOADS
that scenario. Hornshark Island relies on the second: its
`#ifdef MULTIPLAYER_HORNSHARK_ISLAND_LOAD` blocks, which hold
`MODIFY_BOWMAN` and its prestart events, are live only because
`define=MULTIPLAYER_HORNSHARK_ISLAND_LOAD` sits in its own
`[multiplayer]` block (`data/multiplayer/scenarios/2p_Hornshark_Island.cfg:31`).
No difficulty symbol (`EASY`, `NORMAL`, `HARD`, `NIGHTMARE`) is
defined, so `{QUANTITY ...}` (`data/core/macros/utils.cfg:8`) expands
to nothing in multiplayer.

Implemented by `tools/scenario_events.evaluate_conditionals`.

## Vision and fog: what a side sees, and when it is recomputed (added 2026-09-24)

**Rule.**

1. **What one unit sees.** Every hex it could reach in one turn by
   spending its vision points at its vision costs, other units and zones
   of control ignored, plus every hex adjacent to one of those, on the
   board or off it. Vision points are the unit's `vision=` when its type
   declares one, else its maximum movement; vision costs are its
   `[vision_costs]` when declared, else its movement costs; a slowed
   unit pays double; a jamming enemy adds its jamming cost per hex. In
   the default era no unit type declares `vision=`, `[vision_costs]` or
   jamming, so a unit sees what it could reach with its full movement,
   plus the ring around it.
2. **What a side sees is its fog, and fog is state.** Hexes a side's
   units see are cleared, and they stay cleared until the side's fog is
   recalculated. Hexes are cleared by: the side's units, at its turn
   start; a moving unit, at every hex it enters; a recruit, at its hex;
   an advancing unit of either side, at its hex, whoever's turn it is.
   Recalculation (refog everything, then clear from every unit's
   current hex) happens at the side's turn start (after the turn-refresh
   events), at its turn end (after its units' slow expires and after the
   turn-end events), and, for the defending side, after a fight in which
   the defending unit died, was slowed or was petrified.
3. **So, during its own turn,** a side sees its turn-start vision plus
   everything its movers and recruits saw since; nothing is refogged
   during its own turn, not even where its own units died.

**Source (1.18.4).** `src/pathfind/pathfind.cpp:576-588`, the vision
area:

    vision_path::vision_path(const unit& viewer, const map_location& loc,
    ...
    	const int sight_range = viewer.vision();
    	// The three nullptr parameters indicate (in order):
    	// ignore units, ignore ZoC (no effect), and don't build a cost_map.
    	...
    	find_routes(loc, viewer.movement_type().get_vision(),
    	            viewer.get_state(unit::STATE_SLOWED), sight_range, sight_range,
    	            0, destinations, &edges, &viewer, nullptr, nullptr, &viewing_team, &jamming_map, nullptr, true);

`find_routes` collects the edges: off-board neighbours of every
collected hex (`:349-352`, `edges->insert(off_board_it, adj_locs.end());`)
and every neighbour it cannot enter (`:392-398`):

    			if ( next.moves_left < 0 || next.turns_left < 0 ) {
    				// Either can never enter this hex or out of turns.
    				if ( edges != nullptr )
    					edges->insert(next_hex);
    				continue;
    			}

with the cost of entering at `:378`, `int cost = costs.cost(map[next_hex], slowed);`
and the jamming added at `:379-384`. `src/actions/vision.cpp:354-369`
clears both sets (`for (const pathfind::paths::step &dest : sight.destinations)`
and `for (const map_location &dest : sight.edges)`). Vision points,
`src/units/unit.hpp:1415-1418`:

    	int vision() const
    	{
    		return vision_ < 0 ? max_movement_ : vision_;
    	}

and a `movement` effect moves vision with movement unless it says
`apply_to_vision=no` (`src/units/unit.cpp:2175-2198`). Vision costs fall
back to movement costs, `src/movetype.cpp:822`,
`vision_(cfg.child_or_empty("vision_costs"), mvj_params_, &movement_),`.
Slowed doubles, `src/movetype.hpp:69-72`,
`return  slowed  &&  result != movetype::UNREACHABLE ? 2 * result : result;`.

When fog is cleared and recalculated:
- Turn start, `src/play_controller.cpp:524-525`, after the turn-refresh
  events: `// Make sure vision is accurate.` /
  `actions::clear_shroud(current_side(), true);` -- the second argument
  is `reset_fog`, which ends in `recalculate_fog(side);`
  (`src/actions/vision.cpp:774-779`).
- Turn end, `src/play_controller.cpp:582-590`:
  `gamestate().board_.end_turn(current_side());` (which clears slow,
  `src/units/unit.cpp:1284` `set_state(STATE_SLOWED,false);`), then
  `// Clear shroud, in case units had been slowed for the turn.`, the
  turn-end events, and
  `// This is where we refog, after all of a side's events are done.` /
  `actions::recalculate_fog(current_side());`.
- `recalculate_fog`, `src/actions/vision.cpp:702-736`: `tm.refog();`
  (`:718`), then `clearer.clear_unit(u.get_location(), u, tm, &visible_locs);`
  for every unit of the side (`:724-728`).
- A move, `src/actions/move.cpp:972-976`, after each step:
  `// Update the fog.` / `if ( current_uses_fog_ )` /
  `handle_fog(*real_end_, new_animation);`, which clears around the
  entered hex (`:548-557`).
- A recruit, `src/actions/create.cpp:695-698`: `// Fog clearing.` ...
  `clearer.clear_unit(current_loc, *new_unit_itor);`.
- An advancement, `src/actions/advancement.cpp:397-399`:
  `// Update fog/shroud.` / `clearer.clear_unit(loc, *new_unit);`.
- A fight, `src/actions/attack.cpp`: in `perform_hit`,
  `bool& update_fog = attacker_turn ? update_def_fog_ : update_att_fog_;`
  (`:965`) is set when the struck unit dies (`:1151-1154`), is slowed
  (`:1174-1178`) or is petrified (`:1181-1184`), and
  `// update_att_fog_ is not used, other than making some code simpler.`
  (`:751`); at the end of the fight, `if(update_def_fog_) {` /
  `actions::recalculate_fog(defender_side);` (`:1456-1458`).

A move and a recruit clear immediately only while the player's "delay
shroud updates" preference is off (`current_uses_fog_(current_team_->fog_or_shroud() && current_team_->auto_shroud_updates())`,
`src/actions/move.cpp:371`, and `create.cpp:697`); that is the default,
and the simulator models it. A unit placed by WML (`[unit]`) or a
plague corpse clears nothing (`src/actions/unit_creator.cpp` and
`attack::unit_killed` have no clearing call).

**Why non-obvious.** Until 2026-09-24 the simulator drew a disc of
radius `max_moves` around each unit's current hex. That is wrong three
ways: it sees across terrain the unit could not cross (mountains,
water, cave walls), it misses the ring one hex beyond the reachable
area, and it forgets, after a unit moves, what the side saw earlier in
the turn. On 31,137 decisions of 92 fogged corpus games (three per
map), the engine's turn-accumulated view differs from the disc at
30,814; it holds 63.1 hexes per decision the disc lacks and lacks 26.4
the disc holds (522.9 against 486.1 hexes seen); of 309,711 enemy units
on the board at those decisions, 19,439 are shown only by the engine's
view and 3,341 only by the disc
(`tools/analysis/vision_rule_census.py`,
`training/metrics/fidelity/vision_rule_census_20260924.json`).

**Implemented by** `wesnoth_ai/visibility.py` (`unit_vision`, and the
fog each side has cleared on `global_info._fog_cleared`, kept by the
hooks in `tools/replay_dataset._apply_command`) and, for the Rust core,
`rust/wesnoth_core/src/core_fog.rs`; pinned by tests/test_vision.py.

**Not modelled.** `vision=` and `[vision_costs]` (declared by the Dune
Falconer, the Dune Sky Hunter, the Dragonfly and the Grand Dragonfly,
none of which is in the default era, the pool or the corpus; such a
unit logs a warning and sees with its movement), jamming, shared vision
between allies, and the delayed-shroud preference.
