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
- File path + line number where it's enforced (`wesnoth_src/...`)
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
- [Combat](#combat)
- [Units, traits, leaders](#units-traits-leaders)
- [Recruit and recall](#recruit-and-recall)
- [Replay structure](#replay-structure)
- [Scenario events](#scenario-events)
- [Common pitfalls](#common-pitfalls)
- [File map (where to look first)](#file-map-where-to-look-first)
- [Search recipes](#search-recipes)
- [Verification protocol](#verification-protocol)

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

### ZoC and incapacitation

`wesnoth_src/src/units/unit.hpp:1352-1355`:
```cpp
/** Tests whether the unit has a zone-of-control, considering @ref incapacitated. */
bool get_emit_zoc() const
{
    return emit_zoc_  && !incapacitated();
}
```

Petrified (`STATE_PETRIFIED`) → `incapacitated()` is true →
emits no ZoC. Also has `attacks_left() = 0` and `movement_left() = 0`
(unit.hpp:998 and 1299).

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

The synced engine doesn't enforce this (you can construct an
attack on a petrified target via direct WML), but Wesnoth's UI
gates the action so a player can never click-attack a statue.
Our legality mask should match the UI rule: petrified targets
excluded from attack mask.

### Attacking a petrified defender

`wesnoth_src/src/actions/attack.cpp` and `unit.hpp:1352-1355`:
the petrified defender has `attacks_left()=0` (no counter-attack)
and the attacker's strikes proceed normally. Our combat resolver
must skip the counter-attack when defender is petrified (we already
do this in `tools/replay_dataset.py`).

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

`wesnoth_src/src/team.cpp:235-243`:
```cpp
const std::string& village_support = cfg["village_support"];
if(village_support.empty()) {
    support_per_village = game_config::village_support;  // default 1
} else {
    support_per_village = lexical_cast_default<int>(...);
}
```

`village_gold=` is on the `[side]` block, not global. In MP, the
host's game-options dialog sets it identically across all sides, but
`replay_extract.py` MUST capture it per-side -- our
`SideState.village_income` field reads `[side] village_gold` directly.
Default Era uses **5** gold per village (not the historic 1). The
host can also customize it. NOT reading the per-side value and
defaulting to a hardcoded 2 was the cause of a multi-replay diff
divergence (commit 2026-05-03, Den of Onis #14f7a7c1a17f).

Same for `village_support=` (default 1, mainline always 1 but capture
it for completeness — `SideState.village_support`).

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
  - Set `current_moves = 0` on truncation -- a fog-ambushed unit
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
tends to land:

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
   pin to the version. Our tree is `wesnoth_src/` at tag 1.18.4.

5. **Note non-obvious paths.** If the rule lives in Lua but the
   superficial search would land on C++, write that down.

6. **Update entries when we discover errors.** Don't add a
   contradicting entry. Find the earlier one and edit it, with
   a brief note on what changed and when.
