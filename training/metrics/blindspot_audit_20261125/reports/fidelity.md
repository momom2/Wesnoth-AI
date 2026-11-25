# Fidelity of the emitted columns (auditor: "truth of the columns that exist")

Scope: every column the encoder emits under `observation_parity` with unit row
version 2 (global row 15, hex dynamic 5 + static 3 + terrain mask, unit and
recruit rows 341), checked against the Rust that fills it and against Wesnoth
1.18.4 (`src/actions/attack.cpp`, `src/units/abilities.cpp`, `src/reports.cpp`
fetched from the 1.18.4 tag; WML from `wesnoth_src/data/`).

Method: reading only, plus one standalone script (no project import, `python -I`)
that re-implements `rust/wesnoth_core/src/terrain.rs` over `terrain_db.json` and
`unit_stats.json` and walks every terrain code of the 21 ladder maps against all
89 distinct (defence, movement) tables of the scrape:
`scratchpad/audit_20261125/fid/terr.py`.

Headline: I found no column whose number is wrong against the engine. The fight
lines follow `battle_context_unit_stats` and `choose_defender_weapon` step by
step. The findings are about what the columns cannot express, about hidden units
reaching a few columns exactly as they reach the engine's own displays, and
about stale comments and thin tests.

---

## 1. Findings (most serious first)

### F1. Per-class chance-to-be-hit drops the defence caps, so mixed hexes do not follow from the columns for mounted units and fliers (low to moderate)

- **What the player sees.** Hovering a hex with a unit selected shows the unit's
  defence there. The sidebar tooltip breaks a mixed terrain into its parts, each
  marked "minimum" or "maximum" (`src/reports.cpp:577-616`, `unit_defense`:
  `underlying_def_terrain` with `(revert ? _("maximum^max.") : _("minimum^min."))`).
  The number shown is `defense_modifier(terrain)`, cap included.
- **What the network gets.**
  - Cols 111-126 hold `def_pct` of each class's archetype
    (`core_parity.rs:184-193`, `core.rs:382-385`). `terminal_cost` takes `.abs()`
    of a negative (cap) entry (`terrain.rs:123`), so a cap reads as an ordinary
    value: Horseman forest = 0.70, flat = 0.60.
  - The hex token carries the class set only (`terrain_mask_parity`,
    `terrain.rs:307-337`). It merges the movement and defence alias lists and
    drops the +/- (best/worst) markers.
  - The only exact defence is col 110, "where it stands" (`defense_at`,
    `core_attack.rs:168-174`), and only for the unit's current hex.
- **Measured (terr.py).**
  - Codes where the true chance to be hit differs from "best of the mask's
    classes" for some unit, cap-driven only:
    - every Flat+Forest and Hills+Forest code (`Gs^Fp` 437 cells, `Gs^Fms` 113,
      `Hh^Fms` 69, `Hh^Fp` 52, and about 30 more) for the `mounted` and
      `dunearmoredhorse` movetypes: Horseman, Knight, Lancer, Cavalryman,
      Dragoon, Paladin, Grand Knight, Dune Cataphract and similar. Truth 70,
      best-of 60 (50 on hills).
    - every Flat+Fungus code (`Gg^Tf` 92, `Re^Tf` 76, ...) for the fliers with
      `fungus=-70` (Gryphon Rider, Gryphon Master) and the mounted units
      (`fungus=-80`). Truth 70, best-of 50 or 60.
  - Every other deviation from best-of / worst-of is a fixed property of the
    mask. Villages, bridges, fords, castles on water and the like are best-of
    for movement too.
  - The mask determines the rule on these maps. The only codes sharing a mask
    with different values are `Mm^Xm`/`Md^Xm`/`Ms^Xm` against `Mm^Xo`, and those
    differ in defence only, on impassable hexes. So "best or worst" is learnable
    from the mask's identity, but the cap is learnable only through the unit's
    type embedding.
- **Why it matters.** A Knight choosing between `Gs^Fp` and open grass reads
  60 / 60 from the columns. In fact it is 70 vs 60 (30% against 40% defence). The
  network can only memorise this per type id, and the exact number appears only
  after the unit stands there.
- **Fix.** Within the rulings, one option is a per-class sign bit or "cap" column.
  Another is showing the class value as the engine composes it.
- **Confidence:** verified by reading (`terrain.rs:120-196`, `core.rs:376-386`) and
  by the script. The engine side (cap semantics) is taken from
  `def_pct`/`collect_neg_caps`, which the project's parity tests cover; I did not
  re-read `movetype.cpp`.
- **Run to confirm:** see "To check by a run" #2.

### F2. Hidden units reach four columns, as they reach the engine's own displays; one Python comment says otherwise (low; record it)

- **Where hidden units count.** In each place below, every unit counts, seen or
  not:
  - col 107, a unit's time-of-day modifier: `illuminated` scans every unit
    (`core_attack.rs:200-207`).
  - col 108 of a visible enemy: leadership from an adjacent enemy leader that
    stands in fog (`core_attack.rs:179-196`).
  - hex col 4, a seen hex's lawful bonus: illumination of every unit on or next
    to it (`core_parity.rs:349-377`; its doc says "seen or not").
  - the fight lines' answer damage: `d_leadership_bonus` and `d_lawful_bonus`
    (`core_attack.rs:249-252`).
- **The engine does the same.**
  - `under_leadership` / `get_abilities` in `src/units/abilities.cpp` (1.18.4)
    has no visibility test; a grep for visible, fogged or invisible finds only a
    comment at :166.
  - The sidebar weapon report uses `get_visible_time_of_day_at`, which on a seen
    hex calls `get_illuminated_time_of_day(rc.units(), ...)` with all units
    (`reports.cpp:100-112`, `:791-795`).
  - So these columns hold what the player's screen holds. This is not a leak
    beyond the player, but it is an exception to Principle 6's "no hidden unit
    contributes", and it is not written down.
- **Comment disagreement.** `encoder.py:310-312` describes hex col 4 as "the
  illumination of the units the side sees". The Rust (`core_parity.rs:346-348`,
  `352-363`) uses every unit, seen or not, which is what the engine does. The
  Python comment is wrong.
- **Example.** An enemy Lieutenant two hexes behind a fogged ridge, next to the
  Grunt our Spearman faces. The answer's damage in our line includes the +25%.
  A human sees the same number in the attack dialog.
- **Confidence:** verified by reading (Rust and 1.18.4 C++).

### F3. No column is clipped anywhere; the normalisation comments are stale (low)

- **No clamp exists.** There is no clamp on any feature in `encode.rs:108-177`,
  `core_parity.rs:145-216`, `core_encode.rs`, `game_core.py:907-921`, or
  `model.py` (its two `clamp_min`s are on value variances, `model.py:437`,
  `:1191`). Divisors only scale, so nothing is blind above a divisor.
- **Values above 1 that occur**, over the 238 types reachable from the
  default-era recruits and leaders plus plague corpses (scrape read by script):
  - cost: Elvish Sylph 161 → 2.0; Great Mage 110 → 1.4 (COST_NORM 80);
  - max HP: Drake Enforcer 85, Grand Knight 84, plus resilient, AMLA and
    feeding → above 1.06 (HP_NORM 80);
  - moves: Elvish Outrider 11, plus quick → 1.2 (MOVES_NORM 10);
  - fight damage per hit: above 40, for example a Dwarvish Dragonguard (40
    base) against a 150%-fire target, or a charging Grand Knight → about 1.5
    (WEAPON_DAMAGE_NORM 40).
  - Strikes peak at 6 (Inferno Drake), so the strikes divisor holds. Experience
    peaks at 200 × 0.7 = 140 < 150. Level peaks at 4, equal to LEVEL_NORM. The
    most weapons is 3, equal to the slot count.
- **Stale comments.** `constants.py:106-109` ("ladder Drake max =~70", "ladder
  caps around 7", "ladder caps around 60") are false for reachable types. The
  first global-feature block, `encoder.py:198-204`, is a stale 6-feature list
  next to the real one (`:205-213`).
- **Confidence:** verified by reading and by the script.

### F4. The fight-column tests leave most rules unexercised, and their played-game checks are circular (test gap, not a wrong number)

`tests/test_unit_rows2.py`:
- **What the hand-computed tests cover.** Night ToD, resistance, a single-weapon
  answer, the sword/bow choice by range, slow on the attacker, Lieutenant
  leadership (L2 → L1), a spent attack, a petrified enemy, and one fogged
  ambusher. All from side 1's point of view (`:193-269`).
- **Why the played-game tests are circular.**
  - `test_on_played_games_the_lines_are_the_fights_own` compares each line with
    `core.fight_stats` and `core.counter_weapon_choice` (`:388-401`). Those are
    the same functions `fight_line` calls (`outcomes.rs:567-575`).
  - `test_each_attack_played_strikes_as_its_line_said` compares with the
    simulator's own `checkup_strikes` (`:427`, `:440-476`).
  - Neither ties a line to Wesnoth. The tie rests on the combat-parity suite
    (`tests/test_combat_seed_alignment.py`, `tools/diff_replay.py`).
- **Never hand-checked against the engine formula:**
  - marksman (offence only, `max(60, cth)`); magical (70 on defence too,
    `cumulative=no` replaces the base even when the base is higher);
  - backstab with a flanker; charge doubling both sides; steadfast (the answer to
    our attack against a steadfast unit);
  - swarm at partial HP; drain; firststrike order with a firststrike defender;
    berserk;
  - a slowed defender's halved answer; illumination; a lawful unit at day;
    liminal;
  - a defender with two weapons of one range, where `better_combat` decides;
    traits (strong, dextrous, quick, resilient, intelligent) in the rows;
  - the "stands" column on a mixed hex: the played test checks one-class hexes
    only (`:372-375`); the side-2 point of view in the hand tests.
- **What I verified by reading instead.**
  - These rules are implemented as the engine does them
    (`combat.rs:354-420` against `attack.cpp:167-224`; Coverage below).
  - The defender's choice is the engine's: same-range candidates; a single one
    is taken unsimulated; otherwise the best by
    `better_combat(def, att, ..., harm 1.0)`. In `attack.cpp:617-636` the
    `min_rating` filter is inert (max_weight is assigned before the comparison),
    as `outcomes.rs:514-521` says.
  - A human defender cannot choose: `battle_context` with `defender_weapon == -1`
    always calls `choose_defender_weapon` (`attack.cpp:412-415`).

### F5. Global col 5 under fog depends on a checkpoint flag whose default is False on several paths (to confirm, not a finding yet)

- **The leak, if the flag is off.** `their_villages` is the true count unless
  `fog_hides_enemy_villages` (`core_encode.rs:428-433`). The engine never shows
  an enemy's village count under fog (`docs/wesnoth_rules.md:3116-3144`, citing
  `game_stats.cpp:90`, `:139`).
- **Defaults.** `transformer_policy.py:93` defaults to True. `game_core.py:854`
  and `:888`, `encoder.py:655`, `tools/encode_worker.py:120`,
  `tools/inference_seam.py:767` and `tools/policy_anchor.py:185` default to
  False.
- The live loop's value was not determinable by reading: "To check by a run" #1.

---

## 2. Coverage (checked, present and right)

### Global row (15)

- **0** turn / 60: `core_encode.rs:440-446`, `encode.rs:165`. Unclipped.
- **1** side to move ±1: `encode.rs:166`. Own/enemy are relative to `side` on
  every path (`opponent_of`, `core_encode.rs:42-49`, `:287`).
- **2** own gold / 500: `core_encode.rs:425`.
- **3** own base income (2 + the scenario's side income, team.hpp
  `base_income`): `core_encode.rs:426`, `scenario_pool.py:806`, `:817`.
- **4** own villages; **5** enemy villages, fog-gated (see F5):
  `core_encode.rs:427-433`.
- **6, 7** this turn's and next turn's board lawful bonus. The default cycle
  honours `tod_start_offset` (`core_step.rs:59`). Every ladder scenario uses
  DEFAULT_SCHEDULE or the SECOND_WATCH variant (grep of the 21 .cfg files).
- **8** village gold, **9** village support: `core_parity.rs:405-415`, from the
  scenario economy (`scenario_pool.py:789-799`).
- **10** own net income = base + villages × gold − max(0, upkeep − villages ×
  support). Upkeep is the sum of levels, leaders and loyal excluded. Matches the
  `team_data` income report (`core_parity.rs:385-399`, `reports.cpp:1489-1530`)
  and the turn-start payment (`core_step.rs:272-291`).
- **11** fog on. **12-14** the enemy's gold, net income and upkeep, 0 under fog
  as `game_stats.cpp` hides them: `core_parity.rs:416-421`.

### Hex token

- **Static village bit** on every village hex under parity: `core_encode.rs:351-353`.
- **Keep and castle bits**: `core_encode.rs:338-339`, from `parse_hex_code`
  (`terrain.rs:363-395`).
- **Terrain set**: 16 classes; mushroom grove and reef have their own
  (`terrain.rs:264-309`).
  - Every ladder code maps into the 16 classes or into impassable. The only
    aliases outside the 16 are `_off^_usr`, `^_fme` and `M*^Xm`, all impassable
    (script).
  - Rails (`Rt`) share flat, as the code says.
- **Dynamic 0** (recruit-rejected) is written 0 under parity, with the reason
  given: `core_encode.rs:370-381`.
- **Dynamic 1** village ours, always; **dynamic 2** village theirs, only on a
  seen hex (or fog off): `core_encode.rs:359-369`, `encode.rs:281-293`.
- **Col 3** seen now (1 with fog off): `core_parity.rs:368-369`.
- **Col 4** the hex's lawful bonus minus the board's: the area alone on a
  fogged hex; area + terrain light + illumination on a seen hex. Matches
  `get_visible_time_of_day_at` (`reports.cpp:100-112`). For the comment, see F2.
- **Fogged hexes**: no unit, no ZoC and no occupancy from a hidden enemy.
  Unit tokens are only `visible` units (`core_encode.rs:315`, `observe.rs:99-131`).
  `occupied`, `enemy` and `zoc` are built from visible units only
  (`observe.rs:139-166`). A fogged enemy village shows neutral. Terrain changes
  (events) update the mask and the movement classes
  (`events.rs:429-483`). Fog does not hide terrain in Wesnoth.

### Unit row, first 13 columns (`encode.rs:108-130`, inputs at `core_encode.rs:390-403`)

- **0, 1** max HP after traits, HP ratio.
  - Traits: strong +1, quick −5%, resilient +4 +1/level, healthy +1 +1/level
    (`units.rs:100-118`, `:213-261`).
  - Feeding +1 max (`core_attack.rs:75-89`).
  - Advancement resets to the new type at full HP and re-applies traits
    (`core_units.rs:121-173`).
- **2, 3** max moves after quick or slow, and the moves ratio. Moves refresh at
  the side's init (`core_step.rs:251-254`). Recruits arrive with 0 moves and
  no attack (`core_units.rs:226-227`).
- **4, 5** max XP: scaled by the game's modifier, then intelligent −20% or dim
  +20%, in the engine's order (`units.rs:30-32`, `:232-235`,
  `core_units.rs:141`, `:164-166`).
- **6** cost (the new type's on advancement: `core_units.rs:144`).
- **7** leader. **8** has_attacked: set by the attack, cleared at the side's
  refresh (`core_attack.rs:427-431`, `core_step.rs:253`).
- **9-12** alignment one-hot: `classes.Alignment` 0-3 equals the Rust coding
  (`classes.py:19-24`, `combat.rs:10-12`).

### Parity columns 13-108

- **13-72 weapon slots**: the unit's own damage after strong or dextrous, base
  strikes, range, damage type, and the type's specials united with the unit's
  (`core_attack.rs:97-124`, `core_parity.rs:236-243`). At most 3 attacks among
  reachable types, so no weapon is lost.
- **73-78 resistances** of the type: no default-era trait or advancement changes
  them. Steadfast is shown through the ability bit and applied in the fight line.
- **79-91 traits**, **92-104 abilities**, **105 poisoned**, **106 slowed**
  (slowed is dropped at the end of the slowed unit's side turn:
  `core_step.rs:301-312`).
- **107** `combat_modifier` with the type's alignment and fearless, under the
  illuminated lawful bonus of the unit's hex.
  - Lawful ±b, chaotic ∓b, liminal 25 − |b|, fearless floors at 0
    (`combat.rs:207-220`).
  - Illumination is +25 capped at 25 (`core_attack.rs:91-94`).
- **108** leadership: 25 × (leader level − unit level) for an adjacent, higher,
  unpetrified same-side leader; best one, not cumulative (`core_attack.rs:179-196`).
  Matches `ABILITY_LEADERSHIP` (`wesnoth_src/data/core/macros/abilities.cfg:192-212`:
  `value="(25 * (level - other.level))"`, `cumulative=no`,
  `formula="level < other.level"`).

### Version-2 columns

- **109** level / 4, the type's level (`core_step.rs:98-104`).
- **110** the chance to be hit where it stands, the movement class's
  `defense_pct` at the hex (`core_attack.rs:168-174`). The same number feeds the
  fight.
- **111-126 / 127-142** per class: the chance to be hit and 1 / movement cost,
  from the unit's own table (feral's village cap included, `units.rs:110`,
  `core.rs:344-349`).
  - The slowed class doubles costs below 99 (`core.rs:389-395`;
    `core_parity.rs:265`), as `unit::movement_cost` does.
  - 0 where the cost is 99 or more.
  - For caps, see F1.
- **Recruit rows**: the type's level and tables, "stands" 0, fights 0
  (`core_parity.rs:270-279`, `core_encode.rs:565-580`).
  - The 13 base columns: type HP, ratio 1, type moves, moves ratio 0, scaled
    XP, cost, has_attacked 1 (`encode.rs:135-151`, `core_encode.rs:572`).
  - The type's weapons, resistances and abilities (`core_parity.rs:171-178`).
- **143-340 fight lines**, only for own, on-board, unpetrified units with an
  attack left (`core_parity.rs:284-287`), against visible, non-allied,
  non-inert units on the six neighbours (`observe.rs:72-74`). In `nbrs` order
  N, NE, SE, S, SW, NW, as documented (`observe.rs:28-34`).
  - **Chance to hit**: the target's defence, plus accuracy, minus parry,
    clamped. Magical = 70 on offence and defence, replacing the base. Marksman =
    max(60, ·) on offence only (`combat.rs:365-383`). This matches
    `WEAPON_SPECIAL_MAGICAL` (`cumulative=no`, no `active_on`) and `_MARKSMAN`
    (`cumulative=yes`, `active_on=offense`) in
    `wesnoth_src/data/core/macros/weapon_specials.cfg:81-121`, and the
    composite at `attack.cpp:167-180`.
  - **Damage**:
    - 100 + ToD (the attacker's hex, illuminated) + leadership, times the
      resistance, rounded by `round_damage` (`combat.rs:197-204`, `:384-406`;
      `attack.cpp:182-207`);
    - steadfast doubles positive resistance below 50, capped at 50, on defence
      (`combat.rs:386-393`);
    - backstab ×2 on offence with an unpetrified enemy of the defender opposite
      (`core_attack.rs:211-221`, macro at `weapon_specials.cfg:16-33`);
    - charge ×2 for both sides when the attacker charges (`combat.rs:399-403`).
  - **Slowed**: a slowed side's line shows the halved damage, as
    `if(is_slowed) damage = slow_damage` (`outcomes.rs:469-471`,
    `attack.cpp:205-207`).
  - **Swarm** strikes at current HP (`combat.rs:222-232`, `:407-408`).
  - **Firststrike, drain, poison, slow, berserk** are not numbers in the line.
    They are the special bits in both units' weapon slots.
  - **The answer** is the slot chosen by `counter_weapon`, the engine's
    `choose_defender_weapon` (`outcomes.rs:522-560` against `attack.cpp:570-656`,
    `better_combat` `:475-515`). There is no answer when no weapon of that range
    exists, or when the defender is petrified.
  - **When they are current**: recomputed at every encoding from the live state,
    so after a move, a fight, an advancement (`core_units.rs:179-196`), and
    turn-start healing and poison (`core_step.rs:172-255`).

### Point of view

- `side_code`, `is_ours`, own villages, the recruit list, the leader position,
  `may_attack`, and the economy's own/enemy are all computed from `side`, the
  side to move (`core_encode.rs:397`, `:412`, `:425-428`, `:485`, `:491`).
- Nothing is hard-wired to side 1.

### Python against Rust

- These layouts agree: `encoder.py:276-299`, `:316-323` and `:329-361` against
  `core_parity.rs` and `core_encode.rs`.
- They disagree in two places:
  - F2: `encoder.py:310-312`;
  - the per-class defence comment `encoder.py:337-338` does not say that a cap
    is stored as its size. The test docstring does (`test_unit_rows2.py:53-55`).

---

## 3. To check by a run

1. **The live loop's village flag (F5).** Load the live checkpoint and print
   `ckpt.get("fog_hides_enemy_villages")`, and the encoder attribute
   `fog_hides_enemy_villages` that the simple loop's worker actually uses.
   Expect True.
2. **The cap gap (F1).**
   - Setup: a Horseman on `Gs^Fp` and one on `Gg`, side 1, fog off.
   - Encode with unit row version 2.
   - Expect col 110 = 0.70 on `Gs^Fp`, while min(col 111+FLAT, col 111+FOREST)
     = 0.60.
   - Repeat with a Gryphon Rider on `Gg^Tf`: expect 0.70 against 0.50.
3. **The defender's choice against the engine.**
   - Over the human replay corpus, take each attack command's recorded
     defender weapon (the engine chose it).
   - Compare it with `GameCore.counter_weapon_choice` on the reconstructed state
     just before the command.
   - Count disagreements and DP fallbacks.
4. **Hand-computed special lines**, as tests to add, each with the engine
   formula:
   - Marksman Elvish Marksman attacking a unit at 30% chance to be hit → 60.
   - A Dark Adept attacked in melee: no answer.
   - A Mage's magical answer to a ranged attack → 70.
   - A Horseman charging (both doubled).
   - A Thief backstabbing with a flanker.
   - A Dwarvish Guardsman defending (steadfast resistance).
   - Swarm: check whether any reachable default-era type has it. If none, the
     swarm path is never exercised in these games.
   - A slowed defender's answer.
   - A Mage of Light next to the attacker at night.
5. **The hidden leader (F2).** Under fog, put an enemy Lieutenant next to a
   visible enemy Grunt that is adjacent to our Spearman, on a hex our side does
   not see. Confirm the answer's damage includes +25%, and record the behaviour
   as the engine's.

## 4. Could not determine

- **Backstab and facing.** The engine's backstab uses `direction_from(loc,
  other.facing)` (`weapon_specials.cfg:27-31`), the attacker's facing. The sim
  uses the hex opposite the attacker (`core_attack.rs:211-221`). This is
  identical when the attacker faces the defender. I did not confirm whether the
  attack dialog's prediction is computed before or after the facing is set.
- **Time areas.** `area_lawful_bonus` indexes a time area's cycle as
  `(turn − 1) mod len`, ignoring any `current_time` of the area
  (`core_step.rs:86-95`). Tombs of Kesorak (`random_start_time=no`, four areas)
  and Elensefar Courtyard (an area added by an event) are the ladder maps
  concerned. Checking the phase against `tod_manager::get_time_of_day` was out
  of my reading budget.
- **The live checkpoint's `fog_hides_enemy_villages`** (F5).

## 5. For other parts

- Missing columns: the turn limit, and the enemy's fight lines against our
  units (the "Damage versus" tooltip, `reports.cpp:904-930`). Also the
  `resting` status, which drives +2 healing and has no column.
- `Koa^Xo`, `Chw^Xo` and `Cv^Xo` (Aethermaw, Hellhole, Walls of Pyrennis) get the
  keep or castle bit from the letter in the base code (`terrain.rs:377-384`),
  yet they are impassable. Whether the engine treats them as a keep is for the
  recruit-mask auditor.
- `set_terrain` sets `village_mod = 0` (`events.rs:448`). A village created by
  an event would rely on `village_terrain` alone.
