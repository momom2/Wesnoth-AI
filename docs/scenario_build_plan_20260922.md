# Dev plan: one scenario builder, checked against the game (2026-09-22, rev 3)

Rev 2 after an independent review of rev 1; rev 3 after a second
review of rev 2. Both reviews found factual errors that changed the
work, and both are recorded rather than quietly dropped, because the
errors are themselves instances of the bug class this plan exists to
kill: something was asserted, nothing checked it.

## Objective

Build a sim scenario from a scenario request and Wesnoth's own files,
by one path, such that **anything we fail to process and anything we
get wrong is detected automatically**. No Wesnoth install at runtime.

## The bug class

Every scenario-fidelity defect found on 2026-09-21/22 had one shape:
the scenario declared something, we did not read it, and nothing said
so. Hardcoded village gold, unread per-side fog, starting gold
declared inside a macro, `[effect]` members keyed by tag instead of id.

The class is alive elsewhere in the tree. `tools/scenario_events.py:1882`
dispatches event actions through `_ACTION_HANDLERS.get(action.tag,
_no_op_action)`, so an unrecognised action is silently discarded, and
`remove_unit` is explicitly mapped to the no-op under the comment
"state-affecting tags we don't (yet) interpret". `_COSMETIC_MACROS` at
`:59-72` substitutes a fixed list of macros to the empty string.

**Correction 1, the plan's own cautionary tale.** Rev 1 claimed our
expander fails to recurse into `{DEFAULT_SCHEDULE}` and made that the
argument for writing our own preprocessor. False. The expander
recurses to a fixpoint; the macro is on `_COSMETIC_MACROS` and is
deleted on purpose. Verified: with the entry present the expansion
yields zero `[time]` blocks, with it removed all six, `dawn`,
`morning 25`, `afternoon 25`, `dusk`, `first_watch -25`,
`second_watch -25`, which is the cycle the sim hardcodes. The schedule
is missing because a hardcoded ignore list swallowed it.

**Correction 2.** Rev 1 called the committed templates "the game's own
preprocessor output". They are not. `tools/build_scenario_templates.py`
strips `[side]` 1 and 2, drops `#textdomain` and translation markers,
inlines `map_data`, and injects `turns="-1"`,
`experience_modifier="70"`, `has_mod_events`, `loaded_resources`,
objectives and two hand-written era events. So a generation path
reading the template would read **our** injected 70 as though the
scenario declared it, recreating the defect 716a1c3 removed; and a
byte-for-byte check of our own expander cannot use the templates.

**Correction 3.** Rev 1 implied the quick-leader rule was unmodelled.
It is modelled at `tools/traits.py:275-285`, citing the era's Lua, from
one call site both pipelines share. The unmodelled gates are
`make_4mp_leaders_quick` and `unit.variables.dont_make_me_quick`
(`eras.lua:6-16`); the engine has no side-count gate.
`turns_over_advantage` is display-only (`eras.lua:143` calls only
`gui.show_popup`) and unreachable under the injected `turns=-1`. Our
turn-cap tiebreak scores differently from the engine's popup formula,
income times five plus gold plus unit worth excluding the leader,
against our villages at 2.0, gold at 0 and material including the
leader, and that divergence is **nowhere recorded**, which by W6's own
rule makes it the first thing W6 writes down.

## The three detectors

1. **Unprocessed input, in two halves**, because one classifier cannot
   see both. A census of the expanded tree cannot fire on what the
   *expander* dropped: the gold declared inside a macro and
   `DEFAULT_SCHEDULE` never reach the tree. So the expand stage needs
   W0's diff as a standing test, and the classify stage needs the
   tree census. Values are classified as well as names, since
   `apply_to=new_ability` was a value of a modelled attribute.
2. **Unread input.** A MODELLED classification that names no reader
   proves nothing. `village_gold`, `[side] fog=` and `income=` were all
   ordinary known names, present in the tree, when they were bugs; a
   name-and-value classifier passes every one. Each MODELLED entry
   names the function that reads it, and a test asserts the value
   reaches the built state. **This is the detector that earns the
   plan**; the other two are necessary and insufficient.
3. **Discrepancy.** What we build against what the engine builds (W4).

Classification therefore has four buckets, not three. **SUBSTITUTED**
is the one rev 2 lacked: input we read and then deliberately do
something else with, because reimplementing Wesnoth's version is out
of scope. `[side][ai]` is the case that forces it. We replace the
engine's RCA AI with `tools/neutral_ai.py`, a stationary-only port,
by user ruling 2026-07-14.

A SUBSTITUTED entry records its reason **and its preconditions**,
because a substitution is only valid while its assumptions hold. The
neutral AI's precondition is that side-3 units cannot move: the
enclave maps pin them through their own `turn refresh` event, and
2p_mini through terrain. Nothing checks that today, so if a scenario
change or an expansion change let those units move, a stationary-only
AI would keep driving them and nothing would say so.

## Architecture

| stage | input | output | its oracle |
|---|---|---|---|
| **Expand** | scenario id, WML sources | expanded WML tree | Wesnoth's `--preprocess` output (W0) |
| **Classify** | expanded tree | modelled / ignored; UNKNOWN empty | the census of whole scenarios |
| **Interpret** | modelled subset | initial state | the engine, per scenario (W4) |
| | | event program | `tools/diff_replay.py` over the corpus |

The event-program oracle is thick for the ladder maps, 84 to 1,896
corpus games each, and thin for five minis: Modified Close Relation 4
games, Mini Fallenstar 9, 2p_mini 10, Small Fallenstar 16, Benji 16.

## Work items

Each carries acceptance, cost and a kill criterion, as
docs/plan_20260904.md's steps do. Mostly local; W1 owes a corpus sweep
and W4 needs the Wesnoth install.

### W0. Diff our expansion against the game's, per scenario

Compare our expansion of each pool scenario against the committed
engine expansion, cosmetics excluded, and enumerate every divergence
cluster. No Wesnoth run, no new harness.

This is the **only** check covering the expand stage, so it ships as a
committed regression test rather than a one-off; detector 1 is
incomplete without it.

*Cost:* an afternoon. *Acceptance:* a record naming every cluster and
its cause, and a test that fails when a new cluster appears. *Kill:*
none; this is measurement.

### W1. The failing default

Turn the silent-ignore sites into classified ones: the event action
dispatch, `_COSMETIC_MACROS`, and the unknown-macro path. Unrecognised
input warns once with its name and scenario, or fails, per a config
switch. Take `DEFAULT_SCHEDULE` off the cosmetic list and read the
schedule.

`IGNORED by name` is `_COSMETIC_MACROS` renamed unless it is
disciplined: every entry carries a reason and a test that the ignored
thing is inert. `DEFAULT_SCHEDULE` sat on that list for months and was
anything but.

The size is measured, not guessed. Unknown event-action tags falling
through to the no-op: five in the game's expansion (`scroll` 30,
`delay` 12, `screen_fade` 12, `foreach` 3, `end_turn` 1) and one in
ours (`end_turn`). `foreach` must be implemented rather than listed,
because W5 depends on it.

**This item is not local.** Changing `_COSMETIC_MACROS` and the action
dispatch changes what `load_scenario_wml` returns on the
*reconstruction* path, which `diff_replay` over 17,039 replays
certifies, and by the project's own rule that sweep is box work and is
no acceptance test unless it fails under the old behaviour.

*Cost:* a day local, plus a corpus sweep, about 20 minutes and $0.20.
*Acceptance:* every currently-swallowed name appears in a census or a
log; the 28-scenario snapshot moves only where intended; the sweep
stays clean. *Kill:* if the warn-everything run produces more
unmodelled names than can be triaged in a day, narrow to the pool's
own tags first.

### W2. Repair and regenerate the templates

Fix the dangling `DRILL_SCENARIO_IDS` import, unimportable since
2026-08-10. That fix is a deletion rather than a repair: the drill
sources under `add-ons/wesnoth_ai/scenarios/drills/` no longer exist,
so the three drill templates and `around_mini` are retired.

**Keeping sides 1 and 2 is new code on the export path, not a move.**
Rev 2 claimed the exporter already strips them at emit time. It does
not: `_strip_player_sides` lives in the template builder and runs at
build time, and `sim_to_replay` only splices its own blocks before the
first column-0 `[side]`, which after stripping is the scenery side.
Keeping the player sides, which the builder dedents to column 0, makes
that anchor find side 1 and breaks side 2's leader spawn at playback,
the failure recorded at `sim_to_replay.py:1294`. So this item either
teaches the exporter to strip, or leaves the templates stripped and
commits the raw preprocessor output separately for generation. Decide
before starting.

Keep the injected attributes out of anything generation reads, or mark
them so they cannot be mistaken for the scenario's own. The injection
is conditional on a regex over every line of the body, so keeping the
player sides can suppress an injection, and "differs only in the
player sides" is not a prediction the code supports.

*Cost:* half a day. The preprocessor is 60-90 ms per scenario in
steady state, but the builder runs it once per source tree with a
180-second timeout and a cold WML cache, so budget minutes.
*Acceptance:* the builder runs, and every difference between the
regenerated and committed templates is explained in the record. Check
at run time that the Steam tree still matches `wesnoth_src/`: identical
today, but the templates were built 2026-06-12 and the install was
updated 2026-09-19. *Kill:* none.

### W3. The classification manifest, bound to readers

Classify the whole surface over **whole** scenarios, which is why it
follows W2: the current templates omit `recruit`, `type`, `[village]`
and `[unit]` for the player sides and see `gold`, `fog`, `income` and
`village_gold` only through scenery side 3, which 10 of the 28
templates carry. The 64 tag paths and 210 path-attribute pairs
measured over the current templates are a floor; re-measure after W2.

Each MODELLED entry names its reader, per detector 2.

*Cost:* a day and a half. *Acceptance:* zero UNKNOWN across the pool;
an injected unknown tag, an unknown attribute on a modelled tag, and
an unknown dispatch value each fail. **And the detectors must fire on
the known-bad past**: replay each cited defect's pre-fix state. Expect
this to show honestly that a name-and-value classifier catches almost
none of them, since village gold, per-side fog and `[hides] id=` were
all ordinary known pairs and the macro-declared gold never reaches the
tree; the reader binding is what catches them. *Kill:* if the
reader-binding detector cannot be made to fire on the village-gold and
per-side-fog defects specifically, the classification is decoration
and the item stops.

### W4. The engine oracle for scenario init

Rev 1 treated this as two collector fields; rev 2 claimed no
multiplayer start exists. Both wrong.
`tools/validate_replay_wesnoth.py:233` launches `wesnoth --load` on a
`sim_to_replay` save, and `tools/templates/wesnoth_save_scaffold.wml`
carries `campaign_type="multiplayer"`, `era_id="era_default"` and the
`quick_4mp_leaders` prestart event. A real multiplayer start is
already driven here under `WESNOTH_E2E=1`, so the era half of W6 is
answerable.

What the collector lacks is narrow: the current time of day is
reported and already checked against the engine, 54 of 54 on
2026-09-20; the schedule *definition* is absent; per-hex village
ownership is fog-gated though the per-side count is not; the unit list
is vision-filtered, but `oracle_units` already emits every unit with
per-side visibility.

*Cost:* half a day to scope on the existing harness, 200-400 lines.
*Acceptance:* a record naming every field compared and the agreeing
fraction per scenario. *Kill:* if the exported save is the only way
in, the oracle tests our exporter as much as our builder, and that
circularity is stated rather than hidden.

### W5. One expansion source for generation

Three risks to price first, all found by review:

- **Switching to the game's expansion would break a SUBSTITUTED
  entry's precondition.** `enclave_micro_isar.cfg:86-91` pins its
  `role=monster` units to 0 movement every `turn refresh`, and our
  expander reduces that macro to a `[modify_unit]` form we interpret
  (`scenario_events.py:259-275`). The game's expansion is
  `[store_unit kill=yes]`, `[foreach]`, `[unstore_unit]`, none of
  which we handle, so the pin would vanish. The pin is the scenario's
  rule, not our fix; what depends on it is our stationary-only neutral
  AI, which would then drive mobile units and say nothing. W1's
  "implemented or listed" would pass on listing, so the items are
  linked: `foreach` before W5, and the precondition gets a test.
- **`[side] fog=` is declared on all 56 player sides**: yes on 50,
  no on the six of `2p_mini`, `2p_mini_edited` and
  Modified_Tiny_Close_Relation (read from the engine by W4). Reading it
  without a precedence rule overrides `setup.fogless`, a training
  lever with its own tests. The rule is: the scenario supplies the
  default, our choice wins.
- **The defect W5 names is mostly empty.** `[side] recruit=` and
  `type=` appear on zero player sides across the pool, and `fog=` is
  already True by default.

Generation also already reads the committed template, through three
private regexes in `scenario_pool._scenario_tod_info`, while
`wml_state.read_tod` has no production caller. The expansion source is
not an open choice; it is already two sources read two ways, and W5's
job is to make it one.

*Cost:* a day, plus `foreach`. *Acceptance:* the 28-scenario snapshot
moves only where intended; W0's divergence set shrinks to the intended
entries; the `.cfg` trees are unread at generation time; the tentacle
pin still holds. *Kill:* none.

### W6. Assumptions that become reads

Decide each by evidence. The six-slot cycle is hardcoded and every
pool scenario happens to use exactly it. The experience modifier has
several homes, of which the template's injected 70 is read by nothing
in the sim.

Two known SUBSTITUTED entries get written down here, since both exist
today and neither is recorded anywhere: the neutral AI with its
immobility precondition, and the turn-cap tiebreak, which scores
income times five plus gold plus unit worth excluding the leader in
the engine, against our villages at 2.0, gold at 0 and material
including the leader.

*Cost:* half a day per assumption. *Acceptance:* each is a read, or a
record of the evidence that justified keeping it. *Kill:* an
assumption that cannot be checked without W4 waits for W4.

## Consequences for numbers

Any W5 or W6 change that moves the built state makes matches run after
it **cross-build** against every Elo measured before it, as the
2026-09-13 hide-cover change did, and wants
`constants.OBSERVATION_EPOCH` bumped from 3. The bump is blunt: it
gates the pre-encoded corpus and the human anchor, both replay-derived,
which a generation-only change does not touch, and it only warns on a
checkpoint while refusing caches. So the item that moves the
observation says which half it moves. Mini self-play is already
non-comparable with anything before 716a1c3.

## Priority, stated plainly

Nothing here is known to change a ladder game today: all 28 pool
scenarios carry the identical six-slot cycle, and no **pool** mainline
map declares an experience modifier. Both statements need their
qualifiers. Two pool scenarios start at second watch, Tombs of Kesorak
carries three time areas and Elensefar Courtyard one, four minis roll
a random start, and four minis declare `experience_modifier="70%"`;
outside the pool, Dark Forecast and Isle of Mists declare 100 and 90.

This is insurance against the next scenario, the next Wesnoth version
and transfer to real Wesnoth. Price it as insurance while phase 2's
first measurement waits for a word.

## Deferred on purpose

Writing our own preprocessor. Rev 1 scheduled it on evidence that was
a hardcoded ignore list. It buys nothing the earlier items do not, and
its requirement list is larger than rev 1 stated, omitting
`#ifver`/`#ifhave`, `#arg`/`#endarg` optional arguments and the
`{./...}` include form. It becomes worth doing behind a trigger, such
as wanting programmatically generated scenarios, with W0's diff as its
acceptance test.

## Not in this plan, and ahead of it

The encoder feeds the network no time-of-day signal: `GLOBAL_FEAT_DIM`
is 6 (turn, side, gold, income, own villages, enemy villages), and
units carry alignment but no time of day or lawful bonus. Since
alignment only matters through the current time, the network cannot
tell a lawful unit's good hour from its bad one, and on the two
second-watch maps and the random-start minis the turn number does not
even encode the phase.

This is a model-input question with an 800-game answer, not fidelity
work, and by user ruling 2026-09-22 it is **the next thing worked on**,
ahead of this plan. It sits in BACKLOG.md under the training-signal
items.

---

# Record: W0 and W1 (2026-09-22)

Written on the run. Local work only; the corpus sweep W1 owes is still
outstanding and is listed at the end.

## W0. The expansion diff — DONE

`tools/analysis/expansion_diff.py` compares our expansion of each pool
scenario against the committed engine expansion and reports every
difference as a **cluster**, a `(kind, detail)` pair with the scenarios
it affects, so a systematic gap reads as one line rather than 28. Over
the 28 pool scenarios it takes 0.2 s, which is why it runs in the fast
tier (`tests/test_expansion_diff.py`).

Compared: the board schedule, `[time_area]` count, per-event-name
action counts, scenery sides 3+ with their attributes and unit and
village counts, the resolved map, and the remaining scenario
attributes. Excluded, with the reason in the source: player sides 1
and 2 (stripped at build), the attributes the builder injects, the era
events it appends, and presentation tags.

Two comparisons were wrong on the first pass and are worth recording,
because both would have read as findings:

- **The map was compared by attribute.** A scenario names a `.map`
  file and the engine's expansion inlines the grid, so `map_data` vs
  `map_file` differed on 21 of 28 and checked nothing. It now resolves
  both to a grid.
- **The grid was then compared as raw row text**, which reported three
  mini maps as differing. The inline form loses the last row's
  trailing spaces to its closing quote; every cell was identical. It
  now compares rows of stripped cells, as every consumer reads them.

**Found, and fixed below: 6 clusters, of which 4 were bugs.**

| cluster | scenarios | verdict |
|---|---|---|
| ours 0 `[time]` blocks, the game's 6 | 28 | **bug**: `{DEFAULT_SCHEDULE}` was deleted |
| `user_team_name` keeps a `_ "` prefix | 4 | **bug**: the translatable marker was not stripped |
| `map_data` vs `map_file` | 21 | the tool's; now resolved |
| three grids "differ" | 3 | the tool's; trailing whitespace |
| era `[lua]` prestart absent | 26 | SUBSTITUTED |
| era `[lua]` prestart absent, second prestart present | 2 | SUBSTITUTED |

The two survivors are the same cause — the era's
`quick_4mp_leaders` prestart
(`wesnoth_src/data/multiplayer/eras.lua:5-22`), which
`tools/traits.py`'s `roll_traits` applies natively. Both are recorded
in `tests/data/expansion_diff_expected.json` with their reason, and
the test refuses an entry whose `stands_in` no longer names a file.

**The check fails under the old behaviour**, which is the bar this
project sets for a sweep: re-adding `DEFAULT_SCHEDULE` to the cosmetic
list produces the 28-scenario cluster again and
`test_no_unexpected_divergence` fails.

## W1. The failing default — DONE except the sweep

**The three silent sites are now classified and the default fails.**

1. **Macro classification split in two.** `_COSMETIC_MACROS` now means
   presentation only; `_SUBSTITUTED_MACROS` means the sim implements
   the behaviour and names what stands in for it. `DEFAULT_SCHEDULE`
   came off the first (it belonged on neither), `TURNS_OVER_ADVANTAGE`
   moved to the second.
2. **The event-action dispatch no longer defaults to a no-op.**
   `_IGNORED_ACTIONS` and `_SUBSTITUTED_ACTIONS` each carry a reason
   per tag; anything else warns once with its tag and scenario and
   raises `UnmodelledWML` under `WESNOTH_STRICT_WML`. The dead
   `_no_op_action` is gone.
3. **The unknown-macro path** warned at DEBUG, which is where this
   class of bug hides. It warns at WARNING, counts, and raises under
   the same switch.

**Measured after the change**: over all 28 pool scenarios, with
prestart, start, turn-refresh and per-side turn events fired, **zero
tags fall through and zero macros are unknown**.
`tests/test_action_classification.py` holds that as a test, along with
the `end_turn` substitution's precondition — its only use in the pool
is Silverhead Crossing's `side 3 turn`, on a `controller="null"` side
that neither the engine nor we ever run.

**The corpus's whole scenario surface is classified, not just the
pool's.** The corpus is 21 ladder maps, the minis, and exactly two
off-whitelist mainline maps -- Cynsaun Battlefield (334 games) and
Hornshark Island (248). Both build and fire their events with nothing
falling through and no unknown macros, so the sweep will not spam and
`WESNOTH_STRICT_WML` would pass over it.

Measured the same way over ALL mainline scenario `.cfg` files, which
is a much wider set: 23 unclassified event-action tags do exist
(`have_unit`, `store_side`, `terrain_mask`, `allow_recruit`,
`unstore_unit`, `micro_ai`, `kill`, ...), and every one of them lives
in a 4p/5p/6p map or in the two 2p survival maps, Dark Forecast and
Isle of Mists. None of those maps is in the corpus. They are what the
failing default is FOR: if the corpus is ever widened to them, they
warn instead of building a quietly wrong game.

**Two preprocessor-grammar bugs surfaced by the census**, both now in
docs/wesnoth_rules.md with sources:

- **Macro names may contain `:`.** `#define INTERNAL:SPECIAL_NOTES_*`
  and friends all collapsed onto the single name `INTERNAL`, so every
  use expanded to nothing. 2 uses in the pool (Silverhead Crossing).
- **`#arg NAME … #endarg` declares an optional named argument with a
  default**, inside the body rather than on the `#define` line. It was
  unimplemented, and the comment stripper ate the markers while
  leaving the default as a stray line, so `{OVERLAY}` survived
  unsubstituted. 9 uses in the pool (the three enclave scenarios).

Both were cosmetic in effect here — a special-note string and a hero
icon — but the grammar is now read rather than approximated, and
`_extract_inline_macros` is the single extractor for both the core
macro files and a scenario's own body, which it was not before.

**A third mirror, and a latent wrong default, both in the time of
day.** `scenario_pool._scenario_tod_info` read `current_time`,
`random_start_time` and the slot count with three private regexes over
the raw template text (including a regex that stripped `[time_area]`
first), while reconstruction read the same three keys off a parsed
node, and `wml_state.read_tod` — written for exactly this — had no
production caller. Both now call it; the strip is unnecessary because
`WMLNode.all` does not descend into `[time_area]`, which is pinned.
Reading the parsed node also exposed that `random_start_time` has a
**third form**, a value list such as `"2,4"`, which a yes/no coercion
folds onto "no random start" and so onto dawn. The guard that meant to
drop those replays sat inside a branch only a plain `yes` could enter.
`wml_state.wml_bool_or_none` distinguishes it and the caller drops.
No corpus replay uses the form, so nothing had diverged.

**The board schedule is expanded but still not read.** All 28 pool
scenarios declare exactly the standard six-phase cycle with the
standard bonuses, so the hardcoded `TOD_DEFAULT_CYCLE` is correct for
the pool today and `{DEFAULT_SCHEDULE}`'s deletion was latent rather
than live. Turning the cycle into a read is W6 and is listed there.
`[time_area]` cycles were already read, and Tombs of Kesorak's three
zones are the reason.

## What moved, and what did not

The 28-scenario built-state snapshot and the 120-replay extract
snapshot are both unchanged, which is the intended result: everything
above is a change to what we *notice*, not yet to what we *build*.

## Still owed

- **The corpus sweep.** `_COSMETIC_MACROS`, the macro-name class, the
  `#arg` handling and the translatable-marker strip all change what
  `load_scenario_wml` returns on the RECONSTRUCTION path too. The
  snapshots cover 120 replays and 28 scenarios; `diff_replay` over the
  full corpus is box work, about 20 minutes and $0.20, and by this
  project's own rule it is not an acceptance test unless it fails
  under the old behaviour, so it runs with the old predicates
  monkeypatched back as the control.
- `foreach` / `do` / `unstore_unit`, which W5 depends on. `[foreach]`
  appears 3 times in the engine's expansion of the three enclave
  scenarios, always as `[store_unit kill=yes]` → `[foreach]` →
  `[unstore_unit]`; `[unstore_unit]` has no handler, so on the game's
  expansion those units would be killed and never put back. Our own
  expansion does not hit this because `MODIFY_UNIT` is a SUBSTITUTED
  macro reduced to `[modify_unit]`, which is precisely the
  precondition W5 warns about.

## W2. The templates — DONE, one deletion pending the user

**The builder imports again.** It took `DRILL_SCENARIO_IDS` from
`scenario_pool`, which stopped existing when the user ruled the drill
scenarios out (fba0513) and their sources were deleted with them. The
drill path is gone rather than repaired, for that reason.

**Rev 3 was wrong about `around_mini`.** It listed it with the drills
as an orphan to retire. Its `.cfg` and `.map` are still shipped in the
Mini Maps Collection and `tests/test_mini_tentacle_spawns.py` builds
it; retiring the template would have broken that test. It left the
training POOL in 2026-07-14, which is a different thing. It is now
named in `EXTRA_MINI_TEMPLATE_IDS`, which also fixes a latent bug:
`--only around_mini` intersected the request with the pool constants
to decide which source tree to preprocess, found nothing, and reported
the id as NOT FOUND.

**The acceptance run: all 29 templates regenerate BYTE-IDENTICAL** to
what is committed, from the current Steam install through Wesnoth's
own preprocessor. So the templates are certified as this install's
output, and the plan's worry that the 2026-09-19 Steam update might
have drifted them is answered: it did not, for these scenarios.

`tests/test_template_builder.py` keeps the module importable, checks
every scenario it would build resolves to a source through the
production resolver (a scenario's id is not its filename --
`2p_mini_edited` lives in `2p_mini_1.cfg`), and pins that the only
unbuildable templates on disk are the three drills.

**Deleted on the user's ruling (2026-09-22):**
`drill_chokepoint.wml`, `drill_duel.wml` and `drill_village_rush.wml`,
tracked files whose sources went with the drill scenarios in fba0513.
They could not be regenerated or checked against anything. 29
templates remain, which is exactly the set the builder regenerates
byte-identically, and `tests/test_template_builder.py` now asserts the
unbuildable set is EMPTY rather than naming the three.

## W5. One expansion source for generation — DONE

The plan called this a day's work plus `foreach`. It turned out to be
neither, because W0 changed what the question was.

Generation did not have an open choice of source; it had **two
sources read two ways**. Everything went through `load_scenario_wml`
except the time of day, which `_scenario_tod_info` read off the
committed template with three private regexes. One scenario, built
from two renderings of itself.

Measured before changing it: our expansion and the engine's agree on
`(current_time, random_start_time, slot count)` for **28 of 28** pool
scenarios. So `_scenario_tod_info` now reads `load_scenario_wml`, and
generation reads exactly one rendering. The values are unchanged --
Fallenstar Lake and Ruined Passage at `current_time=5`, four minis
random-start, the rest default.

`tools/templates/scenarios/` is now read by the EXPORT path
(`sim_to_replay`), the builder that writes it, and the two checks.
Nothing on the generation path reads it.

The three risks rev 3 priced:

- **The `foreach` precondition is not reached, and the reason is
  recorded.** Switching to the engine's expansion would have needed
  `[foreach]` / `[unstore_unit]`, because `enclave_micro_isar`'s
  monster pin is `[store_unit kill=yes]` → `[foreach]` →
  `[unstore_unit]` there. We do not switch: our expander reduces
  `MODIFY_UNIT` to `[modify_unit]`, which is a SUBSTITUTED macro and
  now says so. `foreach` is still owed before anyone reads the engine
  expansion at generation time, and `[unstore_unit]` now WARNS instead
  of silently dropping a killed unit, which is what made this
  invisible.
- **`[side] fog=` reaches generation since W4 (2026-09-23).**
  `build_scenario_gamestate` reads each player side's fog and shroud
  through `wml_state.read_side`, and `setup.fogless` still turns fog
  off: the scenario supplies the default, our choice wins. Until then
  the three minis that declare `fog=no` were played under fog.
- **The defect was indeed mostly empty**: `[side] recruit=` and
  `type=` appear on zero player sides across the pool.

## W6. Assumptions that become reads — DONE

### The six-slot day: the assumption is TRUE, and now it is checked

Measured rather than argued:

- **28 of 28 pool scenarios** declare exactly the default cycle
  (`dawn 0, morning 25, afternoon 25, dusk 0, first_watch -25,
  second_watch -25`).
- **17,019 of 17,019 corpus games** declare exactly that cycle
  (`tools/analysis/corpus_census.py`, which now reads the schedule;
  record refreshed at `training/metrics/corpus_census.json`).

So `combat.TOD_DEFAULT_CYCLE` is a fact about our data, not a guess.
It is NOT a fact about Wesnoth: a sample of the raw `replays_raw/`
tree, which is much wider than the curated corpus, turns up 15-slot
and 24-slot schedules with intermediate bonuses (`0, 5, 15, 25, …`)
from other eras and map packs. If the corpus is ever widened, the
constant breaks.

**It stays a constant, and here is why**, stated so it can be argued
with: the cycle is hardcoded in BOTH engines --
`combat.TOD_DEFAULT_CYCLE` and `rust/wesnoth_core`'s `DEFAULT_CYCLE`
in `core_step.rs` -- and they must agree on every combat. Making it a
read in Python alone would split them; the Rust half cannot be built
or certified on this laptop (the wheel is phase 3 against source 11),
so it is box work. Turning a zero-risk latent assumption into a
cross-engine parity risk, locally untestable, is the wrong trade.

**What lands instead is the detector.** `wml_state.check_board_cycle`
reads the scenario's own `[time]` blocks and compares them with the
constant, on BOTH pipelines -- generation in
`build_scenario_gamestate`, reconstruction in `replay_extract`. A
non-default schedule warns once with the scenario's name and raises
`UnsupportedSchedule` under `WESNOTH_STRICT_WML`. Note this was only
possible AFTER W1: until `{DEFAULT_SCHEDULE}` expanded, there were no
`[time]` blocks to compare against.

Also measured, and worth keeping: a replay header can carry the
scenario TWICE (`[replay_start]` and `[snapshot]`), on 50 of 1,200
sampled games. A whole-header regex therefore sees every `[time]`
block twice and reports a 12-slot schedule. The census now slices to
the first container, as reconstruction does.

### Finding: the quick-leader substitution has two gates we do not read

Surfaced by a review agent checking W6's claims, and verified against
the source. The era's `quick_4mp_leaders`
(`wesnoth_src/data/multiplayer/eras.lua:5-22`) is the SUBSTITUTED
entry behind both of W0's accepted divergences, and the expectation
file said flatly that "the behaviour is present". It is not the whole
behaviour. The macro has two gates:

- the WML variable `make_4mp_leaders_quick`, which when false skips
  the rule entirely (`eras.lua:6-11`);
- a per-unit `dont_make_me_quick` variable, which skips that unit
  (`eras.lua:16`).

`tools/traits.py` reads neither and applies the rule unconditionally.
**The gates are not hypothetical**: `2p_Dark_Forecast.cfg:68` and
`2p_Isle_of_Mists.cfg:75` set `dont_make_me_quick=yes` on units. The
same two survival maps carry the unclassified event tags, and neither
is in the pool or the corpus.

Measured: **0 gate declarations across all 30 scenarios we build** —
the 28-map pool plus Cynsaun Battlefield and Hornshark Island — and
**0 of 1,500 sampled corpus replays** touch either gate. So the
unconditional rule is exact, for the third time in this block for a
reason nobody had written down.

The rule is live on BOTH paths, which is easy to miss: a replay does
not carry leader traits (docs/wesnoth_rules.md "Pitfall 5" — the
engine re-rolls them from the recorded seed), so reconstruction rolls
them too and applies the quick-leader rule as well.
`wml_state.check_quick_leader_gates` therefore runs at generation
(`build_scenario_gamestate`) and at reconstruction (`replay_extract`),
warning and raising under `WESNOTH_STRICT_WML`.
`tests/test_wml_state.py` pins the precondition over all 30
scenarios, and the expectation file's `why` no longer overclaims.

### The two SUBSTITUTED entries, written down

- **The turn-cap tiebreak** is in the tables:
  `_SUBSTITUTED_MACROS["TURNS_OVER_ADVANTAGE"]` and
  `_SUBSTITUTED_ACTIONS["endlevel"]`, each naming what stands in.
- **The neutral AI, with its immobility precondition** — and checking
  it found a live defect.

### The neutral AI's precondition, checked for the first time

Our neutral-side AI is combat-only (user scoping, 2026-07-14). That
substitution is EXACT only for a unit the real AI would not move,
because Wesnoth's default AI then reduces to its combat candidate
action. The justification in `neutral_ai.py` was "the enclaves pin MP,
2p_mini is terrain-locked", and it had never been checked.

Checked now, through the production reachability code, over the six
pool scenarios with an acting neutral side:

| scenario | neutral units | why they stay put |
|---|---|---|
| enclave_micro_isar | 3 | pinned to 0 MP every `turn refresh` |
| enclave_mini_fallenstar_1v1 | 3 | pinned to 0 MP |
| enclave_small_fallenstar_1v1 | 3 | pinned to 0 MP |
| 2p_mini | 2 | `ai_special=guardian`, and terrain-locked |
| 2p_mini_edited | 2 | `ai_special=guardian`, and terrain-locked |
| Modified_Tiny_Close_Relation | 1 | `ai_special=guardian` ONLY |

**The precondition holds on all six, and the written justification was
wrong about why.** `Modified_Tiny_Close_Relation`'s Tentacle has full
MP and two adjacent water hexes it can legally enter; neither stated
reason covers it. What keeps it still is `ai_special=guardian`, which
sets STATE_GUARDIAN (1.18.4 `src/units/unit.cpp:659`) and makes the
move phase hand the unit a move from its own hex to its own hex --
"is guardian, staying still"
(`src/ai/default/ca_move_to_targets.cpp:269-277`).

Until now `ai_special` was read by nothing in this project. So the
substitution was correct for a reason nobody had written down, on a
flag nobody had parsed -- one map edit away from being wrong silently.

Fixed at the root rather than documented: `_unit_action` reads
`ai_special=guardian` onto the unit, and
`neutral_ai._check_units_are_stationary` runs at the start of every
neutral turn and demands one of the three reasons, warning (and
raising under `WESNOTH_STRICT_WML`) otherwise.
`tests/test_neutral_ai_precondition.py` pins each scenario's reason
separately, because they have different failure modes: a guardian flag
is lost to an unread attribute, a pin to an unrun event, a terrain lock
to a map edit.

This is also the clearest example of why detector 2 is reader-binding
rather than name-and-value: `ai_special=guardian` is an ordinary known
attribute with an ordinary known value, and every classifier that
looks at names and values calls it fine.

## W3. The classification manifest, bound to readers — DONE

`tools/analysis/scenario_surface.py` enumerates every tag path and
attribute the pool's 28 scenarios declare, and
`tests/data/scenario_surface.json` says what we do with each.

**The surface, measured on our own expansion** (which is what
generation reads, since W5): **51 tag paths, 178 path-attribute
pairs** — 72 MODELLED, 98 IGNORED, 8 SUBSTITUTED, **0 UNKNOWN**.

Rev 3 measured 64 paths and 210 pairs over the templates and called
them a floor. They are not a floor; they are a different tree. The
templates strip player sides 1 and 2 and carry the builder's injected
attributes and the engine's music blocks, so the counts are not
comparable in either direction.

A MODELLED entry names its reader as `file.py:symbol`, and the test
checks the symbol is in that file. That is the whole point of the
item: **every defect this plan was written for was an ordinary
attribute with an ordinary value that nothing read.** Village gold,
per-side fog, `[hides] id=`, `[specials] id=`, `ai_special=guardian` —
a classifier over names and values calls all five fine.

**The detectors fire on the known-bad past**, which rev 3 set as the
kill criterion. `tests/test_scenario_surface.py` replays five defects
this project actually shipped by deleting the entry that records each
fix, and asserts the attribute comes back UNKNOWN. All five fire. A
deleted reader is caught separately, and so is an injected attribute.

### The design flaw the kill-criterion test found

The first version allowed a **path default**: classify
`scenario/event/object/effect/abilities/hides` as IGNORED once instead
of listing its six attributes. Running the kill criterion showed that
`[hides] id=` — the 2026-09-13 bug — did NOT come back as UNKNOWN,
because the path default swallowed it. The shortcut reproduced the
original bug's shape exactly: a tag that looks like a display block,
whose `id` is the only thing that matters.

So a path default is now refused wherever any attribute under it is
MODELLED or SUBSTITUTED (`masking_path_defaults`, checked by a test).
Enforcing it turned up four more live mixtures — `scenario/time`,
`scenario/time_area/time` and the two `[modifications][trait]` paths,
each hiding a MODELLED `id` or `lawful_bonus` under a presentation
default. All are enumerated now.

Checked while classifying, and not a bug: Elensefar Courtyard's event
`[time_area]` declares one `[time]` with no `lawful_bonus`, so our
cycle for it is `[0]`. The engine's own expansion declares it the same
way, and an absent `lawful_bonus` is 0 in Wesnoth too.

---

# Where the plan stands (2026-09-22, end of the local block)

| item | state |
|---|---|
| W0 expansion diff | **done**, committed test, fails under the old rule |
| W1 the failing default | **done** locally; owes the corpus sweep |
| W2 templates | **done**; one deletion needs a ruling |
| W3 classification manifest | **done**, 0 UNKNOWN, detectors fire on 5 shipped defects |
| W4 engine oracle | **done** 2026-09-23 — see "W4. The engine oracle" |
| W5 one expansion source | **done** |
| W6 assumptions become reads | **done** |

## Four detectors now stand where there were none

1. **Did we expand it the way the game does?**
   `tests/test_expansion_diff.py`, per scenario, against the engine's
   own rendering. Two accepted divergences, both recorded with a
   reason.
2. **Of what we expanded, what do we read?**
   `tests/test_scenario_surface.py`, 178 pairs, 0 UNKNOWN, every
   MODELLED entry naming a reader whose symbol is checked.
3. **Does anything reach a silent no-op?**
   `tests/test_action_classification.py`: the event-action dispatch,
   the macro tables and the unknown-macro path all fail rather than
   swallow, and `WESNOTH_STRICT_WML` turns every warning into a raise.
4. **Do the substitutions' preconditions hold?**
   `tests/test_neutral_ai_precondition.py` and
   `wml_state.check_board_cycle`, checked at the moment each is
   relied on rather than asserted in a docstring.

## W4. The engine oracle for scenario init — DONE (2026-09-23)

`tools/scenario_init_oracle.py` launches a real multiplayer game per pool
scenario (`--multiplayer --scenario=<id> --era=era_default`, both
factions named, every side played by the AI) with
`add-ons/wesnoth_ai/init_oracle_ai.cfg` on side 1. Its Lua
(`lua/init_oracle.lua`) reports the whole board at side 1's first turn;
the tool builds the same game the way self-play does, with the leaders
the engine drew, and compares 26 fields (village owners only where a
scenario pre-owns one): each player side's gold, base,
total and net income, village gold and support, fog, recruit list and
faction; whether each extra side takes turns; every unit's presence,
type, leader flag, named traits, statuses, hit points, moves and
experience; village owners; every hex's terrain code and lawful bonus;
the time of day; and the lobby's experience modifier.

**Record:** `training/metrics/fidelity/scenario_init_oracle_20260923.json`:
all 28 pool scenarios, every field agreeing, after the fixes below.

**What the harness has to supply, because a command-line start is not a
lobby** (docs/wesnoth_rules.md, "A command-line `--multiplayer` start
skips the lobby's parameter writes"): the lobby's fog, shroud, village
gold and support go in with `--parm`, for the sides Wesnoth's own
preprocessing shows without them; and ours is built at the experience
modifier the engine applies (100 on the command line, read from a unit
type whose base is 100), while the lobby's value is its own field.

**Found and fixed:**

- Three minis declare `fog=no` and were played under fog (above, W5).
- The statues of Caves of the Basilisk, Sullas Ruins and Thousand Stings
  Garrison carry modifications that leave them 1 hp and no moves (a
  custom `remove_hp` trait, or the same effects in an `[object]`).
  Units placed in a `[side]` block got none of them, and an event's
  `[unit]` got its custom traits but not its objects;
  `scenario_events.own_modification_effects` now serves both paths.
  The manifest had the effects as IGNORED, "presentation".
- Two representation differences, mapped rather than changed:
  `not_living` is the engine's name for its three parts, and a
  guardian's status is our `_ai_guardian` flag.

**Not built:** a headless run. The Windows binary opens a window even
minimized, so a sweep runs when the user says (28 launches, about 25
minutes); `--frames DIR` saves the engine's reports so later changes
can be rechecked with `--from-frames` without launching it.

## Two decisions, taken

1. **The three drill templates**: deleted (user ruling 2026-09-22).
2. **The corpus sweep W1 owed**: not run. Reconstruction reads only
   events and time areas from our expander, and diffing both, old
   against new, over all 31 corpus scenarios shows only display text
   and one heals block with identical sim abilities, so the sweep would
   pass both ways and certify nothing (review below).

---

# Review of the uncommitted work (2026-09-23, Opus 5.5)

A second model re-checked the W0-W6 work before anything was
committed. Most of it held; the classification manifest did not, and
checking it turned up one hazard that had nearly shipped.

## What held

- **Python and Rust compute the same time-of-day features.** Same
  index formula (`(max(turn,1) - 1 + max(offset,0)) % 6`), same cycle
  constant, both divide by 25 (`global_feature_row`), and Python hands
  the kernel all eight values. Checked in source because nothing local
  can run the kernel; two Rust doc comments still said six globals and
  are fixed.
- **The neutral-AI stationary check behaves in real play**, not just on
  a freshly built state: over three full rounds on an enclave map, a
  guardian map and Modified_Tiny_Close_Relation, the turn-refresh pin
  is already applied when the check runs, and nothing warns.
- **The parser's translatable-marker strip is structurally a no-op on
  replays.** Replays DO carry `_ "` markers -- 400 of 400 sampled, the
  builder's "saves carry none" was wrong and is corrected -- but old
  and new parsers produce the same tree on all 400. Only display
  values change, and no reader consumes those keys.

## Fixed

- **Nested actions lost their scenario.** Tags inside `[if]`/`[switch]`
  bodies dispatch from handlers that are never told the scenario, so an
  unmodelled nested tag warned as `(tag, "")`; reports dedupe on that
  key, so a second scenario hitting the same nested tag stayed silent.
  A ContextVar set by `fire_event` carries it. The test fails with the
  ContextVar neutralised.

## The hazard: Hornshark Island's Mermaids healed correctly by luck

Diffing what reconstruction reads from each corpus scenario, old
expander against new, found 14 of 31 scenarios changed. Almost all of
it was display text. The exception: on Hornshark Island (248 corpus
games) the preplaced Mermaid Initiates' `{ABILITY_HEALS}` expanded,
under the old expander, to an **empty `[heals]` block** -- the
`INTERNAL:` name collapse had sent `{INTERNAL:ABILITY_HEALS_NO_NOTES}`
to whichever `INTERNAL` body was defined last.

The sim still gave them `heals_4`, because the reader defaulted a
missing `value` to 4, which happens to equal the macro. Measured: the
old and new builds give both Mermaids identical abilities. So nothing
shipped wrong. But a map using `{ABILITY_HEALS_8}` would have healed
half what Wesnoth does, silently. And the default was not
engine-faithful anyway: the engine builds the heal effect with a
default of 0 (`src/actions/heal.cpp:211`) and a block without `value=`
contributes nothing (`src/units/abilities.cpp:2061`).

`scenario_events._heals_ability` now reads the value, returns no heal
for a value-less block and warns (raising under `WESNOTH_STRICT_WML`),
and warns on any value other than the 4 and 8 the sim models.

## The manifest overclaimed

W3's binding test checked that a MODELLED entry's reader **exists**,
not that it **reads the attribute**. An audit extracting each reader's
own source found **22 of 72 MODELLED entries bound to a function that
never names the attribute.** Some were just imprecise (map_file is read
by the caller, `[modify_unit] moves` through a table). Others were
false: `read_side` does not read `[side] id`, and `random_traits`,
`affect_self` and `cumulative` are read by nothing in the project.

The manifest was also blind by construction in two ways. It covered
the 28-map pool, while reconstruction loads corpus maps outside it
through the same expander (Cynsaun Battlefield, Hornshark Island,
around_mini). And it keyed on raw nesting, so a `[unit]` inside
`[switch][case]` was a new, unclassified path although
`_apply_action` dispatches it with the same handler. Hornshark's
Mermaids sat exactly there.

Now: the surface covers 31 scenarios with control flow folded out of
the path; a MODELLED reader must name its attribute in its own source
or say in `generic` why the read happens elsewhere; and the test
replays three of the false bindings. **MODELLED 68, IGNORED 121,
SUBSTITUTED 25, UNKNOWN 0** over 214 pairs. SUBSTITUTED went from 8 to
25: seventeen behaviours that were claimed as read are now visibly
unread, each with the precondition that makes that safe.

## The pattern, six more times

Each of these is right today for a reason nobody had written down,
measured on 2026-09-23:

| unread | why it is still right |
|---|---|
| `[heals] value` defaulted to 4 | equals `{ABILITY_HEALS}`; the only carrier |
| `random_traits` | the Tentacles carrying `=yes` are race=monster, `num_traits=0` (`units.cfg:300`) |
| `apply_to=movement_costs` (castle=99) | carried only by guardian Tentacles that never move |
| `[object] duration` | all 8 objects in the corpus scenarios say `forever` |
| `[object][filter] type=Bowman` | x, y alone pin the Bowman, over all 36 faction pairings |
| `#ifdef` / `#ifndef` | FIXED in 0.2.1: now evaluated as the engine does (see below) |

## Done after the review: preprocessor conditionals are evaluated (0.2.1)

The expander used to resolve no `#ifdef` at all: the comment stripper
kept the directive lines, the WML parser skipped them, and every
branch's content survived. `scenario_events.evaluate_conditionals` now
follows the engine (`src/serialization/preprocessor.cpp:1322-1400`):
`#ifdef` / `#ifndef` / `#else` / `#endif`, nested, against a define set
that holds `MULTIPLAYER`, the scenario's own `define=` (found by a
pre-scan, since Wesnoth applies it before preprocessing) and every
macro `#define`d so far -- the engine keeps macros and symbols in one
map, so `#ifdef SOME_MACRO` turns true at the line that defines it.
`#ifhave` / `#ifver` and malformed nesting raise; nothing we expand
uses them.

All three text sources -- core macros, an add-on's utility files, the
scenario -- go through one `_preprocess_text`, where they used to
repeat the two strip calls and skip this step.

What it changed: the core macros' difficulty branches
(`QUANTITY`, `QUANTITY4`: `#ifdef EASY` / `NORMAL` / `HARD` /
`NIGHTMARE`) and the AI controller's `#ifdef __UNUSED` / `#ifndef
MULTIPLAYER` blocks now expand to what a multiplayer game gets.
What it did not change: anything we build. The expansion diff still
shows only the two accepted clusters, the 28-scenario and 120-replay
snapshots are unchanged, and Hornshark Island -- the one scenario with
a conditional of its own -- still defines MODIFY_BOWMAN and still
gives both Loyalist Bowmen firststrike. The new tests fail with the
evaluator switched off.
