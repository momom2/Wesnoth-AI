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
- **`[side] fog=yes` is declared on all 56 player sides.** Reading it
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
