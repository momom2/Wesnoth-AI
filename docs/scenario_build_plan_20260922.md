# Dev plan: one scenario builder, checked against the game (2026-09-22)

## Objective

Build a sim scenario from a scenario request and Wesnoth's own files, by
one path, such that **anything we fail to process and anything we get
wrong is detected automatically** rather than found as a bug months
later. No Wesnoth install at runtime.

This replaces defect-by-defect patching. Every fidelity bug found in
the 2026-09-21/22 work had the same shape: the scenario declared
something, we did not read it, and nothing said so. Hardcoded village
gold, unread per-side fog, hide-cover globs, `[effect]` ids, starting
gold declared inside a macro. A list of fixes does not stop the next
one; two automatic detectors do.

## The two detectors, which come first

1. **Unprocessed input.** Every tag and attribute of the expanded
   scenario is classified as MODELLED, IGNORED (by name, on purpose)
   or UNKNOWN. UNKNOWN fails the build. The classification is checked
   against what the files actually contain, not against what we
   remembered to handle, so a new scenario or a Wesnoth upgrade
   surfaces on the next test run.
2. **Discrepancy.** The state we build is compared against the state
   the real engine builds for the same scenario. The precedent is
   `tools/hidden_units_oracle.py`, which settled the hide-cover rules
   at 54 of 54 scripted positions; the same harness can start a
   scenario and dump what the engine made of it.

Everything else in this plan is downstream of those two.

## Architecture

Three stages, explicit contracts, one implementation each.

| stage | input | output | its oracle |
|---|---|---|---|
| **Expand** | scenario id, WML sources | fully expanded WML tree | Wesnoth's own preprocessor |
| **Classify** | expanded tree | modelled subset; ignored subset; UNKNOWN must be empty | the census of the pool's own files |
| **Interpret** | modelled subset | initial state + event program | the real engine, per scenario |

The sim's other contract, reconstruction from a replay, keeps its own
oracle (`tools/diff_replay.py` over the corpus) and is unchanged by
this plan. Both paths already share one WML reader (`tools/wml_state.py`,
commits 09bdae1 and be00037).

## What is already measured

- Wesnoth's preprocessor costs 60-90 ms per scenario, about 2 seconds
  for the 28-map pool, run offline. Speed is not a reason to
  reimplement it.
- Our expander does not expand `{DEFAULT_SCHEDULE}`: one substitution
  pass deletes it, leaving zero `[time]` blocks where the game emits
  six. Every macro body is in our cache, so this is recursion, not
  missing data.
- The committed templates are the game's own preprocessor output: 32
  files, 544 KB, and all 28 pool templates carry their map inline, so
  they are self-contained.
- The template builder has been unimportable since 2026-08-10, when
  the drill deletion left `DRILL_SCENARIO_IDS` dangling. The templates
  cannot be regenerated today.
- Census of the pool's expanded WML: top-level `music` 700, `time`
  168, `event` 114, `item` 92, `terrain_graphics` 26, `side` 10,
  `time_area` 3. Event children that change play: `unit`, `terrain`,
  `object`, `end_turn`, `time_area`, `store_unit`, `store_locations`,
  `foreach`, `clear_variable`. Three scenarios carry `[side][ai]`.

## Work items

Ordered so each one's acceptance test exists before the work it
gates. All local; no box.

### W1. Scenario-init oracle against the real engine

Start each pool scenario in real Wesnoth, dump the initial state the
engine built, compare field by field with
`build_scenario_gamestate`'s. The collector already reports gold,
base income, villages held, fog, faction, the hex grid and every
unit's stats; the gaps to add are the schedule and per-hex village
ownership.

This is the acceptance test for W3 through W6, so it is built first
and its first run is a measurement of where we stand, not a gate.

*Acceptance:* a record under `training/metrics/fidelity/` naming every
field compared and the agreeing fraction per scenario, plus the list
of fields the engine exposes that we chose not to compare and why.

### W2. Exhaustive classification, failing on unknown

A manifest mapping every tag and attribute in the pool's expanded WML
to MODELLED, IGNORED or UNKNOWN, with the build refusing UNKNOWN. The
manifest is data, not code, so a modder can read it; the test derives
the census from the files and fails when something appears that the
manifest does not cover.

*Acceptance:* zero UNKNOWN across all 28 scenarios; an injected
unknown tag fails the build; an injected unknown attribute on a
modelled tag fails too.

### W3. Repair and regenerate the templates

Fix the builder's dangling import, keep `[side]` 1 and 2 instead of
stripping them (the exporter strips at emit time, where it already
anchors its splice), regenerate all 28.

*Acceptance:* the builder runs; the regenerated templates differ from
the committed ones only in the player sides and whatever W2 flags;
the diff is reviewed line by line and recorded.

### W4. One expansion source for generation

Generation reads the expanded scenario only. Our macro expander
leaves that path, and with it the last place where two renderings of
one `.cfg` could disagree.

*Acceptance:* `tests/data/scenario_state_snapshot.json` moves only
where intended, W1's oracle agreement does not regress, and the
`.cfg` trees are no longer read at generation time.

### W5. Our own expander, differentially verified

Implement the preprocessor requirements: recursive macro expansion,
parameterised macros, file inclusion, `#ifdef`/`#ifndef`/`#else`/
`#endif` under `MULTIPLAYER`, `#undef`, `[+tag]` merges, parallel
assignment. Verify it against Wesnoth's output on all 28 scenarios,
byte-for-byte on the modelled subset W2 defines.

Only this item lets the build step leave a machine with Wesnoth
installed. It is deliberately last: until it passes, committing the
game's output is strictly safer, and the committed output is what
makes the comparison possible.

*Acceptance:* 28 of 28 identical on the modelled subset, as a test
rather than a one-off run.

### W6. Assumptions that become reads

Where the sim assumes what the scenario declares, read it instead, or
record the assumption as a tested choice. Known today: the six-slot
day cycle is hardcoded and every pool scenario happens to use exactly
it; the experience modifier has two homes, the template builder's
injected 70 and our fallback constant; the engine injects a
quick-leader prestart and a turns-over-advantage `time over` that the
sim does not execute.

Each is decided by W1's oracle rather than by argument: if the engine's
initial state differs from ours, the assumption is wrong and becomes a
read; if it does not, the assumption is documented with the evidence.

*Acceptance:* no assumption without either a read or a record of the
oracle run that justified it.

## Decision this plan defers on purpose

Whether to keep Wesnoth as the expander or ship our own. W5 answers it
with a differential result instead of a preference. Until then the
committed templates satisfy the runtime constraint, and the box never
sees a Wesnoth install.

## What generation chooses, and the scenario does not

Factions, leaders and recruit lists come from the era. Fog, the
time-of-day start, gold and income may be overridden per run. The
scenario supplies the defaults for the last four; our choice wins
where we make one, and the choice is config, not an accident of not
reading an attribute.
