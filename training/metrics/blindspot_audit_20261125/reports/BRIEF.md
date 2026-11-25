# Blind-spot audit of the Wesnoth AI's inputs and outputs (2026-11-25)

You are one of seven auditors. Each has one part; yours is named in your
task. This file is the common brief.

## What is audited

A reinforcement-learning player for *Battle for Wesnoth* 1.18, trained by
self-play. The repository is
`C:\Users\momom2\Desktop\perso\projects\wesnoth_ai` (branch `main`, version
0.36.0). Read `CLAUDE.md` there first: it holds the code map and the
project's rules.

The games it plays: two-player multiplayer on the 21 maps of the Ladder
pool (`wesnoth_ai/rules/scenario_pool.py`), default-era factions drawn at
random, with fog. Settings you need beyond that (shroud, turn limits,
gold, income, experience modifier) are for you to read from the code, not
to assume.

## The standard

The owner's ruling: **"For each item of the present state that a player
sees, the network sees it too."** A blind spot is anything a human player
of the same side sees on the screen, can look up in the game's own
interface during play, or can order, which the network cannot see or
cannot order.

The reverse also counts (CLAUDE.md, Principle 6): the network must see
nothing a player cannot, such as an enemy hidden in fog or anything else
from the simulator's god view. Report such a leak as a finding too.

## Where the input and the output are defined

- `wesnoth_ai/encoder.py`: the tokens and their columns (global token,
  hex tokens, unit rows, recruit rows), with layout comments. The unit
  row is at version 2: 341 columns.
- `rust/wesnoth_core/src/`: the Rust core that fills the tensors and the
  legality masks, the state of record. Start with `core_parity.rs`,
  `core_encode.rs`, `observe.rs`, `core_masks.rs`, `lib.rs`.
  `wesnoth_ai/game_core.py` is the Python adapter.
- `wesnoth_ai/model.py`: the transformer, its heads, its memory.
  `wesnoth_ai/action_sampler.py` and `wesnoth_ai/transformer_policy.py`:
  how actions are scored and chosen.
- `tools/wesnoth_sim.py`: the simulator's step loop.
- `docs/wesnoth_rules.md`: Wesnoth rules already established, with source
  quotes. `docs/input_facts_design_20261124.md`: the previous input audit
  and what it added. `docs/status.md`: the project's state.
- Wesnoth's own data (WML and Lua): `wesnoth_src/data/` in the
  repository.
- Wesnoth's C++ at the 1.18.4 tag:
  `https://raw.githubusercontent.com/wesnoth/wesnoth/1.18.4/src/<path>`.
  Useful starting points for "what a player sees": `src/reports.cpp` (the
  sidebar and status bar), `data/themes/default.cfg` (which reports the
  default theme shows), `src/gui/dialogs/` (the attack dialog, the status
  table `game_stats.cpp`, the unit list), `src/game_display.cpp` and
  `src/display.cpp` (reach and attack overlays, fog), `src/hotkey/` and
  `src/menu_events.cpp` (what a player can order or look up).

Do not guess how Wesnoth behaves. Check the source and cite it.

## Already known: do not report these as new

- A unit's weapon is scored without the target it strikes (the design's
  Part 3, not built).
- Level, the chance to be hit (where the unit stands and by terrain),
  movement costs and each fight's numbers were added on 2026-11-25 (unit
  row version 2). The fight columns cover only enemies on the six hexes
  touching the unit, for the side's own units that can still attack. A
  unit attacks only such an enemy: there is no attack ordered from a
  distance.
- The time of day is present, globally (this turn's and next turn's
  bonus) and per hex.
- The owner's rulings: no token per possible attack; no worked-out
  outcome of a fight (expected or most likely) in the input; no hand-set
  rule or table beside the network; one learning rate. Do not propose
  what these rule out. You may still report a blind spot whose only cure
  would cross one of them, saying so.

If you find one of these "known" statements is itself wrong, that is a
finding.

## Rules for you

- **Read only.** Do not edit, create or delete anything in the
  repository. Run no git command that changes state.
- **A training run is live on this machine and holds the GPU.** Do not
  run pytest. Do not run any Python that imports torch or the project's
  packages. Nothing on the GPU, no installs, no builds. Reading and
  searching files is your method. If a claim needs a run to settle, put
  it under "To check by a run" with the exact check to make; the main
  session will run it.
- Do not launch Wesnoth.
- When you fetch Wesnoth's source from GitHub, send nothing about this
  project in the request.
- **Don't assert without checking.** Every finding cites the file and
  line where you read the column or the code, or, for an absence, what
  you searched for and where. Mark each finding "verified by reading" or
  "inferred". A wrong finding costs more than a missing one: the owner
  acts on these.
- Stay in your part. If you see something that belongs to another part,
  note it in one line under "For other parts".

## What to deliver

Write your full report to the file named in your task, in this folder
(`...\scratchpad\audit_20261125\`), in Markdown:

1. **Findings**, most serious first. For each:
   - what the player sees or can order, where in the interface, and the
     source that shows it (file:line);
   - what the network gets instead (file:line), or the search that shows
     it gets nothing;
   - why it matters for play, in one or two sentences, with a concrete
     game situation;
   - confidence: "verified by reading" or "inferred";
   - how a run would confirm it.
2. **Coverage**: every item you checked and found present and right, one
   line each with file:line. The audit's completeness is judged from this
   list, so make it complete for your part.
3. **To check by a run.**
4. **Could not determine.**
5. **For other parts.**

Then reply with a summary under 300 words: how many findings, and the top
ones in one line each.
