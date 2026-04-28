# CLAUDE.md — Wesnoth AI

## Project

A reinforcement-learning AI for *Battle for Wesnoth*, trained by self-play,
aiming to be (a) competitively strong against players and (b) readable
enough that humans can study its strategies. Customization is a first-class
goal: we need to easily incentivize unorthodox strategies, fine-tune on
human game logs, force specific openers, etc. — so code that gates behavior
behind config is preferred to code that gates behavior behind weights.

- **Language:** Python 3.11+
- **ML:** PyTorch
- **Game:** Wesnoth 1.18.x (Steam install on Windows)
- **Run:** `python main.py` (see `--help`)
- **Test:** `pytest` (tests are Python-only; they exercise the WML
  parser and Lua-file generation with synthetic data — they do NOT spin
  up real Wesnoth)

### Wesnoth source pinning

`wesnoth_src/` MUST stay on the **1.18.4 tag** (`git checkout 1.18.4` in
that directory). Wesnoth's `master` branch is 1.19.x development and
unit stats DRIFT between releases — e.g., in master the Ghoul gained
a `[resistance] pierce=90` override that doesn't exist in 1.18.4
(where pierce inherits 70 from the gruefoot movement_type). Using
master's stats made our combat reconstruction overdamage units that
should have lived. Re-run `python tools/scrape_unit_stats.py
wesnoth_src unit_stats.json` after any version change.

Most replays in `replays_raw/` are from 1.18.x clients; pin
accordingly. If a replay's `[scenario] version=` says something
other than 1.18.x, scrape from that version's tag instead.

## Current status (2026-04-23)

**The Python ↔ Wesnoth pipeline has never delivered a single game
state.** Every run in `logs/` ends with `Timeout waiting for state`. The
stdout-based state channel does not work on Windows under
`subprocess.Popen(..., stdout=PIPE)`. Unblocking this is the top
priority; everything else (training signal, model design, scaling) is
downstream of it.

The model, encoder, replay buffer, training step, reward shaping, and
multi-process harness all work in isolation on synthetic data, but none
of that matters until the IPC link is live.

## Architecture

### Two sides of a bridge

**Python side (orchestrator + learner):**
- `main.py` — entry point, setup checks, launcher.
- `game_manager.py` — runs N Wesnoth games in parallel, owns the model,
  replay buffer, and trainer.
- `wesnoth_interface.py` — one Wesnoth process per game; reads state,
  writes actions.
- `state_converter.py` — Wesnoth data format → `GameState`.
- `state_encodings.py` — `GameState` → tensors.
- `transformer.py` — policy/value network.
- `action_selector.py` — samples action from logits; minimal legality
  filtering (will tighten).
- `training.py` — policy + value loss, AdamW.
- `classes.py` — `GameState`, `Unit`, `Hex`, `Map`, `Experience`, etc.
- `constants.py` — all tunables (paths, hyperparameters, terrain codes).

**Wesnoth side (add-on at `add-ons/wesnoth_ai/`):**
- `_main.cfg` loads the scenario.
- `ai_config.cfg` wires two Lua candidate actions into Wesnoth's AI:
  one to send state, one to poll for and execute an action.
- `scenarios/training_scenario.cfg` — 2p_Caves_of_the_Basilisk,
  Knalgan vs Drakes, both sides `controller=ai`, both using our CAs.
- `lua/state_collector.lua` — serializes units, hexes, fog, sides.
- `lua/ca_state_sender.lua` — runs `state_collector`, emits the state.
- `lua/ca_action_executor.lua` — blocks until a new action arrives,
  then dispatches.
- `lua/action_executor.lua` — move / attack / recruit / recall /
  end_turn, using the Wesnoth `ai.*` API.

### IPC (to be revisited in Phase 1)

Today: state via **stdout** with `===WML_STATE_BEGIN===` markers;
actions via a **Lua file** that Wesnoth's CA polls every 50 ms.
The stdout half is the one that does not work. The target replacement
is probably file-based for both directions, or `wesnoth --plugin` if its
Windows story is cleaner.

### Coordinates

- **Wesnoth uses 1-indexed hex coordinates.**
- **Python uses 0-indexed hex coordinates everywhere internally.**
- Conversion happens in `state_converter.py` (both directions). Keep it
  there; do not sprinkle `±1` around the codebase.

## Architecture Principles

### 1. Start simple, measure, then complicate
This project has suffered from premature architectural ambition. We
have a transformer with a memory module, a consistency loss, and
parallel games — and zero training steps. Default stance: the simplest
thing that measurably improves over the previous baseline wins. Add
complexity only when a metric demands it.

### 2. Self-play is non-negotiable
The learning signal comes from self-play. Bootstrapping methods
(imitation from the built-in AI, human games) may be used to warm-start,
but the end state is self-play. Do not design architectures that
preclude it.

### 3. Config-driven, customizable
Rewards, openers, strategic biases, training curriculum — all must live
in data/config, not buried in network weights or scattered constants.
When you add a new behavior, ask: "could a modder flip this without
touching model code?"

### 4. Python and Wesnoth speak through a narrow waist
All state crosses the bridge as one well-defined serialization, all
actions as one well-defined schema. Keep the Lua side dumb: collect
state, execute actions, no game-logic decisions. Keep the Python side
agnostic to IPC details (one `read_state()` / `send_action()` call).

### 5. Failures are visible
Wesnoth hangs are the default failure mode of this system. Any
Python-side wait must have a finite timeout and log which stage timed
out. Lua errors must reach Python's log, not die silently inside a
`pcall`.

### 6. Legality mask = pure function of OBSERVABLE STATE
The action sampler's "legality mask" answers exactly one question:
**what can the policy validly attempt right now, given the
information it has?** It is a pure function of the observable state.

Observable state has two pieces:
1. **Visible game state** — what the encoder sees. Includes
   own-side fog hexes (the encoder retains them after the
   2026-04-28 fix), but NOT enemy units hidden in fog and NOT
   any god-view information from the simulator.
2. **Per-turn rejection history** — the set of hexes where a
   recruit attempt has bounced this turn, stashed on
   `gs.global_info._recruit_rejected_hexes`. Cleared at
   `init_side` (so each side starts a turn with a fresh slate).

What this resolves: the mask is NOT "what the engine will accept"
(which would require god-view fog truth, cheating) AND it is NOT
just "what the model wants to attempt" (which would let the model
infinite-loop on the same fog-hidden hex). It is "what the model
*can* validly attempt given everything it has observed so far,"
which gives the model the same information a human player has.

Concretely:

  - Fog castle hexes ARE legal recruit targets (the model can
    attempt them; like a human, it can't see what's there until
    it tries).
  - After a rejection, that hex becomes illegal AND a per-hex
    "recruit_rejected" bit appears in the encoder feature -- the
    mask consults the rejection set, the model sees the bit; both
    read the same state.
  - Next turn, rejection history clears. The hex is legal again
    (the enemy may have moved away).

Designs that violate this contract are bugs. In particular:

  - **God-view masking is forbidden.** Even though the simulator
    has fog truth, the mask must not consult it. The model must
    play with the same information a human would have.
  - **Rejection history is per-turn.** Persisting it across turns
    would model long-term knowledge that the human player doesn't
    have (the enemy could have moved).
  - **The mask must be a pure function of observable state.** Two
    decisions on the same observable state produce identical
    masks. Mutable state outside `gs.global_info` (e.g. cached
    per-call counters) is forbidden.

This contract applies to all action types -- attack, move,
recruit, end_turn -- not just recruits. The recruit-rejected
case is the most prominent example, but the rule generalizes:
if a future action category needs "we tried this and it
bounced" tracking, it lives on `gs.global_info`, clears at the
right turn boundary, and is mirrored in encoder features.

## Code Style

### File Size
Target ~600 lines. Split by responsibility when a file grows past that.

### Naming
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private members: `_leading_underscore`
- Lua: same conventions (Lua allows `snake_case` fine)

### Imports
- No unused imports.
- No circular imports — if A needs data from B, pass it explicitly.
- `TYPE_CHECKING` blocks for type-only circular deps.

### Data patterns
- Structured data → `@dataclass`, not raw dicts.
- Action payloads on the wire are dicts (they cross the Lua boundary);
  internally prefer typed structures.
- Material/faction properties → LUTs in config, not `if`-chains.

### Encapsulation
- Python systems talk through explicit APIs on `GameManager` /
  `WesnothGame`, not by reaching into private attributes.
- Lua code never decides game logic; it just serializes state and
  executes actions the Python side chose.

## Testing

### Philosophy
Tests exist to catch regressions you'd otherwise only notice after
burning an overnight training run. Prefer few behavioral tests over
many line-coverage tests.

### What tests we have (and what they are NOT)
- `test_wml_parser.py`, `test_integration.py`, `test_lua_actions.py`
  are **Python unit tests with synthetic inputs**. Despite the name,
  `test_integration.py` does NOT launch Wesnoth. A real end-to-end
  test requires a live Wesnoth process — we don't have one yet.

### Guidelines
- Run `pytest` after changes.
- Use `constants.py` values in assertions (not hardcoded duplicates).
- Never weaken a test without explicit user confirmation. A failing
  test is a signal — find the root cause first.

## Working Style

- **High autonomy** on reversible local work (edits, tests, reads).
- **Ask before**: committing, force-pushing, changing branches,
  deleting tracked files, making architectural changes (new IPC,
  replacing the model, etc.).
- **Give best effort**: production-quality code with edge cases handled,
  not sketches.
- **Research first** on non-trivial Wesnoth internals (WML, Lua API,
  scenario/add-on loading, `--plugin`, stdout behavior). The
  [Wesnoth wiki](https://wiki.wesnoth.org/) and
  [Lua API reference](https://wiki.wesnoth.org/LuaAPI) are authoritative.
- **Prefer removing over adding.** This codebase is recovering from
  bloat. When a feature is load-bearing, we'll re-add it with evidence.
