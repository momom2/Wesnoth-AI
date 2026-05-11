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

## Current status (2026-05-10)

**The simulator is the production training path.** `tools/wesnoth_sim.py`
is a pure-Python headless reimplementation of Wesnoth 1.18.4's game
logic, cluster-portable and ~1000× faster than driving a Wesnoth
subprocess. Combat math is bit-exact verified (731/731 strikes
matched against `[mp_checkup]` oracle on strict-sync replays);
full-replay reconstruction at 98.57% clean on the 4,841-replay
competitive-2p corpus after the Stages 1–21 fidelity sweep
(documented under BACKLOG.md).

**Self-play training is end-to-end ready.** `tools/sim_self_play.py`
drives self-play in the simulator with either REINFORCE or MCTS
(`--mcts` flag, AlphaZero-style PUCT + virtual loss + transposition
table). The cluster job (`cluster/job_selfplay.sbatch`) runs on
ENSTA-l40s, auto-resumes from `sim_selfplay.pt` or warm-starts
from the highest `supervised_epoch*.pt`. GUI controls in
`cluster/gui.pyw`.

**Distributional value head + cliffness signal landed 2026-05-10.**
C51 head (K=51 atoms, `[V_MIN, V_MAX] = [-1, +1]`) replaces the
prior tanh-bounded scalar value. `output.cliffness = std(Z(s))`
exposes the network's per-state value uncertainty. Two MCTS
consumers are wired but default OFF pending calibration: Bayesian-
precision bootstrap weighting in `_backup`, and adaptive sim budget
based on root cliffness. Always-on root-cliffness debug log
collects distributions for tuning.

**The live-Wesnoth IPC bridge is now eval-only.** `game_manager.py`
and `main.py --display` were retired 2026-05-11; `tools/sim_demo_game.py`
+ `tools/sim_to_replay.export_replay_from_scratch` cover the demo
case by exporting a Wesnoth-loadable `.bz2`. The only remaining live-
Wesnoth consumer is `tools/eval_vs_builtin.py` (via `tools/eval_runner.py`
+ `wesnoth_interface.py`), which needs real Wesnoth subprocesses to
pit the trained model against the built-in RCA AI.

## Architecture

### Two paths that share encoder + model + trainer

**Production path: in-process simulator (cluster + most local work).**
- `tools/wesnoth_sim.py` — pure-Python game logic. Reuses the
  replay-reconstruction machinery from `tools/replay_dataset.py`
  (which is bit-exact against Wesnoth via `[mp_checkup]` oracle on
  combat); just swaps the data source from "WML command stream"
  to "policy queries."
- `tools/sim_self_play.py` — self-play training entry point. Drives
  N games per iteration through `WesnothSim`, calls `policy.observe`
  for shaping rewards, applies one gradient update per iteration via
  `policy.train_step`. `--mcts` flag swaps in `MCTSPolicy`.
- `tools/scenario_pool.py` / `tools/scenarios.py` — scenario
  randomization for training (Ladder Era 21-map whitelist, faction
  randomization with optional `--forced-faction` lock).
- `tools/mcts.py` / `tools/mcts_policy.py` — MCTS implementation
  and the MCTSPolicy adapter that wraps TransformerPolicy.
- `cluster/job_selfplay.sbatch` / `cluster/gui.pyw` — cluster
  orchestration + Tk GUI.

**Live-Wesnoth path (eval only).**
- `main.py` — setup / maintenance CLI (`--check-setup`,
  `--clean-games`). No longer drives training or `--display`.
- `wesnoth_interface.py` — one Wesnoth process per eval game; state
  channel uses `std_print` → log-file tail (CA-blacklist bypass via
  custom Lua AI stage); actions written atomically as `action.lua`
  and read via `wesnoth.read_file`.
- `add-ons/wesnoth_ai/` — Lua side: `lua/state_collector.lua`,
  `lua/turn_stage.lua` (custom AI stage replacing default RCA so
  failed actions don't blacklist the CA), `lua/action_executor.lua`,
  `lua/json_encoder.lua`. `scenarios/training_scenario.cfg`.
- `tools/eval_vs_builtin.py` + `tools/eval_runner.py` — pits the
  trained model against Wesnoth's default RCA AI across a
  (map × matchup × side-swap) matrix.
- `tools/sim_demo_game.py` — headless one-game demo via the sim;
  exports a Wesnoth-loadable `.bz2` via
  `sim_to_replay.export_replay_from_scratch` (composes save WML
  from `wesnoth_src/` templates, no replays_raw/ dependency).

**Shared by both paths:**
- `classes.py` — `GameState`, `Unit`, `Hex`, `Map`, `SideInfo`,
  `state_key`.
- `encoder.py` — `GameState` → tensors (per-unit, per-hex, recruit
  phantom features, global features).
- `model.py` — `WesnothModel` transformer with distributional C51
  value head; emits `ModelOutput(actor_logits, type_logits,
  target_logits, weapon_logits, value, value_logits, cliffness, ...)`.
- `action_sampler.py` — legal-action enumeration with priors;
  combat-oracle attack-bias on target logits.
- `transformer_policy.py` — Policy adapter (`select_action`,
  `observe`, `train_step`, `save_checkpoint`, `finalize_game`).
- `trainer.py` — REINFORCE + value baseline (`step`) and
  AlphaZero-style soft-target distillation (`step_mcts`); both
  use the categorical CE value loss against C51 atom projections.
- `rewards.py` / `cluster/configs/reward_selfplay.json` — shaping
  reward (terminal ±1, gold/damage/village deltas, per-turn penalty,
  unit-type bonuses, turn-conditional bonuses).

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

### 4. Simulator parity is bit-exact for combat
The simulator's combat math is verified against Wesnoth's own
`[mp_checkup]` oracle on strict-sync replays (731/731 strikes
matched). Any sim change that touches combat, healing, or
advancement must keep that parity. `tools/diff_replay.py` is the
regression check (runs the simulator over a corpus, compares
against the recorded WML command stream); aim to keep clean rate
≥98.5% on competitive-2p. New scenario events go in
`tools/scenario_events.py`; new abilities in `tools/abilities.py`;
both with citations to `wesnoth_src/` file:line.

When live Wesnoth is in the loop (display, eval), the same
narrow-waist principle applies: state crosses the bridge as one
well-defined serialization, actions as one schema, Lua side stays
dumb. But that path is no longer how training data is generated.

### 5. Failures are visible
Both paths log timeouts and stage-of-failure. The simulator returns
typed errors (e.g. `"recruit:insufficient_gold"`) that
`sim_self_play.py` surfaces in the per-game summary. The bridge
path (display, eval) wraps any Python wait in a finite timeout and
logs which stage timed out; Lua errors reach the Python log, not
silently die inside a `pcall`.

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
- **Never run more than one pytest invocation at a time.** Each
  pytest spawns a Python process that imports torch + the model;
  parallel runs balloon memory (5+ GB per process) and a stuck
  test compounds the problem. Wait for each command to fully
  return (foreground) before launching the next. If a test
  hangs, kill the process before starting another — don't queue
  a second behind it. No "let me also kick off X while we wait"
  patterns. (Lesson from 2026-05-10: four parallel pytest jobs
  stuck on a hanging `_select_one` loop produced multiple
  multi-GB zombie Python processes that locked up the user's
  machine.)
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
- **Don't guess at Wesnoth WML attrs / engine semantics — check the
  source.** The wiki sometimes lags, has edge cases wrong, or omits
  attrs. `wesnoth_src/` is pinned to 1.18.4 and is authoritative for
  what the engine actually does. When you'd otherwise hand-wave
  ("`income=` is probably an offset"), grep `wesnoth_src/src/` first
  and cite line numbers in comments / commits.
- **Don't assert without checking.** This applies to factual claims
  about Wesnoth (rules, mechanics, unit stats) AND to claims about
  our own code ("the function does X", "this list covers all cases",
  "the heuristic always fires correctly"). If you find yourself about
  to write a confident-sounding sentence in a commit message, code
  comment, or message to the user, pause: have you actually verified
  the claim with a grep / read / test? If not, either verify first OR
  hedge the language ("I believe X holds because Y; not yet verified").
  Especially: when listing a closed set ("the three races that get
  undrainable are undead, mechanical, elemental"), grep the source for
  the relevant marker and confirm the count matches before publishing.
- **`docs/wesnoth_rules.md` is the source-of-truth catalog.** Read
  it BEFORE researching a Wesnoth rule from scratch — most
  established rules are pinned there with verbatim source quotes.
  When you establish a new rule (or correct an existing one), add
  / edit the entry in `docs/wesnoth_rules.md`. Required: file:line
  citation, verbatim quote of the enforcing code, and (when the
  rule wasn't where you'd naively expect) a "why non-obvious" note.
  Grep recipes and a file map for common Wesnoth-source questions
  also live there. Treat the doc as a force multiplier: each rule
  added saves the next exploration session hours, and the doc
  prevents truth-drift across sessions.
- **Wesnoth rules can live in C++, Lua, OR WML — search all three.**
  Common gotcha: a rule we're hunting in `wesnoth_src/src/` is
  actually in `wesnoth_src/data/multiplayer/eras.lua` or a WML
  macro under `wesnoth_src/data/core/macros/`. After grepping `src/`,
  always also grep `data/multiplayer/`, `data/core/macros/`,
  `data/lua/`. Rules with a "post-pass" feel (applied after unit
  setup) often hide in `[event]name=prestart` Lua callbacks.
- **`changelog.md` is HISTORICAL — verify against current source.**
  Old changelog entries describe behavior at THAT version, which
  may have changed since. Cross-check any changelog quote against
  the live `wesnoth_src/` code path before treating it as authority.
- **`docs/design_constants.md` catalogues DERIVED numerical
  constants** (not arbitrary tuning knobs). Anything with a
  derivation — math, measurement, fixed external standard —
  belongs there rather than buried in a one-line code comment.
  When you find yourself writing "where does this 0.577 come
  from?" or "why exactly 51 atoms?", the rationale belongs in
  `docs/design_constants.md`; cite it from the code with a short
  comment + cross-reference. Pure tuning knobs (learning rate,
  c_puct, etc.) stay in `constants.py` with their own comment
  block — those aren't derived, they're picked, and the picking
  rationale (which may be "AlphaZero paper" or "experiment
  pending") stays nearby.
- **Magic-number principle, more generally:** if a number's
  origin isn't obvious from its name or its surrounding
  comment, the reader will eventually waste time deriving it
  again. Either rename it (`PRIOR_VAR_UNIFORM_M1_P1`), comment
  it inline (1-2 lines, no math), or — for anything more
  involved — write it up in `docs/design_constants.md` and
  cross-reference. Same principle applies to thresholds,
  bucket sizes, atom counts, normalizers, anything where
  someone could reasonably ask "why exactly that value?".
- **Prefer removing over adding.** This codebase is recovering from
  bloat. When a feature is load-bearing, we'll re-add it with evidence.
