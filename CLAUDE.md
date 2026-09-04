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
- **Run:** training via `python tools/sim_self_play.py`; demos via
  `python tools/sim_demo_game.py`; live-Wesnoth setup checks via
  `python main.py --check-setup`
- **Test:** `pytest` (tests are Python-only; they exercise the WML
  parser and Lua-file generation with synthetic data — they do NOT spin
  up real Wesnoth)

### Wesnoth data provenance (updated 2026-06-12)

`wesnoth_src/data/` is a WML-only copy (cfg/lua/map, no art) of the
LOCAL STEAM INSTALL's data tree — currently **1.18.7** — refreshed
via:

    robocopy "C:\Program Files (x86)\Steam\steamapps\common\wesnoth\data" wesnoth_src\data *.cfg *.lua *.map /S

It serves the RUNTIME readers (scenario_pool factions/eras,
scenario_events, map files). It is NOT a git checkout anymore and
has no `src/` tree; for engine-internals research, read the GitHub
**1.18.4 tag** directly (as docs/wesnoth_rules.md entries do).

**The sim's runtime WML subset IS tracked in git (since 2026-07-02):**
`data/multiplayer/{factions,scenarios,maps}` + `data/add-ons/
Mini_Maps_Collection` (~840KB) — so a bare `git clone` can run
self-play training on a GPU node with no Wesnoth install. The rest
of `wesnoth_src/` stays untracked. A robocopy refresh from Steam
will show these as git diffs; that's intentional (data drift
becomes visible in review instead of silent).

**`unit_stats.json` / `terrain_db.json` are pinned 1.18.4 scrapes
and are COMMITTED. Never re-scrape them from this wesnoth_src.**
Unit stats DRIFT between releases — e.g., in 1.19.x the Ghoul
gained a `[resistance] pierce=90` override that doesn't exist in
1.18.4 (where pierce inherits 70 from the gruefoot movement_type);
using drifted stats made combat reconstruction overdamage units.
The sim's bit-exact combat parity and the existing checkpoints
assume the 1.18.4 stats. Re-scraping requires fetching the 1.18.4
tag from GitHub first.

Most replays in `replays_raw/` are from 1.18.x clients; pin
accordingly. If a replay's `[scenario] version=` says something
other than 1.18.x, scrape from that version's tag instead.

## Current status (2026-09-04): new phase, engineering first

**Read `docs/plan_20260904.md` first; `BACKLOG.md` holds the next
actions in order.** Superseded status blocks, plans, leg records and
mechanism specs are in `docs/archive/` (index in its README); the 78
quarantined training mechanisms are in `quarantine/INVENTORY.md`.

State of play:
- The strongest player measured is the imitation seed played at
  temperature 0 (`raw:t0`; HF `tier-b/a3/seed_imit_tierb_start.pt`,
  local `training/checkpoints/seed_imit_tierb_start.pt`). Nothing
  produced by self-play has beaten it.
- Why (2026-09-04 review, docs/raw_argmax_control_20260904.md): the
  seed at argmax beats the seed sampling 37-3 (+412 ± 97), and the
  seed with Gumbel-MCTS-32 loses to the seed at argmax 13-27
  (−124 ± 58). Every earlier "search improves the seed" number used
  the sampling player as its reference. A 32-evaluation search over
  ~350 atomic actions looks 1-2 actions ahead, its targets are the
  prior or a truncation of it, and the value gradient owned 94-99% of
  every update through the shared trunk. No self-play leg ever had a
  teacher better than the prior.
- The 20-agent design panel (docs/selfplay_redesign_20260904.md)
  found no affordable outcome-graded teacher at this budget; the plan
  therefore starts with engineering (10-30x cheaper games, certified
  bit-exact) and then turn-level search built directly against
  `raw:t0` with 800-game gates.
- 2026-09-04/05 engineering day (details in BACKLOG.md and
  docs/box_specs.md): pool generation 141 -> ~330 leaves/s (server
  priors, bf16, packed requests, thread caps; the actor loop's own
  ~50 ms per leaf is the ceiling now, profile in progress); eval
  4.2x through persistent workers (800 games in ~65 min); Rust
  encode_raw; 16 review findings fixed. Measured: `raw:t0` against
  itself stalls (17 of 40 games at the 200-turn cap) while
  `raw:t0.5` scores 22-18 against it with no stalls; at argmax the
  leg-4 self-play product equals the seed (+17 +- 55) and the 5M
  2291k equals the 15M seed (+9 +- 55). The phase-2 prerequisite
  (turn-level value gap, docs/turn_gap_prereg_20260904.md) is
  pre-registered and running.
- Box 49875606 was running the queued measurements at the end of
  that window; destroy it if it is still up. Vast credit about $57.

Standing rules (full list in the plan): the reference player is
`raw:t0`; every strength claim is a PURE match against it with the
standard error stated; no teacher is distilled before it wins such a
match; one factor at a time, each with its own number and kill
criterion; proxies are crash barriers, never verdicts; compute on
rented boxes only, proposed with cost first.

Eval procedure: `tools/run_elo_batch.py ... --mcts-sims 0
--raw-temperature-a 0 --raw-temperature-b 0` for raw players (the
procedure tag is `raw:t0`; the legacy sampler is `raw` and never mixes
in one outdir); searched players carry `mcts:<sims>` (Gumbel root) or
`tcs:<sims>`. Run on a 4090 box with `--device cuda --jobs 10`
(docs/box_specs.md): 40 raw games in 2 minutes, 40 searched games in
14 minutes.

## Architecture

### Two paths that share encoder + model + trainer

**Source layout (2026-07-23 reorg).** The core library modules listed
below now live in the **`wesnoth_ai/`** package (imported as
`from wesnoth_ai.X import ...`), not at the repo root. Scripts keep
their `tools/` prefix; the test suite moved to **`tests/`**
(`tests/conftest.py` bootstraps `sys.path`); `main.py` (the setup CLI)
stays at the root. So a bare name like `classes.py` below means
`wesnoth_ai/classes.py`.

**Production path: in-process simulator.**
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

**Live-Wesnoth path (eval only).**
- `main.py` — setup / maintenance CLI (`--check-setup`,
  `--clean-games`). No longer drives training or `--display`.
- `wesnoth_ai/wesnoth_interface.py` — one Wesnoth process per eval game; state
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

**Shared by both paths (all under `wesnoth_ai/`):**
- `wesnoth_ai/classes.py` — `GameState`, `Unit`, `Hex`, `Map`,
  `SideInfo`, `state_key`.
- `wesnoth_ai/encoder.py` — `GameState` → tensors (per-unit, per-hex,
  recruit phantom features, global features).
- `wesnoth_ai/model.py` — `WesnothModel` transformer with
  distributional C51 value head; emits `ModelOutput(actor_logits,
  type_logits, target_logits, weapon_logits, value, value_logits,
  cliffness, ...)`.
- `wesnoth_ai/action_sampler.py` — legal-action enumeration with
  priors; combat-oracle attack-bias on target logits.
- `wesnoth_ai/transformer_policy.py` — Policy adapter (`select_action`,
  `observe`, `train_step`, `save_checkpoint`, `finalize_game`).
- `wesnoth_ai/trainer.py` — REINFORCE + value baseline (`step`) and
  AlphaZero-style soft-target distillation (`step_mcts`); both
  use the categorical CE value loss against C51 atom projections.
- `wesnoth_ai/rewards.py` / `configs/reward_selfplay.json` — shaping
  reward (terminal ±1, gold/damage/village deltas, per-turn penalty,
  unit-type bonuses, turn-conditional bonuses).

### Coordinates

- **Wesnoth uses 1-indexed hex coordinates.**
- **Python uses 0-indexed hex coordinates everywhere internally.**
- Conversion happens in `wesnoth_ai/state_converter.py` (both
  directions). Keep it there; do not sprinkle `±1` around the codebase.

## Architecture Principles

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

### 4. The simulator must be perfectly faithful to Wesnoth
The simulator's combat math is verified against Wesnoth's own
`[mp_checkup]` oracle on strict-sync replays (731/731 strikes
matched). Any sim change that touches combat, healing, or
advancement must keep that parity. `tools/diff_replay.py` is the
regression check (runs the simulator over a corpus, compares
against the recorded WML command stream). New scenario events go in
`tools/scenario_events.py`; new abilities in `tools/abilities.py`;
both with citations to `wesnoth_src/` file:line.

Any mismatch between the simulator and Wesnoth (usually surfaced
by OOS errors when strict syncing sim-produced replays) is a
critical issue to be investigated and solved at the root.

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

### Linting (adopted 2026-08-05)
- **Run `ruff check .` before committing** — config in `ruff.toml`
  (rules E/F/W; E501/E731 off; E402 allowed in tools//tests//scripts
  for the sys.path bootstrap pattern). The check must come back clean;
  `ruff check --fix` applies the safe autofixes.
- Intentional exceptions get a targeted `# noqa: <rule>` with a short
  reason, never a blanket file ignore.

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
- Run `pytest` after changes. This runs the FAST tier (~2.5 min):
  tests marked `slow` (full-game / subprocess / threading e2e,
  >10s each — see pytest.ini) are excluded by default.
- **Run the FULL suite — `pytest -m ""` (~11 min) — before
  committing sim/trainer/mask changes and before launching any
  training campaign.** The slow tier holds the e2e regression
  guards (MCTS self-play smoke, concurrent train-step races,
  export validation); the fast tier alone does NOT cover them.
- **Never run more than one pytest invocation at a time.** Each
  pytest spawns a Python process that imports torch + the model;
  parallel runs balloon memory (5+ GB per process) and a stuck
  test compounds the problem. Wait for each command to fully
  return (foreground) or be killed before launching the next. 
  If a test hangs, kill the process before starting another — 
  don't queue a second behind it. No "let me also kick off X 
  while we wait" patterns. (Lesson from 2026-05-10: four parallel 
  pytest jobs stuck on a hanging `_select_one` loop produced 
  multiple multi-GB zombie Python processes that locked up the 
  user's machine.)
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
