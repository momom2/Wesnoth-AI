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
- **Run:** imitation training via `python tools/supervised_train.py`
  (every reference checkpoint so far comes from it); self-play legs via
  `python tools/az_loop.py`; demos via `python tools/sim_demo_game.py`;
  live-Wesnoth setup checks via `python main.py --check-setup`
- **Test:** `pytest` (tests are Python-only; they run the simulator,
  encoder, model, trainer and tools on synthetic inputs and committed
  data, and none of them launches Wesnoth). CI runs the full suite with
  a freshly built Rust wheel on every push (see Testing).

### Wesnoth data provenance (updated 2026-06-12)

`wesnoth_src/data/` is a WML-only copy (cfg/lua/map, no art) of the
LOCAL STEAM INSTALL's data tree, taken from **1.18.7**. The install
reports 1.18.8 (2026-09-23), and the files the simulator reads from it
(multiplayer factions, scenarios and maps, core macros and units, the
eras) are identical to this copy, compared that day. Refreshed via:

    robocopy "C:\Program Files (x86)\Steam\steamapps\common\wesnoth\data" wesnoth_src\data *.cfg *.lua *.map /S

It serves the RUNTIME readers (scenario_pool factions/eras,
scenario_events, map files). It is NOT a git checkout anymore and
has no `src/` tree; for engine-internals research, read the GitHub
**1.18.4 tag** directly (as docs/wesnoth_rules.md entries do).

**The sim's runtime WML subset IS tracked in git (since 2026-07-02):**
`data/multiplayer/{factions,scenarios,maps}`, `data/core/macros` (the
31 files the scenario expander reads), and the add-ons
`Mini_Maps_Collection`, `Seamless_Map_Picker` and `WL_Mappack` -- 276
files, counted 2026-09-23 -- so a bare `git clone` can run
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

## Current status (2026-10-08)

**Read `BACKLOG.md` "NEXT" first.** The status log of 2026-09-04 to
2026-10-07 is in `docs/archive/claude_status_20260904_20261007.md`,
verbatim; older blocks are in `docs/archive/claude_status_history.md`,
superseded plans and leg records in `docs/archive/` (index in its
README), and the 78 quarantined training mechanisms in
`quarantine/INVENTORY.md`. `docs/plan_20260904.md` keeps the standing
rules; its phase 1 is closed.

### The reference player

`parity3` at `raw:t0+eo-1.5` with its memory at 64 slots (user ruling
2026-10-04): HF `tier-b/imitation_anneal_20261003/lower.pt`, local
`training/checkpoints/parity3.pt`, pinned in
`configs/reference_player.json`. It is imitation-trained
(`tools/sequence_train.py` under the anneal rule,
docs/imitation_anneal_prereg_20261003.md). Its lineage, each step an
800-decisive-game match under the observation of its day (Elo does not
chain across observation epochs):

| reference | adopted | over its predecessor |
|---|---|---|
| `parity3` | 2026-10-04 | +82 +- 13 over `parity2` |
| `parity2` | 2026-10-02 | +191 +- 14 over `obs8`: the network observes what a player sees, and a 64-slot memory carries it from decision to decision |
| `obs8` | 2026-09-25 | +73 +- 13 over `terrain`: the same recipe on the corrected observation |
| `terrain` | 2026-09-20 | +263 +- 16 over `relset`: the hex's terrain set, composed with the end_turn offset -1.5 |
| `relset` | 2026-09-11 | +56 +- 12 over seed2's one-pass checkpoint: the relevant-set basis |
| seed2 | 2026-09-11 | +33 +- 12 over the original imitation seed |

Measured 2026-10-09 (docs/parity3_baselines_prereg_20261008.md): its
self-pin reads -2.6 +- 12.3 Elo, and **the same checkpoint at 0 slots beats
it by +105.5 +- 12.8 Elo** (16 slots play as 64); the memory player acts
less per side-turn and stalls to the turn cap more. Whether the reference
plays at 0 slots: **no (user ruling 2026-10-09): the memory stays, and
that it plays worse is a design flaw to root-cause** (`parity2`'s memory
cost 37 +- 12; tag `archive/exp-memory-in-play` holds that
investigation, ported to `parity3` on `exp/memory-in-play-parity3`).
Budget (user, 2026-10-09): about $100 on Vast for the foreseeable future;
the lead decides runs, a costly one only when confident it pays.

### Self-play

Nothing produced by self-play has beaten any reference. The 2026-09-04
review (docs/raw_argmax_control_20260904.md): the seed at argmax beats
the same weights sampling by +412 +- 97 Elo; Gumbel-MCTS-32 over atomic
actions loses to argmax by -124 +- 58; the search targets were the prior
or a truncation of it, and the value gradient owned 94-99% of every
update through the shared trunk. Policy-gradient self-play was never
measured on a current network (quarantine/INVENTORY.md 1.9).

User rulings: self-play keeps the memory, whose size is open to
discussion (2026-10-05; the memory reaches every place that plays or
trains the network since 0.14.0, docs/memory_everywhere_20261005.md).
No self-play training on the current algorithm, `az_loop`'s search
distillation; before any training launches, a new algorithm must handle
the reward's sparsity, multi-step turns whose plans depend on the dice
rolled inside the turn, and the simulator's speed without being
bottlenecked on the network (2026-10-06). Imitation's remaining gains
are parked (2026-10-04). The program that answers the 2026-10-06 ruling
is docs/selfplay_program_20261008.md (policy iteration with a one-step
look-ahead over exact combat outcomes, its evaluator chosen by
measurement). Step 1 read Kill on 2026-10-09: no learned critic a fair
player can use selects better than material at this scale; the material
look-ahead itself is neutral against `parity3` (-8.7 +- 12.3 Elo).

Measured facts a new design starts from:
- **Turn-level gaps exist.** Under `terrain`, 7 of 60 holdout positions
  have a sampled alternative turn confirmed better by at least 0.25; in
  6 of the 7 the better turn takes more decisions
  (docs/turn_gap_ref_prereg_20260921.md). Two of the seven carry
  in-turn luck in the alternative's favour, and the playouts saw
  through fog.
- **No cheap turn grader passes.** Ranking 5 candidate turns at 199
  human positions against playout truth (corrected within-position
  correlation): `obs8`'s value head 0.22, a fitted linear grader 0.27,
  the HP margin after the turn 0.40, rollouts read four turns ahead 0.53
  (docs/turn_value_prereg_20260925.md).
- **Acting more is the largest lever on record:** the end_turn logit
  offset -1.5 is worth about +229 Elo at decode
  (docs/endturn_rule_prereg_20260919.md).
- **Fog.** Every search and playout in the tree runs on the true state
  (docs/hidden_information_20260926.md); a search that chooses actions
  in play needs a root drawn from a belief model. A training-time critic
  may see both sides.
- **Common random numbers across branches were measured dead**
  (docs/selfplay_redesign_20260904.md, Q8).

Throughput (docs/box_specs.md): 3,200 leaf evaluations per second per
4090 at serve batch cap 64 (`relset`, no memory); the trainer costs 2.2
ms per experience; the Rust core forks a state in 0.019 ms and steps in
0.050 ms; an attack's exact outcomes take about 0.06-0.11 ms. An
800-decisive-game match at the reference decode takes about 17 minutes
on a 4090 box (944 games of the memory model, 2026-10-04, $0.42 an
hour). A single match wall is not resolvable below about 1.8x: identical
repeats have differed by that much.

### The simulator

The Rust core (`rust/wesnoth_core`, adapter `wesnoth_ai/game_core.py`)
is the state of record of the simulator and of replay reconstruction
(0.9.0), and answers every rule asked of a state: the commands, the
scenario's events, combat, the reach, what a side sees, the observation
and the encoding. It was certified against the Python applier over the
whole corpus (14,376 of 14,376 replays, 2026-10-01); that applier and
the Python rules it carried are retired (docs/rust_core_port_20260928.md),
and the core is checked against the engine's records and oracles only.
`OBSERVATION_EPOCH` is 11 and
`CORPUS_VERSION` 5: caches of another epoch refuse to load, and Elo
measured under one epoch does not chain onto another. Real Wesnoth
checks the rules: the scenario-init oracle (28 of 28 scenarios), the
hidden-units oracle (54 of 54 positions), and live games against the
default AI with the simulator mirroring the engine's log
(`tools/live_vs_rca.py`), run on the project's patched Wesnoth 1.18.8
(`tools/wesnoth_build/`), which plays a hosted game's 70% experience.

### Standing rulings

Standing rules (full list in the plan): the reference player is
`parity3` at `raw:t0+eo-1.5` with its memory at 64 slots (user ruling
2026-10-04; one checkpoint, its memory and one decode, all in
`configs/reference_player.json`, which
`tools/reference_player.py --flags b` turns into run_elo_batch flags);
every strength claim is a PURE match against it with the standard
error stated; no teacher is distilled before it wins such a
match; one factor at a time, each with its own number and kill
criterion; proxies are crash barriers, never verdicts; compute on
rented boxes only, proposed with cost first.

Also in force:
- 2026-09-05: no optimizations conditioned on the MCTS loop; scope
  every box test and train sparingly; results are written on the run,
  never as atomic dumps; a box job past about 1.5x its estimate is
  inspected and cut.
- 2026-09-28: no faction is forced on eval games; matches before that
  day had a Knalgan side and do not pool with later ones.
- 2026-10-06: CI is the merge gate (Testing); trivial fixes go straight
  to `main` (Branching); an engine parameter is passed to the engine,
  never emulated.

Eval procedure: the match command of README's Quickstart. The
candidate plays side A, and `$(python tools/reference_player.py --flags
b)` puts the reference on side B with its label, checkpoint and decode
flags; the candidate takes the same decode with
`--raw-end-turn-offset-a -1.5`, and `--mcts-sims 0 --raw-temperature-a 0
--raw-temperature-b 0` makes both sides raw players at argmax. The
procedure tag is `raw:t0`, or `raw:t0+eo-1.5` with the offset; the
legacy sampler (no raw temperature) is `raw` and never mixes with them
in one outdir; searched players carry `mcts:<sims>` (Gumbel root) or
`tcs:<sims>`. Every game draws a map from the 21-map Ladder pool with
fog and both factions uniformly (`FORCED_FACTION` in
`wesnoth_ai/rules/scenario_pool.py` is None since 2026-09-28; described
with the scenario pool under Architecture), which each result records as
`forced_faction`. Run on a
4090 box with `--device cuda --jobs 20 --persistent-workers
--shared-inference` (docs/box_specs.md; every match script since
2026-09-19 runs 20 workers). The last two are `store_true` and default
OFF; without them you get the slowest mode in the repo, and every
current match script passes them. `tools/elo_collect.py <outdir>
--no-catalog` fits the decisive games.

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
- `tools/wesnoth_sim.py` — the simulator: it drives a game on the Rust
  core (`rust/wesnoth_core`, adapter `wesnoth_ai/game_core.py`), which
  applies every command and answers every rule, and `sim.gs` is a view
  of that core (`game_core.view_of`). Replay reconstruction runs on the
  same core (`tools/replay_dataset.record_core`, bit-exact against
  Wesnoth's `[mp_checkup]` oracle on combat); the simulator swaps the
  data source from "WML command stream" to "policy queries". A state
  built by hand or copied gets a core built from it
  (`game_core.core_for`). `tools/kernel_status.py` says whether the
  installed wheel serves the adapter and the source.
- `tools/az_loop.py` — the self-play loop
  (docs/archive/az_minimal_spec.md): the actor pool
  (`tools/actor_pool.py`) plays N games per iteration under MCTS, then
  one gradient step toward the visit counts and the game results.
- `tools/sim_self_play.py` — the game loop every producer runs
  (`_play_one_game_safe`, which the actor pool calls), and the earlier
  self-play entry point, which carries the quarantined mechanisms
  (quarantine/INVENTORY.md). Its CLI trains by search distillation by
  default (`--mcts`, with turn search); `--reinforce` selects REINFORCE
  with a value baseline, the only consumer of the shaping reward
  (`wesnoth_ai/rewards.py`, `configs/reward_selfplay.json`: gold,
  damage and village deltas, per-turn penalty, unit-type and
  turn-conditional bonuses). The actor pool plays with a zero reward.
- `wesnoth_ai/rules/scenario_pool.py` / `wesnoth_ai/rules/scenarios.py` — scenario
  randomization: the Ladder Era 21-map whitelist (fogged or fogless),
  the mini maps, the factions and leaders. `random_setup` puts
  `FORCED_FACTION` on one side of the game when it is set, and both
  sides draw uniformly from the six default-era factions when it is None,
  its value since 2026-09-28 (user ruling). From 2026-07-04 to
  2026-09-27 it was the Knalgan Alliance (user request 2026-04-30), so
  every `run_elo_batch` match of that period, the reference players'
  numbers included, had a Knalgan side (`LEGACY_FORCED_FACTION` for
  records without the field); a match under the uniform draw does not
  chain onto them. `sim_self_play --forced-faction NAME` still forces a
  faction for its in-process games; `az_loop`'s actors always drew both
  factions uniformly.
- `tools/mcts.py` / `tools/mcts_policy.py` — MCTS implementation
  and the MCTSPolicy adapter that wraps TransformerPolicy.
- The actor pool (self-play generation): `tools/az_loop.py` drives
  `tools/actor_pool.py`, whose actor processes (`tools/actor_worker.py`)
  play the games and send their leaves to a serve thread or process
  (`tools/serve_worker.py`) that holds the model;
  `wesnoth_ai/server_priors.py` and `wesnoth_ai/leaf_wire.py` are the
  wire, `tools/actor_stream.py` the continuous form (`--stream`) and
  `tools/bench_pool.py` its benchmark.

**Imitation (every reference checkpoint so far).**
- `tools/supervised_train.py` — behavior cloning of the human corpus
  with a value head trained on game outcomes in the same pass; its
  docstring gives the reference recipe's command line, and
  `configs/imitation.json` holds its loss settings (the winners'
  decisions only, equal weight per game, the manifest's holdout split).
- `tools/preencode_corpus.py` — encodes the corpus once, so a run
  streams tensors (`supervised_train --preencoded`).
- `scripts/unit_vocab_retrain_box.sh` — the whole recipe on a box: the
  corpus from Hugging Face, the vocabulary (`tools/unit_vocab.py`), the
  pre-encoding, one pass, the per-phase value table and the 800-game
  match against the reference.

**Evaluation (strength claims).**
- `tools/run_elo_batch.py` — the match driver: resumable, memory-guarded,
  sides alternated, seeds derived from the game index.
- `tools/elo_eval_game.py` — one game, or many as a persistent worker
  (`tools/eval_workers.py`); under `--shared-inference`,
  `tools/eval_inference_server.py` serves the forwards of every worker.
- `tools/raw_player.py` — the raw player's decode (temperature, end_turn
  offset); `tools/reference_player.py` reads
  `configs/reference_player.json`.
- `tools/elo_collect.py` — fits Elo on the decisive games (capped games
  are absences) and refuses to mix estimands within an outdir.
- `tools/game_record.py` — each game is recorded whole beside its
  result (`<game>.game.jsonl.gz`).

**The Rust core.**
- `rust/wesnoth_core/` — the core, installed with `pip install
  ./rust/wesnoth_core`. `lib.rs` declares `__phase__`; the adapter
  (`wesnoth_ai/game_core.py`, `_CORE_PHASE`) refuses an older wheel.
  Python reads the core through `game_core` (`CoreState`),
  `wesnoth_ai/observe.py` (the observation), `wesnoth_ai/encoder.py`
  (`encode_raw`), `tools/pathfind_sim.py` (the planner's context and
  reach), `wesnoth_ai/visibility.py` (what a side sees) and
  `tools/combat_outcomes.py` (fight outcomes, the counter weapon);
  docs/rust_port_plan.md and docs/rust_core_port_20260928.md record the
  port.

**Live-Wesnoth path (evaluation and rule checks only).**
- `main.py` — setup / maintenance CLI (`--check-setup`,
  `--clean-games`).
- `wesnoth_ai/wesnoth_interface.py` — one Wesnoth process per game;
  state channel uses `std_print` → log-file tail; actions written
  atomically as `action.lua` and read via `wesnoth.read_file`.
- `tools/live_vs_rca.py` + `tools/live_mirror.py` — the reference
  against Wesnoth's default AI in a live game, the simulator mirroring it
  from the engine's log and checking the board at every decision; Lua
  side `lua/live_stage.lua` and `lua/board_report.lua`. It runs on the
  project's patched Wesnoth (`tools/wesnoth_build/README.md`).
- `tools/hidden_units_oracle.py` (Lua: `lua/turn_stage.lua`, a custom
  AI stage that avoids the default AI's blacklist-on-failure rule, with
  `lua/state_collector.lua` and `lua/action_executor.lua`, which runs
  moves) and `tools/scenario_init_oracle.py` (`lua/init_oracle.lua`):
  the simulator's rules checked against the engine.
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

### Coordinates

- **Wesnoth uses 1-indexed hex coordinates** (WML, replays, Lua).
- **Python uses 0-indexed hex coordinates everywhere internally.**
- The ±1 conversion happens where Wesnoth data enters or leaves Python:
  the live tools (`tools/live_mirror.py`, `tools/live_vs_rca.py`),
  the WML and replay readers (`tools/replay_extract.py`,
  `wesnoth_ai/rules/scenario_pool.py`, `tools/scenario_events.py`,
  `wesnoth_ai/rules/wml_state.py`) and the writers that emit WML or feed the
  engine (`tools/sim_to_replay.py`, `tools/dump_savestate.py`,
  `tools/hidden_units_oracle.py`, `tools/scenario_init_oracle.py`).
  Keep it at those boundaries; game logic, the encoder and the model
  work in 0-indexed coordinates only.

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
matched; a 2026-05 figure with no record kept, and
tests/test_combat_seed_alignment.py re-checks every strike of a
29-attack strict-sync fixture on each run). Any sim change that
touches combat, healing, or advancement must keep that parity.
`tools/diff_replay.py` is the regression check (runs the simulator
over a corpus, compares against the recorded WML command stream). New
scenario events go in the core's event dispatch
(`rust/wesnoth_core/src/events.rs`; `tools/scenario_events.py` parses
them), new abilities and [effect] forms in the core (`core_attack.rs`,
`core_step.rs`, `effects.rs`); both with citations: `src/<path>:<line>`
at the 1.18.4 tag for C++, `wesnoth_src/data/<path>:<line>` for WML and
Lua.

Any mismatch between the simulator and Wesnoth (usually surfaced
by OOS errors when strict syncing sim-produced replays) is a
critical issue to be investigated and solved at the root.

When live Wesnoth is in the loop (display, eval), the same
narrow-waist principle applies: state crosses the bridge as one
well-defined serialization, actions as one schema, Lua side stays
dumb. But that path is no longer how training data is generated.

### 5. Failures are visible
Both paths log timeouts and stage-of-failure. The simulator does not
apply a refused action (a recruit onto a hex that turns out to be
occupied under fog, a move outside the landable set, an attack with no
hex to strike from): `WesnothSim.step` sets `last_step_rejected` and the
caller decides again, a recruit bounce being recorded on
`gs.global_info._recruit_rejected_hexes` (principle 6). An action it
cannot translate ends the side's turn, and so do eight rejections in a
row from a caller that ignores the mask (with a warning); the recorded
end_turn then carries what was attempted. `tools/diff_replay.py`, which
checks each command of a real replay before applying it, reports typed
divergences (e.g. `"recruit:insufficient_gold"`). The bridge
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
  - A recruit ordered onto a hex a hidden unit holds goes where the
    engine puts it: the vacant castle hex nearest the leader, gold
    spent (docs/wesnoth_rules.md "A recruit onto an occupied hex");
    only with no vacant castle hex is it refused. Either way the
    ordered hex is rejected for the turn: it becomes illegal AND a
    per-hex "recruit_rejected" bit appears in the encoder feature --
    the mask consults the rejection set, the model sees the bit; both
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

### Branching (adopted 2026-09-23)
GitHub flow: `main` holds only finished, checked work. Everything else
happens on a short-lived topic branch cut from `main`, merged back when
it is ready, then deleted.
- **Names:** `feature/<name>`, `fix/<name>`, `test/<name>`,
  `exp/<name>`; short, descriptive, hyphenated
  (`fix/heals-value-default`). One task, one branch.
- **Ready to merge** = `ruff check .` clean and the branch's latest CI
  run green, both tiers (`gh run list --branch <branch>`; see Testing).
  CI is the gate; the laptop's full fast tier is not required.
- **Merge with a merge commit** (`git merge --no-ff`), never squash:
  the commit messages are this project's lab notebook.
- **`exp/` branches differ.** An experiment's code merges only if it
  wins, but its pre-registration and its result land on `main` either
  way (a records-only merge or a direct commit): "rejected: X, because
  Y" is what stops X being proposed again.
- **Status entries** in this file and in BACKLOG.md are written on
  `main` when the work lands, never on the branch. Nearly every change
  touches them, so branch-side edits would conflict with each other.
- **Parallel agent sessions** each use their own branch in their own
  worktree, so no session edits another's files.
- **Pushing** a topic branch is free (user ruling 2026-09-23).
- **Trivial fixes go straight to `main`** (user ruling 2026-10-06):
  corrections to records, docs and status entries, typos, and small
  mechanical fixes that change no behaviour are committed to `main` and
  pushed without asking, with their version bump. Anything else goes
  through a topic branch; merging it, and pushing `main` for it, asks.
- **Deleting a branch** that holds commits `main` does not have: tag
  its tip `archive/<name>` and push the tag first, so the work stays
  reachable.
- Rejected: Git Flow's `develop` / `release/` branches. They serve
  software shipped in numbered releases, and GitHub's own flow has
  neither (docs.github.com, "GitHub flow").

### Versioning (adopted 2026-09-23, revised the same day)
- `wesnoth_ai.__version__` in `wesnoth_ai/__init__.py` is the only
  place the version lives, and it numbers the states of `main`:
  **every change that lands on `main` bumps it once** -- M for a
  feature, P for anything else (fix, test, records) -- and N changes
  only on the user's explicit decision (0 until then).
- Bump it in the commit that lands the change on `main`: for a merge,
  `git merge --no-ff --no-commit <branch>`, bump, then commit. Topic
  branches never touch it, so two branches never claim one number.
  State the new version on the first line of that commit's body.
- History starts at 0.1.0 (the time-of-day encoder commit); nothing
  before it carries a number. The first day bumped per commit; merges
  to `main` are the unit from 0.2.2 on.

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
- Python systems talk through explicit APIs (on the live bridge,
  `WesnothGame`), not by reaching into private attributes.
- Lua code never decides game logic; it just serializes state and
  executes actions the Python side chose.

## Testing

### Philosophy
Tests exist to catch regressions you'd otherwise only notice after
burning an overnight training run. Prefer few behavioral tests over
many line-coverage tests.

### What tests we have (and what they are NOT)
- `test_integration.py` and `test_lua_actions.py` cover the live
  bridge with **synthetic inputs**: JSON payloads shaped like the Lua
  collector's, and the Lua action files Python writes. Despite the
  name, `test_integration.py` does NOT launch Wesnoth, and no test
  does. The engine oracles (`tools/hidden_units_oracle.py`,
  `tools/scenario_init_oracle.py`) launch real Wesnoth and run only by
  hand, with the user's agreement.

### Guidelines
- **Before a push:** `ruff check .` and the test files that exercise
  the change (`pytest tests/test_x.py ...`). The whole fast tier on the
  laptop (a bare `pytest`: tests marked `slow`, full-game / subprocess /
  threading e2e, are excluded by default, see pytest.ini) is optional.
- **CI runs the FULL suite — `pytest -m ""` — on every push, and is the
  merge gate** (`.github/workflows/tests.yml`, GitHub Actions). The slow
  tier holds the e2e regression guards (MCTS self-play smoke,
  concurrent train-step races, export validation); the fast tier alone
  does NOT cover them, and the laptop does not run them (they generate
  self-play games). A branch merges on a green run of its tip, and a
  training campaign launches from a commit that has one. CI has no GPU
  and no corpora: the CUDA tests and the tests that read replay,
  imitation or value data skip there.
- **"Slow" is measured.** Every CI run lists its 30 slowest tests
  (`--durations=30`); a fast-tier test over 10 s moves to the slow tier
  or is slimmed.
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
- **Every run lists its failed, errored and skipped tests** (pytest.ini
  `-rfEs`). A skip count that differs between two runs of one tree is a
  test that sometimes does not run; diff the two SKIPPED lists to name
  it.
- **A test that builds a search policy seeds it**: `MCTSPolicy(...,
  rng_seed=N)` / `TurnCommitPolicy(..., rng_seed=N)`, or `rng=` on a
  direct `mcts_search`. Unseeded, the search draws fresh OS entropy,
  so any assertion or skip that depends on what it chose varies from
  run to run -- the fast tier's 41/42 skip flicker was exactly that
  (2026-09-23).
- Never weaken a test without explicit user confirmation. A failing
  test is a signal — find the root cause first.
- **The Rust paths are tested on CI, not on the laptop.** CI builds
  the wheel from each pushed commit and asserts its `__phase__` equals
  the one `rust/wesnoth_core/src/lib.rs` declares, so every Rust test
  runs there; a green CI run certifies a Rust change, and a box is
  needed only for what CI cannot run (CUDA, throughput). The laptop's
  installed wheel is whatever was last installed and lags the source
  until rebuilt (phase 1 against the source's 31 on 2026-10-08). Every
  rule runs in the core, so a local run tests the installed wheel's
  Rust: one older than the adapter's phase stops the simulator, and the
  tests that need it skip or fail (the suite prints a banner saying so).
  Check `python tools/kernel_status.py` before believing any local
  result.
  The laptop builds the wheel since 2026-09-28, into the cargo target
  directory Defender excludes (any other directory is refused, "Accès
  refusé", os error 5): from `rust/wesnoth_core`,
  `CARGO_TARGET_DIR=C:/Users/amaur/.cargo-target maturin build --release
  -i python --out C:/Users/amaur/.cargo-target/wheels`, then
  `pip install --force-reinstall --no-deps` the wheel it names. Never
  install a wheel while a pytest run is going.

## Working Style

- **High autonomy** on reversible local work (edits, tests, reads),
  including creating, switching, committing on and pushing topic
  branches (see Branching).
- **Ask before**: merging a topic branch into `main` or pushing `main`
  for anything beyond a trivial fix (Branching), force-pushing, deleting
  a branch whose work is not merged,
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
  attrs. **`wesnoth_src/` has NO `src/` tree** -- it is a WML-only
  copy of the local 1.18.7 install (see the provenance note above), so
  `grep wesnoth_src/src/` returns nothing, and the rules catalog's
  citations of C++ files cannot be checked locally. For C++ engine
  internals read the **1.18.4 tag on GitHub raw**, e.g.
  `https://raw.githubusercontent.com/wesnoth/wesnoth/1.18.4/src/units/unit.cpp`,
  and cite `src/...:line` as the rules catalog does. For WML and Lua,
  `wesnoth_src/data/` IS local and authoritative -- grep it first when
  you would otherwise hand-wave ("`income=` is probably an offset").
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
  Common gotcha: a rule we're hunting in the engine's C++ (`src/` at
  the 1.18.4 tag) is actually in
  `wesnoth_src/data/multiplayer/eras.lua` or a WML macro under
  `wesnoth_src/data/core/macros/`. After reading the C++, always also
  grep `wesnoth_src/data/multiplayer/`, `wesnoth_src/data/core/macros/`,
  `wesnoth_src/data/lua/`. Rules with a "post-pass" feel (applied after
  unit setup) often hide in `[event]name=prestart` Lua callbacks.
- **Wesnoth's `changelog.md` is HISTORICAL — verify against current
  source.** It lives in the engine repository on GitHub (`wesnoth_src/`
  has no copy). Old changelog entries describe behavior at THAT
  version, which may have changed since. Cross-check any changelog
  quote against the code at the 1.18.4 tag (C++) or in
  `wesnoth_src/data/` (WML, Lua) before treating it as authority.
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
- **Credentials never reach a transcript** (user decision 2026-09-29).
  What a tool returns to Claude is stored and sent to the model, so
  `tools/secret_guard` redacts credentials from every tool result: Bash
  output through the shell prefix, failed commands included, and the other
  tools' results through a hook; a `<redacted>` in output is the guard at
  work. A new key is created in the user's own terminal or browser and
  saved to its store before any tool call touches it. Keys are restricted
  (Vast: offers and instances; Hugging Face: write on the checkpoint
  repository only), and every checkpoint load uses `weights_only=True`,
  since a leaked Hugging Face token could otherwise plant code in a
  checkpoint we download. Rejected (user, 2026-09-29): pinning by hash
  the code and scripts boxes download from Hugging Face, against a
  dishonest Vast host reusing the box's token; not worth the hardening.
