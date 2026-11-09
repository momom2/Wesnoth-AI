# CLAUDE.md — Wesnoth AI

This file holds instructions for working in this repository, and only
instructions. The project's state is in `docs/status.md`: read it first.
Results, measurements and status logs never go here; they go in
`docs/status.md`, the run's own record, or `docs/archive/`. The CLAUDE.md
that carried status until 2026-11-09 is kept verbatim in
`docs/archive/claude_md_20261109.md`.

## Project

A reinforcement-learning AI for *Battle for Wesnoth*, trained by
self-play. It aims to be (a) strong against players and (b) readable
enough that humans can study its strategies.
- **Strength comes first** (user ruling 2026-10-19). Readability and
  steering come later, once a superhuman player exists.
- **Behaviour is learned** (user ruling 2026-10-21): nothing hand-set
  beside the network decides play or shapes its training signal
  (Principle 3).

- **Language:** Python 3.11+, with a Rust core for the simulator's rules.
- **ML:** PyTorch.
- **Game:** Wesnoth 1.18.x (the Steam install on Windows).
- **Machines:**
  - **The desktop:** Windows, Ryzen 5 7600, 63 GB of RAM, a Radeon RX 6600.
    It is the main machine, and its GPU is free to use (see "The desktop's
    GPU").
  - **Rented 4090 boxes on Vast:** about 5x the desktop's generation rate
    and 10x its training rate, at a price (see "Compute").
  - **GitHub Actions** runs CI.

### Where things are

| What | Where |
|---|---|
| The current state, the reference player and the open questions | `docs/status.md` |
| Open items and older rulings (its NEXT section dates from 2026-10-09) | `BACKLOG.md` |
| How to set up a machine: credentials, the Rust wheel, the ROCm venv | `docs/new_machine.md` |
| Commands: install, the match, the local match | `README.md` |
| The training design the self-play rounds follow | `docs/training_design_20261019/current_design.md` |
| Wesnoth rules, with source quotes | `docs/wesnoth_rules.md` |
| Derived constants | `docs/design_constants.md` |
| Superseded plans, logs and runbooks | `docs/archive/` (index in its README) |
| The 78 quarantined training mechanisms | `quarantine/INVENTORY.md` |
| The standing rules of the 2026-09-04 plan | `docs/plan_20260904.md` |

## Principles

### 1. Results are measured, pre-registered, and recorded either way
- Every run gets a pre-registration (`docs/<topic>_prereg_<date>.md`)
  before it launches. It states what is measured, the number that decides,
  and the kill criteria.
- A change to a registered run is a numbered departure, written before it
  takes effect.
- The result lands in the same document whatever it says: "rejected: X,
  because Y" is what stops X being proposed again.
- Change one factor at a time, each with its own number and kill
  criterion.
- Proxies (losses, KL, held-out readings) are crash barriers, never
  verdicts. Strength is read from games.

### 2. Self-play is non-negotiable
The learning signal comes from self-play. Imitation of human games may
warm-start a network, but the end state is self-play. Do not design
architectures that preclude it.

### 3. Behaviour is learned
- The network decides play and learns it from training (user rulings
  2026-10-19 and 2026-10-21).
- So there is no decode-time rule, table or fitted head beside the network,
  no hand-placed prior bias, and no shaped reward or material score. The
  training signal is the game's outcome. "The model should learn to
  recruit like a human iff recruiting like a human works."
- The player being trained samples from its policy. The reference keeps
  its pinned decode as the yardstick.
- Settings that remain, such as a curriculum or data selection, live in
  config, not in scattered constants.

### 4. The simulator is perfectly faithful to Wesnoth
- The Rust core (`rust/wesnoth_core`) is the state of record for play and
  for replay reconstruction.
- Its combat is bit-exact against Wesnoth's `[mp_checkup]` oracle.
  `tests/test_combat_seed_alignment.py` re-checks every strike of a
  strict-sync fixture on each run, and `tools/diff_replay.py` runs the
  simulator over a corpus against the recorded command stream.
- Any change touching combat, healing or advancement keeps that parity.
- New scenario events go in `rust/wesnoth_core/src/events.rs`, and new
  abilities and `[effect]` forms in `core_attack.rs`, `core_step.rs` and
  `effects.rs`, each with a citation:
  - `src/<path>:<line>` at the 1.18.4 tag for C++;
  - `wesnoth_src/data/<path>:<line>` for WML and Lua.
- Any mismatch with Wesnoth (usually an OOS error when strict-syncing a
  simulator-produced replay) is critical: find the root cause and fix it.

### 5. Failures are visible
- The simulator does not apply a refused action: `WesnothSim.step` sets
  `last_step_rejected`, and the caller decides again.
- An action it cannot translate ends the side's turn, and so do eight
  rejections in a row, with a warning.
- `tools/diff_replay.py` reports typed divergences
  (`"recruit:insufficient_gold"`).
- On the live bridge, every Python wait has a finite timeout and logs which
  stage timed out. Lua errors reach the Python log; they never die silently
  inside a `pcall`.

### 6. The legality mask is a pure function of observable state
The mask answers one question: what can the policy validly attempt now,
given the information it has? Observable state is:
1. the visible game state, which is what the encoder sees: own-side fog
   hexes, but no enemy unit hidden in fog and nothing else from the
   simulator's god view;
2. the per-turn rejection history: the hexes where a recruit bounced this
   turn, on `gs.global_info._recruit_rejected_hexes`, cleared at
   `init_side`.

So:
- Fog castle hexes are legal recruit targets.
- A recruit ordered onto a hex a hidden unit holds goes where the engine
  puts it: the vacant castle hex nearest the leader, gold spent
  (`docs/wesnoth_rules.md`, "A recruit onto an occupied hex"). With no
  vacant castle hex it is refused.
- Either way, the ordered hex becomes illegal for the turn, and the
  encoder shows a `recruit_rejected` bit on it. The mask and the model read
  the same state.

Forbidden:
- god-view masking;
- rejection history that persists across turns;
- mutable state outside `gs.global_info`, such as cached per-call
  counters.

The contract covers every action type. A future "tried and bounced" record
lives on `gs.global_info`, clears at the right turn boundary, and is
mirrored in the encoder's features.

## Rulings in force

The user's, with dates. Ask before acting against one.

- **The reference player** is whatever `configs/reference_player.json`
  pins: one checkpoint, its memory size and one decode.
  - Changing it is the user's decision.
  - A candidate needs a PURE match against it first (see "Strength
    claims").
  - No teacher is distilled before it wins such a match.
- **The memory stays** (2026-10-05, 2026-10-09). Self-play keeps the
  network's memory, whose size is open to discussion. That a memory player
  plays worse is a design flaw to root-cause, not a reason to drop it.
- **Search distillation is out** (2026-10-06). There is no self-play
  training on `tools/az_loop.py`'s search distillation. A new algorithm
  must handle three things before it trains:
  - the reward's sparsity;
  - multi-step turns whose plans depend on dice rolled inside the turn;
  - the simulator's speed, without bottlenecking on the network.

  The policy-iteration rounds (`docs/status.md`) are the current answer.
- **No optimization conditioned on the MCTS loop** (2026-09-05).
- **No faction is forced on evaluation games** (2026-09-28).
  `FORCED_FACTION` in `wesnoth_ai/rules/scenario_pool.py` is None. Matches
  from before that day had a Knalgan side and do not pool with later ones.
- **Engine parameters are passed to the engine, never emulated**
  (2026-10-06).
- **Imitation's remaining gains are parked** (2026-10-04).
- **Runs:**
  - Results are written as the run goes, never as one dump at the end.
  - A run past about 1.5x its estimate is inspected, and cut if it is
    stuck.
  - Rented compute is proposed to the user with its cost first. The
    budget is about $100 on Vast (2026-10-09).

## Strength claims

- **Every strength claim is a PURE match against the reference,** with its
  standard error. It is fitted on decisive games only, since capped games
  are absences.
- **The candidate plays side A** at the reference's decode. The
  reference plays side B.
- **800 decisive games is the minimum** for an adoption.
- **Elo does not chain** across observation epochs (`OBSERVATION_EPOCH` in
  `wesnoth_ai/constants.py`) or across the 2026-09-28 faction change.
- **The procedure tag** names the decode:
  - `raw:t0` is argmax;
  - `raw:t0+eo-1.5` adds the −1.5 end_turn offset, the reference's decode;
  - `raw:t1` samples at temperature 1;
  - `raw` is the legacy sampler;
  - `mcts:<sims>` and `tcs:<sims>` are searched players.

  Tags never mix in one outdir; `tools/elo_collect.py` refuses.
- **Games:** every game draws a map from the 21-map Ladder pool, with fog
  and both factions drawn uniformly.
- **On the desktop:** `scripts/match_local.py` is the whole procedure in
  one command (README, "On the local Windows machine"). It serves both
  sides in fp16 and is a valid claim.
- **On a box:** use README's Quickstart command with `$(python
  tools/reference_player.py --flags b)`, `--raw-end-turn-offset-a -1.5`,
  `--mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0`, and
  `--device cuda --jobs 20 --persistent-workers --shared-inference`. The
  last two flags default off, and without them you get the slowest mode in
  the repo.
- **Pooling:** record the serving (fp16 here, bf16 on a box), and don't
  pool the two in one fit.
- **Fitting:** `python tools/elo_collect.py <outdir> --no-catalog`.

## The desktop's GPU (RX 6600)

- **It is free to use at any time** (user ruling 2026-10-29). When the user
  says they want it, stop or yield the run: a STOP file between steps, or
  `--yield-gpu` (`scripts/gpu_yield.py`).
- **One heavy GPU job at a time.** A benchmark beside a run syncs every
  batch.
- **The card drives the display,** so a GPU stall past 2 s resets the
  display driver (System event 4101). The runners stop on one. No single
  kernel or call may approach 2 s.
- **The ROCm build of torch** lives in its own venv,
  `%USERPROFILE%\venvs\wesnoth_rocm`. Set `HIP_VISIBLE_DEVICES=1`, since
  device 0 is the CPU's integrated GPU. The repository's `.venv` is CPU
  torch, and its CPU kernels are much faster than the ROCm build's.
- **Run the pg tools' fp16 GPU paths with `AMD_SERIALIZE_KERNEL=3`.**
  Unserialized, the evaluator faulted the card, and two faults in a row
  wedged HIP until a reboot (`training/metrics/pg_fp16_20261030/README.md`).
- **Serve matches in fp16** (`--infer-fp16` with `--shared-inference`):
  bf16 matmuls run at a fifth of fp16's rate on this card.
- **CPU diagnostics beside a run** go at Idle priority on at most 4
  threads. At below-normal, one cost a round about an hour.
- **A run must resume from its own markers.** A Windows Update restart ends
  everything, and after a reboot the machine waits at the sign-in screen.
  `--keep-awake` holds it awake while a run lives.

## Code map

- **`wesnoth_ai/`, the library:**
  - `classes.py`: `GameState`, `Unit`, `Hex`, `Map`, `SideInfo`.
  - `game_core.py`: the adapter to the Rust core (`CoreState`, `view_of`,
    `core_for`).
  - `observe.py` and `visibility.py`: what a side sees.
  - `encoder.py`: state to tensors (`encode_raw`).
  - `model.py`: `WesnothModel`, a transformer with a C51 value head and a
    memory.
  - `action_sampler.py`: legal actions with their priors.
  - `transformer_policy.py`: the policy adapter.
  - `trainer.py`: REINFORCE and AlphaZero-style steps.
  - `hip_kernels.py`, `hip_attention_grad.py` and `hip_elementwise.py`: the
    RX 6600's kernels.
  - `pg_*.py`: the self-play rounds' step, loss, protocol, records and
    cross-play.
  - `rules/`: the scenario pool, the scenarios and WML state.
  - `constants.py`.
- **`rust/wesnoth_core/`:** every rule (commands, events, combat, reach,
  vision, observation, encoding), installed as a wheel. `lib.rs` declares
  `__phase__`, and the adapter's `_CORE_PHASE` refuses an older wheel.
- **`tools/`, the programs:**
  - the simulator: `wesnoth_sim.py`, `sim_demo_game.py`, `sim_to_replay.py`;
  - replays: `replay_extract.py`, `replay_dataset.py`, `diff_replay.py`;
  - imitation: `supervised_train.py`, `sequence_train.py`,
    `preencode_corpus.py`;
  - matches: `run_elo_batch.py`, `elo_eval_game.py`,
    `eval_inference_server.py`, `raw_player.py`, `reference_player.py`,
    `elo_collect.py`, `game_record.py`;
  - the rounds' steps: `pg_corrections.py`, `pg_pass.py`, `pg_round.py`,
    `pg_eval.py`, `pg_stage_c.py`, `pg_plant*.py`;
  - live Wesnoth: `live_vs_rca.py`, `live_mirror.py`,
    `hidden_units_oracle.py`, `scenario_init_oracle.py`;
  - the earlier self-play, now unused for training: `az_loop.py`,
    `sim_self_play.py`, `mcts.py`.
- **`scripts/`, the runners:**
  - the desktop's: `selfplay_t1_local.py`, `stage_c_local.py`,
    `p_local.py`, `rounds_local.py`, `chain_local.py`, `match_local.py`,
    `local_machine.py`, `gpu_yield.py`;
  - the box's: `rent_box.py`, `scripts/box/`, `*_box.sh`;
  - the autonomous window's arming, watchers and analyses:
    `scripts/window_20261029/`.
- **The live bridge** (evaluation and rule checks only):
  - `wesnoth_ai/wesnoth_interface.py` runs one Wesnoth process per game.
    State arrives via `std_print`, tailed from the log file; actions are
    written atomically as `action.lua` and read with `wesnoth.read_file`.
  - The Lua side (`lua/`) only serializes state and executes actions:
    it never decides game logic.
  - It runs on the project's patched Wesnoth (`tools/wesnoth_build/`).
- **Also:**
  - `main.py`: the setup CLI (`--check-setup`, `--clean-games`).
  - `configs/`: `reference_player.json`, `imitation.json`.
  - `tests/`: the suite; `tests/conftest.py` bootstraps `sys.path`.

## Wesnoth data and rules

- **Coordinates:** Wesnoth is 1-indexed (WML, replays, Lua), and Python is
  0-indexed everywhere internally. The ±1 happens only where Wesnoth data
  enters or leaves Python:
  - the live tools;
  - the WML and replay readers (`tools/replay_extract.py`,
    `wesnoth_ai/rules/scenario_pool.py`, `tools/scenario_events.py`,
    `wesnoth_ai/rules/wml_state.py`);
  - the writers (`tools/sim_to_replay.py`, `tools/dump_savestate.py`, the
    oracles).
- **`wesnoth_src/data/`** is a WML-only copy (cfg, lua, map; no art and no
  `src/` tree) of the Steam install's data, taken from 1.18.7. It is
  refreshed with:

      robocopy "C:\Program Files (x86)\Steam\steamapps\common\wesnoth\data" wesnoth_src\data *.cfg *.lua *.map /S

  - The sim's runtime subset is tracked in git, so a bare clone can train
    with no Wesnoth install:
    - `data/multiplayer/{factions,scenarios,maps}`;
    - `data/core/macros`;
    - the add-ons `Mini_Maps_Collection`, `Seamless_Map_Picker` and
      `WL_Mappack`.
  - The rest is untracked. A refresh shows up as git diffs, on purpose.
- **`unit_stats.json` and `terrain_db.json` are pinned 1.18.4 scrapes.
  Never re-scrape them from `wesnoth_src/`.** Stats drift between releases
  (1.19's Ghoul gained a pierce resistance), and combat parity and every
  checkpoint assume 1.18.4. A re-scrape starts from the 1.18.4 tag on
  GitHub. If a replay's `[scenario] version=` is not 1.18.x, scrape from
  that version's tag instead.
- **Read `docs/wesnoth_rules.md` before researching a rule.** When you
  establish or correct one, add it there with a file:line citation, a
  verbatim quote of the enforcing code, and a "why non-obvious" note where
  it was not where you'd expect.
- **Don't guess Wesnoth semantics: check the source.**
  - For C++, read the 1.18.4 tag on GitHub raw, e.g.
    `https://raw.githubusercontent.com/wesnoth/wesnoth/1.18.4/src/units/unit.cpp`.
  - For WML and Lua, grep `wesnoth_src/data/`.
  - A rule can live in C++, Lua or WML, so search all three. Rules that
    feel like a post-pass often hide in `[event]name=prestart` Lua in
    `data/multiplayer/eras.lua` or `data/core/macros/`.
  - The wiki (https://wiki.wesnoth.org/, the Lua API reference) sometimes
    lags. The engine's changelog is historical: check any entry against
    the code at 1.18.4.

## Code style

- **Files** target about 600 lines; split by responsibility past that.
- **Naming:** classes `PascalCase`, functions `snake_case`, constants
  `UPPER_SNAKE_CASE`, private members `_leading_underscore`. Lua follows
  the same.
- **Imports:** no unused imports and no circular imports (pass data
  explicitly). Use `TYPE_CHECKING` blocks for type-only cycles.
- **Data:**
  - Structured data is a `@dataclass`, not a raw dict. Action payloads on
    the wire are dicts, since they cross the Lua boundary.
  - Faction and material properties are lookup tables in config, not
    `if`-chains.
- **Encapsulation:** Python systems talk through explicit APIs (on the live
  bridge, `WesnothGame`), never through private attributes.
- **Numbers:** if a number's origin isn't obvious from its name or a
  one-line comment, explain it. A derived constant goes in
  `docs/design_constants.md`, cited from the code. A tuning knob stays in
  `constants.py` with its rationale beside it.
- **Prefer removing over adding.** The codebase is recovering from bloat; a
  feature comes back with evidence.
- **Lint:** `ruff check .` is clean before every commit (config in
  `ruff.toml`). Intentional exceptions get a targeted `# noqa: <rule>` with
  a reason, never a file-wide ignore.

## Branches, versions, pushing

- **GitHub flow:** `main` holds finished, checked work.
  - Everything else goes on a short-lived topic branch cut from `main`:
    `feature/`, `fix/`, `test/` or `exp/<name>`, one task per branch.
  - Parallel sessions each use their own branch in their own worktree.
- **Ready to merge** means ruff is clean and the branch's latest CI run is
  green, both tiers (`gh run list --branch <branch>`; on the desktop the
  `gh` on PATH is an unrelated Python package, and GitHub's CLI installs
  with `winget install GitHub.cli`). When CI cannot run
  because nothing can be pushed, the stand-in is the full suite on the
  desktop at the merged code, named in the merge commit.
- **Merges into `main`:**
  - Always `git merge --no-ff`, never squash: the commit messages are the
    lab notebook.
  - **Ask the user before merging into `main`,** and before pushing `main`
    for anything beyond a trivial fix.
  - Trivial fixes (records, docs, typos, small mechanical fixes that change
    no behaviour) go straight to `main`, with their version bump.
- **`exp/` branches:** the code merges only if it wins. Its
  pre-registration and result land on `main` either way.
- **Records:** status entries (`docs/status.md`, `BACKLOG.md`) are written
  on `main` when the work lands, not on the branch.
- **Deleting a branch** that holds commits `main` lacks: first tag its tip
  `archive/<name>` and push the tag.
- **The version:** `wesnoth_ai.__version__` (in `wesnoth_ai/__init__.py`)
  numbers the states of `main`.
  - Every change that lands bumps it once: M for a feature, P for anything
    else. N changes only on the user's decision.
  - Bump it in the commit that lands the change (`git merge --no-ff
    --no-commit`, bump, commit), and state the new version on the first
    line of the commit body. Topic branches never touch it.
- **Pushing:**
  - Pushing a topic branch is free (2026-09-23).
  - The credentials are the user's: they sign in themselves
    (`docs/new_machine.md`, step 2).
  - When none are stored, a push fails at the password prompt. Commit
    locally and tell the user what is waiting to be pushed.

## Testing

- **Tests catch regressions** you would otherwise find only after burning
  a training run. Prefer a few behavioural tests over line coverage.
  - No test launches Wesnoth. The live bridge's test
    (`tests/test_lua_actions.py`) uses synthetic inputs.
  - The engine oracles run only by hand.
- **Two tiers:**
  - A bare `pytest` runs the fast tier. `slow` tests (full games,
    subprocesses, threading e2e) are excluded by `pytest.ini`.
  - CI (`.github/workflows/tests.yml`) runs `pytest -m ""` on every push,
    with a freshly built Rust wheel, and is the merge gate.
  - A fast-tier test over 10 s moves to the slow tier or is slimmed: CI
    lists the 30 slowest tests on every run.
- **Before a push:** run ruff and the test files that exercise the change.
- **Never run more than one pytest at a time.** Each one imports torch and
  the model (5+ GB), and parallel or stuck runs have locked the machine
  before.
  - Wait for each run to return, or kill it, before starting another.
  - `scripts/window_20261029/pytest_locked.py` serializes runs across
    sessions.
- **Run pytest from the Bash tool,** not the PowerShell tool. In
  PowerShell's environment `bash` is the WSL stub, and 38 box-script tests
  fail there.
- **Every run lists its failed, errored and skipped tests** (`-rfEs`). A
  skip count that differs between two runs of one tree is a test that
  sometimes does not run: diff the SKIPPED lists to name it.
- **A test that builds a search policy seeds it** (`rng_seed=` or `rng=`).
- **Use `constants.py` values in assertions,** not copies.
- **Never weaken a test without the user's confirmation.** A failing test
  is a signal: find the root cause.
- **The Rust core:**
  - CI tests every Rust change.
  - A local run tests whatever wheel is installed, so check `python
    tools/kernel_status.py` before believing one.
  - To rebuild, from `rust/wesnoth_core` (`docs/new_machine.md`, step 5):

        CARGO_TARGET_DIR="$USERPROFILE/.cargo-target" maturin build --release -i python --out "$USERPROFILE/.cargo-target/wheels"
        pip install --force-reinstall --no-deps "$USERPROFILE"/.cargo-target/wheels/wesnoth_core-*.whl

    Defender blocks builds outside the excluded `.cargo-target` folder.
  - Never install a wheel while a pytest run or a training run is going.
- **CI has no GPU and no corpora,** so the CUDA tests and the tests that
  read replay or imitation data skip there. The HIP tests
  (`tests/test_hip_kernels.py`) run on the desktop under the ROCm venv.

## Working style

- **High autonomy on reversible local work:** edits, tests, reads, and
  topic branches (create, commit, push).
- **Ask first before:**
  - merging into `main`, or pushing `main` beyond a trivial fix;
  - force-pushing;
  - deleting a branch whose work is unmerged, or deleting tracked files;
  - architectural changes (new IPC, replacing the model);
  - changing the reference player;
  - spending money.
- **Give best effort:** production-quality code with its edge cases
  handled, not sketches.
- **Don't assert without checking.** This holds for claims about Wesnoth
  and about this code ("the function does X", "this list is complete").
  Before a confident sentence goes in a commit message, a comment or a
  reply, verify it with a grep, a read or a test, or hedge it. Before
  listing a closed set, grep for its marker and count.
- **Credentials never reach a transcript** (user decision 2026-09-29).
  - `tools/secret_guard` redacts credentials from every tool result, so a
    `<redacted>` in output is the guard at work.
  - A new key is created in the user's own terminal or browser.
  - Keys are restricted: Vast to offers and instances, Hugging Face to
    write on the checkpoint repository only.
  - Every checkpoint load uses `weights_only=True`.
