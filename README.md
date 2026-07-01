# Wesnoth AI

A reinforcement-learning AI for *Battle for Wesnoth* 1.18.x, trained
by self-play and warm-started by behavior cloning of human replays.

Two goals shape the design:

1. **Competitively strong** against human players.
2. **Readable** enough that humans can study its strategy. Behavioral
   knobs (rewards, openers, biases) live in code/config, not buried
   in network weights — a modder should be able to flip a behavior
   without touching the model.

## Status (2026-06-30)

**Post-hiatus review done; training path unblocked, all fixes on `main`.**
A 2026-06-29 correctness+optimization review fixed a critical blocker —
`--mcts` self-play was training a **frozen** inference net (the AlphaZero
loop never closed) — plus friendly-unit pathfinding, an actor-pool
watchdog, thread-safe dynamic vocab growth, a combat-oracle anneal that
now actually runs in MCTS mode, and the B1 batched-Gumbel GPU throughput
lever. Full pytest suite green (365+ tests). Itemised in `BACKLOG.md`
§2026-06-29; a new-instance handoff lives in `CLAUDE.md`
"Current status (2026-06-30)". The code is ready to rent GPU compute; the
CUDA path has never run on real hardware, so the next step is the
required GPU smoke + throughput profiling in `docs/running_on_gpu.md`,
launching a fresh campaign with `--reset-decision-step`.

## Status (2026-06-11, superseded)

- **Training is local-only for now, but GPU-ready.** The ENSTA
  Mesogip cluster is permanently inaccessible (2026-06); the
  SLURM/sync/GUI infrastructure was removed (preserved in git
  history). To run on new GPU/cloud compute, the self-play
  entrypoint is device-aware — see
  [docs/running_on_gpu.md](docs/running_on_gpu.md) (launch flags,
  device-aware defaults, and the required first-run CUDA smoke).
  Reward/weight configs live in `configs/`. The project was
  recovered onto a new machine 2026-06-11; the replay corpus
  (`replays_raw/`, `replays_dataset/`) did not survive the move —
  re-download via `tools/download_replays.py` and re-extract via
  `tools/replay_extract.py` before supervised training or the
  corpus-dependent tests (~21 currently skip).
- **Simulator-driven self-play is the production training path.**
  `tools/wesnoth_sim.py` is a pure-Python reimplementation of
  Wesnoth 1.18.4's game logic — bit-exact for combat (731/731
  strikes verified via `[mp_checkup]` oracle on strict-sync
  replays), **99.85% `diff_replay` clean rate on the freshly-
  extracted 5,490-replay competitive-2p corpus** (2026-05-11
  sweep, 8 residual divergences flagged for follow-up).
  ~1000× faster than driving Wesnoth as a subprocess.
- **Self-play training entry point: `python tools/sim_self_play.py`.**
  Drives N games per iteration through `WesnothSim`, applies
  REINFORCE+baseline gradient updates. `--mcts` flag swaps in
  AlphaZero-style PUCT search. Auto-resumes from `sim_selfplay.pt`
  or warm-starts from the highest `supervised_epoch*.pt`.
- **Distributional value head + cliffness signal landed
  2026-05-10.** C51 head over [-1, +1], `output.cliffness =
  std(Z(s))` published per forward. Bayesian-precision bootstrap
  weighting and adaptive sim budget framework are wired but
  default OFF pending calibration.
- **Supervised pre-training** (behavior cloning of 1.18.x human
  replays) produced the `supervised*.pt` checkpoints in
  `training/checkpoints/`. Used as the warm-start for self-play.
- **The live-Wesnoth IPC path** (`wesnoth_interface.py`, Lua
  bridge) is retained only for `tools/eval_vs_builtin.py` (pits
  the trained model against Wesnoth's RCA AI). The training-via-
  subprocess path (`game_manager.py`) and live-watch `--display`
  mode were both retired 2026-05-11; demos now run via
  `tools/sim_demo_game.py` which exports a Wesnoth-loadable
  `.bz2` for the GUI's replay viewer.

## Layout

```
wesnoth_src/                 1.18.4-pinned Wesnoth source (read-only)
add-ons/wesnoth_ai/          Lua side of the live-Wesnoth bridge
                             (display + eval only, NOT training).
                             Junctioned into Wesnoth's userdata at
                             first run.
   _main.cfg                 add-on entry point
   scenarios/training_scenario.cfg
   lua/state_collector.lua   serializes WML state into a framed block
   lua/turn_stage.lua        custom AI turn stage (replaces default RCA)
   lua/action_executor.lua   move/attack/recruit/recall/end_turn dispatcher
   lua/json_encoder.lua      WML → JSON for the state channel
classes.py                   GameState, Unit, Hex, Map, state_key
constants.py                 paths, hyperparameters, IPC tunables
encoder.py                   GameState → tensors (per-unit / per-hex /
                             recruit phantom features / global)
model.py                     transformer policy + C51 distributional
                             value head; cliffness = std(Z(s))
action_sampler.py            enumerate_legal_actions_with_priors,
                             predict_priors (factored), combat-oracle
                             attack-bias on target logits
trainer.py                   REINFORCE+baseline (step) and AlphaZero
                             soft-target distillation (step_mcts);
                             both use categorical CE value loss
transformer_policy.py        Policy adapter (select_action / observe /
                             train_step / save_checkpoint / finalize_game)
dummy_policy.py              scripted-baseline policy
policy.py                    Policy protocol + name registry
rewards.py                   StepDelta + WeightedReward shaping
state_converter.py           Wesnoth WML → GameState (eval bridge)
wesnoth_interface.py         one Wesnoth subprocess per eval game
main.py                      setup / maintenance CLI
                             (--check-setup, --clean-games).
                             Training is via tools/sim_self_play.py;
                             demos via tools/sim_demo_game.py.
tools/
   wesnoth_sim.py            **PURE-PYTHON SIMULATOR** — production
                             training data source. Bit-exact for combat.
   sim_self_play.py          **SELF-PLAY ENTRY POINT.** REINFORCE
                             default; --mcts for AlphaZero-style search.
   sim_demo_game.py          one-game inference + bz2 export (display)
   sim_dummy_smoke.py        pre-flight pipeline check
   sim_to_replay.py          export sim trajectories to .bz2 replays
   mcts.py                   AlphaZero-style MCTS (PUCT, virtual loss,
                             transposition table, cliffness consumers)
   mcts_policy.py            MCTSPolicy adapter wrapping TransformerPolicy
   scenario_pool.py          random scenario+faction selection for training
   scenarios.py              auto-derived 2p competitive scenario allowlist
   replay_extract.py         raw bz2 replays → tagged JSON corpus
   replay_dataset.py         tagged JSON → (state, action) training pairs;
                             also the bit-exact replay-reconstruction
                             machinery the simulator reuses
   replay_builder.py         build .bz2 replays from sim trajectories
   scenario_events.py        scenario [event] dispatch (Hornshark
                             pre-placed units, AMLA [choose] queue, etc.)
   supervised_train.py       behavior-cloning pre-training
   scrape_unit_stats.py      wesnoth_src → unit_stats.json
   scrape_terrain.py         wesnoth_src → terrain_db.json
   terrain_resolver.py       runtime terrain / movement / defense lookups
   traits.py, abilities.py   trait / ability engine (capped defenses,
                             leadership, drain, steadfast, plague, etc.)
   combat.py, combat_oracle.py
                             bit-exact combat simulation (replay parity)
   fog.py                    visibility BFS (per-unit vision range)
   diff_replay.py            sim-vs-Wesnoth replay regression check
   diff_combat_strike.py     per-strike vs Wesnoth [mp_checkup] oracle
   verify_mp_checkup.py      parse [mp_checkup] from strict-sync .bz2
   eval_vs_builtin.py        bridge-path eval vs Wesnoth's RCA AI
   eval_runner.py / eval_scenarios.py
                             eval harness (Wilson CIs, per-faction grid)
   download_replays.py       scraper for the 1.18.x replay server
   filter_replays.py         dataset selection helpers
   purge_mod_replays.py      purge mod-using replays (e.g. Biased RNG)
configs/                     reward_selfplay.json, action_type_weights.json
docs/
   wesnoth_rules.md          authoritative catalog of Wesnoth-engine
                             rules (combat rounding, traits, abilities,
                             terrain alias resolution, etc.) with
                             verbatim source citations
   design_constants.md       derived numerical constants (e.g.
                             cliffness_max = 1/√3) — read this when
                             you wonder where a magic number came from
training/
   checkpoints/              .pt files (supervised + self-play)
   replays/                  per-game logs from self-play
   logs/                     training job logs
benchmarks/                  bench_mcts_tt.py, bench_sim_throughput.py
replays_raw/                 source .bz2 replays (gitignored, ~46k vanilla)
replays_raw_set_aside/       .bz2 replays whose only mod is plan_unit_advance
                             (gitignored, ~28k)
replays_dataset/             6,224 extracted .json.gz replays — strictly
                             default-era 2p, competitive, vanilla map,
                             no mods. The supervised + sim-self-play
                             training corpus. Refreshed 2026-05-10
                             via `tools/sort_replays.py`. Index in
                             `index.jsonl` carries `mods` and `bucket`
                             fields per record.
replays_dataset_quarantine/  Everything else from the source bz2s,
                             classified by criterion (gitignored):
   modded/<mod_id>/            Replays carrying any [modification] or
                               active_mods= entry. Per-mod folders;
                               currently:
                                 plan_unit_advance/   24,327 (UI mod)
                                 Color_Modification/     172 (cosmetic)
                                 Rav_Color_Mod/           34 (cosmetic)
   non_2p/                     37,110 — multi-side / non-default-era
                               (Dunefolk, World-Conquest-II Custom,
                               etc.) / campaign-stage replays.
   non_vanilla_map/              887 — 2p default-era replays on
                               non-mainline scenarios (random map,
                               user maps).
unit_stats.json              scraped from wesnoth_src — rebuild on
                             Wesnoth version bump
terrain_db.json              terrain/defense/move-cost tables, scraped
```

## Architecture in one paragraph

The simulator (`tools/wesnoth_sim.py`) is a pure-Python
reimplementation of Wesnoth 1.18.4's game logic, sharing the
bit-exact replay-reconstruction machinery from
`tools/replay_dataset.py`. Self-play training
(`tools/sim_self_play.py`) drives N games per iteration through the
simulator with both sides controlled by the same policy, then
applies one gradient update via `policy.train_step`. Combat math is
verified bit-exact against Wesnoth's `[mp_checkup]` oracle on
strict-sync replays; full-replay reconstruction at 98.57% clean on
the competitive-2p corpus.

The model is a transformer over tokenized state (per-unit, per-hex,
recruit-phantom, global features). One forward produces actor logits
over "unit slots + recruit slots + end-turn", per-actor type logits
(ATTACK / MOVE), target logits over the hex grid, weapon logits per
attack candidate, plus a categorical (C51) value distribution over
51 atoms in [-1, +1]. The mean of that distribution is the value
estimate (used as REINFORCE baseline); its standard deviation is
exposed as `cliffness` (used by the MCTS bootstrap-weighting +
adaptive-budget consumers, off by default).

The live-Wesnoth IPC bridge (`wesnoth_interface.py` +
`add-ons/wesnoth_ai/`) is now kept only for `eval_vs_builtin.py`
(pits the trained model against Wesnoth's RCA AI). The
training-via-subprocess path (`game_manager.py`) and live-watch
`--display` mode were both retired 2026-05-11; the
`tools/sim_demo_game.py` + Wesnoth-replay-viewer combo covers
the demo case. Wesnoth on Windows is a GUI-subsystem binary (no
piped stdout), the Lua sandbox forbids `io`, and Linux headless
mode wasn't workable on the cluster — all of which were the
original motivations for the simulator pivot.

## Setup

```powershell
# One-time: install Wesnoth 1.18.x (Steam is fine), launch it once so
# the userdata directory exists, then:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Re-pin wesnoth_src to 1.18.4 if you cloned fresh:
git -C wesnoth_src checkout 1.18.4

# Re-scrape unit stats from the pinned source:
python tools/scrape_unit_stats.py wesnoth_src unit_stats.json

# Sanity check Lua add-on / Wesnoth detection:
python main.py --check-setup
```

`--check-setup` verifies Wesnoth is at the expected path, the project
add-on source files are present, and the directory junction from
Wesnoth's userdata to `add-ons/wesnoth_ai/` is in place.

## Self-play

Self-play training runs in the in-process Python simulator
(`tools/wesnoth_sim.py`). Both sides are driven by the same policy.
~1000× faster than driving real Wesnoth, and the
combat math is bit-exact verified against Wesnoth's `[mp_checkup]`
oracle.

### Local

```powershell
# Trainable transformer, warm-started from supervised checkpoint:
python tools/sim_self_play.py `
    --checkpoint-in training/checkpoints/supervised_epoch3.pt `
    --checkpoint-out training/checkpoints/sim_selfplay.pt `
    --iterations 50 --games-per-iter 8 --max-turns 60 `
    --reward-config configs/reward_selfplay.json

# AlphaZero-style MCTS (added 2026-05). PUCT search per move,
# soft-target distillation against visit-count distributions:
python tools/sim_self_play.py `
    --checkpoint-in training/checkpoints/supervised_epoch3.pt `
    --checkpoint-out training/checkpoints/sim_selfplay.pt `
    --mcts --mcts-sims 50 --mcts-c-puct 1.5 `
    --iterations 50 --games-per-iter 8 --max-turns 60

# Random-baseline (no learning — pre-flight pipeline check):
python tools/sim_dummy_smoke.py
```

Useful flags:
  - `--workers N` — rollout worker threads. CPU-bound; leave 1-2
    cores free for the main thread and the OS.
  - `--mcts` — enable AlphaZero-style search. Per-step shaping
    rewards are silently ignored in MCTS mode (only terminal z is
    distilled).
  - `--forced-faction "Knalgan Alliance"` — lock at least one side
    to a given default-era faction. Empty string = use the
    sim_self_play module default; `none` = explicit disable.
  - `--reward-config PATH` — JSON file defining shaping reward
    weights. See `configs/reward_selfplay.json` for the
    canonical template (terminal ±1, gold/damage/village deltas,
    per-turn penalty, unit-type bonuses, turn-conditional bonuses).

Training auto-resumes from `sim_selfplay.pt` if present, else
warm-starts from the highest `supervised_epoch*.pt`, else from
`supervised.pt`, else from random init (warning, not error).

### What to watch for in the logs

```
[iter 042] games=8 wins/draws/losses=3/0/5 timeouts=0
           mean_return=0.18 entropy=4.21 grad_norm=0.83
           policy_loss=0.42 value_loss=0.31
mcts: root cliffness=0.34 (adaptive=off, n_sims=50)
```

`mean_return` and `wins/draws/losses` are the headline numbers.
`mcts: root cliffness=...` (when MCTS is on) is always logged at
debug level so you can collect distributions for tuning the
adaptive sim budget later.

### Watch a trained model play one game

```powershell
# Auto-picks the latest supervised*.pt checkpoint, plays one game
# headlessly via the sim, exports a Wesnoth-loadable .bz2, copies
# it into your Wesnoth saves dir (visible under File -> Load Game).
python tools/sim_demo_game.py

# Force a specific scenario:
python tools/sim_demo_game.py --scenario multiplayer_Hamlets
```

The `.bz2` is composed entirely from `wesnoth_src/` + the sim's
command history -- no source replay needed. Works for any of the
21 Ladder Era maps.

### Live-Wesnoth eval (against Wesnoth's RCA AI)

```powershell
# Pit the trained model against Wesnoth's built-in RCA across a
# (map x matchup x side-swap) matrix. Uses real Wesnoth subprocesses
# via the eval bridge.
python tools/eval_vs_builtin.py --resume <ckpt>
```

## Supervised pre-training (behavior cloning)

The supervised path uses the same encoder+model architecture as
self-play and produces a `--resume`-compatible checkpoint.

### Build the dataset (once per Wesnoth version bump or pipeline change)

```powershell
# 1. Scrape ~200k 1.18.x replays from the add-on server (~5GB bz2):
python tools/download_replays.py replays_raw

# 2. Extract → tagged JSON, applying scenario + version filters:
python tools/replay_extract.py replays_raw replays_dataset

# 3. (Optional) Inspect or filter a single replay:
python tools/dump_savestate.py replays_dataset/<id>.json.gz
```

### Train locally (CPU, slow)

```powershell
python tools/supervised_train.py replays_dataset \
    --bs 32 --lr 1e-4 --epochs 10 \
    --device cpu \
    --log-every 100 --ckpt-every 500 \
    --checkpoint training/checkpoints/supervised.pt
```

Throughput on a typical desktop CPU: ~20 pairs/sec. Single epoch is a
multi-day run. DirectML on AMD GPUs (RX 6600 here) was tried, did
not help (the bottleneck is the encoder's Python state-build on CPU,
not the tensor work) and triggered Windows TDR on backward passes —
default device stays at CPU.

`--workers N>0` prefetches the per-pair `encode_raw` step in `N`
subprocesses. **Helpful only when the model side is fast enough that
encoding is the bottleneck — i.e., on a CUDA GPU.** Don't enable on
local CPU runs: the workers compete with the main thread for CPU and
make things slower.

After supervised pre-training, continue with self-play:

```powershell
python tools/sim_self_play.py `
    --checkpoint-in training/checkpoints/supervised_epoch3.pt `
    --checkpoint-out training/checkpoints/sim_selfplay.pt `
    --iterations 10 --games-per-iter 8
```

To watch a trained model play, use `tools/sim_demo_game.py` (exports
a Wesnoth-loadable `.bz2` for the replay viewer — see above).

## Evaluating against Wesnoth's built-in AI

`tools/eval_vs_builtin.py` runs the trained transformer against
Wesnoth's default RCA AI across a matrix of (map × faction matchup ×
side-swap), reports overall and per-faction / per-map / per-matchup
win rates, and dumps a JSON for further analysis.

```powershell
# Quick read (~30 games, ~2h on 4 parallel) -- 2 maps, no mirrors, no swap:
python tools/eval_vs_builtin.py \
    --checkpoint training/checkpoints/supervised_epoch3.pt \
    --maps caves den --pairs cross --no-swap \
    --parallel 4 --save-json eval_results.json

# Full grid (252 games -- many hours): 6 maps × 21 pairs × 2 swaps.
python tools/eval_vs_builtin.py \
    --checkpoint training/checkpoints/supervised_epoch3.pt
```

Key flags:
- `--checkpoint PATH` (required): trained `.pt` to evaluate.
- `--maps {caves,den,sablestone,hornshark,hamlets,freelands}`: subset of
  the 6 popular 2p maps (default: all).
- `--pairs {all,cross}`: `all` includes mirrors (21 pairs); `cross` is
  non-mirror only (15).
- `--no-swap`: skip the side-swap rerun. Halves game count but
  introduces first-mover bias.
- `--parallel N`: max parallel Wesnoth processes.
- `--save-json PATH`: persist structured results.

The script generates per-matchup `.cfg` scenarios under
`add-ons/wesnoth_ai/scenarios/eval/` (gitignored — rebuilt each run)
and adds a glob include to `_main.cfg` on first run.

## Tests

Tests are Python-only — they exercise the WML parser, combat oracle,
and Lua-action serialization with synthetic inputs. They do **NOT**
launch Wesnoth. End-to-end testing is live self-play (above).

```powershell
pytest                   # all
pytest test_integration.py       # synthetic-input integration tests
pytest test_lua_actions.py
```

When tests fail, fix the underlying code; **do not relax the
assertions**. A failing test is a signal.

## Coordinates

- Wesnoth is 1-indexed in WML and Lua.
- Python is 0-indexed everywhere internally.

`state_converter.py` does the ±1 conversion in both directions. Don't
sprinkle ±1 elsewhere.

## Wesnoth source pinning

`wesnoth_src/` MUST be on the **1.18.4 tag**:

```powershell
git -C wesnoth_src checkout 1.18.4
```

Wesnoth's `master` is 1.19.x and unit stats drift between releases
(Ghoul resistance overrides changed, etc.). Re-running
`scrape_unit_stats.py` against `master` gave wrong damage numbers and
broke replay reconstruction. After any Wesnoth version change,
re-scrape.

## Useful ad-hoc commands

```powershell
# Inspect a single extracted replay:
python -c "import gzip,json; print(json.loads(gzip.decompress(open(r'replays_dataset/<id>.json.gz','rb').read()))['header'])"

# Count training-eligible replays under the 2p competitive allowlist:
python -c "from tools.replay_dataset import filter_competitive_2p; from pathlib import Path; print(len(filter_competitive_2p([p for p in Path('replays_dataset').glob('*.json.gz')], Path('replays_dataset'))))"

# Dump the auto-derived 2p scenario allowlist:
python -c "from tools.scenarios import COMPETITIVE_2P_SCENARIOS; [print(s) for s in sorted(COMPETITIVE_2P_SCENARIOS)]"

# What checkpoints do I have:
ls training/checkpoints
```

## Troubleshooting

**`Timeout waiting for state` in self-play.** The Wesnoth process
launched but never wrote a state frame. Check
`%USERPROFILE%\Documents\My Games\Wesnoth1.18\logs\wesnoth-*.out.log`
for Lua errors — they should appear there if the Lua side `pcall`-wrapped
something that threw. If the log file is empty, Wesnoth itself didn't
launch (bad path in `constants.py`?).

**`mklink` permission errors at first run.** The add-on directory
junction needs write access to
`%USERPROFILE%\Documents\My Games\Wesnoth1.18\data\add-ons\`. Run from
a writable shell, not as a different user.

**Replay extraction crashes on a specific file.** `replay_extract.py`
salvages most malformed WML via `_safe_int` and tag-skip logic. If a
replay still kills it, log the file path, skip it, and move on —
207k inputs, you can afford to drop the genuinely-broken few.
