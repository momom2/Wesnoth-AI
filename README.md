# Wesnoth AI

A reinforcement-learning AI for *Battle for Wesnoth* 1.18.x, trained
by self-play and warm-started by behavior cloning of human replays.

Two goals shape the design:

1. **Competitively strong** against human players.
2. **Readable** enough that humans can study its strategy. Behavioral
   knobs (rewards, openers, biases) live in code/config, not buried
   in network weights — a modder should be able to flip a behavior
   without touching the model.

## Status (2026-05-10)

- **Simulator-driven self-play is the production training path.**
  `tools/wesnoth_sim.py` is a pure-Python reimplementation of
  Wesnoth 1.18.4's game logic — bit-exact for combat (731/731
  strikes verified via `[mp_checkup]` oracle on strict-sync
  replays), **99.85% `diff_replay` clean rate on the freshly-
  extracted 5,490-replay competitive-2p corpus** (2026-05-11
  sweep, 8 residual divergences flagged for follow-up).
  ~1000× faster than driving Wesnoth as a subprocess and
  trivially cluster-portable.
- **Self-play training entry point: `python tools/sim_self_play.py`.**
  Drives N games per iteration through `WesnothSim`, applies
  REINFORCE+baseline gradient updates. `--mcts` flag swaps in
  AlphaZero-style PUCT search. Cluster job:
  `cluster/job_selfplay.sbatch` (ENSTA-l40s, auto-resume from
  `sim_selfplay.pt` or warm-start from the highest
  `supervised_epoch*.pt`).
- **Distributional value head + cliffness signal landed
  2026-05-10.** C51 head over [-1, +1], `output.cliffness =
  std(Z(s))` published per forward. Bayesian-precision bootstrap
  weighting and adaptive sim budget framework are wired but
  default OFF pending calibration.
- **Supervised pre-training** (behavior cloning of 1.18.x human
  replays) ran on ENSTA Mesogip; checkpoints `supervised.pt`,
  `supervised_epoch3.pt` available locally. Used as the warm-start
  for self-play.
- **The live-Wesnoth IPC path** (`python main.py`,
  `wesnoth_interface.py`, Lua bridge) still works (Phase 2b
  resolved the CA blacklist via custom Lua AI stage) but is no
  longer the training path — it stays for `--display` (watching
  a trained model play in Wesnoth's GUI) and the
  `tools/eval_vs_builtin.py` harness against Wesnoth's RCA AI.

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
state_converter.py           Wesnoth WML → GameState (bridge path only)
game_manager.py              N parallel Wesnoth subprocesses (bridge)
wesnoth_interface.py         one Wesnoth subprocess per game (bridge)
main.py                      bridge-path entry point (display + eval;
                             NOT training. Use sim_self_play.py for
                             training.)
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
cluster/                     SLURM scripts for ENSTA Mesogip
   build_bundle.ps1          tarball the project for upload
   setup.sh                  venv + torch+CUDA on a fresh login node
   run.sh                    sbatch wrapper (start/status/tail/stop)
   job.sbatch                supervised pre-training job
   job_selfplay.sbatch       SELF-PLAY training job (the production
                             training path on the cluster)
   sync.ps1                  push code-only updates without re-shipping
                             the corpus; -Mode supervised|selfplay
   pull_checkpoint.ps1       pull trained .pt files back to the laptop
   gui.pyw / gui.bat         Tk GUI for cluster + local ops
   configs/                  reward_selfplay.json, action_type_weights.json
   RUNBOOK.md                step-by-step deployment guide
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
   logs/                     SLURM job logs
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

The live-Wesnoth IPC bridge (`main.py` + `game_manager.py` +
`wesnoth_interface.py` + `add-ons/wesnoth_ai/`) is kept for the
`--display` mode and the `eval_vs_builtin` harness, but is no
longer the training path. Wesnoth on Windows is a GUI-subsystem
binary (no piped stdout), the Lua sandbox forbids `io`, and Linux
headless mode wasn't workable on the cluster — all of which were
the original motivations for the simulator pivot.

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
Cluster-portable, ~1000× faster than driving real Wesnoth, and the
combat math is bit-exact verified against Wesnoth's `[mp_checkup]`
oracle.

### Local

```powershell
# Trainable transformer, warm-started from supervised checkpoint:
python tools/sim_self_play.py `
    --checkpoint-in training/checkpoints/supervised_epoch3.pt `
    --checkpoint-out training/checkpoints/sim_selfplay.pt `
    --iterations 50 --games-per-iter 8 --max-turns 60 `
    --reward-config cluster/configs/reward_selfplay.json

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
  - `--workers N` — rollout worker threads. CPU-bound. Cluster
    default is 6 (8 CPUs allocated, 1 main + 1 OS slack + 6 rollouts).
  - `--mcts` — enable AlphaZero-style search. Per-step shaping
    rewards are silently ignored in MCTS mode (only terminal z is
    distilled).
  - `--forced-faction "Knalgan Alliance"` — lock at least one side
    to a given default-era faction. Empty string = use the
    sim_self_play module default; `none` = explicit disable.
  - `--reward-config PATH` — JSON file defining shaping reward
    weights. See `cluster/configs/reward_selfplay.json` for the
    canonical template (terminal ±1, gold/damage/village deltas,
    per-turn penalty, unit-type bonuses, turn-conditional bonuses).

### Cluster (ENSTA Mesogip)

```bash
# Submit one self-play training link (~3h55 walltime):
bash cluster/run.sh start selfplay

# With overrides via env-var passthrough:
sbatch --export=USE_MCTS=1,MCTS_SIMS=50,GAMES_PER_ITER=4 \
       cluster/job_selfplay.sbatch

# Stop the chain:
touch training/checkpoints/selfplay_done.flag

# Sync local code changes + continue:
powershell -ExecutionPolicy Bypass -File cluster/sync.ps1 -Mode selfplay
bash cluster/run.sh continue selfplay
```

The job auto-resumes from `sim_selfplay.pt` if present, else
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

### Live-Wesnoth bridge (display only, NOT training)

```powershell
# Watch a trained model play in Wesnoth's GUI:
python main.py --policy transformer --display `
    --resume training/checkpoints/supervised_epoch3.pt
```

The bridge path (`main.py` + `game_manager.py`) used to be the
training path before the simulator pivot. It's kept for `--display`
and for `tools/eval_vs_builtin.py` (which needs Wesnoth's RCA AI
on the opponent side).

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
make things slower. The cluster job (next section) sets it to 6.

### Train on the ENSTA Mesogip cluster (NVIDIA L40S, ~6× faster)

See `cluster/RUNBOOK.md` for the full step-by-step.

```powershell
# 1. Build the bundle (atomic — won't leave a partial tarball):
powershell -ExecutionPolicy Bypass -File cluster/build_bundle.ps1
# Output: wai_cluster_bundle.tar.gz (~390 MB, dominated by the corpus)

# 2. Upload (CASCAD password ×2 from off-site, ×1 on ENSTA wired):
scp wai_cluster_bundle.tar.gz mesogip_outside:~/

# 3. SSH in and bootstrap:
ssh mesogip_outside
```

On the cluster:
```bash
mkdir -p ~/wesnoth-ai && cd ~/wesnoth-ai
tar -xzf ~/wai_cluster_bundle.tar.gz
bash cluster/setup.sh                  # ~2 min: venv + torch+CUDA + sanity check
bash cluster/run.sh start supervised   # sbatch the supervised job
bash cluster/run.sh status             # squeue + last 15 log lines + GPU info
bash cluster/run.sh tail               # follow the live SLURM log
bash cluster/run.sh stop               # scancel the recorded job id
```

(`cluster/run.sh` is polymorphic: `start supervised` for behavior
cloning, `start selfplay` for self-play, `continue {supervised,
selfplay}` for the next chain link after the previous one ended.)

The supervised job (`cluster/job.sbatch`):
- Runs on `ENSTA-l40s` with 1 GPU, 8 CPUs, 32GB RAM.
- Walltime 03:55:00 — under the student-QoS 4h cap.
- `--workers 6`: 6 prefetch subprocesses run `encode_raw` (CPU-bound,
  ~3-4ms/pair) while the main thread runs `encode_from_raw` + model
  forward/backward on the GPU. Hides CPU encode time behind GPU step
  time. Workers receive a frozen vocab snapshot at startup
  (pre-seeded from `unit_stats.json`); out-of-vocab unit types hit
  the overflow embedding row.
- Auto-resumes from `training/checkpoints/supervised.pt` if present.
- No automatic chain — the operator continues each link manually
  via `bash cluster/run.sh continue supervised` or the GUI's
  "Sync + Continue" button. (Earlier auto-chain didn't survive
  walltime kills reliably; a manual continue also gives the
  operator a chance to drop fixes between links.)
- `done.flag` (touched manually) stops the chain.

The self-play job (`cluster/job_selfplay.sbatch`) follows the same
pattern: no auto-chain, manual continue, env-var overrides for
`USE_MCTS` / `MCTS_SIMS` / `MCTS_C_PUCT` / `MCTS_BATCH_SIZE` /
`GAMES_PER_ITER` / `MAX_TURNS` / `WORKERS` / `SAVE_EVERY` / `SEED`
/ `FORCED_FACTION` / `REWARD_CONFIG`.

Pull the trained checkpoint back when an epoch finishes:
```powershell
# What's available on the cluster:
powershell -ExecutionPolicy Bypass -File cluster\pull_checkpoint.ps1 -List

# Pull the latest per-epoch snapshot (default — stable, immutable):
powershell -ExecutionPolicy Bypass -File cluster\pull_checkpoint.ps1

# Pull the live rolling supervised.pt (may be mid-epoch, but freshest):
powershell -ExecutionPolicy Bypass -File cluster\pull_checkpoint.ps1 -Rolling

# Pull a specific epoch:
powershell -ExecutionPolicy Bypass -File cluster\pull_checkpoint.ps1 -Epoch 3
```

Then continue training locally OR watch a game in Wesnoth's GUI:

```powershell
# Continue self-play training locally (simulator-driven, fast):
python tools/sim_self_play.py `
    --checkpoint-in training/checkpoints/supervised_epoch3.pt `
    --checkpoint-out training/checkpoints/sim_selfplay.pt `
    --iterations 10 --games-per-iter 8

# OR: watch ONE game in Wesnoth's GUI (bridge path, animations on,
# training disabled):
powershell -ExecutionPolicy Bypass -File run_self_play.ps1 -Display

# Specific checkpoint for the display:
powershell -ExecutionPolicy Bypass -File run_self_play.ps1 `
    -Checkpoint training\checkpoints\supervised_epoch3.pt -Display
```

`run_self_play.ps1` is the bridge-path launcher — it spins up real
Wesnoth processes via `python main.py`. It supports both `-Display`
(one GUI window, 2x turbo, training off) and a `TRAINING` mode (N
parallel Wesnoth windows, 10x turbo, training on). The TRAINING
mode predates the simulator pivot and is mostly a curiosity now;
`tools/sim_self_play.py` is faster and produces the same gradient
updates. Use `run_self_play.ps1 -Display` when you want to watch
the trained model play in Wesnoth's GUI.

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

### One-click GUI for the common ops

If repeatedly typing `powershell -ExecutionPolicy Bypass -File ...`
is wearing thin, double-click `cluster\gui.bat` (or `cluster\gui.pyw`
directly if Windows associated `.pyw` with `pythonw.exe`). A small
Tk window appears with:

- A password field (kept in memory only, deleted on window close).
- Buttons for Status / Sync / Sync+Restart / Pull checkpoint
  (the cluster-side ones use the password automatically via
  OpenSSH's `SSH_ASKPASS` hook -- you type it once per session).
- Buttons for Run self-play / Run eval (local, no password).
- A scrolled output panel showing live stdout from the launched
  scripts.
- A Cancel button to terminate the running op.

Stdlib only -- no pip installs needed.

### Push code-only updates without re-shipping the corpus

When you've edited Python locally and want the cluster to pick up the
change without rebuilding the 390MB bundle:

```powershell
# One ssh connection. Lists what would be sent first:
powershell -ExecutionPolicy Bypass -File cluster/sync.ps1 -DryRun

# Actually send (one set of password prompts):
powershell -ExecutionPolicy Bypass -File cluster/sync.ps1

# Send + immediately bounce the running job so the chained follow-up
# picks up the new code now (loses up to ~500 steps from the last
# periodic checkpoint). Without -Restart the new code waits until
# the next natural walltime hop.
powershell -ExecutionPolicy Bypass -File cluster/sync.ps1 -Restart
```

The script sends every `*.py` at the project root, all of `tools/`,
the `cluster/` scripts, and `unit_stats.json`. It deliberately does
**not** touch checkpoints, the corpus, or the Wesnoth source on the
cluster.

Then evaluate locally in self-play:
```powershell
python main.py --policy transformer --resume training/checkpoints/supervised_epoch0.pt --games 4
```

## Tests

Tests are Python-only — they exercise the WML parser, combat oracle,
and Lua-action serialization with synthetic inputs. They do **NOT**
launch Wesnoth. End-to-end testing is live self-play (above).

```powershell
pytest                   # all
pytest test_integration.py       # synthetic-input integration tests
pytest test_wml_parser.py
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

# Is a cluster job still queued (run on cluster):
bash cluster/run.sh status
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

**`cuda not available` in cluster setup.** The login node has no GPU
— that's expected. Training happens via SLURM on a compute node;
`bash cluster/run.sh start` is what gets you onto a GPU. If you see
the message after `bash cluster/run.sh tail` shows the actual job
running, the wheel didn't match the driver — edit `CUDA_TAG=cu118` /
`cu124` at the top of `cluster/setup.sh`.

**Cluster job dies with QoSMaxWallDurationPerJobLimit.** Student QoS
caps a single job at 4h. The sbatch already requests 03:55:00; don't
raise it. Use the auto-chain instead.

**Replay extraction crashes on a specific file.** `replay_extract.py`
salvages most malformed WML via `_safe_int` and tag-skip logic. If a
replay still kills it, log the file path, skip it, and move on —
207k inputs, you can afford to drop the genuinely-broken few.
