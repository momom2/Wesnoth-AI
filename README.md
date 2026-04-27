# Wesnoth AI

A reinforcement-learning AI for *Battle for Wesnoth* 1.18.x, trained
by self-play and warm-started by behavior cloning of human replays.

Two goals shape the design:

1. **Competitively strong** against human players.
2. **Readable** enough that humans can study its strategy. Behavioral
   knobs (rewards, openers, biases) live in code/config, not buried
   in network weights — a modder should be able to flip a behavior
   without touching the model.

## Status

- The Python ↔ Wesnoth IPC works. Self-play games run end-to-end at
  ~150 games/hour with 4 parallel processes.
- Supervised pre-training (behavior cloning of 5,399 1.18.x human
  replays) is currently running on the ENSTA Mesogip cluster (NVIDIA
  L40S). The trained encoder+model is `--resume`-compatible with the
  self-play path.
- Random-init self-play stalled in a "never finds combat" local
  optimum. Warm-starting from the supervised checkpoint should
  unblock that.

## Layout

```
wesnoth_src/                 1.18.4-pinned Wesnoth source (read-only)
add-ons/wesnoth_ai/          Lua side of the bridge — junctioned into
                             Wesnoth's userdata at first run
   _main.cfg                 add-on entry point
   scenarios/training_scenario_mp.cfg
   lua/state_collector.lua   serializes WML state into a framed block
   lua/turn_stage.lua        custom AI turn stage (replaces default)
classes.py                   GameState, Unit, Hex, Map, ...
constants.py                 paths, hyperparameters, IPC tunables
state_converter.py           Wesnoth WML → GameState (and back)
encoder.py                   GameState → tensors (per-actor / per-hex)
model.py                     transformer policy/value network
action_sampler.py            sample legal action from logits
trainer.py                   REINFORCE + value baseline + entropy
transformer_policy.py        Policy implementation (rollout + train)
dummy_policy.py              random-action baseline
policy.py                    Policy protocol + name registry
game_manager.py              N parallel games, per-step reward feed
wesnoth_interface.py         one Wesnoth subprocess per game
rewards.py                   StepDelta + WeightedReward shaping
main.py                      entry point (`python main.py --help`)
tools/
   replay_extract.py         raw bz2 replays → tagged JSON corpus
   replay_dataset.py         tagged JSON → (state, action) training pairs
   supervised_train.py       behavior-cloning pre-training
   scrape_unit_stats.py      wesnoth_src → unit_stats.json
   scenarios.py              auto-derived 2p competitive scenario list
   traits.py, abilities.py   trait / ability engine (capped defenses, etc.)
   combat.py, combat_oracle.py
                             bit-exact combat simulation (replay parity)
   fog.py                    visibility BFS (per-unit vision range)
   filter_replays.py         dataset selection helpers
   download_replays.py       scraper for the 1.18.x replay server
cluster/                     SLURM scripts for ENSTA Mesogip
   build_bundle.ps1          tarball the project for upload
   setup.sh                  venv + torch+CUDA on a fresh login node
   run.sh                    sbatch wrapper (start/status/tail/stop)
   job.sbatch                the actual training job (auto-resume + chain)
   RUNBOOK.md                step-by-step deployment guide
training/
   checkpoints/              .pt files (supervised + self-play)
   replays/                  per-game logs from self-play
replays_raw/                 207k bz2 replays (gitignored, ~5GB)
replays_dataset/             53k extracted JSON.gz replays (~390MB)
unit_stats.json              scraped from wesnoth_src — rebuild on
                             Wesnoth version bump
```

## Architecture in one paragraph

Wesnoth is a closed game on Windows: no headless mode that we could get
working, no stdout from a piped subprocess (it's a GUI-subsystem
binary), no `io` in the Lua sandbox. Each game is a real Wesnoth
process running our scenario; a custom Lua AI stage publishes state
by `std_print()`-ing a framed block (Wesnoth routes this to
`<userdata>/logs/wesnoth-*.out.log`); Python tails that file. For
actions, Python writes `action.lua` atomically with a monotonic seq;
Lua reads it via `wesnoth.read_file` and ignores stale seqs. See
`wesnoth_interface.py`'s docstring for the gory details.

The model is a transformer over tokenized state (per-unit + per-hex
embeddings + global features). One forward produces actor logits over
"unit slots + recruit slots + end-turn", target logits over the hex
grid, and weapon logits per attack candidate. `action_sampler.py`
samples a legal triple. The same forward also produces a value
estimate used as the REINFORCE baseline.

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

Self-play training runs N parallel Wesnoth processes, both sides AI,
both controlled by the same policy.

```powershell
# Random-baseline self-play (no learning — verifies the bridge):
python main.py --policy dummy --games 4

# Trainable transformer, fresh init:
python main.py --policy transformer --games 4

# Resume from a checkpoint (e.g. the supervised warm-start):
python main.py --policy transformer --resume training/checkpoints/supervised.pt --games 4

# Or pick the latest periodic self-play checkpoint automatically:
python main.py --policy transformer --resume latest --games 4
```

Useful flags:
  - `--games N` — parallelism. CPU-bound. 4 is the practical max on
    typical desktops; 1 trades throughput for faster gradient updates
    (see the per-step CPU contention discussion in
    `transformer_policy.py`).
  - `--check-setup` — verify the install link, exit.

What to watch for in the logs:
```
Game G done: turn=N outcome=timeout | s1 acts=A(inv=I) mv=M atk=T rec=R(Gg) ...
Stats: games=100 entropy=7.17 avg_loss=... wins=3 timeouts=97 ...
```
`atk=` is the load-bearing number. If it stays at 0 after a few
hundred games of fresh-init training, you're stuck in the
"never finds combat" local optimum — warm-start from a supervised
checkpoint instead.

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
bash cluster/setup.sh        # ~2 min: venv + torch+CUDA + sanity check
bash cluster/run.sh start    # sbatch the job; auto-chains across walltime
bash cluster/run.sh status   # squeue + last 15 log lines + GPU info
bash cluster/run.sh tail     # follow the live SLURM log
bash cluster/run.sh stop     # scancel the recorded job id
```

The job (`cluster/job.sbatch`):
- Runs on `ENSTA-l40s` with 1 GPU, 8 CPUs, 32GB RAM.
- Walltime 03:55:00 — under the student-QoS 4h cap.
- `--workers 6`: 6 prefetch subprocesses run `encode_raw` (CPU-bound,
  ~3-4ms/pair) while the main thread runs `encode_from_raw` + model
  forward/backward on the GPU. Hides CPU encode time behind GPU step
  time. Workers receive a frozen vocab snapshot at startup
  (pre-seeded from `unit_stats.json`); out-of-vocab unit types hit
  the overflow embedding row.
- Auto-resumes from `training/checkpoints/supervised.pt` if present.
- Auto-chains a follow-up job when the trainer ends cleanly OR via
  SIGTERM (143 = walltime hit), unless `done.flag` exists.
- `done.flag` is created when `supervised_epoch9.pt` shows up — i.e.
  all 10 epochs completed.

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

Then launch self-play locally with the pulled checkpoint:
```powershell
# Auto-pick the freshest supervised*.pt by mtime:
powershell -ExecutionPolicy Bypass -File run_self_play.ps1

# Specific checkpoint, single game (good for watching one play through):
powershell -ExecutionPolicy Bypass -File run_self_play.ps1 `
    -Checkpoint training\checkpoints\supervised_epoch3.pt -Games 1
```

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
