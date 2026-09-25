# Wesnoth AI

A reinforcement-learning AI for *Battle for Wesnoth* 1.18.x, warm-started
by behavior cloning of human replays, with self-play training being
rebuilt (status: [CLAUDE.md](CLAUDE.md); plan:
[docs/plan_20260904.md](docs/plan_20260904.md); next actions:
[BACKLOG.md](BACKLOG.md)). Two goals shape the design:

1. **Competitively strong** against human players.
2. **Readable** — behavioral knobs (rewards, openers, biases) live in
   code/config, not buried in weights, so a modder can flip a behavior
   without retraining.

## Quickstart

```powershell
# 1. Environment: Python 3.11 or later (the laptop and CI run 3.13).
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# Optional: the simulator's Rust kernels (needs a Rust toolchain,
# https://rustup.rs). Without them the Python paths run, more slowly.
pip install ./rust/wesnoth_core

# 2. The test suite's fast tier, after every change.
pytest

# 3. Watch the reference player play one game against itself. The game
#    is exported as a .bz2 replay that Wesnoth opens with Load Game.
python tools/reference_player.py --ensure
python tools/sim_demo_game.py --checkpoint $(python tools/reference_player.py --path)
```

On a GPU box, torch comes with the CUDA image instead
(`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`, docs/box_specs.md).

The checkpoints and the replay corpus live in the Hugging Face
repository `momom2/wesnoth-model-checkpoints`, which is private:
`reference_player.py --ensure` and the box scripts need an account with
access to it, logged in with `huggingface-cli login`.

CI (`.github/workflows/tests.yml`) runs the full suite, `pytest -m ""`,
on every push: the fast tier, the slow tier (which plays self-play
games) and the Rust tests against a wheel built from the pushed commit.

Matches run in the simulator (no Wesnoth needed) on a rented GPU box
(docs/box_specs.md). A match of 40 decisive games between a checkpoint
and the reference player (`configs/reference_player.json`), both
players at the reference's decode (temperature 0, end_turn logit offset
-1.5):

```bash
python tools/reference_player.py --ensure      # fetch the reference checkpoint if missing
python tools/run_elo_batch.py \
    --label-a cand --spec-a training/checkpoints/cand.pt --raw-end-turn-offset-a -1.5 \
    $(python tools/reference_player.py --flags b) \
    --outdir eval_games/cand_vs_ref --games 40 --mcts-sims 0 \
    --raw-temperature-a 0 --raw-temperature-b 0 \
    --device cuda --jobs 20 --persistent-workers --shared-inference
python tools/elo_collect.py eval_games/cand_vs_ref --no-catalog
```

Every match game is drawn from the 21-map Ladder pool with fog, and one
of its sides plays the Knalgan Alliance (`FORCED_FACTION` in
`tools/scenario_pool.py`; CLAUDE.md, "Eval procedure").

Imitation training is `tools/supervised_train.py` (its docstring gives
the reference recipe); self-play legs run through `tools/az_loop.py`
(docs/plan_20260904.md).

A bare `git clone` can train: the simulator's runtime WML inputs are
committed, so no Wesnoth install is required for self-play. Wesnoth
itself is needed only for the eval bridge (below) and to watch exported
replays; `python main.py --check-setup` verifies that install.

## Overview

**The simulator is the production training path.**
[`tools/wesnoth_sim.py`](tools/wesnoth_sim.py) reimplements Wesnoth
1.18.4's game logic in-process, with no rendering and no engine
subprocess. The game logic is Python; when the wheel is installed, Rust
kernels (`rust/wesnoth_core`) compute reach, legal moves, the
observation, the encoding and combat. Combat is bit-exact, checked
strike-for-strike against Wesnoth's `[mp_checkup]` oracle on
strict-sync replays, and replay reconstruction is checked against the
whole corpus command by command (`tools/diff_replay.py`). Self-play runs
through [`tools/az_loop.py`](tools/az_loop.py): each iteration plays N
games through the simulator with both sides on the current policy under
MCTS, then applies one gradient step toward the search's visit counts
and the games' results. [`tools/sim_self_play.py`](tools/sim_self_play.py),
the earlier entry point, trains by search distillation by default
(`--mcts`) and by REINFORCE with a value baseline under `--reinforce`.

**The model** is a transformer over tokenized state (per-unit, per-hex,
recruit-phantom, and global features). One forward produces the action
distribution (unit/recruit/end-turn slots, per-actor type, target hex,
weapon) plus a categorical **C51** value distribution over 51 atoms in
[-1, +1]; its mean is the value estimate and its standard deviation is
exposed as `cliffness`.

**Warm-start** comes from behavior cloning of 1.18.x human replays via
[`tools/supervised_train.py`](tools/supervised_train.py). Every
reference checkpoint so far is such an imitation product
(`configs/reference_player.json` names the current one), and self-play
starts from one (`az_loop --seed-checkpoint`).

**The live-Wesnoth bridge is eval-only.**
[`tools/eval_vs_builtin.py`](tools/eval_vs_builtin.py) (plus
`wesnoth_ai.wesnoth_interface` and the Lua add-on under
[`add-ons/wesnoth_ai/`](add-ons/wesnoth_ai/)) pits the trained model
against Wesnoth's built-in RCA AI. Training no longer touches real
Wesnoth.

## Layout

```
wesnoth_ai/       Core library: GameState/encoder/model/trainer/rewards,
                  combat + visibility, policy adapters (imported as
                  `from wesnoth_ai.X import ...`).
tools/            Scripts + the simulator: imitation, self-play, MCTS,
                  matches, replay extraction/reconstruction,
                  scenario/faction pools; tools/analysis/ reads the
                  measurements.
rust/             The simulator's Rust kernels (wesnoth_core).
tests/            pytest suite: the simulator, encoder, model, trainer
                  and tools on synthetic inputs and committed data; no
                  test launches Wesnoth. conftest.py bootstraps sys.path.
scripts/          Box scripts: rent a GPU box (rent_box.py) and run one
                  measurement or retrain end to end (*_box.sh).
configs/          JSON configs: the reference player, the imitation
                  recipe, the self-play reward, map whitelists, ...
docs/             Plans, pre-registrations, measurements and the
                  Wesnoth rules catalog (index: docs/README.md).
training/         checkpoints/ (a few small tracked .pt files; the rest
                  on Hugging Face) and metrics/ (measurement records).
eval_games/       Records of past matches.
quarantine/       The training mechanisms set aside on 2026-09-03, and
                  their inventory.
signal_profiler/  Instruments that break the training signal down into
                  gradient amplitudes per part of the training system.
benchmarks/       Micro-benchmarks.
main.py           Setup/maintenance CLI (--check-setup, --clean-games).
wesnoth_src/      WML-only copy of the Wesnoth 1.18.7 data tree (runtime
                  subset tracked; CLAUDE.md, "Wesnoth data provenance").
add-ons/wesnoth_ai/   Lua side of the eval bridge.
```

## Where to look next

- **[CLAUDE.md](CLAUDE.md)** — architecture, invariants, and working
  agreements (the authoritative orientation for contributors).
- **[BACKLOG.md](BACKLOG.md)** — the next actions in order, and the
  open findings.
- **[docs/README.md](docs/README.md)** — what each document in docs/ is,
  and the environment variables the code reads.
- **[docs/wesnoth_rules.md](docs/wesnoth_rules.md)** — catalog of
  Wesnoth-engine rules with verbatim source citations.
- **[docs/plan_20260904.md](docs/plan_20260904.md)** — the current
  plan: engineering first, then search over turns;
  **[docs/box_specs.md](docs/box_specs.md)** for box shapes and
  measured throughput.
- **[docs/design_constants.md](docs/design_constants.md)** — where the
  derived magic numbers come from.

## Two things to get right

- **Coordinates:** Wesnoth is 1-indexed (WML, replays, Lua); Python is
  0-indexed internally. The ±1 conversion happens only where Wesnoth
  data enters or leaves Python: the live bridge
  (`wesnoth_ai/state_converter.py`) and the WML readers and writers
  under `tools/` (listed in CLAUDE.md, "Coordinates"). Game logic, the
  encoder and the model work in 0-indexed coordinates only.
- **Version pin:** `unit_stats.json` / `terrain_db.json` are committed
  1.18.4 scrapes. Unit stats might drift between releases and break combat
  parity — never re-scrape from a different Wesnoth version.
