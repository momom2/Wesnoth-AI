# Wesnoth AI

A reinforcement-learning AI for *Battle for Wesnoth* 1.18.x, warm-started
by behavior cloning of human replays, with self-play training being
rebuilt (status: [CLAUDE.md](CLAUDE.md); plan:
[docs/plan_20260904.md](docs/plan_20260904.md)). Two goals shape the
design:

1. **Competitively strong** against human players.
2. **Readable** — behavioral knobs (rewards, openers, biases) live in
   code/config, not buried in weights, so a modder can flip a behavior
   without retraining.

## Quickstart

```powershell
# 1. Environment (Windows dev box; requirements.txt pins torch-directml).
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Run the test suite (fast tier ~2.5 min).
pytest                     # fast tier — run after every change
pytest -m ""               # full suite (~11 min) — before commits/campaigns

# 3. Watch a trained model play one game (exports a Wesnoth-loadable .bz2).
python tools/sim_demo_game.py
```

Matches run in the simulator (no Wesnoth needed) on a rented GPU box
(docs/box_specs.md). A 40-game match of a checkpoint against the
reference player (`configs/reference_player.json`), both players at
the reference's decode (temperature 0, end_turn logit offset -1.5):

```bash
python tools/reference_player.py --ensure      # fetch the reference checkpoint if missing
python tools/run_elo_batch.py \
    --label-a cand --spec-a training/checkpoints/cand.pt --raw-end-turn-offset-a -1.5 \
    $(python tools/reference_player.py --flags b) \
    --outdir eval_games/cand_vs_ref --games 40 --mcts-sims 0 \
    --raw-temperature-a 0 --raw-temperature-b 0 \
    --device cuda --jobs 10 --persistent-workers --shared-inference
python tools/elo_collect.py eval_games/cand_vs_ref --no-catalog
```

Imitation training is `tools/supervised_train.py`; self-play legs run
through `tools/az_loop.py` (docs/plan_20260904.md).

A bare `git clone` can train: the simulator's runtime WML inputs are
committed, so no Wesnoth install is required for self-play. Wesnoth
itself is only needed for the eval bridge (below); `python main.py
--check-setup` verifies that install.

## Overview

**The simulator is the production training path.**
[`tools/wesnoth_sim.py`](tools/wesnoth_sim.py) is a Python
reimplementation (with optional Rust kernels) of Wesnoth 1.18.4's game
logic — ~1000× faster than driving Wesnoth as a subprocess, and
bit-exact for combat (verified strike-for-strike against Wesnoth's
`[mp_checkup]` oracle on strict-sync replays). Self-play runs through
[`tools/az_loop.py`](tools/az_loop.py): each iteration plays N games
through the simulator with both sides on the current policy under
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
tools/            Scripts + the simulator: self-play, MCTS, eval, replay
                  extraction/reconstruction, scenario/faction pools.
tests/            pytest suite (Python-only synthetic inputs; does NOT
                  launch Wesnoth). conftest.py bootstraps sys.path.
configs/          Reward + weight JSON (reward_selfplay.json, ...).
docs/             Reference docs — see below.
training/         checkpoints/ (tracked .pt files), logs/.
main.py           Setup/maintenance CLI (--check-setup, --clean-games).
wesnoth_src/      WML-only copy of the Wesnoth 1.18.7 data tree (runtime
                  subset tracked; CLAUDE.md, "Wesnoth data provenance").
add-ons/wesnoth_ai/   Lua side of the eval bridge.
```

## Where to look next

- **[CLAUDE.md](CLAUDE.md)** — architecture, invariants, and working
  agreements (the authoritative orientation for contributors).
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
