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

# 3. A 40-game match between two checkpoints in the simulator (no Wesnoth
#    needed; in practice this runs on a rented GPU box, docs/box_specs.md).
python tools/run_elo_batch.py `
    --label-a cand --spec-a training/checkpoints/cand.pt `
    --label-b seed --spec-b training/checkpoints/seed_imit_tierb_start.pt `
    --outdir eval_games/cand_vs_seed --games 40 --mcts-sims 0 `
    --raw-temperature-a 0 --raw-temperature-b 0
python tools/elo_collect.py eval_games/cand_vs_seed --no-catalog
#   Self-play training (tools/sim_self_play.py, tools/az_loop.py) is being
#   rebuilt; see docs/plan_20260904.md.

# 4. Watch a trained model play one game (exports a Wesnoth-loadable .bz2).
python tools/sim_demo_game.py
```

A bare `git clone` can train: the simulator's runtime WML inputs are
committed, so no Wesnoth install is required for self-play. Wesnoth
itself is only needed for the eval bridge (below); `python main.py
--check-setup` verifies that install.

## Overview

**The simulator is the production training path.**
[`tools/wesnoth_sim.py`](tools/wesnoth_sim.py) is a pure-Python
reimplementation of Wesnoth 1.18.4's game logic — ~1000× faster than
driving Wesnoth as a subprocess, and bit-exact for combat (verified
strike-for-strike against Wesnoth's `[mp_checkup]` oracle on strict-sync
replays). [`tools/sim_self_play.py`](tools/sim_self_play.py) drives N
games per iteration through it with both sides on the same policy, then
applies one gradient update — REINFORCE + value baseline by default, or
AlphaZero-style soft-target distillation with `--mcts`.

**The model** is a transformer over tokenized state (per-unit, per-hex,
recruit-phantom, and global features). One forward produces the action
distribution (unit/recruit/end-turn slots, per-actor type, target hex,
weapon) plus a categorical **C51** value distribution over 51 atoms in
[-1, +1]; its mean is the value estimate and its standard deviation is
exposed as `cliffness`.

**Warm-start** comes from behavior cloning of 1.18.x human replays via
[`tools/supervised_train.py`](tools/supervised_train.py), producing the
`supervised*.pt` checkpoints that self-play resumes from.

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
wesnoth_src/      1.18.4-pinned Wesnoth WML (runtime inputs tracked).
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

- **Coordinates:** Wesnoth is 1-indexed (WML/Lua); Python is 0-indexed
  internally. The ±1 conversion lives only in
  `wesnoth_ai/state_converter.py` — don't sprinkle it elsewhere.
- **Version pin:** `unit_stats.json` / `terrain_db.json` are committed
  1.18.4 scrapes. Unit stats might drift between releases and break combat
  parity — never re-scrape from a different Wesnoth version.
