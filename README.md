# Wesnoth AI

A reinforcement-learning AI for *Battle for Wesnoth* 1.18.x, trained by
self-play and warm-started by behavior cloning of human replays. Two
goals shape the design:

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

# 3. Self-play training in the in-process simulator (no Wesnoth needed).
python tools/sim_self_play.py `
    --checkpoint-in  training/checkpoints/supervised_epoch3.pt `
    --checkpoint-out training/checkpoints/sim_selfplay.pt `
    --iterations 50 --games-per-iter 8 --max-turns 60 `
    --reward-config configs/reward_selfplay.json
#   add --mcts for AlphaZero-style PUCT search instead of REINFORCE: better but much longer.

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
- **[docs/tier_a_runbook.md](docs/tier_a_runbook.md)** — the current
  go-forward training plan; **[docs/running_on_gpu.md](docs/running_on_gpu.md)**
  for GPU/cloud launch flags.
- **[docs/design_constants.md](docs/design_constants.md)** — where the
  derived magic numbers come from.

## Two things to get right

- **Coordinates:** Wesnoth is 1-indexed (WML/Lua); Python is 0-indexed
  internally. The ±1 conversion lives only in
  `wesnoth_ai/state_converter.py` — don't sprinkle it elsewhere.
- **Version pin:** `unit_stats.json` / `terrain_db.json` are committed
  1.18.4 scrapes. Unit stats might drift between releases and break combat
  parity — never re-scrape from a different Wesnoth version.
