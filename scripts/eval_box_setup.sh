#!/usr/bin/env bash
# Elo-eval box bring-up (docs/box_specs.md "Elo-eval batch box").
# Reusable and idempotent -- the no-hand-assembly rule applies to
# eval boxes too (2026-08-19 amendment). Usage, from /workspace:
#
#   bash Wesnoth-AI/scripts/eval_box_setup.sh \
#       tier-a/campaign_live_20260730.pt=2516k.pt \
#       tier-b/tier_b_l4.pt=2516k-b-294k-l4-430k.pt
#
# Each arg is HF_PATH=LOCAL_NAME; files land in
# Wesnoth-AI/training/checkpoints/. Reads the HF token from
# /workspace/.hf_token if present (public repo works without).
# Ends with a one-game CPU smoke (mcts:8, 10 turns) that must
# produce a game record -- silence is never success.
set -euo pipefail
# vast pytorch images put torch in /venv/main; ssh sessions don't
# always inherit the profile activation.
[ -x /venv/main/bin/python ] && export PATH=/venv/main/bin:$PATH
mkdir -p /workspace
cd /workspace

if [ ! -d Wesnoth-AI ]; then
    git clone --depth 1 https://github.com/momom2/Wesnoth-AI.git
fi
cd Wesnoth-AI
git fetch --depth 1 origin main && git reset --hard origin/main

if [ -f /workspace/.hf_token ]; then
    export HF_TOKEN="$(cat /workspace/.hf_token)"
fi

python - "$@" <<'EOF'
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

dst = Path("training/checkpoints")
dst.mkdir(parents=True, exist_ok=True)
for arg in sys.argv[1:]:
    hf_path, _, local = arg.partition("=")
    if not local:
        sys.exit(f"bad arg (want HF_PATH=LOCAL_NAME): {arg!r}")
    out = dst / local
    p = hf_hub_download("momom2/wesnoth-model-checkpoints", hf_path)
    out.write_bytes(Path(p).read_bytes())
    print(f"fetched {hf_path} -> {out} ({out.stat().st_size//2**20} MB)")
EOF

# The Rust core (masks and enumeration): 7% on the eval path
# (docs/box_specs.md, 2026-09-11), and the production path elsewhere.
if [ "${WESNOTH_RUST_BUILD:-1}" = "1" ] && ! python -c "import wesnoth_core" 2>/dev/null; then
    command -v cc >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq gcc) >/dev/null 2>&1 || true
    command -v cargo >/dev/null 2>&1 || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
    export PATH="$HOME/.cargo/bin:$PATH"
    python -m pip install -q maturin >/dev/null 2>&1
    python -m pip install -q rust/wesnoth_core 2>&1 | tail -1
    python -c "import wesnoth_core; print('wesnoth_core built')" || echo "WARNING: wesnoth_core build failed; Python path"
fi

# Smoke: one cheap self-game through the REAL eval worker. Uses the
# first fetched checkpoint against itself.
first_local=$(printf '%s\n' "$@" | head -1 | cut -d= -f2)
ck="training/checkpoints/${first_local}"
rm -rf /workspace/smoke_games && mkdir -p /workspace/smoke_games
python tools/elo_eval_game.py smokeA "$ck" smokeB "$ck" 1 424242 \
    /workspace/smoke_games --max-turns 10 --mcts-sims 8 \
    --device cpu --log-level INFO
n=$(ls /workspace/smoke_games/*.json 2>/dev/null | wc -l)
if [ "$n" -lt 1 ]; then
    echo "SMOKE FAILED: no game record in /workspace/smoke_games" >&2
    exit 1
fi
echo "EVAL_BOX_READY smoke_records=$n"
