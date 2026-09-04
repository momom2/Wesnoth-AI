#!/usr/bin/env bash
# Generation throughput through the actor pool, server_priors off then
# on (plan step 1.3). Expects /workspace/{wai.tar.gz,.hf_token}.
# Writes /workspace/bench/pool_{off,on}.json and /workspace/bench/DONE.
set -euo pipefail
[ -x /venv/main/bin/python ] && export PATH=/venv/main/bin:$PATH
DPH="${DPH:-0.31}"
ACTORS="${ACTORS:-19}"
GAMES="${GAMES:-19}"
SIMS="${SIMS:-32}"
LEAF_BATCH="${LEAF_BATCH:-16}"
MAX_TURNS="${MAX_TURNS:-30}"
cd /workspace
rm -rf Wesnoth-AI && mkdir -p Wesnoth-AI && tar xzf /workspace/wai.tar.gz -C Wesnoth-AI
cd Wesnoth-AI
export HF_TOKEN="$(cat /workspace/.hf_token)"
python -m pip install -q huggingface_hub psutil 2>&1 | tail -1 || true
python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download
dst = Path("training/checkpoints"); dst.mkdir(parents=True, exist_ok=True)
p = hf_hub_download("momom2/wesnoth-model-checkpoints", "tier-b/a3/seed_imit_tierb_start.pt")
(dst / "seed.pt").write_bytes(Path(p).read_bytes())
print("seed staged", (dst / "seed.pt").stat().st_size)
PY
if [ "${WESNOTH_RUST_BUILD:-1}" = "1" ] && ! python -c "import wesnoth_core" 2>/dev/null; then
    command -v cc >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq gcc) >/dev/null 2>&1 || true
    command -v cargo >/dev/null 2>&1 || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
    export PATH="$HOME/.cargo/bin:$PATH"
    python -m pip install -q maturin >/dev/null 2>&1
    python -m pip install -q rust/wesnoth_core 2>&1 | tail -1
    python -c "import wesnoth_core; print('wesnoth_core built')" || echo "WARNING: wesnoth_core build failed; Python path"
fi
mkdir -p /workspace/bench
echo "cores quota: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)"
for mode in off on; do
    flag=""; [ "$mode" = "on" ] && flag="--server-priors"
    python tools/bench_pool.py --checkpoint training/checkpoints/seed.pt \
        --actors "$ACTORS" --games "$GAMES" --sims "$SIMS" --leaf-batch "$LEAF_BATCH" \
        --max-turns "$MAX_TURNS" --dollars-per-hour "$DPH" $flag \
        --out "/workspace/bench/pool_$mode.json" 2>&1 | tee "/workspace/bench/pool_$mode.log"
done
touch /workspace/bench/DONE
echo POOL_DONE
