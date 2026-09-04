#!/usr/bin/env bash
# Baseline pipeline benchmark on a rented GPU box (plan step 1.1).
# Expects in /workspace: wai.tar.gz (repo working tree), bench_dataset.tar.gz
# (tools/bench_pipeline.py --pack-states output), .hf_token.
# Writes /workspace/bench/<label>.json and .md, and /workspace/bench/DONE.
set -euo pipefail
[ -x /venv/main/bin/python ] && export PATH=/venv/main/bin:$PATH
LABEL="${LABEL:-baseline}"
DPH="${DPH:-0.334}"
GAMES="${GAMES:-20}"
JOBS="${JOBS:-10}"
cd /workspace
rm -rf Wesnoth-AI && mkdir -p Wesnoth-AI && tar xzf /workspace/wai.tar.gz -C Wesnoth-AI
rm -rf bench_dataset && mkdir -p bench_dataset && tar xzf /workspace/bench_dataset.tar.gz -C bench_dataset
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
# wesnoth_core (Rust reach + enumeration) is the production path; the
# baseline of 2026-09-04 ran the Python fallback because it was missing.
if [ "${WESNOTH_RUST_BUILD:-1}" = "1" ] && ! python -c "import wesnoth_core" 2>/dev/null; then
    command -v cc >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq gcc) >/dev/null 2>&1 || true
    command -v cargo >/dev/null 2>&1 || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
    export PATH="$HOME/.cargo/bin:$PATH"
    python -m pip install -q maturin >/dev/null 2>&1
    python -m pip install -q rust/wesnoth_core 2>&1 | tail -1
    python -c "import wesnoth_core; print('wesnoth_core built')" || echo "WARNING: wesnoth_core build failed; Python path"
fi
mkdir -p /workspace/bench
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "cores quota: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)"
python tools/bench_pipeline.py --checkpoint training/checkpoints/seed.pt \
    --device cuda --manifest configs/bench_states.json --dataset /workspace/bench_dataset \
    --games "$GAMES" --jobs "$JOBS" --dollars-per-hour "$DPH" --label "$LABEL" \
    --games-outdir /workspace/bench/games --out "/workspace/bench/$LABEL.json" \
    2>&1 | tee "/workspace/bench/$LABEL.log"
tar czf /workspace/bench/games.tar.gz -C /workspace/bench games 2>/dev/null || true
touch /workspace/bench/DONE
echo BENCH_DONE
