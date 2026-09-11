#!/usr/bin/env bash
# seed2 against the seed (user order 2026-09-11): the PURE 800-game
# match that rates the clean imitation seed. raw:t0 on both sides
# (--mcts-sims 0, temperature 0), sides alternated, ladder maps, the
# standard eval procedure of CLAUDE.md. seed2 = HF
# tier-b/clean_seed_20260909/arm_epoch2.pt (docs/checkpoint_naming.md).
#
# ON THE BOX after /workspace/{stage.tar.gz,.hf_token}:
#   bash seed2_match_box.sh
# Results land under /workspace/match/; the fit and logs are uploaded
# to HF tier-b/seed2_vs_seed_20260911/ at the end and the game records
# as one tarball. Re-entry resumes the match from its files.
set -uo pipefail
WORKDIR=/workspace
OUT=$WORKDIR/match
GAMES="${GAMES:-800}"
JOBS="${JOBS:-10}"
export HF_DIR=tier-b/seed2_vs_seed_20260911
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export HF_HUB_DISABLE_XET=1
mkdir -p "$OUT"
cd $WORKDIR
if [ ! -d Wesnoth-AI/tools ]; then
    mkdir -p Wesnoth-AI && tar xzf $WORKDIR/stage.tar.gz -C Wesnoth-AI
fi
cd Wesnoth-AI
export HF_TOKEN="$(tr -d '\r\n' < $WORKDIR/.hf_token)"
python -m pip install -q huggingface_hub psutil 2>&1 | grep -v "WARNING: Running pip" | tail -1 || true

if [ ! -f "$OUT/STAGED" ]; then
python - <<'EOF' || { echo "staging failed" >&2; exit 1; }
import pathlib
from huggingface_hub import HfApi, hf_hub_download
dst = pathlib.Path("training/checkpoints"); dst.mkdir(parents=True, exist_ok=True)
for remote, local in (("tier-b/a3/seed_imit_tierb_start.pt", "seed.pt"),
                      ("tier-b/clean_seed_20260909/arm_epoch2.pt", "seed2.pt")):
    p = hf_hub_download("momom2/wesnoth-model-checkpoints", remote)
    (dst / local).write_bytes(pathlib.Path(p).read_bytes())
    print("staged", local, (dst / local).stat().st_size, flush=True)
# The alias file on HF, so later boxes fetch seed2 by name.
api = HfApi()
api.upload_file(path_or_fileobj=str(dst / "seed2.pt"), path_in_repo="tier-b/seed2.pt",
                repo_id="momom2/wesnoth-model-checkpoints")
print("tier-b/seed2.pt uploaded", flush=True)
EOF
touch "$OUT/STAGED"
fi
{ nproc --all; free -g | head -2; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; } > "$OUT/box.txt" 2>&1

# The Rust core (masks and enumeration; 7% on the eval path, measured
# 2026-09-11 in docs/box_specs.md). bench_box.sh recipe.
if ! python -c "import wesnoth_core" 2>/dev/null; then
    command -v cc >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq gcc) >/dev/null 2>&1 || true
    command -v cargo >/dev/null 2>&1 || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
    export PATH="$HOME/.cargo/bin:$PATH"
    python -m pip install -q maturin >/dev/null 2>&1
    python -m pip install -q rust/wesnoth_core 2>&1 | tail -1
    python -c "import wesnoth_core; print('wesnoth_core built')" || echo "WARNING: wesnoth_core build failed; Python path"
fi

decisive_results() {              # decisive_results OUTDIR -> count of win/loss files
    python - "$1" <<'EOF'
import json, pathlib, sys
n = 0
for p in pathlib.Path(sys.argv[1]).glob("game_*.json"):
    try:
        n += json.loads(p.read_text(encoding="utf-8")).get("outcome_a") in ("win", "loss")
    except Exception:
        pass
print(n)
EOF
}

DIR=$OUT/seed2_vs_seed
if [ ! -f "$DIR.DONE" ]; then
    python tools/run_elo_batch.py --label-a seed2 --spec-a training/checkpoints/seed2.pt \
        --label-b seed_t0 --spec-b training/checkpoints/seed.pt \
        --outdir "$DIR" --games "$GAMES" --seed-base 40000 \
        --mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0 \
        --persistent-workers --shared-inference --no-infer-compile --device cuda --jobs "$JOBS" \
        --time-budget-min 150 2>&1 | grep --line-buffered -v "wesnoth_core is not importable" | tee -a "$DIR.log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then echo "match: run_elo_batch FAILED rc=$rc" >&2; exit 1; fi
    have=$(decisive_results "$DIR")
    if [ "$have" -lt "$GAMES" ]; then
        echo "match SHORT: $have/$GAMES decisive results; re-run to continue" >&2; exit 1
    fi
    python tools/elo_collect.py "$DIR" --no-catalog --save-json "$DIR.fit.json" 2>&1 | tee -a "$DIR.log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ] || [ ! -f "$DIR.fit.json" ]; then echo "match: elo_collect FAILED rc=$rc" >&2; exit 1; fi
    touch "$DIR.DONE"
fi

cd $OUT && tar czf seed2_vs_seed_games.tgz seed2_vs_seed && cd /workspace/Wesnoth-AI
python - <<'EOF'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for name in ("seed2_vs_seed.fit.json", "seed2_vs_seed.log", "seed2_vs_seed_games.tgz", "box.txt"):
    p = "/workspace/match/" + name
    if os.path.exists(p):
        api.upload_file(path_or_fileobj=p, path_in_repo=f"{os.environ['HF_DIR']}/{name}",
                        repo_id="momom2/wesnoth-model-checkpoints")
        print("uploaded", name, flush=True)
EOF
touch "$OUT/ALL_DONE"
echo SEED2_MATCH_DONE
