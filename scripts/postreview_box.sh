#!/usr/bin/env bash
# One rental, three things that are owed (2026-09-13, after the
# adversarial review of the hide-cover fix).
#
#   1. CERTIFICATION. Today's sim changes touch combat and visibility
#      and none has been through the corpus:
#        - 2p Silverhead Crossing's Tentacle now gets the submerge and
#          the `magical` its scenario grants (tools/scenario_events.py).
#          `magical` is a 70% chance-to-hit floor, so this CHANGES
#          COMBAT on a Ladder map and 351 corpus games.
#        - the start-position label is any text before a space, which
#          moved five terrain-code parsers (tools/terrain_resolver.py).
#      Acceptance is the whole corpus through tools/diff_replay.py plus
#      tools/diff_core.py, exactly as scripts/hide_cover_cert_box.sh
#      does. This is a NO-REGRESSION test, not a proof of the new
#      rules -- see docs/box_specs.md "Hide cover certified after the
#      root fix" for why the distinction matters.
#
#   2. THE SECOND SERVE PROCESS. Built on 2026-09-04 and never
#      measured; 1.3-1.5x expected on generation, which is the biggest
#      named unmeasured win in the project (BACKLOG.md). The inference
#      server is the ceiling on BOTH paths, so this is the one that
#      pays.
#
#   3. PIDS CALIBRATION. `az_loop --actors` is now 0 = auto, clamped by
#      the container's READ pids limit instead of a guess that has cost
#      throughput on every box since a 2026-09-04 run died at 38
#      actors. host_resources.PIDS_PER_ACTOR_ESTIMATE is still a guess
#      until a box reports the real number; phase 3 prints it.
#
# Every phase writes its records as it goes and uploads them, so the
# run can be inspected or cut at any point (standing rule: results on
# the run, never an atomic dump at the end). A phase that fails does
# not stop the others.
#
# Expects /workspace/.hf_token (chmod 600). Box: a 4090 with >=24
# cores does all three; phase 2 alone wants cores, not a GPU.
#
# Build the staging tarball with tools/stage_code.py, which REFUSES a
# payload missing a file the run needs -- a hand-built payload silently
# dropped a new tool earlier today and the run measured nothing:
#
#   python tools/stage_code.py --out /tmp/stage_20260913d.tar.gz \
#       --require scripts/postreview_box.sh tools/diff_replay.py \
#                 tools/diff_core.py tools/bench_pool.py \
#       --upload tier-b/staging/stage_20260913d.tar.gz
set -uo pipefail
[ -x /venv/main/bin/python ] && export PATH=/venv/main/bin:$PATH
OUT=/workspace/postreview
HF_DIR="${HF_DIR:-tier-b/postreview_20260913}"
STAGE="${STAGE:-tier-b/staging/stage_20260913d.tar.gz}"
SHARDS="${SHARDS:-24}"
CORE_REPLAYS="${CORE_REPLAYS:-600}"
GAMES="${GAMES:-48}"
SIMS="${SIMS:-32}"
LEAF_BATCH="${LEAF_BATCH:-16}"
MAX_TURNS="${MAX_TURNS:-30}"
DPH="${DPH:-0.34}"
SKIP_CERT="${SKIP_CERT:-0}"
SKIP_POOL="${SKIP_POOL:-0}"
mkdir -p "$OUT"
cd /workspace
export HF_TOKEN="$(tr -d '\r\n' < /workspace/.hf_token)" HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export HF_DIR
python -m pip install -q huggingface_hub pytest psutil scipy 2>&1 | tail -1 || true

upload() {
    python - <<'PY' 2>/dev/null || true
import os, glob
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
for p in sorted(glob.glob("/workspace/postreview/*")):
    if os.path.isfile(p) and os.path.getsize(p) < 50_000_000:
        api.upload_file(path_or_fileobj=p,
                        path_in_repo=os.environ["HF_DIR"] + "/" + os.path.basename(p),
                        repo_id="momom2/wesnoth-model-checkpoints")
PY
}

# ---- phase 0: code, wheel, box facts --------------------------------
rm -rf Wesnoth-AI && mkdir -p Wesnoth-AI
python - "$STAGE" <<'PY'
import sys, tarfile
from huggingface_hub import hf_hub_download
p = hf_hub_download("momom2/wesnoth-model-checkpoints", sys.argv[1])
with tarfile.open(p, "r:gz") as tf:
    tf.extractall("/workspace/Wesnoth-AI")
print("code staged")
PY
cd /workspace/Wesnoth-AI
command -v cc >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq gcc) >/dev/null 2>&1 || true
command -v cargo >/dev/null 2>&1 || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1
export PATH="$HOME/.cargo/bin:$PATH"
python -m pip install -q maturin >/dev/null 2>&1
touch rust/wesnoth_core/src/*.rs
python -m pip install --force-reinstall --no-deps rust/wesnoth_core > "$OUT/build.log" 2>&1
# Phase 9 carries the rows_from_landable token bounds check. Two tests
# skip below it, so record the number rather than only printing it.
python -c "import wesnoth_core; p = wesnoth_core.__phase__; print('wheel phase', p); \
assert p >= 9, f'wheel is phase {p}; the token bounds check landed in 9'" | tee -a "$OUT/build.log" \
    || echo "BUILD_FAILED or PRE-PHASE-9 (the Python path still certifies; the core half does not)" | tee -a "$OUT/build.log"
{
    echo "cores(nproc)  $(nproc --all)"
    echo "cpu.max       $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)"
    echo "pids.max      $(cat /sys/fs/cgroup/pids.max 2>/dev/null || cat /sys/fs/cgroup/pids/pids.max 2>/dev/null || echo n/a)"
    echo "pids.current  $(cat /sys/fs/cgroup/pids.current 2>/dev/null || cat /sys/fs/cgroup/pids/pids.current 2>/dev/null || echo n/a)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU"
    python - <<'PY'
import sys; sys.path.insert(0, ".")
from tools.host_resources import max_actors, pids_headroom
print("pids headroom", pids_headroom())
print("max_actors(64) ->", max_actors(64))
PY
} > "$OUT/box.txt" 2>&1
cat "$OUT/box.txt"
upload

# ---- phase 1: the corpus, both paths --------------------------------
if [ "$SKIP_CERT" != "1" ]; then
python - <<'PY'
import pathlib, tarfile
from huggingface_hub import hf_hub_download
if not pathlib.Path("replays_dataset_imitation").exists():
    p = hf_hub_download("momom2/wesnoth-model-checkpoints",
                        "tier-b/replays_dataset_imitation_dedup_20260908.tar.gz")
    with tarfile.open(p, "r:gz") as tf:
        tf.extractall(".")
print("corpus", len(list(pathlib.Path("replays_dataset_imitation").glob("*.json.gz"))))
PY
ls replays_dataset_imitation/*.json.gz > "$OUT/files.txt"
rm -f "$OUT"/shard_*
split -n "l/$SHARDS" -d -a 2 "$OUT/files.txt" "$OUT/shard_"
t0=$(date +%s)
for f in "$OUT"/shard_??; do
    ( xargs -a "$f" python tools/diff_replay.py > "$f.log" 2>&1 || true ) &
done
wait
echo "diff_replay wall $(( $(date +%s) - t0 )) s" | tee "$OUT/replay.wall"
python - <<'PY' | tee "$OUT/replay_summary.txt"
import glob, re
tot = clean = div = 0
lines = []
for log in sorted(glob.glob("/workspace/postreview/shard_??.log")):
    for line in open(log, encoding="utf-8", errors="replace"):
        if "replays" in line and ("clean" in line or "diverg" in line):
            lines.append(line.rstrip())
        for pat, key in ((r"(\d+) replays", "tot"), (r"(\d+) clean", "clean"),
                         (r"(\d+) with divergence", "div")):
            mm = re.search(pat, line)
            if mm:
                if key == "tot": tot += int(mm.group(1))
                elif key == "clean": clean += int(mm.group(1))
                else: div += int(mm.group(1))
print(f"diff_replay: {tot} replays, {clean} clean, {div} with divergences")
print(f"shards: {len(lines)}")
for ln in lines:
    print("  " + ln)
print("CLEAN" if tot and tot == clean else "NOT CLEAN -- investigate before trusting today's sim changes")
PY
grep -h "DIVERGENCE\|divergence" "$OUT"/shard_??.log 2>/dev/null | head -25 > "$OUT/replay_divergences.txt" || true
head -25 "$OUT/replay_divergences.txt"
python tools/diff_core.py replays_dataset_imitation --limit "$CORE_REPLAYS" > "$OUT/diff_core.log" 2>&1
grep -v WARNING "$OUT/diff_core.log" | head -6
# The suites, both states of record. Silverhead's fix is the new one.
SUITES="tests/test_effect_ids.py tests/test_start_positions.py tests/test_hide_cover.py \
tests/test_visibility.py tests/test_game_core.py tests/test_combat_rules.py \
tests/test_addon_events.py tests/test_fork_isolation.py tests/test_sim_determinism.py \
tests/test_rust_observe.py tests/test_rust_combat.py"
python -m pytest $SUITES -q -p no:cacheprovider -m "" > "$OUT/tests_python.log" 2>&1
tail -3 "$OUT/tests_python.log"
WESNOTH_RUST_CORE=1 python -m pytest $SUITES -q -p no:cacheprovider -m "" > "$OUT/tests_core.log" 2>&1
tail -3 "$OUT/tests_core.log"
upload
fi

# ---- phase 2: does a second serve process pay? ----------------------
if [ "$SKIP_POOL" != "1" ]; then
python - <<'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download
dst = Path("training/checkpoints"); dst.mkdir(parents=True, exist_ok=True)
p = hf_hub_download("momom2/wesnoth-model-checkpoints",
                    "tier-b/seed2_relset_20260911/arm_epoch0.pt")
(dst / "relset.pt").write_bytes(Path(p).read_bytes())
print("reference player staged", (dst / "relset.pt").stat().st_size)
PY
# One factor: serve processes. Everything else is the committed
# configuration. Actors track the games so the comparison is fair.
for np_ in 1 2; do
    echo "=== serve_processes=$np_ ==="
    python tools/bench_pool.py --checkpoint training/checkpoints/relset.pt \
        --actors "$GAMES" --games "$GAMES" --sims "$SIMS" \
        --leaf-batch "$LEAF_BATCH" --max-turns "$MAX_TURNS" \
        --dollars-per-hour "$DPH" --server-priors \
        --serve-processes "$np_" \
        > "$OUT/pool_p$np_.json" 2> "$OUT/pool_p$np_.log"
    tail -3 "$OUT/pool_p$np_.log"
    python - "$np_" <<'PY'
import json, sys
try:
    d = json.load(open(f"/workspace/postreview/pool_p{sys.argv[1]}.json"))
    print(f"  serve_processes={d.get('serve_processes')} "
          f"leaves/s={d.get('leaves_per_s')} "
          f"saturated={d.get('saturated_leaves_per_s')} "
          f"games/$={d.get('games_per_dollar')}")
except Exception as e:                       # noqa: BLE001 -- record and move on
    print("  unreadable:", e)
PY
    upload
done
python - <<'PY' | tee "$OUT/pool_verdict.txt"
import json
def rd(n):
    try:
        return json.load(open(f"/workspace/postreview/pool_p{n}.json"))
    except Exception:
        return None
a, b = rd(1), rd(2)
if not (a and b):
    print("one arm missing; no verdict")
else:
    for k in ("leaves_per_s", "saturated_leaves_per_s", "games_per_dollar"):
        x, y = a.get(k), b.get(k)
        if x and y:
            print(f"{k:26s} 1 proc {x:10.1f}   2 proc {y:10.1f}   {y / x:.2f}x")
    print("\nExpectation on record was 1.3-1.5x. A second server halves the "
          "mean batch, which is why the EVAL path could not win from one "
          "(docs/box_specs.md); generation posts more leaves per round trip, "
          "so it is a different question. Under 1.1x, drop the idea and say so.")
PY
upload
fi

touch "$OUT/ALL_DONE"
upload
echo POSTREVIEW_DONE
