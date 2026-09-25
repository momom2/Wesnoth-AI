#!/usr/bin/env bash
# The turn-ranking value function under the reference player (`obs8` at
# raw:t0+eo-1.5). Bars, predictions and cost:
# docs/turn_value_prereg_20260925.md. Stages:
#   1. validation: the turn-gap procedure on the first 200 holdout
#      boundary positions, flat (3 alternatives, 28 playouts each, horizon
#      reads and playout luck recorded), then its features at once: the
#      first CUDA use of the features path, before the long stage;
#   2. training data: candidate turns and one playout each at up to 15
#      turn starts of each of the 800 recorded games of obs8 against
#      terrain (tools/turn_value_data.py);
#   3. the training features, the fit of the arms, the verdict.
# The two long stages are cut at twice their estimates and a cut is
# final; every other step has its own timeout. Unattended: the onstart of
# scripts/rent_box.py fetches this script from HF and runs it detached.
# Expects /workspace/.hf_token (chmod 600). Records under
# /workspace/turn_value go to HF $HF_DIR every 30 minutes (the data log as
# a copy of its whole games) and at the end. Re-entry, on this machine or
# a new one (files absent here come back from HF), skips finished stages
# and resumes cut ones. Every exit, clean or not, leaves ALL_DONE on HF and
# stops the instance (`stop_self`, with the id and key Vast puts in the
# container). Never `set -x`: the HF token and the instance key are in
# the environment.
set -uo pipefail
[ -x /venv/main/bin/python ] && export PATH=/venv/main/bin:$PATH
WORKDIR=/workspace
OUT=$WORKDIR/turn_value
REPO_DIR=$WORKDIR/Wesnoth-AI
GAMES=$WORKDIR/obs_games
UPLOADER=$WORKDIR/turn_value_upload.py
UPLOAD_LOCK=$WORKDIR/.turn_value_upload.lock
STAGE="${STAGE:-tier-b/staging/stage_20260925w.tar.gz}"
export HF_DIR="${HF_DIR:-tier-b/turn_value_20260925}"
GAMES_TAR="${GAMES_TAR:-tier-b/observation_retrain_20260924/games_obs_e1_vs_terrain.tar.gz}"
CORPUS_TAR=tier-b/replays_dataset_imitation_dedup_20260908.tar.gz
JOBS="${JOBS:-40}"                                  # capped at start by the box's CPU quota and memory
SEED="${SEED:-25}"
DPH="${DPH:-0.48}"
VALIDATION_CUT_MIN="${VALIDATION_CUT_MIN:-264}"     # estimate 2.2 h
DATA_CUT_MIN="${DATA_CUT_MIN:-498}"                 # estimate 4.1 h
DATA_STALL_MIN="${DATA_STALL_MIN:-75}"              # the data log has not grown this long: the generator is stuck
                                                    # (a proxy game alone at the tail can take 30-45 minutes)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1
export HF_HUB_DISABLE_XET=1

FINISHED=0
MONITOR_PID=""
RC=0                                                # the exit code of the last bounded step
VALIDATION_FILE="-" VALIDATION_POSITIONS="-" VALIDATION_RC="-" DATA_GAMES="-" DATA_RC="-"
JOBS_WHY="-"

# ---- small helpers ----------------------------------------------------
now() { date +%s; }
stamp() { date -u +%FT%TZ; }
write_file() {                   # write_file PATH TEXT: the whole file or none of it
    printf '%s\n' "$2" > "$1.tmp" && mv -f "$1.tmp" "$1"
}
stage_wall() {                   # stage_wall NAME RC T0
    echo "$1 rc=$2 $(( $(now) - $3 )) s" | tee -a "$OUT/walls.txt"
}
is_final_rc() {                  # is_final_rc RC: the step finished (0) or ran into its cut (124)
    [ "$1" = 0 ] || [ "$1" = 124 ]
}
cut_rc() {                       # cut_rc RC DEADLINE: 124 also when the KILL that follows an ignored TERM ended the step
    if [ "$1" -eq 137 ] && [ "$(now)" -ge "$2" ]; then echo 124; else echo "$1"; fi
}
bounded() {                      # bounded NAME CUT LOG CMD...: CMD under its timeout, appended to LOG; sets RC
    local name=$1 cut=$2 log=$3 t0
    shift 3
    t0=$(now)
    timeout -k 1m "$cut" "$@" >> "$OUT/$log" 2>&1
    RC=$?
    stage_wall "$name" "$RC" "$t0"
}
step_dir() {                     # step_dir NAME: an empty directory for a step's outputs
    rm -rf "$OUT/tmp/$1" && mkdir -p "$OUT/tmp/$1" && echo "$OUT/tmp/$1"
}
publish() {                      # publish DIR: a finished step's files into $OUT, by rename (never half-written)
    local f
    for f in "$1"/*; do
        [ -e "$f" ] && mv -f "$f" "$OUT/"
    done
}
notes() {                        # the counts and exit codes every finish reason carries
    echo "jobs=$JOBS validation=${VALIDATION_FILE##*/} positions=$VALIDATION_POSITIONS" \
         "validation_rc=$VALIDATION_RC data_games=$DATA_GAMES data_rc=$DATA_RC"
}
log_memory() {
    { stamp; free -m; } >> "$OUT/memory.log" 2>&1
}

# ---- records to HF, and every exit ------------------------------------
write_uploader() {               # the uploader, as a file so that upload can run it under flock and timeout
    cat > "$UPLOADER.tmp" <<'EOF'
"""Upload the run's records to HF $HF_DIR, one line per file in upload.log.

A file unchanged since it last landed is skipped. The data log grows by
appended gzip members, so what goes up is a copy up to its last whole
member. A done-marker goes up only after the files it vouches for, and a
cache built from a stage's output only once that stage is done, so a
restore onto a new machine never pairs a marker or a cache with an older
file. With --final, ALL_DONE goes up last.
"""
import argparse
import glob
import json
import os
import sys
import tempfile
import threading
import time

from huggingface_hub import HfApi

REPO = "momom2/wesnoth-model-checkpoints"
ATTEMPT_SECS = 300
MARKERS = {"VALIDATION_DONE": ("validation.json", "validation.partial.json"),
           "DATA_DONE": ("data.jsonl.gz",)}
BUILT_FROM = {"validation.pt": "VALIDATION_DONE", "train.pt": "DATA_DONE", "arms.pt": "DATA_DONE"}
NOT_LISTED = {"ALL_DONE", "upload.log", "data.jsonl.gz", *MARKERS}

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--code", required=True, help="The staged repository (tools.game_record).")
ap.add_argument("--extra", action="append", default=[], help="Another file, sent under its own name.")
ap.add_argument("--final", action="store_true", help="Send ALL_DONE last.")
ARGS = ap.parse_args()
OUT = ARGS.out
HF_DIR = os.environ["HF_DIR"]
API = HfApi(token=os.environ["HF_TOKEN"])
STATE = os.path.join(OUT, "tmp", "uploaded.json")


def log(line):
    with open(os.path.join(OUT, "upload.log"), "a", encoding="utf-8") as fh:
        fh.write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + line + "\n")


def attempt(path, name):
    """None when the file landed within ATTEMPT_SECS, else the error's type name."""
    outcome = {}

    def run():
        try:
            API.upload_file(path_or_fileobj=path, path_in_repo=f"{HF_DIR}/{name}", repo_id=REPO)
            outcome["error"] = None
        except Exception as e:  # noqa: BLE001 - reported by type name only
            outcome["error"] = type(e).__name__

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(ATTEMPT_SECS)
    return outcome.get("error", "Timeout")


def send(path, name):
    """Three attempts and one line in upload.log; True when the file landed."""
    size = os.path.getsize(path)
    error = None
    for i in range(3):
        error = attempt(path, name)
        if error is None:
            log(f"{name} {size} ok")
            return True
        if i < 2:
            time.sleep(20)
    log(f"{name} {size} failed {error}")
    return False


def load_landed():
    """name -> the key of the version on HF (size and mtime; the data log's end)."""
    try:
        with open(STATE, encoding="utf-8") as fh:
            state = json.load(fh)
    except (OSError, ValueError):
        return {}
    return state["files"] if state.get("hf_dir") == HF_DIR else {}


def save_landed(landed):
    with open(STATE + ".tmp", "w", encoding="utf-8") as fh:
        json.dump({"hf_dir": HF_DIR, "files": landed}, fh)
    os.replace(STATE + ".tmp", STATE)


def records():
    """(path, name, key) of the files to send, smallest first: every file in
    OUT, the dot-prefixed inference server logs and stats included, then
    --extra."""
    items = []
    for name in os.listdir(OUT):
        path = os.path.join(OUT, name)
        if not os.path.isfile(path) or name in NOT_LISTED or name.endswith(".tmp"):
            continue
        if name in BUILT_FROM and not os.path.exists(os.path.join(OUT, BUILT_FROM[name])):
            continue                  # built from a partial file: stays on this machine
        items.append((path, name))
    items += [(p, os.path.basename(p)) for p in ARGS.extra if os.path.isfile(p)]
    keyed = []
    for path, name in items:
        try:
            st = os.stat(path)
        except OSError:               # removed since the listing
            continue
        keyed.append((path, name, [st.st_size, st.st_mtime_ns]))
    return sorted(keyed, key=lambda item: item[2][0])


def data_snapshot():
    """(copy, end): the data log up to its last whole gzip member, or None."""
    src = os.path.join(OUT, "data.jsonl.gz")
    if not os.path.exists(src):
        return None
    sys.path.insert(0, ARGS.code)
    from tools.game_record import complete_members_end
    end = complete_members_end(src)
    if end == 0:
        return None
    fd, copy = tempfile.mkstemp(prefix="data_snapshot_", suffix=".jsonl.gz",
                                dir=os.path.join(OUT, "tmp"))
    with os.fdopen(fd, "wb") as dst, open(src, "rb") as fh:
        left = end
        while left > 0:
            chunk = fh.read(min(left, 1 << 20))
            if not chunk:
                break
            dst.write(chunk)
            left -= len(chunk)
    return copy, end


def remove_stale_snapshots():
    """Copies left by a killed upload (an upload lasts 21 minutes at most)."""
    for path in glob.glob(os.path.join(OUT, "tmp", "data_snapshot_*")):
        try:
            if time.time() - os.path.getmtime(path) > 3600:
                os.remove(path)
        except OSError:
            pass


def main():
    os.makedirs(os.path.join(OUT, "tmp"), exist_ok=True)
    remove_stale_snapshots()
    landed = load_landed()
    current = {}
    unchanged = 0

    def send_if_changed(path, name, key):
        nonlocal unchanged
        current[name] = key
        if landed.get(name) == key:
            unchanged += 1
        elif send(path, name):
            landed[name] = key
            save_landed(landed)

    def vouched(inputs):
        """Every input present here has landed in its current version."""
        return all(name in current and landed.get(name) == current[name]
                   for name in inputs if os.path.exists(os.path.join(OUT, name)))

    for path, name, key in records():
        send_if_changed(path, name, key)
    try:
        snapshot = data_snapshot()
    except Exception as e:  # noqa: BLE001 - reported by type name only
        log(f"data.jsonl.gz snapshot failed {type(e).__name__}")
        snapshot = None
    if snapshot is not None:
        copy, end = snapshot
        send_if_changed(copy, "data.jsonl.gz", [end])
        os.remove(copy)
    for marker, inputs in MARKERS.items():
        path = os.path.join(OUT, marker)
        if not os.path.exists(path):
            continue
        if vouched(inputs):
            st = os.stat(path)
            send_if_changed(path, marker, [st.st_size, st.st_mtime_ns])
        else:
            log(f"{marker} held back: the files it vouches for have not all landed")
    log(f"{'final' if ARGS.final else 'escrow'} round: {unchanged} files unchanged")
    send(os.path.join(OUT, "upload.log"), "upload.log")
    if ARGS.final and os.path.exists(os.path.join(OUT, "ALL_DONE")):
        send(os.path.join(OUT, "ALL_DONE"), "ALL_DONE")


main()
sys.stdout.flush()
os._exit(0)                           # an upload thread still hung on a socket must not hold the exit
EOF
    mv -f "$UPLOADER.tmp" "$UPLOADER"
}
upload() {                       # upload [--final]: the records to HF; one uploader at a time
    (
        # An upload lasts 21 minutes at most, so a 22-minute wait outlasts
        # any holder; a final upload that still cannot lock goes anyway.
        if command -v flock >/dev/null 2>&1 && ! flock -w 1320 9; then
            echo "$(stamp) upload lock busy for 22 minutes" >> "$OUT/upload.log"
            [ "${1:-}" = --final ] || exit 0
        fi
        timeout -k 1m 20m python "$UPLOADER" --out "$OUT" --code "$REPO_DIR" \
            --extra "$WORKDIR/onstart_script.log" "$@" >> "$OUT/upload.log" 2>&1
    ) 9> "$UPLOAD_LOCK"
}
monitor() {                      # every 30 minutes: the memory to memory.log, the records to HF
    while sleep 1800; do
        log_memory
        upload
    done
}
instance_var() {                 # instance_var NAME: from our environment, else from PID 1's
    local value="${!1:-}"
    [ -n "$value" ] || value=$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n "s/^$1=//p" | head -1)
    printf '%s' "$value"
}
stop_self() {                    # the final upload, then stop this instance: its GPU stops billing, its disk stays
    local id key
    id=$(instance_var CONTAINER_ID)
    key=$(instance_var CONTAINER_API_KEY)
    if [ -z "$id" ] || [ -z "$key" ]; then
        echo "stop_self $(stamp): no instance id or key in the environment; the laptop watcher stops the box" >> "$OUT/stop.log"
        upload --final
        return
    fi
    echo "stop_self $(stamp): stopping instance $id" >> "$OUT/stop.log"
    upload --final
    # The key travels in the environment and in the Authorization header.
    # Only status codes and exception type names are printed, and stderr
    # is dropped: a message or a traceback can carry the URL, which holds
    # the key in the query form.
    INSTANCE_KEY="$key" timeout -k 30s 5m python - "$id" >> "$OUT/stop.log" 2>/dev/null <<'EOF'
import os
import sys
import time

try:
    import requests
except ImportError:
    print("stop: requests is not installed", flush=True)
    sys.exit(1)

URL = f"https://console.vast.ai/api/v0/instances/{sys.argv[1]}/"
KEY = os.environ["INSTANCE_KEY"]
AUTH = {"bearer": {"headers": {"Authorization": f"Bearer {KEY}"}},
        "query": {"params": {"api_key": KEY}}}

form = "bearer"
for attempt in range(1, 4):
    try:
        status = requests.put(URL, json={"state": "stopped"}, timeout=60, **AUTH[form]).status_code
    except Exception as e:  # noqa: BLE001 - the type name only
        print(f"stop attempt {attempt} ({form}): {type(e).__name__}", flush=True)
        time.sleep(20)
        continue
    print(f"stop attempt {attempt} ({form}): HTTP {status}", flush=True)
    if 200 <= status < 300:
        sys.exit(0)
    # Refused (Vast answers 404, not 401, to a key it does not accept):
    # the key in the query, as the vastai SDK also sends it, tried once.
    if 400 <= status < 500 and status != 429:
        if form == "query":
            sys.exit(1)
        form = "query"
        continue
    time.sleep(20)
sys.exit(1)
EOF
}
finish() {                       # finish REASON [RC]: every exit records why, leaves ALL_DONE on HF, stops the box
    FINISHED=1
    trap '' TERM INT                                # nothing interrupts the final upload and the stop
    [ -n "$MONITOR_PID" ] && kill "$MONITOR_PID" 2>/dev/null
    echo "$1 $(stamp)" | tee -a "$OUT/status.txt"
    touch "$OUT/ALL_DONE"
    stop_self
    exit "${2:-0}"
}
on_exit() {                      # an exit that did not go through finish (set -u, a signal) still records and stops
    local rc=$?
    [ "$FINISHED" -eq 1 ] || finish "UNEXPECTED_EXIT rc=$rc $(notes)" "$rc"
}

# ---- bring-up -----------------------------------------------------------
restore() {                      # restore NAME...: files absent here come back from HF (a restart may land on a new machine)
    timeout -k 1m 10m python - "$OUT" "$@" >> "$OUT/restore.log" 2>&1 <<'EOF'
import os
import shutil
import sys
import time

from huggingface_hub import hf_hub_download

ABSENT = ("RemoteEntryNotFoundError", "EntryNotFoundError")   # not on HF: nothing to restore
out, names = sys.argv[1], sys.argv[2:]
failed = []
for name in names:
    dst = os.path.join(out, name)
    if os.path.exists(dst):
        continue
    for attempt in range(3):
        try:
            src = hf_hub_download("momom2/wesnoth-model-checkpoints", f"{os.environ['HF_DIR']}/{name}")
        except Exception as e:  # noqa: BLE001 - reported by type name only
            if type(e).__name__ in ABSENT:
                break
            print(f"restore {name}: attempt {attempt + 1} {type(e).__name__}", flush=True)
            time.sleep(20)
            continue
        shutil.copyfile(src, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
        print(f"restored {name}: {os.path.getsize(dst)} bytes", flush=True)
        break
    else:
        failed.append(name)
# HF unreachable: starting over here would overwrite what HF holds.
sys.exit(1 if failed else 0)
EOF
}
stage_code() {                   # the code tarball, unless this STAGE is in place already
    [ "$(cat "$REPO_DIR/.staged_from" 2>/dev/null)" = "$STAGE" ] && return 0
    rm -rf "$REPO_DIR.tmp" && mkdir -p "$REPO_DIR.tmp" || return 1
    timeout -k 1m 10m python - "$STAGE" "$REPO_DIR.tmp" <<'EOF' || return 1
import sys
import tarfile

from huggingface_hub import hf_hub_download

path = hf_hub_download("momom2/wesnoth-model-checkpoints", sys.argv[1])
with tarfile.open(path, "r:gz") as tf:
    tf.extractall(sys.argv[2])
print("code staged from", sys.argv[1], flush=True)
EOF
    echo "$STAGE" > "$REPO_DIR.tmp/.staged_from"
    rm -rf "$REPO_DIR" && mv "$REPO_DIR.tmp" "$REPO_DIR"
}
build_wheel() {                  # the Rust wheel from this code, its phase checked against the source
    if ! command -v cc >/dev/null 2>&1; then
        { timeout 5m apt-get update -qq && timeout 5m apt-get install -y -qq gcc; } > "$OUT/cc_install.log" 2>&1 \
            || { timeout 10m conda install -y -q -c conda-forge c-compiler >> "$OUT/cc_install.log" 2>&1 \
                 && ln -sf "$(ls /opt/conda/bin/x86_64-conda-linux-gnu-cc 2>/dev/null | head -1)" /usr/local/bin/cc; }
    fi
    { echo "cc: $(command -v cc || echo none)"; cc --version 2>&1 | head -1; } > "$OUT/build.log"
    if ! command -v cargo >/dev/null 2>&1; then
        curl --max-time 120 -sSf https://sh.rustup.rs -o /tmp/rustup.sh \
            && timeout 10m sh /tmp/rustup.sh -y --profile minimal >> "$OUT/build.log" 2>&1
    fi
    timeout 5m python -m pip install -q maturin >> "$OUT/build.log" 2>&1
    touch rust/wesnoth_core/src/*.rs
    timeout -k 1m 20m python -m pip install --force-reinstall --no-deps rust/wesnoth_core >> "$OUT/build.log" 2>&1
    local want
    want=$(grep -oP '__phase__",\s*\K[0-9]+' rust/wesnoth_core/src/lib.rs | head -1)
    timeout 1m python -c "import wesnoth_core, sys; p = wesnoth_core.__phase__; \
print('wheel phase', p, 'source', sys.argv[1]); assert str(p) == sys.argv[1]" "$want" >> "$OUT/build.log" 2>&1
}
stage_corpus() {                 # the replay corpus the holdout positions rebuild from, when missing
    [ -d replays_dataset_imitation ] && return 0
    rm -rf .corpus_tmp && mkdir .corpus_tmp || return 1
    timeout -k 1m 15m python - "$CORPUS_TAR" <<'EOF' || return 1
import pathlib
import sys
import tarfile

from huggingface_hub import hf_hub_download

path = hf_hub_download("momom2/wesnoth-model-checkpoints", sys.argv[1])
with tarfile.open(path, "r:gz") as tf:
    tf.extractall(".corpus_tmp")
corpus = pathlib.Path(".corpus_tmp/replays_dataset_imitation")
assert (corpus / "manifest.jsonl").is_file(), "no manifest.jsonl in the corpus"
print("corpus staged", len(list(corpus.glob("*.json.gz"))), flush=True)
EOF
    mv .corpus_tmp/replays_dataset_imitation replays_dataset_imitation && rm -rf .corpus_tmp
}
games_dir() {                    # the directory holding the recorded games (empty when there is none)
    local first
    first=$(find "$GAMES" -name '*.game.jsonl.gz' -print -quit 2>/dev/null)
    [ -n "$first" ] && dirname "$first"
}
count_games() {                  # count_games DIR: the recorded games directly in DIR, where the tools read them
    [ -n "$1" ] || { echo 0; return; }
    find "$1" -maxdepth 1 -name '*.game.jsonl.gz' | wc -l
}
stage_games() {                  # the 800 recorded games of obs8 against terrain, when missing
    [ "$(count_games "$(games_dir)")" -eq 800 ] && return 0
    rm -rf "$GAMES" "$GAMES.tmp" && mkdir -p "$GAMES.tmp" || return 1
    timeout -k 1m 10m python - "$GAMES_TAR" "$GAMES.tmp" <<'EOF' || return 1
import sys
import tarfile

from huggingface_hub import hf_hub_download

path = hf_hub_download("momom2/wesnoth-model-checkpoints", sys.argv[1])
with tarfile.open(path, "r:gz") as tf:
    tf.extractall(sys.argv[2])
print("games staged from", sys.argv[1], flush=True)
EOF
    mv "$GAMES.tmp" "$GAMES"
}
choose_jobs() {                  # prints "JOBS why": JOBS capped by the CPU quota less 4 and by available memory / 1.5 GB
    timeout 1m python - "$JOBS" <<'EOF'
import math
import os
import sys

sys.path.insert(0, "tools")
import host_resources  # noqa: E402

asked = int(sys.argv[1])
cores = min(host_resources.effective_cores(), len(os.sched_getaffinity(0)))
cpu_cap = math.floor(cores) - 4
available = host_resources.available_mb()
mem_cap = asked if available is None else math.floor(available / 1024 / 1.5)
memory = "unknown" if available is None else f"{available / 1024:.0f} GB"
jobs = max(1, min(asked, cpu_cap, mem_cap))
print(jobs, f"(asked {asked}; {cores:g} cores less 4 = {cpu_cap}; {memory} available / 1.5 GB = {mem_cap})")
EOF
}
box_facts() {                    # what this box is, for the record
    echo "cores(all) $(nproc --all), usable $(nproc)"
    grep -m1 "model name" /proc/cpuinfo
    echo "cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)"
    echo "pids.max $(cat /sys/fs/cgroup/pids.max 2>/dev/null || echo n/a)"
    free -m | head -2
    timeout 1m nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    timeout 1m python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
    timeout 1m python -c "import wesnoth_ai; from wesnoth_ai.constants import OBSERVATION_EPOCH as E; \
print('code', wesnoth_ai.__version__, 'epoch', E)"
    timeout 2m python tools/kernel_status.py 2>/dev/null | tail -8
    echo "stage $STAGE"
    echo "jobs $JOBS_WHY"
}
run_tests() {                    # the tools against the built wheel, once per code stage
    [ "$(cat "$OUT/TESTED" 2>/dev/null)" = "$STAGE" ] && return 0
    : > "$OUT/tests.log"
    bounded tests 30m tests.log python -m pytest tests/test_turn_value.py tests/test_turn_gap.py \
        tests/test_game_record.py tests/test_playout_reads.py -q -p no:cacheprovider -m ""
    tail -3 "$OUT/tests.log"
    [ "$RC" -eq 0 ] || finish "TESTS_FAILED rc=$RC (tests.log)" 1
    write_file "$OUT/TESTED" "$STAGE"
}

# ---- 1. validation --------------------------------------------------------
keep_server_logs() {             # keep_server_logs TAG: a rerun reopens the inference server's files, so the last ones move aside
    local f
    for f in "$OUT/.inference_server_$1.log" "$OUT/.inference_server_$1.json"; do
        [ -f "$f" ] && mv -f "$f" "${f%.*}.$(now).${f##*.}"
    done
}
validation_file() {              # the validation records: the whole file, else the partial one
    if [ -f "$OUT/validation.json" ]; then
        echo "$OUT/validation.json"
    elif [ -f "$OUT/validation.partial.json" ]; then
        echo "$OUT/validation.partial.json"
    fi
}
positions_in() {                 # positions_in FILE: the positions it holds (0 when absent or unreadable)
    local n
    n=$(timeout 2m python -c 'import json, sys; print(len(json.load(open(sys.argv[1]))["positions"]))' \
        "$1" 2>/dev/null)
    [[ "$n" =~ ^[0-9]+$ ]] && echo "$n" || echo 0
}
run_validation() {               # run_validation DEADLINE: one attempt, resuming what earlier ones measured; sets RC
    local t0 rc
    t0=$(now)
    log_memory
    keep_server_logs turn_gap
    timeout -k 2m "$(( $1 - t0 ))s" python tools/turn_gap.py --reference --device cuda --jobs "$JOBS" \
        --shared-inference --no-infer-compile --playout-temperature 0.5 --cap-turns 30 \
        --seed "$SEED" --dollars-per-hour "$DPH" --n-states 200 --alternatives 3 --continue-edits 1 \
        --temperature 1.0 --playouts 28 --horizon-reads 8 --playout-luck --resume \
        --out "$OUT/validation.json" >> "$OUT/validation.log" 2>&1
    rc=$?
    RC=$(cut_rc "$rc" "$1")
    stage_wall validation "$RC" "$t0"
}
validation_stage() {             # a cut is final; a crash that left a partial file retries once
    [ -f "$OUT/validation.json" ] && [ ! -f "$OUT/VALIDATION_DONE" ] && write_file "$OUT/VALIDATION_DONE" "rc=0"
    if [ -f "$OUT/VALIDATION_DONE" ]; then
        VALIDATION_RC=$(sed -n 's/^rc=//p' "$OUT/VALIDATION_DONE")
        return
    fi
    rm -f "$OUT/validation.pt"                      # built from an earlier state of the partial file
    local deadline=$(( $(now) + VALIDATION_CUT_MIN * 60 ))
    run_validation "$deadline"
    if ! is_final_rc "$RC" && [ "$(positions_in "$(validation_file)")" -gt 0 ] \
            && [ $(( deadline - $(now) )) -ge 600 ]; then
        run_validation "$deadline"
    fi
    VALIDATION_RC=$RC
    if is_final_rc "$RC" && [ "$(positions_in "$(validation_file)")" -gt 0 ]; then
        write_file "$OUT/VALIDATION_DONE" "rc=$RC"
    fi
}
validation_summary() {           # the gap summary of the validation file (the whole file also writes validation.md)
    timeout 5m python tools/turn_gap.py --summarize "$VALIDATION_FILE" --dollars-per-hour "$DPH" \
        > "$OUT/validation_summary.md.tmp" 2>> "$OUT/validation.log" \
        && mv -f "$OUT/validation_summary.md.tmp" "$OUT/validation_summary.md"
}
validation_features() {          # the first CUDA use of the features path, before the long stage
    [ -f "$OUT/validation.pt" ] && return 0
    local dir
    dir=$(step_dir validation_features)
    bounded "validation features" 20m features.log python tools/turn_value.py features --reference \
        --device cuda --jobs "$JOBS" --positions "$VALIDATION_FILE" --out "$dir/validation.pt"
    [ "$RC" -eq 0 ] || finish "VALIDATION_FEATURES_FAILED rc=$RC (features.log) $(notes)" 1
    publish "$dir"
}

# ---- 2. training data -------------------------------------------------------
finished_games() {               # the games the data log holds whole (0 when there is no log)
    local n
    [ -f "$OUT/data.jsonl.gz" ] || { echo 0; return; }
    n=$(timeout 5m python - "$OUT/data.jsonl.gz" 2>> "$OUT/data.log" <<'EOF'
import sys

from tools.turn_value_data import read_log

header, positions, done = read_log(sys.argv[1])
print(len(done))
EOF
)
    [[ "$n" =~ ^[0-9]+$ ]] && echo "$n" || echo 0
}
watch_growth() {                 # watch_growth PID FILE: stop PID when FILE has not grown for DATA_STALL_MIN minutes
    local pid=$1 file=$2 last=-1 still=0 size
    while kill -0 "$pid" 2>/dev/null; do
        sleep 60
        size=$(stat -c %s "$file" 2>/dev/null || echo 0)
        if [ "$size" -gt "$last" ]; then
            last=$size
            still=0
        else
            still=$(( still + 1 ))
        fi
        if [ "$still" -ge "$DATA_STALL_MIN" ]; then
            echo "$(stamp) $file has not grown for $DATA_STALL_MIN minutes; stopping the generator" >> "$OUT/watchdog.log"
            kill -TERM "$pid" 2>/dev/null      # timeout passes it on to the generator's whole process group
            return
        fi
    done
}
run_data() {                     # run_data DEADLINE: one attempt, resuming by game; sets RC
    local t0 pid watchdog rc
    t0=$(now)
    log_memory
    keep_server_logs turn_value
    timeout -k 2m "$(( $1 - t0 ))s" python tools/turn_value_data.py --reference --games-dir "$GAMES_DIR" \
        --device cuda --jobs "$JOBS" --shared-inference --no-infer-compile --seed "$SEED" \
        --horizon-reads 8 --playout-luck --proxy-playouts 2 \
        --out "$OUT/data.jsonl.gz" >> "$OUT/data.log" 2>&1 &
    pid=$!
    watch_growth "$pid" "$OUT/data.jsonl.gz" &
    watchdog=$!
    wait "$pid"
    rc=$?
    kill "$watchdog" 2>/dev/null
    RC=$(cut_rc "$rc" "$1")
    stage_wall data "$RC" "$t0"
}
needs_rerun() {                  # needs_rerun RC GAMES: a crash or a stall, or a clean exit that left games unmeasured
    is_final_rc "$1" || return 0
    [ "$1" -eq 0 ] && [ "$2" -lt 800 ]
}
data_stage() {                   # a cut is final; a crash, a stall or a short log reruns once, resuming
    if [ -f "$OUT/DATA_DONE" ]; then
        DATA_RC=$(sed -n 's/^rc=//p' "$OUT/DATA_DONE")
        return
    fi
    rm -f "$OUT/train.pt" "$OUT/arms.pt"            # built from an earlier state of the log
    local deadline=$(( $(now) + DATA_CUT_MIN * 60 ))
    run_data "$deadline"
    if needs_rerun "$RC" "$(finished_games)" && [ $(( deadline - $(now) )) -ge 600 ]; then
        run_data "$deadline"
    fi
    DATA_RC=$RC
    is_final_rc "$RC" && write_file "$OUT/DATA_DONE" "rc=$RC"
}

# ---- 3. features, the arms, the verdict ----------------------------------------
train_features() {
    [ -f "$OUT/train.pt" ] && return 0
    rm -f "$OUT/arms.pt"                            # fitted on an earlier cache
    local dir
    dir=$(step_dir train_features)
    bounded "train features" 60m features.log python tools/turn_value.py features --reference \
        --device cuda --jobs "$JOBS" --games-dir "$GAMES_DIR" --positions "$OUT/data.jsonl.gz" \
        --out "$dir/train.pt"
    [ "$RC" -eq 0 ] || finish "TRAIN_FEATURES_FAILED rc=$RC (features.log) $(notes)" 1
    publish "$dir"
}
fit_arms() {
    [ -f "$OUT/arms.pt" ] && return 0
    local dir
    dir=$(step_dir fit)
    bounded fit 60m fit.log python tools/turn_value.py fit --train "$OUT/train.pt" --out-dir "$dir"
    [ "$RC" -eq 0 ] || finish "FIT_FAILED rc=$RC (fit.log) $(notes)" 1
    publish "$dir"
}
evaluate() {                     # always rerun: cheap, and never older than the heads and caches it reads
    rm -f "$OUT/verdict.json" "$OUT/verdict.md"
    local dir
    dir=$(step_dir evaluate)
    bounded evaluate 30m verdict.log python tools/turn_value.py evaluate --heads "$OUT" \
        --validation "$VALIDATION_FILE=$OUT/validation.pt" --train "$OUT/train.pt" \
        --out "$dir/verdict.json"
    [ "$RC" -eq 0 ] || finish "EVALUATE_FAILED rc=$RC (verdict.log) $(notes)" 1
    publish "$dir"
}

# ==== the run ====================================================================
mkdir -p "$OUT/tmp"
cd "$WORKDIR" || exit 1
write_uploader
trap on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
rm -f "$OUT/ALL_DONE"
HF_TOKEN="$(tr -d '\r\n' < "$WORKDIR/.hf_token" 2>/dev/null)"
export HF_TOKEN
[ -n "$HF_TOKEN" ] || finish "NO_HF_TOKEN (/workspace/.hf_token)" 1
timeout 5m python -m pip install -q huggingface_hub psutil pytest scipy requests 2>&1 \
    | grep -v "WARNING: Running pip" | tail -1 || true

# Bring-up: what this entry needs, whatever an earlier entry left.
restore VALIDATION_DONE DATA_DONE validation.json validation.partial.json data.jsonl.gz \
    validation.pt train.pt arms.pt status.txt walls.txt || finish "RESTORE_FAILED (restore.log)" 1
# An ALL_DONE a previous entry left on HF would tell the laptop watcher this entry is over.
timeout 2m python - >> "$OUT/restore.log" 2>&1 <<'EOF_CLEAR'
import os

from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
name = f"{os.environ['HF_DIR']}/ALL_DONE"
try:
    if api.file_exists("momom2/wesnoth-model-checkpoints", name):
        api.delete_file(name, repo_id="momom2/wesnoth-model-checkpoints")
        print("removed the previous entry's ALL_DONE from HF", flush=True)
except Exception as e:  # noqa: BLE001 - reported by type name only
    print(f"clearing ALL_DONE on HF: {type(e).__name__}", flush=True)
EOF_CLEAR
stage_code >> "$OUT/staging.log" 2>&1 || finish "CODE_STAGING_FAILED (staging.log)" 1
cd "$REPO_DIR" || finish "CODE_STAGING_FAILED (no $REPO_DIR)" 1
export PATH="$HOME/.cargo/bin:$PATH"
build_wheel || finish "BUILD_FAILED (build.log)" 1
timeout -k 1m 15m python tools/reference_player.py --ensure >> "$OUT/staging.log" 2>&1 \
    || finish "REFERENCE_MISSING (staging.log)" 1
timeout 1m python tools/reference_player.py > "$OUT/reference.json.tmp" && mv -f "$OUT/reference.json.tmp" "$OUT/reference.json"
stage_corpus >> "$OUT/staging.log" 2>&1 || finish "CORPUS_STAGING_FAILED (staging.log)" 1
stage_games >> "$OUT/staging.log" 2>&1 || finish "GAMES_STAGING_FAILED (staging.log)" 1
GAMES_DIR=$(games_dir)
n_games=$(count_games "$GAMES_DIR")
[ "$n_games" -eq 800 ] || finish "GAMES_STAGING_FAILED ($n_games records in ${GAMES_DIR:-no directory})" 1
jobs_line=$(choose_jobs)
[[ "${jobs_line%% *}" =~ ^[0-9]+$ ]] || finish "JOBS_UNKNOWN (host_resources)" 1
JOBS=${jobs_line%% *}
JOBS_WHY=$jobs_line
box_facts > "$OUT/box.txt.tmp" 2>&1
mv -f "$OUT/box.txt.tmp" "$OUT/box.txt"
cat "$OUT/box.txt"
timeout 1m python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || finish "NO_CUDA (box.txt)" 1
run_tests
monitor &
MONITOR_PID=$!

# 1. Validation, then its features at once.
validation_stage
VALIDATION_FILE=$(validation_file)
VALIDATION_POSITIONS=$(positions_in "$VALIDATION_FILE")
[ "$VALIDATION_POSITIONS" -gt 0 ] || finish "VALIDATION_FAILED rc=$VALIDATION_RC (validation.log) $(notes)" 1
validation_summary
validation_features
upload &                                            # the validation records land now, not at the next round

# 2. Training data.
data_stage
DATA_GAMES=$(finished_games)
[ "$DATA_GAMES" -gt 0 ] || finish "DATA_FAILED rc=$DATA_RC (data.log) $(notes)" 1
upload &

# 3. Features, the arms, the verdict.
train_features
fit_arms
evaluate
finish "TURN_VALUE_DONE $(notes)"
