# shellcheck shell=bash
# boxlib.sh: what every box run needs, sourced by scripts/*_box.sh
# (docs/box_runbook.md). A script sets its configuration, sources this file
# and calls box_init:
#
#     set -uo pipefail
#     WORKDIR=/workspace
#     BOX_OUT=$WORKDIR/myrun               # the records: every file here goes to HF
#     export HF_DIR=tier-b/myrun_20260926  # where on the model host
#     BOX_MAX_H=9                          # the dead-man's switch, in hours
#     STAGE="${STAGE:-}"                   # the code stage (tools/stage_code.py)
#     . "${BOX_LIB:-$WORKDIR/box}/boxlib.sh" || exit 1
#     box_init
#
# From box_init on, every exit, clean or not, goes through box_finish: the
# reason goes to status.txt (and to FAILED when the exit code is not 0),
# ALL_DONE is written, the records go to HF with ALL_DONE last, and the
# instance is stopped through box_stop.py. The dead-man's switch that
# box_init starts, a process of its own, does the same after BOX_MAX_H hours
# whatever the script is doing. Every step goes through box_bounded, which
# cuts it at its deadline. The library expects `set -uo pipefail`, not
# `set -e`, and the script's working directory is its own business: the
# library uses absolute paths.
#
# Records (in BOX_OUT): status.txt (one line per finish), stages.txt (one
# line per entry), walls.txt (one line per step), upload.log, stop.log and
# stop.jsonl, deadman.log, watchdog.log, restore.log, staging.log, build.log.
# Machine-local markers (the wheel built, the tests passed) live in
# BOX_STATE, which never goes to HF: a new machine has neither.

# ---- configuration: set before sourcing to change -----------------------
BOX_LIB=${BOX_LIB:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
WORKDIR=${WORKDIR:-/workspace}
BOX_REPO=${BOX_REPO:-$WORKDIR/Wesnoth-AI}      # the code stage, unpacked
BOX_STATE=${BOX_STATE:-$WORKDIR/.box_state}    # machine-local markers
BOX_ROUND_MIN=${BOX_ROUND_MIN:-30}             # between two periodic upload rounds
BOX_UPLOAD_BUDGET_MIN=${BOX_UPLOAD_BUDGET_MIN:-25}
BOX_FINAL_BUDGET_MIN=${BOX_FINAL_BUDGET_MIN:-30}
BOX_FINAL_LOCK_WAIT_MIN=${BOX_FINAL_LOCK_WAIT_MIN:-5}  # then the final round goes alongside
BOX_KILL_GRACE=${BOX_KILL_GRACE:-1m}           # a step's KILL follows its TERM this much later
BOX_WATCH_POLL_S=${BOX_WATCH_POLL_S:-20}       # how often a watched step is looked at
BOX_STOP_ATTEMPTS=${BOX_STOP_ATTEMPTS:-20}
BOX_STOP_INTERVAL_S=${BOX_STOP_INTERVAL_S:-30}

BOX_FINISHED=0
BOX_MONITOR_PID=""
BOX_STEP_PID=""
BOX_DEADMAN_PID=""
BOX_ASYNC_PIDS=()
BOX_RC=0                         # the last bounded step's exit code (124: cut at its deadline)
BOX_WHY=""                       # ...and what ended it: ok, cut, stalled, until, failed

# ---- small helpers ------------------------------------------------------
box_now() { date +%s; }
box_stamp() { date -u +%FT%TZ; }
box_log() {                      # box_log FILE TEXT: a stamped line in BOX_OUT/FILE
    printf '%s %s\n' "$(box_stamp)" "$2" >> "$BOX_OUT/$1"
}
box_seconds() {                  # box_seconds NUMBER UNIT_SECONDS: whole seconds, at least 1
    [[ $1 =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "box_seconds: not a number: $1" >&2; return 1; }
    awk -v n="$1" -v u="$2" 'BEGIN { s = int(n * u + 0.5); print (s < 1 ? 1 : s) }'
}
box_size() {                     # box_size FILE: its size in bytes, 0 when absent
    stat -c %s "$1" 2>/dev/null || echo 0
}
box_descendants() {              # box_descendants PID: every descendant of PID
    local child
    command -v pgrep >/dev/null 2>&1 || return 0
    for child in $(pgrep -P "$1" 2>/dev/null); do
        echo "$child"
        box_descendants "$child"
    done
}
box_kill_tree() {                # box_kill_tree PID: TERM to PID and to every descendant
    [ -n "${1:-}" ] || return 0
    local pids
    pids="$1 $(box_descendants "$1")"
    # shellcheck disable=SC2086 # one word per pid
    kill -TERM $pids 2>/dev/null
    return 0
}
box_instance_id() {              # this Vast instance's id, from the environment or PID 1's
    local id=${CONTAINER_ID:-}
    [ -n "$id" ] || id=$({ tr '\0' '\n' < /proc/1/environ; } 2>/dev/null | sed -n 's/^CONTAINER_ID=//p' | head -1)
    printf '%s' "${id:-unknown}"
}
box_mark() {                     # box_mark PATH [TEXT]: a marker naming the code stage that wrote it
    if mkdir -p "$(dirname "$1")" \
            && printf 'stage=%s time=%s%s\n' "${STAGE:-none}" "$(box_stamp)" "${2:+ $2}" > "$1.tmp" \
            && mv -f "$1.tmp" "$1"; then
        return 0
    fi
    echo "box_mark: cannot write $1" >&2
    return 1
}
box_marked_this_stage() {        # box_marked_this_stage PATH: PATH exists and this STAGE wrote it
    [ -f "$1" ] && [ "$(sed -n 's/^stage=\([^ ]*\).*/\1/p' "$1" | head -1)" = "${STAGE:-none}" ]
}

# ---- entry: lock, traps, dead-man's switch, the previous entry's markers ----
box_take_entry_lock() {          # one entry per records directory; the lock (fd 8) lives as long as the script
    command -v flock >/dev/null 2>&1 || return 0
    exec 8> "$BOX_OUT/tmp/entry.lock" || return 0
    flock -n 8 && return 0
    echo "$(box_stamp) another entry holds $BOX_OUT/tmp/entry.lock; this one leaves it alone" >&2
    return 1
}
box_config_problem() {           # prints what is wrong with the configuration, nothing when it is sound
    [ -n "${BOX_OUT_GIVEN:-}" ] || { echo "BOX_OUT is not set (records went to $BOX_OUT)"; return; }
    [[ $BOX_OUT == /* ]] || { echo "BOX_OUT is not an absolute path: $BOX_OUT"; return; }
    [[ ${HF_DIR:-} =~ ^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$ ]] || { echo "HF_DIR is not a folder path: '${HF_DIR:-}'"; return; }
    box_seconds "${BOX_MAX_H:-}" 3600 >/dev/null 2>&1 || { echo "BOX_MAX_H is not a number of hours: '${BOX_MAX_H:-}'"; return; }
    local f
    for f in box_upload.py box_stop.py box_stage.py; do
        [ -f "$BOX_LIB/$f" ] || { echo "the library in $BOX_LIB lacks $f"; return; }
    done
}
box_init() {                     # the entry's bring-up; after it, every exit finishes through box_finish
    local token problem
    BOX_OUT_GIVEN=${BOX_OUT:-}
    BOX_OUT=${BOX_OUT:-$WORKDIR/box_out}
    mkdir -p "$BOX_OUT/tmp" "$BOX_STATE"
    box_take_entry_lock || exit 1
    trap box_on_exit EXIT
    trap 'exit 143' TERM
    trap 'exit 130' INT
    trap 'exit 129' HUP
    case $- in *x*) set +x; echo "xtrace turned off: the HF token and the instance key are in this shell" ;; esac
    # The token file the onstart wrote, read without echo; the environment's
    # HF_TOKEN when there is none. Exported before the switch starts: the
    # switch is a process of its own.
    token=$({ tr -d '\r\n' < "$WORKDIR/.hf_token"; } 2>/dev/null)
    [ -z "$token" ] || HF_TOKEN=$token
    export HF_TOKEN=${HF_TOKEN:-}
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 HF_HUB_DISABLE_XET=1
    [ ! -x /venv/main/bin/python ] || export PATH=/venv/main/bin:$PATH
    export BOX_LIB BOX_OUT BOX_REPO BOX_STATE BOX_MAX_H WORKDIR HF_DIR STAGE
    export BOX_UPLOAD_BUDGET_MIN BOX_FINAL_BUDGET_MIN BOX_FINAL_LOCK_WAIT_MIN BOX_STOP_ATTEMPTS BOX_STOP_INTERVAL_S
    rm -f "$BOX_OUT/ALL_DONE" "$BOX_OUT/FAILED"
    : > "$BOX_OUT/tmp/upload_spec.tsv"
    box_upload_extra "$WORKDIR/onstart_script.log"
    problem=$(box_config_problem)
    [ -z "$problem" ] || box_finish "BAD_CONFIG: $problem" 2
    box_deadman_start
    [ -n "$HF_TOKEN" ] || box_finish "NO_HF_TOKEN ($WORKDIR/.hf_token)" 1
    # The previous entry's end markers go, locally (above) and on HF, so the
    # laptop cannot mistake them for this entry's; its accumulating records
    # come back when this machine lacks them.
    if ! timeout -k 30s 5m python "$BOX_LIB/box_upload.py" --out "$BOX_OUT" --hf-dir "$HF_DIR" \
            --clear ALL_DONE FAILED >> "$BOX_OUT/restore.log" 2>&1 \
            || ! box_restore status.txt stages.txt walls.txt; then
        box_finish "HF_UNREACHABLE at the entry (restore.log)" 1
    fi
    box_log stages.txt "entry: stage=${STAGE:-none} instance=$(box_instance_id) host=$(hostname) switch=${BOX_MAX_H} h"
}
box_on_exit() {                  # an exit that did not go through box_finish (an error, set -u, a signal) finishes too
    local rc=$? code
    [ "$BOX_FINISHED" = 1 ] && return
    code=$rc
    [ "$code" -ne 0 ] || code=1
    box_finish "UNEXPECTED_EXIT rc=$rc$(box_notes_suffix)" "$code"
}
box_notes_suffix() {             # " <the script's box_notes>", when it defines one
    declare -F box_notes >/dev/null && printf ' %s' "$(box_notes)"
}
box_finish() {                   # box_finish REASON [RC]: record why, upload with ALL_DONE last, stop the instance, exit RC
    local reason=$1 rc=${2:-0} pid
    BOX_FINISHED=1
    trap '' TERM INT HUP
    box_kill_tree "$BOX_MONITOR_PID"
    for pid in ${BOX_ASYNC_PIDS[@]+"${BOX_ASYNC_PIDS[@]}"}; do box_kill_tree "$pid"; done
    box_kill_tree "$BOX_STEP_PID"
    printf '%s %s stage=%s\n' "$(box_stamp)" "$reason" "${STAGE:-none}" >> "$BOX_OUT/status.txt"
    [ "$rc" -eq 0 ] || printf '%s %s\n' "$(box_stamp)" "$reason" > "$BOX_OUT/FAILED"
    printf '%s %s\n' "$(box_stamp)" "$reason" > "$BOX_OUT/ALL_DONE"
    echo "box_finish: $reason"
    box_upload --final
    box_stop_instance
    exit "$rc"
}
box_stop_instance() {            # stop this instance; when Vast refuses, the outcome goes to HF and the switch stays armed
    box_log stop.log "stopping the instance"
    if timeout -k 1m 45m python "$BOX_LIB/box_stop.py" --outcome "$BOX_OUT/stop.jsonl" \
            --max-attempts "$BOX_STOP_ATTEMPTS" --interval "$BOX_STOP_INTERVAL_S" >> "$BOX_OUT/stop.log" 2>&1; then
        box_kill_tree "$BOX_DEADMAN_PID"
        return 0
    fi
    box_log stop.log "the instance is NOT stopped; the laptop must stop it (docs/box_runbook.md)"
    box_upload
    return 1
}
box_deadman_start() {            # a process of its own that finishes the entry after BOX_MAX_H hours
    local seconds at runner=(bash)
    seconds=$(box_seconds "$BOX_MAX_H" 3600) || return 1
    at=$(( $(box_now) + seconds ))
    command -v setsid >/dev/null 2>&1 && runner=(setsid bash)
    # shellcheck disable=SC2016 # expanded by the switch's own shell
    "${runner[@]}" -c '. "$BOX_LIB/boxlib.sh" && box_deadman_main "$1"' box_deadman "$at" \
        >> "$BOX_OUT/deadman.log" 2>&1 < /dev/null 8>&- &
    BOX_DEADMAN_PID=$!
    echo "$BOX_DEADMAN_PID" > "$BOX_OUT/tmp/deadman.pid"
    box_log deadman.log "armed: fires $BOX_MAX_H h from now, at $(date -u -d "@$at" +%FT%TZ 2>/dev/null || echo "epoch $at")"
}
box_deadman_main() {             # (the switch's process) sleep until $1, then finish the entry
    local at=$1 left
    while left=$(( at - $(box_now) )); [ "$left" -gt 0 ]; do
        sleep $(( left < 300 ? left : 300 ))
    done
    box_log deadman.log "firing: $BOX_MAX_H h have passed since the entry began"
    box_finish "DEADMAN: $BOX_MAX_H h since the entry began, whatever it was doing" 1
}

# ---- records to HF ---------------------------------------------------------
box_spec_line() {                # box_spec_line FIELD...: one tab-separated line of the upload spec
    local IFS=$'\t'
    printf '%s\n' "$*" >> "$BOX_OUT/tmp/upload_spec.tsv"
}
box_upload_extra() { box_spec_line extra "$1"; }       # box_upload_extra PATH: a file outside BOX_OUT, under its own name
box_upload_dir() { box_spec_line dir "$1" "$2"; }      # box_upload_dir NAME PATH: PATH goes up as NAME.tar.gz
box_upload_hold() { box_spec_line hold "$@"; }         # box_upload_hold NAME DEP...: NAME waits until each DEP has landed
box_upload_skip() { box_spec_line skip "$1"; }         # box_upload_skip NAME: NAME stays on this machine
box_upload() {                   # box_upload [--final]: one upload round; one round at a time
    local args=() budget wait budget_s wait_s
    if [ "${1:-}" = --final ]; then
        args=(--final) budget=$BOX_FINAL_BUDGET_MIN wait=$BOX_FINAL_LOCK_WAIT_MIN
    else
        budget=$BOX_UPLOAD_BUDGET_MIN wait=$BOX_UPLOAD_BUDGET_MIN
    fi
    budget_s=$(box_seconds "$budget" 60) || return 1
    wait_s=$(box_seconds "$wait" 60) || return 1
    (
        if command -v flock >/dev/null 2>&1 && ! flock -w "$wait_s" 9; then
            box_log upload.log "another round held the upload lock for $wait min"
            [ "${#args[@]}" -gt 0 ] || exit 0
        fi
        timeout -k 1m "$(( budget_s + 120 ))s" python "$BOX_LIB/box_upload.py" --out "$BOX_OUT" \
            --hf-dir "$HF_DIR" --spec "$BOX_OUT/tmp/upload_spec.tsv" --budget-s "$budget_s" \
            ${args[@]+"${args[@]}"} >> "$BOX_OUT/upload.log" 2>&1
    ) 9> "$BOX_OUT/tmp/upload.lock" 8>&-
}
box_upload_async() {             # a round in the background (after a milestone); box_finish ends it
    box_upload &
    BOX_ASYNC_PIDS+=("$!")
}
box_monitor_start() {            # every BOX_ROUND_MIN minutes: the script's box_on_round, then an upload round
    local every
    every=$(box_seconds "$BOX_ROUND_MIN" 60) || return 1
    (
        while sleep "$every"; do
            declare -F box_on_round >/dev/null && box_on_round
            box_upload
        done
    ) < /dev/null 8>&- &
    BOX_MONITOR_PID=$!
}
box_restore() {                  # box_restore NAME...: each NAME absent from BOX_OUT comes back from HF; fails when HF cannot answer
    [ "$#" -gt 0 ] || return 0
    timeout -k 1m 30m python "$BOX_LIB/box_upload.py" --out "$BOX_OUT" --hf-dir "$HF_DIR" \
        --restore "$@" >> "$BOX_OUT/restore.log" 2>&1
}

# ---- steps -----------------------------------------------------------------
box_bounded() {                  # box_bounded [--stall FILE MIN] [--until FILE TEXT] NAME CUT_MIN LOG CMD...
    # Runs CMD, its output appended to BOX_OUT/LOG, and cuts it CUT_MIN
    # minutes in (TERM, then KILL BOX_KILL_GRACE later). --stall ends it
    # when FILE has not grown for MIN minutes; --until ends it once TEXT
    # appears in what FILE gains from now on. Sets BOX_RC (124 for a cut)
    # and BOX_WHY (ok, cut, stalled, until, failed), writes a line to
    # walls.txt and returns BOX_RC. The step runs in the background and is
    # waited for, so a signal reaches the script's traps at once.
    local stall_file="" stall_min=0 until_file="" until_text=""
    while [ "$#" -gt 0 ]; do
        case $1 in
            --stall) stall_file=$2 stall_min=$3; shift 3 ;;
            --until) until_file=$2 until_text=$3; shift 3 ;;
            *) break ;;
        esac
    done
    local name=$1 cut_min=$2 log=$3 cut_s t0 watcher="" verdict until_from=0
    shift 3
    cut_s=$(box_seconds "$cut_min" 60) || { BOX_RC=2 BOX_WHY=failed; return 2; }
    verdict="$BOX_OUT/tmp/verdict.$$.$RANDOM"
    rm -f "$verdict"
    [ -z "$until_file" ] || until_from=$(box_size "$until_file")
    t0=$(box_now)
    timeout -k "$BOX_KILL_GRACE" "${cut_s}s" "$@" >> "$BOX_OUT/$log" 2>&1 <&0 8>&- &
    BOX_STEP_PID=$!
    if [ -n "$stall_file$until_file" ]; then
        box_watch_step "$BOX_STEP_PID" "$verdict" "$stall_file" "$stall_min" \
            "$until_file" "$until_text" "$until_from" < /dev/null 8>&- &
        watcher=$!
    fi
    wait "$BOX_STEP_PID"
    BOX_RC=$?
    BOX_STEP_PID=""
    if [ -n "$watcher" ]; then
        kill "$watcher" 2>/dev/null
        wait "$watcher" 2>/dev/null
    fi
    if [ -f "$verdict" ]; then
        BOX_WHY=$(cat "$verdict")
        rm -f "$verdict"
    elif [ "$BOX_RC" -eq 0 ]; then
        BOX_WHY=ok
    elif [ "$BOX_RC" -eq 124 ] || { [ "$BOX_RC" -eq 137 ] && [ $(( $(box_now) - t0 )) -ge "$cut_s" ]; }; then
        BOX_RC=124 BOX_WHY=cut        # 137: the KILL that followed an ignored TERM at the deadline
    else
        BOX_WHY=failed
    fi
    printf '%s %s rc=%s %s %s s\n' "$(box_stamp)" "$name" "$BOX_RC" "$BOX_WHY" "$(( $(box_now) - t0 ))" \
        >> "$BOX_OUT/walls.txt"
    return "$BOX_RC"
}
box_watch_step() {               # (background) end step PID when STALL_FILE stops growing or UNTIL_TEXT appears past byte FROM
    local pid=$1 verdict=$2 stall_file=$3 stall_min=$4 until_file=$5 until_text=$6 from=$7
    local stall_s=0 last_size last_growth size
    [ -z "$stall_file" ] || stall_s=$(box_seconds "$stall_min" 60) || return
    last_size=$(box_size "$stall_file")
    last_growth=$(box_now)
    while kill -0 "$pid" 2>/dev/null; do
        sleep "$BOX_WATCH_POLL_S"
        if [ -n "$until_file" ] && tail -c "+$(( from + 1 ))" "$until_file" 2>/dev/null | grep -qF -- "$until_text"; then
            box_end_step "$pid" "$verdict" until "$until_file shows '$until_text'"
            return
        fi
        [ -n "$stall_file" ] || continue
        size=$(box_size "$stall_file")
        if [ "$size" -ne "$last_size" ]; then
            last_size=$size last_growth=$(box_now)
        elif [ $(( $(box_now) - last_growth )) -ge "$stall_s" ]; then
            box_end_step "$pid" "$verdict" stalled "$stall_file has not grown for $stall_min min"
            return
        fi
    done
}
box_end_step() {                 # box_end_step PID VERDICT_FILE VERDICT TEXT
    printf '%s\n' "$3" > "$2"
    box_log watchdog.log "$4: ending the step"
    kill -TERM "$1" 2>/dev/null   # timeout passes it to the step's whole process group
}

# ---- the code and the machine --------------------------------------------------
box_stage_code() {               # STAGE into BOX_REPO: unpacked into BOX_REPO.tmp, then swapped in whole; staging.log
    [ -n "${STAGE:-}" ] || { box_log staging.log "no STAGE"; return 1; }
    if [ "$(cat "$BOX_REPO/.staged_from" 2>/dev/null)" != "$STAGE" ]; then
        rm -rf "$BOX_REPO.tmp" || return 1
        timeout -k 1m 15m python "$BOX_LIB/box_stage.py" fetch "$STAGE" --into "$BOX_REPO.tmp" \
            >> "$BOX_OUT/staging.log" 2>&1 || return 1
        printf '%s\n' "$STAGE" > "$BOX_REPO.tmp/.staged_from" || return 1
        rm -rf "$BOX_REPO" && mv "$BOX_REPO.tmp" "$BOX_REPO" || return 1
        box_log staging.log "staged $STAGE into $BOX_REPO"
    fi
    box_same_library "$BOX_REPO/scripts/box"
}
box_same_library() {             # box_same_library DIR: the library running here is DIR's, file for file
    local f differ=""
    for f in "$BOX_LIB"/*; do
        [ -f "$f" ] || continue
        cmp -s "$f" "$1/${f##*/}" || differ="$differ ${f##*/}"
    done
    [ -z "$differ" ] && return 0
    box_log staging.log "the library in $BOX_LIB differs from the stage's ($1):$differ"
    return 1
}
box_pip() {                      # box_pip PACKAGE...: pip install, bounded; pip.log
    timeout -k 30s 10m python -m pip install -q "$@" >> "$BOX_OUT/pip.log" 2>&1
}
box_wheel_phase_ok() {           # prints both phases; true when the installed wesnoth_core is the phase lib.rs declares
    local want got
    want=$(sed -n 's/.*"__phase__"[^0-9]*\([0-9][0-9]*\).*/\1/p' "$BOX_REPO/rust/wesnoth_core/src/lib.rs" 2>/dev/null | head -1)
    got=$(cd "$BOX_REPO" && timeout 2m python -c 'import wesnoth_core; print(wesnoth_core.__phase__)' 2>/dev/null)
    echo "wheel phase ${got:-none}, source phase ${want:-none}"
    [ -n "$want" ] && [ "$got" = "$want" ]
}
box_build_wheel() {              # the Rust wheel from BOX_REPO, once per stage on this machine; build.log
    local log="$BOX_OUT/build.log"
    if box_marked_this_stage "$BOX_STATE/WHEEL" && box_wheel_phase_ok >> "$log" 2>&1; then
        return 0
    fi
    # An apt-less host gets conda-forge's compiler (2026-09-19).
    local conda_cc=/opt/conda/bin/x86_64-conda-linux-gnu-cc
    if ! command -v cc >/dev/null 2>&1; then
        if ! { timeout 5m apt-get update -qq && timeout 5m apt-get install -y -qq gcc; } >> "$log" 2>&1; then
            timeout 10m conda install -y -q -c conda-forge c-compiler >> "$log" 2>&1
            [ ! -x "$conda_cc" ] || ln -sf "$conda_cc" /usr/local/bin/cc
        fi
    fi
    { echo "cc: $(command -v cc || echo none)"; cc --version 2>&1 | head -1; } >> "$log"
    if ! command -v cargo >/dev/null 2>&1 && [ ! -x "$HOME/.cargo/bin/cargo" ]; then
        curl --max-time 120 -sSf https://sh.rustup.rs -o /tmp/rustup.sh \
            && timeout 10m sh /tmp/rustup.sh -y --profile minimal >> "$log" 2>&1
    fi
    export PATH="$HOME/.cargo/bin:$PATH"
    timeout 5m python -m pip install -q maturin >> "$log" 2>&1
    # The stage keeps the laptop's modification times, older than any
    # earlier build here: without this cargo reuses a stale build.
    touch "$BOX_REPO"/rust/wesnoth_core/src/*.rs
    (cd "$BOX_REPO" && timeout -k 1m 20m python -m pip install --force-reinstall --no-deps rust/wesnoth_core) \
        >> "$log" 2>&1 || return 1
    box_wheel_phase_ok >> "$log" 2>&1 || return 1
    box_mark "$BOX_STATE/WHEEL"
}
box_facts() {                    # what this box is and runs, for box.txt
    echo "stage ${STAGE:-none}"
    echo "instance $(box_instance_id), host $(hostname)"
    echo "cores(all) $(nproc --all 2>/dev/null), usable $(nproc 2>/dev/null)"
    grep -m1 "model name" /proc/cpuinfo 2>/dev/null
    echo "cpu.max $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo n/a)"
    echo "pids.max $(cat /sys/fs/cgroup/pids.max 2>/dev/null || echo n/a)"
    free -m 2>/dev/null | head -2
    df -h "$WORKDIR" 2>/dev/null | tail -1
    timeout 1m nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1
    timeout 2m python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1
    [ -d "$BOX_REPO" ] || return 0
    (cd "$BOX_REPO" && timeout 2m python -c "import wesnoth_ai; from wesnoth_ai.constants import \
OBSERVATION_EPOCH as E; print('code', wesnoth_ai.__version__, 'epoch', E)" 2>&1)
    box_wheel_phase_ok 2>&1
    (cd "$BOX_REPO" && timeout 2m python tools/kernel_status.py 2>/dev/null | tail -8)
}
