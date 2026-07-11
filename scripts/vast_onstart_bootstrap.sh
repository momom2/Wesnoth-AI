#!/usr/bin/env bash
# Thin onstart bootstrap -- THIS is what goes in the Vast template's
# "On-start Script" box (or `vastai create/update instance --onstart`).
#
# Why it exists (2026-07-11): the onstart stored at instance creation
# is frozen -- a 21:45Z interruptible restart re-ran a pre-supervisor
# copy of vast_onstart.sh with stale env, OOM'd once, and stranded
# the box idle. This bootstrap never goes stale: it pulls the repo
# and execs the CURRENT scripts/vast_onstart.sh, so onstart fixes
# ship with a git push instead of an instance rebuild.
set -u
WORKDIR="${DATA_DIRECTORY:-/workspace}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"
exec >> "$WORKDIR/onstart.log" 2>&1
echo "==== bootstrap $(date -u +%FT%TZ) ===="
if [ ! -d Wesnoth-AI ]; then
    git clone --depth 1 https://github.com/momom2/Wesnoth-AI.git \
        || { echo "[bootstrap] FATAL: clone failed"; exit 1; }
fi
cd Wesnoth-AI
git pull --ff-only \
    || echo "[bootstrap] WARN: git pull failed; running existing checkout"
exec bash scripts/vast_onstart.sh
