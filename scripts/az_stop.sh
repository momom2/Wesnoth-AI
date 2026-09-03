#!/bin/bash
# Stop the minimal loop cleanly on the box WITHOUT stopping the box:
# launcher first (so it cannot relaunch), then the loop, then its
# daemons, then any actor processes orphaned by the kill.
# Usage (on the box): bash /workspace/wai/scripts/az_stop.sh
set -u
WORKDIR="${WORKDIR:-/workspace}"

for p in $(pgrep -f 'scripts/az_launc[h]'); do kill "$p" 2>/dev/null; done
pkill -f 'stall_watchdo[g]' 2>/dev/null
pkill -f 'tools/az_loo[p]' 2>/dev/null
for i in $(seq 1 60); do
    pgrep -f 'tools/az_loo[p]' >/dev/null || break
    sleep 1
done
pkill -f 'hf_upload_loo[p]' 2>/dev/null
pkill -f 'probe_escrow_loo[p]' 2>/dev/null
sleep 2
n=0
for p in $(pgrep -f 'multiprocessing.spaw[n]'); do
    if [ "$(ps -o ppid= -p "$p" | tr -d ' ')" = "1" ]; then
        kill "$p" 2>/dev/null && n=$((n + 1))
    fi
done
echo "[az_stop] orphaned actors killed: $n"
left=$(pgrep -fa 'az_launc[h]|tools/az_loo[p]|stall_watchdo[g]|hf_upload_loo[p]|probe_escrow_loo[p]' | wc -l)
echo "[az_stop] remaining loop processes: $left"
ls "$WORKDIR"/ABORTED_* 2>/dev/null
