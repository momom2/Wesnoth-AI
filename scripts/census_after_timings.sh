#!/usr/bin/env bash
# After the trainer timings (TRAINER_TIMINGS_DONE), run the pair census
# (scripts/pair_census_box.sh). Takes the HF token from the running
# relset script's environment (ssh sessions do not carry it).
OUT=/workspace/relset
PID=$(pgrep -of "seed2_relset_box")
HF_TOKEN=$(python - "$PID" <<'EOF'
import sys
for item in open(f"/proc/{sys.argv[1]}/environ", "rb").read().split(b"\0"):
    if item.startswith(b"HF_TOKEN="):
        print(item[len(b"HF_TOKEN="):].decode())
EOF
)
export HF_TOKEN
echo "census watcher up $(date -u +%H:%M), token chars ${#HF_TOKEN}" >> "$OUT/switch.log"
until [ -f /workspace/TRAINER_TIMINGS_DONE ]; do sleep 60; done
echo "census start $(date -u +%H:%M)" >> "$OUT/switch.log"
bash /workspace/pair_census_box.sh > /workspace/census.log 2>&1
echo "census done $(date -u +%H:%M)" >> "$OUT/switch.log"
