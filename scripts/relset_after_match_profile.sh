#!/usr/bin/env bash
# After the relset arm's matches (ALL_DONE), on the then-idle GPU:
# time the imitation trainer before and after the batched flow
# (2026-09-11, wesnoth_ai/imitation_loss.py), then profile the new
# one. /workspace/Wesnoth-AI is the arm's checkout (the old trainer);
# /workspace/Wesnoth-AI-new links to it and carries the patched
# files (scripts/ship_new_trainer.sh). Takes the HF token from the
# running arm script's environment.
OUT=/workspace/relset
OLD=$(pgrep -of "seed2_relset_box")
HF_TOKEN=$(python - "$OLD" <<'EOF'
import sys
for item in open(f"/proc/{sys.argv[1]}/environ", "rb").read().split(b"\0"):
    if item.startswith(b"HF_TOKEN="):
        print(item[len(b"HF_TOKEN="):].decode())
EOF
)
export HF_TOKEN
echo "profile watcher up $(date -u +%H:%M), token chars ${#HF_TOKEN}" >> "$OUT/switch.log"
until [ -f "$OUT/ALL_DONE" ]; do sleep 60; done
sleep 30
echo "trainer timings start $(date -u +%H:%M)" >> "$OUT/switch.log"
MODE=time CODE=/workspace/Wesnoth-AI OUT=/workspace/trainprof_old HF_DIR=tier-b/train_profile_20260911b/old \
    bash /workspace/train_profile_box.sh > /workspace/trainprof_old.log 2>&1
MODE=time CODE=/workspace/Wesnoth-AI-new OUT=/workspace/trainprof_new HF_DIR=tier-b/train_profile_20260911b/new \
    bash /workspace/train_profile_box.sh > /workspace/trainprof_new.log 2>&1
MODE=pyspy CODE=/workspace/Wesnoth-AI-new OUT=/workspace/trainprof_new_pyspy HF_DIR=tier-b/train_profile_20260911b/new_pyspy \
    bash /workspace/train_profile_box.sh > /workspace/trainprof_new_pyspy.log 2>&1
echo "trainer timings done $(date -u +%H:%M)" >> "$OUT/switch.log"
touch /workspace/TRAINER_TIMINGS_DONE
