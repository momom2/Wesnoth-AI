#!/usr/bin/env bash
# Switch the running relset arm to one pass (user order 2026-09-11):
# once the trainer has written arm_epoch0.pt and its holdout eval,
# stop the old script tree and the trainer, then run the patched
# seed2_relset_box.sh (shipped as seed2_relset_box_e1.sh), which skips
# training and goes to the phase eval and the matches against seed2's
# one-pass checkpoint. Takes the HF token from the running script's
# environment (ssh sessions do not carry it).
OUT=/workspace/relset
OLD=$(pgrep -of "seed2_relset_box\.sh")
HF_TOKEN=$(python - "$OLD" <<'EOF'
import sys
for item in open(f"/proc/{sys.argv[1]}/environ", "rb").read().split(b"\0"):
    if item.startswith(b"HF_TOKEN="):
        print(item[len(b"HF_TOKEN="):].decode())
EOF
)
export HF_TOKEN
echo "watcher up $(date -u +%H:%M), old script pid $OLD, token chars ${#HF_TOKEN}" >> "$OUT/switch.log"
until grep -q "EVAL\[epoch0-end\]" "$OUT/train.log" 2>/dev/null; do sleep 20; done
sleep 5
pkill -f "seed2_relset_box\.sh"; sleep 2
for p in $(pgrep -f "^python tools/supervised_train"); do kill $p; done; sleep 20
pgrep -f "^python tools/supervised_train" >/dev/null && pkill -9 -f "^python tools/supervised_train"
sleep 5
echo "switched $(date -u +%H:%M): trainer stopped after one pass" >> "$OUT/switch.log"
cd /workspace && STOP_AFTER_EPOCH=1 bash /workspace/seed2_relset_box_e1.sh >> /workspace/relset_e1.log 2>&1
echo "e1 script exited $(date -u +%H:%M) rc=$?" >> "$OUT/switch.log"
