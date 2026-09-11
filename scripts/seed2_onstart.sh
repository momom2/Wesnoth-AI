#!/usr/bin/env bash
# Unattended bring-up for the seed2 match (2026-09-11: Vast boxes came
# up without a reachable sshd all morning, so the box stages itself
# from HF and reports through HF). Fetched and started by the
# instance's onstart command with HF_TOKEN in the environment:
#   fetch the code tarball and the match script, start a progress
#   uploader (every 10 minutes: game count, log tails), run the match.
set -uo pipefail
cd /workspace
export HF_TOKEN="$(tr -d '\r\n' < /workspace/.hf_token)"
export HF_HUB_DISABLE_XET=1
python - <<'EOF'
import shutil
from huggingface_hub import hf_hub_download
for remote, local in (("tier-b/staging/stage_20260911.tar.gz", "/workspace/stage.tar.gz"),
                      ("tier-b/staging/seed2_match_box.sh", "/workspace/seed2_match_box.sh")):
    shutil.copyfile(hf_hub_download("momom2/wesnoth-model-checkpoints", remote), local)
    print("fetched", local, flush=True)
EOF
progress() {
    local n
    n=$(ls /workspace/match/seed2_vs_seed/game_*.json 2>/dev/null | wc -l)
    { date -u; echo "games=$n"; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null;
      echo "--- match log"; tail -6 /workspace/match/seed2_vs_seed.log 2>/dev/null;
      echo "--- run log"; tail -4 /workspace/match.log 2>/dev/null; } > /workspace/progress.txt
    python - <<'EOF'
import os
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).upload_file(
    path_or_fileobj="/workspace/progress.txt", path_in_repo="tier-b/seed2_vs_seed_20260911/progress.txt",
    repo_id="momom2/wesnoth-model-checkpoints")
EOF
}
( while [ ! -f /workspace/match/ALL_DONE ]; do sleep 600; progress; done; progress ) &
bash /workspace/seed2_match_box.sh > /workspace/match.log 2>&1
progress
