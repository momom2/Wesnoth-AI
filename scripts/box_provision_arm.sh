#!/bin/bash
# One-shot box provisioning for an arm launch (2026-09-02).
# Run ON the box after uploading to /workspace: the repo tarball,
# datasets.tar.gz, hf_token.txt, vast_api_key.txt, and the probe
# reference seed. Usage:
#   bash box_provision_arm.sh <instance_id> <repo_tarball> \
#        [hf_path_of_calibrated_seed]
# Leaves /workspace/wai ready for `arm_vg_launch.sh all`.
set -e
IID="$1"; TAR="$2"; CAL="${3:-}"
cd /workspace
mkdir -p wai && tar xzf "$TAR" -C wai
cd wai && tar xzf /workspace/datasets.tar.gz
mv -f /workspace/hf_token.txt /workspace/.hf_token && chmod 600 /workspace/.hf_token
mv -f /workspace/vast_api_key.txt /workspace/.vast_api_key && chmod 600 /workspace/.vast_api_key
echo "$IID" > /workspace/.instance_id
mkdir -p training/checkpoints
[ -f /workspace/seed_imit_tierb_start.pt ] && \
    mv -f /workspace/seed_imit_tierb_start.pt training/checkpoints/
pip install -q numpy huggingface_hub psutil pytest requests vastai \
    2>&1 | grep -v WARNING | tail -1 || true
if [ -n "$CAL" ]; then
    python - "$CAL" <<'PYEOF'
import sys, shutil, torch
from huggingface_hub import hf_hub_download
tok = open("/workspace/.hf_token").read().strip()
p = hf_hub_download("momom2/wesnoth-model-checkpoints", sys.argv[1], token=tok)
dst = "training/checkpoints/" + sys.argv[1].split("/")[-1]
shutil.copy(p, dst)
m = torch.load(dst, map_location="cpu", weights_only=False).get("training_meta", {}).get("vg2", {})
print("staged", dst, "| meta lambda", round(m.get("trust_lambda", 0), 2),
      "bias", round(m.get("consist_bias", 0), 3), "sigma2", round(m.get("consist_sigma2", 0), 3))
PYEOF
fi
echo "cores: $(nproc)"
echo PROVISIONED
