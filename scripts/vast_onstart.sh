#!/usr/bin/env bash
# Vast.ai on-start script for the Tier-a Phase 2 calibration run
# (docs/tier_a_runbook.md). Do NOT paste THIS file into the template:
# use scripts/vast_onstart_bootstrap.sh there instead -- it pulls the
# repo and execs the current copy of this script, so onstart fixes
# ship with a git push (2026-07-11 lesson: a frozen create-time copy
# re-ran stale config after an instance restart and stranded the
# box). This script runs at EVERY instance (re)start, so it encodes
# the first-launch vs preemption-resume distinction that must not be
# fumbled: --reset-decision-step ONLY on the very first launch.
#
# Layout: /workspace persists across interruptible stop/restart
# cycles on Vast (it is the instance's disk; it is lost only if the
# instance is DESTROYED -- download checkpoints before destroying).
set -uo pipefail
# Vast images differ in data dir + where python lives (observed
# 2026-07-02 on vastai/pytorch:cuda-13.0.3-auto: no /workspace at
# onstart time, and python sits in /venv/main which only interactive
# shells activate). Be robust to both.
WORKDIR="${DATA_DIRECTORY:-/workspace}"
export WORKDIR   # read by hf_upload_loop.py and the HF seed block
mkdir -p "$WORKDIR"
exec >> "$WORKDIR/onstart.log" 2>&1
echo "==== onstart $(date -u +%FT%TZ) ===="

cd "$WORKDIR"

# Box-local env overrides (2026-08-11): create-time env is baked and
# unchangeable, so mid-leg config changes (arm switches, knob tweaks)
# go in $WORKDIR/.leg_env -- sourced here and echoed for the audit
# trail. NO SECRETS in this file (it is logged verbatim); the token
# stays in .hf_token.
if [ -f "$WORKDIR/.leg_env" ]; then
    echo "[onstart] sourcing .leg_env:"
    sed 's/^/[onstart]   /' "$WORKDIR/.leg_env"
    . "$WORKDIR/.leg_env"
fi

# Post-create env overrides: container env is frozen at instance
# creation, so knob changes on a LIVE instance go into
# $WORKDIR/env.sh (plain `export VAR=...` lines). Persisted on the
# instance disk -> survives stop/restart cycles and wins over the
# create-time -e values.
if [ -f "$WORKDIR/env.sh" ]; then
    . "$WORKDIR/env.sh"
    echo "[onstart] sourced $WORKDIR/env.sh overrides"
fi

# Resolve python: prefer the image's venv, then conda, then PATH.
if [ -x /venv/main/bin/python ]; then
    export PATH="/venv/main/bin:$PATH"
elif [ -x /opt/conda/bin/python ]; then
    export PATH="/opt/conda/bin:$PATH"
fi
PY="$(command -v python || command -v python3 || true)"
if [ -z "$PY" ]; then
    echo "[onstart] FATAL: no python found on PATH/venv/conda"; exit 1
fi
echo "[onstart] using python: $PY"

# CUDA forward-compat libs (shipped in vastai images for newer-than-
# driver CUDA runtimes) only work on datacenter GPUs; on GeForce
# cards the compat libcuda poisons CUDA init with Error 804 while
# nvidia-smi looks fine (hit 2026-07-12 on an RTX 3090 + driver 550).
# Drop the compat layer and rely on CUDA minor-version compatibility
# against the host driver.
if [ -d /usr/local/cuda/compat ] && nvidia-smi --query-gpu=name \
        --format=csv,noheader 2>/dev/null | grep -qi geforce; then
    mv /usr/local/cuda/compat /usr/local/cuda/compat.disabled \
        2>/dev/null || true
    ldconfig 2>/dev/null || true
    echo "[onstart] disabled CUDA compat libs (GeForce + compat = 804)"
fi

# Hard requirements: CUDA torch + Python >= 3.11 (project floor).
"$PY" - <<'EOF' || { echo "[onstart] FATAL: env check failed"; exit 1; }
import sys, torch
assert sys.version_info >= (3, 11), f"need Python >=3.11, got {sys.version}"
assert torch.cuda.is_available(), "no CUDA device visible"
# is_available() PASSES on a torch build that lacks this GPU's SM arch
# (e.g. a pre-Blackwell wheel on an sm_120 card). The real failure would
# otherwise surface hours in, at the first kernel launch, as "no kernel
# image is available for execution on the device". One tiny real matmul
# NOW is the ground truth -- a rented box must fail in the first seconds,
# not after we have paid for an evening of nothing.
_cap = torch.cuda.get_device_capability(0)
_sm = f"sm_{_cap[0]}{_cap[1]}"
try:
    _x = torch.randn(64, 64, device="cuda")
    _s = float((_x @ _x).sum().item())
    assert _s == _s, "CUDA matmul returned NaN"
except Exception as e:
    raise SystemExit(
        f"[onstart] FATAL: CUDA kernel smoke test failed on "
        f"{torch.cuda.get_device_name(0)} ({_sm}): {e}\n"
        f"torch {torch.__version__} was compiled for "
        f"{torch.cuda.get_arch_list()} -- this build cannot drive this "
        f"GPU. Use a newer image/wheel, or pick an older-arch GPU.")
# A matmul can still SUCCEED via PTX JIT on an arch-mismatched build:
# functional, but the first kernels are slow and perf is unpredictable.
if _sm not in torch.cuda.get_arch_list():
    print(f"[onstart] WARNING: {_sm} not in compiled arch list "
          f"{torch.cuda.get_arch_list()} -- running via PTX JIT fallback; "
          f"expect slow first kernels and possible perf loss.")
print(f"[onstart] python {sys.version.split()[0]}, torch {torch.__version__}, "
      f"gpu {torch.cuda.get_device_name(0)}, "
      f"vcpus reported: {__import__('os').cpu_count()}")
EOF

if [ ! -d Wesnoth-AI ]; then
    git clone --depth 1 https://github.com/momom2/Wesnoth-AI.git || exit 1
fi
cd Wesnoth-AI
# Pick up fixes pushed since the instance was created. --ff-only so a
# locally-dirtied tree (shouldn't happen; checkpoints write to an
# untracked path) fails loudly instead of merging silently.
git pull --ff-only || echo "[onstart] WARN: git pull failed; running existing checkout"

# A tripwire abort (exit 4 = all-draws, 5 = holdout stall) needs a
# human decision -- do NOT auto-relaunch over it.
if ls "$WORKDIR"/ABORTED_* >/dev/null 2>&1; then
    echo "[onstart] ABORTED_* marker present -- NOT relaunching."
    echo "[onstart] Read the tail of $WORKDIR/train.log, diagnose,"
    echo "[onstart] delete the marker, then restart the instance."
    exit 0
fi

# First launch warm-starts from the committed 5M grow WITH the anneal
# reset; any restart after that resumes the campaign checkpoint
# WITHOUT it. The campaign file doubles as the marker (it exists iff
# training has saved at least once; the save is atomic + .bak'd).
# Architecture + campaign identity. Defaults reproduce the Tier-a
# calibration run exactly; a Tier-b campaign overrides all six at
# instance-create time (or via $WORKDIR/env.sh on a live box).
#
# THESE MUST AGREE WITH THE SEED CHECKPOINT'S `arch` -- sim_self_play
# raises on a warm-start arch mismatch, which is the guard that keeps
# a half-changed override from training a differently-shaped net.
#
# CAMPAIGN_FILE is also the run's IDENTITY: it names the local rolling
# checkpoint AND the HF escrow object. `tier_a_campaign.pt` is
# RESERVED for the Tier-a lineage -- a Tier-b run that leaves it at
# the default would roll its own weights forward over that name.
# Defaults = the tier-b HANDOFF leg (2026-08-10 technique-review
# config): 15M arch, seeded from the imitation checkpoint
# (imit_tierb_start.pt = rescued 2368k ckpt, aux heads stripped,
# holdout CE 3.102 = the leg's t0 reference). A tier-a revival must
# override all six.
D_MODEL="${D_MODEL:-384}"
NUM_LAYERS="${NUM_LAYERS:-8}"
NUM_HEADS="${NUM_HEADS:-12}"
D_FF="${D_FF:-1536}"
CAMPAIGN_FILE="${CAMPAIGN_FILE:-tier_b_handoff.pt}"
SEED_CKPT="${SEED_CKPT:-training/checkpoints/imit_tierb_start.pt}"
# HF repo folder for this lineage -- shared with hf_upload_loop.py so
# the seed-fetch and the escrow can never drift apart.
HF_PREFIX="${HF_PREFIX:-tier-b/}"
export HF_PREFIX
# HF path of the first-launch seed (the fetch block below).
SEED_HF_NAME="${SEED_HF_NAME:-${HF_PREFIX}$(basename "$SEED_CKPT")}"
echo "[onstart] arch d_model=$D_MODEL layers=$NUM_LAYERS heads=$NUM_HEADS" \
     "d_ff=$D_FF | campaign=$CAMPAIGN_FILE | seed=$SEED_CKPT"

CAMPAIGN="training/checkpoints/$CAMPAIGN_FILE"

# Seed the campaign from HF Hub on a FRESH instance so a brand-new
# node RESUMES the campaign instead of silently starting over
# (2026-07-05 incident: token scp'd after onstart had already begun a
# fresh --reset-decision-step run). Requires HF_TOKEN in the template
# env (pass at create time: vastai create instance ... --env '-e
# HF_TOKEN=hf_...') or $WORKDIR/.hf_token pre-seeded some other way.
# HF_SEED_FILE selects WHICH repo file seeds the campaign (default:
# the rolling campaign checkpoint). Pass -e HF_SEED_FILE=... at
# create time to start a run from a different escrowed checkpoint,
# e.g. human_value_allgames.pt (the 2026-07-09 human-corpus value
# fine-tune: late-game AUC 0.89 vs the old head's ~0.50). It lands
# AS the local campaign file, so the resume path (no anneal reset,
# decision_step carried) applies and the uploader's tier_a_campaign
# escrow rolls forward from it.
if [ ! -f "$CAMPAIGN" ]; then
    # Fresh campaign: a stale persisted holdout probe (from a
    # previous campaign on this disk) would anchor holdout CE to the
    # wrong distribution -- clear it so the new campaign samples its
    # own (sim_self_play persists it as <checkpoint>.holdout).
    rm -f "$CAMPAIGN.holdout"
    if [ -n "${HF_TOKEN:-}" ] || [ -f "$WORKDIR/.hf_token" ]; then
        "$PY" -m pip install --quiet huggingface_hub || true
        HF_SEED_TOKEN="${HF_TOKEN:-}" \
        HF_SEED_FILE="${HF_SEED_FILE:-${HF_PREFIX}$CAMPAIGN_FILE}" \
        HF_SEED_DEST="$CAMPAIGN" \
        "$PY" - <<'EOF' && echo "[onstart] seeded campaign from HF" \
            || echo "[onstart] HF seed unavailable (first campaign?)"
import os, pathlib, shutil, sys
from huggingface_hub import hf_hub_download
tok = os.environ.get("HF_SEED_TOKEN") or pathlib.Path(
    os.environ.get("WORKDIR", "/workspace"), ".hf_token"
).read_text().strip()
fname = os.environ.get("HF_SEED_FILE", "tier-a/tier_a_campaign.pt")
try:
    p = hf_hub_download("momom2/wesnoth-model-checkpoints", fname, token=tok)
except Exception as e:                                  # noqa: BLE001
    print(f"[onstart] hf seed download failed: {e}")
    sys.exit(1)
# Must land on THIS campaign's path: onstart decides first-launch vs
# resume by testing that exact file, so seeding a renamed campaign to
# the old hardcoded path would silently start over from scratch.
dst = pathlib.Path(os.environ.get(
    "HF_SEED_DEST", "training/checkpoints/tier_a_campaign.pt"))
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(p, dst)
print(f"[onstart] seed file: {fname}")
# Also carry the frozen-holdout probe sidecar (escrowed alongside the
# checkpoint). Without it a fresh box RESAMPLES the probe and the
# holdout-CE curve loses cross-box comparability (2026-07-18 concern;
# observed live 2026-08-11 on the on-demand migration). Best-effort:
# a missing sidecar just means the probe resamples, as before.
try:
    hp = hf_hub_download("momom2/wesnoth-model-checkpoints",
                         fname + ".holdout", token=tok)
    shutil.copy2(hp, str(dst) + ".holdout")
    print("[onstart] holdout sidecar carried from escrow")
except Exception as e:                                  # noqa: BLE001
    print(f"[onstart] no holdout sidecar on HF ({e.__class__.__name__}); "
          f"probe will resample")
EOF
    fi
fi

# Stage the human value corpus for midgame starts. Since the
# 2026-07-20 absolute-mix redesign, training ERRORS at startup when
# --midgame-ratio > 0 and the corpus is missing (the 2026-07-15
# incident -- 56 min trained with midgame 0/0 on an unseen warning
# -- can no longer happen silently; it now fails loudly here
# instead). Idempotent: skips when the index is already on disk.
if [ ! -f replays_dataset/value_corpus_index.jsonl ] \
        && { [ -n "${HF_TOKEN:-}" ] || [ -f "$WORKDIR/.hf_token" ]; }; then
    HF_SEED_TOKEN="${HF_TOKEN:-}" \
    "$PY" - <<'EOF' && echo "[onstart] value corpus staged" \
        || echo "[onstart] WARN: corpus staging failed (midgame will be OFF)"
import os, pathlib, sys, tarfile
from huggingface_hub import hf_hub_download
tok = os.environ.get("HF_SEED_TOKEN") or pathlib.Path(
    os.environ.get("WORKDIR", "/workspace"), ".hf_token"
).read_text().strip()
try:
    p = hf_hub_download("momom2/wesnoth-model-checkpoints", "tier-a/value_corpus.tar.gz",
                        token=tok)
except Exception as e:                                  # noqa: BLE001
    print(f"[onstart] corpus download failed: {e}")
    sys.exit(1)
dst = pathlib.Path("replays_dataset")
dst.mkdir(parents=True, exist_ok=True)
with tarfile.open(p, "r:gz") as tf:
    tf.extractall(dst)                          # flat ./*.json.gz + index
print(f"[onstart] corpus: {len(list(dst.glob('*.json.gz')))} games")
EOF
fi

# Stage the IMITATION dataset (games + manifest) for the human-holdout
# CE probe (the handoff observable) and the anchor builders. Escrowed
# as tier-b/replays_dataset_imitation.tar.gz (2026-08-10). Idempotent.
if [ ! -f replays_dataset_imitation/manifest.jsonl ] \
        && { [ -n "${HF_TOKEN:-}" ] || [ -f "$WORKDIR/.hf_token" ]; }; then
    HF_SEED_TOKEN="${HF_TOKEN:-}" \
    "$PY" - <<'EOF' && echo "[onstart] imitation dataset staged" \
        || echo "[onstart] WARN: imitation dataset staging failed (probe OFF)"
import os, pathlib, sys, tarfile
from huggingface_hub import hf_hub_download
tok = os.environ.get("HF_SEED_TOKEN") or pathlib.Path(
    os.environ.get("WORKDIR", "/workspace"), ".hf_token"
).read_text().strip()
try:
    p = hf_hub_download("momom2/wesnoth-model-checkpoints",
                        "tier-b/replays_dataset_imitation.tar.gz",
                        token=tok)
except Exception as e:                                  # noqa: BLE001
    print(f"[onstart] imitation dataset download failed: {e}")
    sys.exit(1)
dst = pathlib.Path("replays_dataset_imitation")
dst.mkdir(parents=True, exist_ok=True)
with tarfile.open(p, "r:gz") as tf:
    tf.extractall(dst)              # flat ./*.json.gz + manifest.jsonl
print(f"[onstart] imitation dataset: "
      f"{len(list(dst.glob('*.json.gz')))} games")
EOF
fi

# Fetch the FIRST-LAUNCH seed from HF when it is not in the git clone.
# Tier-a's 5M seed is committed (20 MB); larger tier seeds are escrowed
# on HF instead of bloating the repo, so a fresh box must be able to
# pull one. Only runs when the campaign has not started AND the seed is
# genuinely absent -- never overwrites a local file, and never touches
# a resume (which reads $CAMPAIGN, not $SEED_CKPT).
if [ ! -f "$CAMPAIGN" ] && [ ! -f "$SEED_CKPT" ] \
        && { [ -n "${HF_TOKEN:-}" ] || [ -f "$WORKDIR/.hf_token" ]; }; then
    "$PY" -m pip install --quiet huggingface_hub || true
    HF_SEED_TOKEN="${HF_TOKEN:-}" HF_SEED_DEST="$SEED_CKPT" \
    HF_SEED_NAME="$SEED_HF_NAME" \
    "$PY" - <<'EOF' || echo "[onstart] WARN: seed fetch failed"
import os, pathlib, shutil, sys
from huggingface_hub import hf_hub_download
tok = os.environ.get("HF_SEED_TOKEN") or pathlib.Path(
    os.environ.get("WORKDIR", "/workspace"), ".hf_token").read_text().strip()
name = os.environ["HF_SEED_NAME"]
try:
    p = hf_hub_download("momom2/wesnoth-model-checkpoints", name, token=tok)
except Exception as e:                                  # noqa: BLE001
    print(f"[onstart] seed download failed ({name}): {e}")
    sys.exit(1)
dst = pathlib.Path(os.environ["HF_SEED_DEST"])
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(p, dst)
print(f"[onstart] fetched first-launch seed: {name} -> {dst}")
EOF
fi

# A missing seed must fail HERE, loudly, rather than inside the training
# launch several steps later (where it would land in the relaunch loop
# and burn restarts against an unfixable condition).
if [ ! -f "$CAMPAIGN" ] && [ ! -f "$SEED_CKPT" ]; then
    echo "[onstart] FATAL: no campaign at $CAMPAIGN and no seed at $SEED_CKPT."
    echo "[onstart] Commit the seed, or escrow it on HF and pass HF_TOKEN."
    exit 1
fi

if [ -f "$CAMPAIGN" ]; then
    CKPT_IN="$CAMPAIGN"; RESET=""
    echo "[onstart] RESUME from $CAMPAIGN (no --reset-decision-step)"
else
    CKPT_IN="$SEED_CKPT"
    # Handoff default: NO decision-step reset. The combat-oracle
    # alphas are pinned 0.0 (the anneal a reset used to restart), and
    # keeping the imitation checkpoint's step (2,809,659) keeps the
    # telemetry/lineage numbering monotonic across the handoff. Pass
    # -e RESET_DECISION_STEP=1 for a genuinely fresh campaign.
    if [ "${RESET_DECISION_STEP:-0}" = "1" ]; then
        RESET="--reset-decision-step"
        echo "[onstart] FIRST LAUNCH from $CKPT_IN (+decision-step reset)"
    else
        RESET=""
        echo "[onstart] FIRST LAUNCH from $CKPT_IN (step carried)"
    fi
fi

# ---- Human-anchor rehearsal cache -----------------------------------
# HUMAN_ANCHOR_FILE points at a pre-encoded (RawEncoded, z, ml)
# pickle. It is deliberately NOT escrowed: the cache is invalid
# across encoder feature-layout changes (2026-07-11 grew the hex
# dynamic flags and side codes), so a fresh node REBUILDS it from
# the escrowed value corpus (value_corpus.tar.gz: index + games at
# tar root -> extracted into replays_dataset/). ~few minutes.
# DEFAULT ON since 2026-08-10 (A2 ruling: value rehearsal protects
# the fresh value head, AUC 0.951, from the documented self-play
# erosion 0.88->0.60). Disable with -e HUMAN_ANCHOR_FILE= (empty).
HUMAN_ANCHOR_FILE="${HUMAN_ANCHOR_FILE-replays_dataset/human_anchor.pkl}"
if [ -n "${HUMAN_ANCHOR_FILE:-}" ] && [ ! -f "$HUMAN_ANCHOR_FILE" ]; then
    if [ -n "${HF_TOKEN:-}" ] || [ -f "$WORKDIR/.hf_token" ]; then
        "$PY" -m pip install --quiet huggingface_hub || true
        HF_SEED_TOKEN="${HF_TOKEN:-}" "$PY" - <<'EOF' \
            && echo "[onstart] value corpus ready" \
            || echo "[onstart] WARN: value corpus fetch failed"
import os, pathlib, sys, tarfile
from huggingface_hub import hf_hub_download
tok = os.environ.get("HF_SEED_TOKEN") or pathlib.Path(
    os.environ.get("WORKDIR", "/workspace"), ".hf_token"
).read_text().strip()
dst = pathlib.Path("replays_dataset")
if (dst / "value_corpus_index.jsonl").is_file():
    print("[onstart] value corpus already extracted")
    sys.exit(0)
try:
    p = hf_hub_download("momom2/wesnoth-model-checkpoints", "tier-a/value_corpus.tar.gz",
                        token=tok)
except Exception as e:                                  # noqa: BLE001
    print(f"[onstart] corpus download failed: {e}")
    sys.exit(1)
dst.mkdir(parents=True, exist_ok=True)
with tarfile.open(p) as t:
    t.extractall(dst)
print(f"[onstart] extracted corpus -> {dst}")
EOF
        if [ -f replays_dataset/value_corpus_index.jsonl ]; then
            echo "[onstart] building human anchor -> $HUMAN_ANCHOR_FILE"
            NW=$(nproc); [ "$NW" -gt 24 ] && NW=24
            "$PY" tools/build_human_anchor.py \
                --out "$HUMAN_ANCHOR_FILE" --workers "$NW" \
                >> "$WORKDIR/onstart.log" 2>&1 \
                || echo "[onstart] WARN: anchor build failed"
        fi
    fi
    if [ ! -f "$HUMAN_ANCHOR_FILE" ]; then
        echo "[onstart] WARN: no anchor file; training will run WITHOUT"
        echo "[onstart]       the human rehearsal anchor."
        unset HUMAN_ANCHOR_FILE
    fi
fi

# POLICY-head anchor cache (F1; leg-2 arm). Built on box from the
# staged imitation dataset when HUMAN_ANCHOR_POLICY_FILE is set and
# the cache is absent. Activation is a USER decision (one prior
# protection per leg): pass -e HUMAN_ANCHOR_POLICY_FILE=... only for
# the leg that runs this arm.
if [ -n "${HUMAN_ANCHOR_POLICY_FILE:-}" ] \
        && [ ! -f "$HUMAN_ANCHOR_POLICY_FILE" ]; then
    if [ -f replays_dataset_imitation/manifest.jsonl ]; then
        echo "[onstart] building policy anchor -> $HUMAN_ANCHOR_POLICY_FILE"
        "$PY" tools/policy_anchor.py \
            --out "$HUMAN_ANCHOR_POLICY_FILE" \
            --games "${POLICY_ANCHOR_GAMES:-500}" \
            >> "$WORKDIR/onstart.log" 2>&1 \
            || echo "[onstart] WARN: policy anchor build failed"
    fi
    if [ ! -f "$HUMAN_ANCHOR_POLICY_FILE" ]; then
        echo "[onstart] WARN: no policy-anchor cache; policy rehearsal OFF"
        unset HUMAN_ANCHOR_POLICY_FILE
    fi
fi

# ---- Periodic checkpoint export (Hugging Face Hub) ------------------
# Opt-in: put a fine-grained write token (scoped to ONE model repo) in
# $WORKDIR/.hf_token, or set HF_TOKEN in the template env. Uploads the
# campaign checkpoint + CSV immediately and every 30 min -- a stopped
# instance's disk is unreachable (learned 2026-07-03), so anything not
# pushed off the node is hostage to the next outbid.
pkill -f 'hf_upload_loo[p].py' 2>/dev/null || true
if [ -n "${HF_TOKEN:-}" ] || [ -f "$WORKDIR/.hf_token" ]; then
    "$PY" -m pip install --quiet huggingface_hub || true
    WORKDIR="$WORKDIR" CAMPAIGN_FILE="$CAMPAIGN_FILE" \
        nohup "$PY" scripts/hf_upload_loop.py \
        >> "$WORKDIR/hf_upload.log" 2>&1 &
    echo "[onstart] HF checkpoint uploader ON (see hf_upload.log)"
else
    echo "[onstart] HF uploader off (no HF_TOKEN / $WORKDIR/.hf_token)"
fi

# ---- Periodic human-holdout CE probe (handoff observable) -----------
# THE pre-registered A1/F1 observable (2026-08-10): the imitation
# prior's human-play CE (t0 = 3.102) must stay flat through
# self-play. CPU-only subprocess (keeps the GPU for the learner);
# writes training/logs/holdout_probe.csv, which hf_upload_loop
# escrows. Needs replays_dataset_imitation/ staged on the box;
# skipped (with a loud line) when absent. Disable with PROBE_EVERY=0.
pkill -f 'holdout_probe_loo[p].py' 2>/dev/null || true
if [ "${PROBE_EVERY:-3600}" != "0" ]; then
    if [ -f "replays_dataset_imitation/manifest.jsonl" ]; then
        # PROBE_T0: the seed checkpoint's CE under this probe's exact
        # protocol -- 3.207 for the imit_tierb_start lineage (measured
        # 2026-08-10, 1200 pairs). With it set, the probe ABORTS
        # training after PROBE_ABORT_N consecutive reads above
        # t0+PROBE_ABORT_DELTA (default 0.5 x3) -- the guard the
        # 2026-08-12 diagnosis found missing. Override or empty it
        # (-e PROBE_T0=) for a different lineage.
        CAMPAIGN_FILE="$CAMPAIGN_FILE" \
        PROBE_T0="${PROBE_T0-3.207}" \
            nohup "$PY" scripts/holdout_probe_loop.py \
            >> "$WORKDIR/holdout_probe.log" 2>&1 &
        echo "[onstart] holdout probe ON (see holdout_probe.log)"
    else
        echo "[onstart] holdout probe OFF: no replays_dataset_imitation/"
        echo "[onstart]   (stage the imitation dataset to arm the"
        echo "[onstart]   handoff observable -- BACKLOG item 3)"
    fi
fi

# ---- SL_MODE: supervised behavior-cloning pass ----------------------
# SL_MODE=1 runs tools/supervised_train.py on the staged human corpus
# INSTEAD of self-play (user directive 2026-07-16: SL pass resumed
# from the latest campaign checkpoint, never fresh). Preemption-safe:
# a restart re-enters here and resumes from supervised.pt if it
# exists, else seeds from the campaign checkpoint (which the HF seed
# block above already fetched). Escrow: set
# HF_EXTRA_FILES='training/checkpoints/supervised.pt:supervised.pt,training/checkpoints/supervised_eval.jsonl:supervised_eval.jsonl'
# at create time so the uploader ships the SL artifacts too.
if [ "${SL_MODE:-0}" = "1" ]; then
    # NOT supervised.pt: that name is a git-TRACKED 471K-era (d_model
    # 128) checkpoint, so a fresh clone ships it and the
    # resume-if-exists logic below would pick it up -- the arch
    # guard caught exactly that on the first SL launch (2026-07-16).
    SL_OUT="${SL_OUT:-training/checkpoints/supervised_5m.pt}"
    if [ -f "$SL_OUT" ]; then
        SL_RESUME="$SL_OUT"
    else
        SL_RESUME="$CAMPAIGN"
    fi
    # Idempotency: Vast can re-fire onstart on a running instance
    # (observed 2026-07-16: a second trainer spawned mid-run and
    # contended for the GPU). One SL trainer at a time.
    pkill -f 'supervised_trai[n].py' 2>/dev/null || true
    sleep 2
    echo "[onstart] SL_MODE: behavior cloning, resume from $SL_RESUME"
    nohup "$PY" tools/supervised_train.py replays_dataset         --checkpoint "$SL_OUT"         --resume "$SL_RESUME"         --epochs "${SL_EPOCHS:-8}"         --bs "${SL_BS:-64}"         --lr "${SL_LR:-1e-4}"         --device cuda         --workers "${SL_WORKERS:-24}"         --d-model $D_MODEL --num-layers $NUM_LAYERS --num-heads $NUM_HEADS --d-ff $D_FF         --holdout-games "${SL_HOLDOUT:-300}"         --value-loss-weight "${SL_VALUE_WEIGHT:-1.0}"         --value-states-per-game "${SL_VALUE_SPG:-16}"         --eval-every "${SL_EVAL_EVERY:-50000}"         --eval-pairs "${SL_EVAL_PAIRS:-1200}"         >> "$WORKDIR/train.log" 2>&1 &
    echo "[onstart] SL training launched (tail -f $WORKDIR/train.log)"
    exit 0
fi

# Rotate a bloated train.log (the 2026-07-03 fd-leak spammed 134MB of
# tracebacks; keep restarts snappy and greps fast).
if [ -f "$WORKDIR/train.log" ] && \
        [ "$(stat -c%s "$WORKDIR/train.log")" -gt 50000000 ]; then
    mv "$WORKDIR/train.log" "$WORKDIR/train.log.1"
    echo "[onstart] rotated oversized train.log -> train.log.1"
fi

# Both spellings: torch <=2.7 reads PYTORCH_CUDA_ALLOC_CONF, newer
# reads PYTORCH_ALLOC_CONF.
# SPOOL_WORKERS / TRAIN_BATCH: GPU-memory knobs, overridable via
# -e at create time for smaller cards (16GB: 12 / 48; the 24GB
# defaults 16 / 64 measured ~17GB with creep).
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# GPU memory budget (24GB card, learned from the 2026-07-06 OOM):
# each spool worker pins a ~560MB CUDA context + model, so
# 24 workers (13.7GB) + trainer peak at batch 128 (~10GB) OOM'd
# mid-train-step (and allocator thrash made the step take 22 min).
# 16 workers (~9GB) + batch 64 (~5GB) leaves real headroom.
# 48 actor processes x (ctrl/resp queues + shipped experience pipes)
# exceed the container's default 1024-fd soft limit (observed
# 2026-07-02: OSError errno 24 in multiprocessing resource_sharer).
ulimit -n 65536 2>/dev/null || ulimit -n 4096 2>/dev/null || true
echo "[onstart] fd limit: $(ulimit -n)"
# Record the RESOLVED training mix in train.log (2026-07-20: the
# overnight campaign's effective mix was unrecoverable from the repo
# -- env overrides lived only on the box; one echo makes every
# future run's distribution auditable from the escrowed log alone).
# The five scenario-mix ratios MUST sum to 1 or sim_self_play exits rc=2.
# They used to be five independent env vars whose DEFAULTS happened to sum
# to 1, so changing any single one (historically: zeroing one ratio after its pool was
# retired) silently produced 0.95 and an unbootable box -- the supervisor
# then relaunched into the same error 20 times. Derive LADDER as the
# remainder so any single-ratio change is valid by construction; an
# explicit LADDER_RATIO still wins, and an over-subscribed mix fails loudly
# here rather than 20 relaunches deep.
MIDGAME_RATIO="${MIDGAME_RATIO:-0.2}"
# MINI default 0 since 2026-08-10 (A1 caveat: draw rate climbed under
# prior-discount damping on minis; keep them out of the handoff leg).
MINI_RATIO="${MINI_RATIO:-0.0}"
FOGLESS_RATIO="${FOGLESS_RATIO:-0.2}"
if [ -z "${LADDER_RATIO:-}" ]; then
    LADDER_RATIO=$("$PY" -c "r = round(1.0 - ($MIDGAME_RATIO + $MINI_RATIO + $FOGLESS_RATIO), 6); print(r if r >= 0 else 'NEGATIVE')")
    if [ "$LADDER_RATIO" = "NEGATIVE" ] || [ -z "$LADDER_RATIO" ]; then
        echo "[onstart] FATAL: mix ratios over-subscribe 1.0 (midgame=$MIDGAME_RATIO"\
             "mini=$MINI_RATIO fogless=$FOGLESS_RATIO)"
        exit 1
    fi
fi

# Topology (F3 ruling 2026-08-10): the ACTOR POOL is the tier-b
# production path -- weightless actor processes ship every leaf
# forward to the learner's central GPU batching server. The measured
# ~200 req/s server ceiling is 3-10x what 15M needs (20-62 req/s at
# 4-7k steps/hr), while spool workers' 15M CPU forwards project to
# ~2k steps/hr. Spool stays as the debug fallback: pass
# -e SPOOL_WORKERS=N (>0) to get the old topology. Caveat: training
# is not bit-deterministic under the pool (dynamic cross-actor
# batching). games-per-iter stays decoupled from the actor count
# (extra actors deepen the replay buffer, not the iteration).
if [ "${SPOOL_WORKERS:-0}" -gt 0 ]; then
    TOPO_ARGS="--spool-workers ${SPOOL_WORKERS} --spool-worker-device ${SPOOL_WORKER_DEVICE:-auto}${SPOOL_CUDA_WORKERS:+ --spool-cuda-workers $SPOOL_CUDA_WORKERS}"
    TOPO_DESC="spool=${SPOOL_WORKERS}"
else
    # Size from the CGROUP CPU QUOTA, not nproc: inside a Vast
    # container nproc reports the HOST's cores (measured 2026-08-10:
    # nproc=120 on a 38.4-core slice; /proc/loadavg is host-wide for
    # the same reason). cgroup v2 cpu.max = "quota period"; v1 =
    # cfs_quota_us/cfs_period_us; "max"/absent = uncapped -> nproc.
    _CORES=$("$PY" - <<'PYEOF'
import os
def cores():
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            return max(1, int(int(q) / int(p)))
    except OSError:
        pass
    try:
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0:
            return max(1, q // p)
    except OSError:
        pass
    return os.cpu_count() or 8
print(cores())
PYEOF
)
    ACTOR_POOL="${ACTOR_POOL:-$(( _CORES - 4 ))}"
    [ "$ACTOR_POOL" -lt 8 ] && ACTOR_POOL=8
    # Server fuse cap: fuse<=32 STILL OOM'd 19 min in (2026-08-10,
    # attempt 2) -- the collision is a train_step backward overlapping
    # BOTH serve threads' fused forwards (the MHA python path
    # materializes S^2 attention per layer). 16 x 2 threads + the
    # B=32 training chunks fit the 24GB card; raise only with
    # measured VRAM headroom. TRAIN_BATCH is the chunk size, not the
    # gradient batch (loss is /N-accumulated) -- memory-neutral to
    # training dynamics.
    TOPO_ARGS="--actor-pool ${ACTOR_POOL} --actor-max-batch ${ACTOR_MAX_BATCH:-16}"
    TOPO_DESC="actor-pool=${ACTOR_POOL} (quota ${_CORES} cores, fuse<=${ACTOR_MAX_BATCH:-16})"
fi
GAMES_PER_ITER="${GAMES_PER_ITER:-24}"

echo "[onstart] games_per_iter=${GAMES_PER_ITER} (${TOPO_DESC})"
echo "[onstart] training mix: midgame=${MIDGAME_RATIO}" \
     "mini=${MINI_RATIO}" \
     "fogless=${FOGLESS_RATIO} ladder=${LADDER_RATIO}" \
     >> "$WORKDIR/train.log"
# moves-left parked ENTIRELY (user 2026-07-21, "the training
# signal is already complicated enough"): neither the -0.2*Q*M
# PUCT utility (made losing positions prefer dragging) nor the
# aux head trains -- both flags removed, defaults OFF. The
# moves_left loss column keeps logging (0) so the CSV schema and
# log format stay stable. Checkpoint head weights load as
# tolerated unexpected keys.
# ---- SIM_FORK_GUARD smoke iteration (A6 ruling 2026-08-10) ----------
# One cheap in-process iteration with the deep-state fingerprint guard
# armed: the handoff is a new weight/config combination, and the guard
# caught three fork-aliasing bugs invisible to state_key. Runs against
# a THROWAWAY checkpoint path (never touches the campaign file); a
# guard trip is a FATAL config bug -- abort the launch rather than
# train on a corrupting fork. One ladder game at 8 turns exercises
# the historical bug surface (scenario events, combat forks) in
# ~10-15 min of 15M CPU forwards + fingerprint overhead (measured
# 2026-08-10: 2 games x 15 turns took ~30 min on the laptop).
# Skip: FORK_GUARD_SMOKE=0.
# Pass-marker keyed on the git rev: the smoke certifies a CODE+seed
# combination, so a re-run of the same rev (config-tuning reboots)
# skips the ~12 min. A new commit re-arms it.
_SMOKE_REV="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
if [ "${FORK_GUARD_SMOKE:-1}" = "1" ] \
        && [ -f "$WORKDIR/.fork_guard_passed_$_SMOKE_REV" ]; then
    echo "[onstart] fork-guard smoke already PASSED for $_SMOKE_REV; skipping"
    FORK_GUARD_SMOKE=0
fi
if [ "${FORK_GUARD_SMOKE:-1}" = "1" ]; then
    echo "[onstart] fork-guard smoke iteration (SIM_FORK_GUARD=1)..."
    if SIM_FORK_GUARD=1 "$PY" tools/sim_self_play.py \
        --mcts --mcts-sims 8 --device cpu \
        --d-model $D_MODEL --num-layers $NUM_LAYERS \
        --num-heads $NUM_HEADS --d-ff $D_FF \
        --iterations 1 --games-per-iter 1 --max-turns 8 \
        --ladder-ratio 1.0 --midgame-ratio 0 --mini-ratio 0 \
        --fogless-ratio 0 \
        --game-log-dir "" --validate-export-every 0 \
        --checkpoint-in "$CKPT_IN" \
        --checkpoint-out "$WORKDIR/fork_guard_smoke.pt" \
        --save-every 1000 --log-level WARNING \
        >> "$WORKDIR/onstart.log" 2>&1; then
        echo "[onstart] fork-guard smoke PASSED"
        touch "$WORKDIR/.fork_guard_passed_$_SMOKE_REV"
        rm -f "$WORKDIR/fork_guard_smoke.pt" \
              "$WORKDIR/fork_guard_smoke.pt.holdout" 2>/dev/null || true
    else
        echo "[onstart] FATAL: fork-guard smoke FAILED -- a search fork"
        echo "[onstart] mutated real game state (see onstart.log)."
        touch "$WORKDIR/ABORTED_fork_guard"
        exit 1
    fi
fi

# Supervised launch: relaunch on ordinary crashes (rc 1/2 -- OOM,
# transient CUDA errors) with a 60s backoff, capped at 20 restarts
# per onstart so a hard config bug can't burn the box all night
# (2026-07-06 lesson: 22 unsupervised OOM deaths). Tripwire aborts
# (rc >= 3) still stop everything and leave an ABORTED_* marker.
# After the first save, $RESET is dropped automatically: the
# campaign file exists, so a relaunch resumes it.
nohup bash -c "
  RESET='$RESET'
  tries=0
  while [ \$tries -lt 20 ]; do
    [ -f '$CAMPAIGN' ] && RESET=''
    '$PY' tools/sim_self_play.py --device cuda \
      --mcts --mcts-sims 32 \
      --d-model $D_MODEL --num-layers $NUM_LAYERS \
      --num-heads $NUM_HEADS --d-ff $D_FF \
      --replay-buffer --replay-updates 16 --value-coef 1.0 \
      --replay-minibatch ${REPLAY_MINIBATCH:-128} --replay-capacity 24000 \
      --train-batch-size ${TRAIN_BATCH:-32} --mcts-batch-size 16 \
      --mini-ratio ${MINI_RATIO} \
      --midgame-ratio ${MIDGAME_RATIO} --fogless-ratio ${FOGLESS_RATIO} \
      --ladder-ratio ${LADDER_RATIO} \
      ${MAX_TURNS:+--max-turns $MAX_TURNS} \
      --max-turns-min ${MAX_TURNS_MIN:-60} \
      ${RELEVANT_SET_HEXES:+--relevant-set-hexes} \
      --mcts-aux-score \
      ${AUX_VALUE_BONUS:+--mcts-aux-value-bonus $AUX_VALUE_BONUS} \
      --validate-export-every ${VALIDATE_EXPORT_EVERY:-1} \
      --value-label-smoothing 0.02 \
      --holdout-size 512 --holdout-per-game-cap 64 \
      ${HUMAN_ANCHOR_FILE:+--human-anchor-file $HUMAN_ANCHOR_FILE} \
      ${HUMAN_ANCHOR_UPDATES:+--human-anchor-updates $HUMAN_ANCHOR_UPDATES} \
      ${HUMAN_ANCHOR_BATCH:+--human-anchor-batch $HUMAN_ANCHOR_BATCH} \
      ${HUMAN_ANCHOR_POLICY_FILE:+--human-anchor-policy-file $HUMAN_ANCHOR_POLICY_FILE} \
      ${HUMAN_ANCHOR_POLICY_UPDATES:+--human-anchor-policy-updates $HUMAN_ANCHOR_POLICY_UPDATES} \
      ${HUMAN_ANCHOR_POLICY_BATCH:+--human-anchor-policy-batch $HUMAN_ANCHOR_POLICY_BATCH} \
      ${DRAW_VALUE_WEIGHT:+--draw-value-weight $DRAW_VALUE_WEIGHT} \
      --abort-decisive-rate ${ABORT_DECISIVE_RATE:-0.35} \
      --abort-window ${ABORT_WINDOW:-20} \
      --abort-holdout-stall ${ABORT_HOLDOUT_STALL:-60} \
      --distill-prior-discount ${DISTILL_PRIOR_DISCOUNT:-0.9} \
      ${TOPO_ARGS} --games-per-iter ${GAMES_PER_ITER} \
      \$RESET \
      --checkpoint-in  \$([ -f '$CAMPAIGN' ] && echo '$CAMPAIGN' || echo '$CKPT_IN') \
      --checkpoint-out $CAMPAIGN \
      --iterations 100000 --save-every 2 --log-level INFO \
      >> '$WORKDIR/train.log' 2>&1
    rc=\$?
    echo \"[onstart] training exited rc=\$rc at \$(date -u +%FT%TZ)\" >> '$WORKDIR/train.log'
    if [ \$rc -eq 0 ]; then break; fi
    # rc >= 128 = killed by signal. If the STALL WATCHDOG did it
    # (marker present), treat as a crash: consume the marker and
    # relaunch -- a hung leg loses minutes, not days (BACKLOG item 1;
    # the 2026-08-08 imitation hang billed ~2 idle days). Otherwise
    # it was an operator pkill / preemption shutdown: stand down
    # quietly WITHOUT an ABORTED marker.
    if [ \$rc -ge 128 ]; then
      if [ -f '$WORKDIR/WATCHDOG_STALL' ]; then
        cat '$WORKDIR/WATCHDOG_STALL' >> '$WORKDIR/train.log'
        rm -f '$WORKDIR/WATCHDOG_STALL'
        echo \"[onstart] watchdog kill; relaunching\" >> '$WORKDIR/train.log'
      else
        echo \"[onstart] signal exit; supervisor stands down\" >> '$WORKDIR/train.log'
        break
      fi
    fi
    # Tripwire aborts (3=reserved, 4=all-draws, 5=holdout stall,
    # 6=systemic index-basis mismatch between workers and learner)
    # need a human: marker blocks auto-relaunch until removed.
    if [ \$rc -ge 3 ] && [ \$rc -le 9 ]; then
      touch '$WORKDIR/ABORTED_'\$rc; break
    fi
    tries=\$((tries + 1))
    echo \"[onstart] relaunch \$tries/20 in 60s\" >> '$WORKDIR/train.log'
    sleep 60
  done
" >/dev/null 2>&1 &
echo "[onstart] training launched, supervised (tail -f $WORKDIR/train.log)"

# ---- Stall watchdog (BACKLOG item 1, 2026-08-10) --------------------
# Kills the training process when its CPU burn flatlines (the silent-
# hang symptom the tripwires and the relaunch loop both miss); the
# supervisor above sees the WATCHDOG_STALL marker and relaunches.
# Disable with -e STALL_WINDOW=0.
pkill -f 'stall_watchdo[g].py' 2>/dev/null || true
rm -f "$WORKDIR/WATCHDOG_STALL"
if [ "${STALL_WINDOW:-1800}" != "0" ]; then
    WORKDIR="$WORKDIR" nohup "$PY" scripts/stall_watchdog.py \
        >> "$WORKDIR/watchdog.log" 2>&1 &
    echo "[onstart] stall watchdog ON (see watchdog.log)"
fi
