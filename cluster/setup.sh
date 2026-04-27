#!/usr/bin/env bash
# Set up the project on a fresh mesogip user account.
#
# Run from the project root after the bundle is extracted.
# Idempotent: re-run is safe; will reuse existing venv.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
echo "[setup] project dir: ${PROJECT_DIR}"

# Pick a Python. Cluster's Ubuntu 24.04 ships Python 3.12; that's fine for
# our deps. Fall back to system python3 if the user has something custom.
PYBIN="${PYBIN:-python3}"
echo "[setup] python: $($PYBIN --version 2>&1)"

# venv at project root, named .venv. Conda would also work; sticking with
# venv to avoid pulling miniconda just for this.
if [ ! -d "${PROJECT_DIR}/.venv" ]; then
    echo "[setup] creating venv at .venv"
    "$PYBIN" -m venv "${PROJECT_DIR}/.venv"
fi
# shellcheck disable=SC1091
source "${PROJECT_DIR}/.venv/bin/activate"
python -m pip install --quiet --upgrade pip wheel

# Probe driver/CUDA to pick the right torch wheel. nvidia-smi prints
# "CUDA Version: X.Y" on its top-right corner. We default to cu121 (works
# with driver >= 530); if the driver is older, fall back to cu118.
CUDA_TAG="cu121"
if command -v nvidia-smi >/dev/null; then
    DRIVER_LINE=$(nvidia-smi 2>/dev/null | head -3 | tail -1 || true)
    echo "[setup] nvidia-smi: ${DRIVER_LINE}"
    if echo "${DRIVER_LINE}" | grep -q "CUDA Version: 11"; then
        CUDA_TAG="cu118"
    fi
else
    echo "[setup] WARNING: nvidia-smi not found; installing cu121 anyway"
fi
echo "[setup] using torch wheel index: ${CUDA_TAG}"

# Install torch + numpy. Skip torchvision/torchaudio — we don't use them.
python -m pip install --quiet \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
    torch
# numpy comes from default PyPI; PyTorch's index doesn't host it.
python -m pip install --quiet numpy

# Quick CUDA sanity check.
python - <<'PY'
import torch
print(f"[setup] torch:        {torch.__version__}")
print(f"[setup] cuda avail:   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[setup] device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"[setup]   [{i}] {torch.cuda.get_device_name(i)}  cap={torch.cuda.get_device_capability(i)}")
else:
    raise SystemExit("[setup] CUDA not available — wrong wheel or no driver. "
                     "Re-run with PYBIN= or change CUDA_TAG manually.")
PY

mkdir -p "${PROJECT_DIR}/training/checkpoints"
echo "[setup] done."
echo "[setup] activate the env later with: source ${PROJECT_DIR}/.venv/bin/activate"
