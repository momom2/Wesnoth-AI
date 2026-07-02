#!/usr/bin/env python3
"""Sampler-on-CPU split + B2 batch reads (gpu_perf_patches.md #1/#2).

On CPU the split must be a strict NO-OP (same objects back, no
copies) so the whole existing suite pins the CPU path byte-identical.
The CUDA behavior (bulk D2H instead of per-actor syncs) can only be
validated on a GPU node -- that's the Kaggle smoke + re-profile's
job, per the spec's explicit warning that CPU tests cannot catch a
missed field.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

import glob
import torch

from transformer_policy import TransformerPolicy
from tools.mcts import _leaf_to_cpu

from test_inference_snapshot import _gs


def test_leaf_to_cpu_is_identity_on_cpu():
    """CPU inputs must pass through as the SAME objects -- the gate
    keeps the CPU path (and therefore every other test) untouched."""
    policy = TransformerPolicy()
    with torch.no_grad():
        encoded = policy._encoder.encode(_gs())
        output = policy._model(encoded)
    assert output.actor_logits.device.type == "cpu"
    enc2, out2 = _leaf_to_cpu(encoded, output)
    assert enc2 is encoded, "CPU encoded must not be copied"
    assert out2 is output, "CPU output must not be copied"
