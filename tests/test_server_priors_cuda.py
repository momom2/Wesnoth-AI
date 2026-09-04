"""CUDA-only checks of the staged priors path (wesnoth_ai/
server_priors.py, docs/gpu_forward_design_20260904.md §6.2): the
compact output equals the CPU path's, nothing between the staging
copy and the final transfer synchronizes with the host, and the
serve-side timing hook reports device time. Skipped without CUDA;
runs on the box."""
from __future__ import annotations

import copy
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
CUDA = torch.device("cuda")


def _setup():
    """CPU policy plus CUDA copies of its inference encoder and model
    (same weights), the harvested states, encodings and packs."""
    from tests.test_server_priors import _policy, _states
    from wesnoth_ai.server_priors import pack_masks
    policy = _policy()
    enc, model = policy._inference_encoder, policy._inference_model
    enc_cuda = copy.deepcopy(enc).to(CUDA)
    model_cuda = copy.deepcopy(model).to(CUDA)
    states = _states()
    encs = [enc.encode(gs) for gs in states]
    packs = [pack_masks(e, gs) for e, gs in zip(encs, states)]
    return enc, model, enc_cuda, model_cuda, states, encs, packs


def _to_device(padded, device):
    kw = {f: (v.to(device) if f in padded._TENSOR_FIELDS and v is not None else v)
          for f, v in ((f.name, getattr(padded, f.name)) for f in dataclasses.fields(padded))}
    return type(padded)(**kw)


def _to_device_encoded(e):
    kw = {f.name: (v.to(CUDA) if torch.is_tensor(v) else v)
          for f in dataclasses.fields(e) for v in (getattr(e, f.name),)}
    return type(e)(**kw)


def _same_compact(x, y, rtol):
    for f in ("actor", "kind", "target", "weapon"):
        assert np.array_equal(getattr(x, f), getattr(y, f)), f
    assert x.prior.dtype == y.prior.dtype == np.float64
    assert np.allclose(x.prior, y.prior, rtol=rtol, atol=0.0)


def test_cuda_priors_equal_cpu_priors_on_the_same_forward():
    """Same head outputs fed to both devices: identical legal entries,
    priors equal up to the softmax kernels' rounding."""
    from wesnoth_ai.server_priors import batched_priors
    _, model, _, _, _, encs, packs = _setup()
    with torch.no_grad():
        padded = model.forward_padded(encs)
        cpu = batched_priors(padded, packs)
        gpu = batched_priors(_to_device(padded, CUDA), packs)
    assert len(cpu) == len(gpu) == len(packs)
    for x, y in zip(cpu, gpu):
        _same_compact(x, y, rtol=1e-6)
    assert sum(len(c.actor) for c in gpu) > 0


def test_cuda_seam_matches_reference_enumeration():
    from tools.inference_seam import InferenceServer, RemoteEncoder, RemoteModel
    from tests.test_server_priors import _same
    from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors
    enc, model, enc_cuda, model_cuda, states, encs, _ = _setup()
    with torch.no_grad():
        ref = [enumerate_legal_actions_with_priors(e, model(e), gs)
               for e, gs in zip(encs, states)]
        renc = RemoteEncoder(enc.unit_type_to_id, enc.faction_to_id, server_priors=True)
        rmodel = RemoteModel(InferenceServer(model_cuda, enc_cuda))
        lencs = [renc.encode(gs) for gs in states]
        outs = rmodel.forward_batch(lencs)
    for le, out, r, gs in zip(lencs, outs, ref, states):
        assert out.value.device.type == "cpu"
        _same(r, enumerate_legal_actions_with_priors(le, out, gs))


def test_no_host_sync_between_staging_copy_and_final_transfer():
    """torch raises on any implicit sync (blocking copy, `.item()`,
    `nonzero`, boolean-mask indexing, stream/device synchronize) while
    the debug mode is "error"; `start_priors` must issue none. The
    first call warms the caching allocators (a cudaMalloc/cudaHostAlloc
    on a miss synchronizes outside torch's instrumentation)."""
    from wesnoth_ai.server_priors import start_priors
    _, model, _, model_cuda, _, encs, packs = _setup()
    encs_cuda = [_to_device_encoded(e) for e in encs]
    with torch.no_grad():
        padded = model_cuda.forward_padded(encs_cuda)
        extras = [padded.value, padded.value_logits, padded.cliffness]
        warm, warm_extras = start_priors(padded, packs, extras).finish()
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            pending = start_priors(padded, packs, extras)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        compact, got_extras = pending.finish()
    for x, y in zip(warm, compact):
        _same_compact(x, y, rtol=0.0)
    for a, b in zip(warm_extras, got_extras):
        assert np.array_equal(a, b)


def test_gpu_timing_hook_reports_device_time():
    from tools.inference_seam import InferenceServer, RemoteEncoder
    enc, _, enc_cuda, model_cuda, states, _, _ = _setup()
    renc = RemoteEncoder(enc.unit_type_to_id, enc.faction_to_id, server_priors=True)
    items = [(le._raw, le._masks) for le in (renc.encode(gs) for gs in states)]
    server = InferenceServer(model_cuda, enc_cuda)
    st = {"gpu_ms": 0.0}
    server.infer_batch(items, stats=st)
    first = st["gpu_ms"]
    assert first > 0.0
    server.infer_batch(items, stats=st)
    assert st["gpu_ms"] > first
    assert server.infer_batch(items)[0].legal_compact is not None    # no stats: no timing
