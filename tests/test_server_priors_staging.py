"""The staged priors path (wesnoth_ai/server_priors.py, design note
docs/gpu_forward_design_20260904.md §6.2) on CPU: the two-phase API,
the ride-along outputs, the capacity bound, the buffer layout, and a
batch holding a sample with no legal action. The CUDA-only checks
(no host sync, timing) are in tests/test_server_priors_cuda.py."""
from __future__ import annotations

import copy
import functools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


@functools.lru_cache(maxsize=None)
def _batch():
    """(policy, encoded states, packs, game states), harvested once."""
    from tests.test_server_priors import _policy, _states
    from wesnoth_ai.server_priors import pack_masks
    policy = _policy()
    enc = policy._inference_encoder
    states = _states()
    encs = [enc.encode(gs) for gs in states]
    packs = [pack_masks(e, gs) for e, gs in zip(encs, states)]
    return policy, encs, packs, states


def _same_compact(x, y, exact=True):
    for f in ("actor", "kind", "target", "weapon"):
        assert np.array_equal(getattr(x, f), getattr(y, f)), f
    if exact:
        assert np.array_equal(x.prior, y.prior)
    else:
        assert np.allclose(x.prior, y.prior, rtol=1e-6, atol=0.0)


def test_two_phase_api_carries_extras_in_the_same_transfer():
    from wesnoth_ai.server_priors import batched_priors, start_priors
    policy, encs, packs, _ = _batch()
    model = policy._inference_model
    with torch.no_grad():
        padded = model.forward_padded(encs)
        ref = batched_priors(padded, packs)
        pending = start_priors(padded, packs, [padded.value, padded.value_logits])
        compact, extras = pending.finish()
    for x, y in zip(ref, compact):
        _same_compact(x, y)
    assert len(extras) == 2
    assert np.array_equal(extras[0], padded.value.numpy())
    assert np.array_equal(extras[1], padded.value_logits.numpy())
    assert extras[1].shape == tuple(padded.value_logits.shape)


def test_capacity_counted_from_bits_bounds_the_legal_entries():
    from wesnoth_ai.server_priors import _legal_capacity, batched_priors
    policy, encs, packs, _ = _batch()
    model = policy._inference_model
    with torch.no_grad():
        padded = model.forward_padded(encs)
        compact = batched_priors(padded, packs)
    W = padded.weapon_logits.shape[2]
    for p, c in zip(packs, compact):
        cap = _legal_capacity(p, W)
        assert len(c.actor) <= cap
        # Only softmax underflow (a logit gap > ~100) can make the bound
        # strict; a fresh small net has no such gap.
        assert len(c.actor) == cap


def test_sample_with_no_legal_actor_is_empty_and_isolated():
    from wesnoth_ai.server_priors import batched_priors
    policy, encs, packs, _ = _batch()
    model = policy._inference_model
    with torch.no_grad():
        padded = model.forward_padded(encs)
        ref = batched_priors(padded, packs)
        dead = copy.deepcopy(packs[1])
        dead.actor_mask[:] = 0
        got = batched_priors(padded, [packs[0], dead] + packs[2:])
    assert len(got[1].actor) == 0 and got[1].prior.dtype == np.float64
    for b in (0, *range(2, len(packs))):
        _same_compact(ref[b], got[b])


def test_layout_places_every_field_at_an_aligned_offset():
    from wesnoth_ai.server_priors import _Layout
    layout = _Layout([("k", torch.int8, (5,)), ("p", torch.float64, (3,)),
                      ("a", torch.int32, (2, 3)), ("m", torch.uint8, (0, 4)),
                      ("f", torch.float32, (7,))])
    seen = 0
    for name, dt, shape, off, n in layout.fields:
        assert off % dt.itemsize == 0, name
        assert n == int(np.prod(shape)) * dt.itemsize
        seen += n
    assert layout.nbytes == seen == 5 + 24 + 24 + 0 + 28
    buf = torch.zeros(layout.nbytes, dtype=torch.uint8)
    hv = layout.numpy_views(buf.numpy())
    hv["p"][:] = [1.5, -2.0, 3.25]
    hv["a"][:] = np.arange(6).reshape(2, 3)
    hv["k"][:] = -1
    tv = layout.torch_views(buf)
    assert torch.equal(tv["p"], torch.tensor([1.5, -2.0, 3.25], dtype=torch.float64))
    assert torch.equal(tv["a"], torch.arange(6, dtype=torch.int32).view(2, 3))
    assert torch.equal(tv["k"], torch.full((5,), -1, dtype=torch.int8))
    assert tv["m"].shape == (0, 4)


def test_server_stats_untouched_on_cpu():
    from tools.inference_seam import InferenceServer, RemoteEncoder
    policy, _, _, states = _batch()
    enc, model = policy._inference_encoder, policy._inference_model
    renc = RemoteEncoder(enc.unit_type_to_id, enc.faction_to_id, server_priors=True)
    le = renc.encode(states[0])
    st = {"gpu_ms": 0.0}
    outs = InferenceServer(model, enc).infer_batch([(le._raw, le._masks)], stats=st)
    assert st == {"gpu_ms": 0.0}
    assert outs[0].legal_compact is not None and outs[0].value.device.type == "cpu"
