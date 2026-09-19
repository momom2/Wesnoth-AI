"""The static-shape serve path (wesnoth_ai/graphed_serve.py) serves the
same compact actions, priors and values as the eager path, with the
same shapes padded to its caps; CPU, so the body runs eagerly and the
graph capture itself is certified on a box."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from tests.test_server_priors import _policy, _same, _states  # noqa: E402


def _pairs(policy, states):
    from tools.inference_seam import RemoteEncoder
    enc = policy._inference_encoder
    renc = RemoteEncoder(enc.unit_type_to_id, enc.faction_to_id, terrain_multi_hot=enc.terrain_multi_hot,
                         fog_hides_enemy_villages=enc.fog_hides_enemy_villages, server_priors=True)
    lencs = [renc.encode(gs) for gs in states]
    return lencs, [(le._raw, le._masks) for le in lencs]


def test_static_index_extends_the_packed_layout():
    from wesnoth_ai.graphed_serve import static_index
    from wesnoth_ai.model import ActorKind, TokenKind
    from wesnoth_ai.packed_trunk import build_packed_layout
    sizes = [(3, 2, 40), (1, 0, 25), (5, 1, 33)]
    B, A_max, H_max = 3, 5 + 2 + 1, 40
    real = build_packed_layout(sizes, H_max, 5, 2, TokenKind, ActorKind, source="streams")
    b_cap, a_cap, h_cap = 5, 8, 48
    total = int(real.cu_seqlens[-1])
    s = static_index(sizes, b_cap, a_cap, h_cap, t_cap=total + 2 + 5)
    assert s.cu_seqlens.dtype == np.int32 and len(s.cu_seqlens) == b_cap + 1
    np.testing.assert_array_equal(s.cu_seqlens[:B + 1], real.cu_seqlens)
    # Pad segments: one token each, right after the real tokens.
    np.testing.assert_array_equal(s.cu_seqlens[B:], total + np.arange(b_cap - B + 1))
    np.testing.assert_array_equal(s.actor.reshape(b_cap, a_cap)[:B, :A_max],
                                  real.actor.reshape(B, A_max))
    np.testing.assert_array_equal(s.hex.reshape(b_cap, h_cap)[:B, :H_max],
                                  real.hex.reshape(B, H_max))
    np.testing.assert_array_equal(s.glob[:B], real.glob)
    np.testing.assert_array_equal(s.actor_kind[:B, :A_max], real.actor_kind)
    # Every padded slot points at a row inside its own segment.
    act = s.actor.reshape(b_cap, a_cap)
    hx = s.hex.reshape(b_cap, h_cap)
    for b in range(b_cap):
        lo, hi = s.cu_seqlens[b], s.cu_seqlens[b + 1]
        assert (act[b] >= lo).all() and (act[b] < hi).all()
        assert (hx[b] >= lo).all() and (hx[b] < hi).all()
        assert lo <= s.glob[b] < hi
    assert (s.actor_kind[B:] == ActorKind.END_TURN).all()


def test_static_body_matches_the_eager_priors_path():
    """GraphedServe with graphs off runs the static-shape body on the
    same tensors a graph would replay: its compact actions, priors and
    values equal the eager seam's on every state, including one whose
    batch is padded from 5 to the 8-segment cap."""
    from tools.inference_seam import InferenceServer
    from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors
    from wesnoth_ai.graphed_serve import Caps, GraphedServe
    policy = _policy()
    enc, model = policy._inference_encoder, policy._inference_model
    states = _states()
    lencs, pairs = _pairs(policy, states)
    eager = InferenceServer(model, enc)
    with torch.no_grad():
        ref = eager.infer_batch(pairs)
        graphed = GraphedServe(model, enc, torch.device("cpu"), graphs=False,
                               caps=Caps(b_caps=(4, 8), a_caps=(128,), h_caps=(4096,),
                                         t_caps=(8192, 16384), max_len=4096))
        got = InferenceServer(model, enc, graphed=graphed).infer_batch(pairs)
    assert graphed.served == 1 and not graphed.fallbacks, graphed.summary()
    for le, gs, r, g in zip(lencs, states, ref, got):
        _same(enumerate_legal_actions_with_priors(le, r, gs),
              enumerate_legal_actions_with_priors(le, g, gs))
        assert torch.allclose(r.value, g.value, atol=1e-4), (r.value, g.value)
        assert torch.allclose(r.cliffness, g.cliffness, atol=1e-4)
    # A second batch reuses the bucket's buffers: the masks of the first
    # must not leak into it (fewer, different states).
    with torch.no_grad():
        ref2 = eager.infer_batch(pairs[1:3])
        got2 = InferenceServer(model, enc, graphed=graphed).infer_batch(pairs[1:3])
    for le, gs, r, g in zip(lencs[1:3], states[1:3], ref2, got2):
        _same(enumerate_legal_actions_with_priors(le, r, gs),
              enumerate_legal_actions_with_priors(le, g, gs))
    assert graphed.served == 2
    # The 7-state batch took the 8-segment bucket, the 2-state one the 4.
    assert {k.split("x")[0] for k in graphed.summary()["buckets"]} == {"8", "4"}


def test_batches_past_a_cap_take_the_eager_path():
    from tools.inference_seam import InferenceServer
    from wesnoth_ai.graphed_serve import Caps, GraphedServe
    policy = _policy()
    enc, model = policy._inference_encoder, policy._inference_model
    states = _states()
    _lencs, pairs = _pairs(policy, states)
    with torch.no_grad():
        ref = InferenceServer(model, enc).infer_batch(pairs)
        too_few_segments = GraphedServe(model, enc, torch.device("cpu"), graphs=False,
                                        caps=Caps(b_cap=2, max_len=4096))
        got = InferenceServer(model, enc, graphed=too_few_segments).infer_batch(pairs)
    assert too_few_segments.fallbacks == {"segments": 1} and too_few_segments.served == 0
    for r, g in zip(ref, got):
        assert torch.equal(r.value, g.value)
        assert np.array_equal(r.legal_compact.prior, g.legal_compact.prior)


def test_a_factory_gives_each_serve_thread_its_own_instance():
    """InferenceServer(graphed=<factory>) builds one GraphedServe per
    calling thread and reports them all in graphed_summary."""
    import threading
    from tools.inference_seam import InferenceServer
    from wesnoth_ai.graphed_serve import Caps, GraphedServe
    policy = _policy()
    enc, model = policy._inference_encoder, policy._inference_model
    _lencs, pairs = _pairs(policy, _states())
    caps = Caps(b_caps=(8,), a_caps=(128,), h_caps=(4096,), t_caps=(8192, 16384), max_len=4096)
    server = InferenceServer(model, enc, graphed=lambda: GraphedServe(
        model, enc, torch.device("cpu"), graphs=False, caps=caps))
    outs = {}

    def serve(name):
        with torch.no_grad():
            outs[name] = server.infer_batch(pairs[:3])

    threads = [threading.Thread(target=serve, args=(n,)) for n in ("t1", "t2")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    summary = server.graphed_summary()
    assert set(summary) == {"thread0", "thread1"}, summary
    assert all(v["served"] == 1 for v in summary.values()), summary
    for a, b in zip(outs["t1"], outs["t2"]):
        assert torch.equal(a.value, b.value)
        assert np.array_equal(a.legal_compact.prior, b.legal_compact.prior)
