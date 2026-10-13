"""The static-shape serve path (wesnoth_ai/graphed_serve.py) serves the
same compact actions, priors and values as the eager path, with the
same shapes padded to its caps; CPU, so the body runs eagerly and the
graph capture itself is certified on a box."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from helpers.priors_parity import _policy, _same, _states  # noqa: E402


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


def test_static_layout_carries_sightings_and_memory_as_the_packed_layout():
    """With sightings and memory slots, the static heads, offsets and
    memory slots are the packed layout's for the real segments, and every
    real row's source in the static token table holds what the packed
    layout's source holds in the stream-ordered table."""
    from wesnoth_ai.graphed_serve import static_index, static_sources
    from wesnoth_ai.model import ActorKind, TokenKind
    from wesnoth_ai.packed_trunk import build_packed_layout
    sizes = [(3, 2, 40), (1, 0, 25), (5, 1, 33)]
    sights, slots, k_cap = [2, 0, 4], [6, 6, 3], 8
    B, U_max, R_max, H_max = 3, 5, 2, 40
    real = build_packed_layout(sizes, H_max, U_max, R_max, TokenKind, ActorKind, source="streams",
                               sightings=sights, memory=slots, S_max=4, K_max=6)
    total = int(real.cu_seqlens[-1])
    b_cap, a_cap, h_cap = 5, 8, 48
    s = static_index(sizes, b_cap, a_cap, h_cap, t_cap=total + 9, sightings=sights,
                     memory=slots, k_cap=k_cap)
    np.testing.assert_array_equal(s.cu_seqlens[:B + 1], real.cu_seqlens)
    np.testing.assert_array_equal(s.actor.reshape(b_cap, a_cap)[:B, :U_max + R_max + 1],
                                  real.actor.reshape(B, -1))
    np.testing.assert_array_equal(s.hex.reshape(b_cap, h_cap)[:B, :H_max], real.hex.reshape(B, H_max))
    np.testing.assert_array_equal(s.glob[:B], real.glob)
    mem = s.memory.reshape(b_cap, k_cap)
    np.testing.assert_array_equal(mem[:B, :6], real.memory.reshape(B, 6))
    ends = s.cu_seqlens[1:B + 1] - 1
    assert (mem[:B, 6:] == ends[:, None]).all(), "slots past the widest memory point at end_turn"
    assert s.total == total
    # One label per (block, sample, row): the stream-ordered table of the
    # eager path and the static table of the graphed one.
    blocks = list(zip(*[(H, U, R, S, K) for (U, R, H), S, K in zip(sizes, sights, slots)]))
    stream, static = [], {}
    rows = (128, 40, 40, 40, b_cap * k_cap, b_cap)           # the static table's block rows
    bases = np.concatenate([[0], np.cumsum(rows)])
    for j, counts in enumerate(blocks):
        for b, n in enumerate(counts):
            for r in range(n):
                stream.append((j, b, r))
                row = (bases[j] + b * k_cap + r if j == 4
                       else bases[j] + sum(counts[:b]) + r)
                static[int(row)] = (j, b, r)
    for b in range(B):
        stream.append((5, b, 0))
        static[int(bases[5] + b)] = (5, b, 0)
    stream.append((6, 0, 0))
    static[int(bases[6])] = (6, 0, 0)
    counts = s.counts
    src, kind = static_sources(counts, bases[:6], k_cap, int(bases[6]))
    assert [static[int(i)] for i in src] == [stream[int(i)] for i in real.src]
    np.testing.assert_array_equal(kind, real.kind)


def _parity_policy(slots: int):
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(5)
    return TransformerPolicy(d_model=32, num_layers=1, num_heads=2, d_ff=64, device=torch.device("cpu"),
                             relevant_set_hexes=True, fog_hides_enemy_villages=True,
                             terrain_multi_hot=True, observation_parity=True, memory_slots=slots,
                             relevant_set_version=2)


def _parity_triples(policy, states, memories):
    from tools.inference_seam import RemoteEncoder
    enc = policy._inference_encoder
    renc = RemoteEncoder(enc.unit_type_to_id, enc.faction_to_id, relevant_set=True, server_priors=True,
                         fog_hides_enemy_villages=True, terrain_multi_hot=True,
                         observation_parity=True, relevant_set_version=2)
    lencs = [renc.encode(gs) for gs in states]
    return lencs, [(le._raw, le._masks, m) for le, m in zip(lencs, memories)]


def test_static_body_serves_the_parity_memory_model_as_the_eager_path():
    """The parity observation's sighting stream and faction posterior and
    an 8-slot memory through the static-shape body: the same compact
    actions, priors, values and new memory states as the eager path, for
    first decisions (the learned initial memory), later ones and a player
    at 4 slots; a second, smaller batch in the same bucket reads none of
    the first batch's rows."""
    from helpers.core_positions import core_positions
    from tools.inference_seam import InferenceServer
    from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors
    from wesnoth_ai.graphed_serve import Caps, GraphedServe
    from wesnoth_ai.memory import MemoryState
    policy = _parity_policy(8)
    enc, model = policy._inference_encoder, policy._inference_model
    pool = list(core_positions())
    _, probe = _parity_triples(policy, pool, [None] * len(pool))
    sighted = [gs for gs, t in zip(pool, probe) if t[0].sight_xs.shape[0]]
    states = sighted[:3] + [gs for gs, t in zip(pool, probe) if not t[0].sight_xs.shape[0]][:4]
    g = torch.Generator().manual_seed(3)
    memories = [MemoryState(8, None), MemoryState(8, 0.5 * torch.randn(8, 32, generator=g)),
                MemoryState(4, 0.5 * torch.randn(4, 32, generator=g)), MemoryState(8, None),
                MemoryState(8, 0.5 * torch.randn(8, 32, generator=g)),
                MemoryState(4, None), MemoryState(8, 0.5 * torch.randn(8, 32, generator=g))]
    lencs, triples = _parity_triples(policy, states, memories)
    assert any(t[0].sight_xs.shape[0] for t in triples), "no sighting in the batch"
    graphed = GraphedServe(model, enc, torch.device("cpu"), graphs=False,
                           caps=Caps(b_caps=(4, 8), a_caps=(128,), h_caps=(4096,),
                                     t_caps=(8192, 16384), max_len=4096))
    eager = InferenceServer(model, enc)
    served = InferenceServer(model, enc, graphed=graphed)
    for batch in (slice(0, 7), slice(2, 5)):
        with torch.no_grad():
            ref = eager.infer_batch(triples[batch])
            got = served.infer_batch(triples[batch])
        for le, gs, r, o, m in zip(lencs[batch], states[batch], ref, got, memories[batch]):
            _same(enumerate_legal_actions_with_priors(le, r, gs),
                  enumerate_legal_actions_with_priors(le, o, gs))
            assert torch.allclose(r.value, o.value, atol=1e-5), (r.value, o.value)
            assert torch.allclose(r.cliffness, o.cliffness, atol=1e-5)
            assert o.memory.shape == (m.k, 32)
            assert torch.allclose(r.memory, o.memory, atol=1e-5)
    assert graphed.served == 2 and not graphed.fallbacks, graphed.summary()


def _small_parity_serve(policy, **kw):
    from wesnoth_ai.graphed_serve import Caps, GraphedServe
    caps = Caps(b_caps=(4, 8), a_caps=(64, 128), h_caps=(512, 1024), t_caps=(2048, 4096, 8192),
                max_len=2048)
    return GraphedServe(policy._inference_model, policy._inference_encoder, torch.device("cpu"),
                        graphs=False, caps=caps, **kw)


def test_the_preflight_runs_the_largest_bucket():
    """The preflight's synthetic batch lands on the largest bucket of the
    caps (the most segments, actor and hex slots and token rows)."""
    policy = _parity_policy(8)
    serve = _small_parity_serve(policy)
    with torch.no_grad():
        info = serve.preflight()
    assert info["ok"] and serve.disabled is None, info
    assert list(serve.summary()["buckets"]) == ["8x128x1024x8192"] == [info["bucket"]]
    assert serve.served == 0, "the synthetic batch is not a served one"


def test_a_bucket_without_a_graph_goes_to_the_eager_path(monkeypatch):
    """A capture that fails, or that too little memory is left for, leaves
    the batch to the caller's eager path, which serves what it always
    serves; a failed bucket is not captured again; a failed preflight
    disables the instance; a batch carrying the oracle's biases goes
    eager without `biases`."""
    from helpers.core_positions import core_positions
    from tools.inference_seam import InferenceServer
    from wesnoth_ai.memory import MemoryState
    policy = _parity_policy(8)
    enc, model = policy._inference_encoder, policy._inference_model
    states = list(core_positions()[:3])
    _, triples = _parity_triples(policy, states, [MemoryState(8, None)] * 3)
    serve = _small_parity_serve(policy)
    serve.graphs = True                       # the CUDA branch of _run, on the CPU
    probes, captures = [], []
    monkeypatch.setattr(serve, "_probe_device", lambda: probes.append(1))
    monkeypatch.setattr(serve, "_available_bytes", lambda: 10 * serve.capture_reserve_bytes)

    def failing_capture(st):
        captures.append(st.bucket)
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    monkeypatch.setattr(serve, "_capture", failing_capture)
    eager = InferenceServer(model, enc)
    served = InferenceServer(model, enc, graphed=serve)
    with torch.no_grad():
        for _ in range(2):
            ref, got = eager.infer_batch(triples), served.infer_batch(triples)
            for r, o in zip(ref, got):
                assert torch.equal(r.value, o.value) and torch.equal(r.memory, o.memory)
                assert np.array_equal(r.legal_compact.prior, o.legal_compact.prior)
    assert len(captures) == 1 and len(probes) == 1, "one capture attempt, the device probed after it"
    assert serve.fallbacks == {"capture_failed": 2} and serve.served == 0
    assert "out of memory" in serve.capture_failures[0]
    assert served._graphed_serve() is serve, "a failed capture does not turn the graphed path off"
    # Too little memory left: no capture is tried, and a later batch may try.
    low = _small_parity_serve(policy)
    low.graphs = True
    monkeypatch.setattr(low, "_available_bytes", lambda: low.capture_reserve_bytes - 1)
    monkeypatch.setattr(low, "_capture", failing_capture)
    with torch.no_grad():
        assert low.infer([t[0] for t in triples], [t[1] for t in triples], [t[2] for t in triples]) is None
    assert low.fallbacks == {"memory": 1} and len(captures) == 1
    # A preflight that cannot capture disables the instance.
    dead = _small_parity_serve(policy)
    dead.graphs = True
    monkeypatch.setattr(dead, "_probe_device", lambda: None)
    monkeypatch.setattr(dead, "_available_bytes", lambda: 10 * dead.capture_reserve_bytes)
    monkeypatch.setattr(dead, "_capture", failing_capture)
    with torch.no_grad():
        assert not dead.preflight()["ok"]
        assert dead.infer([t[0] for t in triples], [t[1] for t in triples], [t[2] for t in triples]) is None
    assert dead.disabled and dead.fallbacks["disabled"] == 1
    # The oracle's biases ride the static masks only on request.
    biased = (triples[0][0], dataclasses.replace(triples[0][1], type_bias=np.zeros_like(
        triples[0][1].type_valid, dtype=np.float32)), triples[0][2])
    plain = _small_parity_serve(policy)
    with torch.no_grad():
        assert plain.infer([biased[0]], [biased[1]], [biased[2]]) is None
    assert plain.fallbacks == {"biases": 1}
