"""The inference server's host work in Rust (tools/serve_stage.py) answers
as the Python path does, byte for byte, and the memory states the server
keeps between decisions serve each decision the state the wire would
have carried, or refuse it."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from wesnoth_ai import game_core as gc  # noqa: E402

pytestmark = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")
D, SLOTS = 32, 8
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def leaves():
    """(RawEncoded, PackedMasks) pairs of random legal games, as own1's
    recipe encodes them. The map of seed 6 starts each leader on its keep,
    so the first positions offer recruits."""
    from tools.bench_serve_host import harvest_leaves
    out = harvest_leaves(24, seed=6)
    assert any(m.actor_mask[m.n_units:m.n_units + m.n_recruits].any() for _, m in out), "no legal recruit"
    return out


@pytest.fixture(scope="module")
def server():
    from tools.inference_seam import InferenceServer
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.model import WesnothModel
    torch.manual_seed(3)
    model = WesnothModel(d_model=D, num_layers=1, num_heads=2, d_ff=64, observation_parity=True,
                         memory_slots=SLOTS).eval()
    encoder = GameStateEncoder(d_model=D, relevant_set_hexes=True, fog_hides_enemy_villages=True,
                               terrain_multi_hot=True, observation_parity=True, relevant_set_version=2).eval()
    return InferenceServer(model, encoder, device=CPU, output_device=CPU, autocast_bf16=False,
                           packed_embed=True)


class _Req:
    def __init__(self, rid, message, cid=1):
        self.rid, self.message, self.cid = rid, message, cid
        self.leaves = 1


def _request(rid, items, served=None, cid=1):
    from tools.serve_loop import KIND_INFER
    from wesnoth_ai.byte_wire import encode_request
    return _Req(rid, encode_request(items, prefix=(KIND_INFER, rid, len(items)), served=served), cid)


def _python(server, requests, memory=None):
    from tools.eval_inference_server import _model_infer_fn
    return _model_infer_fn(server, rust_stage=False, memory=memory)(requests)


def _rust(server, requests, memory=None):
    from tools.serve_stage import serve_requests
    return serve_requests(server, requests, memory=memory, cids=[r.cid for r in requests])


def test_the_rust_stage_answers_as_the_python_path(leaves, server):
    """Requests of one and two leaves, each memory on the wire (a state, or
    none at a game-side's first decision): the Rust stage's replies are
    the Python path's bytes."""
    from wesnoth_ai.memory import MemoryState
    rng = np.random.default_rng(1)
    reqs = []
    i = 0
    rid = 0
    while i < len(leaves):
        n = 2 if rid % 3 == 1 else 1
        items = []
        for raw, masks in leaves[i:i + n]:
            state = None if (i + len(items)) % 4 == 0 else rng.standard_normal((SLOTS, D)).astype(np.float32)
            items.append((raw, masks, MemoryState(SLOTS, state)))
        reqs.append(_request(rid, items))
        i += n
        rid += 1
    want = _python(server, reqs)
    got = _rust(server, reqs)
    assert len(got) == len(reqs) and all(len(m) > 100 for m in got)
    assert got == want


def test_the_packed_index_is_build_packed_layouts(leaves, server):
    """The packed trunk's index arrays the Rust stage writes are
    `build_packed_layout(source="streams")`'s and `stream_row_index`'s,
    for memory slot counts that differ between leaves."""
    from tools.serve_stage import _ACTOR_KINDS, _TOKEN_KINDS, parse_requests
    from wesnoth_ai.memory import MemoryState
    ks = [SLOTS, 3, 0, SLOTS, 5, 1] * 4
    reqs = [_request(i, [(raw, masks, MemoryState(ks[i], None))]) for i, (raw, masks) in enumerate(leaves)]
    parsed = parse_requests([r.message for r in reqs])
    counts = parsed.memory_counts(SLOTS)
    assert counts == ks[:len(leaves)]
    dims = parsed.index_dims(counts)
    buf = np.zeros(8 * (2 * dims[0] + len(leaves) * (2 * dims[1] + dims[2] + dims[3] + 1 + dims[6])
                        + len(leaves) + 1 + dims[7]), dtype=np.uint8)
    assert parsed.fill_packed_index(buf, 0, _TOKEN_KINDS, _ACTOR_KINDS, counts) == buf.nbytes
    got = buf.view(np.int64)
    sizes = parsed.sizes()
    want = server._model.trunk_index(sizes, parsed.sighting_counts(), counts, CPU, packed=True)
    lay = want.layout
    parts = [lay.src, lay.kind, lay.actor, lay.hex, lay.unit, lay.glob, lay.memory, lay.cu_seqlens,
             lay.actor_kind.reshape(-1), want.memory_rows.numpy()]
    assert np.array_equal(got, np.concatenate(parts))
    assert dims[8] == lay.max_len


def test_the_masks_and_capacity_are_write_masks(leaves, server):
    """The masks the Rust stage writes, with the compaction capacity, are
    `server_priors._stage_masks`'s."""
    from tools.serve_stage import _MASK_NAMES, parse_requests
    from wesnoth_ai.server_priors import _stage_masks, mask_layout
    reqs = [_request(i, [(raw, masks)]) for i, (raw, masks) in enumerate(leaves)]
    parsed = parse_requests([r.message for r in reqs])
    packs = [m for _, m in leaves]
    A = max(m.n_units + m.n_recruits + 1 for m in packs)
    H = max(m.n_hexes for m in packs)
    layout, host, capacity = _stage_masks(packs, A, H, 2, 4, False)
    ml = mask_layout(len(packs), A, H, 2, 4)
    assert [f[:3] for f in ml.fields] == [f[:3] for f in layout.fields]
    buf = np.full(ml.nbytes, 0xAB, dtype=np.uint8)
    offsets = {f[0]: f[3] for f in ml.fields}
    got = parsed.fill_masks(buf, [offsets[n] for n in _MASK_NAMES], A, 2, (H + 7) // 8, H, 4, (0, ml.nbytes))
    assert got == capacity
    assert np.array_equal(buf, host.numpy())


def test_the_memory_books():
    """Each game-side keeps its last two states; a read of another version,
    a key twice in a batch, another slot count or a released state is
    refused; an aborted write leaves its row holding nothing."""
    import wesnoth_core
    books = wesnoth_core.MemoryTable(3)
    reads, writes, why = books.plan([1, 1], [0, 1], [0, 0], [8, 8])
    assert why == [None, None] and reads == [0, 0] and writes[0] != writes[1] and 0 not in writes
    books.commit()
    a1 = writes[0]
    reads, writes, why = books.plan([1], [0], [1], [8])           # version 1 is held
    assert why == [None] and reads == [a1] and writes[0] != a1
    books.commit()
    a2 = writes[0]
    # The decision re-decided (a refused recruit): version 1 again, written
    # into the row the refused decision wrote.
    reads, writes, why = books.plan([1], [0], [1], [8])
    assert (reads, writes, why) == ([a1], [a2], [None])
    books.commit()
    # A write that did not complete over the row holding version 1: that row
    # holds nothing from then on, and version 2 is still served.
    reads, writes, why = books.plan([1], [0], [2], [8])
    assert (reads, writes, why) == ([a2], [a1], [None])
    books.abort()
    _, _, why = books.plan([1], [0], [1], [8])
    assert why[0] is not None and "not held" in why[0]
    books.abort()
    reads, _, why = books.plan([1], [0], [2], [8])
    assert reads == [a2] and why == [None]
    books.abort()
    for cids, keys, versions, ks in (([1], [0], [5], [8]), ([2], [0], [1], [8]), ([1], [0], [2], [4]),
                                     ([1, 1], [1, 1], [1, 1], [8, 8])):
        _, _, why = books.plan(cids, keys, versions, ks)
        assert why[-1] is not None, (cids, keys, versions)
        books.abort()
    _, _, why = books.plan([1, 1], [2, 3], [0, 0], [8, 8])        # past three game-sides
    assert why[0] is None and "already holds" in why[1]
    books.abort()
    # The aborted write of key 2's version 1 holds nothing.
    _, _, why = books.plan([1], [2], [1], [8])
    assert why[0] is not None
    books.abort()
    rows = books.rows
    assert books.release_connection(1) == 3 and len(books) == 0
    _, _, why = books.plan([1], [0], [2], [8])
    assert "not held" in why[0]
    books.abort()
    reads, writes, why = books.plan([7], [0], [0], [8])           # the freed rows serve again
    assert why == [None] and books.rows == rows


def _decisions(leaves):
    """A scripted line of decisions of two game-sides (key 0 and 1): each
    (key, version read, leaf), with a re-decided decision (a refused
    recruit) at key 0's third."""
    return [[(0, 0, 0), (1, 0, 1)], [(0, 1, 2), (1, 1, 3)], [(0, 2, 4)], [(0, 2, 5), (1, 2, 6)],
            [(0, 3, 7), (1, 3, 8)]]


@pytest.mark.parametrize("rust", [True, False])
def test_the_served_memory_plays_as_the_wire_memory(leaves, server, rust):
    """A line of decisions with the server keeping the memory, re-decided
    decision included, answers as the same line with each state on the
    wire (the state the previous decision wrote, before it at the
    re-decision); a read of a version the server does not hold is refused
    alone, and none is served after the connection releases them."""
    from tools.serve_loop import STATUS_OK, reply_head
    from tools.serve_stage import ServerMemory
    from wesnoth_ai.byte_wire import decode_replies_with_versions
    from wesnoth_ai.memory import MemoryState
    serve = _rust if rust else _python
    memory = ServerMemory(server._model, CPU)
    held = {0: [None], 1: [None]}            # per key, the wire state of each version
    rid = 0
    for batch in _decisions(leaves):
        wire_reqs, served_reqs = [], []
        for key, version, leaf in batch:
            raw, masks = leaves[leaf]
            wire_reqs.append(_request(rid, [(raw, masks, MemoryState(SLOTS, held[key][version]))]))
            served_reqs.append(_request(rid, [(raw, masks, MemoryState(SLOTS, None))], served=[(key, version)]))
            rid += 1
        wire = serve(server, wire_reqs)
        served = serve(server, served_reqs, memory)
        for (key, version, _), w, s in zip(batch, wire, served):
            (wv,), wvers = decode_replies_with_versions(bytearray(w), 16)
            (sv,), svers = decode_replies_with_versions(bytearray(s), 16)
            assert reply_head(s)[1] == STATUS_OK and svers == [version + 1] and wvers == [-1]
            assert sv[10] is None and wv[10] is not None
            for a, b in zip(wv[3:9], sv[3:9]):
                assert (a is None and b is None) or np.array_equal(a, b)
            for name in ("actor", "kind", "target", "weapon", "prior"):
                assert np.array_equal(getattr(wv[9], name), getattr(sv[9], name))
            held[key][version + 1:] = [np.array(wv[10])]
    # Key 0 holds versions 4 and 3: version 2 is gone, and refused alone.
    raw, masks = leaves[9]
    reqs = [_request(rid, [(raw, masks, MemoryState(SLOTS, None))], served=[(0, 2)]),
            _request(rid + 1, [(raw, masks, MemoryState(SLOTS, None))], served=[(1, 3)])]
    out = serve(server, reqs, memory)
    assert reply_head(out[0])[1] != STATUS_OK and b"not held" in out[0]
    assert reply_head(out[1])[1] == STATUS_OK
    memory.release_connection(1)
    out = serve(server, [_request(rid + 2, [(raw, masks, MemoryState(SLOTS, None))], served=[(1, 4)])], memory)
    assert reply_head(out[0])[1] != STATUS_OK
    assert memory.summary()["held"] == 0


def test_a_server_without_memory_refuses_a_served_state(leaves, server):
    """A leaf naming a state the server keeps, sent to a server that keeps
    none, is refused, never served from the initial memory."""
    from wesnoth_ai.memory import MemoryState
    raw, masks = leaves[0]
    req = _request(0, [(raw, masks, MemoryState(SLOTS, None))], served=[(0, 1)])
    for serve in (_rust, _python):
        with pytest.raises(ValueError, match="keeps none"):
            serve(server, [req])


def test_the_client_refuses_a_state_it_did_not_hand_out():
    """A served state names its client: another client refuses to send it,
    and a server that keeps no states refuses it too."""
    from tools.eval_inference_server import EvalInferenceClient
    from wesnoth_ai.byte_wire import ServedMemory
    from wesnoth_ai.memory import MemoryState
    client = object.__new__(EvalInferenceClient)
    client.server_memory, client._token, client._next_key, client._holds = True, 11, 0, False
    client.address = "test"
    assert client._served_keys([(None, None, MemoryState(8, None)), (None, None, MemoryState(8, None))]) \
        == [(0, 0), (1, 0)]
    assert client._served_keys([(None, None, MemoryState(8, ServedMemory(11, 1, 4)))]) == [(1, 4)]
    with pytest.raises(ValueError, match="another connection"):
        client._served_keys([(None, None, MemoryState(8, ServedMemory(12, 1, 4)))])
    with pytest.raises(ValueError, match="on the wire is refused"):
        client._served_keys([(None, None, MemoryState(8, np.zeros((8, 4), np.float32)))])
    client.server_memory = False
    with pytest.raises(ValueError, match="keeps none"):
        client._served_keys([(None, None, MemoryState(8, ServedMemory(11, 1, 4)))])


def _graphed_server(server, device_memory: bool):
    """The eager server's model behind the static-shape path, its body run
    eagerly on the CPU."""
    from tools.inference_seam import InferenceServer
    from wesnoth_ai.graphed_serve import Caps, GraphedServe
    caps = Caps(b_caps=(4, 8, 16, 32), a_caps=(64, 128), h_caps=(512, 1024),
                t_caps=(4096, 8192, 16384, 32768), max_len=4096)
    gserve = GraphedServe(server._model, server._encoder, CPU, graphs=False, caps=caps,
                          device_memory=device_memory)
    return InferenceServer(server._model, server._encoder, device=CPU, output_device=CPU,
                           autocast_bf16=False, packed_embed=True, graphed=gserve), gserve


def test_the_graphed_path_stages_from_the_rust_core_as_from_numpy(leaves, server):
    """The static-shape path's embed, index and mask buffers written by the
    Rust core (`infer_parsed`) are the numpy staging's (`infer`), a larger
    batch's leftovers cleared, and the replies from them the same bytes;
    every batch fits a bucket."""
    from wesnoth_ai.memory import MemoryState
    rng = np.random.default_rng(2)
    reqs = [_request(i, [(raw, masks, MemoryState(SLOTS, None if i % 3 == 0 else
                                                  rng.standard_normal((SLOTS, D)).astype(np.float32)))])
            for i, (raw, masks) in enumerate(leaves[:12])]
    big, small = reqs[:7], reqs[7:12]                 # both in the bucket of 8 segments
    bufs, replies = {}, {}
    for name, serve in (("numpy", _python), ("rust", _rust)):
        graphed_server, gserve = _graphed_server(server, device_memory=False)
        replies[name] = serve(graphed_server, big) + serve(graphed_server, small)
        assert gserve.served == 2 and not gserve.fallbacks and len(gserve._states) == 1
        st = next(iter(gserve._states.values()))
        bufs[name] = [st.embed_host.numpy().copy(), st.index_host.numpy().copy(), st.mask_host.numpy().copy()]
    for part, a, b in zip(("embed", "index", "mask"), bufs["numpy"], bufs["rust"]):
        assert np.array_equal(a, b), part
    assert replies["rust"] == replies["numpy"]


@pytest.mark.parametrize("rust", [True, False])
def test_the_graphed_path_serves_the_kept_memory_as_the_wire(leaves, server, rust):
    """The static-shape path with the memory states on the device answers
    a line of decisions, re-decided decision included, as the eager path
    with the states on the wire."""
    from tools.serve_stage import ServerMemory
    from wesnoth_ai.byte_wire import decode_replies_with_versions
    from wesnoth_ai.memory import MemoryState
    graphed_server, gserve = _graphed_server(server, device_memory=True)
    serve = _rust if rust else _python
    memory = ServerMemory(server._model, CPU)
    held = {0: [None], 1: [None]}
    rid = 0
    for batch in _decisions(leaves):
        wire_reqs, kept_reqs = [], []
        for key, version, leaf in batch:
            raw, masks = leaves[leaf]
            wire_reqs.append(_request(rid, [(raw, masks, MemoryState(SLOTS, held[key][version]))]))
            kept_reqs.append(_request(rid, [(raw, masks, MemoryState(SLOTS, None))], served=[(key, version)]))
            rid += 1
        wire = _python(server, wire_reqs)
        kept = serve(graphed_server, kept_reqs, memory)
        for (key, version, _), w, k in zip(batch, wire, kept):
            (wv,), _ = decode_replies_with_versions(bytearray(w), 16)
            (kv,), versions = decode_replies_with_versions(bytearray(k), 16)
            assert versions == [version + 1]
            for name in ("actor", "kind", "target", "weapon"):
                assert np.array_equal(getattr(wv[9], name), getattr(kv[9], name))
            np.testing.assert_allclose(kv[9].prior, wv[9].prior, rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(kv[3], wv[3], rtol=1e-4, atol=1e-5)
            held[key][version + 1:] = [np.array(wv[10])]
    assert gserve.served == len(_decisions(leaves)) and not gserve.fallbacks


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _cuda_server(server, graphed: Optional[bool] = None):
    """The tiny model on the GPU as the eval server runs it: bf16, the
    packed trunk and embed; with `graphed` the static-shape path, its
    memory states on the device when True, from the host when False."""
    import copy
    from tools.inference_seam import InferenceServer
    from wesnoth_ai.graphed_serve import Caps, GraphedServe
    dev = torch.device("cuda")
    model, encoder = copy.deepcopy(server._model).to(dev), copy.deepcopy(server._encoder).to(dev)
    model.infer_packed_trunk = True
    gserve = None
    if graphed is not None:
        caps = Caps(b_caps=(4, 8, 16, 32), a_caps=(64, 128), h_caps=(512, 1024),
                    t_caps=(4096, 8192, 16384, 32768), max_len=4096)
        gserve = GraphedServe(model, encoder, dev, caps=caps, device_memory=graphed)
    return InferenceServer(model, encoder, device=dev, output_device=CPU, autocast_bf16=True,
                           packed_embed=True, graphed=gserve), gserve


@needs_cuda
def test_on_cuda_the_rust_stage_answers_as_the_python_path(leaves, server):
    """On the GPU (bf16, packed trunk and embed): the Rust stage's replies
    are the Python path's bytes, memory on the wire."""
    from wesnoth_ai.memory import MemoryState
    cuda_server, _ = _cuda_server(server)
    rng = np.random.default_rng(4)
    reqs = [_request(i, [(raw, m, MemoryState(SLOTS, None if i % 4 == 0 else
                                               rng.standard_normal((SLOTS, D)).astype(np.float32)))])
            for i, (raw, m) in enumerate(leaves)]
    assert _rust(cuda_server, reqs) == _python(cuda_server, reqs)


@needs_cuda
@pytest.mark.parametrize("graphed", [None, True])
def test_on_cuda_the_kept_memory_plays_as_the_wire(leaves, server, graphed):
    """On the GPU, a line of decisions with the memory kept on the device
    (eager, or graphed with its device buffers) answers as the same path
    with the states on the wire (graphed from the host), byte for byte."""
    from tools.serve_stage import ServerMemory
    from wesnoth_ai.byte_wire import decode_replies_with_versions
    from wesnoth_ai.memory import MemoryState
    kept_server, kept_g = _cuda_server(server, graphed)
    wire_server, wire_g = _cuda_server(server, None if graphed is None else False)
    memory = ServerMemory(kept_server._model, torch.device("cuda"))
    held = {0: [None], 1: [None]}
    rid = 0
    for batch in _decisions(leaves):
        wire_reqs, kept_reqs = [], []
        for key, version, leaf in batch:
            raw, masks = leaves[leaf]
            wire_reqs.append(_request(rid, [(raw, masks, MemoryState(SLOTS, held[key][version]))]))
            kept_reqs.append(_request(rid, [(raw, masks, MemoryState(SLOTS, None))], served=[(key, version)]))
            rid += 1
        wire = _rust(wire_server, wire_reqs)
        kept = _rust(kept_server, kept_reqs, memory)
        for (key, version, _), w, k in zip(batch, wire, kept):
            (wv,), _ = decode_replies_with_versions(bytearray(w), 16)
            (kv,), versions = decode_replies_with_versions(bytearray(k), 16)
            assert versions == [version + 1]
            for i in range(3, 9):
                assert (wv[i] is None and kv[i] is None) or np.array_equal(wv[i], kv[i])
            for name in ("actor", "kind", "target", "weapon", "prior"):
                assert np.array_equal(getattr(wv[9], name), getattr(kv[9], name))
            held[key][version + 1:] = [np.array(wv[10])]
    if graphed is not None:
        assert kept_g.served == len(_decisions(leaves)) and not kept_g.fallbacks
        assert wire_g.served == len(_decisions(leaves)) and not wire_g.fallbacks


@pytest.mark.parametrize("path", ["rust", "python", "graphed"])
def test_priors_only_serves_the_same_priors_without_the_head_outputs(leaves, server, path):
    """A priors-only model (model.infer_priors_only, the server's
    --priors-only) answers with the same priors, actions, actor kinds and
    memory states, byte for byte, and no value, value distribution or
    cliffness; on the eager Rust and Python paths and the static-shape one."""
    from wesnoth_ai.byte_wire import decode_replies_with_versions
    from wesnoth_ai.memory import MemoryState
    rng = np.random.default_rng(6)
    reqs = [_request(i, [(raw, m, MemoryState(SLOTS, None if i % 3 == 0 else
                                               rng.standard_normal((SLOTS, D)).astype(np.float32)))])
            for i, (raw, m) in enumerate(leaves[:10])]
    model = server._model
    replies = {}
    for priors_only in (False, True):
        model.infer_priors_only = priors_only
        try:
            if path == "graphed":
                served, _ = _graphed_server(server, device_memory=False)
                replies[priors_only] = _rust(served, reqs)
            else:
                replies[priors_only] = (_rust if path == "rust" else _python)(server, reqs)
        finally:
            model.infer_priors_only = False
    for full, slim in zip(replies[False], replies[True]):
        (fv,), _ = decode_replies_with_versions(bytearray(full), 16)
        (sv,), _ = decode_replies_with_versions(bytearray(slim), 16)
        assert fv[3] is not None and fv[4] is not None and fv[5] is not None
        assert sv[3] is None and sv[4] is None and sv[5] is None
        assert (fv[1], fv[2]) == (sv[1], sv[2]) and np.array_equal(fv[8], sv[8])
        for name in ("actor", "kind", "target", "weapon", "prior"):
            assert getattr(fv[9], name).tobytes() == getattr(sv[9], name).tobytes(), name
        assert fv[10].tobytes() == sv[10].tobytes()
        assert len(slim) < len(full)


@needs_cuda
def test_on_cuda_priors_only_serves_the_same_priors(leaves, server):
    """On the GPU (bf16, packed trunk and embed), the priors-only model's
    replies carry the full model's priors, actions and memory states byte
    for byte."""
    from wesnoth_ai.byte_wire import decode_replies_with_versions
    from wesnoth_ai.memory import MemoryState
    cuda_server, _ = _cuda_server(server)
    reqs = [_request(i, [(raw, m, MemoryState(SLOTS, None))]) for i, (raw, m) in enumerate(leaves)]
    full = _rust(cuda_server, reqs)
    cuda_server._model.infer_priors_only = True
    slim = _rust(cuda_server, reqs)
    for f, s_ in zip(full, slim):
        (fv,), _ = decode_replies_with_versions(bytearray(f), 16)
        (sv,), _ = decode_replies_with_versions(bytearray(s_), 16)
        assert sv[3] is None and fv[3] is not None
        for name in ("actor", "kind", "target", "weapon", "prior"):
            assert getattr(fv[9], name).tobytes() == getattr(sv[9], name).tobytes(), name
        assert fv[10].tobytes() == sv[10].tobytes()
