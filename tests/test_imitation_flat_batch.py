"""The batched imitation loss equals the per-sample reference.

The trainer's batched flow (tools/supervised_train._flush_batch) embeds
a batch from one host buffer, runs the trunk once and computes every
head's cross-entropy over the PaddedOutput (wesnoth_ai/imitation_loss.py).
Here the same pairs go through the per-sample reference
(`encode_from_raw` + `forward_batch` + `_loss_parts_for_output`) and
the two must agree on every head's value, on which heads fired, on the
batched flow's total and on the gradient of every parameter. The
holdout eval's cache must reproduce the streamed probe exactly.
"""
import random
from pathlib import Path

import pytest
import torch

from tools.replay_dataset import ActionIndices
from wesnoth_ai.encoder import GameStateEncoder, encode_raw
from wesnoth_ai.imitation_loss import build_imitation_targets, imitation_loss_parts
from wesnoth_ai.model import WesnothModel

_ARCH = dict(d_model=64, num_layers=2, num_heads=2, d_ff=128)
_TYPE_W = {"move": 0.189, "attack": 0.628, "recruit": 1.748, "end_turn": 1.435}


def _states(n):
    from tools.bench_infer import harvest_states
    from tools.scenario_pool import build_scenario_gamestate, random_setup
    states = [build_scenario_gamestate(random_setup(random.Random(7)))]
    states += harvest_states(n - 1, seed=11)
    return states[:n]


def _labels(raws, rng):
    """One label per sample over every head shape: unit moves and
    attacks (type, target, weapon), a recruit, end_turn, an actor
    outside the sample (contributes nothing), a value-only pair and a
    pair without an outcome."""
    kinds = ["move", "attack", "recruit", "end_turn", "skip", "value_only", "no_value", "attack"]
    ais, zw = [], []
    for b, raw in enumerate(raws):
        U, R, H = len(raw.unit_ids), len(raw.recruit_types), int(raw.hex_xs.shape[0])
        A = U + R + 1
        kind = kinds[b % len(kinds)]
        z, vw, pw = (1 if b % 2 else -1), 0.7, 1.0
        if kind == "move" and U:
            ai = ActionIndices("move", rng.randrange(U), target_idx=rng.randrange(H), type_idx=1)
        elif kind == "attack" and U:
            ai = ActionIndices("attack", rng.randrange(U), target_idx=rng.randrange(H),
                               weapon_idx=rng.randrange(2), type_idx=0)
        elif kind == "recruit" and R:
            ai = ActionIndices("recruit", U + rng.randrange(R), target_idx=rng.randrange(H))
        elif kind == "skip":
            ai = ActionIndices("move", A + 3, target_idx=0, type_idx=1)
        elif kind == "value_only":
            ai, pw = ActionIndices("end_turn", A - 1), 0.0
        elif kind == "no_value":
            ai, z, vw = ActionIndices("end_turn", A - 1), None, 0.0
        else:
            ai = ActionIndices("end_turn", A - 1)
        ais.append(ai)
        zw.append((z, vw, pw))
    return ais, zw


def _reference(model, enc, raws, ais, zw, dev):
    from tools.supervised_train import _loss_parts_for_output
    encoded = [enc.encode_from_raw(r, device=dev) for r in raws]
    outs = model.forward_batch(encoded)
    parts = [_loss_parts_for_output(o, ai, dev, type_loss_weights=_TYPE_W,
                                    value_z=z, value_weight=vw, policy_weight=pw)
             for o, ai, (z, vw, pw) in zip(outs, ais, zw)]
    total = (sum(pw * (p.actor + p.type + p.target + p.weapon) for p, (_, _, pw) in zip(parts, zw))
             + sum(vw * p.value for p, (_, vw, _) in zip(parts, zw)))
    return parts, total


def _grad_close(a, b):
    if a is None or b is None:
        return (a is None or float(a.abs().max()) < 1e-7) and (b is None or float(b.abs().max()) < 1e-7)
    scale = max(1e-6, float(b.abs().max()))
    return torch.allclose(a, b, atol=1e-6 + 1e-4 * scale, rtol=1e-3)


def test_batched_imitation_loss_matches_per_sample_reference():
    torch.manual_seed(3)
    dev = torch.device("cpu")
    enc = GameStateEncoder(d_model=_ARCH["d_model"])
    model = WesnothModel(**_ARCH)
    states = _states(8)
    for s in states:
        enc.register_names(s)
    raws = [encode_raw(s, type_to_id=enc.unit_type_to_id, faction_to_id=enc.faction_to_id)
            for s in states]
    ais, zw = _labels(raws, random.Random(5))
    model.eval()
    enc.eval()            # no dropout: the two paths must agree exactly
    params = list(model.parameters()) + list(enc.parameters())

    ref_parts, ref_total = _reference(model, enc, raws, ais, zw, dev)
    ref_total.backward()
    ref_grads = [None if p.grad is None else p.grad.clone() for p in params]
    for p in params:
        p.grad = None

    streams = enc.encode_from_raw_embedded(raws, device=dev)
    padded = model.forward_embedded(streams)
    targets = build_imitation_targets(
        ais, zw, streams.sizes, n_types=padded.type_logits.shape[2],
        n_weapons=padded.weapon_logits.shape[2], n_atoms=padded.value_logits.shape[1],
        type_loss_weights=_TYPE_W, device=dev)
    parts = imitation_loss_parts(padded, targets)

    fired = {"actor": [p.actor_fired for p in ref_parts], "type": [p.type_fired for p in ref_parts],
             "target": [p.target_fired for p in ref_parts],
             "weapon": [p.weapon_fired for p in ref_parts],
             "value": [p.value_fired for p in ref_parts]}
    for head, flags in fired.items():
        assert list(targets.ok[head]) == flags, head
        assert any(flags), f"no {head} label fired: the test exercises nothing for it"
    assert not all(fired["actor"]) and not all(fired["value"])

    heads = [("actor", parts.actor_raw, [p.actor for p in ref_parts]),
             ("type", parts.type, [p.type for p in ref_parts]),
             ("target", parts.target, [p.target for p in ref_parts]),
             ("weapon", parts.weapon, [p.weapon for p in ref_parts]),
             ("value", parts.value_raw, [p.value for p in ref_parts])]
    for head, batched, ref in heads:
        ref_t = torch.tensor([float(v.detach()) for v in ref])
        assert torch.allclose(batched.detach(), ref_t, atol=1e-5, rtol=1e-4), (head, batched, ref_t)
    assert abs(float(parts.total.detach()) - float(ref_total.detach())) < 1e-4 * max(1.0, abs(float(ref_total)))

    parts.total.backward()
    n_checked = 0
    for p, g in zip(params, ref_grads):
        assert _grad_close(p.grad, g)
        n_checked += g is not None and float(g.abs().max()) > 0
    assert n_checked > 10


@pytest.mark.skipif(not Path("replays_dataset").exists(), reason="needs the replay dataset")
def test_holdout_eval_cache_replays_the_same_probe(monkeypatch):
    from tools import supervised_train as st
    from tools.replay_dataset import filter_competitive_2p
    files = filter_competitive_2p(Path("replays_dataset"))[:2]
    torch.manual_seed(1)
    enc = GameStateEncoder(d_model=_ARCH["d_model"])
    model = WesnothModel(**_ARCH)
    cache: list = []
    first = st._evaluate(model, enc, files, torch.device("cpu"), eval_pairs=6, cache=cache)
    assert first["n"] == 6 and len(cache) >= 6

    def _no_stream(*a, **k):
        pytest.fail("the cached probe must not reconstruct the holdout games")
    monkeypatch.setattr(st, "_pair_stream_serial", _no_stream)
    again = st._evaluate(model, enc, files, torch.device("cpu"), eval_pairs=6, cache=cache)
    assert again == first


def test_flush_batch_splits_on_out_of_memory(monkeypatch):
    """A batch that does not fit the device is split and accumulated,
    never dropped: the step equals the unsplit step and every pair
    reaches the log. Simulated by an out-of-memory error on any chunk
    larger than two pairs."""
    import copy
    from collections import deque
    from tools import supervised_train as st

    torch.manual_seed(4)
    dev = torch.device("cpu")
    enc = GameStateEncoder(d_model=_ARCH["d_model"])
    model = WesnothModel(**_ARCH)
    model.eval()
    enc.eval()
    states = _states(6)
    for s in states:
        enc.register_names(s)
    raws = [encode_raw(s, type_to_id=enc.unit_type_to_id, faction_to_id=enc.faction_to_id)
            for s in states]
    ais, zw = _labels(raws, random.Random(9))
    ref_model, ref_enc = copy.deepcopy(model), copy.deepcopy(enc)

    def run(m, e, batch_loss):
        monkeypatch.setattr(st, "_batch_loss", batch_loss)
        opt = torch.optim.SGD(list(m.parameters()) + list(e.parameters()), lr=0.1)
        dq = {k: deque(maxlen=50) for k in ("t", "a", "ty", "tg", "w", "v")}
        splits = st._flush_batch(m, e, raws, ais, zw, opt, list(m.parameters()), len(raws), dev,
                                 dq["t"], dq["a"], dq["ty"], dq["tg"], dq["w"], dq["v"],
                                 type_loss_weights=_TYPE_W)
        return splits, dq

    original = st._batch_loss
    ref_splits, ref_dq = run(ref_model, ref_enc, original)
    assert ref_splits == 0

    def oom_above_two(m, e, chunk, *args):
        if len(chunk) > 2:
            raise torch.cuda.OutOfMemoryError("simulated")
        return original(m, e, chunk, *args)

    splits, dq = run(model, enc, oom_above_two)
    assert splits == 2                      # halvings: 6 -> 2 x 3 -> 4 x 2 fits
    assert list(dq["a"]) == pytest.approx(list(ref_dq["a"]), abs=1e-5)
    assert len(dq["a"]) == len(ref_dq["a"]) > 0
    for p, q in zip(list(model.parameters()) + list(enc.parameters()),
                    list(ref_model.parameters()) + list(ref_enc.parameters())):
        assert torch.allclose(p, q, atol=1e-6, rtol=1e-4)


def test_bf16_autocast_matches_fp32_within_tolerance():
    """--bf16 runs the batched flow's forward and loss under bf16
    autocast with fp32 weights: the loss stays within a few percent of
    the fp32 loss and the gradient keeps its direction."""
    from tools.supervised_train import _batch_loss

    torch.manual_seed(5)
    dev = torch.device("cpu")
    enc = GameStateEncoder(d_model=_ARCH["d_model"])
    model = WesnothModel(**_ARCH)
    model.eval()
    enc.eval()
    states = _states(6)
    for s in states:
        enc.register_names(s)
    raws = [encode_raw(s, type_to_id=enc.unit_type_to_id, faction_to_id=enc.faction_to_id)
            for s in states]
    ais, zw = _labels(raws, random.Random(11))
    params = list(model.parameters()) + list(enc.parameters())

    def run(dtype):
        for p in params:
            p.grad = None
        parts, _ = _batch_loss(model, enc, raws, ais, zw, dev, _TYPE_W, autocast_dtype=dtype)
        parts.total.backward()
        grads = torch.cat([p.grad.flatten() for p in params if p.grad is not None])
        return float(parts.total.detach()), grads.clone()

    loss32, g32 = run(None)
    loss16, g16 = run(torch.bfloat16)
    assert all(p.dtype == torch.float32 for p in params)
    assert torch.isfinite(g16).all()
    assert abs(loss16 - loss32) <= 0.05 * abs(loss32) + 1e-3, (loss16, loss32)
    cos = torch.nn.functional.cosine_similarity(g16, g32, dim=0)
    assert float(cos) > 0.95, float(cos)


def test_out_of_memory_inside_backward_does_not_double_count_the_gradient(monkeypatch):
    """The peak is inside backward(), so that is where the device runs
    out. A backward that raises part way has ALREADY accumulated the
    gradients of the layers it walked; retrying the batch without
    zeroing adds those contributions a second time and the step is
    wrong in a way nothing reports. The retry must start from zeroed
    gradients, so the result equals the unsplit step exactly."""
    import copy
    from collections import deque
    from tools import supervised_train as st

    torch.manual_seed(11)
    dev = torch.device("cpu")
    enc = GameStateEncoder(d_model=_ARCH["d_model"])
    model = WesnothModel(**_ARCH)
    model.eval()
    enc.eval()
    states = _states(4)
    for s in states:
        enc.register_names(s)
    raws = [encode_raw(s, type_to_id=enc.unit_type_to_id, faction_to_id=enc.faction_to_id)
            for s in states]
    ais, zw = _labels(raws, random.Random(3))
    ref_model, ref_enc = copy.deepcopy(model), copy.deepcopy(enc)

    def run(m, e, patch):
        if patch is not None:
            monkeypatch.setattr(st, "_batch_loss", patch)
        opt = torch.optim.SGD(list(m.parameters()) + list(e.parameters()), lr=0.1)
        dq = {k: deque(maxlen=50) for k in ("t", "a", "ty", "tg", "w", "v")}
        splits = st._flush_batch(m, e, raws, ais, zw, opt, list(m.parameters()), len(raws), dev,
                                 dq["t"], dq["a"], dq["ty"], dq["tg"], dq["w"], dq["v"],
                                 type_loss_weights=_TYPE_W)
        return splits, dq

    original = st._batch_loss
    ref_splits, ref_dq = run(ref_model, ref_enc, None)
    assert ref_splits == 0

    class _PartialBackward:
        """Loss parts whose backward accumulates into the FIRST
        parameter and only then runs out of memory -- what a real
        out-of-memory inside backward() leaves behind."""

        def __init__(self, inner, victim):
            self._inner = inner
            self._victim = victim

        def __getattr__(self, name):
            return getattr(self._inner, name)

        @property
        def total(self):
            return self

        def __truediv__(self, other):
            return self

        def backward(self):
            with torch.no_grad():                      # the partial accumulation
                if self._victim.grad is None:
                    self._victim.grad = torch.ones_like(self._victim)
                else:
                    self._victim.grad += torch.ones_like(self._victim)
            raise torch.cuda.OutOfMemoryError("simulated: out of memory inside backward")

    victim = next(iter(model.parameters()))
    fired = {"n": 0}

    def oom_in_backward_once(m, e, chunk, *args):
        parts, targets = original(m, e, chunk, *args)
        if len(chunk) == len(raws) and fired["n"] == 0:
            fired["n"] += 1
            return _PartialBackward(parts, victim), targets
        return parts, targets

    splits, dq = run(model, enc, oom_in_backward_once)
    assert fired["n"] == 1, "the simulated failure never fired"
    assert splits >= 1
    assert list(dq["a"]) == pytest.approx(list(ref_dq["a"]), abs=1e-5)
    for p_, q_ in zip(list(model.parameters()) + list(enc.parameters()),
                      list(ref_model.parameters()) + list(ref_enc.parameters())):
        assert torch.allclose(p_, q_, atol=1e-6, rtol=1e-4), \
            "the retry double-counted the gradients the failed backward left behind"
