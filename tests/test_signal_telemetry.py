"""The imitation trainer's signal telemetry (tools/signal_telemetry.py):
each loss term's gradient, and its image in the optimizer's update
space, as the rows record them; the probe leaves training bit for bit as
it was; a failure costs its row, not the run; a resumed run's file reads
as one run. tests/test_supervised_resume.py checks the rows through
`train()`: a cut and resumed run records the uncut run's readings."""
import copy
import json
import logging
import math
import random
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from helpers.imitation import ARCH, TYPE_W, labels, per_sample_reference, states
from tools import supervised_train as st
from tools import signal_telemetry as sig
from tools.signal_telemetry import (
    IMITATION_SOURCES, POLICY_SOURCES, SIGNAL_GROUPS, GradientProbe, ImitationSignal,
    named_model_parameters, read_signal_rows, signal_group,
)
from wesnoth_ai.encoder import GameStateEncoder, encode_raw
from wesnoth_ai.model import WesnothModel

DEV = torch.device("cpu")


@pytest.fixture(scope="module")
def game_states():
    return states(8)


def _net(game_states, seed: int, dropout: float = 1e-4):
    torch.manual_seed(seed)
    enc = GameStateEncoder(d_model=ARCH["d_model"])
    model = WesnothModel(**ARCH, dropout=dropout)
    for s in game_states:
        enc.register_names(s)
    raws = [encode_raw(s, type_to_id=enc.unit_type_to_id, faction_to_id=enc.faction_to_id)
            for s in game_states]
    return model, enc, raws


def _weighted(zw):
    """The recipe's per-game weights run from about 0.25 to 4: give every
    policy and value weight of the helper's labels a different size."""
    policy = (0.25, 0.5, 2.0, 4.0)
    value = (0.35, 0.7, 1.4)
    return [(z, vw * value[i % 3] / 0.7 if vw else vw, policy[i % 4] if pw else pw)
            for i, (z, vw, pw) in enumerate(zw)]


def _signal(path, model, enc, **kw) -> ImitationSignal:
    return ImitationSignal(path, lambda r, a, z: st._batch_loss(model, enc, r, a, z, DEV, TYPE_W),
                           model, enc, **kw)


def _params(model, enc):
    return list(model.parameters()) + list(enc.parameters())


def _probe_rows(path):
    return [r for r in read_signal_rows(path) if r["kind"] == "probe"]


def _reference_grams(model, enc, raws, ais, zw, optimizer=None):
    """Per group, the Gram matrix of the five terms' gradients built from
    the per-sample loss path (tests/test_imitation_flat_batch.py holds it
    equal to the batched one), in gradient space and, given an optimizer
    with state, as the step AdamW takes for it: times lr over its
    bias-corrected scale."""
    parts, _ = per_sample_reference(model, enc, raws, ais, zw, DEV)
    terms = {
        "actor": sum(w * p.actor for p, (_, _, w) in zip(parts, zw)),
        "type": sum(w * p.type for p, (_, _, w) in zip(parts, zw)),
        "target": sum(w * p.target for p, (_, _, w) in zip(parts, zw)),
        "weapon": sum(w * p.weapon for p, (_, _, w) in zip(parts, zw)),
        "value": sum(w * p.value for p, (_, w, _) in zip(parts, zw)),
    }
    named = named_model_parameters(model, enc)
    params = [p for _, p in named]
    flat = {group: [[] for _ in IMITATION_SOURCES] for group in SIGNAL_GROUPS}
    scaled = {group: [[] for _ in IMITATION_SOURCES] for group in SIGNAL_GROUPS}
    for i, term in enumerate(IMITATION_SOURCES):
        grads = torch.autograd.grad(terms[term] / len(raws), params, retain_graph=True,
                                    allow_unused=True)
        for (name, p), g in zip(named, grads):
            g = torch.zeros_like(p) if g is None else g
            flat[signal_group(name)][i].append(g.reshape(-1).double())
            st_p = optimizer.state.get(p) if optimizer is not None else None
            if st_p:
                hyper = optimizer.param_groups[0]
                beta2, eps, lr = hyper["betas"][1], hyper["eps"], hyper["lr"]
                v_hat = st_p["exp_avg_sq"].double() / (1 - beta2 ** float(st_p["step"]))
                scaled[signal_group(name)][i].append((g.double() * lr / (v_hat.sqrt() + eps)).reshape(-1))

    def gram(vectors):
        out = {}
        for group, rows in vectors.items():
            m = torch.stack([torch.cat(r) for r in rows])
            out[group] = (m @ m.T).tolist()
        return out
    return gram(flat), (gram(scaled) if optimizer is not None else None)


def _assert_gram_close(got, want):
    """Entry by entry, within 1e-3 of the Cauchy-Schwarz bound
    sqrt(G_ii G_jj), so that small terms are checked as closely as
    large ones."""
    for i, row_want in enumerate(want):
        for j, b in enumerate(row_want):
            bound = math.sqrt(abs(want[i][i]) * abs(want[j][j]))
            assert abs(got[i][j] - b) <= 1e-3 * bound + 1e-15, (i, j, got[i][j], b)


def test_each_term_is_the_gradient_the_trainer_steps_on(tmp_path, game_states):
    """Probing a whole batch at the recipe's spread of policy and value
    weights: every entry of the five terms' Gram matrix, in every group,
    in gradient space and in update space, matches the per-sample loss
    path; and the terms add up to the gradient the trainer's own
    accumulation leaves before the clip."""
    model, enc, raws = _net(game_states, 3)
    model.eval()
    enc.eval()                    # no dropout: the two paths must agree exactly
    ais, zw = labels(raws, random.Random(5))
    zw = _weighted(zw)
    opt = torch.optim.AdamW(_params(model, enc), lr=1e-3, weight_decay=1e-4)
    st._flush_batch(model, enc, raws[:4], ais[:4], zw[:4], opt, _params(model, enc), 4, DEV,
                    *[deque(maxlen=50) for _ in range(6)], type_loss_weights=TYPE_W)

    signal = _signal(tmp_path / "s.jsonl", model, enc, optimizer=opt, every=1,
                     probe_pairs=len(raws))
    row = signal.record((raws, ais, zw), epoch=0, step=1, pairs=len(raws))
    assert "probe_error" not in row and row["probe_pairs"] == len(raws)
    assert "update_stateless" not in row
    # A policy term covers the pairs whose actor is in the sample and whose
    # policy weight is not zero; the value-only pair counts for value alone.
    in_sample = [ai.actor_idx < len(r.unit_ids) + len(r.recruit_types) + 1 for r, ai in zip(raws, ais)]
    assert row["fired"]["actor"] == sum(ok and pw > 0 for ok, (_, _, pw) in zip(in_sample, zw))
    assert row["fired"]["value"] == sum(ok and z is not None and vw > 0
                                        for ok, (z, vw, _) in zip(in_sample, zw))
    assert 0 < row["fired"]["actor"] < sum(in_sample) and row["fired"]["weapon"] > 0

    gradient, update = _reference_grams(model, enc, raws, ais, zw, optimizer=opt)
    for group in SIGNAL_GROUPS:
        _assert_gram_close(row["gradient_gram"][group], gradient[group])
        _assert_gram_close(row["update_gram"][group], update[group])

    squares = dict.fromkeys(SIGNAL_GROUPS, 0.0)
    trainer = torch.optim.SGD(_params(model, enc), lr=0.0)
    st._accumulate_batch(model, enc, raws, ais, zw, len(raws), DEV, TYPE_W, [], trainer)
    for name, p in named_model_parameters(model, enc):
        if p.grad is not None:
            squares[signal_group(name)] += float(p.grad.double().pow(2).sum())
    for group in SIGNAL_GROUPS:
        assert squares[group] > 0
        assert row["gradient"][group]["total_norm"] ** 2 == pytest.approx(squares[group], rel=1e-4)
    policy = [IMITATION_SOURCES.index(s) for s in POLICY_SOURCES]
    trunk = row["gradient_gram"]["trunk"]
    assert sum(trunk[a][b] for a in policy for b in policy) > 0 and trunk[4][4] > 0

    # A parameter never stepped is left out of the update space, and said so.
    largest_trunk = max((p for name, p in named_model_parameters(model, enc)
                         if signal_group(name) == "trunk" and opt.state.get(p)),
                        key=lambda p: p.numel())
    del opt.state[largest_trunk]
    partial = signal.record((raws, ais, zw), epoch=0, step=2, pairs=2 * len(raws))
    assert partial["update_stateless"] == {"trunk": 1}


def test_the_probe_leaves_training_bit_identical(tmp_path, game_states):
    """Three steps of the batched flow with and without a probe after
    each: the weights come out bit for bit the same and the global
    `random` is untouched. The model draws dropout masks at 0.1 here,
    and one extra draw of torch's generator between the steps does
    change the weights, so a probe that moved the generator would show."""
    model, enc, raws = _net(game_states, 4, dropout=0.1)
    ais, zw = labels(raws, random.Random(6))
    batches = [(raws[i:i + 4], ais[i:i + 4], zw[i:i + 4]) for i in (0, 4, 2)]

    def run(tag: str, probe: bool = False, draw: bool = False):
        m, e = copy.deepcopy(model), copy.deepcopy(enc)
        m.train()
        e.train()
        opt = torch.optim.AdamW(_params(m, e), lr=1e-3, weight_decay=1e-4)
        signal = _signal(tmp_path / f"{tag}.jsonl", m, e, optimizer=opt, every=1, probe_pairs=3)
        torch.manual_seed(99)
        python_state = random.getstate()
        for k, (r, a, z) in enumerate(batches):
            st._flush_batch(m, e, r, a, z, opt, _params(m, e), len(r), DEV,
                            *[deque(maxlen=50) for _ in range(6)],
                            type_loss_weights=TYPE_W, step_norms=signal.step_norms)
            if probe:
                signal.record((r, a, z), epoch=0, step=k + 1, pairs=4 * (k + 1))
            if draw:
                torch.rand(1)
        assert random.getstate() == python_state
        return [p.detach().clone() for p in _params(m, e)], signal

    plain, _ = run("plain")
    probed, signal = run("probed", probe=True)
    drawn, _ = run("drawn", draw=True)
    assert all(torch.equal(a, b) for a, b in zip(plain, probed))
    assert not all(torch.equal(a, b) for a, b in zip(plain, drawn))

    rows = _probe_rows(signal.path)
    assert len(rows) == 3 and signal.failures == 0
    for row in rows:
        assert row["probe_pairs"] == 3
        assert row["steps"]["n"] == 1 and row["steps"]["nonfinite"] == 0
        assert row["steps"]["mean"] > 0 and 0.0 <= row["steps"]["clipped"] <= 1.0
        for space in ("gradient", "update"):
            for group in (*SIGNAL_GROUPS, "all"):
                node = row[space][group]
                shares = [node[s]["share"] for s in IMITATION_SOURCES]
                assert sum(shares) == pytest.approx(1.0, abs=1e-6), (space, group)
            assert "policy_value_cos" not in row[space]["heads"]


def test_a_failed_probe_costs_its_row_not_the_run(tmp_path, game_states, caplog):
    model, enc, raws = _net(game_states, 5)
    ais, zw = labels(raws, random.Random(7))

    def broken(*_):
        raise RuntimeError("probe broke")

    signal = ImitationSignal(tmp_path / "s.jsonl", broken, model, enc, every=1)
    with caplog.at_level(logging.WARNING, logger="signal_telemetry"):
        row = signal.record((raws, ais, zw), epoch=0, step=7, pairs=64)
    assert signal.failures == 1
    assert "probe broke" in row["probe_error"] and "gradient" not in row
    assert "signal telemetry failed at step 7" in caplog.text
    assert [r["kind"] for r in read_signal_rows(signal.path)] == ["start", "probe"]


def test_the_probe_forks_the_generators_of_the_parameters_devices(monkeypatch):
    """A model on cuda:1 while the current device is 0: the probe's
    dropout draws must come from a fork of cuda:1's generator."""
    forked = []
    monkeypatch.setattr(sig.torch.random, "fork_rng", lambda devices: forked.append(devices))
    on_one = SimpleNamespace(requires_grad=True, is_cuda=True, device=torch.device("cuda", 1))
    GradientProbe([("model.x", on_one), ("encoder.y", on_one)], signal_group, SIGNAL_GROUPS).fork_rng()
    on_cpu = SimpleNamespace(requires_grad=True, is_cuda=False, device=torch.device("cpu"))
    GradientProbe([("model.x", on_cpu)], signal_group, SIGNAL_GROUPS).fork_rng()
    assert forked == [[1], []]


def test_a_resumed_runs_file_reads_as_one_run(tmp_path):
    """Rows written past the checkpoint a run resumed from are dropped
    when the file is read; the rest keep their order."""
    path = tmp_path / "s.jsonl"
    rows = [{"kind": "start", "pairs": 0}, {"kind": "probe", "pairs": 100},
            {"kind": "probe", "pairs": 200}, {"kind": "steps", "pairs": 230},
            {"kind": "start", "pairs": 150}, {"kind": "probe", "pairs": 200},
            {"kind": "probe", "pairs": 300}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"kind": "pro',
                    encoding="utf-8")        # the trainer killed mid-write
    assert [(r["kind"], r["pairs"]) for r in read_signal_rows(path)] == [
        ("start", 0), ("probe", 100), ("start", 150), ("probe", 200), ("probe", 300)]
