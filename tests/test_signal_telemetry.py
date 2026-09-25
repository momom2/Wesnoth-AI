"""The imitation trainer's signal telemetry (tools/signal_telemetry.py
`ImitationSignal`): the loss terms it splits the gradient into add up
to the gradient the trainer steps on, the probe leaves training bit for
bit as it was, and a failed probe costs its row, not the run.
tests/test_supervised_resume.py checks the rows through `train()`: a
cut and resumed run records the uncut run's readings."""
import copy
import json
import logging
import math
import random
from collections import deque

import pytest
import torch

from imitation_helpers import ARCH, TYPE_W, labels, states
from tools import supervised_train as st
from tools.signal_telemetry import (
    IMITATION_SOURCES, POLICY_SOURCES, SIGNAL_GROUPS, ImitationSignal, signal_group,
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


def _signal(path, model, enc, **kw) -> ImitationSignal:
    return ImitationSignal(path, lambda r, a, z: st._batch_loss(model, enc, r, a, z, DEV, TYPE_W),
                           model, enc, **kw)


def _params(model, enc):
    return list(model.parameters()) + list(enc.parameters())


def _trained_gradient_norms(model, enc, raws, ais, zw):
    """Per telemetry group, the norm of the gradient the trainer's own
    accumulation (`_accumulate_batch`) leaves for these pairs as one
    batch, before the clip."""
    opt = torch.optim.SGD(_params(model, enc), lr=0.0)
    st._accumulate_batch(model, enc, raws, ais, zw, len(raws), DEV, TYPE_W, [], opt)
    squares = dict.fromkeys(SIGNAL_GROUPS, 0.0)
    named = ([("model." + n, p) for n, p in model.named_parameters()]
             + [("encoder." + n, p) for n, p in enc.named_parameters()])
    for name, p in named:
        if p.grad is not None:
            squares[signal_group(name)] += float(p.grad.double().pow(2).sum())
    opt.zero_grad(set_to_none=True)
    return {g: math.sqrt(v) for g, v in squares.items()}


def test_the_terms_add_up_to_the_gradient_the_trainer_steps_on(tmp_path, game_states):
    """Probing a whole batch: the summed terms' gradient is the trainer's,
    the value term is the trainer's with the policy silenced, and the
    four policy terms together are the trainer's with the value
    silenced, in every group."""
    model, enc, raws = _net(game_states, 3)
    model.eval()
    enc.eval()                    # no dropout: the two paths must agree exactly
    ais, zw = labels(raws, random.Random(5))
    signal = _signal(tmp_path / "s.jsonl", model, enc, every=1, probe_pairs=len(raws))
    row = signal.record((raws, ais, zw), epoch=0, step=1, pairs=len(raws))
    assert "probe_error" not in row and row["probe_pairs"] == len(raws)

    trained = _trained_gradient_norms(model, enc, raws, ais, zw)
    value_only = _trained_gradient_norms(model, enc, raws, ais, [(z, vw, 0.0) for z, vw, _ in zw])
    policy_only = _trained_gradient_norms(model, enc, raws, ais, [(z, 0.0, pw) for z, _, pw in zw])
    policy = [IMITATION_SOURCES.index(s) for s in POLICY_SOURCES]
    for group in SIGNAL_GROUPS:
        node, gram = row["groups"][group], row["gram"][group]
        assert min(trained[group], value_only[group], policy_only[group]) > 0, group
        assert node["total_norm"] == pytest.approx(trained[group], rel=1e-4), group
        assert node["value"]["norm"] == pytest.approx(value_only[group], rel=1e-4), group
        policy_norm = math.sqrt(sum(gram[a][b] for a in policy for b in policy))
        assert policy_norm == pytest.approx(policy_only[group], rel=1e-4), group
    total = row["groups"]["all"]["total_norm"]
    assert total == pytest.approx(math.sqrt(sum(v * v for v in trained.values())), rel=1e-4)


def test_the_probe_leaves_training_bit_identical(tmp_path, game_states):
    """Three steps of the batched flow with and without a probe after
    each: the weights come out bit for bit the same. The model draws
    dropout masks at 0.1 here, and one extra draw of torch's generator
    between the steps does change them, so a probe that moved the
    generator would show."""
    model, enc, raws = _net(game_states, 4, dropout=0.1)
    ais, zw = labels(raws, random.Random(6))
    batches = [(raws[i:i + 4], ais[i:i + 4], zw[i:i + 4]) for i in (0, 4, 2)]

    def run(tag: str, probe: bool = False, draw: bool = False):
        m, e = copy.deepcopy(model), copy.deepcopy(enc)
        m.train()
        e.train()
        opt = torch.optim.Adam(_params(m, e), lr=1e-3)
        signal = _signal(tmp_path / f"{tag}.jsonl", m, e, every=1, probe_pairs=3)
        torch.manual_seed(99)
        for k, (r, a, z) in enumerate(batches):
            st._flush_batch(m, e, r, a, z, opt, _params(m, e), len(r), DEV,
                            *[deque(maxlen=50) for _ in range(6)],
                            type_loss_weights=TYPE_W, step_norms=signal.step_norms)
            if probe:
                signal.record((r, a, z), epoch=0, step=k + 1, pairs=4 * (k + 1))
            if draw:
                torch.rand(1)
        return [p.detach().clone() for p in _params(m, e)], signal

    plain, _ = run("plain")
    probed, signal = run("probed", probe=True)
    drawn, _ = run("drawn", draw=True)
    assert all(torch.equal(a, b) for a, b in zip(plain, probed))
    assert not all(torch.equal(a, b) for a, b in zip(plain, drawn))

    rows = [json.loads(line) for line in signal.path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == signal.rows == 3 and signal.failures == 0
    for row in rows:
        assert row["probe_pairs"] == 3
        assert row["steps"]["n"] == 1 and row["steps"]["nonfinite"] == 0
        assert row["steps"]["mean"] > 0 and 0.0 <= row["steps"]["clipped"] <= 1.0
        for group in (*SIGNAL_GROUPS, "all"):
            node = row["groups"][group]
            assert sum(node[s]["share"] for s in IMITATION_SOURCES) == pytest.approx(1.0, abs=1e-6)
            assert -1.0 - 1e-9 <= node["policy_value_cos"] <= 1.0 + 1e-9


def test_a_failed_probe_costs_its_row_not_the_run(tmp_path, game_states, caplog):
    model, enc, raws = _net(game_states, 5)
    ais, zw = labels(raws, random.Random(7))

    def broken(*_):
        raise RuntimeError("probe broke")

    signal = ImitationSignal(tmp_path / "s.jsonl", broken, model, enc, every=1)
    with caplog.at_level(logging.WARNING, logger="signal_telemetry"):
        row = signal.record((raws, ais, zw), epoch=0, step=7, pairs=64)
    assert signal.failures == 1 and signal.rows == 1
    assert "probe broke" in row["probe_error"] and "groups" not in row
    assert "signal probe failed at step 7" in caplog.text
