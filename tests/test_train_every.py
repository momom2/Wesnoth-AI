"""`--train-every K` (tools/sequence_train.py): one window in K trains, the
others run the forward only. At a learning rate of 0 the forward-only
windows read the same losses and carry the same memories as trained ones,
to the bit; at the recipe's rate the weights move on trained windows only,
every window's positions count toward the pass, the steps count the trained
windows, the steps file marks the others, and a resumed pass keeps the
alternation."""
from __future__ import annotations

import json

import pytest
import torch
from test_sequence_launches import QUIET, _trainer

from tools.bench_sequence_step import set_dropout


def _pass(pass_inputs, tmp_path, *extra, rate_zero=False):
    """A whole pass in windows of two time steps, dropout 0: per window its
    result, whether it computed gradients, the weights after it and the
    carried memories."""
    trainer = _trainer(pass_inputs, tmp_path, "--window", "2", *extra)
    set_dropout(trainer, 0.0)
    if rate_zero:
        trainer._lr_now = lambda: 0.0
    with_gradients, zero_grad = set(), trainer.opt.zero_grad

    def hooked(*a, **kw):
        if any(p.grad is not None for p in trainer.params):
            with_gradients.add(trainer.state["windows"])
        return zero_grad(*a, **kw)
    trainer.opt.zero_grad = hooked
    out = []
    while True:
        index = trainer.state["windows"]
        result = trainer.train_window()
        if not result:
            return trainer, out
        out.append({"result": result, "trained": index in with_gradients,
                    "params": [p.detach().clone() for p in trainer.params],
                    "memories": {s: m.clone() for s, m in trainer.memories.items()}})


@pytest.mark.slow
@pytest.mark.parametrize("forward", [(), ("--packed-trunk", "--window-batched")], ids=["stepped", "batched"])
def test_a_forward_only_window_carries_what_a_trained_window_carries(tmp_path, pass_inputs, forward):
    _, every = _pass(pass_inputs, tmp_path / "every", *forward, rate_zero=True)
    _, half = _pass(pass_inputs, tmp_path / "half", "--train-every", "2", *forward, rate_zero=True)
    assert len(every) == len(half) >= 3
    assert [w["trained"] for w in half] == [i % 2 == 0 for i in range(len(half))]
    assert all(w["trained"] for w in every)
    for a, b in zip(every, half):
        assert a["result"]["loss"] == b["result"]["loss"]
        assert a["result"]["positions"] == b["result"]["positions"]
        assert a["memories"].keys() == b["memories"].keys()
        assert all(torch.equal(a["memories"][s], b["memories"][s]) for s in a["memories"])


@pytest.mark.slow
def test_the_weights_move_on_trained_windows_only_and_a_resume_keeps_the_alternation(tmp_path, pass_inputs):
    from tools import sequence_train
    trainer, half = _pass(pass_inputs, tmp_path / "half", "--train-every", "2")
    before = [p.detach().clone() for p in _trainer(pass_inputs, tmp_path / "fresh", "--window", "2").params]
    for w in half:
        moved = any(not torch.equal(a, b) for a, b in zip(before, w["params"]))
        assert moved == w["trained"]
        before = w["params"]
    assert trainer.state["steps"] == sum(w["trained"] for w in half) < len(half) == trainer.state["windows"]
    assert trainer.state["positions"] == sum(w["result"]["positions"] for w in half)
    rows = [json.loads(line) for line in (tmp_path / "half" / "arm.steps.jsonl").read_text().splitlines()]
    assert [bool(r.get("forward_only")) for r in rows] == [not w["trained"] for w in half]
    assert all(r["grad"] == {} and not r["applied"] for r in rows if r.get("forward_only"))

    out = tmp_path / "resumed.pt"
    common = [*pass_inputs, *QUIET, "--window", "2", "--train-every", "2", "--out", str(out)]
    assert sequence_train.main([*common, "--max-positions", "4", "--checkpoint-every", "1"]) == 0
    assert sequence_train.main([*common, "--resume"]) == 0
    rows = [json.loads(line) for line in out.with_suffix(".steps.jsonl").read_text().splitlines()]
    assert [bool(r.get("forward_only")) for r in rows] == [i % 2 == 1 for i in range(len(rows))]
