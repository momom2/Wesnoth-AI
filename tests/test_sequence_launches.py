"""The sequence trainer's launch-count work (docs/trainer_speed_20261013.md,
round 2): the memory batch gathered in one indexing trains what per-row
memories padded one by one train, to the bit."""
from __future__ import annotations

import pytest
import torch

from tools.bench_sequence_step import set_dropout

QUIET = ["--probe-every", "1000000", "--barrier-positions", "1000000", "--signal-every", "1000000"]


def _trainer(pass_inputs, tmp_path, *extra):
    from tools import sequence_train
    tmp_path.mkdir(parents=True, exist_ok=True)
    args = sequence_train.parse_args([*pass_inputs, *QUIET, "--out", str(tmp_path / "arm.pt"), *extra])
    return sequence_train.build_trainer(args)


class PerRowMemories:
    """The trainer's memory handling before the router: each row's state a
    slice of the step that wrote it, the batch padded by `SlotMemory.batch`."""

    def __init__(self, model, device, carried):
        self.model, self.device = model, device
        self.states = dict(carried)

    def batch(self, steps):
        return [self.model.initial_memory(s.k).to(self.device) if s.starts else self.states[s.slot]
                for s in steps]

    def wrote(self, steps, padded):
        for b, s in enumerate(steps):
            self.states[s.slot] = padded[b, :s.k]

    def memories(self):
        return self.states


def _windows(pass_inputs, tmp_path, *extra):
    """The whole pass in windows of two time steps, every dropout at 0.3:
    per window its returned values, its gradients before they are zeroed,
    the parameters after its step and the carried memories."""
    trainer = _trainer(pass_inputs, tmp_path, "--window", "2", *extra)
    set_dropout(trainer, 0.3)
    seen, gate, zero_grad = [], [], trainer.opt.zero_grad

    def hooked(*a, **kw):
        seen.append([None if p.grad is None else p.grad.clone() for p in trainer.params])
        gate.append(trainer.model.slot_memory.gate.weight.grad is not None)
        return zero_grad(*a, **kw)

    trainer.opt.zero_grad = hooked
    out = []
    while True:
        result = trainer.train_window()
        if not result:
            return out
        out.append({"result": result, "grads": seen[-1], "write_trained": gate[-1], "params": [p.detach().clone() for p in trainer.params],
                    "memories": {s: m.clone() for s, m in trainer.memories.items()}})


def _assert_same(a, b):
    for x, y in zip(a, b):
        assert x["result"] == y["result"]
        for g, h in zip(x["grads"], y["grads"]):
            assert (g is None) == (h is None) and (g is None or torch.equal(g, h))
        assert all(torch.equal(p, q) for p, q in zip(x["params"], y["params"]))
        assert x["memories"].keys() == y["memories"].keys()
        assert all(torch.equal(x["memories"][s], y["memories"][s]) for s in x["memories"])


@pytest.mark.slow
@pytest.mark.parametrize("own", [False, True], ids=["corpus", "own"])
@pytest.mark.parametrize("trunk", [(), ("--packed-trunk",)], ids=["padded", "packed"])
def test_the_routed_memories_train_what_per_row_memories_train(tmp_path, request, monkeypatch, own, trunk):
    """A whole pass in windows of two time steps, game-sides ending and
    starting inside them (the corpus's game-sides are three decisions long),
    on the corpus alone and with own game-sides on half the streams: the
    memory batch gathered in one indexing (tools/sequence_memory.py) gives
    the losses, every gradient, the weights after each step and the carried
    memories that per-row memories padded one by one give, to the bit; the
    memory's write untrained in a window where no step reads what an earlier
    one wrote, as it was."""
    from tools import sequence_train
    pass_inputs = request.getfixturevalue("own_pass_inputs" if own else "pass_inputs")
    routed = _windows(pass_inputs, tmp_path / "routed", *trunk)
    monkeypatch.setattr(sequence_train, "MemoryRouter", PerRowMemories)
    per_row = _windows(pass_inputs, tmp_path / "per_row", *trunk)
    assert len(routed) == len(per_row) >= 3
    _assert_same(routed, per_row)
    assert [w["write_trained"] for w in routed] == [w["write_trained"] for w in per_row]
    assert not all(w["write_trained"] for w in per_row), "a window in which no step reads an earlier write"
