"""The sequence trainer's optimizers over whole tiny passes (the pass_inputs
fixture: tests/conftest.py): the default trainer trains the weights the
trainer trained before `--optimizer` existed, and a weight average leaves
them as they are; Muon, SOAP and schedule-free AdamW each end a pass cut and
resumed with the uncut pass's weights, averages and optimizer state; a
resume refuses another optimizer; schedule-free's checkpoint and probe read
its average, not its training point; a finished pass's resume reads its
average (the optimizer bench's EMA readings)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

pytestmark = pytest.mark.slow


def _trainer(pass_inputs, out_dir, *extra):
    from tools import sequence_train
    out_dir.mkdir(parents=True, exist_ok=True)
    args = sequence_train.parse_args([*pass_inputs, "--probe-every", "1000000", "--barrier-positions",
                                      "1000000", "--signal-every", "3", "--window", "1", "--out",
                                      str(out_dir / "arm.pt"), *extra])
    torch.manual_seed(args.seed)
    return sequence_train.Trainer(args, torch.device("cpu"))


def _weights(trainer):
    return {f"{space}.{k}": v.detach().clone() for space, m in (("model", trainer.model), ("encoder", trainer.encoder))
            for k, v in m.state_dict().items()}


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def test_the_default_trains_the_old_adamw_s_weights_and_an_average_changes_none(tmp_path, pass_inputs):
    """The old trainer's optimizer line, `torch.optim.AdamW(self.params,
    lr=args.lr, weight_decay=args.weight_decay)`, set in place of the built
    one: the same weights to the bit after the pass; and a pass that keeps a
    weight average and probes it ends with the same live weights."""
    ends = {}
    for name, extra in (("default", ()), ("old", ()), ("averaged", ("--weight-ema", "0.9", "--probe-weights", "both"))):
        trainer = _trainer(pass_inputs, tmp_path / name, *extra)
        if name == "old":
            trainer.opt = torch.optim.AdamW(trainer.params, lr=trainer.args.lr, weight_decay=trainer.args.weight_decay)
            trainer.signal = trainer._signal_probe()
        assert trainer.run() == 0
        ends[name] = (trainer, _weights(trainer))
    assert ends["default"][0].state["steps"] >= 5
    for name in ("old", "averaged"):
        for key, w in ends["default"][1].items():
            assert torch.equal(w, ends[name][1][key]), (name, key)
    averaged = ends["averaged"][0]
    ck = torch.load(tmp_path / "averaged" / "arm.pt", map_location="cpu", weights_only=True)
    assert ck["ema_updates"] == averaged.state["steps"] and ck["training_meta"]["weight_ema"] == 0.9
    assert any(not torch.equal(ck["ema_model_state"][k], ck["model_state"][k]) for k in ck["model_state"])
    live = _rows(tmp_path / "averaged" / "arm.probe.jsonl")
    ema = _rows(tmp_path / "averaged" / "arm.ema.probe.jsonl")
    assert [r["weights"] for r in live] == ["live"] and [r["weights"] for r in ema] == ["ema"]
    assert live[-1]["final"] and ema[-1]["final"] and live[-1]["k8"]["ce_all"] != ema[-1]["k8"]["ce_all"]
    assert (tmp_path / "averaged" / "arm.ema.probe_sides.jsonl").exists()
    default_ck = torch.load(tmp_path / "default" / "arm.pt", map_location="cpu", weights_only=True)
    assert default_ck["training_meta"]["optimizer"] == {"name": "adamw"} and "ema_model_state" not in default_ck


@pytest.mark.parametrize("name, extra", [
    ("muon", ("--weight-ema", "0.9")),
    ("soap", ("--soap-precondition-every", "2")),
    ("schedulefree", ()),
])
def test_each_candidate_ends_a_cut_and_resumed_pass_as_the_uncut_one(tmp_path, pass_inputs, monkeypatch, name, extra):
    from tools import sequence_train
    common = [*pass_inputs, "--probe-every", "1000000", "--barrier-positions", "1000000", "--window", "1",
              "--optimizer", name, *extra]
    probed = []
    real_probe = sequence_train.probe

    def spy(model, *args, **kwargs):
        probed.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        return real_probe(model, *args, **kwargs)

    monkeypatch.setattr(sequence_train, "probe", spy)
    uncut, cut = tmp_path / "uncut.pt", tmp_path / "cut.pt"
    assert sequence_train.main([*common, "--out", str(uncut)]) == 0
    assert sequence_train.main([*common, "--out", str(cut), "--max-positions", "4"]) == 0
    with pytest.raises(SystemExit, match="was trained with optimizer="):
        sequence_train.main([*pass_inputs, "--out", str(cut), "--resume", "--probe-every", "1000000",
                             "--barrier-positions", "1000000", "--window", "1", "--optimizer", "adamw_hygiene"])
    assert sequence_train.main([*common, "--out", str(cut), "--resume"]) == 0
    a = torch.load(uncut, map_location="cpu", weights_only=True)
    b = torch.load(cut, map_location="cpu", weights_only=True)
    assert a["training_meta"]["optimizer"]["name"] == name
    assert a["sequence_resume"]["state"]["positions"] == b["sequence_resume"]["state"]["positions"] == 12
    keys = ["model_state", "encoder_state"] + (["ema_model_state", "ema_encoder_state"] if "--weight-ema" in extra
                                               else [])
    for key in keys:
        for k, tensor in a[key].items():
            assert torch.equal(tensor, b[key][k]), (key, k)
    _equal_states(a["optimizer_state"], b["optimizer_state"])
    if name == "schedulefree":
        # The checkpoint and the final probe hold the average x; a resume trains on from y.
        from wesnoth_ai.optimizers import evaluation_point
        resumed = sequence_train.build_trainer(sequence_train.parse_args([*common, "--out", str(uncut), "--resume"]))
        y = _weights(resumed)
        assert any(not torch.equal(y[f"model.{k}"], v) for k, v in a["model_state"].items()), "not y"
        with evaluation_point(resumed.opt):
            x = _weights(resumed)
        assert all(torch.equal(x[f"model.{k}"], v) for k, v in a["model_state"].items()), "x"
        assert all(torch.equal(probed[0][k], v) for k, v in a["model_state"].items()), "the probe read x"
    if "--weight-ema" in extra:
        assert sequence_train.main([*common, "--out", str(uncut), "--resume", "--probe-weights", "ema"]) == 0
        ema = _rows(uncut.with_suffix(".ema.probe_sides.jsonl"))
        assert len(ema) == 1 and ema[0]["final"] and ema[0]["weights"] == "ema"
        assert all(torch.equal(probed[-1][k], a["ema_model_state"][k]) for k in a["model_state"])


def _equal_states(x, y, where="optimizer_state"):
    if torch.is_tensor(x):
        assert torch.is_tensor(y) and torch.equal(x, y), where
    elif isinstance(x, dict):
        assert set(x) == set(y), where
        for k in x:
            _equal_states(x[k], y[k], f"{where}.{k}")
    elif isinstance(x, (list, tuple)):
        assert len(x) == len(y), where
        for i, (u, v) in enumerate(zip(x, y)):
            _equal_states(u, v, f"{where}[{i}]")
    else:
        assert x == y, (where, x, y)
