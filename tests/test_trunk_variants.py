"""The trunk variants in the sequence trainer (docs/trainer_speed_20261013.md,
round 3; wesnoth_ai/trunk_modes.py): the fused ReLU trains what the packed
trunk trains to the bit on the CPU; the bf16 residual and the compiled trunk
train near it, kept or recomputed, stepped or batched; the flags need the
packed trunk; a pass resumes on its own variants; the gradient check reads
several candidates against one reference."""
from __future__ import annotations

import json

import pytest
import torch
from test_sequence_launches import QUIET, _assert_same, _rate_zero_pass, _windows
from test_sequence_speed import _bench


@pytest.mark.slow
@pytest.mark.parametrize("batched", [(), ("--window-batched",)], ids=["stepped", "batched"])
@pytest.mark.parametrize("own", [False, True], ids=["corpus", "own"])
def test_the_fused_relu_trains_what_the_packed_trunk_trains_to_the_bit(tmp_path, request, own, batched):
    """A whole pass in windows of two time steps: the losses, every gradient,
    the weights after each step and the carried memories, to the bit."""
    pass_inputs = request.getfixturevalue("own_pass_inputs" if own else "pass_inputs")
    base = _windows(pass_inputs, tmp_path / "base", "--packed-trunk", *batched)
    fused = _windows(pass_inputs, tmp_path / "fused", "--packed-trunk", "--fused-relu", *batched)
    assert len(base) == len(fused) >= 3
    _assert_same(base, fused)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.flatten().double(), b.flatten().double(), dim=0))


@pytest.mark.slow
@pytest.mark.parametrize("budget", ["0", "1000"], ids=["recomputed", "kept"])
@pytest.mark.parametrize("variant, loss_rel, min_cos", [
    (("--trunk-bf16",), 2e-2, 0.99),
    (("--compiled-trunk", "aot_eager"), 1e-6, 0.999999),
], ids=["bf16 residual", "compiled"])
def test_a_variant_trains_near_the_packed_trunk(tmp_path, own_pass_inputs, budget, variant, loss_rel, min_cos):
    """The production path (own game-sides on half the streams, the window
    batched) from the same weights at rate 0, every time step recomputed in
    the backward pass or every one kept: each window's loss and gradients
    near the float32 packed trunk's; the bf16 residual not equal to it."""
    base = _rate_zero_pass(own_pass_inputs, tmp_path / "base", budget, "--window-batched")
    other = _rate_zero_pass(own_pass_inputs, tmp_path / "variant", budget, "--window-batched", *variant)
    assert len(base) == len(other) >= 2
    exact = True
    for (ra, ga, _), (rb, gb, _) in zip(base, other):
        assert rb["loss"] == pytest.approx(ra["loss"], rel=loss_rel)
        trained = 0
        for x, y in zip(ga, gb):
            assert (x is None) == (y is None)
            if x is not None and x.abs().max() > 1e-6:
                assert _cosine(x, y) >= min_cos
                trained += 1
                exact = exact and torch.equal(x, y)
        assert trained > 10
    if "--trunk-bf16" in variant:
        assert not exact


def test_the_variants_need_the_packed_trunk_and_the_fused_relu_is_not_compiled(tmp_path):
    from tools import sequence_train
    common = ["--sequences", str(tmp_path), "--dataset", str(tmp_path), "--device", "cpu",
              "--out", str(tmp_path / "x.pt")]
    for extra, match in ((["--trunk-bf16"], "--packed-trunk"), (["--compiled-trunk"], "--packed-trunk"),
                         (["--packed-trunk", "--fused-relu", "--compiled-trunk"], "not both")):
        with pytest.raises(SystemExit, match=match):
            sequence_train.build_trainer(sequence_train.parse_args(common + extra))
    assert sequence_train.parse_args(common + ["--compiled-trunk"]).compiled_trunk == "inductor"
    assert sequence_train.parse_args(common + ["--compiled-trunk", "aot_eager"]).compiled_trunk == "aot_eager"


@pytest.mark.slow
def test_a_pass_resumes_on_its_own_trunk_variants(tmp_path, pass_inputs):
    from tools import sequence_train
    common = [*pass_inputs, *QUIET, "--packed-trunk"]
    first = tmp_path / "first.pt"
    assert sequence_train.main([*common, "--fused-relu", "--out", str(first), "--max-positions", "4"]) == 0
    assert torch.load(first, map_location="cpu", weights_only=True)["training_meta"]["trunk_mode"] == "fused_relu"
    with pytest.raises(SystemExit, match="trunk_mode"):
        sequence_train.main([*common, "--out", str(first), "--resume"])
    assert sequence_train.main([*common, "--fused-relu", "--out", str(first), "--resume",
                                "--max-positions", "8"]) == 0
    older = torch.load(first, map_location="cpu", weights_only=True)
    del older["training_meta"]["trunk_mode"]                 # written before the variants were recorded
    torch.save(older, first)
    with pytest.raises(SystemExit, match="trunk_mode='none'"):
        sequence_train.main([*common, "--fused-relu", "--out", str(first), "--resume"])


@pytest.mark.slow
def test_the_gradient_check_reads_every_candidate_against_one_reference(tmp_path, pass_inputs):
    """The fused ReLU's gradients are the packed trunk's (cosines of 1), the
    bf16 residual's pass the check without being them; each candidate gets
    its own verdict and steps file."""
    done = _bench("gradcheck", "--out", str(tmp_path / "check.json"), "--windows", "2",
                  "--reference=--packed-trunk", "--candidate=--packed-trunk --fused-relu",
                  "--candidate=--packed-trunk --trunk-bf16", "--",
                  *pass_inputs, *QUIET, "--out", str(tmp_path / "arm.pt"))
    assert done.returncode == 0, done.stderr[-2000:]
    check = json.loads((tmp_path / "check.json").read_text(encoding="utf-8"))
    fused, bf16 = check["candidates"]
    assert check["pass"] and fused["pass"] and bf16["pass"]
    assert fused["candidate"] == "--packed-trunk --fused-relu" and len(fused["windows"]) == 2
    assert all(c == 1.0 for w in fused["windows"] for c in w["cosine"].values() if c is not None)
    assert all(w["loss_rel"] == 0 for w in fused["windows"])
    cos = [c for w in bf16["windows"] for c in w["cosine"].values() if c is not None]
    assert len(cos) > 5 and 0.99 < min(cos) < 1.0
    assert (tmp_path / "arm_candidate0.steps.jsonl").exists() and (tmp_path / "arm_candidate1.steps.jsonl").exists()


@pytest.mark.slow
def test_a_whole_pass_on_the_variants_writes_its_telemetry_and_its_probe(tmp_path, own_pass_inputs):
    """The production path with the bf16 residual and the compiled trunk, run
    by main: the telemetry rows (a backward pass per loss term through the
    compiled blocks, every 3 positions) and the final probe (the padded eval
    trunk) are written, their numbers finite."""
    import math
    from tools import sequence_train
    out = tmp_path / "variants.pt"
    assert sequence_train.main([*own_pass_inputs, "--out", str(out), "--packed-trunk", "--window-batched",
                                "--trunk-bf16", "--compiled-trunk", "aot_eager"]) == 0
    signal = [json.loads(line) for line in out.with_suffix(".signal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(signal) >= 2
    numbers = [v for row in signal for v in _numbers(row["gradient"])]
    assert len(numbers) > 10 and all(math.isfinite(v) for v in numbers)
    probe = [json.loads(line) for line in out.with_suffix(".probe.jsonl").read_text(encoding="utf-8").splitlines()]
    assert probe[-1]["final"] and math.isfinite(probe[-1]["k0"]["ce_all"]) and probe[-1]["k0"]["n_nonfinite"] == 0
    meta = torch.load(out, map_location="cpu", weights_only=True)["training_meta"]
    assert meta["trunk_mode"] == "bf16_residual+compiled:aot_eager"


def _numbers(obj):
    """Every number in a nested row."""
    if isinstance(obj, dict):
        return [v for x in obj.values() for v in _numbers(x)]
    if isinstance(obj, (list, tuple)):
        return [v for x in obj for v in _numbers(x)]
    return [float(obj)] if isinstance(obj, (int, float)) and not isinstance(obj, bool) else []
