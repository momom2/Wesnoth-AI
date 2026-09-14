"""The training-only device knobs (wesnoth_ai/train_perf.py).

TF32 must reach the training paths and NOTHING else: the simulator, the
corpus sweeps and eval matches are compared across runs and must not
have their fp32 numerics changed underneath them. Both knobs are opt-in
(a recipe change is one factor with its own match); the optimizer
helper takes the fused kernel only on request and only when every
parameter is on CUDA, and a resumed optimizer keeps the kernel it was
built with rather than the checkpoint's.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from wesnoth_ai.train_perf import adamw, enable_tf32, reassert_step_kernel, tf32_training  # noqa: E402


def test_enable_tf32_reports_the_previous_setting_and_is_a_noop_without_cuda():
    cudnn_before = bool(torch.backends.cudnn.allow_tf32)
    before = enable_tf32(True, quiet=True)
    assert set(before) == {"matmul"}
    assert all(isinstance(v, bool) for v in before.values())
    if not torch.cuda.is_available():
        # No CUDA: nothing may change, so restoring is trivially exact.
        assert torch.backends.cuda.matmul.allow_tf32 == before["matmul"]
        return
    assert torch.backends.cuda.matmul.allow_tf32 is True
    enable_tf32(False, quiet=True)
    assert torch.backends.cuda.matmul.allow_tf32 is False
    enable_tf32(before["matmul"], quiet=True)
    assert bool(torch.backends.cudnn.allow_tf32) == cudnn_before, "cuDNN's switch is not ours"


def test_adamw_steps_and_declines_the_fused_kernel_off_cuda():
    p = torch.nn.Parameter(torch.zeros(4))
    opt = adamw([p], lr=1e-3, weight_decay=1e-4, fused=True)
    assert not opt.param_groups[0].get("fused", False), \
        "a CPU parameter must not take the fused path even when asked"
    p.grad = torch.ones(4)
    opt.step()
    assert torch.all(p.detach() < 0), "the optimizer must actually move the parameter"


def test_adamw_passes_lr_and_weight_decay_through():
    p = torch.nn.Parameter(torch.zeros(2))
    opt = adamw([p], lr=0.5, weight_decay=0.25)
    assert opt.param_groups[0]["lr"] == 0.5
    assert opt.param_groups[0]["weight_decay"] == 0.25


def test_adamw_is_the_reference_step_unless_asked():
    p = torch.nn.Parameter(torch.zeros(4))
    assert not adamw([p], lr=1e-3).param_groups[0].get("fused", False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_adamw_uses_the_fused_kernel_on_cuda_when_asked():
    p = torch.nn.Parameter(torch.zeros(4, device="cuda"))
    opt = adamw([p], lr=1e-3, fused=True)
    assert opt.param_groups[0].get("fused", False)


def test_a_resumed_optimizer_keeps_the_kernel_it_was_built_with():
    """`load_state_dict` takes the checkpoint's param groups, `fused`
    included: a checkpoint from a fused run resumed on a CPU (or an
    old checkpoint resumed into a fused run) would silently switch
    step kernels."""
    p = torch.nn.Parameter(torch.zeros(3))
    opt = adamw([p], lr=1e-3)
    state = opt.state_dict()
    state["param_groups"][0]["fused"] = True         # what a fused run's checkpoint carries
    opt.load_state_dict(state)
    assert opt.param_groups[0]["fused"] is True, "torch takes the checkpoint's groups verbatim"
    reassert_step_kernel(opt)
    assert not opt.param_groups[0]["fused"]
    p.grad = torch.ones(3)
    opt.step()                                        # a CPU step under the reference kernel


def test_tf32_training_restores_what_it_found():
    """az_loop serves the actors' inference from the SAME process as the
    learner, so the switch must not outlive the learner's step."""
    before = bool(torch.backends.cuda.matmul.allow_tf32)
    with tf32_training():
        if torch.cuda.is_available():
            assert torch.backends.cuda.matmul.allow_tf32 is True
    assert bool(torch.backends.cuda.matmul.allow_tf32) == before
    with tf32_training(False):
        assert bool(torch.backends.cuda.matmul.allow_tf32) == before, "off means untouched"


def test_tf32_training_restores_even_when_the_block_raises():
    before = bool(torch.backends.cuda.matmul.allow_tf32)
    with pytest.raises(RuntimeError):
        with tf32_training():
            raise RuntimeError("the learner step failed")
    assert bool(torch.backends.cuda.matmul.allow_tf32) == before


def test_the_sim_and_eval_paths_never_enable_tf32():
    """The knob is training-only by construction: no module outside the
    trainers may call it."""
    root = Path(__file__).parent.parent
    callers = set()
    files = [py for sub in ("wesnoth_ai", "tools", "tools/analysis", "scripts")
             for py in (root / sub).glob("*.py")]
    for py in files:
        if py.name in ("train_perf.py",):
            continue
        text = py.read_text(encoding="utf-8", errors="replace")
        if "enable_tf32" in text or "tf32_training" in text or "allow_tf32" in text:
            callers.add(py.name)
    assert callers <= {"supervised_train.py", "az_loop.py"}, \
        f"TF32 reached a non-training module: {sorted(callers)}"
