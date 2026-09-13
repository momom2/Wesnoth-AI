"""The training-only device knobs (wesnoth_ai/train_perf.py).

TF32 must reach the training paths and NOTHING else: the simulator, the
corpus sweeps and eval matches are compared across runs and must not
have their fp32 numerics changed underneath them. The optimizer helper
must ask for the fused kernel only when every parameter is on CUDA, and
must never fail on a CPU or DirectML parameter set.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from wesnoth_ai.train_perf import adamw, enable_tf32  # noqa: E402


def test_enable_tf32_reports_the_previous_setting_and_is_a_noop_without_cuda():
    before = enable_tf32(True, quiet=True)
    assert set(before) == {"matmul", "cudnn"}
    assert all(isinstance(v, bool) for v in before.values())
    if not torch.cuda.is_available():
        # No CUDA: nothing may change, so restoring is trivially exact.
        assert torch.backends.cuda.matmul.allow_tf32 == before["matmul"]
        return
    assert torch.backends.cuda.matmul.allow_tf32 is True
    enable_tf32(False, quiet=True)
    assert torch.backends.cuda.matmul.allow_tf32 is False
    enable_tf32(before["matmul"], quiet=True)


def test_adamw_steps_and_declines_the_fused_kernel_off_cuda():
    p = torch.nn.Parameter(torch.zeros(4))
    opt = adamw([p], lr=1e-3, weight_decay=1e-4)
    assert not opt.param_groups[0].get("fused", False), \
        "a CPU parameter must not take the fused path"
    p.grad = torch.ones(4)
    opt.step()
    assert torch.all(p.detach() < 0), "the optimizer must actually move the parameter"


def test_adamw_passes_lr_and_weight_decay_through():
    p = torch.nn.Parameter(torch.zeros(2))
    opt = adamw([p], lr=0.5, weight_decay=0.25)
    assert opt.param_groups[0]["lr"] == 0.5
    assert opt.param_groups[0]["weight_decay"] == 0.25


def test_adamw_on_an_empty_parameter_list_does_not_ask_for_fused():
    with pytest.raises(ValueError):
        adamw([], lr=1e-3)          # torch rejects an empty list, fused or not


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_adamw_uses_the_fused_kernel_on_cuda():
    p = torch.nn.Parameter(torch.zeros(4, device="cuda"))
    opt = adamw([p], lr=1e-3)
    assert opt.param_groups[0].get("fused", False)


def test_the_sim_and_eval_paths_never_enable_tf32():
    """The knob is training-only by construction: no module outside the
    trainers may call it."""
    root = Path(__file__).parent.parent
    callers = set()
    for py in list((root / "wesnoth_ai").glob("*.py")) + list((root / "tools").glob("*.py")):
        if py.name in ("train_perf.py",):
            continue
        text = py.read_text(encoding="utf-8", errors="replace")
        if "enable_tf32" in text:
            callers.add(py.name)
    assert callers <= {"supervised_train.py", "az_loop.py", "trainer.py"}, \
        f"TF32 reached a non-training module: {sorted(callers)}"
