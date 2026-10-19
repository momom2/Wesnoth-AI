"""The optimizers (wesnoth_ai/optimizers.py) on CUDA, the box's device: each
candidate steps a tiny network of the recipe's kind toward a target (Muon's
bfloat16 iteration, SOAP's eigendecompositions and QR refreshes on the
device), saves and resumes exactly, and schedule-free's evaluation point
restores its training point to the bit. Skipped without CUDA (the box runs
them, scripts/optimizer_bench_box.sh)."""
from __future__ import annotations

import io

import pytest
import torch

from tools.sequence_setup import build_modules
from wesnoth_ai.optimizers import OPTIMIZERS, OptimizerChoice, build_optimizer, evaluation_point
from wesnoth_ai.param_groups import named_model_parameters

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
ARCH = {"d_model": 64, "num_layers": 2, "num_heads": 2, "d_ff": 128}


def _network(seed):
    torch.manual_seed(seed)
    encoder, model = build_modules({"Spearman": 1}, {"Loyalists": 0}, torch.device("cuda"), ARCH)
    return {"model": model, "encoder": encoder}


def _distance(named, targets):
    return sum(float((p.detach() - t).pow(2).sum()) for (_, p), t in zip(named, targets))


@pytest.mark.parametrize("name", OPTIMIZERS)
def test_each_optimizer_steps_toward_a_target_and_resumes_exactly_on_cuda(name):
    choice = OptimizerChoice(name, 3 if name == "soap" else None)
    modules = _network(0)
    named = named_model_parameters(modules["model"], modules["encoder"])
    opt = build_optimizer(choice, named, modules, 3e-3, 1e-4)
    g = torch.Generator(device="cuda").manual_seed(1)
    targets = [torch.randn(p.shape, device="cuda", generator=g) * 0.05 for _, p in named]

    def steps(opt, named, n):
        for _ in range(n):
            for (_, p), t in zip(named, targets):
                p.grad = p.detach() - t
            opt.step()
            opt.zero_grad(set_to_none=True)

    start = _distance(named, targets)
    steps(opt, named, 8)
    buf = io.BytesIO()
    with evaluation_point(opt):
        torch.save({s: m.state_dict() for s, m in modules.items()} | {"opt": opt.state_dict()}, buf)
    buf.seek(0)
    ck = torch.load(buf, map_location="cuda", weights_only=True)
    other = _network(5)
    for s, m in other.items():
        m.load_state_dict(ck[s])
    other_named = named_model_parameters(other["model"], other["encoder"])
    other_opt = build_optimizer(choice, other_named, other, 3e-3, 1e-4)
    other_opt.load_state_dict(ck["opt"])
    steps(opt, named, 4)
    steps(other_opt, other_named, 4)
    assert all(torch.isfinite(p).all() for _, p in named)
    assert _distance(named, targets) < start, "the steps went toward the target"
    for (n, p), (_, q) in zip(named, other_named):
        assert torch.equal(p, q), (name, n)
