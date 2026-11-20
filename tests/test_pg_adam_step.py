"""The simple loop's optimizer step (tools/pg_adam_step.py): Adam's update
along the ascent direction, its state carried from step to step, and its
refusals."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

LR, B1, B2, EPS = 1e-3, 0.9, 0.999, 1e-8


def _network(tmp_path):
    from helpers.pg_games import save_network, tiny_network
    return save_network(tmp_path / "pi0.pt", *tiny_network())


def _parameters(checkpoint):
    from tools.pg_pass import load_network
    from wesnoth_ai.param_groups import named_model_parameters
    encoder, model = load_network(checkpoint, torch.device("cpu"))
    return {n: p.detach().clone() for n, p in named_model_parameters(model, encoder)}


def _grad(tmp_path, checkpoint, seed, name="grad.pt", still=()):
    """A gradient file for `checkpoint`: random entries, zero on the
    parameters named in `still`."""
    from tools.eval_provenance import file_sha256
    params = _parameters(checkpoint)
    torch.manual_seed(seed)
    total = {n: (torch.zeros_like(p) if n in still else torch.randn_like(p)) for n, p in params.items()}
    path = tmp_path / name
    torch.save({"names": list(params), "total": total, "checkpoint_sha256": file_sha256(checkpoint)}, path)
    return path, total


def test_two_steps_are_adams_and_the_state_carries_the_moments(tmp_path):
    """The first step moves every parameter by the learning rate along its
    gradient's sign (Adam's first step: g over |g| + eps) and leaves a
    parameter without a gradient where it was. The second, from the saved
    state, is Adam's second step by its formulas. Each stepped checkpoint
    records its step and loads as a network."""
    from tools.pg_adam_step import main
    pi0 = _network(tmp_path)
    theta0 = _parameters(pi0)
    still = next(iter(theta0))
    g1_path, g1 = _grad(tmp_path, pi0, 1, "g1.pt", still=(still,))
    assert main(["--checkpoint", str(pi0), "--grad", str(g1_path), "--lr", str(LR), "--out", str(tmp_path / "pi1.pt"),
                 "--state-out", str(tmp_path / "adam1.pt")]) == 0
    theta1 = _parameters(tmp_path / "pi1.pt")
    for n, p in theta0.items():
        want = p if n == still else p + LR * g1[n] / (g1[n].abs() + EPS)
        assert torch.allclose(theta1[n], want, rtol=0, atol=1e-7), n
    moved = torch.cat([(theta1[n] - p).flatten() for n, p in theta0.items() if n != still])
    assert float(moved.abs().median()) == pytest.approx(LR, rel=1e-3)
    record = json.loads((tmp_path / "pi1.step.json").read_text(encoding="utf-8"))
    assert record["step"] == 1 and record["update_norm"] > 0 and sum(record["grad_norms"].values()) > 0
    ck = torch.load(tmp_path / "pi1.pt", weights_only=True)
    assert ck["pg_step"]["step"] == 1 and ck["pg_step"]["lr"] == LR

    g2_path, g2 = _grad(tmp_path, tmp_path / "pi1.pt", 2, "g2.pt")
    assert main(["--checkpoint", str(tmp_path / "pi1.pt"), "--grad", str(g2_path), "--state",
                 str(tmp_path / "adam1.pt"), "--lr", str(LR), "--out", str(tmp_path / "pi2.pt"),
                 "--state-out", str(tmp_path / "adam2.pt")]) == 0
    theta2 = _parameters(tmp_path / "pi2.pt")
    for n in theta0:
        m = (1 - B1) * (B1 * g1[n] + g2[n])
        v = (1 - B2) * (B2 * g1[n] ** 2 + g2[n] ** 2)
        want = theta1[n] + LR * (m / (1 - B1 ** 2)) / ((v / (1 - B2 ** 2)).sqrt() + EPS)
        assert torch.allclose(theta2[n], want, rtol=0, atol=2e-7), n
    assert torch.load(tmp_path / "adam2.pt", weights_only=True)["step"] == 2


def test_a_gradient_or_a_state_that_does_not_belong_is_refused_and_nothing_written(tmp_path):
    """A gradient that is not finite, a gradient of another checkpoint and a
    state left at another checkpoint each end the step with exit 3 before
    any file is written."""
    from tools.pg_adam_step import EXIT_REFUSED, main
    pi0 = _network(tmp_path)
    g_path, _ = _grad(tmp_path, pi0, 1)
    out, state_out = tmp_path / "pi1.pt", tmp_path / "adam1.pt"
    base = ["--checkpoint", str(pi0), "--lr", str(LR), "--out", str(out), "--state-out", str(state_out)]

    bad = torch.load(g_path, weights_only=True)
    first = bad["names"][0]
    bad["total"][first].view(-1)[0] = float("inf")
    torch.save(bad, tmp_path / "inf.pt")
    assert main(base + ["--grad", str(tmp_path / "inf.pt")]) == EXIT_REFUSED

    other = torch.load(g_path, weights_only=True)
    other["checkpoint_sha256"] = "0" * 64
    torch.save(other, tmp_path / "other.pt")
    assert main(base + ["--grad", str(tmp_path / "other.pt")]) == EXIT_REFUSED
    assert not out.exists() and not state_out.exists()

    assert main(base + ["--grad", str(g_path)]) == 0
    stale = ["--checkpoint", str(pi0), "--grad", str(g_path), "--state", str(state_out), "--lr", str(LR),
             "--out", str(tmp_path / "pi2.pt"), "--state-out", str(tmp_path / "adam2.pt")]
    assert main(stale) == EXIT_REFUSED            # the state was left at pi1, not at pi0
    assert not (tmp_path / "pi2.pt").exists()


def test_a_changed_learning_rate_takes_effect_on_a_carried_state(tmp_path):
    """The step's learning rate is this step's, whatever the state was saved
    with: the same gradient at ten times the rate moves ten times as far."""
    from tools.pg_adam_step import adam_step
    pi0 = _network(tmp_path)
    g1_path, _ = _grad(tmp_path, pi0, 1, "g1.pt")
    adam_step(pi0, g1_path, None, LR, tmp_path / "pi1.pt", tmp_path / "adam1.pt")
    g2_path, _ = _grad(tmp_path, tmp_path / "pi1.pt", 2, "g2.pt")
    slow = adam_step(tmp_path / "pi1.pt", g2_path, tmp_path / "adam1.pt", LR, tmp_path / "slow.pt",
                     tmp_path / "adam_slow.pt")
    fast = adam_step(tmp_path / "pi1.pt", g2_path, tmp_path / "adam1.pt", 10 * LR, tmp_path / "fast.pt",
                     tmp_path / "adam_fast.pt")
    assert fast["update_norm"] == pytest.approx(10 * slow["update_norm"], rel=1e-4)
