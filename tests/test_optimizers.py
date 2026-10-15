"""The sequence trainer's optimizers (wesnoth_ai/optimizers.py) on a tiny
network of the recipe's kind, on the CPU: the default is the recipe's AdamW
to the bit; each parameter takes the role the design gives it (no norm, bias,
embedding or memory row is decayed by the hygiene or taken by Muon or SOAP);
Muon's updates are near-orthogonal, the query, key and value apart, at
AdamW's update RMS; every candidate steps, saves and resumes exactly;
schedule-free is read at its average and trains on from its training point
exactly; the weight average follows its recurrence and leaves the weights as
they were; the command line refuses what a run cannot hold together."""
from __future__ import annotations

import io
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.sequence_args import check_optimizer_args, parse_args  # noqa: E402
from tools.sequence_setup import build_modules  # noqa: E402
from wesnoth_ai.muon import RMS_MATCH, Muon, newton_schulz  # noqa: E402
from wesnoth_ai.optimizers import (DECAY, MATRIX, NO_DECAY, OPTIMIZERS, CompositeOptimizer,  # noqa: E402
                                   OptimizerChoice, ScheduleFreeAdamW, build_optimizer, evaluation_point,
                                   parameter_roles)
from wesnoth_ai.param_groups import named_model_parameters  # noqa: E402
from wesnoth_ai.weight_ema import WeightEMA  # noqa: E402

ARCH = {"d_model": 32, "num_layers": 2, "num_heads": 2, "d_ff": 64}
LR, WD = 1e-3, 1e-2


def _network(seed=0):
    torch.manual_seed(seed)
    encoder, model = build_modules({"Spearman": 1, "Cavalryman": 2}, {"Loyalists": 0}, torch.device("cpu"), ARCH)
    return {"model": model, "encoder": encoder}


def _named(modules):
    return named_model_parameters(modules["model"], modules["encoder"])


def _optimizer(name, modules, **choice):
    return build_optimizer(OptimizerChoice(name, **choice), _named(modules), modules, LR, WD)


def _gradients(named, step):
    """A loss's gradient at the current weights, pulling each toward a
    target of its step: a wrong restored weight changes every later step."""
    g = torch.Generator().manual_seed(1000 + step)
    for _, p in named:
        target = torch.randn(p.shape, generator=g)
        p.grad = (p.detach() - target) * 0.1


def _train(opt, modules, steps, start=0):
    named = _named(modules)
    for t in range(start, start + steps):
        _gradients(named, t)
        opt.step()
        opt.zero_grad(set_to_none=True)


def _weights(modules):
    return {f"{space}.{k}": v.clone() for space, m in modules.items() for k, v in m.state_dict().items()}


def test_the_default_is_the_recipe_s_adamw_to_the_bit():
    """--optimizer adamw builds what the trainer built before the choice
    existed: AdamW over every parameter in the trainer's order, the recipe's
    rate and weight decay, betas at torch's defaults."""
    a, b = _network(), _network()
    old = torch.optim.AdamW([p for _, p in _named(a)], lr=LR, weight_decay=WD)
    new = _optimizer("adamw", b)
    assert type(new) is torch.optim.AdamW and len(new.param_groups) == 1
    _train(old, a, 5)
    _train(new, b, 5)
    for name, w in _weights(a).items():
        assert torch.equal(w, _weights(b)[name]), name
    assert new.param_groups[0]["betas"] == (0.9, 0.999) and new.param_groups[0]["weight_decay"] == WD


def _where(opt):
    """Each parameter's optimizer part and weight decay: {id: (part, decay)}."""
    parts = opt.parts if isinstance(opt, CompositeOptimizer) else {"adamw": opt}
    return {id(p): (part, g["weight_decay"]) for part, o in parts.items() for g in o.param_groups for p in g["params"]}


def test_each_parameter_takes_the_role_the_design_gives_it():
    modules = _network()
    named = _named(modules)
    roles = dict(zip([n for n, _ in named], parameter_roles(named, modules)))
    layers = ARCH["num_layers"]
    assert sorted(n for n, r in roles.items() if r == MATRIX) == sorted(
        f"model.encoder.layers.{i}.{w}" for i in range(layers)
        for w in ("self_attn.in_proj_weight", "self_attn.out_proj.weight", "linear1.weight", "linear2.weight"))
    for name in ("model.slot_memory.initial", "model.slot_memory.slot_embed.weight", "model.token_kind_embed.weight",
                 "encoder.unit_type_embed.weight", "model.encoder.layers.0.norm1.weight",
                 "model.encoder.layers.1.norm2.bias", "model.encoder.layers.0.linear1.bias", "encoder.end_turn_token"):
        assert roles[name] == NO_DECAY, name
    for name in ("model.slot_memory.gate.weight", "model.slot_memory.candidate.weight", "model.value_head.0.weight",
                 "encoder.unit_feat_proj.weight"):
        assert roles[name] == DECAY, name
    norms_and_memory = {n for n, p in named if p.dim() < 2 or n.startswith("model.slot_memory.")
                        or ".norm" in n}
    by_name = dict(named)
    for opt_name in OPTIMIZERS:
        where = _where(_optimizer(opt_name, modules) if opt_name != "schedulefree"
                       else _optimizer(opt_name, modules).inner)
        assert len(where) == len(named), "every parameter in exactly one group"
        for n in norms_and_memory:
            part, decay = where[id(by_name[n])]
            assert part == "adamw", (opt_name, n)
            if opt_name != "adamw" and n not in ("model.slot_memory.gate.weight", "model.slot_memory.candidate.weight"):
                assert decay == 0.0, (opt_name, n)
        if opt_name == "adamw":
            assert all(decay == WD for _, decay in where.values())
        if opt_name in ("muon", "soap"):
            assert {n for n, p in named if where[id(p)][0] == opt_name} == {n for n, r in roles.items() if r == MATRIX}
            assert all(where[id(by_name[n])][1] == WD for n, r in roles.items() if r in (MATRIX, DECAY))


def _conditioned(rows, cols, generator, batch=()):
    """Matrices of singular values spread over [0.1, 1] (a condition number
    of 10), in random singular bases: the range in which Newton-Schulz's five
    steps reach their band (a near-singular direction stays short)."""
    k = min(rows, cols)
    U, _ = torch.linalg.qr(torch.randn(*batch, rows, k, generator=generator))
    V, _ = torch.linalg.qr(torch.randn(*batch, cols, k, generator=generator))
    return U @ torch.diag_embed(torch.linspace(1.0, 0.1, k).expand(*batch, k)) @ V.mT, U @ V.mT


def _band(m) -> bool:
    s = torch.linalg.svdvals(m.float())
    return bool(s.min() > 0.6 and s.max() < 1.25)


def test_newton_schulz_gives_near_orthogonal_matrices_along_the_polar_factor():
    G, polar = _conditioned(48, 24, torch.Generator().manual_seed(3), batch=(6,))
    ortho = newton_schulz(G).float()
    assert _band(ortho), "Jordan's band around 1"
    cos = (ortho * polar).sum((-2, -1)) / (ortho.norm(dim=(-2, -1)) * polar.norm(dim=(-2, -1)))
    assert cos.min() > 0.97, "the polar factor's direction"
    wide = newton_schulz(G.mT).float()
    assert torch.allclose(wide, ortho.mT, atol=0.02), "a tall matrix is orthogonalized as its wide transpose"


def test_muon_orthogonalizes_query_key_and_value_apart_at_adamw_s_update_rms():
    d, lr = 32, 1e-2
    g = torch.Generator().manual_seed(5)
    in_proj = torch.nn.Parameter(torch.randn(3 * d, d, generator=g))
    ffn = torch.nn.Parameter(torch.randn(4 * d, d, generator=g))
    opt = Muon([{"params": [in_proj], "blocks": 3}, {"params": [ffn], "blocks": 1}], lr=lr, weight_decay=0.0)
    before = [in_proj.detach().clone(), ffn.detach().clone()]
    # Query, key and value gradients 100 times apart: orthogonalized as one
    # stacked matrix, the query's block would come out about 100 times short.
    in_proj.grad = torch.cat([_conditioned(d, d, g)[0] * scale for scale in (1.0, 10.0, 100.0)])
    ffn.grad = _conditioned(4 * d, d, g)[0]
    opt.step()
    updates = [(b - p.detach()) / lr for b, p in zip(before, (in_proj, ffn))]
    for block in updates[0].reshape(3, d, d):
        assert _band(block / (RMS_MATCH * math.sqrt(d))), "each of q, k and v orthogonal, whatever its scale"
    assert _band(updates[1] / (RMS_MATCH * math.sqrt(4 * d)))
    for u in updates:
        rms = u.pow(2).mean().sqrt().item()
        assert 0.6 * RMS_MATCH < rms < 1.1 * RMS_MATCH, "an update of AdamW's RMS per unit of rate"


def test_muon_decays_its_weights_as_moonshot_does():
    w = torch.nn.Parameter(torch.randn(8, 8))
    opt = Muon([w], lr=0.1, weight_decay=0.5)
    before = w.detach().clone()
    w.grad = torch.zeros_like(w)
    w.grad[0, 0] = 1e-30                      # an update of norm ~0 after the normalization's epsilon
    opt.step()
    assert torch.allclose(w.detach(), before * (1 - 0.1 * 0.5), atol=1e-3)


def _save_and_load(name, modules, opt, **choice):
    """The trainer's checkpoint round trip: the weights an evaluation reads
    and the optimizer state, saved, loaded into a new network and optimizer."""
    buf = io.BytesIO()
    with evaluation_point(opt):
        torch.save({"weights": {s: m.state_dict() for s, m in modules.items()}, "opt": opt.state_dict()}, buf)
    buf.seek(0)
    ck = torch.load(buf, map_location="cpu", weights_only=True)
    fresh = _network(seed=99)
    for space, m in fresh.items():
        m.load_state_dict(ck["weights"][space])
    opt2 = _optimizer(name, fresh, **choice)
    opt2.load_state_dict(ck["opt"])
    return fresh, opt2


@pytest.mark.parametrize("name", OPTIMIZERS)
def test_each_optimizer_steps_saves_and_resumes_exactly(name):
    choice = {"precondition_every": 2} if name == "soap" else {}
    whole, cut = _network(), _network()
    opt_whole, opt_cut = _optimizer(name, whole, **choice), _optimizer(name, cut, **choice)
    start = _weights(whole)
    _train(opt_whole, whole, 7)
    _train(opt_cut, cut, 3)
    resumed, opt_resumed = _save_and_load(name, cut, opt_cut, **choice)
    _train(opt_resumed, resumed, 4, start=3)
    end = _weights(whole)
    assert any(not torch.equal(end[k], start[k]) for k in end), "the steps moved the weights"
    for k, w in end.items():
        assert torch.equal(w, _weights(resumed)[k]), (name, k)
    with evaluation_point(opt_whole), evaluation_point(opt_resumed):
        for k, w in _weights(whole).items():
            assert torch.equal(w, _weights(resumed)[k]), (name, k, "evaluation point")


def test_schedule_free_is_read_at_its_average_and_trains_on_from_its_training_point():
    modules = _network()
    opt = _optimizer("schedulefree", modules)
    assert isinstance(opt, ScheduleFreeAdamW)
    _train(opt, modules, 4)
    named = _named(modules)
    y = {n: p.detach().clone() for n, p in named}
    beta1 = opt.param_groups[0]["betas"][0]
    x = {n: y[n].lerp(opt.state[p]["z"], 1 - 1 / beta1) for n, p in named}
    with evaluation_point(opt):
        held = {n: p.detach().clone() for n, p in named}
        assert opt.state_dict()["train_point"][0] is not None
    assert all(torch.allclose(held[n], x[n], rtol=0, atol=1e-6) for n in x), "the average x in the parameters"
    assert any(not torch.allclose(held[n], y[n]) for n in x), "x is not y"
    assert all(torch.equal(p.detach(), y[n]) for n, p in named), "y restored to the bit"
    with pytest.raises(RuntimeError):
        with evaluation_point(opt):
            opt.step()


def test_the_weight_average_follows_its_recurrence_and_leaves_the_weights_as_they_were():
    p = torch.nn.Parameter(torch.zeros(3))
    ema = WeightEMA([("model.w", p)], decay=0.5)
    expected = torch.zeros(3)
    for t, value in enumerate((1.0, 2.0, 4.0, 8.0)):
        with torch.no_grad():
            p.fill_(value)
        ema.update()
        d = min(0.5, (1 + t) / (10 + t))
        expected = d * expected + (1 - d) * value
    assert torch.allclose(ema.average[0], expected)
    with ema.applied():
        assert torch.equal(p.detach(), ema.average[0])
    assert torch.equal(p.detach(), torch.full((3,), 8.0))
    modules = _network()
    named = _named(modules)
    ema = WeightEMA(named, decay=0.9)
    opt = _optimizer("adamw", modules)
    _train(opt, modules, 3)
    for _ in range(3):
        ema.update()
    states = ema.module_states(modules)
    other = WeightEMA(named, decay=0.9)
    other.load_module_states(states, ema.updates)
    assert other.updates == 3 and all(torch.equal(a, b) for a, b in zip(ema.average, other.average))
    fresh = _network(seed=7)
    for space, m in fresh.items():
        m.load_state_dict(states[space])
    with ema.applied():
        assert all(torch.equal(w, _weights(fresh)[k]) for k, w in _weights(modules).items()), "loadable as weights"


def _args(*extra):
    return parse_args(["--sequences", "s", "--dataset", "d", "--out", "o.pt", "--device", "cpu", *extra])


@pytest.mark.parametrize("extra, refused", [
    (["--optimizer", "schedulefree", "--decay-from", "0.8"], "--decay-from"),
    (["--optimizer", "schedulefree", "--weight-ema", "0.99"], "--weight-ema"),
    (["--soap-precondition-every", "5"], "--soap-precondition-every"),
    (["--weight-ema", "1.5"], "--weight-ema"),
    (["--probe-weights", "both"], "--probe-weights"),
])
def test_the_command_line_refuses_what_a_run_cannot_hold_together(extra, refused):
    with pytest.raises(SystemExit, match=refused):
        check_optimizer_args(_args(*extra))
    check_optimizer_args(_args("--optimizer", "soap", "--soap-precondition-every", "5", "--weight-ema", "0.99",
                               "--probe-weights", "both", "--decay-from", "0.8"))


def test_the_meta_records_each_candidate_apart():
    described = [OptimizerChoice(n).describe() for n in OPTIMIZERS]
    assert described[0] == {"name": "adamw"}, "the value a checkpoint written before the choice reads as"
    assert len({str(d) for d in described}) == len(OPTIMIZERS)
    assert OptimizerChoice("soap", 20).describe() != OptimizerChoice("soap").describe()
