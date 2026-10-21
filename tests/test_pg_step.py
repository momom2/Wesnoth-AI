"""The step math of the policy-gradient round (wesnoth_ai/pg_step.py) on
hand-built inputs: each arm's direction, the KL search, the critical batch on
a Gaussian noise model of known B, the floor against a brute-force bound, and
the ladder's adoption rule."""
from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from wesnoth_ai.muon import newton_schulz  # noqa: E402
from wesnoth_ai.pg_protocol import Evaluator, GameSide  # noqa: E402
from wesnoth_ai.pg_step import (KL_RTOL, critical_batch, debiased_quadratic, decide, direction,  # noqa: E402
                                floor, head_of, kl_search, ladder, muon_matrices)

IN_PROJ = "model.encoder.layers.0.self_attn.in_proj_weight"
LINEAR = "model.encoder.layers.0.linear1.weight"
SHAPES = {IN_PROJ: (6, 4), LINEAR: (5, 3), "model.encoder.layers.0.linear1.bias": (5,),
          "model.token_kind_embed.weight": (4, 3), "model.actor_head.weight": (1, 4), "model.actor_head.bias": (1,),
          "model.type_head.weight": (2, 4), "model.target_q_proj.weight": (4, 4),
          "model.target_k_proj.weight": (4, 4), "model.weapon_head.0.weight": (4, 4),
          "model.weapon_head.2.weight": (3, 4), "model.weapon_head.2.bias": (3,)}
HEADS = {"model.actor_head.weight", "model.actor_head.bias", "model.type_head.weight",
         "model.target_q_proj.weight", "model.weapon_head.2.weight", "model.weapon_head.2.bias"}


def _random(seed, positive=False):
    g = torch.Generator().manual_seed(seed)
    out = {name: torch.randn(shape, generator=g) for name, shape in SHAPES.items()}
    return {name: t.abs() + 0.1 for name, t in out.items()} if positive else out


def _pass():
    """Two splits of unequal sizes; the in-projection's second moment is 0
    on one row in both."""
    splits = [_random(1), _random(2)]
    sq = [_random(3, positive=True), _random(4, positive=True)]
    for s in sq:
        s[IN_PROJ][0] = 0.0
    total = {name: splits[0][name] + splits[1][name] for name in SHAPES}
    return {"names": list(SHAPES), "total": total, "splits": splits, "sq": sq, "split_games": [30, 10],
            "games": 40, "windows": 40, "decisions": 400}


def test_each_arm_builds_its_direction():
    p = _pass()
    g = p["total"]
    assert all(torch.equal(direction(p, "sgd")[n], g[n]) for n in SHAPES)

    adam = direction(p, "reduced_adamw")
    v = {n: 0.75 * p["sq"][0][n] + 0.25 * p["sq"][1][n] for n in SHAPES}     # weighted by 30 and 10 games
    assert torch.equal(adam[IN_PROJ][0], torch.zeros(4))                   # zero second moment, zero step
    assert torch.allclose(adam[IN_PROJ][1:], g[IN_PROJ][1:] / v[IN_PROJ][1:].sqrt())
    assert torch.allclose(adam[LINEAR], g[LINEAR] / v[LINEAR].sqrt())

    muon = direction(p, "muon")
    assert set(muon_matrices(SHAPES)) == {IN_PROJ, LINEAR}
    for name in SHAPES:
        if name not in (IN_PROJ, LINEAR):
            assert torch.equal(muon[name], adam[name]), name
    for name, blocks in ((IN_PROJ, 3), (LINEAR, 1)):
        rows, cols = SHAPES[name]
        got = muon[name].reshape(blocks, rows // blocks, cols)
        want = adam[name].reshape(blocks, rows // blocks, cols)
        assert torch.allclose(got.pow(2).mean(dim=(1, 2)), want.pow(2).mean(dim=(1, 2)), rtol=1e-4)
        ortho = newton_schulz(g[name].reshape(blocks, rows // blocks, cols)).float()
        for block in range(blocks):                                         # each block its own polar factor
            cos = (got[block] * ortho[block]).sum() / (got[block].norm() * ortho[block].norm())
            assert cos > 0.9999

    heads = direction(p, "heads_only")
    for name in SHAPES:
        if name in HEADS:
            assert torch.equal(heads[name], adam[name]), name
        else:
            assert not heads[name].any(), name


def test_probe_span_moves_along_the_head_temperatures():
    """Each head's part of the step is a multiple c_h of its own weights,
    c = s with no Fisher and c = F^-1 s with one, s the gradient's slopes."""
    p, params = _pass(), _random(9)
    groups = {"actor_head": ["model.actor_head.weight", "model.actor_head.bias"],
              "type_head": ["model.type_head.weight"], "target_q_proj": ["model.target_q_proj.weight"],
              "weapon_head.2": ["model.weapon_head.2.weight", "model.weapon_head.2.bias"]}
    slopes = np.array([sum(float((p["total"][n] * params[n]).sum()) for n in names) for names in groups.values()])
    fisher = np.array([[2.0, 0.3, 0.0, 0.1], [0.3, 1.0, 0.2, 0.0], [0.0, 0.2, 3.0, 0.4], [0.1, 0.0, 0.4, 0.5]])
    for F, coefficients in ((None, slopes), (fisher, np.linalg.solve(fisher, slopes))):
        d = direction(p, "probe_span", params=params, span_fisher=F)
        for c, names in zip(coefficients, groups.values()):
            for name in names:
                assert torch.allclose(d[name], float(c) * params[name], rtol=1e-5, atol=1e-6), name
        for name in SHAPES:
            if head_of(name) is None:
                assert not d[name].any(), name


def test_muon_matrices_and_heads_match_the_network():
    """The name rules read the network as the trainer's roles do: Muon's
    matrices are the trainer's matrix role, and the heads' final maps are
    the parameters named in design §3.4."""
    from tools.sequence_setup import build_modules
    from wesnoth_ai.optimizers import MATRIX, parameter_roles
    from wesnoth_ai.param_groups import named_model_parameters

    torch.manual_seed(0)
    encoder, model = build_modules({"Spearman": 1}, {"Loyalists": 0}, torch.device("cpu"),
                                   {"d_model": 32, "num_layers": 2, "num_heads": 2, "d_ff": 64})
    named = named_model_parameters(model, encoder)
    roles = parameter_roles(named, {"model": model, "encoder": encoder})
    assert set(muon_matrices({n: p.shape for n, p in named})) == {n for (n, _), r in zip(named, roles) if r == MATRIX}
    assert {n for n, _ in named if head_of(n)} == {
        "model.actor_head.weight", "model.actor_head.bias", "model.type_head.weight", "model.type_head.bias",
        "model.target_q_proj.weight", "model.weapon_head.2.weight", "model.weapon_head.2.bias"}


def test_kl_search_hits_its_target():
    for curve in (lambda s: 0.5 * 3.0 * s * s, lambda s: s ** 4 + 0.1 * s * s,
                  lambda s: math.inf if s > 2.0 else 1.0 - math.cos(s)):
        for target in (1e-6, 1e-2, 0.3):
            for start in (1e-5, 1.0, 1e4):
                point = kl_search(curve, target, s0=start)
                assert abs(point.kl - target) <= KL_RTOL * target
                assert abs(curve(point.s) - target) <= KL_RTOL * target
    with pytest.raises(ValueError):
        kl_search(lambda s: 0.0, 0.01)                                       # a zero direction


def test_critical_batch_recovers_B_on_a_gaussian_noise_model():
    """Split gradients drawn as N(g, Sigma / n_i): over many draws the
    estimated signal and noise match g^T M g and tr(M Sigma), M = PFP, and
    their ratio the true B."""
    rng = np.random.default_rng(0)
    dim, n = 12, np.array([2400, 2600, 2500, 2500, 2450, 2550, 2500, 2500])
    P, F = np.diag(rng.uniform(0.5, 2.0, dim)), np.diag(rng.uniform(0.2, 1.5, dim))
    g = rng.normal(size=dim) * 0.02
    A = rng.normal(size=(dim, dim))
    Sigma = A @ A.T / dim
    M = P @ F @ P
    true_signal, true_noise = g @ M @ g, np.trace(M @ Sigma)
    root = np.linalg.cholesky(Sigma)
    signals, noises = [], []
    for _ in range(3000):
        splits = g + (rng.normal(size=(len(n), dim)) @ root.T) / np.sqrt(n)[:, None]
        total = (n[:, None] * splits).sum(axis=0) / n.sum()
        q = [x @ M @ x for x in splits]
        result = critical_batch(q, total @ M @ total, n)
        signals.append(result.signal)
        noises.append(result.noise)
    for got, want in ((signals, true_signal), (noises, true_noise)):
        assert abs(np.mean(got) - want) < 3 * np.std(got) / math.sqrt(len(got))
    assert abs(np.mean(noises) / np.mean(signals) / (true_noise / true_signal) - 1) < 0.05


def test_floor_is_the_best_step_in_the_span_and_bounds_the_full_one():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(5, 5))
    F = A @ A.T + 0.5 * np.eye(5)
    g = rng.normal(size=5)
    U = rng.normal(size=(5, 2))
    K = 0.3
    s, F_s = U.T @ g, U.T @ F @ U
    bound = floor(s, F_s, K)
    angles = np.linspace(0, 2 * np.pi, 200_001)
    circle = np.stack([np.cos(angles), np.sin(angles)])
    c = np.linalg.solve(np.linalg.cholesky(F_s).T, circle) * math.sqrt(2 * K)    # 1/2 c^T F_s c = K
    in_span = (s @ c).max()
    assert in_span <= bound * (1 + 1e-9) and in_span > bound * (1 - 1e-6)
    sphere = rng.normal(size=(5, 200_000))
    sphere /= np.linalg.norm(sphere, axis=0)
    full = (g @ (np.linalg.solve(np.linalg.cholesky(F).T, sphere) * math.sqrt(2 * K))).max()
    assert full >= bound


def test_debiased_quadratic_removes_the_noise():
    rng = np.random.default_rng(2)
    n = np.array([500] * 8)
    true_s, F_s = np.array([0.02, -0.01]), np.array([[1.0, 0.2], [0.2, 0.5]])
    C = np.array([[0.4, 0.1], [0.1, 0.3]])
    root = np.linalg.cholesky(C)
    truth = true_s @ np.linalg.solve(F_s, true_s)
    raw, debiased = [], []
    for _ in range(4000):
        splits = true_s + (rng.normal(size=(8, 2)) @ root.T) / np.sqrt(n)[:, None]
        total = splits.mean(axis=0)
        raw.append(total @ np.linalg.solve(F_s, total))
        debiased.append(debiased_quadratic(total, splits, n, F_s))
    assert abs(np.mean(debiased) - truth) < 3 * np.std(debiased) / math.sqrt(len(debiased))
    assert np.mean(raw) - truth > 10 * np.std(raw) / math.sqrt(len(raw))


class _LadderEvaluator:
    """One decision per game-side, KL s^2, and (W - 1) R on half h equal to
    gains[h][rung] plus fixed noise, the rung read off s."""

    def __init__(self, gains, eps, games=400, seed=0):
        rng = np.random.default_rng(seed)
        self.gains, self.eps = gains, eps
        self.sides = [GameSide(f"g{g}", side, 1.0 if side == 1 else -1.0, "A" if g % 2 == 0 else "B",
                               2 * g + side - 1, 2 * g + side) for g in range(games) for side in (1, 2)]
        self.noise = rng.normal(size=2 * games) * 0.01
        self.R = np.array([gs.R for gs in self.sides])
        self.half = np.array([gs.half for gs in self.sides])

    def game_sides(self):
        return self.sides

    def base_logp(self):
        return np.zeros(len(self.sides))

    def mean_kl(self, s):
        return s * s

    def logp(self, s):
        rung = min(self.gains["A"], key=lambda m: abs(m * self.eps - s * s))
        gain = np.where(self.half == "A", self.gains["A"][rung], self.gains["B"][rung])
        return np.log1p(self.R * (gain + self.noise))


def _ladder(gains, eps=0.01):
    evaluator = _LadderEvaluator(gains, eps)
    assert isinstance(evaluator, Evaluator)
    return ladder(lambda d: evaluator, {}, eps)


def test_the_ladder_adopts_reverts_and_sets_the_next_radius():
    """Selection on half A, confirmation on half B: adopt unless half B
    reads below -2 SE (the boundary read both ways); the next radius is the
    adopted rung's, or half the radius after a revert."""
    eps = 0.01
    middle = _ladder({"A": {0.5: 0.01, 1.0: 0.03, 2.0: 0.02}, "B": {0.5: 0.0, 1.0: 0.02, 2.0: 0.0}})
    assert middle.adopt and middle.chosen == 1 and middle.next_eps == pytest.approx(eps)
    assert middle.s == pytest.approx(math.sqrt(eps), rel=KL_RTOL)

    far = _ladder({"A": {0.5: 0.01, 1.0: 0.02, 2.0: 0.03}, "B": {0.5: 0.0, 1.0: 0.0, 2.0: 0.01}})
    assert far.adopt and far.chosen == 2 and far.next_eps == pytest.approx(2 * eps)

    se = far.rungs[2].halves["B"].exact.se
    shift = far.rungs[2].halves["B"].exact.estimate - 0.01            # the noise's mean on half B
    for z, adopted in ((-1.5, True), (-2.5, False)):
        gains = {"A": {0.5: 0.01, 1.0: 0.02, 2.0: 0.03}, "B": {0.5: 0.0, 1.0: 0.0, 2.0: z * se - shift}}
        decision = _ladder(gains)
        assert decision.rungs[2].halves["B"].exact.estimate == pytest.approx(z * se, rel=1e-6)
        assert decision.adopt is adopted and decision.chosen == 2
        if not adopted:
            assert decision.s == 0.0 and decision.next_eps == pytest.approx(eps / 2)


def test_rungs_past_the_importance_weights_range_are_not_selected():
    """At 1 decision a side-game, eps 0.3 puts the 2 eps rung at K 0.6, past
    the range; with every rung past it, pi_k stays and the radius halves."""
    gains = {"A": {0.5: 0.01, 1.0: 0.02, 2.0: 0.05}, "B": {0.5: 0.0, 1.0: 0.0, 2.0: 0.0}}
    decision = ladder(lambda d: _LadderEvaluator(gains, 0.3), {}, 0.3)
    assert [r.in_range for r in decision.rungs] == [True, True, False]
    assert decision.adopt and decision.chosen == 1
    kept = decide([dataclasses.replace(r, in_range=False) for r in decision.rungs], 0.3)
    assert not kept.adopt and kept.chosen is None and kept.next_eps == pytest.approx(0.15)
