"""The step math of the policy-gradient round (wesnoth_ai/pg_step.py) on
hand-built inputs: each arm's direction, the KL search, the clustered SE, the
Fisher of a few directions, the per-side-game KL, and the ladder's adoption
rule."""
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
from wesnoth_ai.pg_step import (KL_RTOL, clustered_mean, decide, direction, head_of, kl_search,  # noqa: E402
                                ladder, muon_matrices, paired_ratio, read_candidate, second_moment,
                                sides_by_half, span_fisher)

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


def test_the_se_is_clustered_by_game():
    """Two seats of a game reading the same value are one observation: the
    SE is that of the games' means, not of twice as many game-sides."""
    v = np.random.default_rng(3).normal(size=50)
    reading = clustered_mean(np.repeat(v, 2), [f"g{i // 2}" for i in range(100)])
    assert reading.estimate == pytest.approx(v.mean())
    assert reading.se == pytest.approx(v.std(ddof=1) / math.sqrt(len(v)))


class _LadderEvaluator:
    """`decisions` decisions per game-side, KL s^2 per decision, and (W - 1) R
    on half h equal to gains[h][rung] plus fixed noise, the rung read off s
    and each game-side's whole log weight on its first decision; `spike`
    (rung, half, value) adds value to W - 1 of that half's first game-side."""

    def __init__(self, gains, eps, games=400, seed=0, spike=None, decisions=1):
        rng = np.random.default_rng(seed)
        self.gains, self.eps, self.spike, self.decisions = gains, eps, spike, decisions
        self.sides = [GameSide(f"g{g}", side, 1.0 if side == 1 else -1.0, "A" if g % 2 == 0 else "B",
                               (2 * g + side - 1) * decisions, (2 * g + side) * decisions)
                      for g in range(games) for side in (1, 2)]
        self.noise = rng.normal(size=2 * games) * 0.01
        self.R = np.array([gs.R for gs in self.sides])
        self.half = np.array([gs.half for gs in self.sides])
        self.first = np.array([gs.start for gs in self.sides])

    def game_sides(self):
        return self.sides

    def base_logp(self):
        return np.zeros(len(self.sides) * self.decisions)

    def mean_kl(self, s):
        return s * s

    def logp(self, s, halves=None):
        rung = min(self.gains["A"], key=lambda m: abs(m * self.eps - s * s))
        gain = np.where(self.half == "A", self.gains["A"][rung], self.gains["B"][rung])
        x = self.R * (gain + self.noise)
        if self.spike is not None and self.spike[0] == rung:
            x[np.flatnonzero(self.half == self.spike[1])[0]] += self.spike[2]
        out = np.zeros(len(self.sides) * self.decisions)
        out[self.first] = np.log1p(x)
        return out


def _ladder(gains, eps=0.01):
    evaluator = _LadderEvaluator(gains, eps)
    assert isinstance(evaluator, Evaluator)
    return ladder(lambda d: evaluator, {}, eps)


def test_the_ladder_adopts_reverts_and_sets_the_next_radius():
    """Selection on half A, confirmation on half B: adopt unless half B
    reads below -2 SE (the boundary read both ways); the next radius is the
    adopted rung's, or half the radius after a revert."""
    eps = 0.01
    middle = _ladder({"A": {0.5: 0.01, 1.0: 0.03, 2.0: 0.02}, "B": {0.5: 0.04, 1.0: 0.02, 2.0: 0.0}})
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


def test_rungs_importance_weights_cannot_read_are_not_selected():
    """At 1 decision a side-game, eps 0.4 puts the 2 eps rung at K 0.8, past
    the range, with the edge within EDGE_MIN_RATIO of the eps rung; heavy-tailed
    weights (one game-side's W at 51) fail the effective-sample check; with
    no readable rung pi_k stays and the radius halves."""
    gains = {"A": {0.5: 0.01, 1.0: 0.02, 2.0: 0.05}, "B": {0.5: 0.0, 1.0: 0.0, 2.0: 0.0}}
    decision = ladder(lambda d: _LadderEvaluator(gains, 0.4), {}, 0.4)
    assert [r.in_range for r in decision.rungs] == [True, True, False]
    assert decision.adopt and decision.chosen == 1

    heavy = ladder(lambda d: _LadderEvaluator(gains, 0.01, spike=(2.0, "A", 50.0)), {}, 0.01)
    assert heavy.rungs[2].in_range and not heavy.rungs[2].halves["A"].ess_ok
    assert heavy.rungs[2].halves["A"].exact.estimate > heavy.rungs[1].halves["A"].exact.estimate
    assert heavy.adopt and heavy.chosen == 1
    kept = decide([dataclasses.replace(r, in_range=False) for r in decision.rungs], 0.4)
    assert not kept.adopt and kept.chosen is None and kept.next_eps == pytest.approx(0.2)


def _recording(monkeypatch, evaluator):
    """The targets kl_search is asked for and the KLs logp is read at."""
    import wesnoth_ai.pg_step as pg_step
    searched, read = [], []
    monkeypatch.setattr(pg_step, "kl_search", lambda kl_at, target, **kw: searched.append(target)
                        or kl_search(kl_at, target, **kw))
    logp = evaluator.logp
    evaluator.logp = lambda s, halves=None: read.append(s * s) or logp(s, halves)
    return searched, read


def test_a_rung_past_the_range_is_neither_searched_nor_read(monkeypatch):
    """At 1 decision a side-game, eps 0.4 puts the 2 eps rung at K 0.8, past
    the range, and the edge (0.5) is within EDGE_MIN_RATIO of the eps rung:
    the ladder spends no KL search and no reading on it, its summary says
    so, and the decision is the one a read rung would give."""
    gains = {"A": {0.5: 0.01, 1.0: 0.02, 2.0: 0.05}, "B": {0.5: 0.0, 1.0: 0.0, 2.0: 0.0}}
    evaluator = _LadderEvaluator(gains, 0.4)
    searched, read = _recording(monkeypatch, evaluator)
    decision = ladder(lambda d: evaluator, {}, 0.4)
    assert searched == pytest.approx([0.2, 0.4])
    assert read == pytest.approx([0.2, 0.4, 0.4], rel=KL_RTOL)             # rungs 0.5 and 1 on A, rung 1 on B
    far = decision.rungs[2]
    assert not far.in_range and far.s is None and far.kl is None and far.halves == {}
    assert decision.summary()["rungs"][2] == {"multiplier": 2.0, "target_kl": pytest.approx(0.8), "s": None,
                                              "kl": None, "in_range": False, "halves": {}}
    assert decision.adopt and decision.chosen == 1 and decision.next_eps == pytest.approx(0.4)


def test_the_first_rung_past_the_range_is_read_at_its_edge(monkeypatch):
    """At 1 decision a side-game the edge is K 0.5. At eps 0.3 the 2 eps rung
    (0.6) is read at the edge, which then sets the next radius. At eps 0.6
    the eps rung is read at the edge and the 2 eps rung, past it too, is not
    read; a known point at eps is not reused for a rung moved to the edge."""
    import wesnoth_ai.pg_step as pg_step
    assert pg_step.EDGE_MIN_RATIO == pytest.approx(math.sqrt(2.0))
    gains = {"A": {0.5: 0.01, 1.0: 0.02, 2.0: 0.05}, "B": {0.5: 0.0, 1.0: 0.0, 2.0: 0.01}}
    evaluator = _LadderEvaluator(gains, 0.3)
    searched, read = _recording(monkeypatch, evaluator)
    top = ladder(lambda d: evaluator, {}, 0.3)
    assert searched == pytest.approx([0.15, 0.3, 0.5])
    assert read == pytest.approx([0.15, 0.3, 0.5, 0.5], rel=KL_RTOL)
    edge = top.rungs[2]
    assert edge.in_range and edge.multiplier == 2.0 and edge.target_kl == pytest.approx(0.5)
    assert edge.kl == pytest.approx(0.5, rel=KL_RTOL) and edge.halves["B"].side_kl == pytest.approx(0.5, rel=KL_RTOL)
    assert top.adopt and top.chosen == 2 and top.next_eps == pytest.approx(0.5)

    gains = {"A": {0.5: 0.01, 1.0: 0.03, 2.0: 0.05}, "B": {0.5: 0.0, 1.0: 0.01, 2.0: 0.0}}
    evaluator = _LadderEvaluator(gains, 0.6)
    searched, read = _recording(monkeypatch, evaluator)
    known = {1.0: kl_search(evaluator.mean_kl, 0.6)}
    middle = ladder(lambda d: evaluator, {}, 0.6, known=known)
    assert searched == pytest.approx([0.3, 0.5])
    assert [r.in_range for r in middle.rungs] == [True, True, False]
    assert middle.rungs[1].target_kl == pytest.approx(0.5) and middle.rungs[1].kl == pytest.approx(0.5, rel=KL_RTOL)
    assert middle.rungs[2].s is None and middle.rungs[2].target_kl == pytest.approx(1.2)
    assert middle.adopt and middle.chosen == 1 and middle.next_eps == pytest.approx(0.5)


class _Prefetching(_LadderEvaluator):
    """A `_LadderEvaluator` that also takes `prefetch` and `base_logp_on`
    (tools/pg_eval.py's), recording its calls in order, by KL."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def mean_kl(self, s):
        self.calls.append(("kl",))
        return super().mean_kl(s)

    def prefetch(self, steps, halves):
        self.calls.append(("prefetch", [s * s for s in steps], halves))

    def base_logp_on(self, halves):
        self.calls.append(("base", halves))
        return self.base_logp()

    def logp(self, s, halves=None):
        self.calls.append(("read", s * s, halves))
        return super().logp(s, halves)


def test_the_ladder_searches_every_rung_before_one_prefetch_a_half():
    """Every rung's KL search comes before any reading; the rungs are then
    prefetched on half A in one call, theta_k read there, the chosen rung
    prefetched on half B and theta_k read there; the decision is the one an
    evaluator without them gives."""
    gains = {"A": {0.5: 0.01, 1.0: 0.03, 2.0: 0.02}, "B": {0.5: 0.04, 1.0: 0.02, 2.0: 0.0}}
    evaluator = _Prefetching(gains, 0.01)
    decision = ladder(lambda d: evaluator, {}, 0.01)
    kinds = [c[0] for c in evaluator.calls]
    first = kinds.index("prefetch")
    assert set(kinds[:first]) == {"kl"} and "kl" not in kinds[first:]
    prefetched = [c for c in evaluator.calls if c[0] == "prefetch"]
    assert [c[2] for c in prefetched] == [("A",), ("B",)]
    assert prefetched[0][1] == pytest.approx([0.005, 0.01, 0.02], rel=KL_RTOL)
    assert prefetched[1][1] == pytest.approx([0.01], rel=KL_RTOL)
    assert [c[1] for c in evaluator.calls if c[0] == "base"] == [("A",), ("B",)]
    assert [c[2] for c in evaluator.calls if c[0] == "read"] == [("A",)] * 3 + [("B",)]
    assert decision.adopt and decision.chosen == 1
    assert decision.summary() == _ladder(gains).summary()


def test_half_b_weights_must_pass_before_its_reading_counts():
    """The rung chosen on half A whose weights fail the check on half B is
    not adopted on half B's reading: pi_k stays and the radius halves."""
    gains = {"A": {0.5: 0.01, 1.0: 0.02, 2.0: 0.05}, "B": {0.5: 0.0, 1.0: 0.0, 2.0: 0.01}}
    decision = ladder(lambda d: _LadderEvaluator(gains, 0.01, spike=(2.0, "B", 50.0)), {}, 0.01)
    assert decision.chosen == 2 and decision.rungs[2].halves["A"].ess_ok
    assert not decision.rungs[2].halves["B"].ess_ok
    assert not decision.adopt and decision.s == 0.0 and decision.next_eps == pytest.approx(0.005)


class _FixedWeights:
    """Each game-side's log weight on its first decision, whatever s."""

    def __init__(self, log_weights, decisions):
        self.decisions = decisions
        self.out = np.zeros(len(log_weights) * decisions)
        self.out[::decisions] = log_weights

    def logp(self, s, halves=None):
        return self.out

    def base_logp(self):
        return np.zeros_like(self.out)


def test_the_kl_is_read_per_side_game():
    """With 50 decisions a side-game: eps 0.008 a decision puts the 2 eps
    rung at K 0.8 a side-game, past the range (and the edge within
    EDGE_MIN_RATIO of the eps rung's K 0.4); and weights of effective-sample
    share 1/3.25 pass the check at K 0.3 a side-game (floor 0.27), which read
    per decision (K 0.006, floor 0.49) they would fail."""
    gains = {"A": {0.5: 0.01, 1.0: 0.02, 2.0: 0.05}, "B": {0.5: 0.0, 1.0: 0.0, 2.0: 0.0}}
    decision = ladder(lambda d: _LadderEvaluator(gains, 0.008, decisions=50), {}, 0.008)
    assert [r.in_range for r in decision.rungs] == [True, True, False]
    assert decision.adopt and decision.chosen == 1

    weights = np.array([0.25] * 160 + [4.0] * 40)                  # mean 1, mean square 3.25
    sides = [GameSide(f"g{i // 2}", 1 + i % 2, 1.0, "A", 50 * i, 50 * (i + 1)) for i in range(200)]
    reading = read_candidate(_FixedWeights(np.log(weights), 50), 1.0, 0.3 / 50, {"A": sides})["A"]
    assert reading.side_kl == pytest.approx(0.3)
    assert reading.ess_share == pytest.approx(1 / 3.25)
    assert reading.ess_ok


def test_halves_are_drawn_per_game():
    sides = [GameSide("g0", 1, 1.0, "A", 0, 1), GameSide("g0", 2, -1.0, "B", 1, 2),
             GameSide("g1", 1, 1.0, "A", 2, 3), GameSide("g1", 2, -1.0, "A", 3, 4)]
    with pytest.raises(ValueError, match="two halves"):
        sides_by_half(sides, ("A", "B"))


class _CubicKL:
    """KL(s) = s^2 d^T F d / 2 + s^3 sum(d^3): a Fisher F, and a cubic term
    the symmetric reading cancels."""

    def __init__(self, F, d):
        self.F, self.d = F, d

    def mean_kl(self, s):
        return 0.5 * s * s * float(self.d @ self.F @ self.d) + s ** 3 * float((self.d ** 3).sum())


def test_span_fisher_reads_every_entry():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(5, 5))
    F = A @ A.T + np.eye(5)
    U = rng.normal(size=(3, 5))
    directions = [{"p": torch.from_numpy(u)} for u in U]
    got = span_fisher(lambda d: _CubicKL(F, d["p"].numpy()), directions, 0.05)
    assert np.allclose(got, U @ F @ U.T, rtol=1e-9, atol=1e-12)


def test_the_pass_s_scale_and_windows():
    """tools/pg_pass.py squares each window's gradient before the 1/2N that
    total carries: a second moment c^2 times larger only divides reduced
    AdamW, Muon and heads-only by c. Its window counts, when present, weight
    the splits' second moments."""
    p = _pass()
    c = 2.0 * 20_000
    scaled = {**p, "sq": [{n: t * c * c for n, t in s.items()} for s in p["sq"]]}
    for arm in ("reduced_adamw", "muon", "heads_only"):
        plain, wide = direction(p, arm), direction(scaled, arm)
        for name in SHAPES:
            assert torch.allclose(wide[name] * c, plain[name], rtol=1e-4, atol=1e-7), (arm, name)
    v = second_moment({**p, "split_windows": [1, 3]})
    for name in SHAPES:
        assert torch.allclose(v[name], 0.25 * p["sq"][0][name] + 0.75 * p["sq"][1][name])


def test_the_paired_ratio_se_matches_its_spread():
    """Two readings sharing most of their per-game noise: the paired ratio's
    SE matches the ratio's spread over many draws (an unpaired SE would be
    about twice it)."""
    rng = np.random.default_rng(5)
    games = [f"g{i // 2}" for i in range(400)]
    ratios, ses = [], []
    for _ in range(2000):
        common = rng.normal(size=400)
        a = clustered_mean(1.0 + common + 0.1 * rng.normal(size=400), games)
        b = clustered_mean(2.0 + common + 0.1 * rng.normal(size=400), games)
        ratio, se = paired_ratio(a, b)
        ratios.append(ratio)
        ses.append(se)
    assert 0.85 < math.sqrt(np.mean(np.square(ses))) / np.std(ratios) < 1.15


def test_the_average_weighs_each_direction_once_at_unit_norm():
    """Prereg departure 37: each direction scaled to unit norm, the mean of
    them, over the first's names; a direction lacking one, or zero, is
    refused."""
    from wesnoth_ai.pg_step import averaged, norm
    a = {"x": torch.tensor([3.0, 0.0]), "y": torch.tensor([4.0])}
    b = {"x": torch.tensor([0.0, 10.0]), "y": torch.tensor([0.0]), "z": torch.tensor([1.0])}
    avg = averaged([a, b])
    assert set(avg) == {"x", "y"}
    assert torch.allclose(avg["x"], torch.tensor([0.3, 0.5])) and torch.allclose(avg["y"], torch.tensor([0.4]))
    assert norm(averaged([a])) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="lacks"):
        averaged([b, a])
    with pytest.raises(ValueError, match="zero"):
        averaged([a, {"x": torch.zeros(2), "y": torch.zeros(1)}])


class _Jointly(_LadderEvaluator):
    """A `_LadderEvaluator` that also takes `prefetch_jointly`, recording
    every call in `log` by the evaluator's name and the steps' KLs."""

    def __init__(self, name, log, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name, self.log = name, log

    def mean_kl(self, s):
        self.log.append(("kl", self.name))
        return super().mean_kl(s)

    def prefetch(self, steps, halves):
        self.log.append(("prefetch", self.name, [s * s for s in steps], halves))

    def prefetch_jointly(self, requests, halves):
        self.log.append(("jointly", [(e.name, [s * s for s in steps]) for e, steps in requests], halves))


def _paired(gains_plain, gains_average, eps=0.01, log=None, cls=_LadderEvaluator):
    """The paired ladder over a plain direction (norm 2) and an averaged one
    (norm 1), each with its evaluator."""
    from wesnoth_ai.pg_step import AVERAGE_MULTIPLIERS, LADDER_MULTIPLIERS, paired_ladder
    log = [] if log is None else log
    make = (lambda name, gains: cls(name, log, gains, eps)) if cls is _Jointly else (lambda name, gains: cls(gains, eps))
    evaluators = {2.0: make("plain", gains_plain), 1.0: make("average", gains_average)}
    directions = {"plain": {"x": torch.tensor([2.0])}, "average": {"x": torch.tensor([1.0])}}
    return paired_ladder(lambda d: evaluators[float(d["x"])], directions, eps,
                         {"plain": LADDER_MULTIPLIERS, "average": AVERAGE_MULTIPLIERS}, s0=0.5)


def test_the_paired_ladder_selects_across_both_directions_and_confirms_the_winner():
    """The rung of highest ΔĴ on half A among both directions' rungs is read
    on half B, and decide reads that direction's rungs; the other's rungs
    are read on half A alone. A tie goes to the plain direction, and the
    decision with no average to win is the plain ladder's own."""
    plain = {"A": {0.5: 0.01, 1.0: 0.03, 2.0: 0.02}, "B": {0.5: 0.0, 1.0: 0.02, 2.0: 0.0}}
    better = {"A": {1.0: 0.05}, "B": {1.0: 0.04}}
    won = _paired(plain, better)
    assert won.chosen == "average" and won.decision.adopt and won.decision.chosen == 0
    assert won.decision.rungs == won.rungs["average"] and len(won.rungs["average"]) == 1
    assert won.rungs["average"][0].multiplier == 1.0 and set(won.rungs["average"][0].halves) == {"A", "B"}
    assert all(set(r.halves) <= {"A"} for r in won.rungs["plain"])
    assert won.decision.s == pytest.approx(math.sqrt(0.01), rel=KL_RTOL)
    assert won.summary()["chosen"] == "average" and len(won.summary()["rungs"]["plain"]) == 3

    lost = _paired(plain, {"A": {1.0: 0.01}, "B": {1.0: 0.04}})
    assert lost.chosen == "plain" and lost.decision.summary() == _ladder(plain).summary()
    assert set(lost.rungs["average"][0].halves) == {"A"}
    tied = _paired(plain, {"A": {1.0: 0.03}, "B": {1.0: 0.04}})               # the same games, gains and noise
    assert tied.rungs["average"][0].halves["A"].exact.estimate == tied.rungs["plain"][1].halves["A"].exact.estimate
    assert tied.chosen == "plain" and tied.decision.summary() == _ladder(plain).summary()


def test_the_paired_ladder_searches_both_then_reads_half_a_in_one_joint_prefetch():
    """Every rung of both directions is searched before any reading; the
    selection half is prefetched once for both, the confirmation half for
    the winner alone, and the decision is the one evaluators without
    prefetches give."""
    plain = {"A": {0.5: 0.01, 1.0: 0.03, 2.0: 0.02}, "B": {0.5: 0.0, 1.0: 0.02, 2.0: 0.0}}
    better = {"A": {1.0: 0.05}, "B": {1.0: 0.04}}
    log = []
    decision = _paired(plain, better, log=log, cls=_Jointly)
    first = next(i for i, c in enumerate(log) if c[0] != "kl")
    assert {c[0] for c in log[:first]} == {"kl"} and all(c[0] != "kl" for c in log[first:])
    joint = [c for c in log if c[0] == "jointly"]
    assert len(joint) == 1 and joint[0][2] == ("A",)
    (n1, kl1), (n2, kl2) = joint[0][1]
    assert (n1, n2) == ("plain", "average")
    assert kl1 == pytest.approx([0.005, 0.01, 0.02], rel=KL_RTOL) and kl2 == pytest.approx([0.01], rel=KL_RTOL)
    assert [c[1:] for c in log if c[0] == "prefetch"] == [("average", pytest.approx([0.01], rel=KL_RTOL), ("B",))]
    assert decision.chosen == "average"
    assert decision.summary() == _paired(plain, better).summary()
