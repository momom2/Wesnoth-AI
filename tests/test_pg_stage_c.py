"""Stage C of the policy-gradient program (tools/pg_stage_c.py) and the
step's estimators on the enumerable toy game (tests/helpers/toy_game.py):
the estimators read its exact gains and fail without the seat sign; the
driver reads every part of its readout, the critical batch against the toy's
true B; and on hand-built inputs, the critical batch and the floor with
their SEs, the optimizer criterion and the gate."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from helpers import toy_game  # noqa: E402
from tools.pg_stage_c import (CriticalBatch, RoundPrices, choose_arm, critical_batch, fisher_from_pass,  # noqa: E402
                              floor, floor_reading, round_size_for, run_stage_c, slopes_from_pass, stage_c_gate)
from wesnoth_ai.pg_protocol import Evaluator  # noqa: E402
from wesnoth_ai.pg_step import (ARMS, KL_RTOL, LADDER_MULTIPLIERS, Decision, HalfReading, Reading,  # noqa: E402
                                Rung, clustered_mean, direction, exact_gain, first_order_gain, kl_search,
                                paired_difference, pooled_gradient, second_moment)


@pytest.fixture(scope="module")
def toy():
    game, params = toy_game.ToyGame(0), toy_game.initial_params(0)
    train = game.sample(params, 20_000, seed=1)
    heldout = game.sample(params, 20_000, seed=2)
    return game, params, toy_game.toy_pass(params, train), heldout


def _read(evaluator, s):
    log_ratio = evaluator.logp(s) - evaluator.base_logp()
    sides = evaluator.game_sides()
    return exact_gain(log_ratio, sides), first_order_gain(log_ratio, sides)


def test_the_estimators_read_the_exact_gains_and_need_the_seat_sign(toy):
    """ΔĴ reads the exact gain of a seat switched to the candidate and Ĝ the
    exact first-order (per-decision switch) gain, each within 3 SE; with the
    seat sign sigma dropped from R, ΔĴ misses by more than 3 SE."""
    game, params, pass_, heldout = toy
    d = direction(pass_, "reduced_adamw")
    evaluator = toy_game.ToyEvaluator(params, heldout, d)
    assert isinstance(evaluator, Evaluator)
    point = kl_search(evaluator.mean_kl, 0.05)
    assert abs(point.kl - 0.05) <= KL_RTOL * 0.05
    candidate = toy_game.step_params(params, d, point.s)
    exact, surrogate = game.exact_gain(params, candidate), game.exact_surrogate(params, candidate)
    dJ, G = _read(evaluator, point.s)
    assert abs(dJ.estimate - exact) < 3 * dJ.se
    assert abs(G.estimate - surrogate) < 3 * G.se
    # Paired by game-side the two readings resolve their estimands' gap (the
    # step's second-order terms), so a W summed rather than multiplied fails.
    gap = paired_difference(dJ, G)
    assert abs(gap.estimate - (exact - surrogate)) < 3 * gap.se
    assert abs(exact - surrogate) > 3 * gap.se

    unsigned, _ = _read(toy_game.ToyEvaluator(params, heldout, d, drop_sign=True), point.s)
    assert abs(unsigned.estimate - exact) > 3 * unsigned.se


def test_stage_c_reads_every_part_on_the_toy(toy, tmp_path):
    game, params, pass_, heldout = toy
    out = tmp_path / "stage_c.json"
    readout = run_stage_c(pass_, params, toy_game.factory(params, heldout), out, k_side=0.1,
                          prices=RoundPrices(7.76e-5, 2.42))
    assert json.loads(out.read_text())["gate"] == readout["gate"]

    eps = readout["heldout"]["eps"]
    assert eps == pytest.approx(0.1 / 2)                              # two decisions a side-game
    assert set(readout["arms"]) == set(ARMS)
    for arm, reading in readout["arms"].items():
        assert abs(reading["kl"] - eps) <= KL_RTOL * eps, arm
    assert readout["optimizer"]["winner"] in ARMS
    assert [r["target_kl"] for r in readout["ladder"]["rungs"]] == pytest.approx(
        [m * eps for m in LADDER_MULTIPLIERS])
    assert readout["gate"]["verdict"] == "resolved"                   # the toy's gradient is strong

    # The Fisher forms read through the KL against the toy's Fisher matrix.
    # At the toy's KL eps/4 = 0.0125 a decision the KL's quartic term leaves
    # about 0.4% in a form (at the network's eps/4, about 2.6e-4, far less).
    F = toy_game.fisher_matrix(params, heldout)
    head = toy_game.flatten(toy_game.head_direction(params))
    assert readout["span"]["fisher"][0][0] == pytest.approx(head @ F @ head, rel=1e-3)
    v = toy_game.flatten(second_moment(pass_))
    P = np.where(v > 0, 1.0 / np.sqrt(np.where(v > 0, v, 1.0)), 0.0)
    M = P[:, None] * F * P[None, :]
    total = toy_game.flatten(pass_["total"])
    batch = readout["critical_batch"]["reduced_adamw"]
    assert batch["q_total"] == pytest.approx(total @ M @ total, rel=1e-2)
    for i in range(8):
        split = toy_game.flatten(pooled_gradient(pass_, [i]))
        assert batch["q_splits"][i] == pytest.approx(split @ M @ split, rel=2e-2)
        assert batch["q_deviations"][i] == pytest.approx((split - total) @ M @ (split - total), rel=2e-2)

    # The toy's true B, from the exact per-game gradient's moments.
    g, Sigma = game.gradient_moments(params)
    true_noise, true_signal = np.trace(M @ Sigma), g @ M @ g
    assert batch["status"] == "resolved"
    assert abs(batch["noise"] - true_noise) < 3 * batch["noise_se"]
    assert abs(batch["signal"] - true_signal) < 3 * batch["signal_se"]
    assert abs(batch["B"] - true_noise / true_signal) < 3 * batch["B_se"]

    # The gain curve's points read the exact first-order gains of their steps,
    # and their ratios to the whole pass's the exact ratios.
    full_s = readout["arms"]["reduced_adamw"]["s"]
    full = game.exact_surrogate(params, toy_game.step_params(params, direction(pass_, "reduced_adamw"), full_s))
    points = readout["gain_curve"]["points"]
    assert [p["games"] for p in points] == [5_000, 10_000]
    for p in points:
        d = direction(pass_, "reduced_adamw", gradient=pooled_gradient(pass_, range(p["splits"])))
        exact = game.exact_surrogate(params, toy_game.step_params(params, d, p["s"]))
        assert abs(p["first_order"]["estimate"] - exact) < 3 * p["first_order"]["se"]
        assert abs(p["ratio"] - exact / full) < 3 * p["ratio_se"]
        assert 0 < p["model_ratio"] < 1

    floor_read = readout["floor"]
    assert 0 < floor_read["floor_debiased"] <= floor_read["floor"] and floor_read["floor_se"] > 0

    # Each split's gradient is on the total's per-game scale: weighted by
    # their games the splits average to the total, and all eight pool to it.
    games = pass_["split_games"]
    mean = {n: sum(k / sum(games) * pooled_gradient(pass_, [i])[n] for i, k in enumerate(games))
            for n in pass_["names"]}
    whole = pooled_gradient(pass_, range(8))
    for n in pass_["names"]:
        assert torch.allclose(mean[n], pass_["total"][n], rtol=1e-4, atol=1e-8)
        assert torch.allclose(whole[n], pass_["total"][n], rtol=1e-5, atol=1e-9)


def test_another_arm_replaces_the_default_only_beyond_two_paired_se():
    """Paired by game, beyond 2 SE, and only when both arms' importance
    weights pass the check."""
    rng = np.random.default_rng(0)
    games = [f"g{i // 2}" for i in range(400)]
    base = rng.normal(size=400)

    def arm(shift):
        return clustered_mean(base + shift + rng.normal(size=400) * 0.5, games)

    default = clustered_mean(base, games)
    all_readable = dict.fromkeys(ARMS, True)
    close = {"reduced_adamw": default, "sgd": arm(0.0), "muon": arm(0.03)}
    winner, table, _ = choose_arm(close, all_readable)
    assert winner == "reduced_adamw" and not any(row["beyond"] for row in table.values())
    assert 0 < table["muon"]["z"] < 2

    far = {"reduced_adamw": default, "sgd": arm(0.4), "muon": arm(0.8), "heads_only": arm(-0.8)}
    winner, table, _ = choose_arm(far, all_readable)
    assert winner == "muon" and table["sgd"]["beyond"] and not table["heads_only"]["beyond"]
    winner, table, _ = choose_arm(far, {**all_readable, "muon": False})
    assert winner == "sgd" and not table["muon"]["beyond"]
    winner, table, reason = choose_arm(far, {**all_readable, "reduced_adamw": False})
    assert winner == "reduced_adamw" and table == {} and "fail" in reason


def _gaussian_splits(rng, g, root, n):
    splits = g + (rng.normal(size=(len(n), len(g))) @ root.T) / np.sqrt(n)[:, None]
    return splits, (n[:, None] * splits).sum(axis=0) / n.sum()


def test_critical_batch_recovers_B_and_its_se_on_a_gaussian_noise_model():
    """Split gradients drawn as N(g, Sigma / n_i): over many draws the
    signal and noise read g^T M g and tr(M Sigma), M = PFP, their ratio the
    true B, and each SE matches its reading's spread."""
    rng = np.random.default_rng(0)
    dim, n = 12, np.array([2400, 2600, 2500, 2500, 2450, 2550, 2500, 2500])
    P, F = np.diag(rng.uniform(0.5, 2.0, dim)), np.diag(rng.uniform(0.2, 1.5, dim))
    g = rng.normal(size=dim) * 0.02
    A = rng.normal(size=(dim, dim))
    Sigma = A @ A.T / dim
    M = P @ F @ P
    true_signal, true_noise = g @ M @ g, np.trace(M @ Sigma)
    root = np.linalg.cholesky(Sigma)
    reads = []
    for _ in range(3000):
        splits, total = _gaussian_splits(rng, g, root, n)
        result = critical_batch([x @ M @ x for x in splits], [(x - total) @ M @ (x - total) for x in splits],
                                total @ M @ total, n)
        reads.append((result.signal, result.signal_se, result.noise, result.noise_se))
    signal, signal_se, noise, noise_se = np.array(reads).T
    for got, se, want in ((signal, signal_se, true_signal), (noise, noise_se, true_noise)):
        assert abs(got.mean() - want) < 3 * got.std() / math.sqrt(len(got))
        assert 0.8 < math.sqrt(np.mean(se ** 2)) / got.std() < 1.25
    assert abs(noise.mean() / signal.mean() / (true_noise / true_signal) - 1) < 0.05


def test_an_unresolved_part_leaves_B_unknown():
    """No noise (identical splits) or no signal leaves B NaN with its status,
    never B = 0."""
    no_noise = critical_batch([1.0] * 8, [0.0] * 8, 1.0, [2500] * 8)
    assert no_noise.status == "noise_unresolved" and math.isnan(no_noise.B)
    no_signal = critical_batch([0.5] * 8, [0.4] * 8, 0.05, [2500] * 8)       # signal 0.05 - 2800/20000 < 0
    assert no_signal.status == "signal_unresolved" and math.isnan(no_signal.B)


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


def test_the_floor_s_noise_and_se():
    """Over many draws of the slopes the debiased quadratic reads the true
    s^T F^-1 s, the raw one sits above it, and the quadratic's and the floor's
    SEs match their spreads."""
    rng = np.random.default_rng(2)
    n = np.array([500] * 8)
    true_s, F_s, K = np.array([0.06, -0.03]), np.array([[1.0, 0.2], [0.2, 0.5]]), 0.25
    root = np.linalg.cholesky(np.array([[0.4, 0.1], [0.1, 0.3]]))
    truth = true_s @ np.linalg.solve(F_s, true_s)
    reads = [floor_reading(total, splits, n, F_s, K)
             for splits, total in (_gaussian_splits(rng, true_s, root, n) for _ in range(4000))]
    raw = np.array([r.quadratic for r in reads])
    debiased = raw - np.array([r.quadratic_noise for r in reads])
    assert abs(debiased.mean() - truth) < 3 * debiased.std() / math.sqrt(len(debiased))
    assert raw.mean() - truth > 5 * raw.std() / math.sqrt(len(raw))
    for got, se in ((raw, [r.quadratic_se for r in reads]), ([r.floor for r in reads], [r.floor_se for r in reads])):
        assert 0.8 < math.sqrt(np.mean(np.square(se))) / np.std(got) < 1.25


def _decision(estimate, se, ess_ok=True):
    reading = Reading(estimate, se, ("g0", "g1"), np.zeros(2), np.ones(2))
    half = HalfReading(reading, reading, 1.0, 0.1, ess_ok)
    rung = Rung(1.0, 0.01, 0.1, 0.01, True, {"A": half, "B": half})
    return Decision(True, 0.1, 0.01, 0.01, 0, "", [rung])


def test_round_size_reaches_the_target_along_the_gain_curve():
    size = round_size_for(1.5, 20_000, 20_000)
    assert size == pytest.approx(160_000)
    assert 1.5 * math.sqrt(size * 40_000 / (20_000 * (size + 20_000))) == pytest.approx(2.0)
    assert math.isinf(round_size_for(0.5, 20_000, 20_000))       # the asymptote z 0.71 stays below 2
    assert math.isinf(round_size_for(-1.0, 20_000, 20_000))


def test_the_gate_reads_design_section_6():
    """Resolved at a lower 2-SE bound above 0; else rounds at the size the
    gain curve resolves within the budget; else neither (over budget, B
    unresolved, no prices, half B's weights failing, no rung)."""
    batch = CriticalBatch(20_000.0, 1_000.0, 0.01, 0.001, 200.0, 10.0, "resolved")
    prices = RoundPrices(7.76e-5, 2.42)
    assert stage_c_gate(_decision(0.05, 0.02), "B", batch, 20_000, prices)["verdict"] == "resolved"

    sized = stage_c_gate(_decision(0.03, 0.02), "B", batch, 20_000, prices)       # z 1.5
    assert sized["verdict"] == "rounds_at_size" and sized["round_size"] == 160_000
    assert sized["round_cost"] == pytest.approx(2.42 + 7.76e-5 * 160_000)

    unresolved = CriticalBatch(math.nan, math.nan, 0.01, 0.001, 1.0, 10.0, "noise_unresolved")
    for gate, why in ((stage_c_gate(_decision(0.03, 0.02), "B", batch, 20_000,
                                    RoundPrices(7.76e-5, 2.42, budget=10.0)), "over $10"),
                      (stage_c_gate(_decision(0.03, 0.02), "B", unresolved, 20_000, prices), "noise_unresolved"),
                      (stage_c_gate(_decision(0.03, 0.02), "B", batch, 20_000, None), "no prices"),
                      (stage_c_gate(_decision(0.05, 0.02, ess_ok=False), "B", batch, 20_000, prices), "weights")):
        assert gate["verdict"] == "neither" and why in gate["reason"], gate
    none = Decision(False, 0.0, 0.01, 0.005, None, "no rung readable by importance weights", [])
    assert stage_c_gate(none, "B", batch, 20_000, prices)["verdict"] == "neither"


def test_the_head_fisher_comes_from_the_pass_in_the_head_order():
    """tools/pg_pass.py's head readings, keyed by its own head names and in
    its own order, reach the floor per decision and in pg_step's head
    order; a pass without them gives None."""
    heads = ["actor_head", "type_head", "target_q_proj", "weapon_head.2"]
    F_side = np.array([[4.0, 0.1, 0.2, 0.3], [0.1, 3.0, 0.4, 0.5], [0.2, 0.4, 2.0, 0.6], [0.3, 0.5, 0.6, 1.0]])
    pass_ = {"games": 100, "decisions": 200 * 50, "head_fisher": F_side.tolist(),     # 50 decisions a side-game
             "head_slopes": {"weapon": 4.0, "target": 3.0, "type": 2.0, "actor": 1.0}}
    order = [3, 2, 1, 0]                                    # pg_step's heads in the pass's order
    assert np.allclose(fisher_from_pass(pass_, heads), F_side[np.ix_(order, order)] / 50)
    assert slopes_from_pass(pass_, heads) == [1.0, 2.0, 3.0, 4.0]
    assert fisher_from_pass({"games": 100, "decisions": 10_000}, heads) is None
    assert fisher_from_pass(pass_, heads[:3]) is None
