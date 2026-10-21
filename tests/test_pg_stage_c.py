"""The policy-gradient step on the enumerable toy game
(tests/helpers/toy_game.py): the production estimators read its exact gains
and fail without the seat sign, and the stage C driver (tools/pg_stage_c.py)
reads every part of its readout through the Evaluator protocol."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from helpers import toy_game  # noqa: E402
from tools.pg_stage_c import choose_arm, run_stage_c  # noqa: E402
from wesnoth_ai.pg_protocol import Evaluator  # noqa: E402
from wesnoth_ai.pg_step import (ARMS, KL_RTOL, LADDER_MULTIPLIERS, clustered_mean, direction,  # noqa: E402
                                exact_gain, first_order_gain, kl_search, paired_difference, pooled_gradient)


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


def _exact_fisher(evaluator, d):
    """d^T F d per decision in the toy: the mean over held-out decisions of
    the variance under pi_k of the logits' derivative along d."""
    w, table = evaluator.params[toy_game.HEAD].item(), evaluator.params[toy_game.TABLE].numpy()
    dw, dtable = d[toy_game.HEAD].double().item(), d[toy_game.TABLE].double().numpy()
    slope = (dw * table + w * dtable)[evaluator.states]
    probs = np.exp(evaluator.base[evaluator.states])
    mean = (probs * slope).sum(axis=1, keepdims=True)
    return float((probs * (slope - mean) ** 2).sum(axis=1).mean())


def test_stage_c_reads_every_part_on_the_toy(toy, tmp_path):
    _, params, pass_, heldout = toy
    factory = toy_game.factory(params, heldout)
    out = tmp_path / "stage_c.json"
    readout = run_stage_c(pass_, params, factory, out, k_side=0.1)
    assert json.loads(out.read_text())["floor"] == readout["floor"]

    eps = readout["heldout"]["eps"]
    assert eps == pytest.approx(0.1 / 2)                              # two decisions a side-game
    assert set(readout["arms"]) == set(ARMS)
    for arm, reading in readout["arms"].items():
        assert abs(reading["kl"] - eps) <= KL_RTOL * eps, arm
    assert readout["optimizer"]["winner"] in ARMS
    rungs = readout["ladder"]["rungs"]
    assert [r["target_kl"] for r in rungs] == pytest.approx([m * eps for m in LADDER_MULTIPLIERS])

    head = toy_game.head_direction(params)
    assert readout["span"]["fisher"][0][0] == pytest.approx(
        _exact_fisher(toy_game.ToyEvaluator(params, heldout, head), head), rel=1e-3)
    batch = readout["critical_batch"]["reduced_adamw"]
    d = direction(pass_, "reduced_adamw")
    # Read at the toy's KL eps/4 = 0.0125 a decision, the KL's quartic term
    # leaves about 0.4% in q (at the network's eps/4, about 2.6e-4, far less).
    assert batch["q_total"] == pytest.approx(_exact_fisher(toy_game.ToyEvaluator(params, heldout, d), d), rel=1e-2)
    assert len(batch["q_splits"]) == 8 and batch["B"] >= 0
    assert [p["games"] for p in readout["gain_curve"]["points"]] == [5_000, 10_000]
    assert 0 < readout["floor"]["floor_debiased"] <= readout["floor"]["floor"]

    whole = pooled_gradient(pass_, range(8))
    assert all(torch.allclose(whole[n], pass_["total"][n], rtol=1e-5, atol=1e-9) for n in pass_["names"])


def test_another_arm_replaces_the_default_only_beyond_two_paired_se():
    rng = np.random.default_rng(0)
    games = [f"g{i // 2}" for i in range(400)]
    base = rng.normal(size=400)

    def arm(shift):
        return clustered_mean(base + shift + rng.normal(size=400) * 0.5, games)

    default = clustered_mean(base, games)
    close = {"reduced_adamw": default, "sgd": arm(0.0), "muon": arm(0.03)}
    winner, table = choose_arm(close)
    assert winner == "reduced_adamw" and not any(row["beyond"] for row in table.values())
    assert 0 < table["muon"]["z"] < 2

    far = {"reduced_adamw": default, "sgd": arm(0.4), "muon": arm(0.8), "heads_only": arm(-0.8)}
    winner, table = choose_arm(far)
    assert winner == "muon" and table["sgd"]["beyond"] and not table["heads_only"]["beyond"]
