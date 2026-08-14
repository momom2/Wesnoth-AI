"""TCS rung-1 probe: unit + end-to-end tests driving the production
probe code paths (tools/turn_counterfactual_probe.py) on real sims
and a real (tiny, random-init) policy -- no mirrored logic.

Covers the contracts docs/tcs_spec.md par.3/par.8 relies on:
  * spine recording invariants (forks are pre-action, legal lists
    populated, terminal end_turn semantics);
  * materialization = grade-what-you-commit (identity replay lands
    fully; a doubled command bounces cleanly, never crashes; every
    variant terminates at a real boundary);
  * the pure decision helpers (two-stage acceptance, Gumbel-top-k
    alternative sampling, KL target math, spearman);
  * probe_state end-to-end on a mini scenario, real + placebo arms,
    JSON-serializable output.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_test_helpers import fresh_scenario_sim  # noqa: E402
from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402
from tools.mcts import MCTSConfig  # noqa: E402
from tools.turn_counterfactual_probe import (  # noqa: E402
    ProbeConfig, gumbel_top_k_alternatives, materialize, probe_state,
    record_spine, spearman, tcs_target_kl, two_stage_accept,
)


@pytest.fixture(scope="module")
def tiny_policy():
    torch.manual_seed(0)
    return TransformerPolicy(device=torch.device("cpu"), d_model=32,
                             num_layers=1, num_heads=4, d_ff=64)


@pytest.fixture(scope="module")
def mini_sim():
    return fresh_scenario_sim(seed=3, max_turns=6, mini=True)


def test_record_spine_invariants(tiny_policy, mini_sim):
    rng = np.random.default_rng(0)
    side = mini_sim.gs.global_info.current_side
    steps, boundary = record_spine(tiny_policy, mini_sim, side, 0, rng,
                                   max_spine=12)
    assert steps, "spine must record at least one coordinate"
    for st in steps:
        assert st.legal, "every coordinate carries its full legal list"
        assert st.pre_fork.gs.global_info.current_side == side, \
            "pre_fork must snapshot the state BEFORE the action"
        assert math.isfinite(st.pre_value)
        assert st.legal[st.action_idx].action == st.action
    # The walk ends by end_turn, side switch, game end, or the cap.
    ended = (steps[-1].action.get("type") == "end_turn"
             or boundary.done
             or boundary.gs.global_info.current_side != side
             or len(steps) == 12)
    assert ended
    # The probed sim itself is untouched (probe works on forks only).
    assert mini_sim.gs.global_info.current_side == side
    assert not mini_sim.done


def test_materialize_identity_and_bounce(tiny_policy, mini_sim):
    rng = np.random.default_rng(1)
    side = mini_sim.gs.global_info.current_side
    steps, _ = record_spine(tiny_policy, mini_sim, side, 0, rng,
                            max_spine=8)
    incumbent = [s.action for s in steps]

    # Identity replay: every incumbent command lands (turn-1 mini map:
    # no contact, so bounces would indicate a replay defect).
    m = materialize(tiny_policy, mini_sim, side, incumbent,
                    "test:salt", 0)
    assert not m.invalid
    assert m.survival == 1.0
    assert math.isfinite(m.value)
    assert -1.0 <= m.value <= 1.0

    # Doubling a non-end_turn command: the duplicate must bounce
    # cleanly (unit already moved / hex taken), never crash, and the
    # turn still terminates at a real boundary.
    non_et = [a for a in incumbent if a.get("type") != "end_turn"]
    if non_et:  # a policy could legitimately open with end_turn
        doubled = [non_et[0], non_et[0]] + incumbent[1:]
        m2 = materialize(tiny_policy, mini_sim, side, doubled,
                         "test:salt2", 0)
        assert m2.invalid or m2.accepted < m2.attempted
        if not m2.invalid:
            assert math.isfinite(m2.value)


def test_two_stage_accept_math():
    # Clear signal, zero spread: accepted.
    ok, mean, thr = two_stage_accept(np.array([0.2, 0.2, 0.2]), 0.01)
    assert ok and mean == pytest.approx(0.2) and thr == 0.01
    # Pure noise straddling zero: rejected (threshold = 2*sd/sqrt(n)).
    ok, _, thr = two_stage_accept(np.array([0.30, -0.25, 0.05]), 0.01)
    assert not ok and thr > 0.01
    # Deterministic single replicate: floor gates it.
    ok, _, _ = two_stage_accept(np.array([0.005]), 0.01)
    assert not ok
    ok, _, _ = two_stage_accept(np.array([0.05]), 0.01)
    assert ok


def test_gumbel_top_k_alternatives():
    rng = np.random.default_rng(2)
    priors = np.array([0.5, 0.3, 0.15, 0.04, 0.01])
    picks = gumbel_top_k_alternatives(priors, exclude_idx=0,
                                      end_turn_idx=4, k=3, rng=rng)
    assert len(picks) == 3
    assert 0 not in picks, "incumbent's choice must be excluded"
    assert 4 in picks, "end_turn is force-included (spec par.3)"
    assert gumbel_top_k_alternatives(np.array([1.0]), 0, None, 3,
                                     rng) == []


def test_tcs_target_kl_transform():
    cfg = MCTSConfig()
    priors = np.array([0.4, 0.3, 0.2, 0.1])
    # Uninformative: all evaluated values equal. The 0.04 rescale
    # floor must fade the target to the prior (KL ~ 0) -- the exact
    # behavior the 2026-08-12 fix installed.
    kl0 = tcs_target_kl(priors, np.full(4, 0.1),
                        np.array([True] * 4), 0.1, 16.0, 1.0, cfg)
    assert kl0 == pytest.approx(0.0, abs=1e-9)
    # One decisively better action: target must move off the prior.
    vals = np.array([0.0, 0.5, 0.0, 0.0])
    kl1 = tcs_target_kl(priors, vals, np.array([True] * 4), 0.0,
                        16.0, 1.0, cfg)
    assert kl1 > 0.1
    # Unevaluated actions complete at v_mix (never at raw 0): with
    # only the good action evaluated, KL is still positive and finite.
    kl2 = tcs_target_kl(priors, vals,
                        np.array([False, True, False, False]), 0.0,
                        16.0, 1.0, cfg)
    assert 0.0 < kl2 < 10.0


def test_spearman():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(x, x * 10) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)
    assert math.isnan(spearman(x[:2], x[:2]))


def test_probe_state_end_to_end(tiny_policy, mini_sim):
    rng = np.random.default_rng(4)
    side = mini_sim.gs.global_info.current_side
    cfg = ProbeConfig(n_alt=2, rounds=1, reval_salts=2,
                      baseline="first", baseline_sims=4,
                      max_spine=6)
    rec = probe_state(tiny_policy, mini_sim, side, cfg, rng, "t0")
    assert not rec.get("empty")
    assert rec["K_spine"] >= 1
    assert len(rec["rounds"]) <= 1
    for rd in rec["rounds"]:
        assert rd["n_variants"] >= 1
        assert math.isfinite(rd["naive_delta"])
        assert rd["reval_n"] >= 1
        assert isinstance(rd["accepted"], bool)
    # KL lists cover the final incumbent's coordinates.
    assert len(rec["kl_own"]) == len(rec["kl_matched"])
    assert all(math.isfinite(k) for k in rec["kl_own"])
    # Baseline arm ran on the first coordinate.
    assert len(rec["baseline_kl"]) == 1
    assert math.isfinite(rec["baseline_kl"][0])
    # Every variant tuple is (delta, survival, stochastic, vis_changed).
    for t in rec["variants"]:
        assert len(t) == 4
        assert 0.0 <= t[1] <= 1.0
    # The whole record must serialize (it is written as JSONL).
    json.dumps(rec)

    # Placebo arm: same machinery, shuffled selection.
    rec_p = probe_state(tiny_policy, mini_sim, side, cfg, rng, "t0p",
                        placebo=True)
    assert rec_p["placebo"] is True
    json.dumps(rec_p)
