"""Multi-turn projection tests (TCS, docs/tcs_spec.md par.3;
user directive 2026-08-17) -- production code paths only.

Covers:
  * depth-0 identity: project_value == boundary_value (the default-
    off contract: projection disabled changes nothing);
  * forced-close parity: with max_actions=0 every projected half-turn
    is exactly one end_turn, so H=2 must equal grading a manual
    end_turn/end_turn fork -- this pins the half-turn advance AND the
    perspective flip (odd H grades with the opponent to move);
  * fork isolation: the live sim is never mutated by projection;
  * placement: project="reval" runs projections in stage 2 while
    "none" runs zero; TurnCommitPolicy accumulates and drains the
    tcs_projections counter under project="all".
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_test_helpers import fresh_scenario_sim  # noqa: E402
from wesnoth_ai.classes import state_key  # noqa: E402
from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402
from tools.mcts import MCTSConfig  # noqa: E402
from tools.turn_policy import TurnCommitPolicy  # noqa: E402
from tools.turn_search import (  # noqa: E402
    TurnSearchConfig, boundary_value, plan_turn, project_value,
)


def _base_policy() -> TransformerPolicy:
    torch.manual_seed(0)
    return TransformerPolicy(device=torch.device("cpu"), d_model=32,
                             num_layers=1, num_heads=4, d_ff=64)


def _cfg(**kw) -> TurnSearchConfig:
    defaults = dict(n_alt=2, rounds=1, fast_rounds=0, reval_salts=2,
                    max_spine=6, turn_full_prob=1.0)
    defaults.update(kw)
    return TurnSearchConfig(**defaults)


def test_project_depth0_equals_boundary_value():
    sim = fresh_scenario_sim(seed=3, max_turns=6, mini=True)
    policy = _base_policy()
    side = sim.gs.global_info.current_side
    rng = np.random.default_rng(0)
    v0 = boundary_value(policy, sim, side, 0)
    vp = project_value(policy, sim, side, 0, half_turns=0,
                       max_actions=40, rng=rng)
    assert vp == pytest.approx(v0, abs=1e-9)


def test_project_forced_close_parity_and_perspective():
    """max_actions=0 makes each projected half-turn exactly one forced
    end_turn: H=1 and H=2 must equal grading manual end_turn forks.
    The H=1 case grades with the OPPONENT to move, so it also pins
    `_value_for`'s perspective flip."""
    sim = fresh_scenario_sim(seed=3, max_turns=6, mini=True)
    policy = _base_policy()
    side = sim.gs.global_info.current_side
    r = sim.fork()
    r.step({"type": "end_turn"})
    expect_h1 = boundary_value(policy, r, side, 0)
    r2 = r.fork()
    r2.step({"type": "end_turn"})
    expect_h2 = boundary_value(policy, r2, side, 0)
    got_h1 = project_value(policy, sim, side, 0, half_turns=1,
                           max_actions=0,
                           rng=np.random.default_rng(0))
    got_h2 = project_value(policy, sim, side, 0, half_turns=2,
                           max_actions=0,
                           rng=np.random.default_rng(0))
    assert got_h1 == pytest.approx(expect_h1, abs=1e-9)
    assert got_h2 == pytest.approx(expect_h2, abs=1e-9)
    # Sanity: the two boundary states genuinely differ (side to move
    # flips), so the parity above is not a trivial equality.
    assert state_key(r.gs) != state_key(r2.gs)


def test_projection_never_mutates_live_sim():
    sim = fresh_scenario_sim(seed=3, max_turns=6, mini=True)
    policy = _base_policy()
    side = sim.gs.global_info.current_side
    key0 = state_key(sim.gs)
    rng0 = sim._rng_requests
    project_value(policy, sim, side, 0, half_turns=3, max_actions=4,
                  rng=np.random.default_rng(1))
    assert state_key(sim.gs) == key0
    assert sim._rng_requests == rng0


def test_placement_reval_projects_and_none_does_not():
    sim = fresh_scenario_sim(seed=3, max_turns=6, mini=True)
    policy = _base_policy()
    side = sim.gs.global_info.current_side
    mcts_cfg = MCTSConfig()
    plan_off = plan_turn(policy, sim, side, 0, _cfg(),
                         mcts_cfg, np.random.default_rng(0), "t0",
                         full=True)
    assert plan_off.projections == 0
    plan_on = plan_turn(policy, sim, side, 0,
                        _cfg(project="reval", project_halfturns=1,
                             project_max_actions=2),
                        mcts_cfg, np.random.default_rng(0), "t1",
                        full=True)
    assert plan_on.commands, "projection must still yield a plan"
    # Stage 2 always runs under projection (the deterministic-pair
    # shortcut is disabled), so any candidate round projects both
    # sides of >=1 pairing.
    assert plan_on.projections >= 2


def test_turn_policy_serves_and_drains_projections_under_all():
    sim = fresh_scenario_sim(seed=3, max_turns=6, mini=True)
    torch.manual_seed(0)
    base = TransformerPolicy(device=torch.device("cpu"), d_model=32,
                             num_layers=1, num_heads=4, d_ff=64)
    policy = TurnCommitPolicy(
        base, MCTSConfig(),
        turn_config=_cfg(project="all", project_halfturns=1,
                         project_max_actions=2))
    side = sim.gs.global_info.current_side
    served = 0
    while (not sim.done and sim.gs.global_info.current_side == side
           and served < 10):
        pre = copy.deepcopy(sim.gs)
        action = policy.select_action(pre, game_label="g", sim=sim)
        sim.step(action)
        served += 1
        if action.get("type") == "end_turn":
            break
    assert served >= 1
    pend = policy._pending.get("g")
    assert pend, "full turn under project=all still records targets"
    # The TCS counters ride the distill drain (the leg-3 telemetry
    # gap fix): one call must carry BOTH distill_* means and tcs_*
    # rates, and reset the accumulators.
    merged = policy.drain_distill_stats()
    assert merged is not None
    assert merged["tcs_plans"] == 1.0
    assert merged["tcs_projections_per_plan"] > 0
    assert "distill_sharpen_top" in merged, \
        "full-turn targets also populate the distill accumulator"
    assert policy._tcs_projections == 0, "drain resets the counter"
    assert policy.drain_tcs_stats()["tcs_plans"] == 0
