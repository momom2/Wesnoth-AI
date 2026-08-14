"""TurnCommitPolicy integration tests -- production code paths only.

Covers the contracts docs/tcs_spec.md par.3-6 and the 2026-08-14
default-on integration:
  * one plan per side-turn (no per-decision re-planning when the
    trajectory follows the planned branch);
  * divergence -> warm re-plan (the plan-once-replan-at-chance rule,
    driven by corrupting the expected pre-state key);
  * bounce contract: drop_last_pending pops the pending target, rolls
    back decision_step, and discards the plan;
  * full game -> finalize_game -> train_step through the INHERITED
    MCTS pipeline (experiences carry non-empty 5-tuple targets);
  * config symmetry across the three generation paths (the
    mis-damped-target failure class): sim_self_play's --turn-* flags
    == selfplay_worker's, the spool cmd_tail forwards every one, and
    the actor-pool plumbing carries turn_cfg.
"""
from __future__ import annotations

import copy
import inspect
import re
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_test_helpers import fresh_scenario_sim  # noqa: E402
from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402
from tools.mcts import MCTSConfig  # noqa: E402
from tools.turn_policy import TurnCommitPolicy  # noqa: E402
from tools.turn_search import TurnSearchConfig  # noqa: E402

REPO = Path(__file__).parent.parent


def _policy(turn_cfg: TurnSearchConfig) -> TurnCommitPolicy:
    torch.manual_seed(0)
    base = TransformerPolicy(device=torch.device("cpu"), d_model=32,
                             num_layers=1, num_heads=4, d_ff=64)
    return TurnCommitPolicy(base, MCTSConfig(), turn_config=turn_cfg)


def _cfg(**kw) -> TurnSearchConfig:
    defaults = dict(n_alt=2, rounds=1, fast_rounds=0, reval_salts=2,
                    max_spine=6, turn_full_prob=1.0)
    defaults.update(kw)
    return TurnSearchConfig(**defaults)


def _step_once(policy, sim, label="g"):
    pre = copy.deepcopy(sim.gs)
    action = policy.select_action(pre, game_label=label, sim=sim)
    return action


def test_one_plan_serves_whole_turn():
    sim = fresh_scenario_sim(seed=3, max_turns=6, mini=True)
    policy = _policy(_cfg())
    side = sim.gs.global_info.current_side
    served = 0
    while (not sim.done and sim.gs.global_info.current_side == side
           and served < 10):
        action = _step_once(policy, sim)
        sim.step(action)
        served += 1
        if action.get("type") == "end_turn":
            break
    assert served >= 1
    # Turn-1 mini-map play is deterministic (recruit trait rolls
    # don't alter this turn's legality), so the whole turn must come
    # from ONE planning pass -- no per-decision re-planning.
    assert policy._tcs_plans == 1
    assert policy._tcs_replans == 0
    # Full turn (turn_full_prob=1.0): every served decision recorded
    # a pending 5-tuple target.
    pend = policy._pending.get("g")
    assert pend is not None and len(pend) == served
    for p in pend:
        assert p.visit_counts, "full-turn coordinates carry targets"
        for t in p.visit_counts:
            assert len(t) == 5


def test_divergence_triggers_warm_replan():
    sim = fresh_scenario_sim(seed=3, max_turns=6, mini=True)
    policy = _policy(_cfg())
    action = _step_once(policy, sim)
    sim.step(action)
    if sim.done or action.get("type") == "end_turn":
        pytest.skip("one-action turn; no mid-turn coordinate to test")
    # Corrupt the plan's expected pre-state key for the next
    # coordinate: the live state no longer matches the planned
    # branch, which is exactly what a diverged combat outcome looks
    # like -- select_action must warm re-plan, not serve stale.
    plan = policy._plans["g"]
    plan.pre_keys[plan.cursor] = 0xDEAD
    action2 = _step_once(policy, sim)
    assert policy._tcs_replans == 1
    assert policy._tcs_plans == 2
    sim.step(action2)   # the re-planned action must be legal live


def test_bounce_contract_drops_plan_and_pending():
    sim = fresh_scenario_sim(seed=3, max_turns=6, mini=True)
    policy = _policy(_cfg())
    _ = _step_once(policy, sim)   # served; NOT stepped (bounce path)
    ds_after = policy._base._decision_step
    n_pend = len(policy._pending.get("g", []))
    assert n_pend == 1
    handled = policy.drop_last_pending("g")
    assert handled is True
    assert len(policy._pending.get("g", [])) == 0
    assert policy._base._decision_step == ds_after - 1
    assert "g" not in policy._plans
    # The retry re-plans (fresh plan, legal action).
    action = _step_once(policy, sim)
    assert policy._tcs_plans == 2
    sim.step(action)


@pytest.mark.slow
def test_full_game_trains_through_inherited_pipeline():
    sim = fresh_scenario_sim(seed=5, max_turns=4, mini=True)
    policy = _policy(_cfg())
    guard = 0
    while not sim.done and guard < 400:
        action = _step_once(policy, sim)
        sim.step(action)
        guard += 1
    assert sim.done, "mini game must finish under the turn cap"
    policy.finalize_game("g", sim.winner, final_gs=sim.gs)
    assert len(policy._queue) > 0
    for exp in policy._queue:
        assert exp.visit_counts, \
            "TCS emits only policy-target experiences (no value-only " \
            "boundary rows -- docs/tcs_spec.md par.4 integration note)"
        assert exp.z in (-1.0, 0.0, 1.0)
    stats = policy.train_step()
    assert stats is not None
    tcs = policy.drain_tcs_stats()
    assert tcs["tcs_plans"] > 0


# ---------------------------------------------------------------------
# Config symmetry across the three generation paths
# ---------------------------------------------------------------------

_FLAG_RE = re.compile(r'"(--turn-[a-z-]+)"')


def _flags_in(path: Path, span: str = "") -> set:
    text = path.read_text(encoding="utf-8")
    return {m for m in _FLAG_RE.findall(text)
            if not span or m in text}


def test_turn_flag_symmetry_across_paths():
    """The mis-damped-target failure class: a knob the learner sets
    but a generation path silently ignores. All --turn-* flags must
    exist in BOTH parsers, and the spool cmd_tail must forward each
    (--no-turn-search counts as forwarding --turn-search)."""
    ssp = (REPO / "tools" / "sim_self_play.py").read_text(
        encoding="utf-8")
    wrk = (REPO / "tools" / "selfplay_worker.py").read_text(
        encoding="utf-8")
    ssp_flags = set(_FLAG_RE.findall(ssp))
    wrk_flags = set(_FLAG_RE.findall(wrk))
    assert ssp_flags, "sim_self_play defines --turn-* flags"
    assert ssp_flags - {"--no-turn-search"} <= wrk_flags | {
        "--no-turn-search"}
    assert wrk_flags <= ssp_flags, \
        f"worker-only turn flags: {wrk_flags - ssp_flags}"
    # Every value flag appears in the SpoolWorkers cmd_tail region.
    tail = ssp[ssp.index("_cmd_tail = ["):]
    tail = tail[:tail.index("self._seed0")]
    for flag in sorted(wrk_flags - {"--turn-search",
                                    "--no-turn-search"}):
        assert flag in tail, f"{flag} not forwarded in _cmd_tail"
    assert ("--turn-search" in tail or "--no-turn-search" in tail)


def test_actor_pool_carries_turn_cfg():
    from tools.actor_pool import ActorPool, _actor_loop
    assert "turn_cfg" in inspect.signature(ActorPool.__init__).parameters
    assert "turn_cfg" in inspect.signature(_actor_loop).parameters


def test_config_from_args_roundtrip():
    from types import SimpleNamespace
    from tools.turn_search import config_from_args
    args = SimpleNamespace(turn_search=True, turn_alt=7, turn_rounds=2,
                           turn_fast_rounds=0, turn_reval_salts=5,
                           turn_min_delta=0.02, turn_max_spine=9,
                           turn_full_prob=0.5, turn_reply="reval",
                           turn_reply_max_actions=3)
    cfg = config_from_args(args)
    assert (cfg.n_alt, cfg.rounds, cfg.fast_rounds) == (7, 2, 0)
    assert (cfg.reval_salts, cfg.min_delta) == (5, 0.02)
    assert (cfg.max_spine, cfg.turn_full_prob) == (9, 0.5)
    assert (cfg.reply, cfg.reply_max_actions) == ("reval", 3)
    assert config_from_args(
        SimpleNamespace(turn_search=False)) is None
