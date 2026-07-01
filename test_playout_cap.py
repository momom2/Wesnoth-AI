"""Playout-cap randomization (KataGo, plan §3.5).

Pins the contract: under playout-cap, only the random "full" fraction of
moves run the full sim budget AND record a policy training target; the
rest run a cheap budget and record nothing. The value target is
unaffected (it attaches to recorded states at finalize). Off by default
=> behaviour is byte-identical to before (every move recorded).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

from transformer_policy import TransformerPolicy            # noqa: E402
from tools.mcts import MCTSConfig, mcts_search               # noqa: E402
from tools.mcts_policy import MCTSPolicy                     # noqa: E402
from tools.draw_tiebreak import DrawTiebreakConfig           # noqa: E402
from sim_test_helpers import fresh_scenario_sim              # noqa: E402


def _policy():
    return TransformerPolicy(device=torch.device("cpu"),
                             d_model=48, num_layers=2, num_heads=4, d_ff=96)


def _cfg(**kw):
    base = dict(n_simulations=16, gumbel_root=True, gumbel_m=4,
                chance_nodes=True, exact_outcome_enumeration=True,
                draw_tiebreak=DrawTiebreakConfig(cap=0.3), batch_size=1,
                add_root_noise=False)
    base.update(kw)
    return MCTSConfig(**base)


def _count_forwards(model):
    n = [0]
    h = model.register_forward_pre_hook(
        lambda m, i: n.__setitem__(0, n[0] + 1))
    return n, h


def test_n_sims_override_runs_fewer_forwards():
    pol = _policy()
    sim = fresh_scenario_sim(seed=21, max_turns=12, mini=True)
    enc, mdl = pol._inference_encoder, pol._inference_model

    n_full, h = _count_forwards(mdl)
    mcts_search(sim.fork(), mdl, enc, _cfg(), rng=np.random.default_rng(0))
    h.remove()

    n_fast, h = _count_forwards(mdl)
    mcts_search(sim.fork(), mdl, enc, _cfg(), rng=np.random.default_rng(0),
                n_sims_override=4)
    h.remove()
    assert n_fast[0] < n_full[0], (
        f"override should cut forwards: fast={n_fast[0]} full={n_full[0]}")


def _recorded_count(mp, seed=21):
    """Drive a REAL self-play game through the production rollout
    (tools.sim_self_play.play_one_game -- which deepcopies the per-
    decision snapshot and calls finalize_game) and return how many
    training targets the policy recorded. Driving the actual code path
    (not a hand-rolled select/step/finalize loop) is the point: it can't
    get the snapshot contract subtly wrong. MCTS ignores per-step
    rewards, so a zero reward_fn suffices."""
    from tools.sim_self_play import play_one_game, _recruit_cost_lookup
    mp._rng = np.random.default_rng(seed)
    sim = fresh_scenario_sim(seed=seed, max_turns=8, mini=True)
    play_one_game(sim, mp, lambda delta: 0.0, game_label="g",
                  cost_lookup=_recruit_cost_lookup())
    return len(mp._queue)   # finalize_game drained _pending -> _queue


def test_playout_cap_prob_zero_records_nothing():
    """prob=0 => every move is FAST => no policy targets recorded
    (the game still plays out)."""
    mp = MCTSPolicy(_policy(), _cfg(playout_cap_randomization=True,
                                    playout_cap_prob=0.0,
                                    playout_cap_fast_sims=4))
    assert _recorded_count(mp) == 0


def test_playout_cap_full_records_targets():
    """prob=1 => every move is FULL => the rollout records targets
    (the gate suppresses recording only on FAST moves)."""
    mp = MCTSPolicy(_policy(), _cfg(playout_cap_randomization=True,
                                    playout_cap_prob=1.0,
                                    playout_cap_fast_sims=4))
    assert _recorded_count(mp) > 0


def test_off_by_default_records_targets():
    """playout_cap off (default) => normal recording on every move."""
    mp = MCTSPolicy(_policy(), _cfg())
    assert _recorded_count(mp) > 0


def test_drop_last_pending_pops_tail_and_rolls_back_decision_step():
    """The fog-bounce retry loop calls drop_last_pending to undo the
    rejected decision. It must (a) pop exactly the pending target the
    last select_action recorded and (b) roll back the decision_step
    increment, so the bounced pick is neither trained on nor counted
    toward the combat-oracle anneal (which would otherwise over-advance
    and inject a correlated duplicate target)."""
    import copy
    mp = MCTSPolicy(_policy(), _cfg())
    mp._rng = np.random.default_rng(21)
    sim = fresh_scenario_sim(seed=21, max_turns=8, mini=True)

    ds0 = mp._base._decision_step
    pre = copy.deepcopy(sim.gs)
    mp.select_action(pre, game_label="g", sim=sim)
    # A full move recorded one pending target and advanced the counter.
    assert len(mp._pending.get("g", [])) == 1
    assert mp._base._decision_step == ds0 + 1

    handled = mp.drop_last_pending("g")
    assert handled is True
    assert len(mp._pending.get("g", [])) == 0
    # decision_step rolled back so the retry re-consumes it (no double count).
    assert mp._base._decision_step == ds0
