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


def test_fast_moves_record_no_target():
    """With prob=0 (every move fast), NO pending training states are
    recorded across a whole game, but the game still completes."""
    pol = _policy()
    mp = MCTSPolicy(pol, _cfg(playout_cap_randomization=True,
                              playout_cap_prob=0.0,
                              playout_cap_fast_sims=4))
    sim = fresh_scenario_sim(seed=21, max_turns=8, mini=True)
    gl = "g0"
    steps = 0
    while not sim.done and steps < 40:
        action = mp.select_action(sim.gs, game_label=gl, sim=sim)
        sim.step(action)
        steps += 1
    # prob=0 => every move was fast => nothing recorded.
    assert sum(len(v) for v in mp._pending.values()) == 0


def test_full_prob_records_every_move():
    """prob=1.0 => every move is full => every decision records a
    target (parity with playout-cap OFF), confirming the gate only
    suppresses recording on FAST moves."""
    pol = _policy()
    mp = MCTSPolicy(pol, _cfg(playout_cap_randomization=True,
                              playout_cap_prob=1.0, playout_cap_fast_sims=4))
    mp._rng = np.random.default_rng(0)
    sim = fresh_scenario_sim(seed=21, max_turns=8, mini=True)
    gl = "g1"
    decisions = 0
    while not sim.done and decisions < 12:
        action = mp.select_action(sim.gs, game_label=gl, sim=sim)
        sim.step(action)
        decisions += 1
    assert decisions > 0
    assert sum(len(v) for v in mp._pending.values()) == decisions, \
        "prob=1.0 must record every decision"


def test_off_by_default_records_every_move():
    pol = _policy()
    mp = MCTSPolicy(pol, _cfg())   # playout_cap_randomization defaults False
    sim = fresh_scenario_sim(seed=23, max_turns=8, mini=True)
    mp.select_action(sim.gs, game_label="g3", sim=sim)
    assert sum(len(v) for v in mp._pending.values()) == 1
