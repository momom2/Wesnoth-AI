#!/usr/bin/env python3
"""The trainer must re-encode stored states in the SAME hex space
the search recorded target indices in (2026-08-04: three bare
encode_raw() calls in trainer.py dropped relevant_set, so under
--relevant-set-hexes every policy target landed on wrong/masked
full-space hexes -- policy loss ~6e8, and eval_value_metrics was
measuring garbage; the T2 fine-tune leg was unrunnable)."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

import torch

from wesnoth_ai.transformer_policy import TransformerPolicy
from tools.mcts import MCTSConfig
from tools.mcts_policy import MCTSPolicy
from tools.draw_tiebreak import DrawTiebreakConfig
from tools.scenario_pool import ScenarioSetup, build_scenario_gamestate
from tools.wesnoth_sim import WesnothSim


def test_step_mcts_sane_under_relevant_set_flag():
    pol = TransformerPolicy(device=torch.device("cpu"), d_model=48,
                            num_layers=2, num_heads=4, d_ff=96,
                            relevant_set_hexes=True)
    mp = MCTSPolicy(pol, MCTSConfig(
        n_simulations=8, gumbel_root=True, gumbel_m=4, batch_size=1,
        draw_tiebreak=DrawTiebreakConfig(cap=0.3),
        add_root_noise=False), holdout_size=0)
    setup = ScenarioSetup(
        scenario_id="2p_mini", faction1="Rebels",
        leader1="Elvish Captain", faction2="Knalgan Alliance",
        leader2="Dwarvish Steelclad")
    sim = WesnothSim(build_scenario_gamestate(setup),
                     scenario_id="2p_mini", max_turns=5)
    label = "rs"
    while not sim.done:
        pre = copy.deepcopy(sim.gs)
        act = mp.select_action(pre, game_label=label, sim=sim)
        sim.step(act)
    mp.finalize_game(label, winner=sim.winner or 0, final_gs=sim.gs)
    with mp._lock:
        batch = list(mp._queue)
    assert batch, "game produced no experiences"
    st = pol._trainer.step_mcts(batch)
    # The bug produced ~6e8 (targets on -1e9-masked logits); any
    # sane CE is orders of magnitude below this bound.
    assert float(st.policy_loss) < 100.0, float(st.policy_loss)
    # eval path shares the encode sites; must be finite and sane too.
    m = pol._trainer.eval_value_metrics(batch)
    assert m["ce"] == m["ce"] and m["ce"] < 100.0
