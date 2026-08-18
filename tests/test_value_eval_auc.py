"""value_auc in eval_value_metrics (A1/A3 gate metric, 2026-08-17).
A wrong gate metric silently passes a broken judge into a leg --
the exact leg-3 failure. Pins the tie and missing-class semantics
on the production eval path with a real (tiny) model.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_test_helpers import fresh_scenario_sim  # noqa: E402
from wesnoth_ai.trainer import MCTSExperience  # noqa: E402
from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402


def _exps(zs):
    sim = fresh_scenario_sim()
    return [MCTSExperience(game_state=sim.gs, visit_counts=[], z=z)
            for z in zs]


def _trainer():
    torch.manual_seed(0)
    return TransformerPolicy(device=torch.device("cpu"), d_model=32,
                             num_layers=1, num_heads=4,
                             d_ff=64)._trainer


def test_identical_states_tie_to_exactly_half():
    """Same state for every experience -> identical E[V] -> all
    comparisons are ties -> AUC must be exactly 0.5 (the tie rule),
    not 0 or 1."""
    m = _trainer().eval_value_metrics(_exps([1.0, 1.0, -1.0, -1.0]))
    assert m["value_auc"] == 0.5
    assert m["n_decisive"] == 4


def test_single_class_yields_nan_not_a_verdict():
    """All-win (or all-loss) probes carry no ranking information;
    the gate must see NaN (-> refuse), never a fake number."""
    m = _trainer().eval_value_metrics(_exps([1.0, 1.0, 1.0]))
    assert math.isnan(m["value_auc"])
    m2 = _trainer().eval_value_metrics([])
    assert math.isnan(m2["value_auc"])
