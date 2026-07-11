"""Moves-left head (Lc0-style, 2026-07-04): the model predicts the
fraction of the turn budget still to be played — a dense TEMPO signal
that the sparse win/loss z cannot provide (diagnosed root cause of
action-spam: v is flat across shuffle moves, so nothing in training
distinguishes progress from dawdling).

Pins: the config-gated head (present + (0,1)-bounded when on, absent
from the default arch when off); finalize_game attaching
monotonically shrinking moves_left_targets; the trainer's MSE term
firing only when the head + targets are present. The search-side
utility consumer is deliberately NOT wired yet (default-off pending
head calibration) — nothing here asserts search behavior.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

from transformer_policy import TransformerPolicy               # noqa: E402
from tools.mcts import MCTSConfig                              # noqa: E402
from tools.mcts_policy import MCTSPolicy                       # noqa: E402
from tools.draw_tiebreak import DrawTiebreakConfig             # noqa: E402
from sim_test_helpers import fresh_scenario_sim                # noqa: E402


def _pol(ml):
    return TransformerPolicy(device=torch.device("cpu"), d_model=48,
                             num_layers=2, num_heads=4, d_ff=96,
                             moves_left=ml)


def _cfg():
    return MCTSConfig(n_simulations=10, gumbel_root=True, gumbel_m=4,
                      chance_nodes=True, exact_outcome_enumeration=True,
                      draw_tiebreak=DrawTiebreakConfig(cap=0.3),
                      batch_size=1, add_root_noise=False)


def _play_and_finalize(mp, seed=21):
    # Real production rollout (no hand-mirrored loop; see
    # test_aux_targets._play_and_finalize for the history).
    import numpy as np
    from tools.sim_self_play import play_one_game, _recruit_cost_lookup
    mp._rng = np.random.default_rng(seed)
    sim = fresh_scenario_sim(seed=seed, max_turns=8, mini=True)
    play_one_game(sim, mp, lambda delta: 0.0,
                  game_label="g0", cost_lookup=_recruit_cost_lookup())


# ---- model head -----------------------------------------------------

def test_head_present_and_bounded_when_enabled():
    pol = _pol(True)
    sim = fresh_scenario_sim(seed=21, max_turns=12, mini=True)
    with torch.no_grad():
        out = pol._inference_model(pol._inference_encoder.encode(sim.gs))
    assert out.moves_left is not None
    assert tuple(out.moves_left.shape) == (1, 1)
    assert 0.0 < float(out.moves_left.item()) < 1.0


def test_head_none_and_arch_unchanged_when_disabled():
    pol = _pol(False)
    sim = fresh_scenario_sim(seed=21, max_turns=12, mini=True)
    with torch.no_grad():
        out = pol._inference_model(pol._inference_encoder.encode(sim.gs))
    assert out.moves_left is None
    keys = pol._model.state_dict().keys()
    assert not any("moves_left_head" in k for k in keys), (
        "default arch must stay byte-compatible with old checkpoints")
    assert any("moves_left_head" in k
               for k in _pol(True)._model.state_dict())


# ---- targets + trainer ---------------------------------------------

def test_finalize_sets_shrinking_moves_left_targets():
    mp = MCTSPolicy(_pol(True), _cfg())
    _play_and_finalize(mp)
    assert mp._queue, "expected recorded experiences"
    tgts = [e.moves_left_target for e in mp._queue]
    assert all(t is not None and 0.0 <= t <= 1.0 for t in tgts)
    # Within one game, a later state has no MORE turns remaining.
    turns = [e.game_state.global_info.turn_number for e in mp._queue]
    paired = sorted(zip(turns, tgts))
    for (ta, fa), (tb, fb) in zip(paired, paired[1:]):
        if tb > ta:
            assert fb <= fa + 1e-9, (
                f"turn {tb} has MORE budget left ({fb}) than earlier "
                f"turn {ta} ({fa})")


def test_search_utility_prefers_short_wins_long_losses():
    """The Lc0-style utility must flip its preference with Q's sign:
    winning -> pick the edge with LOWER backed-up moves-left; losing
    -> HIGHER. c_puct=0 isolates the Q+utility part of the score."""
    from tools.mcts import MCTSNode, MCTSEdge, _puct_select
    from action_sampler import LegalActionPrior

    def _node_with_two_edges(q, m_fast, m_slow):
        sim = fresh_scenario_sim(seed=21, max_turns=8, mini=True)
        node = MCTSNode(sim)
        edges = []
        for m in (m_fast, m_slow):
            e = MCTSEdge(LegalActionPrior(
                action={"type": "end_turn"}, prior=0.5, actor_idx=0,
                type_idx=0, target_idx=0, weapon_idx=0))
            e.n_visits = 10
            e.w_value = q * 10
            e.m_sum = m * 10
            edges.append(e)
        node.edges = edges
        node._total_visits = 20
        return node

    # Winning (q=+0.5): prefer the SHORT line (m=0.1 over m=0.9).
    node = _node_with_two_edges(+0.5, 0.1, 0.9)
    assert _puct_select(node, c_puct=0.0,
                        moves_left_utility=0.2) is node.edges[0]
    # Losing (q=-0.5): prefer the LONG line.
    node = _node_with_two_edges(-0.5, 0.1, 0.9)
    assert _puct_select(node, c_puct=0.0,
                        moves_left_utility=0.2) is node.edges[1]
    # Utility off: tie on Q -> first edge wins (legacy behavior).
    node = _node_with_two_edges(+0.5, 0.1, 0.9)
    assert _puct_select(node, c_puct=0.0,
                        moves_left_utility=0.0) is node.edges[0]


def test_search_backs_up_moves_left_through_real_search():
    """A real search with the head + utility on must leave backed-up
    M mass on visited root edges (sigmoid outputs are strictly
    positive, so any network-evaluated backup deposits m_sum > 0)."""
    import numpy as np
    from tools.mcts import MCTSConfig as MC, mcts_search
    pol = _pol(True)
    sim = fresh_scenario_sim(seed=21, max_turns=8, mini=True)
    cfg = MC(n_simulations=12, gumbel_root=False, add_root_noise=False,
             draw_tiebreak=DrawTiebreakConfig(cap=0.3), batch_size=1,
             moves_left_utility=0.2)
    root = mcts_search(sim, pol._inference_model, pol._inference_encoder,
                       cfg, rng=np.random.default_rng(0))
    visited = [e for e in root.edges if e.n_visits > 0]
    assert visited, "search must visit at least one root edge"
    assert any(e.m_sum > 0 for e in visited), (
        "no moves-left mass backed up despite the head being present")


def test_train_step_moves_left_loss_fires_only_when_on():
    mp_on = MCTSPolicy(_pol(True), _cfg())
    _play_and_finalize(mp_on, seed=21)
    stats_on = mp_on.train_step()
    assert stats_on.moves_left_loss > 0.0, (
        "moves-left loss should be active with head+targets")
    assert stats_on.total_loss == stats_on.total_loss  # finite

    mp_off = MCTSPolicy(_pol(False), _cfg())
    _play_and_finalize(mp_off, seed=22)
    stats_off = mp_off.train_step()
    assert stats_off.moves_left_loss == 0.0, "no head -> no loss"


# ---------------------------------------------------------------------
# Aux-head value bonus (2026-07-11): leaf v' = clamp(v + b*aux, -1, 1)
# ---------------------------------------------------------------------

def test_aux_adjusted_math():
    import torch
    from types import SimpleNamespace
    from tools.mcts import _aux_adjusted

    out = SimpleNamespace(aux_score=torch.tensor([[0.5]]))
    assert abs(_aux_adjusted(0.1, out, 0.3) - 0.25) < 1e-9
    # clamped at +1
    assert _aux_adjusted(0.95, out, 0.3) == 1.0
    # negative margin pulls down
    out_neg = SimpleNamespace(aux_score=torch.tensor([[-1.0]]))
    assert abs(_aux_adjusted(0.0, out_neg, 0.3) - (-0.3)) < 1e-9


def test_aux_adjusted_off_by_default():
    import torch
    from types import SimpleNamespace
    from tools.mcts import MCTSConfig, _aux_adjusted

    assert MCTSConfig().aux_value_bonus == 0.0
    out = SimpleNamespace(aux_score=torch.tensor([[0.9]]))
    assert _aux_adjusted(0.2, out, 0.0) == 0.2, "knob 0 = no-op"
    out_no_head = SimpleNamespace(aux_score=None)
    assert _aux_adjusted(0.2, out_no_head, 0.3) == 0.2, \
        "no aux head = no-op"
