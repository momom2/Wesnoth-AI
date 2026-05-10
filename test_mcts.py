#!/usr/bin/env python3
"""Regression tests for tools/mcts.py.

Focus areas:

1. **PUCT selection** — formula, tie-break, exploration vs exploitation
   trade-off across visit counts.

2. **Sign-flip in `_backup` — turns vs actions**. This is the
   Wesnoth-specific bit that diverges from textbook 2-player MCTS:
   a *turn* in Wesnoth contains many *actions* by the same side
   (move, attack, recruit, ...) before `end_turn` flips control.
   Textbook MCTS ("flip every backup step") is WRONG here. The
   correct rule is "flip iff this node's side differs from the leaf's
   side", which our `_backup` implements via
   `parent.side == leaf_side` per edge. These tests construct paths
   with same-side multi-action stretches and mixed paths spanning
   multiple turn transitions to lock that behavior down.

3. **Terminal-value semantics** — win/loss/draw from a given side's
   perspective; terminal-leaf shortcut avoids a model forward.

4. **Dirichlet noise** — mass conservation, eps=0/eps=1 endpoints.

5. **Visit-count extraction** — output format, zero-skip behavior.

These are pure-Python unit tests with hand-rolled MCTSNode/MCTSEdge
fixtures (no live model, no real sim). End-to-end `mcts_search`
integration coverage is not in this file — would require a small
synthetic GameState + model and is a separate test layer.

Dependencies: tools.mcts, action_sampler.LegalActionPrior, classes
Dependents:   pytest only
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "tools"))

from action_sampler import LegalActionPrior  # noqa: E402
from tools.mcts import (  # noqa: E402
    MCTSEdge, MCTSNode, _backup, _puct_select, _terminal_value,
    _add_dirichlet_noise, extract_visit_counts,
)


# ---------------------------------------------------------------------
# Fixtures: hand-rolled nodes with a stub `sim`
# ---------------------------------------------------------------------

def _stub_sim(side: int, *, done: bool = False, winner: int = 0):
    """Minimal SimpleNamespace satisfying what MCTSNode reads from
    `sim`: `sim.gs.global_info.current_side` (for node.side) and
    `sim.done` (for node.is_terminal). For terminal-value tests we
    also set `sim.winner`."""
    gs = SimpleNamespace(global_info=SimpleNamespace(current_side=side))
    return SimpleNamespace(gs=gs, done=done, winner=winner)


def _make_node(side: int, *, done: bool = False, winner: int = 0) -> MCTSNode:
    return MCTSNode(_stub_sim(side, done=done, winner=winner))


def _make_edge(prior: float = 0.5,
               actor_idx: int = 0,
               target_idx=None) -> MCTSEdge:
    lap = LegalActionPrior(
        action={"type": "end_turn"},
        prior=prior,
        actor_idx=actor_idx,
        target_idx=target_idx,
        weapon_idx=None,
        type_idx=None,
    )
    return MCTSEdge(lap)


def _attach(parent: MCTSNode, child: MCTSNode, prior: float = 0.5) -> MCTSEdge:
    """Attach `child` to `parent` via a fresh edge; return the edge."""
    edge = _make_edge(prior=prior)
    edge.child = child
    parent.edges.append(edge)
    return edge


# ---------------------------------------------------------------------
# _terminal_value
# ---------------------------------------------------------------------

def test_terminal_value_win():
    sim = _stub_sim(side=1, done=True, winner=1)
    assert _terminal_value(sim, 1) == 1.0
    assert _terminal_value(sim, 2) == -1.0


def test_terminal_value_draw():
    sim = _stub_sim(side=1, done=True, winner=0)
    assert _terminal_value(sim, 1) == 0.0
    assert _terminal_value(sim, 2) == 0.0


# ---------------------------------------------------------------------
# _backup — turns vs actions sign convention
# ---------------------------------------------------------------------

def test_backup_single_step_opposite_sides():
    """Textbook 2-player case: parent on A, leaf on B. Leaf value v
    is from B's perspective, so parent on A gets -v."""
    parent = _make_node(side=1)
    leaf = _make_node(side=2)
    edge = _attach(parent, leaf)
    _backup([(parent, edge)], v=0.7, leaf_side=2, virtual_loss=0.0)
    assert edge.n_visits == 1
    assert edge.w_value == pytest.approx(-0.7)


def test_backup_same_side_action_no_flip():
    """Wesnoth-specific: a same-side action (move/attack/recruit)
    keeps `current_side` unchanged. The backup MUST NOT flip the
    sign — both nodes are on side 1, so parent sees the leaf value
    as +v, not -v.

    The bug this guards against is the textbook 'flip every step'
    rule which would give -v here, treating the same-side move like
    a turn handover."""
    parent = _make_node(side=1)
    leaf = _make_node(side=1)
    edge = _attach(parent, leaf)
    _backup([(parent, edge)], v=0.6, leaf_side=1, virtual_loss=0.0)
    assert edge.w_value == pytest.approx(+0.6)


def test_backup_multi_action_same_turn():
    """A full Wesnoth turn: A → A → A → A → leaf A. Five actions,
    zero turn transitions. All four parents are on side A and the
    leaf is on side A, so every edge accumulates +v."""
    nodes = [_make_node(side=1) for _ in range(5)]
    path = []
    for parent, child in zip(nodes[:-1], nodes[1:]):
        path.append((parent, _attach(parent, child)))
    leaf = nodes[-1]
    _backup(path, v=0.4, leaf_side=leaf.side, virtual_loss=0.0)
    for parent, edge in path:
        assert parent.side == 1
        assert edge.w_value == pytest.approx(+0.4), (
            f"expected +v for same-side parent, got {edge.w_value}"
        )


def test_backup_path_spans_turn_transition():
    """Mixed path: A → A → B → B → leaf B. The first two parents
    are on the wrong side (A) and should get -v; the third parent
    is on the right side (B, same as leaf) and should get +v.
    Textbook 'flip every step' would give the wrong sign on the
    A→A edge."""
    n1 = _make_node(side=1)  # parent of edge 1
    n2 = _make_node(side=1)  # same-side action: still A
    n3 = _make_node(side=2)  # end_turn happened: now B
    leaf = _make_node(side=2)  # B's first action
    e1 = _attach(n1, n2)
    e2 = _attach(n2, n3)  # this edge is the end_turn
    e3 = _attach(n3, leaf)
    path = [(n1, e1), (n2, e2), (n3, e3)]
    _backup(path, v=0.8, leaf_side=2, virtual_loss=0.0)
    assert e1.w_value == pytest.approx(-0.8), "n1 (A) sees leaf (B) as bad"
    assert e2.w_value == pytest.approx(-0.8), "n2 (A) sees leaf (B) as bad"
    assert e3.w_value == pytest.approx(+0.8), "n3 (B) sees leaf (B) as good"


def test_backup_path_returns_to_original_side():
    """Two transitions: A → B → A → leaf A. With leaf_side=A, the
    A nodes get +v and the single B node gets -v. Tests that the
    rule is 'compare each parent's side to leaf_side', not 'alternate
    based on depth'."""
    nA1 = _make_node(side=1)
    nB  = _make_node(side=2)
    nA2 = _make_node(side=1)
    leaf = _make_node(side=1)
    e1 = _attach(nA1, nB)
    e2 = _attach(nB, nA2)
    e3 = _attach(nA2, leaf)
    path = [(nA1, e1), (nB, e2), (nA2, e3)]
    _backup(path, v=0.5, leaf_side=1, virtual_loss=0.0)
    assert e1.w_value == pytest.approx(+0.5)  # A == A
    assert e2.w_value == pytest.approx(-0.5)  # B != A
    assert e3.w_value == pytest.approx(+0.5)  # A == A


def test_backup_visit_count_increments_regardless_of_side():
    """Sign-flip applies to value only; visit count grows by 1 per
    edge in the path regardless of which side the parent is on.
    A backup that wrongly skipped certain edges would silently
    mis-train on the visit-count distillation target."""
    nodes = [_make_node(side=(i % 2) + 1) for i in range(4)]
    path = [(p, _attach(p, c)) for p, c in zip(nodes[:-1], nodes[1:])]
    _backup(path, v=0.3, leaf_side=nodes[-1].side, virtual_loss=0.0)
    for _, edge in path:
        assert edge.n_visits == 1


def test_backup_virtual_loss_undone():
    """Virtual loss applied during selection must be undone in
    backup, then replaced with the real visit + value."""
    parent = _make_node(side=1)
    leaf = _make_node(side=1)
    edge = _attach(parent, leaf)
    # Simulate selection having already applied virtual loss.
    edge.n_visits = 3.0
    edge.w_value = -3.0
    parent._total_visits = 3.0
    _backup([(parent, edge)], v=0.5, leaf_side=1, virtual_loss=3.0)
    assert edge.n_visits == pytest.approx(1.0), (
        "virtual loss of 3 reverted, then +1 real visit"
    )
    assert edge.w_value == pytest.approx(0.5), (
        "virtual loss of -3 reverted to 0, then +0.5 real value"
    )
    assert parent._total_visits == pytest.approx(1.0)


# ---------------------------------------------------------------------
# _puct_select
# ---------------------------------------------------------------------

def test_puct_select_picks_highest_prior_when_no_visits():
    """With zero visits everywhere, q_value is 0 for all edges and
    PUCT degenerates to picking the largest prior."""
    node = _make_node(side=1)
    node.expanded = True
    e_low = _attach(node, _make_node(side=2), prior=0.1)
    e_mid = _attach(node, _make_node(side=2), prior=0.4)
    e_high = _attach(node, _make_node(side=2), prior=0.5)
    chosen = _puct_select(node, c_puct=1.5)
    assert chosen is e_high


def test_puct_select_prefers_high_q_when_explored():
    """Once visit counts are non-trivial, the edge with the best Q
    wins even if its prior is lower (and other edges have similar
    visit counts so the U bonus is comparable)."""
    node = _make_node(side=1)
    node.expanded = True
    e1 = _attach(node, _make_node(side=2), prior=0.4)
    e2 = _attach(node, _make_node(side=2), prior=0.4)
    # e1: visited 5 times, all losses (Q=-1)
    e1.n_visits = 5
    e1.w_value = -5.0
    # e2: visited 5 times, mostly wins (Q=+0.6)
    e2.n_visits = 5
    e2.w_value = +3.0
    node._total_visits = 10
    chosen = _puct_select(node, c_puct=1.5)
    assert chosen is e2


def test_puct_select_explores_unvisited_edge():
    """An unvisited edge with non-trivial prior beats a heavily-
    visited edge whose Q has dropped below 0. PUCT's U term
    explodes when n_visits == 0 and parent has many total visits."""
    node = _make_node(side=1)
    node.expanded = True
    e_visited = _attach(node, _make_node(side=2), prior=0.5)
    e_unvisited = _attach(node, _make_node(side=2), prior=0.5)
    e_visited.n_visits = 50
    e_visited.w_value = -45.0  # Q ≈ -0.9
    node._total_visits = 50
    chosen = _puct_select(node, c_puct=1.5)
    assert chosen is e_unvisited


# ---------------------------------------------------------------------
# Dirichlet noise
# ---------------------------------------------------------------------

def test_dirichlet_eps_zero_is_noop():
    """eps=0 means original priors untouched."""
    import numpy as np
    node = _make_node(side=1)
    edges = [_attach(node, _make_node(side=2), prior=p)
             for p in (0.1, 0.6, 0.3)]
    rng = np.random.default_rng(42)
    _add_dirichlet_noise(node, alpha=0.3, eps=0.0, rng=rng)
    assert [e.prior for e in edges] == [0.1, 0.6, 0.3]


def test_dirichlet_preserves_mass():
    """Convex combination (1-eps)*prior + eps*noise — total mass
    stays ≈1 since both components sum to 1."""
    import numpy as np
    node = _make_node(side=1)
    edges = [_attach(node, _make_node(side=2), prior=p)
             for p in (0.2, 0.5, 0.3)]
    rng = np.random.default_rng(7)
    _add_dirichlet_noise(node, alpha=0.3, eps=0.25, rng=rng)
    total = sum(e.prior for e in edges)
    assert total == pytest.approx(1.0, abs=1e-9)
    # Priors stay valid probabilities.
    for e in edges:
        assert 0.0 <= e.prior <= 1.0


# ---------------------------------------------------------------------
# extract_visit_counts
# ---------------------------------------------------------------------

def test_extract_visit_counts_skips_zero_visits():
    """`extract_visit_counts` returns one tuple per visited edge and
    omits unvisited ones (saves trainer-side wasted CE-loss work)."""
    node = _make_node(side=1)
    e1 = _attach(node, _make_node(side=2), prior=0.4)
    e1.actor_idx = 3
    e1.target_idx = 7
    e1.n_visits = 12
    e2 = _attach(node, _make_node(side=2), prior=0.6)
    e2.actor_idx = 5
    e2.n_visits = 0   # zero-visit edge: must be dropped
    out = extract_visit_counts(node)
    assert len(out) == 1, "zero-visit edges must be skipped"
    # First element should encode the visited edge's actor_idx.
    assert out[0][0] == 3


def test_extract_visit_counts_preserves_visit_counts():
    """All non-zero visit counts pass through verbatim — no
    normalization at extraction time. The trainer applies the
    softmax-with-temperature when building the target distribution."""
    node = _make_node(side=1)
    e1 = _attach(node, _make_node(side=2), prior=0.5)
    e1.actor_idx, e1.n_visits = 1, 7
    e2 = _attach(node, _make_node(side=2), prior=0.5)
    e2.actor_idx, e2.n_visits = 2, 13
    out = extract_visit_counts(node)
    # Schema is (actor_idx, target_idx, weapon_idx, count, type_idx);
    # count lives at index 3.
    visit_total = sum(t[3] for t in out)
    assert visit_total == 7 + 13


# ---------------------------------------------------------------------
# Terminal-leaf shortcut: terminal nodes don't trigger model forward
# ---------------------------------------------------------------------

def test_terminal_node_marks_terminal_via_sim_done():
    """MCTSNode reads `sim.done` at construction and stores it as
    `is_terminal`. The selection loop uses this to short-circuit
    `_expand` (no model forward needed)."""
    terminal_node = _make_node(side=1, done=True, winner=1)
    assert terminal_node.is_terminal is True

    live_node = _make_node(side=1, done=False)
    assert live_node.is_terminal is False


# ---------------------------------------------------------------------
# Transposition table — node sharing across paths
# ---------------------------------------------------------------------

def test_transposition_shares_node_across_paths():
    """Two parallel selections that converge on a child with the
    same `state_key` must end up pointing at the SAME MCTSNode
    object. Without the TT they'd build duplicate subtrees;
    visit counts on the shared node's outgoing edges then reflect
    every path that reached the state.

    We exercise `_select_one` directly with a hand-rolled root that
    has two edges -- one whose `step()` lands in the same state as
    the other (via a stub sim that ignores the action) -- and check
    that the second descent's edge.child is identical to the first
    descent's. The TT is consulted by `_select_one` when
    `transpositions=` is provided.
    """
    from tools.mcts import _select_one
    from classes import (
        GameState, GlobalInfo, Map, SideInfo, state_key,
    )

    # Build a minimal canonical GameState. Two distinct GameStates
    # with identical content hash to the same state_key, so a stub
    # `sim.fork().step(action)` that always lands at the same gs
    # will collide in the TT.
    def _mk_gs():
        return GameState(
            game_id="t",
            map=Map(size_x=4, size_y=4, mask=set(), fog=set(),
                    hexes=set(), units=set()),
            global_info=GlobalInfo(
                current_side=2, turn_number=3, time_of_day="dawn",
                village_gold=2, village_upkeep=1, base_income=2,
            ),
            sides=[SideInfo(player="x", recruits=[], current_gold=0,
                            base_income=2, nb_villages_controlled=0)],
        )

    # Stub sim: each fork returns a new sim whose `gs` is `_mk_gs()`.
    # `step(action)` does nothing (the stub always lands at the
    # canonical gs regardless of action). This way two different
    # edges from root will both produce children with the same
    # state_key.
    class _StubSim:
        def __init__(self):
            self.gs = _mk_gs()
            self.done = False
            self.winner = 0
            self.ended_by = ""

        def fork(self):
            return _StubSim()

        def step(self, action):  # noqa: ARG002
            # Stay on the same canonical gs -- the TT should
            # collapse all "child" nodes onto one.
            self.gs = _mk_gs()

    root_sim = _StubSim()
    root = MCTSNode(root_sim)
    # Root must be marked expanded so _select_one descends.
    root.expanded = True
    e_a = _attach(root, _make_node(side=1), prior=0.5)
    e_b = _attach(root, _make_node(side=1), prior=0.5)
    # Detach the pre-attached child stubs so _select_one creates
    # fresh ones via TT logic on first descent.
    e_a.child = None
    e_b.child = None
    # Bias selection: PUCT with zero visits picks max-prior; tie
    # broken by edge order. We set priors to differ so the FIRST
    # descent picks e_a, the SECOND picks e_b after virtual loss
    # tilts the choice. With virtual_loss=1.0 a single visit on
    # e_a's PUCT score drops it below e_b.
    e_a.prior = 0.51
    e_b.prior = 0.50

    target_key = state_key(_mk_gs())
    tt = {state_key(root_sim.gs): root}
    leaf_a, _ = _select_one(root, c_puct=1.5, virtual_loss=1.0,
                            transpositions=tt)
    leaf_b, _ = _select_one(root, c_puct=1.5, virtual_loss=1.0,
                            transpositions=tt)
    # Both edges' children must be the SAME node, supplied by the TT.
    assert e_a.child is e_b.child, (
        "TT should have shared the child across both edges"
    )
    # And that shared node sits in the TT at the canonical key.
    assert tt[target_key] is e_a.child


def test_transposition_disabled_creates_separate_nodes():
    """Same setup as above, but with `transpositions=None`. Two
    descents produce two distinct MCTSNode instances even though
    their state_keys collide, confirming that the TT is what
    makes sharing happen (not some other code path)."""
    from tools.mcts import _select_one
    from classes import (
        GameState, GlobalInfo, Map, SideInfo,
    )

    def _mk_gs():
        return GameState(
            game_id="t",
            map=Map(size_x=4, size_y=4, mask=set(), fog=set(),
                    hexes=set(), units=set()),
            global_info=GlobalInfo(
                current_side=2, turn_number=3, time_of_day="dawn",
                village_gold=2, village_upkeep=1, base_income=2,
            ),
            sides=[SideInfo(player="x", recruits=[], current_gold=0,
                            base_income=2, nb_villages_controlled=0)],
        )

    class _StubSim:
        def __init__(self):
            self.gs = _mk_gs()
            self.done = False
            self.winner = 0
            self.ended_by = ""
        def fork(self):
            return _StubSim()
        def step(self, action):  # noqa: ARG002
            self.gs = _mk_gs()

    root = MCTSNode(_StubSim())
    root.expanded = True
    e_a = _attach(root, _make_node(side=1), prior=0.51)
    e_b = _attach(root, _make_node(side=1), prior=0.50)
    e_a.child = None
    e_b.child = None

    _select_one(root, c_puct=1.5, virtual_loss=1.0,
                transpositions=None)
    _select_one(root, c_puct=1.5, virtual_loss=1.0,
                transpositions=None)
    assert e_a.child is not e_b.child, (
        "without TT, each edge gets its own MCTSNode"
    )


# ---------------------------------------------------------------------
# End-to-end smoke: --mcts CLI flag actually runs sim_self_play
# ---------------------------------------------------------------------

def test_mcts_self_play_smoke(tmp_path):
    """Drive `tools/sim_self_play.py --mcts` for one tiny iteration
    and verify exit code 0 + an MCTS-mode log message. Catches:
      - import-graph regressions in tools/mcts_policy.py
      - CLI flag wiring on sim_self_play.py
      - the MCTSPolicy → trainer.step_mcts contract end-to-end

    The default model is 26M params; building it and running 2
    MCTS sims per move for a 5-turn game on CPU completes in ~1
    minute. The 10-minute timeout below leaves slack for slower
    CI machines.
    """
    import os
    import subprocess
    import sys as _sys

    project_root = Path(__file__).parent
    ckpt_out = tmp_path / "smoke_mcts.pt"
    env = os.environ.copy()

    cmd = [
        _sys.executable, "-u",
        str(project_root / "tools" / "sim_self_play.py"),
        "--iterations", "1",
        "--games-per-iter", "1",
        "--max-turns", "5",
        "--workers", "0",          # serial — easier to attribute log lines
        "--mcts",
        "--mcts-sims", "2",
        "--checkpoint-out", str(ckpt_out),
        "--device", "cpu",
        "--log-level", "INFO",
    ]
    proc = subprocess.run(cmd, env=env, cwd=str(project_root),
                          capture_output=True, text=True, timeout=600)
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 0, (
        f"sim_self_play.py --mcts exited {proc.returncode}.\n"
        f"--- stdout ---\n{proc.stdout[-2000:]}\n"
        f"--- stderr ---\n{proc.stderr[-2000:]}"
    )
    assert "MCTS mode enabled" in combined, (
        f"expected MCTS-mode banner in log, got:\n{combined[-2000:]}"
    )
    # Checkpoint should land at the requested path on iter completion.
    assert ckpt_out.exists(), (
        f"expected checkpoint at {ckpt_out}, not present after run"
    )
