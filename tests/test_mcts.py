#!/usr/bin/env python3
"""Regression tests for tools/mcts.py.

Focus areas:

1. **PUCT selection** â€” formula, tie-break, exploration vs exploitation
   trade-off across visit counts.

2. **Sign-flip in `_backup` â€” turns vs actions**. This is the
   Wesnoth-specific bit that diverges from textbook 2-player MCTS:
   a *turn* in Wesnoth contains many *actions* by the same side
   (move, attack, recruit, ...) before `end_turn` flips control.
   Textbook MCTS ("flip every backup step") is WRONG here. The
   correct rule is "flip iff this node's side differs from the leaf's
   side", which our `_backup` implements via
   `parent.side == leaf_side` per edge. These tests construct paths
   with same-side multi-action stretches and mixed paths spanning
   multiple turn transitions to lock that behavior down.

3. **Terminal-value semantics** â€” win/loss/draw from a given side's
   perspective; terminal-leaf shortcut avoids a model forward.

4. **Dirichlet noise** â€” mass conservation, eps=0/eps=1 endpoints.

5. **Visit-count extraction** â€” output format, zero-skip behavior.

These are pure-Python unit tests with hand-rolled MCTSNode/MCTSEdge
fixtures (no live model, no real sim). End-to-end `mcts_search`
integration coverage is not in this file â€” would require a small
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
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from wesnoth_ai.action_sampler import LegalActionPrior  # noqa: E402
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
    """Attach `child` to `parent` via a fresh edge; return the edge.
    Uses id(child) as the outcome key -- tests that exercise real
    state-keyed descent clear `.children` and let _select_one
    populate it."""
    edge = _make_edge(prior=prior)
    edge.children = {id(child): child}
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
# _backup â€” turns vs actions sign convention
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
    sign â€” both nodes are on side 1, so parent sees the leaf value
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
    """A full Wesnoth turn: A â†’ A â†’ A â†’ A â†’ leaf A. Five actions,
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
    """Mixed path: A â†’ A â†’ B â†’ B â†’ leaf B. The first two parents
    are on the wrong side (A) and should get -v; the third parent
    is on the right side (B, same as leaf) and should get +v.
    Textbook 'flip every step' would give the wrong sign on the
    Aâ†’A edge."""
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
    """Two transitions: A â†’ B â†’ A â†’ leaf A. With leaf_side=A, the
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
    e_visited.w_value = -45.0  # Q â‰ˆ -0.9
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
    """Convex combination (1-eps)*prior + eps*noise â€” total mass
    stays â‰ˆ1 since both components sum to 1."""
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
    """All non-zero visit counts pass through verbatim â€” no
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
# Transposition table â€” node sharing across paths
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
    from wesnoth_ai.classes import (
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
    # The root must hash differently from the canonical child state.
    # In a real game step() always progresses the state (MP spent,
    # turn counter advances), so a child can never share an ancestor's
    # state_key. If root collided with the children here, the TT would
    # resolve the root's own child back to root and _select_one would
    # descend the rootâ†’root self-loop forever.
    root_sim.gs.global_info.turn_number = 2
    root = MCTSNode(root_sim)
    # Root must be marked expanded so _select_one descends.
    root.expanded = True
    e_a = _attach(root, _make_node(side=1), prior=0.5)
    e_b = _attach(root, _make_node(side=1), prior=0.5)
    # Detach the pre-attached child stubs so _select_one creates
    # fresh ones via TT logic on first descent.
    e_a.children.clear()
    e_b.children.clear()
    # Bias selection: PUCT with zero visits picks max-prior; tie
    # broken by edge order. We set priors to differ so the FIRST
    # descent picks e_a, the SECOND picks e_b after virtual loss
    # tilts the choice. With virtual_loss=1.0 a single visit on
    # e_a's PUCT score drops it below e_b.
    e_a.prior = 0.51
    e_b.prior = 0.50

    target_key = state_key(_mk_gs())
    tt = {state_key(root_sim.gs): root}
    leaf_a, _, _ = _select_one(root, c_puct=1.5, virtual_loss=1.0,
                               transpositions=tt)
    leaf_b, _, _ = _select_one(root, c_puct=1.5, virtual_loss=1.0,
                               transpositions=tt)
    # Both edges' children must be the SAME node, supplied by the TT.
    assert e_a.sole_child is e_b.sole_child, (
        "TT should have shared the child across both edges"
    )
    # And that shared node sits in the TT at the canonical key.
    assert tt[target_key] is e_a.sole_child


def test_transposition_disabled_creates_separate_nodes():
    """Same setup as above, but with `transpositions=None`. Two
    descents produce two distinct MCTSNode instances even though
    their state_keys collide, confirming that the TT is what
    makes sharing happen (not some other code path)."""
    from tools.mcts import _select_one
    from wesnoth_ai.classes import (
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
    e_a.children.clear()
    e_b.children.clear()

    _select_one(root, c_puct=1.5, virtual_loss=1.0,
                transpositions=None)
    _select_one(root, c_puct=1.5, virtual_loss=1.0,
                transpositions=None)
    assert e_a.sole_child is not e_b.sole_child, (
        "without TT, each edge gets its own MCTSNode"
    )


# ---------------------------------------------------------------------
# No-op resample self-loop guard (the 2026-06-13 OOM regression)
# ---------------------------------------------------------------------

def test_noop_resample_does_not_self_loop():
    """A stochastic edge whose resampled step is a NO-OP (state_key
    unchanged) must NOT make the chance-node descent self-loop.

    Root cause of the 2026-06-13 overnight OOM: a recruit attempted
    on a fog-occupied castle hex returns from `step()` without
    changing anything except the per-turn rejection set, which is
    deliberately EXCLUDED from `state_key`. So `state_key(child) ==
    state_key(parent)`; the per-search transposition table then maps
    that key back to the PARENT node, `child IS parent`, and the
    selection descent loops forever forking a sim per iteration until
    the process OOMs (one game ran 3.5h then MemoryError'd in
    `_select_one`'s `path.append`).

    The fix routes a no-op resample to the `_NOOP_KEY` pseudo-terminal
    sentinel (not shared via the TT), so the descent stops. This test
    builds exactly that setup and asserts `_select_one` returns with a
    bounded path and a terminal leaf.
    """
    from tools.mcts import _select_one, _NOOP_KEY
    from wesnoth_ai.classes import GameState, GlobalInfo, Map, SideInfo, state_key

    def _mk_gs():
        return GameState(
            game_id="t",
            map=Map(size_x=4, size_y=4, mask=set(), fog=set(),
                    hexes=set(), units=set()),
            global_info=GlobalInfo(
                current_side=1, turn_number=5, time_of_day="dawn",
                village_gold=2, village_upkeep=1, base_income=2,
            ),
            sides=[SideInfo(player="x", recruits=[], current_gold=0,
                            base_income=2, nb_villages_controlled=0)],
        )

    # Stub sim whose step() is a NO-OP: the child gs hashes to the
    # same state_key as the parent -- exactly the fog-rejected-recruit
    # case. `last_step_rejected` mirrors what wesnoth_sim sets on the
    # __retry_recruit__ bail-out.
    class _NoopSim:
        def __init__(self):
            self.gs = _mk_gs()
            self.done = False
            self.winner = 0
            self.ended_by = ""
            self.last_step_rejected = False
            self._seed_salt = None

        def fork(self):
            return _NoopSim()

        def step(self, action):  # noqa: ARG002
            # No observable state change (only a rejection-set update
            # in the real sim, which state_key excludes).
            self.last_step_rejected = True

    import numpy as np

    root = MCTSNode(_NoopSim())
    root.expanded = True
    # A single STOCHASTIC recruit edge -- resample fires under
    # chance_nodes, and its step is the no-op above.
    lap = LegalActionPrior(
        action={"type": "recruit", "unit_type": "Spearman",
                "target_hex": SimpleNamespace(x=1, y=1)},
        prior=1.0, actor_idx=0, target_idx=None,
        weapon_idx=None, type_idx=None,
    )
    edge = MCTSEdge(lap)
    root.edges.append(edge)

    tt = {state_key(root.sim.gs): root}
    leaf, path, _ = _select_one(
        root, c_puct=1.5, virtual_loss=0.0,
        transpositions=tt,
        chance_nodes=True,
        sample_rng=np.random.default_rng(0),
    )

    # Terminated (no hang), bounded path, routed to the sentinel.
    assert len(path) == 1, f"expected a 1-edge path, got {len(path)}"
    assert leaf.is_terminal, "no-op leaf must be pseudo-terminal"
    assert _NOOP_KEY in edge.children, (
        "no-op resample must be parked under the _NOOP_KEY sentinel")
    # The sentinel child must NOT be the parent (that was the loop).
    assert edge.children[_NOOP_KEY] is not root


# ---------------------------------------------------------------------
# End-to-end smoke: --mcts CLI flag actually runs sim_self_play
# ---------------------------------------------------------------------

@pytest.mark.slow          # ~371s: see pytest.ini two-tier note
def test_mcts_self_play_smoke(tmp_path):
    """Drive `tools/sim_self_play.py --mcts` for one tiny iteration
    and verify exit code 0 + an MCTS-mode log message. Catches:
      - import-graph regressions in tools/mcts_policy.py
      - CLI flag wiring on sim_self_play.py
      - the MCTSPolicy â†’ trainer.step_mcts contract end-to-end

    The default model is 26M params; building it and running 2
    MCTS sims per move for a 5-turn game on CPU took ~1 minute pre-2026-07 and ~6 minutes since true-reachability
    masks made turns full-length (every unit really moves). The
    15-minute timeout leaves slack for slower machines.
    """
    import os
    import subprocess
    import sys as _sys

    project_root = Path(__file__).parent.parent
    ckpt_out = tmp_path / "smoke_mcts.pt"
    env = os.environ.copy()

    cmd = [
        _sys.executable, "-u",
        str(project_root / "tools" / "sim_self_play.py"),
        "--iterations", "1",
        "--games-per-iter", "1",
        "--max-turns", "5",
        "--workers", "0",          # serial â€” easier to attribute log lines
        "--mcts",
        "--mcts-sims", "2",
        "--checkpoint-out", str(ckpt_out),
        "--device", "cpu",
        "--log-level", "INFO",
    ]
    proc = subprocess.run(cmd, env=env, cwd=str(project_root),
                          capture_output=True, text=True, timeout=900)
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


# ---------------------------------------------------------------------
# First-play urgency (FPU)
# ---------------------------------------------------------------------

def test_fpu_unvisited_q_is_parent_value_minus_reduction():
    """In a losing position (node.value < 0), the legacy Q=0 init
    ranks every unvisited edge above the best visited one, so small
    sim budgets sweep instead of deepening. With fpu_reduction set,
    an unvisited edge scores at (parent value - reduction) and a
    decent visited move wins. c_puct is tiny so the Q term dominates
    and the test isolates the init value."""
    node = _make_node(side=1)
    node.value = -0.6           # network: this position is bad for us
    visited = _attach(node, _make_node(side=2), prior=0.5)
    visited.n_visits = 1
    visited.w_value = -0.5      # q = -0.5: least-bad known move
    unvisited = _attach(node, _make_node(side=2), prior=0.5)
    node._total_visits = 1

    # Legacy: unvisited q_init=0 beats q=-0.5.
    assert _puct_select(node, c_puct=0.01) is unvisited
    # FPU: unvisited q_init = -0.6 - 0.25 = -0.85 loses to -0.5.
    assert _puct_select(node, c_puct=0.01, fpu_reduction=0.25) is visited


def test_fpu_q_init_clamped_to_value_range():
    """node.value - reduction below -1 clamps to -1 (values live in
    [-1, +1]; an init outside the range would distort PUCT)."""
    node = _make_node(side=1)
    node.value = -0.95
    visited = _attach(node, _make_node(side=2), prior=0.5)
    visited.n_visits = 1
    visited.w_value = -0.999    # q = -0.999: nearly-lost move
    unvisited = _attach(node, _make_node(side=2), prior=0.5)
    node._total_visits = 1
    # q_init = clamp(-0.95 - 0.25) = -1.0 < -0.999 -> visited wins.
    assert _puct_select(node, c_puct=0.001, fpu_reduction=0.25) is visited


# ---------------------------------------------------------------------
# Root temperature sampling
# ---------------------------------------------------------------------

def _root_with_visits(*visit_counts):
    """Root whose i-th edge has action {"id": i} and the given visit
    count."""
    import numpy as _np
    from tools.mcts import sample_action  # noqa: F401  (import check)
    root = _make_node(side=1)
    for i, n in enumerate(visit_counts):
        e = _attach(root, _make_node(side=2), prior=1.0 / len(visit_counts))
        e.action = {"id": i}
        e.n_visits = n
        root._total_visits += n
    return root


def test_sample_action_tau1_is_proportional_to_visits():
    import numpy as np
    from tools.mcts import sample_action
    root = _root_with_visits(90, 10)
    rng = np.random.default_rng(7)
    draws = [sample_action(root, 1.0, rng)["id"] for _ in range(2000)]
    frac0 = draws.count(0) / len(draws)
    # Expected 0.9; allow generous sampling slack.
    assert 0.85 <= frac0 <= 0.95, frac0
    assert draws.count(1) > 0


def test_sample_action_zero_temperature_is_argmax():
    import numpy as np
    from tools.mcts import sample_action, best_action
    root = _root_with_visits(3, 42, 1)
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert sample_action(root, 0.0, rng) == best_action(root)


def test_sample_action_never_returns_unvisited():
    import numpy as np
    from tools.mcts import sample_action
    root = _root_with_visits(5, 0)
    rng = np.random.default_rng(3)
    for _ in range(200):
        assert sample_action(root, 1.0, rng)["id"] == 0


# ---------------------------------------------------------------------
# Tree reuse (state-key-checked)
# ---------------------------------------------------------------------

def test_tree_reuse_inherits_subtree_for_deterministic_actions():
    """End-to-end on a real sim + tiny model: search once, step a
    DETERMINISTIC root action (move/end_turn -- combat would diverge
    by RNG), verify the live successor state-key matches the searched
    child, then search again with reuse_root and assert the subtree
    object and its visit statistics are inherited."""
    import random as _random
    import torch
    from wesnoth_ai.classes import state_key
    from wesnoth_ai.transformer_policy import TransformerPolicy
    from tools.mcts import MCTSConfig, mcts_search
    from tools.scenario_pool import (
        random_setup, build_scenario_gamestate, load_factions,
    )
    from tools.wesnoth_sim import WesnothSim

    torch.manual_seed(0)
    policy = TransformerPolicy(
        d_model=64, num_layers=2, num_heads=4, d_ff=128,
        device=torch.device("cpu"),
    )
    load_factions()
    setup = random_setup(_random.Random(11), forced_faction=None)
    gs = build_scenario_gamestate(setup)
    sim = WesnothSim(gs, scenario_id=setup.scenario_id, max_turns=30)

    cfg = MCTSConfig(n_simulations=40, add_root_noise=False)
    root1 = mcts_search(
        sim, policy._inference_model, policy._inference_encoder, cfg)

    # Pick a deterministic, visited, expanded root edge.
    edge = next(
        (e for e in sorted(root1.edges, key=lambda e: -e.n_visits)
         if e.sole_child is not None and e.sole_child.expanded
         and not e.sole_child.is_terminal
         and e.action.get("type") in ("move", "end_turn")),
        None)
    if edge is None:
        import pytest as _pytest
        _pytest.skip("search visited no deterministic expanded edge "
                     "at 10 sims on this seed")
    child = edge.sole_child

    live = sim.fork()
    live.step(edge.action)
    assert state_key(live.gs) == state_key(child.sim.gs), (
        "deterministic action must reproduce the searched child state"
    )

    inherited = child._total_visits
    root2 = mcts_search(
        live, policy._inference_model, policy._inference_encoder, cfg,
        reuse_root=child)
    assert root2 is child
    # Each simulation backs up exactly one root visit on top of the
    # inherited statistics.
    assert root2._total_visits == inherited + cfg.n_simulations


# ---------------------------------------------------------------------
# Gumbel root: completed-Q target + sequential halving
# ---------------------------------------------------------------------

def test_gumbel_target_completed_q_math():
    """Worked example for extract_gumbel_policy_target: two visited
    edges with known Q, one unvisited edge completed with v_mix.
    Checks the mctx mix-value formula and that the softmax favors
    the high-Q visited action over an equal-prior unvisited one."""
    from tools.mcts import MCTSConfig, extract_gumbel_policy_target

    root = _make_node(side=1)
    root.value = 0.10                      # network v(root)
    e_good = _attach(root, _make_node(side=2), prior=0.4)
    e_good.action = {"id": 0}
    e_good.n_visits, e_good.w_value = 4, 0.4     # q = +0.1
    e_bad = _attach(root, _make_node(side=2), prior=0.4)
    e_bad.action = {"id": 1}
    e_bad.n_visits, e_bad.w_value = 4, -0.4      # q = -0.1
    e_unvisited = _attach(root, _make_node(side=2), prior=0.2)
    e_unvisited.action = {"id": 2}
    root._total_visits = 8

    cfg = MCTSConfig()
    target = extract_gumbel_policy_target(root, cfg)
    assert len(target) >= 2
    probs = {t[0]: t[3] for t in target}   # keyed by actor_idx=0 all..
    # All weights form a distribution.
    total = sum(t[3] for t in target)
    assert abs(total - 1.0) < 1e-9
    # Edge order matches root.edges order in the tuple list; map by
    # position instead of actor_idx (the stub edges share actor 0).
    weights = [t[3] for t in target]
    w_good, w_bad = weights[0], weights[1]
    assert w_good > w_bad
    # v_mix = (v_root + sum_visits * weighted_q) / (1 + sum_visits)
    # weighted_q over visited = (0.4*0.1 + 0.4*(-0.1)) / 0.8 = 0.0
    # => v_mix = 0.1 / 9. The unvisited edge's completed Q is small
    # positive => its weight must exceed e_bad's (q=-0.1) despite
    # half the prior.
    if len(weights) == 3:
        assert weights[2] > w_bad


def test_gumbel_root_search_integration():
    """Full mcts_search with the Gumbel root on a real sim + tiny
    model: a gumbel_action is chosen from the legal edges, the sim
    budget is respected, and the policy target is a distribution."""
    import random as _random
    import torch
    from wesnoth_ai.transformer_policy import TransformerPolicy
    from tools.mcts import (
        MCTSConfig, mcts_search, extract_gumbel_policy_target,
    )
    from tools.scenario_pool import (
        random_setup, build_scenario_gamestate, load_factions,
    )
    from tools.wesnoth_sim import WesnothSim

    torch.manual_seed(0)
    policy = TransformerPolicy(
        d_model=64, num_layers=2, num_heads=4, d_ff=128,
        device=torch.device("cpu"),
    )
    load_factions()
    setup = random_setup(_random.Random(5), forced_faction=None)
    gs = build_scenario_gamestate(setup)
    sim = WesnothSim(gs, scenario_id=setup.scenario_id, max_turns=30)

    cfg = MCTSConfig(n_simulations=12, gumbel_m=4, gumbel_root=True)
    root = mcts_search(
        sim, policy._inference_model, policy._inference_encoder, cfg)

    assert root.gumbel_action is not None
    assert any(e.action is root.gumbel_action for e in root.edges)
    assert 1 <= root._total_visits <= cfg.n_simulations
    # Candidates are capped at m: at most gumbel_m root edges visited.
    visited_edges = sum(1 for e in root.edges if e.n_visits > 0)
    assert visited_edges <= cfg.gumbel_m

    target = extract_gumbel_policy_target(root, cfg)
    total = sum(t[3] for t in target)
    assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------
# Chance nodes (per-visit outcome sampling)
# ---------------------------------------------------------------------

def test_seed_salt_samples_distinct_recruit_outcomes():
    """Real sim: forks with DIFFERENT seed salts stepping the SAME
    recruit action roll different traits (distinct state keys), while
    unsalted forks reproduce one predetermined outcome -- the frozen
    single-sample behavior chance nodes exist to fix."""
    import random as _random
    from wesnoth_ai.classes import state_key
    from tools.openers import recruit_type
    from tools.scenario_pool import (
        random_setup, build_scenario_gamestate, load_factions,
    )
    from tools.wesnoth_sim import WesnothSim

    load_factions()
    setup = random_setup(_random.Random(11), forced_faction=None)
    gs = build_scenario_gamestate(setup)
    sim = WesnothSim(gs, scenario_id=setup.scenario_id, max_turns=30)
    side = sim.gs.global_info.current_side
    unit_type = sim.gs.sides[side - 1].recruits[0]
    action = recruit_type(unit_type)(sim.gs, side)
    assert action is not None, "leader should be able to recruit at start"

    # Unsalted forks: bit-exact determinism (replay contract).
    keys_plain = set()
    for _ in range(3):
        f = sim.fork()
        f.step(action)
        keys_plain.add(state_key(f.gs))
    assert len(keys_plain) == 1

    # Salted forks: independent trait rolls.
    keys_salted = set()
    for i in range(6):
        f = sim.fork()
        f._seed_salt = f"test-salt-{i}"
        f.step(action)
        keys_salted.add(state_key(f.gs))
    assert len(keys_salted) >= 2, (
        "six independently-salted trait rolls should produce at "
        "least two distinct outcomes"
    )


def _chance_stub_root():
    """Root with ONE stochastic edge whose stub sim produces a NEW
    distinct state on every fork+step (monotone counter)."""
    from wesnoth_ai.classes import GameState, GlobalInfo, Map, SideInfo

    counter = {"n": 10}

    def _mk_gs(turn):
        return GameState(
            game_id="t",
            map=Map(size_x=4, size_y=4, mask=set(), fog=set(),
                    hexes=set(), units=set()),
            global_info=GlobalInfo(
                current_side=2, turn_number=turn, time_of_day="dawn",
                village_gold=2, village_upkeep=1, base_income=2,
            ),
            sides=[SideInfo(player="x", recruits=[], current_gold=0,
                            base_income=2, nb_villages_controlled=0)],
        )

    class _StubSim:
        def __init__(self, turn=1):
            self.gs = _mk_gs(turn)
            self.done = False
            self.winner = 0
            self.ended_by = ""
            self._seed_salt = ""
        def fork(self):
            return _StubSim(self.gs.global_info.turn_number)
        def step(self, action):  # noqa: ARG002
            counter["n"] += 1
            self.gs = _mk_gs(counter["n"])

    root = MCTSNode(_StubSim())
    root.expanded = True
    lap = LegalActionPrior(
        action={"type": "attack"}, prior=1.0,
        actor_idx=0, target_idx=None, weapon_idx=None, type_idx=None,
    )
    root.edges = [MCTSEdge(lap)]
    return root


def test_chance_nodes_accumulate_outcome_children():
    import numpy as np
    from tools.mcts import _select_one
    root = _chance_stub_root()
    edge = root.edges[0]
    rng = np.random.default_rng(0)
    for _ in range(4):
        _select_one(root, c_puct=1.5, virtual_loss=0.0,
                    chance_nodes=True, sample_rng=rng)
    assert len(edge.children) == 4, (
        "each traversal of a stochastic edge must sample a fresh "
        "outcome child"
    )


def test_chance_nodes_off_freezes_first_sample():
    import numpy as np
    from tools.mcts import _select_one
    root = _chance_stub_root()
    edge = root.edges[0]
    rng = np.random.default_rng(0)
    for _ in range(4):
        _select_one(root, c_puct=1.5, virtual_loss=0.0,
                    chance_nodes=False, sample_rng=rng)
    assert len(edge.children) == 1, (
        "legacy mode: first sampled outcome is frozen"
    )
