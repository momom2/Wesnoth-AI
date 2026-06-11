"""AlphaZero-style MCTS for the Wesnoth AI.

Walks the game tree from a root state, uses PUCT to select edges,
expands leaves via the policy/value network, backs up the value
estimate. Returns root visit counts -- the soft training target
the trainer's `step_mcts` consumes.

Design choices:

  - **Factored action space**: each MCTS edge is a concrete
    (actor, target, weapon) triple. `LegalActionPrior` from
    `action_sampler.enumerate_legal_actions_with_priors` produces
    them; we wrap each in an MCTSEdge.

  - **No rollouts**: AlphaZero-style. The leaf's value comes from
    the model's value head, NOT from a Monte-Carlo playout. Saves
    the cost of rolling out the rest of the game per simulation
    -- a 30-action playout would be 30x more model forwards or 30x
    more sim steps with a fast random policy. Trades some accuracy
    for tree depth.

  - **Transposition table** (default ON). When two paths converge
    on the same `state_key(gs)`, share a single MCTSNode. Visit
    counts and Q-values then reflect every path that reached the
    state. Empirical hit rate on Wesnoth states is low (~0.4% at
    20 sims/move) because per-action mutations to HP/MP/villages
    keep state_keys unique; the win comes mostly from intra-turn
    move-reordering convergence.

  - **Cliffness-scaled bootstrap weighting** (opt-in). On backup,
    scale a non-terminal leaf's `v` by
    `1 - cliffness_bootstrap_alpha * cliffness/cliffness_max` so
    high-cliffness leaves contribute less to ancestor Q-values.
    Terminal leaves are unaffected (their value is exact).

  - **Cliffness-driven adaptive sim budget** (opt-in). After root
    expansion, scale total sims between `n_simulations_min` and
    `n_simulations_max` based on root cliffness. Spend more
    compute when the network is uncertain about the position.

  - **Sign convention**: the model's value head is "from
    current_side's perspective". Each leaf's value is the leaf's
    own-side estimate; on backup we flip the sign whenever the
    edge's parent side differs from the leaf's side. For 2p ladder
    games sides alternate every action, so the sign typically
    flips at every backup step.

  - **Dirichlet root noise**: AlphaZero's exploration trick. At the
    root only, blend Dirichlet draws into the priors so the search
    explores actions the model is currently dismissive of. Off
    during evaluation; on by default during training data
    generation.

Usage:

    from tools.mcts import MCTSConfig, mcts_search, extract_visit_counts
    config = MCTSConfig(n_simulations=200, c_puct=1.5)
    root = mcts_search(sim, model, encoder, config)
    visits = extract_visit_counts(root)
    z = ...   # terminal outcome from sim.current_side perspective
    exp = MCTSExperience(
        game_state=sim.gs, visit_counts=visits, z=z,
    )
    trainer.step_mcts([exp])

Performance: dominated by the model forward at each leaf expansion
(~30-35ms on CPU for our default-sized model: d=128, 3 layers).
50 simulations => ~1.6s per move single-thread.

Batched-leaf evaluation (`MCTSConfig.batch_size > 1`) is wired up
via virtual loss but on CPU it's a NET SLOWDOWN at our token
sequence length: B=1 = 33ms/sim, B=4 = 54ms/sim, B=8 = 56ms/sim
(measured 2026-04, default model, 30 sims/run). Same root cause
as `train_batch_size = 1` in the trainer: torch's MHA on CPU
prefers many small invocations over one big padded one at
~1750-token sequences. On GPU the inverse holds; flip
`batch_size` up there.
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Make project root importable when run as a script.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

from action_sampler import (
    LegalActionPrior,
    enumerate_legal_actions_with_priors,
)
from classes import GameState, state_key
from encoder import GameStateEncoder
from model import WesnothModel
from wesnoth_sim import WesnothSim


log = logging.getLogger("mcts")


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class MCTSConfig:
    """Hyperparameters for one MCTS search.

    Defaults are AlphaZero-paper-ish, scaled down for our smaller
    branching factor (~80-200 legal actions vs Go's 362)."""

    n_simulations:    int   = 100
    # PUCT exploration constant. Higher = more exploration. AlphaZero
    # uses ~1.0; chess-AlphaZero uses ~1.25-2.5. We default 1.5
    # because legal-action counts on a typical Wesnoth state vary
    # wildly across game phases (3 actions early game, 200+
    # mid-game) so a slightly aggressive constant prevents the
    # search from getting stuck on the model's first guess.
    c_puct:           float = 1.5

    # Dirichlet noise added to root priors for exploration. AlphaZero
    # paper uses alpha = 10/avg-legal-actions. Chess-AlphaZero uses
    # 0.3. With our ~100 average legal actions, alpha=0.3 is roughly
    # the right scale; tune up for more uniform exploration, down
    # for more peaked.
    dirichlet_alpha:  float = 0.3
    dirichlet_eps:    float = 0.25
    add_root_noise:   bool  = True

    # Optional cap on per-search wall time (seconds). When set, the
    # search aborts after `time_budget` even if it hasn't hit
    # n_simulations yet. Useful for self-play data generation where
    # you'd rather collect more games than over-search any one.
    time_budget:      Optional[float] = None

    # Batched leaf evaluation: how many simulations to run in parallel
    # via virtual-loss before doing one batched model forward. 1 is
    # serial (equivalent to the simple algorithm). Setting >1
    # diverges parallel rollouts via temporary "this looks bad"
    # virtual losses on selected edges so each sim explores a
    # different path; once `batch_size` leaves accumulate, they're
    # forwarded together via model.forward_batch.
    #
    # GPU: bigger batches amortize per-launch overhead; B=8-32 is
    # typically a 5-10x speedup over B=1.
    # CPU: PyTorch's batched MHA can be SLOWER than B=1 at our
    # token sequence lengths (see the trainer.py train_batch_size
    # comment for benchmark rationale). Default B=1 for that reason
    # -- override on GPU.
    batch_size:       int   = 1
    # Virtual-loss strength. Higher = stronger discouragement of
    # parallel rollouts hitting the same edge. AlphaZero paper uses
    # 1; AlphaGo Zero used 3. With a small-to-medium batch_size,
    # 1 is plenty.
    virtual_loss:     float = 1.0

    # Transposition table: when two paths converge on the same
    # `state_key(gs)`, share a single MCTSNode rather than building
    # two parallel subtrees. Visit counts and Q-values then reflect
    # ALL paths that reached the state, which is the correct PUCT
    # semantic ("N(s) = total visits to state s"). The TT is built
    # fresh per `mcts_search` call (cleared between root searches),
    # so memory is bounded by states-explored-per-search.
    #
    # In Wesnoth this hits whenever within-turn action ordering is
    # commutative (move A then move B vs move B then move A both
    # reach the same end-of-turn state). Default ON; toggle off to
    # benchmark the win or to debug a suspect tree.
    use_transposition_table: bool = True

    # ----- Cliffness consumers (all default off; enable per-search) -----
    #
    # Cliffness = std(Z(s)), the spread of the categorical value
    # distribution the network predicts. A low value means the
    # network is committing to a definite outcome at this state;
    # high means it admits the value is uncertain. We expose two
    # knobs that let a search take that signal into account:
    #
    # 1. BOOTSTRAP WEIGHTING: on backup, shrink a non-terminal
    #    leaf's v contribution toward 0 by treating it as a noisy
    #    estimate with variance = cliffness². The Bayesian-optimal
    #    blend toward the prior (uniform on [-1, +1], variance
    #    1/3) is
    #        scale = sigma_prior_sq / (sigma_prior_sq + cliffness²)
    #    Terminal leaves bypass this -- their value is exact.
    #    `cliffness_bootstrap_alpha` is a multiplier on the
    #    cliffness² term; alpha=1.0 is the Bayes-optimal default,
    #    alpha=0 disables shrinkage entirely.
    #
    # 2. ADAPTIVE SIM BUDGET: after root expansion, override
    #    `n_simulations` to interpolate between
    #    `n_simulations_min` and `n_simulations_max` based on
    #    root cliffness. Spend more sims at uncertain positions.
    #    Default OFF: the linear-interp shape is uncalibrated --
    #    we want to log root cliffness on real positions before
    #    picking a schedule.
    #
    # `cliffness_max` is the normalizer for the adaptive budget.
    # See the constant definition near `_value_atoms` in model.py
    # (and BACKLOG.md "Cliffness magic number 0.577") for the
    # full derivation; in short: 1/sqrt(3) is the std of the
    # continuous uniform on [-1, +1], and the discrete uniform
    # on K=51 atoms matches it to 3 decimal places. Above that
    # the network is saying "any outcome possible".
    cliffness_bootstrap_alpha:     float = 0.0

    adaptive_sim_budget:           bool  = False
    n_simulations_min:             int   = 100
    n_simulations_max:             int   = 400

    cliffness_max:                 float = 0.577


# ---------------------------------------------------------------------
# Tree structures
# ---------------------------------------------------------------------

class MCTSEdge:
    """One outgoing action from a node. Stores PUCT statistics:
    visit count, summed value, prior, lazy-created child node."""
    __slots__ = ("action", "actor_idx", "type_idx", "target_idx",
                 "weapon_idx", "prior", "n_visits", "w_value", "child")

    def __init__(self, lap: LegalActionPrior):
        self.action     = lap.action
        self.actor_idx  = lap.actor_idx
        self.type_idx   = lap.type_idx
        self.target_idx = lap.target_idx
        self.weapon_idx = lap.weapon_idx
        self.prior      = lap.prior
        self.n_visits   = 0
        self.w_value    = 0.0
        self.child: Optional["MCTSNode"] = None

    @property
    def q_value(self) -> float:
        # Initialize unvisited edges to 0 (AlphaZero convention; some
        # MCTS variants use parent-Q or pessimistic-init).
        return 0.0 if self.n_visits == 0 else self.w_value / self.n_visits


class MCTSNode:
    """One game state in the search tree. Owns its sim fork so
    multiple branches don't interfere."""
    __slots__ = ("sim", "side", "edges", "is_terminal",
                 "expanded", "_total_visits",
                 "tt_hits", "tt_misses",
                 "cliffness")

    def __init__(self, sim: WesnothSim):
        self.sim:           WesnothSim   = sim
        self.side:          int          = sim.gs.global_info.current_side
        self.edges:         List[MCTSEdge] = []
        self.is_terminal:   bool         = sim.done
        self.expanded:      bool         = False
        self._total_visits: int          = 0
        # TT stats are populated only on the ROOT node returned by
        # mcts_search (default 0 elsewhere); see the search loop.
        self.tt_hits:       int          = 0
        self.tt_misses:     int          = 0
        # std(Z(s)) from the network's distributional value head.
        # Set during _expand / _populate_leaf. Stays at 0 for
        # terminal leaves and unexpanded nodes (we have no
        # network estimate for those, but cliffness=0 = "exact"
        # is the right semantic — terminal values are exact).
        self.cliffness:     float        = 0.0


# ---------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------

def _terminal_value(sim: WesnothSim, side: int) -> float:
    """v(terminal_state) from `side`'s perspective.
    Wesnoth ties / timeouts: 0. Win: +1. Loss: -1."""
    if sim.winner == 0:
        return 0.0
    return 1.0 if sim.winner == side else -1.0


def _expand(
    node: MCTSNode,
    model: WesnothModel,
    encoder: GameStateEncoder,
) -> float:
    """Forward the model on `node.sim.gs`, build edges from the
    legal-action priors, return v(s) from node.side's perspective.

    No-op + returns terminal value if the node is already terminal
    (saves a model forward)."""
    if node.is_terminal:
        node.expanded = True
        return _terminal_value(node.sim, node.side)
    with torch.no_grad():
        encoded = encoder.encode(node.sim.gs)
        output = model(encoded)
        priors = enumerate_legal_actions_with_priors(
            encoded, output, node.sim.gs)
        v = float(output.value.squeeze().item())
        node.cliffness = float(output.cliffness.squeeze().item())
    if not priors:
        # No legal actions left -- treat the node as terminal with
        # neutral value. Shouldn't happen on real maps (end_turn is
        # always legal), but guards against pathological states.
        node.is_terminal = True
        node.expanded = True
        return 0.0
    # Sort by descending prior so the early simulations naturally
    # explore the model's preferred moves first (cosmetic; PUCT
    # would discover them anyway).
    priors.sort(key=lambda p: -p.prior)
    node.edges = [MCTSEdge(p) for p in priors]
    node.expanded = True
    return v


def _puct_select(node: MCTSNode, c_puct: float) -> MCTSEdge:
    """Pick the edge maximizing
        U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
    AlphaZero's standard PUCT score. Initial Q is 0 for unvisited
    edges, so the first few selections track P (the prior) closely."""
    sqrt_total = math.sqrt(max(1, node._total_visits))
    best_edge: Optional[MCTSEdge] = None
    best_score = -float("inf")
    for edge in node.edges:
        u = (edge.q_value
             + c_puct * edge.prior * sqrt_total / (1 + edge.n_visits))
        if u > best_score:
            best_score = u
            best_edge  = edge
    return best_edge  # type: ignore[return-value]


def _add_dirichlet_noise(
    node: MCTSNode, alpha: float, eps: float, rng: np.random.Generator,
) -> None:
    """Mix Dirichlet(alpha) noise into root priors for exploration:
       P'(a) = (1 - eps) * P(a) + eps * eta(a)
    where eta ~ Dirichlet([alpha] * |A|). Off-root noise would just
    add variance to deep search; AlphaZero applies it only at the
    root."""
    n = len(node.edges)
    if n == 0:
        return
    noise = rng.dirichlet([alpha] * n)
    for edge, eta in zip(node.edges, noise):
        edge.prior = (1.0 - eps) * edge.prior + eps * float(eta)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def _select_one(
    root:           MCTSNode,
    c_puct:         float,
    virtual_loss:   float,
    transpositions: Optional[Dict[int, "MCTSNode"]] = None,
    stats:          Optional[Dict[str, int]] = None,
) -> Tuple[MCTSNode, List[Tuple[MCTSNode, MCTSEdge]]]:
    """Walk down from root picking PUCT-best edges. Lazy-create
    child nodes by forking the parent sim and applying the edge's
    action. Apply virtual loss to each selected edge so the next
    parallel selection in the same batch is biased away from this
    path. Returns (leaf, path) where path is the list of
    (parent_node, edge_taken) pairs; backup walks it in reverse.

    If `transpositions` is provided, the table is consulted before
    creating a new child: if `state_key(child_sim.gs)` already maps
    to a node, that node is shared (multiple edges from different
    parents point at the same MCTSNode). Visit counts and Q-values
    on the shared node's outgoing edges then reflect every path
    that reached the state. Backup is unaffected -- it walks the
    edges of the path, not the nodes themselves.
    """
    path: List[Tuple[MCTSNode, MCTSEdge]] = []
    node = root
    while node.expanded and not node.is_terminal:
        edge = _puct_select(node, c_puct)
        if edge.child is None:
            child_sim = node.sim.fork()
            try:
                child_sim.step(edge.action)
            except Exception as e:
                log.debug(f"mcts: step rejected: {e}; treating as terminal")
                child_sim.done = True
                child_sim.winner = 0
                child_sim.ended_by = "step_error"
            # Transposition lookup: if another path already reached
            # this state, share its MCTSNode. Skip the lookup for
            # error-marked sims (state_key on a malformed gs would
            # itself raise, and there's nothing useful to share
            # with anyway).
            if (transpositions is not None and not child_sim.done) or (
                transpositions is not None and child_sim.done
                and child_sim.ended_by != "step_error"
            ):
                key = state_key(child_sim.gs)
                cached = transpositions.get(key)
                if cached is not None:
                    edge.child = cached
                    if stats is not None:
                        stats["hits"] = stats.get("hits", 0) + 1
                else:
                    new_child = MCTSNode(child_sim)
                    transpositions[key] = new_child
                    edge.child = new_child
                    if stats is not None:
                        stats["misses"] = stats.get("misses", 0) + 1
            else:
                edge.child = MCTSNode(child_sim)
        # Apply virtual loss BEFORE descending so the next
        # parallel selection in the same batch sees an edge that
        # currently looks bad (q drops, PUCT score drops). Real
        # backup undoes this and adds the actual update.
        if virtual_loss > 0:
            edge.n_visits += virtual_loss
            edge.w_value -= virtual_loss
            node._total_visits += virtual_loss
        path.append((node, edge))
        node = edge.child
    return node, path


# Variance of the prior distribution over outcomes, used as
# σ²_prior in the Bayesian-precision bootstrap shrinkage. We
# treat the prior over v as uniform on [V_MIN, V_MAX] = [-1, +1],
# whose variance is (V_MAX - V_MIN)² / 12 = 4/12 = 1/3.
# Choice of "uniform" prior matches what a freshly-initialized
# C51 head would output (uniform logits → uniform softmax →
# uniform distribution over atoms): the value-head's max-entropy
# state is the natural "I know nothing" prior.
_BOOTSTRAP_PRIOR_VAR = 1.0 / 3.0


def _backup(
    path:                List[Tuple[MCTSNode, MCTSEdge]],
    v:                   float,
    leaf_side:           int,
    virtual_loss:        float,
    leaf_cliffness:      float = 0.0,
    bootstrap_alpha:     float = 0.0,
    leaf_is_terminal:    bool  = False,
) -> None:
    """Walk the path in reverse, undoing virtual loss and adding the
    real visit. `v` is from `leaf_side`'s perspective; flip per edge
    when the parent's side differs.

    If `bootstrap_alpha > 0` AND the leaf is non-terminal, treat
    `v` as a noisy estimate of the true value with variance
    `bootstrap_alpha * cliffness²` (cliffness IS the std of the
    network's predicted value distribution, so cliffness² is its
    variance — the network is already telling us its own
    uncertainty). The Bayesian-optimal blend toward the prior
    (uniform on [-1, +1], variance 1/3) is

        scale = sigma_prior² / (sigma_prior² + alpha * cliffness²)

    At cliffness=0 the scale is 1 (full trust); at cliffness=∞
    the scale is 0 (full shrink to prior). At alpha=1 this is the
    Bayes-optimal posterior mean under the variance assumption;
    alpha lets the caller dial overall aggressiveness.

    Linear-schedule alternative (which is what we shipped first
    before reasoning through the Bayesian form): scale linearly
    from 1 at cliffness=0 to 0 at cliffness=cliffness_max. Strictly
    more aggressive than Bayesian at every cliffness > 0 and not
    grounded in any specific assumption about value noise; the
    Bayesian form has zero free hyperparameters once you accept
    the prior. See BACKLOG.md for the comparison table.

    Terminal leaves bypass shrinkage: their value is the true
    game outcome, not a network estimate, and cliffness is
    meaningless there.

    Visit counts (`edge.n_visits`, `parent._total_visits`) always
    increment by 1 regardless of cliffness — visits are about
    PUCT exploration credit, which is independent of how much we
    trust the value backed up. Only `w_value` is scaled."""
    if bootstrap_alpha > 0 and not leaf_is_terminal:
        var_v = bootstrap_alpha * (leaf_cliffness ** 2)
        scale = _BOOTSTRAP_PRIOR_VAR / (_BOOTSTRAP_PRIOR_VAR + var_v)
        v_eff = v * scale
    else:
        v_eff = v
    for parent, edge in reversed(path):
        if virtual_loss > 0:
            edge.n_visits -= virtual_loss
            edge.w_value += virtual_loss
            parent._total_visits -= virtual_loss
        edge.n_visits += 1
        parent._total_visits += 1
        edge.w_value += v_eff if parent.side == leaf_side else -v_eff


def _adaptive_n_sims(config: MCTSConfig, root_cliffness: float) -> int:
    """How many simulations the search should run given the root
    state's cliffness. When `config.adaptive_sim_budget` is False,
    this is just `config.n_simulations` (the caller's request).
    Otherwise it's a linear interpolation from
    `n_simulations_min` (cliffness=0) to `n_simulations_max`
    (cliffness >= `cliffness_max`).

    Pulled out as a helper so it can be unit-tested without a
    full search; called once per `mcts_search` invocation."""
    if not config.adaptive_sim_budget:
        return config.n_simulations
    norm = min(1.0, max(0.0,
                        root_cliffness / max(1e-6, config.cliffness_max)))
    return int(round(
        config.n_simulations_min
        + norm * (config.n_simulations_max - config.n_simulations_min)
    ))


def _populate_leaf(
    leaf:    MCTSNode,
    encoded,
    output,
) -> float:
    """Build the leaf's edges from the model's enumerated priors and
    return v from leaf.side's perspective. Mirrors the post-forward
    half of `_expand`. Also records the network's cliffness on the
    leaf node so the backup phase can downweight unreliable
    bootstraps."""
    leaf.cliffness = float(output.cliffness.squeeze().item())
    priors = enumerate_legal_actions_with_priors(
        encoded, output, leaf.sim.gs)
    if not priors:
        leaf.is_terminal = True
        leaf.expanded = True
        return 0.0
    priors.sort(key=lambda p: -p.prior)
    leaf.edges = [MCTSEdge(p) for p in priors]
    leaf.expanded = True
    return float(output.value.squeeze().item())


def mcts_search(
    sim:     WesnothSim,
    model:   WesnothModel,
    encoder: GameStateEncoder,
    config:  Optional[MCTSConfig] = None,
    *,
    rng:     Optional[np.random.Generator] = None,
) -> MCTSNode:
    """Run MCTS from `sim`'s state. Returns the root node with
    populated visit counts on outgoing edges.

    The caller's `sim` is NOT mutated -- the search runs on a fork.
    With `config.batch_size > 1`, leaves accumulate and are forwarded
    through `model.forward_batch` together (virtual-loss
    parallelization).

    Cost (B=1, CPU, default model): ~30-50 ms per simulation, model
    forward dominated. With B=8 on GPU: typically 5-10x speedup as
    forward overhead amortizes.
    """
    if config is None:
        config = MCTSConfig()
    if rng is None:
        rng = np.random.default_rng()
    import time as _time

    # Fork so the caller's sim is untouched.
    root_sim = sim.fork()
    root = MCTSNode(root_sim)

    # Per-search transposition table. Built fresh and dropped at
    # function exit -- a stale TT across searches would carry
    # outdated visit counts after the live game advances. Bounded
    # in size by states-explored-this-search (worst case ~n_sims).
    transpositions: Optional[Dict[int, MCTSNode]] = (
        {state_key(root_sim.gs): root}
        if config.use_transposition_table else None
    )
    # TT instrumentation: hit/miss counts surface as attrs on the
    # returned root so callers can audit whether the table is
    # actually saving work. Wesnoth's per-action state mutations
    # (MP/HP/villages) make true collisions rare in practice;
    # intra-turn move-reorderings are the main source.
    tt_stats: Dict[str, int] = {"hits": 0, "misses": 0}

    # Initial expand at root + optional Dirichlet noise on the priors.
    # Always serial (single state to evaluate at the start). Reads
    # the network's cliffness as a side-effect, which we then use
    # for the adaptive sim budget.
    _expand(root, model, encoder)
    if config.add_root_noise and root.edges:
        _add_dirichlet_noise(root, config.dirichlet_alpha,
                             config.dirichlet_eps, rng)

    # ----- Adaptive sim budget ---------------------------------------
    # Override n_simulations based on root cliffness. Linear
    # interpolation from n_min (cliffness=0, network confident) to
    # n_max (cliffness >= cliffness_max, network maxed-out
    # uncertain). Default OFF -- when off, n_simulations stays
    # whatever the caller asked for.
    n_sims = _adaptive_n_sims(config, root.cliffness)
    # Always log root cliffness (cheap; callers want this even
    # when adaptive_sim_budget is off, so they can decide what
    # an empirically reasonable budget schedule would look like
    # before flipping the switch). Surfaces on the returned
    # root via `root.cliffness` for programmatic consumers.
    log.debug(
        f"mcts: root cliffness={root.cliffness:.3f} "
        f"(adaptive={'on' if config.adaptive_sim_budget else 'off'}, "
        f"n_sims={n_sims})"
    )

    deadline = (_time.perf_counter() + config.time_budget
                if config.time_budget is not None else None)

    B = max(1, int(config.batch_size))
    V_LOSS = float(config.virtual_loss) if B > 1 else 0.0
    sims_done = 0

    while sims_done < n_sims:
        if deadline is not None and _time.perf_counter() > deadline:
            log.info(f"mcts: stopping at {sims_done}/{n_sims} "
                     f"(time budget hit)")
            break

        # How many sims to run in this batch.
        n_this_batch = min(B, n_sims - sims_done)

        # ----- Phase 1: select leaves --------------------------------
        # Each iteration descends from root via PUCT, applying virtual
        # loss as it goes. Terminal-leaf paths are backed up
        # immediately (no model forward needed). Non-terminal leaves
        # accumulate in `pending` for batched forward.
        pending: List[Tuple[MCTSNode, List[Tuple[MCTSNode, MCTSEdge]]]] = []
        for _ in range(n_this_batch):
            leaf, path = _select_one(
                root, config.c_puct, V_LOSS,
                transpositions=transpositions,
                stats=tt_stats,
            )
            if leaf.is_terminal:
                # Immediate backup -- no model forward needed.
                # Terminal v is exact; bootstrap weighting must be
                # disabled for this path (cliffness has no meaning
                # at a terminal, and we shouldn't shrink an exact
                # outcome anyway).
                v = _terminal_value(leaf.sim, leaf.side)
                _backup(
                    path, v, leaf.side, V_LOSS,
                    leaf_cliffness=0.0,
                    bootstrap_alpha=0.0,
                    leaf_is_terminal=True,
                )
                sims_done += 1
            else:
                pending.append((leaf, path))

        # ----- Phase 2: batch evaluate the non-terminal leaves -------
        if pending:
            # Deduplicate leaves: with virtual loss, two parallel
            # selections rarely converge on the same unexpanded leaf,
            # but it CAN happen (e.g. tiny tree, tied priors). Encode
            # + forward each unique leaf once; share the value across
            # all paths that reached it.
            unique_leaves: Dict[int, MCTSNode] = {}
            for leaf, _ in pending:
                if id(leaf) not in unique_leaves:
                    unique_leaves[id(leaf)] = leaf
            unique_list = list(unique_leaves.values())

            with torch.no_grad():
                encoded_list = [encoder.encode(l.sim.gs) for l in unique_list]
                outputs = model.forward_batch(encoded_list)

            # Build edges + record value per unique leaf.
            # _populate_leaf also stamps cliffness onto the leaf
            # node, which the backup phase below reads.
            leaf_values: Dict[int, float] = {}
            for leaf, encoded, output in zip(
                    unique_list, encoded_list, outputs):
                leaf_values[id(leaf)] = _populate_leaf(leaf, encoded, output)

            # Backup all pending paths (each may share or own its leaf).
            for leaf, path in pending:
                v = leaf_values[id(leaf)]
                _backup(
                    path, v, leaf.side, V_LOSS,
                    leaf_cliffness=leaf.cliffness,
                    bootstrap_alpha=config.cliffness_bootstrap_alpha,
                    leaf_is_terminal=leaf.is_terminal,
                )
                sims_done += 1

    # Surface TT stats on the returned root so callers (test code,
    # logging in sim_self_play) can decide whether the table is
    # actually pulling its weight. Setattr on a __slots__ class
    # would fail, so we add the slot at MCTSNode definition; if
    # this is on the slots-free fallback path, the attribute
    # assignment just lands in __dict__.
    root.tt_hits = tt_stats["hits"]
    root.tt_misses = tt_stats["misses"]
    if tt_stats["hits"] + tt_stats["misses"] > 0:
        log.debug(
            f"mcts: TT hits={tt_stats['hits']} "
            f"misses={tt_stats['misses']} "
            f"hit_rate={tt_stats['hits'] / max(1, tt_stats['hits'] + tt_stats['misses']):.1%}"
        )
    return root


def extract_visit_counts(
    root: MCTSNode,
) -> List[Tuple]:
    """Convert the root's edge visit counts into the tuple format
    `MCTSExperience.visit_counts` expects. Skips zero-visit edges.

    Schema: 5-tuples (actor_idx, target_idx, weapon_idx, count, type_idx).
    Trainer's `_mcts_factored_policy_loss` consumes this; legacy 4-
    tuples without type_idx still work (caller-side fallback).

    Cast to int because virtual-loss accounting promotes `n_visits` to
    float during batched search (vloss=1.0 + add/sub round-trip leaves
    an integer-valued float). Trainer math is float-safe but we
    canonicalize to int here."""
    return [
        (e.actor_idx, e.target_idx, e.weapon_idx,
         int(e.n_visits), e.type_idx)
        for e in root.edges if e.n_visits > 0
    ]


def best_action(root: MCTSNode) -> Optional[dict]:
    """The action with the highest visit count at the root. Used at
    play time (after MCTS finishes) to pick a move. Visit count is
    the canonical AlphaZero choice -- prior + Q both feed into the
    PUCT that drove visits, so visits already integrate them."""
    if not root.edges:
        return None
    best = max(root.edges, key=lambda e: e.n_visits)
    return best.action if best.n_visits > 0 else None
