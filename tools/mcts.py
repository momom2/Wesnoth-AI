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
from draw_tiebreak import DrawTiebreakConfig, draw_tiebreak_z
from combat_outcomes import (
    enumerate_attack_outcomes, outcome_key_for_child,
)


log = logging.getLogger("mcts")

# Action types whose sim.step consumes synced RNG (combat damage
# rolls; recruit trait rolls). Chance-node sampling re-forks and
# re-steps these on EVERY traversal; everything else is
# deterministic and keeps its single cached successor.
_STOCHASTIC_ACTION_TYPES = frozenset({"attack", "recruit"})

# Children-dict key for the step-rejected pseudo-terminal (state_key
# on a malformed post-error state could itself raise).
_STEP_ERROR_KEY = "step_error"

# Sentinel: edge.outcome_probs not yet computed (None = computed and
# unavailable -> pure sampling).
_OUTCOMES_UNSET = "unset"

# Coverage threshold for switching from sampled to exact-probability
# outcome selection: once the observed children carry >= 1 - this of
# the exact mass, traversals select among them with renormalized
# exact probabilities and stop forking the sim. The ignored tail
# (< 0.1%) parallels the engine's own truncations (berserk rounds
# stop at 99% dead mass, probs snapped at 1e-9 -- see
# docs/wesnoth_rules.md); the alternative, waiting for full
# coverage, would keep Monte-Carlo forking for thousands of visits
# to chase outcomes that cannot influence a 16-200-sim search.
_EXACT_COVERAGE_EPSILON = 1e-3


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

    # Material tiebreaker for drawn terminals (see
    # tools/draw_tiebreak.py): a turn-cap draw at a search horizon
    # scores by village/gold/unit-value differential in
    # (-cap, +cap) instead of a flat 0, so PUCT prefers
    # material-ahead lines even before the policy can win outright,
    # and the value head's z targets carry gradient in drawn games.
    # None disables (legacy z=0 draws).
    draw_tiebreak: Optional[DrawTiebreakConfig] = None

    # First-play urgency (FPU): the Q used in PUCT for an edge that
    # has never been visited. AlphaZero's Q=0 init is fine when
    # values hover near 0, but in a clearly losing position (all
    # visited Q < 0) it makes every UNVISITED edge look better than
    # the best known move, so the search degenerates into a one-visit
    # sweep of all 100-200 legal actions and never deepens -- fatal
    # at our 25-100 sim budgets. KataGo / Leela instead initialize
    # an unvisited edge to (parent value - reduction): "assume an
    # unexplored child is slightly worse than the position itself".
    # 0.25 follows Leela's default; KataGo's paper ("Accelerating
    # Self-Play Learning in Go", Wu 2020, sec 5.1) reports the same
    # shape. Set to None to restore the legacy Q=0 init.
    fpu_reduction:    Optional[float] = 0.25
    # FPU at the ROOT when Dirichlet noise is on: noise exists to
    # force exploration of dismissed actions, and a root FPU penalty
    # would cancel it (KataGo applies no root FPU reduction with
    # noise for this reason). 0.0 = unvisited root edges score at
    # parent value, letting noise-boosted priors actually win a
    # visit.
    root_fpu_reduction: float = 0.0

    # Action selection at the root during SELF-PLAY: for the first
    # `temperature_decisions` decisions of each game, sample an
    # action proportional to visit_count^(1/temperature) instead of
    # argmax-visits (AlphaZero's tau=1 for the first 30 moves).
    # Without it self-play games are near-deterministic (root noise
    # is the only diversity source) and the data distribution
    # collapses. After the threshold, argmax. Consumed by
    # MCTSPolicy, not mcts_search itself.
    temperature:           float = 1.0
    temperature_decisions: int   = 30

    # Gumbel AlphaZero root (Danihelka et al. 2022, "Policy
    # improvement by planning with Gumbel"). Replaces
    # Dirichlet-noise + visit-count-temperature at the ROOT with:
    #   1. Gumbel-Top-k: sample `gumbel_m` DISTINCT candidate
    #      actions ∝ the prior (g(a) + logits(a), g ~ Gumbel(0,1)).
    #   2. Sequential halving: split the sim budget tournament-style
    #      across candidates, halving the field each phase by
    #      g + logits + sigma(q̂) -- the bandit-optimal way to find
    #      the best arm under a fixed budget.
    #   3. Play argmax g + logits + sigma(q̂): provably a policy
    #      improvement in expectation, even at tiny budgets.
    # The policy target becomes softmax(logits + sigma(completed_q))
    # over ALL legal actions (`extract_gumbel_policy_target`) --
    # unvisited actions fall back to the mixed value estimate
    # instead of an implicit zero, so no simulation is wasted.
    # Interior nodes keep PUCT + FPU (the paper's interior variant
    # is a smaller win; documented divergence).
    # sigma(q) = (c_visit + max_b N(b)) * c_scale * q, paper defaults.
    gumbel_root:    bool  = True
    gumbel_m:       int   = 16
    gumbel_c_visit: float = 50.0
    gumbel_c_scale: float = 1.0

    # Chance nodes for stochastic actions (combat, recruit traits).
    # When ON, every traversal of a stochastic edge re-forks the
    # parent sim with a fresh seed salt and re-steps -- sampling the
    # outcome from the TRUE distribution (the sim is bit-exact) --
    # and the edge keeps one child PER DISTINCT OUTCOME state
    # (keyed by state_key; strike-order permutations collapse).
    # Selection then continues below the sampled outcome, so the
    # subtree represents outcome-CONDITIONAL play: concentrate where
    # you hit high, spread where you hit low. The edge's Q converges
    # to E_outcome[V(adaptive response)] -- expectation AFTER the
    # max, not before (EV-collapse would systematically undervalue
    # multi-pronged aggression; see BACKLOG chance-node discussion).
    # When OFF, the legacy behavior: the first sampled outcome is
    # frozen and every visit descends it (single-sample
    # determinization).
    chance_nodes: bool = True

    # Exact outcome enumeration (Tier 1, see tools/combat_outcomes):
    # attack edges lazily compute the exact outcome distribution via
    # the prob_matrix-style DP (same parameters as the bit-exact
    # resolver). While unseen mass remains, traversals keep sampling
    # through the real sim (true-distribution Monte Carlo); once the
    # observed children cover ~all the mass, selection switches to
    # exact-probability choice among them with NO sim fork -- zero
    # Monte-Carlo noise and cheaper traversals in the common
    # ~10-outcome case. Fights the DP refuses (petrify, possible
    # advancement, berserk/complexity caps, any DP/sim mismatch)
    # fall back to sampling automatically.
    exact_outcome_enumeration: bool = True

    # Tree reuse across consecutive decisions (state-key-checked).
    # After the live game plays the chosen action, the subtree under
    # that root edge describes the searched successor state -- which
    # for DETERMINISTIC actions (moves, end_turn: the vast majority
    # of intra-turn decisions) is exactly the live successor, but for
    # combat is conditioned on the ONE RNG outcome the search
    # sampled, which the live game almost never reproduces. The
    # caller (MCTSPolicy) therefore reuses the subtree iff
    # state_key(live state) == state_key(searched child state):
    # deterministic steps match and inherit the whole subtree's
    # visits; combat mismatches and rebuilds from scratch. Zero
    # unsoundness by construction (modulo 64-bit state_key
    # collisions, the same assumption the transposition table
    # already makes).
    tree_reuse: bool = True

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
    """One outgoing action from a node. Stores PUCT statistics
    (visit count, summed value, prior) and the lazily-created
    successor node(s).

    `children` maps state_key(outcome state) -> MCTSNode. A
    deterministic action has exactly one entry; a stochastic action
    (attack, recruit) accumulates one entry per distinct sampled
    outcome when chance nodes are on. Edge statistics aggregate
    across outcomes, so q_value estimates
    E_outcome[V(best response | outcome)]."""
    __slots__ = ("action", "actor_idx", "type_idx", "target_idx",
                 "weapon_idx", "prior", "n_visits", "w_value",
                 "children",
                 "outcome_probs", "outcome_keys", "seen_mass")

    def __init__(self, lap: LegalActionPrior):
        self.action     = lap.action
        self.actor_idx  = lap.actor_idx
        self.type_idx   = lap.type_idx
        self.target_idx = lap.target_idx
        self.weapon_idx = lap.weapon_idx
        self.prior      = lap.prior
        self.n_visits   = 0
        self.w_value    = 0.0
        self.children: Dict = {}
        # Exact-enumeration bookkeeping (attack edges only):
        # outcome_probs: _OUTCOMES_UNSET | None | OutcomeDistribution.
        # outcome_keys maps children state_key -> OutcomeKey;
        # seen_mass = total exact probability of observed children.
        self.outcome_probs = _OUTCOMES_UNSET
        self.outcome_keys: Dict = {}
        self.seen_mass: float = 0.0

    @property
    def sole_child(self) -> Optional["MCTSNode"]:
        """The single successor of a deterministic edge, or None if
        the edge has zero or multiple (outcome-branched) children."""
        if len(self.children) == 1:
            return next(iter(self.children.values()))
        return None

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
                 "cliffness", "value", "gumbel_action")

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
        # Network value estimate v(s) from this node's own side's
        # perspective, stamped at expansion. Used as the FPU anchor
        # for this node's unvisited edges. 0.0 for terminal /
        # unexpanded nodes.
        self.value:         float        = 0.0
        # Action chosen by the Gumbel root procedure (set on the
        # ROOT by mcts_search when config.gumbel_root). None
        # elsewhere / in classic mode.
        self.gumbel_action: Optional[dict] = None


# ---------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------

def _terminal_value(
    sim:  WesnothSim,
    side: int,
    tiebreak: Optional[DrawTiebreakConfig] = None,
) -> float:
    """v(terminal_state) from `side`'s perspective. Win: +1.
    Loss: -1. Ties / turn-cap timeouts: 0, or the material
    differential in (-cap, +cap) when `tiebreak` is configured
    (see tools/draw_tiebreak.py)."""
    if sim.winner == 0:
        if tiebreak is not None:
            return draw_tiebreak_z(sim.gs, side, tiebreak)
        return 0.0
    return 1.0 if sim.winner == side else -1.0


def _expand(
    node: MCTSNode,
    model: WesnothModel,
    encoder: GameStateEncoder,
    tiebreak: Optional[DrawTiebreakConfig] = None,
) -> float:
    """Forward the model on `node.sim.gs`, build edges from the
    legal-action priors, return v(s) from node.side's perspective.

    No-op + returns terminal value if the node is already terminal
    (saves a model forward)."""
    if node.is_terminal:
        node.expanded = True
        return _terminal_value(node.sim, node.side, tiebreak)
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
    node.value = v
    return v


def _puct_select(
    node:   MCTSNode,
    c_puct: float,
    fpu_reduction: Optional[float] = None,
) -> MCTSEdge:
    """Pick the edge maximizing
        U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
    AlphaZero's standard PUCT score.

    Unvisited edges score with first-play urgency when
    `fpu_reduction` is set: Q_init = clamp(node.value - reduction)
    -- "an unexplored child is probably a bit worse than the
    position itself". With the legacy `None`, Q_init = 0, which in
    losing positions ranks every unexplored action above the best
    known one and flattens the search into a breadth-1 sweep (see
    MCTSConfig.fpu_reduction)."""
    sqrt_total = math.sqrt(max(1, node._total_visits))
    if fpu_reduction is None:
        q_init = 0.0
    else:
        q_init = max(-1.0, min(1.0, node.value - fpu_reduction))
    best_edge: Optional[MCTSEdge] = None
    best_score = -float("inf")
    for edge in node.edges:
        q = q_init if edge.n_visits == 0 else edge.q_value
        u = (q
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
    fpu_reduction:      Optional[float] = None,
    root_fpu_reduction: Optional[float] = None,
    forced_first_edge:  Optional[MCTSEdge] = None,
    chance_nodes:       bool = False,
    sample_rng:         Optional[np.random.Generator] = None,
    exact_outcomes:     bool = False,
) -> Tuple[MCTSNode, List[Tuple[MCTSNode, MCTSEdge]]]:
    """Walk down from root picking PUCT-best edges. Lazy-create
    child nodes by forking the parent sim and applying the edge's
    action. Apply virtual loss to each selected edge so the next
    parallel selection in the same batch is biased away from this
    path. Returns (leaf, path) where path is the list of
    (parent_node, edge_taken) pairs; backup walks it in reverse.

    With `chance_nodes`, every traversal of a STOCHASTIC edge
    (attack / recruit) re-forks the parent sim under a fresh seed
    salt and re-steps, sampling the outcome from the true
    distribution; the edge keeps one child per distinct outcome
    state. Deterministic edges keep their single cached successor
    (no re-fork). Without it, the first sampled outcome is frozen
    (legacy single-sample determinization).

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
        if forced_first_edge is not None:
            # Gumbel sequential halving pins the ROOT edge; interior
            # selection below stays PUCT.
            edge = forced_first_edge
            forced_first_edge = None
        else:
            fpu = root_fpu_reduction if node is root else fpu_reduction
            edge = _puct_select(node, c_puct, fpu)

        resample = (chance_nodes
                    and isinstance(edge.action, dict)
                    and edge.action.get("type")
                    in _STOCHASTIC_ACTION_TYPES)
        if edge.children and not resample:
            # Deterministic edge (or legacy frozen-sample mode):
            # descend the cached successor without re-stepping.
            child = next(iter(edge.children.values()))
        else:
            child = None
            dist = None
            if (resample and exact_outcomes
                    and edge.action.get("type") == "attack"):
                if edge.outcome_probs is _OUTCOMES_UNSET:
                    try:
                        edge.outcome_probs = enumerate_attack_outcomes(
                            node.sim.gs, edge.action)
                    except Exception as e:
                        log.warning(
                            f"mcts: outcome enumeration failed ({e}); "
                            f"sampling instead")
                        edge.outcome_probs = None
                dist = edge.outcome_probs
                if (dist is not None and sample_rng is not None
                        and edge.seen_mass >= 1.0 - _EXACT_COVERAGE_EPSILON
                        and edge.children):
                    # Full support observed: exact-probability choice
                    # among the known outcome children -- no sim
                    # fork, no Monte-Carlo noise.
                    keys = [k for k in edge.children
                            if k != _STEP_ERROR_KEY]
                    if keys:
                        w = np.array(
                            [dist.probs[edge.outcome_keys[k]]
                             for k in keys])
                        w /= w.sum()
                        pick = keys[int(sample_rng.choice(
                            len(keys), p=w))]
                        child = edge.children[pick]
            if child is None:
                child_sim = node.sim.fork()
                if resample and sample_rng is not None:
                    # Fresh salt => this traversal rolls independent
                    # synced RNG, sampling a fresh outcome.
                    # Search-only: live sims never carry a salt
                    # (replay fidelity).
                    child_sim._seed_salt = (
                        f"mcts{int(sample_rng.integers(1 << 62))}")
                step_error = False
                try:
                    child_sim.step(edge.action)
                except Exception as e:
                    log.debug(f"mcts: step rejected: {e}; "
                              f"treating as terminal")
                    child_sim.done = True
                    child_sim.winner = 0
                    child_sim.ended_by = "step_error"
                    step_error = True
                if step_error:
                    # state_key on a malformed post-error state could
                    # itself raise; park all error outcomes under one
                    # sentinel child.
                    child = edge.children.get(_STEP_ERROR_KEY)
                    if child is None:
                        child = MCTSNode(child_sim)
                        edge.children[_STEP_ERROR_KEY] = child
                else:
                    key = state_key(child_sim.gs)
                    child = edge.children.get(key)
                    if child is None:
                        # Transposition lookup: if another path
                        # already reached this state, share its
                        # MCTSNode.
                        cached = (transpositions.get(key)
                                  if transpositions is not None
                                  else None)
                        if cached is not None:
                            child = cached
                            if stats is not None:
                                stats["hits"] = stats.get("hits", 0) + 1
                        else:
                            child = MCTSNode(child_sim)
                            if transpositions is not None:
                                transpositions[key] = child
                                if stats is not None:
                                    stats["misses"] = (
                                        stats.get("misses", 0) + 1)
                        edge.children[key] = child
                    # Exact-enumeration bookkeeping: tie this child
                    # to its outcome and accumulate observed mass.
                    # A sampled outcome the DP doesn't know is a
                    # modeling gap -- distrust the whole distribution
                    # for this edge rather than risk biased
                    # selection.
                    if dist is not None and key not in edge.outcome_keys:
                        okey = outcome_key_for_child(
                            child_sim.gs, dist.attacker_id,
                            dist.defender_id)
                        p_o = dist.probs.get(okey)
                        if p_o is None:
                            log.warning(
                                "mcts: sampled combat outcome absent "
                                "from exact support; disabling "
                                "enumeration for this edge")
                            edge.outcome_probs = None
                        else:
                            edge.outcome_keys[key] = okey
                            edge.seen_mass += p_o
        # Apply virtual loss BEFORE descending so the next
        # parallel selection in the same batch sees an edge that
        # currently looks bad (q drops, PUCT score drops). Real
        # backup undoes this and adds the actual update.
        if virtual_loss > 0:
            edge.n_visits += virtual_loss
            edge.w_value -= virtual_loss
            node._total_visits += virtual_loss
        path.append((node, edge))
        node = child
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
    leaf.value = float(output.value.squeeze().item())
    return leaf.value


def _run_one_sim(
    root:           MCTSNode,
    model:          WesnothModel,
    encoder:        GameStateEncoder,
    config:         MCTSConfig,
    transpositions: Optional[Dict[int, MCTSNode]],
    tt_stats:       Dict[str, int],
    forced_first_edge: Optional[MCTSEdge] = None,
    sample_rng:        Optional[np.random.Generator] = None,
) -> None:
    """One serial simulation: select (optionally pinned through a
    given root edge), evaluate/expand the leaf, back up. Used by the
    Gumbel root procedure; the classic loop keeps its own batched
    variant."""
    leaf, path = _select_one(
        root, config.c_puct, 0.0,
        transpositions=transpositions, stats=tt_stats,
        fpu_reduction=config.fpu_reduction,
        root_fpu_reduction=config.fpu_reduction,
        forced_first_edge=forced_first_edge,
        chance_nodes=config.chance_nodes,
        sample_rng=sample_rng,
        exact_outcomes=config.exact_outcome_enumeration,
    )
    if leaf.is_terminal:
        v = _terminal_value(leaf.sim, leaf.side, config.draw_tiebreak)
        _backup(path, v, leaf.side, 0.0, leaf_is_terminal=True)
        return
    with torch.no_grad():
        encoded = encoder.encode(leaf.sim.gs)
        output = model(encoded)
    v = _populate_leaf(leaf, encoded, output)
    _backup(
        path, v, leaf.side, 0.0,
        leaf_cliffness=leaf.cliffness,
        bootstrap_alpha=config.cliffness_bootstrap_alpha,
        leaf_is_terminal=leaf.is_terminal,
    )


def _gumbel_sigma(q: float, max_visits: float, config: MCTSConfig) -> float:
    """The paper's monotone Q transform: sigma(q) =
    (c_visit + max_b N(b)) * c_scale * q. Scaling by the visit count
    keeps logits and Q commensurate as the search deepens."""
    return (config.gumbel_c_visit + max_visits) * config.gumbel_c_scale * q


def _gumbel_root_search(
    root:           MCTSNode,
    model:          WesnothModel,
    encoder:        GameStateEncoder,
    config:         MCTSConfig,
    transpositions: Optional[Dict[int, MCTSNode]],
    tt_stats:       Dict[str, int],
    rng:            np.random.Generator,
    n_sims:         int,
    deadline:       Optional[float] = None,
) -> None:
    """Gumbel-Top-k candidates + sequential halving at the root.
    Sets `root.gumbel_action` to argmax(g + logits + sigma(q̂)) over
    the final survivors -- in expectation a policy improvement over
    the raw prior (Danihelka et al. 2022, thm. on one-step
    improvement)."""
    import time as _time
    edges = root.edges
    if not edges:
        root.gumbel_action = None
        return
    logits = np.log(np.maximum(
        np.array([e.prior for e in edges], dtype=np.float64), 1e-12))
    g = rng.gumbel(size=len(edges))
    base = g + logits

    m = max(1, min(config.gumbel_m, len(edges)))
    cands: List[int] = list(np.argsort(-base)[:m])

    def _score(ci: int, max_v: float) -> float:
        return base[ci] + _gumbel_sigma(edges[ci].q_value, max_v, config)

    sims_done = 0
    num_phases = max(1, math.ceil(math.log2(m))) if m > 1 else 1
    for phase in range(num_phases):
        if len(cands) == 1 or sims_done >= n_sims:
            break
        # Even split of the remaining budget over remaining phases,
        # then over surviving candidates; every candidate gets at
        # least one sim per phase (subject to the global budget).
        phase_budget = max(len(cands),
                           (n_sims - sims_done) // (num_phases - phase))
        sims_per = max(1, phase_budget // len(cands))
        for ci in cands:
            for _ in range(sims_per):
                if sims_done >= n_sims:
                    break
                if deadline is not None and _time.perf_counter() > deadline:
                    log.info(f"mcts(gumbel): stopping at "
                             f"{sims_done}/{n_sims} (time budget hit)")
                    sims_done = n_sims
                    break
                _run_one_sim(root, model, encoder, config,
                             transpositions, tt_stats,
                             forced_first_edge=edges[ci],
                             sample_rng=rng)
                sims_done += 1
        max_v = max(e.n_visits for e in edges)
        keep = max(1, math.ceil(len(cands) / 2))
        cands = sorted(cands, key=lambda ci: -_score(ci, max_v))[:keep]

    max_v = max(e.n_visits for e in edges)
    best_ci = max(cands, key=lambda ci: _score(ci, max_v))
    root.gumbel_action = edges[best_ci].action


def mcts_search(
    sim:     WesnothSim,
    model:   WesnothModel,
    encoder: GameStateEncoder,
    config:  Optional[MCTSConfig] = None,
    *,
    rng:        Optional[np.random.Generator] = None,
    reuse_root: Optional[MCTSNode] = None,
) -> MCTSNode:
    """Run MCTS from `sim`'s state. Returns the root node with
    populated visit counts on outgoing edges.

    The caller's `sim` is NOT mutated -- the search runs on a fork.
    With `config.batch_size > 1`, leaves accumulate and are forwarded
    through `model.forward_batch` together (virtual-loss
    parallelization).

    `reuse_root`: a subtree node from a PREVIOUS search whose state
    the caller has verified (by state_key) to equal `sim`'s current
    state. The search continues from it, inheriting its visit
    statistics, and still runs the full `n_simulations` on top.
    Dirichlet root noise is (re-)applied to its priors, matching
    Leela/KataGo's reuse behavior. Pass only state-key-verified
    nodes -- see MCTSConfig.tree_reuse.

    Cost (B=1, CPU, default model): ~30-50 ms per simulation, model
    forward dominated. With B=8 on GPU: typically 5-10x speedup as
    forward overhead amortizes.
    """
    if config is None:
        config = MCTSConfig()
    if rng is None:
        rng = np.random.default_rng()
    import time as _time

    if (reuse_root is not None and reuse_root.expanded
            and not reuse_root.is_terminal):
        root = reuse_root
        root_sim = root.sim
        log.debug(f"mcts: reusing subtree with "
                  f"{root._total_visits} inherited visits")
    else:
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
    if not root.expanded:
        _expand(root, model, encoder, tiebreak=config.draw_tiebreak)
    # Gumbel mode replaces Dirichlet noise at the root: exploration
    # comes from sampling the candidate set without replacement.
    use_gumbel = bool(config.gumbel_root and root.edges
                      and not root.is_terminal)
    if config.add_root_noise and root.edges and not use_gumbel:
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

    if use_gumbel:
        _gumbel_root_search(root, model, encoder, config,
                            transpositions, tt_stats, rng, n_sims,
                            deadline)
        n_sims = 0   # the classic PUCT-at-root loop below is skipped

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
                fpu_reduction=config.fpu_reduction,
                root_fpu_reduction=(
                    config.root_fpu_reduction
                    if (config.add_root_noise
                        and config.fpu_reduction is not None)
                    else config.fpu_reduction),
                chance_nodes=config.chance_nodes,
                sample_rng=rng,
                exact_outcomes=config.exact_outcome_enumeration,
            )
            if leaf.is_terminal:
                # Immediate backup -- no model forward needed.
                # Terminal v is exact; bootstrap weighting must be
                # disabled for this path (cliffness has no meaning
                # at a terminal, and we shouldn't shrink an exact
                # outcome anyway).
                v = _terminal_value(leaf.sim, leaf.side,
                                    config.draw_tiebreak)
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


def extract_gumbel_policy_target(
    root:   MCTSNode,
    config: MCTSConfig,
) -> List[Tuple]:
    """The Gumbel-AlphaZero "completed Q" policy target:

        pi_target = softmax(logits + sigma(completed_q))

    over ALL legal root actions, where completed_q(a) = q̂(a) for
    visited actions and the MIXED VALUE estimate v_mix for unvisited
    ones (instead of the implicit 0 that visit-count targets give
    them). v_mix interpolates the root's own network value with the
    prior-weighted mean of the visited actions' search Q
    (mctx's `qtransform_completed_by_mix_value`):

        v_mix = (v_root + sum_visits * weighted_q) / (1 + sum_visits)

    Same 5-tuple schema as `extract_visit_counts`
    (actor, target, weapon, weight, type); weights are floats
    summing to ~1 -- the trainer normalizes by total weight, so
    counts and probabilities train identically."""
    edges = root.edges
    if not edges:
        return []
    priors = np.maximum(
        np.array([e.prior for e in edges], dtype=np.float64), 1e-12)
    logits = np.log(priors)
    visits = np.array([float(e.n_visits) for e in edges])
    qs = np.array([e.q_value for e in edges], dtype=np.float64)
    visited = visits > 0

    sum_visits = float(visits.sum())
    if visited.any():
        p_vis = priors[visited]
        weighted_q = float((p_vis * qs[visited]).sum() / p_vis.sum())
        v_mix = (root.value + sum_visits * weighted_q) / (1.0 + sum_visits)
    else:
        v_mix = root.value

    max_v = float(visits.max()) if len(visits) else 0.0
    completed_q = np.where(visited, qs, v_mix)
    t = logits + (config.gumbel_c_visit + max_v) \
        * config.gumbel_c_scale * completed_q
    t -= t.max()
    p = np.exp(t)
    p /= p.sum()
    return [
        (e.actor_idx, e.target_idx, e.weapon_idx, float(w), e.type_idx)
        for e, w in zip(edges, p) if w > 1e-9
    ]


def sample_action(
    root:        MCTSNode,
    temperature: float,
    rng:         Optional[np.random.Generator] = None,
) -> Optional[dict]:
    """Sample a root action with probability proportional to
    visit_count^(1/temperature). AlphaZero plays this way for the
    first ~30 moves of each SELF-PLAY game (tau=1) so the data
    distribution doesn't collapse onto one deterministic line of
    play, then switches to argmax (`best_action`). temperature <= 0
    is treated as argmax.

    Only visited edges participate (an unvisited edge carries no
    search evidence)."""
    if temperature <= 0.0:
        return best_action(root)
    if rng is None:
        rng = np.random.default_rng()
    visited = [e for e in root.edges if e.n_visits > 0]
    if not visited:
        return None
    counts = np.array([float(e.n_visits) for e in visited])
    # visits^(1/tau) in log space to dodge overflow at low tau.
    logits = np.log(counts) / temperature
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    idx = int(rng.choice(len(visited), p=probs))
    return visited[idx].action
