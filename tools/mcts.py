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

from wesnoth_ai.action_sampler import (
    LegalActionPrior,
    enumerate_legal_actions_with_priors,
)
from wesnoth_ai.classes import GameState, state_key
from wesnoth_ai.encoder import GameStateEncoder
from wesnoth_ai.model import WesnothModel
from wesnoth_sim import WesnothSim
from draw_tiebreak import DrawTiebreakConfig, draw_tiebreak_z
from combat_outcomes import (
    enumerate_attack_outcomes, outcome_key_for_child,
)
from outcome_buckets import (
    event_class as _event_class,
    initial_buckets as _initial_buckets,
    propose_split as _propose_split,
    split as _bucket_split,
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

# Children-dict key for a NO-OP resample: a stochastic step that
# advanced the game by nothing observable, so state_key(child) ==
# state_key(parent). The canonical case is a recruit attempt on a
# fog-occupied castle hex: the sim adds the hex to the per-turn
# rejection set and returns WITHOUT applying anything (see
# wesnoth_sim.step's __retry_recruit__ branch), and the rejection
# set is deliberately EXCLUDED from state_key (it's per-turn
# observable history, not game state). Descending into such a child
# is catastrophic: the per-search transposition table maps the
# unchanged key back to the PARENT node, so child IS parent and the
# selection descent self-loops forever, forking a sim per iteration
# until the process OOMs (observed 2026-06-13: one game ran 3.5h
# then MemoryError'd in _select_one's path.append). Routing no-ops
# to a distinct pseudo-terminal sentinel (NOT shared via the
# transposition table) stops the loop, lets the edge accrue visits
# so PUCT stops over-selecting it, and backs up a neutral value.
_NOOP_KEY = "noop"

# Hard cap on selection-descent depth. No legitimate line of play in
# our games (max_turns ~24, low-hundreds of actions/turn worst case)
# approaches this; exceeding it means an undiscovered cycle, so we
# break and treat the current node as the leaf rather than hang.
# Defense-in-depth behind the _NOOP_KEY guard above.
_MAX_SELECT_DEPTH = 4096

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
    # Lc0-style moves-left utility weight (2026-07-05). 0.0 = OFF
    # (default; byte-identical search). When > 0, in-tree selection
    # adds  -moves_left_utility * Q(s,a) * M(s,a)  to each visited
    # edge's PUCT score, where M is the edge's backed-up mean
    # moves-left prediction in [0,1] (terminals back up 0). Effect:
    # from winning positions (Q>0) the search prefers lines that END
    # the game sooner; from losing positions it prefers dragging.
    # Motivated by the Tier-a verdict: 42/66 eval games died to the
    # action cap -- nothing priced time. Requires a checkpoint whose
    # model has the moves-left head (--mcts-moves-left); without M
    # data the term is inert. Tuning knob, start ~0.2 (term bounded
    # by the weight since |Q|<=1, M<=1).
    moves_left_utility: float = 0.0
    # Aux-head value bonus (2026-07-11): leaf value used by search
    # becomes clamp(v + aux_value_bonus * aux_pred, -1, 1), with
    # aux_pred = tanh-bounded predicted material margin in (-1, 1).
    # At 0.3 the bonus spans (-0.3, +0.3): material/village gains
    # become visible WITHIN the search horizon instead of only at
    # in-search terminals (never reached mid-game). Root cause it
    # fixes: the anatomy diagnostic showed ~zero village captures in
    # 100-turn games -- nothing in-horizon rewarded expansion.
    # 0.0 = off (legacy).
    aux_value_bonus:  float = 0.0
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

    # ----- Tier-2 adaptive outcome bucketing (default off) -----------
    # Group an attack edge's combat outcomes into buckets that share
    # ONE representative network forward, refined only where members'
    # values diverge (see tools/outcome_buckets + BACKLOG "Tier 2").
    # The first outcome sampled into a bucket is forwarded and becomes
    # the representative; later same-bucket outcomes COPY its edges +
    # value instead of forwarding (amortizes the dominant cost) while
    # still recursing from their own real state. Requires
    # `exact_outcome_enumeration` (operates on the DP distribution)
    # and the serial search path (batch_size == 1; the production
    # default). `bucket_z_sig` is the significance multiplier for the
    # split trigger; `bucket_v_min` / `bucket_min_half_visits` gate it
    # (see outcome_buckets.propose_split).
    outcome_buckets:        bool  = False
    bucket_v_min:           int   = 16
    bucket_z_sig:           float = 2.0
    bucket_min_half_visits: int   = 2

    # ----- Playout-cap randomization (KataGo, Wu 2019) ----------------
    # Decouple the moves that ADVANCE a self-play game (cheap) from the
    # moves that GENERATE training data (expensive full search). When
    # on, each decision is a "full" move with probability
    # `playout_cap_prob` -- full `n_simulations` AND its policy target
    # is recorded -- otherwise a "fast" move with only
    # `playout_cap_fast_sims` sims and NO recorded target. Most moves
    # are fast, so games/GPU-hour rises ~3-10x (KataGo) while targets
    # still come from full-strength searches. `playout_cap_fast_sims`
    # == 0 derives a default of max(1, n_simulations // 4). The value
    # target (terminal z) still attaches to every recorded full state.
    playout_cap_randomization: bool  = False
    playout_cap_prob:          float = 0.25
    playout_cap_fast_sims:     int   = 0


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
                 "m_sum",
                 "children",
                 "outcome_probs", "outcome_keys", "seen_mass",
                 "bucket_rep", "bucket_of")

    def __init__(self, lap: LegalActionPrior):
        self.action     = lap.action
        self.actor_idx  = lap.actor_idx
        self.type_idx   = lap.type_idx
        self.target_idx = lap.target_idx
        self.weapon_idx = lap.weapon_idx
        self.prior      = lap.prior
        self.n_visits   = 0
        self.w_value    = 0.0
        # Summed moves-left predictions backed up through this edge
        # (perspective-free: game length, not value -- no sign flip).
        # Sibling-relative comparisons are what the utility consumes,
        # so the constant per-ply offset is irrelevant (Lc0-style).
        self.m_sum      = 0.0
        self.children: Dict = {}
        # Exact-enumeration bookkeeping (attack edges only):
        # outcome_probs: _OUTCOMES_UNSET | None | OutcomeDistribution.
        # outcome_keys maps children state_key -> OutcomeKey;
        # seen_mass = total exact probability of observed children.
        self.outcome_probs = _OUTCOMES_UNSET
        self.outcome_keys: Dict = {}
        self.seen_mass: float = 0.0
        # Tier-2 outcome bucketing (only populated when
        # config.outcome_buckets; see tools/outcome_buckets):
        #   bucket_rep: id(Bucket) -> representative MCTSNode (the one
        #     outcome that forwarded; later same-bucket outcomes copy
        #     its edges/value).
        #   bucket_of:  OutcomeKey -> Bucket it currently belongs to
        #     (re-pointed when a bucket splits). None until lazily
        #     initialized from the DP distribution on first use.
        self.bucket_rep: Dict[int, "MCTSNode"] = {}
        self.bucket_of: Optional[Dict] = None

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
                 "cliffness", "value", "gumbel_action",
                 "moves_left",
                 "_bucket_copy_from")

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
        # Moves-left prediction (fraction of the turn budget still to
        # be played, in (0,1)) from the network's optional moves-left
        # head, stamped at expansion. None when the model lacks the
        # head or the node is terminal/unexpanded (terminals back up
        # M=0 directly -- game over IS zero moves left).
        self.moves_left: Optional[float] = None
        # Tier-2 bucketing: when set (to the bucket's representative
        # node), this node is a NON-representative member -- on
        # expansion it COPIES the representative's edges + value
        # instead of running its own network forward. None otherwise.
        self._bucket_copy_from: Optional["MCTSNode"] = None


def tree_depth_stats(root: "MCTSNode") -> Tuple[int, float, int]:
    """(max_depth, visit_weighted_depth, n_nodes) of a finished
    search tree; root depth = 0.

    `visit_weighted_depth` weights each non-root node's depth by its
    incoming-edge visits and divides by total incoming visits -- an
    "average lookahead" in ACTIONS (not turns: at ~7 actions per
    side-turn, depth 7 is ~one own side-turn of lookahead). Answers
    the standing question "how deep does MCTS actually search?"
    (user 2026-07-21) -- visit counts concentrate depth near the
    root, so max_depth alone flatters the search. Cost: one walk of
    <= n_sims nodes per finished search; only called from the
    per-decision diag hook."""
    max_depth = 0
    w_sum = 0.0
    v_sum = 0
    n_nodes = 1
    stack = [(root, 0)]
    seen = {id(root)}
    while stack:
        node, depth = stack.pop()
        for e in node.edges:
            for child in e.children.values():
                if id(child) in seen:      # TT can alias subtrees
                    continue
                seen.add(id(child))
                d = depth + 1
                n_nodes += 1
                max_depth = max(max_depth, d)
                # Edge visits aggregate across outcomes; apportion
                # to children by their own subtree visits + 1 (the
                # +1 counts the visit that created the node).
                cv = child._total_visits + 1
                w_sum += d * cv
                v_sum += cv
                stack.append((child, d))
    return max_depth, (w_sum / v_sum if v_sum else 0.0), n_nodes


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


def _aux_adjusted(v: float, output, aux_value_bonus: float) -> float:
    """clamp(v + bonus * aux_pred). No-op when the knob is 0 or the
    model has no aux head."""
    if not aux_value_bonus or output.aux_score is None:
        return v
    aux = float(output.aux_score.reshape(()).item())
    adj = v + aux_value_bonus * aux
    return max(-1.0, min(1.0, adj))


def _expand(
    node: MCTSNode,
    model: WesnothModel,
    encoder: GameStateEncoder,
    tiebreak: Optional[DrawTiebreakConfig] = None,
    *,
    decision_step: int = 0,
    aux_value_bonus: float = 0.0,
) -> float:
    """Forward the model on `node.sim.gs`, build edges from the
    legal-action priors, return v(s) from node.side's perspective.

    `decision_step` drives the combat-oracle anneal in the priors (see
    `combat_alphas_at`); it MUST be the same value the trainer's
    distillation loss later uses for this state, or the search priors
    and the loss reference logits would disagree. Default 0 =
    full-strength oracle.

    No-op + returns terminal value if the node is already terminal
    (saves a model forward)."""
    if node.is_terminal:
        node.expanded = True
        return _terminal_value(node.sim, node.side, tiebreak)
    with torch.no_grad():
        encoded = encoder.encode(node.sim.gs)
        output = model(encoded)
        # Sampler-on-CPU split: one bulk D2H here instead of dozens of
        # per-actor syncs inside the enumeration (no-op on CPU).
        encoded, output = _leaf_to_cpu(encoded, output)
        priors = enumerate_legal_actions_with_priors(
            encoded, output, node.sim.gs, decision_step=decision_step)
        v = _aux_adjusted(float(output.value.squeeze().item()),
                          output, aux_value_bonus)
        node.cliffness = float(output.cliffness.squeeze().item())
        if output.moves_left is not None:
            node.moves_left = float(output.moves_left.squeeze().item())
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
    moves_left_utility: float = 0.0,
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
        # Lc0-style moves-left utility (see MCTSConfig): winning lines
        # (q > 0) are nudged toward LOW expected remaining moves,
        # losing lines toward HIGH. Only visited edges carry M data;
        # the term vanishes at moves_left_utility=0 (default) and for
        # M-less edges, keeping legacy behavior byte-identical.
        if moves_left_utility > 0.0 and edge.n_visits > 0 \
                and edge.m_sum > 0.0:
            u -= moves_left_utility * q * (edge.m_sum / edge.n_visits)
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
    moves_left_utility: float = 0.0,
    forced_first_edge:  Optional[MCTSEdge] = None,
    chance_nodes:       bool = False,
    sample_rng:         Optional[np.random.Generator] = None,
    exact_outcomes:     bool = False,
    outcome_buckets:    bool = False,
) -> Tuple[MCTSNode, List[Tuple[MCTSNode, MCTSEdge]]]:
    """Walk down from root picking PUCT-best edges. Lazy-create
    child nodes by forking the parent sim and applying the edge's
    action. Apply virtual loss to each selected edge so the next
    parallel selection in the same batch is biased away from this
    path. Returns (leaf, path, member_path): `path` is the list of
    (parent_node, edge_taken) pairs (backup walks it in reverse);
    `member_path` is the parallel list of the OutcomeKey descended
    into at each bucketed chance edge (else None), which backup uses
    to attribute values to member ground stats (Tier-2 bucketing).

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
    # Parallel to `path`: the OutcomeKey descended into at each step
    # when that edge is a bucketed chance edge, else None. _backup
    # uses it to attribute backed-up values to member ground stats
    # (Tier-2 stage 2). Always built; harmless (all None) when
    # bucketing is off.
    member_path: List[Optional[tuple]] = []
    node = root
    while node.expanded and not node.is_terminal:
        if len(path) >= _MAX_SELECT_DEPTH:
            # Cycle backstop (see _MAX_SELECT_DEPTH): treat the
            # current node as the leaf rather than descend forever.
            log.warning(
                "mcts: selection depth hit %d; breaking (cycle?). "
                "Treating current node as leaf.", _MAX_SELECT_DEPTH)
            break
        step_okey: Optional[tuple] = None   # set iff bucketed this step
        if forced_first_edge is not None:
            # Gumbel sequential halving pins the ROOT edge; interior
            # selection below stays PUCT.
            edge = forced_first_edge
            forced_first_edge = None
        else:
            fpu = root_fpu_reduction if node is root else fpu_reduction
            edge = _puct_select(node, c_puct, fpu, moves_left_utility)

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
                    # Exclude BOTH sentinel keys: _STEP_ERROR_KEY and
                    # _NOOP_KEY are inserted into `children` but never into
                    # `outcome_keys`, so indexing dist.probs[outcome_keys[k]]
                    # below would KeyError. A _NOOP_KEY child can't currently
                    # coexist with an attack edge (noop arises only on recruit
                    # rejections / unchanged-state_key), but filter it here so
                    # this stays correct if that invariant ever changes.
                    keys = [k for k in edge.children
                            if k not in (_STEP_ERROR_KEY, _NOOP_KEY)]
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
                noop = (not step_error
                        and (getattr(child_sim, "last_step_rejected", False)
                             or state_key(child_sim.gs)
                             == state_key(node.sim.gs)))
                if step_error or noop:
                    # state_key on a malformed post-error state could
                    # itself raise; park all error outcomes under one
                    # sentinel child. A NO-OP resample (e.g. a recruit
                    # rejected on a fog-occupied hex) is routed the
                    # same way: its unchanged state_key would alias
                    # the parent through the transposition table and
                    # self-loop the descent forever (see _NOOP_KEY).
                    # Mark the sentinel sim terminal so the descent
                    # stops here and a neutral value backs up.
                    sentinel = _STEP_ERROR_KEY if step_error else _NOOP_KEY
                    child = edge.children.get(sentinel)
                    if child is None:
                        if noop:
                            child_sim.done = True
                            child_sim.winner = 0
                            child_sim.ended_by = "noop_resample"
                        child = MCTSNode(child_sim)
                        edge.children[sentinel] = child
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
                            # Expected graceful fallback, NOT an error:
                            # the DP occasionally misses a rare-tail
                            # outcome on complex mid-game combats
                            # (pre-existing slow/poison, backstab /
                            # leadership / illuminate shifts, swarm /
                            # berserk). We distrust the whole
                            # distribution for this edge and sample
                            # instead -- always correct, just slower.
                            # debug-level because it fired ~1866x in a
                            # 44-iter overnight run and flooded the log
                            # (2026-06-13); the loss is enumeration
                            # speedup on that edge, not correctness.
                            # BACKLOG (Tier-2 outcome modeling) tracks
                            # closing the gap.
                            log.debug(
                                "mcts: sampled combat outcome absent "
                                "from exact support; disabling "
                                "enumeration for this edge")
                            edge.outcome_probs = None
                        else:
                            edge.outcome_keys[key] = okey
                            edge.seen_mass += p_o
                    # --- Tier-2 bucketing. Group this outcome's child
                    # into its bucket: the first outcome forwarded in a
                    # bucket is the representative; later same-bucket
                    # outcomes copy its edges + value instead of running
                    # their own network forward (copy-at-expansion).
                    # Members still recurse from their own real sim
                    # (re-separation), keeping an independent value
                    # signal that the backup-time significance test
                    # (stage 2) uses to split buckets. `step_okey`
                    # threads the member to _backup for ground-stat
                    # attribution. Only active with a live DP dist.
                    if outcome_buckets and dist is not None:
                        okey = edge.outcome_keys.get(key)
                        if okey is not None:
                            if edge.bucket_of is None:
                                # Lazy init: one bucket per event-class
                                # (coarsest), refined by splits later.
                                edge.bucket_of = {
                                    k: b
                                    for b in _initial_buckets(dist.probs)
                                    for k in b.members
                                }
                            b = edge.bucket_of.get(okey)
                            if b is not None:
                                step_okey = okey
                                rep = edge.bucket_rep.get(id(b))
                                if rep is None or rep is child:
                                    edge.bucket_rep[id(b)] = child
                                elif (not child.expanded
                                      and child._bucket_copy_from is None):
                                    child._bucket_copy_from = rep
        # Apply virtual loss BEFORE descending so the next
        # parallel selection in the same batch sees an edge that
        # currently looks bad (q drops, PUCT score drops). Real
        # backup undoes this and adds the actual update.
        if virtual_loss > 0:
            edge.n_visits += virtual_loss
            edge.w_value -= virtual_loss
            node._total_visits += virtual_loss
        path.append((node, edge))
        member_path.append(step_okey)
        node = child
    return node, path, member_path


# Variance of the prior distribution over outcomes, used as
# σ²_prior in the Bayesian-precision bootstrap shrinkage. We
# treat the prior over v as uniform on [V_MIN, V_MAX] = [-1, +1],
# whose variance is (V_MAX - V_MIN)² / 12 = 4/12 = 1/3.
# Choice of "uniform" prior matches what a freshly-initialized
# C51 head would output (uniform logits → uniform softmax →
# uniform distribution over atoms): the value-head's max-entropy
# state is the natural "I know nothing" prior.
_BOOTSTRAP_PRIOR_VAR = 1.0 / 3.0


def _record_and_maybe_split(edge, okey, mv, config) -> None:
    """Tier-2 stage 2: attribute a backed-up member value `mv` (in the
    edge's parent-perspective frame, consistent across the bucket) to
    its ground stat, then split the bucket if the significance test
    fires. On split, re-point members to the two sub-buckets and
    retire the old representative (sub-buckets re-elect one lazily on
    their next sampled member). Retained member ground stats carry
    into the sub-buckets, so the split is warm and unbiased."""
    if edge.bucket_of is None:
        return
    b = edge.bucket_of.get(okey)
    if b is None:
        return
    b.record(okey, mv)
    out = _propose_split(b, config.bucket_v_min, config.bucket_z_sig,
                         config.bucket_min_half_visits)
    if out is None:
        return
    axis, thr, _gap = out
    lo, hi = _bucket_split(b, axis, thr)
    for k in lo.members:
        edge.bucket_of[k] = lo
    for k in hi.members:
        edge.bucket_of[k] = hi
    edge.bucket_rep.pop(id(b), None)


def _backup(
    path:                List[Tuple[MCTSNode, MCTSEdge]],
    v:                   float,
    leaf_side:           int,
    virtual_loss:        float,
    leaf_cliffness:      float = 0.0,
    bootstrap_alpha:     float = 0.0,
    leaf_is_terminal:    bool  = False,
    member_path:         Optional[List] = None,
    config:              Optional["MCTSConfig"] = None,
    leaf_moves_left:     Optional[float] = None,
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
    bucketing = member_path is not None and config is not None
    for i in range(len(path) - 1, -1, -1):
        parent, edge = path[i]
        if virtual_loss > 0:
            edge.n_visits -= virtual_loss
            edge.w_value += virtual_loss
            parent._total_visits -= virtual_loss
        edge.n_visits += 1
        parent._total_visits += 1
        contrib = v_eff if parent.side == leaf_side else -v_eff
        edge.w_value += contrib
        # Moves-left backup (perspective-free -- game length, not
        # value, so no sign flip). Terminal leaves back up 0 ("game
        # over IS zero moves left"); leaves without the head back up
        # nothing, leaving m_sum's mean over the visits that DID
        # carry M (consumed sibling-relative by _puct_select).
        if leaf_moves_left is not None:
            edge.m_sum += leaf_moves_left
        # Tier-2 stage 2: attribute to the traversed member's ground
        # stat (same parent-perspective frame as edge.w_value, so it's
        # consistent across the bucket) and split on significance.
        if bucketing:
            okey = member_path[i]
            if okey is not None:
                _record_and_maybe_split(edge, okey, contrib, config)


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


def _leaf_to_cpu(encoded, output):
    """Forward-on-GPU / sampler-on-CPU split (gpu_perf_patches.md #1).

    `enumerate_legal_actions_with_priors` does dozens of per-actor
    `.item()`/`.tolist()` reads; on CUDA each one is a serializing
    D2H sync, × n_sims leaves per move (measured 2026-07-02 on a T4:
    enumerate = 26% of the rollout even before counting the syncs
    buried in `forward`'s 41%). Move the model output (every tensor
    field, via the actor pool's proven `move_model_output` seam) and
    the two encoded tensors the sampler consumes as values
    (`unit_is_ours`, `recruit_is_ours` — `build_light_encoded`
    documents that the sampler reads the token tensors only for
    `.size()`) to CPU in ONE pass, then enumerate on host tensors.

    CPU inputs pass through untouched, so the CPU path — and the
    whole test suite — is byte-identical. The CUDA-only behavior is
    validated by the Kaggle GPU smoke + re-profile (the device
    mismatch this could cause cannot manifest on CPU)."""
    if output.actor_logits.device.type == "cpu":
        return encoded, output
    from dataclasses import replace as _dc_replace
    from tools.inference_seam import move_model_output
    cpu = torch.device("cpu")
    output_cpu = move_model_output(output, cpu)
    encoded_cpu = _dc_replace(
        encoded,
        unit_is_ours=encoded.unit_is_ours.to(cpu),
        recruit_is_ours=encoded.recruit_is_ours.to(cpu),
    )
    return encoded_cpu, output_cpu


def _populate_leaf(
    leaf:    MCTSNode,
    encoded,
    output,
    *,
    decision_step: int = 0,
    value:     Optional[float] = None,
    cliffness: Optional[float] = None,
    aux_value_bonus: float = 0.0,
) -> float:
    """Build the leaf's edges from the model's enumerated priors and
    return v from leaf.side's perspective. Mirrors the post-forward
    half of `_expand`. Also records the network's cliffness on the
    leaf node so the backup phase can downweight unreliable
    bootstraps. `decision_step` drives the combat-oracle anneal (must
    match the loss's value for this state; see `_expand`).

    `value`/`cliffness`: pre-read scalars from a batched D2H transfer
    (B2, see `_run_sim_batch`); when None, read them from `output`
    (identical values -- the batch read is a pure transfer coalesce)."""
    encoded, output = _leaf_to_cpu(encoded, output)
    leaf.cliffness = (float(output.cliffness.squeeze().item())
                      if cliffness is None else float(cliffness))
    if output.moves_left is not None:
        leaf.moves_left = float(output.moves_left.squeeze().item())
    priors = enumerate_legal_actions_with_priors(
        encoded, output, leaf.sim.gs, decision_step=decision_step)
    if not priors:
        leaf.is_terminal = True
        leaf.expanded = True
        return 0.0
    priors.sort(key=lambda p: -p.prior)
    leaf.edges = [MCTSEdge(p) for p in priors]
    leaf.expanded = True
    # `value`, when provided (B2 batched read), is the RAW network
    # value -- the aux adjustment is applied here either way so both
    # paths agree.
    raw_v = (float(output.value.squeeze().item())
             if value is None else float(value))
    leaf.value = _aux_adjusted(raw_v, output, aux_value_bonus)
    return leaf.value


def _copy_expansion(leaf: MCTSNode, rep: MCTSNode) -> float:
    """Tier-2 copy-at-expansion: expand a non-representative bucket
    member by COPYING the representative's edges + value/cliffness
    instead of running a network forward. Valid because all members
    of an event-class share the same legal actions (HP doesn't change
    legality); the shared priors are an approximation that the
    significance-aware split corrects when member values diverge. Each
    copied edge is a FRESH MCTSEdge (independent stats) built from the
    representative edge's LegalActionPrior, so the member's subtree
    still re-separates from its own real sim. Returns the value to
    back up (the representative's V)."""
    leaf.edges = [
        MCTSEdge(LegalActionPrior(
            action=e.action, prior=e.prior, actor_idx=e.actor_idx,
            type_idx=e.type_idx, target_idx=e.target_idx,
            weapon_idx=e.weapon_idx))
        for e in rep.edges
    ]
    leaf.value = rep.value
    leaf.cliffness = rep.cliffness
    leaf.moves_left = rep.moves_left
    leaf.expanded = True
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
    decision_step:     int = 0,
) -> None:
    """One serial simulation: select (optionally pinned through a
    given root edge), evaluate/expand the leaf, back up. Used by the
    Gumbel root procedure; the classic loop keeps its own batched
    variant. `decision_step` drives the combat-oracle anneal in leaf
    priors (see `_expand`)."""
    leaf, path, member_path = _select_one(
        root, config.c_puct, 0.0,
        transpositions=transpositions, stats=tt_stats,
        fpu_reduction=config.fpu_reduction,
        root_fpu_reduction=config.root_fpu_reduction,
        moves_left_utility=config.moves_left_utility,
        forced_first_edge=forced_first_edge,
        chance_nodes=config.chance_nodes,
        sample_rng=sample_rng,
        exact_outcomes=config.exact_outcome_enumeration,
        outcome_buckets=config.outcome_buckets,
    )
    # Member-attribution args: only thread them when bucketing is on
    # (no-op otherwise, and keeps _backup's hot loop branch-free).
    _bp = ({"member_path": member_path, "config": config}
           if config.outcome_buckets else {})
    if leaf.is_terminal:
        v = _terminal_value(leaf.sim, leaf.side, config.draw_tiebreak)
        _backup(path, v, leaf.side, 0.0, leaf_is_terminal=True,
                leaf_moves_left=0.0, **_bp)
        return
    if leaf.expanded:
        # Depth-cap break (see _MAX_SELECT_DEPTH) returned an already-
        # expanded interior node. Re-running _populate_leaf would
        # REPLACE leaf.edges and discard its accumulated stats, so
        # back up its stored value estimate instead.
        _backup(
            path, leaf.value, leaf.side, 0.0,
            leaf_cliffness=leaf.cliffness,
            bootstrap_alpha=config.cliffness_bootstrap_alpha,
            leaf_moves_left=leaf.moves_left, **_bp,
        )
        return
    # Tier-2 copy-at-expansion: a non-representative bucket member
    # copies its representative's edges/value instead of forwarding --
    # but only once the representative itself is expanded with real
    # edges; otherwise fall back to a normal forward (no saving this
    # time, still correct).
    rep = leaf._bucket_copy_from
    if (rep is not None and rep.expanded and not rep.is_terminal
            and rep.edges):
        v = _copy_expansion(leaf, rep)
    else:
        with torch.no_grad():
            encoded = encoder.encode(leaf.sim.gs)
            output = model(encoded)
        v = _populate_leaf(leaf, encoded, output,
                           aux_value_bonus=config.aux_value_bonus,
                           decision_step=decision_step)
    _backup(
        path, v, leaf.side, 0.0,
        leaf_cliffness=leaf.cliffness,
        bootstrap_alpha=config.cliffness_bootstrap_alpha,
        leaf_is_terminal=leaf.is_terminal,
        leaf_moves_left=(0.0 if leaf.is_terminal else leaf.moves_left),
        **_bp,
    )


def _run_sim_batch(
    root:           MCTSNode,
    model:          WesnothModel,
    encoder:        GameStateEncoder,
    config:         MCTSConfig,
    transpositions: Optional[Dict[int, MCTSNode]],
    tt_stats:       Dict[str, int],
    rng:            Optional[np.random.Generator],
    v_loss:         float,
    decision_step:  int,
    *,
    forced_edges:   Optional[List[MCTSEdge]] = None,
    n:              int = 0,
) -> int:
    """Run a batch of simulations sharing ONE `model.forward_batch` over
    their leaves (virtual-loss parallelization). Shared by the classic
    PUCT-at-root loop and the batched Gumbel phase so there is a single
    tested select -> batch-forward -> backup code path.

    `forced_edges`: per-sim pinned root edges (Gumbel sequential halving
    descends each sim through a fixed candidate). When None, run `n`
    classic PUCT-at-root selections. Returns the number of sims
    completed (terminal / already-expanded leaves are backed up inline;
    non-terminal unexpanded leaves share the batched forward).

    Does NOT support outcome bucketing (member_path attribution) -- the
    callers gate the batched path on `not config.outcome_buckets` and
    fall back to the serial `_run_one_sim` when buckets are on.
    """
    specs: List[Optional[MCTSEdge]] = (
        list(forced_edges) if forced_edges is not None else [None] * n)
    root_fpu = (config.root_fpu_reduction
                if (config.add_root_noise and config.fpu_reduction is not None)
                else config.fpu_reduction)
    pending: List[Tuple[MCTSNode, List[Tuple[MCTSNode, MCTSEdge]]]] = []
    completed = 0
    # ----- Phase 1: select leaves (virtual loss diversifies them) ----
    for fe in specs:
        leaf, path, _member = _select_one(
            root, config.c_puct, v_loss,
            transpositions=transpositions, stats=tt_stats,
            fpu_reduction=config.fpu_reduction,
            root_fpu_reduction=root_fpu,
            moves_left_utility=config.moves_left_utility,
            forced_first_edge=fe,
            chance_nodes=config.chance_nodes,
            sample_rng=rng,
            exact_outcomes=config.exact_outcome_enumeration,
        )
        if leaf.is_terminal:
            v = _terminal_value(leaf.sim, leaf.side, config.draw_tiebreak)
            _backup(path, v, leaf.side, v_loss,
                    leaf_cliffness=0.0, bootstrap_alpha=0.0,
                    leaf_is_terminal=True, leaf_moves_left=0.0)
            completed += 1
        elif leaf.expanded:
            # Depth-cap break: already-expanded interior node. Back up
            # its stored value rather than re-populating (which would
            # wipe its edge stats).
            _backup(path, leaf.value, leaf.side, v_loss,
                    leaf_cliffness=leaf.cliffness,
                    bootstrap_alpha=config.cliffness_bootstrap_alpha,
                    leaf_moves_left=leaf.moves_left)
            completed += 1
        else:
            pending.append((leaf, path))
    # ----- Phase 2: one batched forward over the unique leaves --------
    if pending:
        unique_leaves: Dict[int, MCTSNode] = {}
        for leaf, _ in pending:
            unique_leaves.setdefault(id(leaf), leaf)
        unique_list = list(unique_leaves.values())
        with torch.no_grad():
            encoded_list = [encoder.encode(l.sim.gs) for l in unique_list]
            outputs = model.forward_batch(encoded_list)
        # B2 (gpu_perf_patches.md #2): read every leaf's scalar value
        # + cliffness in ONE batched D2H transfer instead of 2
        # serializing syncs per leaf. Values are identical to the
        # per-leaf `.item()` reads; CPU path skips the coalesce (the
        # reads are already host ops there).
        vals: Optional[List[float]] = None
        cliffs: Optional[List[float]] = None
        if outputs and outputs[0].value.device.type != "cpu":
            with torch.no_grad():
                packed = torch.stack([
                    torch.stack((o.value.reshape(()),
                                 o.cliffness.reshape(())))
                    for o in outputs
                ]).cpu()
            vals = packed[:, 0].tolist()
            cliffs = packed[:, 1].tolist()
        leaf_values: Dict[int, float] = {}
        for i, (leaf, encoded, output) in enumerate(
                zip(unique_list, encoded_list, outputs)):
            leaf_values[id(leaf)] = _populate_leaf(
                leaf, encoded, output, decision_step=decision_step,
                value=None if vals is None else vals[i],
                aux_value_bonus=config.aux_value_bonus,
                cliffness=None if cliffs is None else cliffs[i])
        for leaf, path in pending:
            _backup(path, leaf_values[id(leaf)], leaf.side, v_loss,
                    leaf_cliffness=leaf.cliffness,
                    bootstrap_alpha=config.cliffness_bootstrap_alpha,
                    leaf_is_terminal=leaf.is_terminal,
                    leaf_moves_left=(0.0 if leaf.is_terminal
                                     else leaf.moves_left))
            completed += 1
    return completed


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
    decision_step:  int = 0,
) -> int:
    """Gumbel-Top-k candidates + sequential halving at the root.
    Sets `root.gumbel_action` to argmax(g + logits + sigma(q̂)) over
    the final survivors -- in expectation a policy improvement over
    the raw prior (Danihelka et al. 2022, thm. on one-step
    improvement). `decision_step` drives the combat-oracle anneal in
    leaf priors (see `_expand`)."""
    import time as _time
    edges = root.edges
    if not edges:
        root.gumbel_action = None
        return 0
    logits = np.log(np.maximum(
        np.array([e.prior for e in edges], dtype=np.float64), 1e-12))
    g = rng.gumbel(size=len(edges))
    base = g + logits

    m = max(1, min(config.gumbel_m, len(edges)))
    cands: List[int] = list(np.argsort(-base)[:m])

    def _score(ci: int, max_v: float) -> float:
        return base[ci] + _gumbel_sigma(edges[ci].q_value, max_v, config)

    # Batched leaf evaluation: when batch_size > 1 (GPU), evaluate each
    # phase's sims through one model.forward_batch with virtual loss
    # instead of B=1-per-sim. The sequential-halving SCHEDULE (num_phases,
    # sims_per, candidate reduction) is identical either way -- only the
    # forward grouping changes -- so the total sim count matches. Gated
    # off when outcome bucketing is on (the batched path doesn't carry
    # member-path attribution); falls back to serial _run_one_sim.
    use_batch = config.batch_size > 1 and not config.outcome_buckets
    B = max(1, int(config.batch_size))
    V_LOSS = float(config.virtual_loss) if use_batch else 0.0

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
        # Flat list of this phase's sims, each pinned through a candidate
        # root edge, capped by the remaining global budget. Same order
        # (candidate-major) the serial loop used.
        forced: List[MCTSEdge] = []
        for ci in cands:
            for _ in range(sims_per):
                if sims_done + len(forced) >= n_sims:
                    break
                forced.append(edges[ci])
            if sims_done + len(forced) >= n_sims:
                break

        if use_batch:
            j = 0
            while j < len(forced):
                if deadline is not None and _time.perf_counter() > deadline:
                    log.info(f"mcts(gumbel): stopping at "
                             f"{sims_done}/{n_sims} (time budget hit)")
                    sims_done = n_sims
                    break
                chunk = forced[j:j + B]
                j += len(chunk)
                sims_done += _run_sim_batch(
                    root, model, encoder, config, transpositions, tt_stats,
                    rng, V_LOSS, decision_step, forced_edges=chunk)
        else:
            for fe in forced:
                if deadline is not None and _time.perf_counter() > deadline:
                    log.info(f"mcts(gumbel): stopping at "
                             f"{sims_done}/{n_sims} (time budget hit)")
                    sims_done = n_sims
                    break
                _run_one_sim(root, model, encoder, config,
                             transpositions, tt_stats,
                             forced_first_edge=fe,
                             sample_rng=rng,
                             decision_step=decision_step)
                sims_done += 1

        max_v = max(e.n_visits for e in edges)
        keep = max(1, math.ceil(len(cands) / 2))
        cands = sorted(cands, key=lambda ci: -_score(ci, max_v))[:keep]

    max_v = max(e.n_visits for e in edges)
    best_ci = max(cands, key=lambda ci: _score(ci, max_v))
    root.gumbel_action = edges[best_ci].action
    return sims_done


def mcts_search(
    sim:     WesnothSim,
    model:   WesnothModel,
    encoder: GameStateEncoder,
    config:  Optional[MCTSConfig] = None,
    *,
    rng:        Optional[np.random.Generator] = None,
    reuse_root: Optional[MCTSNode] = None,
    n_sims_override: Optional[int] = None,
    decision_step:   int = 0,
) -> MCTSNode:
    """Run MCTS from `sim`'s state. Returns the root node with
    populated visit counts on outgoing edges.

    `decision_step`: training-progress counter threaded into leaf-prior
    expansion to drive the combat-oracle anneal (`combat_alphas_at`).
    The policy passes its current step here AND records it on the stored
    MCTSExperience so the distillation loss rebuilds reference logits at
    the SAME alpha -- search priors and loss must agree. Default 0 =
    full-strength oracle.

    `n_sims_override`: when set, run exactly this many simulations,
    bypassing both `config.n_simulations` and the adaptive budget.
    Used by playout-cap randomization (KataGo) to run cheap "fast"
    searches on the majority of self-play moves.

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
        _expand(root, model, encoder, tiebreak=config.draw_tiebreak,
                aux_value_bonus=config.aux_value_bonus,
                decision_step=decision_step)
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
    if n_sims_override is not None:
        # Playout-cap "fast" move: a fixed cheap budget, no adaptive
        # scaling (the move won't be recorded as a training target, so
        # its search just needs to pick a reasonable action).
        n_sims = max(1, int(n_sims_override))
    else:
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
        _gumbel_done = _gumbel_root_search(
            root, model, encoder, config, transpositions, tt_stats,
            rng, n_sims, deadline, decision_step=decision_step)
        # Spill any UNSPENT budget into the classic PUCT loop below.
        # Sequential halving can under-consume: a single-candidate
        # root runs 0 sims (nothing to halve), and the per-phase
        # floor split can drop a few for odd candidate counts. The
        # spill keeps the "n_simulations sims total" contract (and
        # the tree-reuse visit accounting built on it) regardless of
        # edge count -- exposed 2026-07-17 when true-reachability
        # masks changed root edge counts. Gumbel's action CHOICE
        # (root.gumbel_action) is already set and unaffected.
        n_sims = max(0, n_sims - _gumbel_done)

    B = max(1, int(config.batch_size))
    V_LOSS = float(config.virtual_loss) if B > 1 else 0.0
    sims_done = 0

    while sims_done < n_sims:
        if deadline is not None and _time.perf_counter() > deadline:
            log.info(f"mcts: stopping at {sims_done}/{n_sims} "
                     f"(time budget hit)")
            break

        # Run up to B sims sharing one batched forward (B=1 => serial).
        # Outcome bucketing is unsupported on the batched classic root
        # (v1 supports it only on the serial Gumbel path); the
        # outcome_buckets+classic-root combo is warned about at the CLI.
        n_this_batch = min(B, n_sims - sims_done)
        sims_done += _run_sim_batch(
            root, model, encoder, config, transpositions, tt_stats,
            rng, V_LOSS, decision_step, n=n_this_batch)

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
