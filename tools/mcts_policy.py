"""MCTS+visit-count distillation policy wrapper.

Wraps a `TransformerPolicy` so that the rollout loop in
`tools/sim_self_play.py` can opt into AlphaZero-style training
without touching the loop's interface beyond the new `sim=` kwarg
on `select_action`. Same five-method duck type as
`TransformerPolicy`: `select_action`, `observe`, `finalize_game`,
`train_step`, `save_checkpoint` / `load_checkpoint`.

What MCTS mode changes vs REINFORCE:

  - **Action picking**: an MCTS search produces a visit-count
    distribution over legal actions; we sample (or argmax) from
    that instead of the raw policy logits.
  - **Per-step shaping reward**: ignored. AlphaZero distills the
    final game outcome (z = ±1 for win/loss, 0 for draws/timeouts)
    onto every visited state — no Bellman-style reward shaping.
    The user's `--reward-config` is silently unused in MCTS mode;
    the iter log still reports per-step shaping totals so REINFORCE
    runs and MCTS runs are easy to compare visually.
  - **Trainer step**: `trainer.step_mcts(experiences)` instead of
    `trainer.step(transitions)`. Cross-entropy against the
    visit-count distribution + value loss against z.

Threading: `select_action` and `observe` may be called from
worker threads in parallel rollouts (one game per thread). The
`_pending` and `_queue` accesses are guarded by `_lock` to mirror
TransformerPolicy's thread-safety contract.

Dependencies: tools.mcts, transformer_policy, trainer
Dependents:   tools.sim_self_play (when --mcts is passed)
"""

from __future__ import annotations

import logging
import random
import threading
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from classes import GameState, state_key
from trainer import MCTSExperience, TrainStats
from tools.draw_tiebreak import draw_tiebreak_z, material_margin
from tools.mcts import (
    MCTSConfig, mcts_search, extract_visit_counts, best_action,
    sample_action, extract_gumbel_policy_target,
)


log = logging.getLogger("mcts_policy")


@dataclass
class ReplayConfig:
    """Experience-replay + multi-epoch training for the MCTS path.

    Diagnosis (2026-06-15): the default one-gradient-step-per-fresh-
    batch-then-discard schedule is severely sample-inefficient. Overfit
    probes showed the value head needs ~80-100 gradient steps to
    converge on a batch, but live training gave it ONE step per
    (shifting, high-variance) batch -> the value head never left its
    ~uniform floor (val loss ~3.56 vs ln(51)=3.93), MCTS produced
    near-uniform visit targets, and the policy plateaued. A replay
    buffer + several minibatch updates per iteration is the AlphaZero-
    standard fix: it decouples gradient steps from data generation so
    the value head can actually converge.

    `enabled=False` reproduces the exact legacy one-pass behavior, so
    this is a safe, A/B-able default-off addition.
    """
    enabled:          bool = False
    capacity:         int  = 4000   # max experiences retained (memory!)
    updates_per_iter: int  = 8      # gradient steps per train_step
    minibatch:        int  = 128    # experiences per gradient step
    min_size:         int  = 512    # warm up (legacy one-pass) until this


@dataclass
class _PendingMCTSState:
    """One mid-game record awaiting a terminal z. We stash the
    `gs` reference (NOT a deepcopy — `_run_one_game` already passes
    in a deepcopy as `pre_state`), the MCTS visit counts emitted
    at this state, and the side-to-move so finalize_game can
    compute z = +1 / -1 / 0 from the perspective of that side."""
    gs:           GameState
    visit_counts: List[Tuple]
    side:         int


class MCTSPolicy:
    """Wraps a `TransformerPolicy` to do MCTS+distillation."""

    trainable = True
    # Optimization #6 (2026-06-14): MCTS distills terminal z, not
    # per-step shaping -- `observe` is a no-op. Signals the rollout
    # loop to skip the (expensive, Dijkstra-bearing) compute_delta +
    # reward_fn calls entirely.
    uses_step_rewards = False

    def __init__(self, base, mcts_config: Optional[MCTSConfig] = None,
                 replay_config: Optional[ReplayConfig] = None):
        self._base = base
        self._mcts_config = mcts_config or MCTSConfig()
        self._replay_config = replay_config or ReplayConfig()
        # Per-game pending: game_label -> List[_PendingMCTSState].
        # Mid-game states accumulate here; `finalize_game` drains
        # them into `_queue` with their terminal z attached.
        self._pending: Dict[str, List[_PendingMCTSState]] = {}
        # Completed: flat list of MCTSExperience, drained by
        # `train_step` and handed to `trainer.step_mcts`.
        self._queue: List[MCTSExperience] = []
        # Experience-replay buffer (only used when replay_config is
        # enabled). Bounded FIFO of the most recent experiences; each
        # holds a per-decision deepcopy of the game state (see
        # _PendingMCTSState), so retaining it across iterations is
        # safe -- no aliasing with the live sim.
        self._replay: "deque[MCTSExperience]" = deque(
            maxlen=self._replay_config.capacity)
        # Dedicated seeded RNG for minibatch sampling (separate from
        # the search RNG) so replay runs are reproducible given a seed.
        self._replay_rng = random.Random(0)
        self._lock = threading.Lock()
        # RNG for temperature sampling at the root (AlphaZero's
        # tau=1 phase). Unseeded like mcts_search's noise RNG --
        # self-play data generation wants diversity, not
        # reproducibility; tests construct their own Generator.
        self._rng = np.random.default_rng()
        # Tree-reuse stash: game_label -> (child_node, child_state_key)
        # for the action played at the previous decision. The next
        # select_action reuses the subtree iff the LIVE state's key
        # matches (deterministic actions match; combat RNG diverges
        # and rebuilds). See MCTSConfig.tree_reuse.
        self._reuse: Dict[str, Tuple] = {}
        # `_inference_*` already exist on base — under the lock used
        # by base's snapshot. We just borrow them. mcts_search runs
        # `torch.no_grad()` internally so concurrent gradient steps
        # via `train_step` won't corrupt these forwards.
        self._inference_model = base._inference_model
        self._inference_encoder = base._inference_encoder

    # ------------------------------------------------------------------
    # Policy duck-type
    # ------------------------------------------------------------------

    def select_action(
        self,
        game_state: GameState,
        *,
        game_label: str = "default",
        sim=None,
    ) -> Dict:
        if sim is None:
            raise RuntimeError(
                "MCTSPolicy.select_action requires `sim=` to fork "
                "the search tree. Update the rollout loop to pass "
                "the live sim through."
            )
        # Enforce the snapshot contract: `game_state` must be a
        # DEEPCOPY of the pre-action state, not the live `sim.gs`. We
        # stash `game_state` in `_pending` as the AlphaZero training
        # target; if it's the live object, subsequent `sim.step` calls
        # mutate it and the recorded action indices stop matching the
        # state's re-encoding at train time (actor-slot drift -> the
        # policy loss IndexErrors or trains on a corrupted target).
        # Every real rollout/eval caller already deepcopies (see
        # play_one_game). Fail loudly here so a future caller/test that
        # forgets can't silently corrupt training data.
        if game_state is sim.gs:
            raise ValueError(
                "MCTSPolicy.select_action was passed the LIVE sim.gs; it "
                "must be a deepcopy snapshot of the pre-action state "
                "(sim.step would otherwise mutate the recorded training "
                "target). See play_one_game's `copy.deepcopy(sim.gs)`."
            )
        reuse_root = None
        if self._mcts_config.tree_reuse:
            with self._lock:
                stash = self._reuse.pop(game_label, None)
            if stash is not None:
                # The stash maps outcome state_key -> searched child
                # node. Deterministic actions have one entry; combat
                # has one per sampled outcome -- if the live RNG
                # produced an outcome the search visited, its subtree
                # is reusable.
                cand = stash.get(state_key(sim.gs))
                if (cand is not None and cand.expanded
                        and not cand.is_terminal):
                    reuse_root = cand
        # Playout-cap randomization (KataGo): most moves are "fast"
        # (cheap search, NOT recorded as a training target); a random
        # fraction `playout_cap_prob` are "full" (full budget AND
        # recorded). Decoupling game-advancing moves from data-
        # generating moves yields ~3-10x more self-play games per
        # GPU-hour while targets still come from full-strength searches.
        cfg = self._mcts_config
        full_move = True
        n_override = None
        if cfg.playout_cap_randomization:
            full_move = bool(self._rng.random() < cfg.playout_cap_prob)
            if not full_move:
                n_override = (cfg.playout_cap_fast_sims
                              or max(1, cfg.n_simulations // 4))
        root = mcts_search(
            sim,
            self._inference_model,
            self._inference_encoder,
            self._mcts_config,
            reuse_root=reuse_root,
            n_sims_override=n_override,
        )
        if self._mcts_config.gumbel_root:
            # Gumbel mode: the search already chose
            # argmax(g + logits + sigma(q̂)) -- stochastic via the
            # Gumbel draws, provably an improved policy. No
            # temperature schedule needed.
            action = root.gumbel_action or best_action(root)
        else:
            # Classic AlphaZero exploration schedule: sample
            # proportional to visit counts for the first
            # `temperature_decisions` decisions of each game, argmax
            # afterwards. The decision index is exactly how many
            # states this game has already recorded.
            with self._lock:
                decision_idx = len(self._pending.get(game_label, ()))
            if decision_idx < self._mcts_config.temperature_decisions:
                action = sample_action(
                    root, self._mcts_config.temperature, self._rng)
            else:
                action = best_action(root)
        if action is None:
            # No legal action and no terminal — pathological state.
            # The base policy's select_action would have raised; we
            # raise too rather than let the rollout loop wedge.
            raise RuntimeError(
                f"MCTS produced no action at game_label={game_label!r}; "
                f"root.is_terminal={root.is_terminal}, "
                f"n_edges={len(root.edges)}"
            )
        # Record a training target ONLY for full-budget moves. Fast
        # moves advance the game but generate no policy target
        # (playout-cap randomization); their search ran a cheap budget
        # purely to pick a reasonable action.
        if full_move:
            if self._mcts_config.gumbel_root:
                # Completed-Q target: every legal action gets a weight
                # (search Q if visited, mixed value if not) -- denser
                # signal than raw visit counts at small sim budgets.
                visits = extract_gumbel_policy_target(
                    root, self._mcts_config)
            else:
                visits = extract_visit_counts(root)
            side = game_state.global_info.current_side
            with self._lock:
                self._pending.setdefault(game_label, []).append(
                    _PendingMCTSState(gs=game_state, visit_counts=visits,
                                      side=side)
                )
        # Stash the played edge's outcome children for
        # state-key-checked reuse at the next decision. Action dicts
        # are returned by identity from the edge, so `is` finds the
        # edge; `==` is the fallback for wrappers that copy.
        if self._mcts_config.tree_reuse:
            edge = next(
                (e for e in root.edges
                 if e.action is action or e.action == action), None)
            if edge is not None and edge.children:
                stash = {k: n for k, n in edge.children.items()
                         if isinstance(k, int)}   # skip error sentinel
                if stash:
                    with self._lock:
                        self._reuse[game_label] = stash
        return action

    def observe(self, game_label: str, side: int, reward: float,
                done: bool = False) -> None:
        """MCTS ignores per-step shaping rewards. We only act on
        the rollout loop's terminal-observe pulse to know when the
        game is over for *this side* — but the actual MCTSExperience
        building happens in `finalize_game`, which needs the winner
        (not just per-side done flags). Treat this as a no-op."""
        # Intentionally empty.

    def finalize_game(self, game_label: str, winner: int,
                      final_gs: Optional[GameState] = None) -> None:
        """Drain `_pending[game_label]` into `_queue` with one
        `MCTSExperience` per recorded state. `winner == 0` means
        draw/timeout: z is 0, or the material tiebreaker score of
        the FINAL state when `MCTSConfig.draw_tiebreak` is set --
        the same function the search applies at turn-cap terminals,
        so targets and search horizon agree. `final_gs` is the
        game's actual end state (passed by the rollout loop); if
        absent we fall back to the last recorded pre-action state,
        which trails the true final state by one action."""
        with self._lock:
            states = self._pending.pop(game_label, [])
            self._reuse.pop(game_label, None)
        tiebreak = self._mcts_config.draw_tiebreak
        if winner == 0 and tiebreak is not None and final_gs is None \
                and states:
            final_gs = states[-1].gs
            log.debug(
                f"finalize_game({game_label!r}): no final_gs passed; "
                f"tiebreak scored on the last pre-action state")
        # Auxiliary margin target (KataGo §3.5): the final MATERIAL
        # margin from each state's side, computed from the game's end
        # state for EVERY outcome (denser than z). Needs the tiebreak
        # weights + a final state; falls back to the last recorded
        # state if the rollout didn't pass `final_gs`. `None` when no
        # tiebreak config is set (trainer then skips the aux loss).
        aux_gs = final_gs if final_gs is not None else (
            states[-1].gs if states else None)
        for s in states:
            if winner == 0:
                if tiebreak is not None and final_gs is not None:
                    z = draw_tiebreak_z(final_gs, s.side, tiebreak)
                else:
                    z = 0.0
            elif winner == s.side:
                z = +1.0
            else:
                z = -1.0
            aux = (material_margin(aux_gs, s.side, tiebreak)
                   if (tiebreak is not None and aux_gs is not None)
                   else None)
            exp = MCTSExperience(
                game_state=s.gs,
                visit_counts=s.visit_counts,
                z=z,
                aux_target=aux,
            )
            with self._lock:
                self._queue.append(exp)
        if states:
            log.debug(
                f"finalize_game({game_label!r}, winner={winner}): "
                f"queued {len(states)} MCTSExperiences"
            )

    def drop_pending(self, game_label: str) -> None:
        """Match TransformerPolicy's drop_pending API for error
        recovery: the rollout loop calls it when a game errors mid-
        rollout. We just drop pending without queuing."""
        with self._lock:
            self._pending.pop(game_label, None)
            self._reuse.pop(game_label, None)

    def train_step(self) -> TrainStats:
        with self._lock:
            batch = self._queue
            self._queue = []
        rc = self._replay_config
        if not rc.enabled:
            # Legacy: one gradient step over this iteration's fresh
            # experiences, then discard. Byte-for-byte the old path.
            if not batch:
                return TrainStats()
            return self._base._trainer.step_mcts(batch)

        # --- Experience replay + multi-epoch (default-off) -----------
        # Add the fresh experiences to the bounded buffer, then take
        # several minibatch gradient steps sampled from it. This gives
        # the value head the many steps it needs to converge (see
        # ReplayConfig) instead of one-and-discard.
        self._replay.extend(batch)
        if len(self._replay) < rc.min_size:
            # Warm-up: not enough history yet -- mirror legacy (one
            # pass over the fresh batch) so early iters aren't wasted.
            if not batch:
                return TrainStats()
            return self._base._trainer.step_mcts(batch)

        pool = list(self._replay)
        n = len(pool)
        mb = min(rc.minibatch, n)
        stats: List[TrainStats] = []
        for _ in range(rc.updates_per_iter):
            sample = self._replay_rng.sample(pool, mb)
            stats.append(self._base._trainer.step_mcts(sample))
        return self._combine_stats(stats, buffer_size=n)

    @staticmethod
    def _combine_stats(stats: List[TrainStats],
                       buffer_size: int) -> TrainStats:
        """Aggregate the multi-epoch minibatch steps into one TrainStats
        for the iter log: losses/grad-norm are MEANED over the updates
        (representative of the pass), transitions SUMMED (total samples
        touched), grad_norm uses the LAST step (post-update magnitude)."""
        if not stats:
            return TrainStats()
        k = len(stats)
        return TrainStats(
            policy_loss=sum(s.policy_loss for s in stats) / k,
            value_loss=sum(s.value_loss for s in stats) / k,
            entropy=sum(s.entropy for s in stats) / k,
            total_loss=sum(s.total_loss for s in stats) / k,
            grad_norm=stats[-1].grad_norm,
            mean_return=sum(s.mean_return for s in stats) / k,
            n_transitions=sum(s.n_transitions for s in stats),
            n_trajectories=buffer_size,
            aux_loss=sum(s.aux_loss for s in stats) / k,
        )

    # ------------------------------------------------------------------
    # Checkpoint forwarding
    # ------------------------------------------------------------------

    def save_checkpoint(self, path) -> None:
        return self._base.save_checkpoint(path)

    def load_checkpoint(self, path, *, strict: bool = False) -> None:
        return self._base.load_checkpoint(path, strict=strict)
