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
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from classes import GameState
from trainer import MCTSExperience, TrainStats
from tools.mcts import (
    MCTSConfig, mcts_search, extract_visit_counts, best_action,
)


log = logging.getLogger("mcts_policy")


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

    def __init__(self, base, mcts_config: Optional[MCTSConfig] = None):
        self._base = base
        self._mcts_config = mcts_config or MCTSConfig()
        # Per-game pending: game_label -> List[_PendingMCTSState].
        # Mid-game states accumulate here; `finalize_game` drains
        # them into `_queue` with their terminal z attached.
        self._pending: Dict[str, List[_PendingMCTSState]] = {}
        # Completed: flat list of MCTSExperience, drained by
        # `train_step` and handed to `trainer.step_mcts`.
        self._queue: List[MCTSExperience] = []
        self._lock = threading.Lock()
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
        root = mcts_search(
            sim,
            self._inference_model,
            self._inference_encoder,
            self._mcts_config,
        )
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
        visits = extract_visit_counts(root)
        side = game_state.global_info.current_side
        with self._lock:
            self._pending.setdefault(game_label, []).append(
                _PendingMCTSState(gs=game_state, visit_counts=visits,
                                  side=side)
            )
        return action

    def observe(self, game_label: str, side: int, reward: float,
                done: bool = False) -> None:
        """MCTS ignores per-step shaping rewards. We only act on
        the rollout loop's terminal-observe pulse to know when the
        game is over for *this side* — but the actual MCTSExperience
        building happens in `finalize_game`, which needs the winner
        (not just per-side done flags). Treat this as a no-op."""
        # Intentionally empty.

    def finalize_game(self, game_label: str, winner: int) -> None:
        """Drain `_pending[game_label]` into `_queue` with one
        `MCTSExperience` per recorded state. `winner == 0` means
        draw/timeout (z=0). Called from `_run_one_game` after the
        per-side terminal observes."""
        with self._lock:
            states = self._pending.pop(game_label, [])
        for s in states:
            if winner == 0:
                z = 0.0
            elif winner == s.side:
                z = +1.0
            else:
                z = -1.0
            exp = MCTSExperience(
                game_state=s.gs,
                visit_counts=s.visit_counts,
                z=z,
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

    def train_step(self) -> TrainStats:
        with self._lock:
            batch = self._queue
            self._queue = []
        if not batch:
            return TrainStats(
                policy_loss=0.0, value_loss=0.0, entropy=0.0,
                total_loss=0.0, n_trajectories=0, n_transitions=0,
                mean_return=0.0, grad_norm=0.0,
            )
        return self._base._trainer.step_mcts(batch)

    # ------------------------------------------------------------------
    # Checkpoint forwarding
    # ------------------------------------------------------------------

    def save_checkpoint(self, path) -> None:
        return self._base.save_checkpoint(path)

    def load_checkpoint(self, path, *, strict: bool = False) -> None:
        return self._base.load_checkpoint(path, strict=strict)
