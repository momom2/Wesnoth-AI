"""Learned policy driven by a transformer over tokenized state.

Registers in ``policy.py``'s registry as ``--policy transformer``.

Lifecycle used by game_manager:
  1. ``select_action(game_state, game_label=L)`` — forward pass, samples
     an action, appends a `Transition` (with grad-tracked log_prob,
     value, entropy) to the policy's per-(game_label, side) pending list.
     Returns the action dict.
  2. ``observe(game_label, side, reward, done)`` — called after the
     next state arrives; adds `reward` to the last pending transition
     for that (game, side). If `done=True`, the trajectory is moved
     from pending into the training queue.
  3. ``train_step()`` — runs one REINFORCE+baseline gradient update
     over all queued trajectories; clears the queue. Called between
     batches of games by game_manager.

Checkpointing: save/load serialize model weights, encoder weights,
and the encoder's name→id dict (so recruit and unit tokens keep
addressing the same rows across sessions). An architecture dict is
stored so load can refuse an incompatible model.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from action_sampler import sample_action
from classes import GameState
from device import describe, select_device
from encoder import GameStateEncoder
from model import WesnothModel
from trainer import Trainer, TrainerConfig, Transition, TrainStats


class TransformerPolicy:
    """Learned policy — conforms to the ``Policy`` Protocol and exposes
    the optional training hooks ``observe`` and ``train_step``."""

    trainable = True

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        d_ff: int = 256,
        device=None,
        trainer_config: Optional[TrainerConfig] = None,
    ):
        # CPU by default for training: the DirectML backend on this
        # RX 6600 doesn't support the `scatter` op that
        # `torch.distributions.Categorical.log_prob` needs during
        # backward. Forward-only paths (a checkpoint evaluator, say)
        # can construct with `device=select_device()` to get GPU.
        # See memory/user_gpu_setup.md for the DML-op limitations.
        self._device = (
            device if device is not None else torch.device("cpu")
        )
        self._encoder = GameStateEncoder(d_model=d_model).to(self._device)
        self._model = WesnothModel(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
        ).to(self._device)
        self._trainer = Trainer(
            self._model,
            encoder=self._encoder,
            config=trainer_config,
            device=self._device,
        )
        self._logger = logging.getLogger("transformer_policy")

        # Pending transitions keyed by (game_label, side). Each entry
        # is a growing list; the most-recent Transition is the one
        # that `observe` will attach a reward to when called.
        self._pending: Dict[Tuple[str, int], List[Transition]] = {}

        # Trajectories completed (terminal reward received) and
        # awaiting the next train_step.
        self._queue: List[List[Transition]] = []

        # Architecture record (for checkpoint compat checks).
        self._arch = {
            "d_model":     d_model,
            "num_layers":  num_layers,
            "num_heads":   num_heads,
            "d_ff":        d_ff,
        }

        # Eval mode avoids dropout noise during rollout; Trainer.step
        # flips to train() for its backward pass.
        self._encoder.eval()
        self._model.eval()

    # ------------------------------------------------------------------
    # Policy Protocol
    # ------------------------------------------------------------------

    def select_action(
        self,
        game_state: GameState,
        *,
        game_label: str = "default",
    ) -> Dict:
        """Forward pass + sample + record pending Transition.

        Runs under torch.no_grad(): we only need the sampled indices
        at collection time, not the grad graph. The trainer re-forwards
        the stored game_state at training time. This is the key
        memory-saving move — see trainer.py's docstring.
        """
        with torch.no_grad():
            encoded = self._encoder.encode(game_state)
            output = self._model(encoded)
            sampled = sample_action(encoded, output, game_state)

        side = game_state.global_info.current_side
        key = (game_label, side)
        self._pending.setdefault(key, []).append(Transition(
            game_state=game_state,
            actor_idx=sampled.actor_idx,
            target_idx=sampled.target_idx,
            weapon_idx=sampled.weapon_idx,
        ))
        return sampled.action

    # ------------------------------------------------------------------
    # Trainable-policy hooks (game_manager uses these)
    # ------------------------------------------------------------------

    def observe(
        self,
        game_label: str,
        side: int,
        reward: float,
        done: bool = False,
    ) -> None:
        """Attach `reward` to the last pending transition for
        (game_label, side). If `done`, seal the trajectory and queue
        it for training."""
        key = (game_label, side)
        pending = self._pending.get(key)
        if not pending:
            return
        pending[-1].reward += reward
        if done:
            pending[-1].done = True
            self._queue.append(pending)
            del self._pending[key]

    def drop_pending(self, game_label: str) -> None:
        """Forget all pending transitions for a game (called when a
        game errors out mid-run and never reaches a terminal)."""
        keys = [k for k in self._pending if k[0] == game_label]
        for k in keys:
            del self._pending[k]

    def train_step(self) -> TrainStats:
        """Apply one gradient update over queued trajectories.

        Clears the queue regardless of outcome. No-op if empty.
        Logs duration — this is the most likely source of a perceived
        'freeze' in the rolling-pool game_manager, since train_step
        blocks the asyncio event loop while it runs. See trainer.py
        for the re-forward design that keeps this tractable.
        """
        if not self._queue:
            return TrainStats()
        n_transitions_total = sum(len(t) for t in self._queue)
        self._logger.info(
            f"train_step starting: {len(self._queue)} trajectories, "
            f"{n_transitions_total} transitions pending"
        )
        t0 = time.perf_counter()
        self._model.train()
        self._encoder.train()
        try:
            stats = self._trainer.step(self._queue)
        finally:
            self._model.eval()
            self._encoder.eval()
            self._queue.clear()
        dt = time.perf_counter() - t0
        self._logger.info(
            "train_step done in %.2fs: "
            "%d trajectories / %d transitions "
            "(of %d queued, capped by trainer) "
            "total_loss=%.4f policy=%.4f value=%.4f entropy=%.4f "
            "mean_return=%.4f grad_norm=%.4f",
            dt, stats.n_trajectories, stats.n_transitions, n_transitions_total,
            stats.total_loss, stats.policy_loss, stats.value_loss,
            stats.entropy, stats.mean_return, stats.grad_norm,
        )
        return stats

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "arch":            self._arch,
                "model_state":     self._model.state_dict(),
                "encoder_state":   self._encoder.state_dict(),
                "unit_type_to_id": dict(self._encoder.unit_type_to_id),
                "optimizer_state": self._trainer.optimizer.state_dict(),
            },
            path,
        )
        self._logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path) -> None:
        path = Path(path)
        ckpt = torch.load(path, map_location=self._device, weights_only=False)
        saved_arch = ckpt.get("arch", {})
        for k, v in self._arch.items():
            if saved_arch.get(k) != v:
                raise RuntimeError(
                    f"Checkpoint arch mismatch on '{k}': "
                    f"saved={saved_arch.get(k)!r} vs current={v!r}"
                )
        self._model.load_state_dict(ckpt["model_state"])
        self._encoder.load_state_dict(ckpt["encoder_state"])
        self._encoder.unit_type_to_id = dict(ckpt["unit_type_to_id"])
        if "optimizer_state" in ckpt:
            try:
                self._trainer.optimizer.load_state_dict(ckpt["optimizer_state"])
            except Exception as e:
                self._logger.warning(
                    f"Couldn't restore optimizer state: {e}. Training "
                    f"will re-accumulate momentum from scratch."
                )
        self._logger.info(f"Loaded checkpoint from {path}")


def _register() -> None:
    import policy
    policy.register("transformer", TransformerPolicy)


_register()
