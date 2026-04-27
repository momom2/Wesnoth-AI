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


# Disable PyTorch's "better transformer" MHA fast path. The fused op
# it dispatches to (aten::_transformer_encoder_layer_fwd) is not
# registered in torch-directml, so every forward silently falls back
# to CPU (with per-layer PCI-e round-trips) — the opposite of what we
# want by moving to GPU. The slow path uses primitive ops (linear,
# bmm, softmax) that DML implements natively, so we pay a ~10-20% hit
# on CPU-only runs in exchange for unblocking the GPU path. Call it
# once at module import so it's in effect before any model is built.
try:
    torch.backends.mha.set_fastpath_enabled(False)
except AttributeError:
    # Older PyTorch without this toggle; nothing to disable.
    pass


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
        # Default to select_device() — picks the discrete DirectML GPU
        # when available, falls back to CPU otherwise. The prior
        # hard-coded CPU default was needed because Categorical's
        # scatter-in-backward broke on DML; action_sampler.py now uses
        # a gather-based log_prob / entropy and Gumbel-max sampling
        # (no scatter, no multinomial), so DML is usable for training
        # too. Override via `device=torch.device("cpu")` or env var
        # WESNOTH_AI_DEVICE=cpu if we hit a new DML op gap.
        # CPU default again. DML runs work for rollout (single-sample
        # forwards are competitive with CPU once the MHA/TransformerEncoder
        # fast paths are off), but training on the RX 6600 via DirectML
        # triggered "The GPU device instance has been suspended" (Windows
        # TDR) ~90s into the first train_step — even at batch=4 with the
        # two-pass trainer. The backward-heavy workload on this driver /
        # DML-plugin combination is unstable for our model. Code paths
        # remain in place: set WESNOTH_AI_DEVICE=dml:1 to opt in for
        # rollout-only experiments, or pass `device=` explicitly.
        self._device = (
            device if device is not None else torch.device("cpu")
        )
        self._logger = logging.getLogger("transformer_policy")
        self._logger.info(f"TransformerPolicy device: {describe(self._device)}")

        # Batch size: DML likes B≥4 (amortizes kernel launch) but our
        # runs OOM/TDR above B=4 at seq=1622. CPU prefers B=1 (batched
        # MHA doesn't speed up; see trainer.py config note). Keep the
        # auto-select so if we opt into DML for rollout it picks 4,
        # and CPU stays at 1.
        if trainer_config is None:
            is_dml = "privateuseone" in str(self._device)
            trainer_config = TrainerConfig(train_batch_size=4 if is_dml else 1)

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

        SAFE TO RUN IN A WORKER THREAD while the main thread's rollout
        loop continues. The concurrency contract:
          * We only pop from the FRONT of self._queue; main-thread
            observe() only appends to the END. list.pop(0) and
            list.append() are both atomic under the GIL, so neither
            corrupts the other regardless of interleaving.
          * `self._pending` is only touched by the main thread (in
            `select_action` / `observe`), never by us — no race.
          * Model weights ARE mutated here and read by `select_action`
            on the main thread. We rely on GIL-protected tensor ops +
            RL tolerance for off-policy samples: a rollout step that
            reads weights mid-optimizer-update gets noisy output, which
            the next train_step washes out. No lock because it would
            serialize rollouts with training and defeat the point of
            running async.

        Queue management: we drain OLDEST trajectories first (FIFO) up
        to max_transitions_per_step, leaving any excess for the next
        call. A safety cap then trims the remaining queue to 2x the
        per-step budget so it can't grow unboundedly if rollouts out-
        pace training forever; the dropped trajectories are the oldest
        (most off-policy), which is the right set to lose.
        """
        if not self._queue:
            return TrainStats()
        t0 = time.perf_counter()

        target = self._trainer.config.max_transitions_per_step
        taken: List[List[Transition]] = []
        trans_count = 0
        # FIFO drain up to the per-step budget. pop(0) under GIL is
        # atomic — safe while main thread appends.
        while self._queue and trans_count < target:
            try:
                traj = self._queue.pop(0)
            except IndexError:
                break  # race with overflow-trim below, shouldn't happen
            taken.append(traj)
            trans_count += len(traj)

        # Safety cap: if the tail of the queue has more than `target`
        # transitions of unprocessed data still sitting there, drop
        # the oldest. Rollouts producing data faster than training
        # can keep up means older samples are becoming progressively
        # more off-policy; training on them later hurts more than
        # losing them.
        max_remaining = target
        dropped_trajs = 0
        dropped_trans = 0
        def _remaining_trans():
            return sum(len(t) for t in self._queue)
        while self._queue and _remaining_trans() > max_remaining:
            try:
                t = self._queue.pop(0)
            except IndexError:
                break
            dropped_trajs += 1
            dropped_trans += len(t)

        queued_after = len(self._queue)
        msg = (
            f"train_step starting: drained {len(taken)} trajectories "
            f"({trans_count} transitions); queue has {queued_after} "
            f"trajectories remaining"
        )
        if dropped_trajs:
            msg += (f" (dropped {dropped_trajs} stale trajectories / "
                    f"{dropped_trans} transitions to bound queue)")
        self._logger.info(msg)

        if not taken:
            return TrainStats()

        self._model.train()
        self._encoder.train()
        try:
            stats = self._trainer.step(taken)
        finally:
            self._model.eval()
            self._encoder.eval()
        # Keep the historical log key names so dashboards don't break.
        n_transitions_total = trans_count
        queue = taken  # for the log line below
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
                "faction_to_id":   dict(self._encoder.faction_to_id),
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
        # Faction vocab — present in checkpoints saved after faction
        # conditioning landed. Older checkpoints lack it; fall back to
        # the encoder's default-seeded vocab so names still resolve.
        if "faction_to_id" in ckpt:
            self._encoder.faction_to_id = dict(ckpt["faction_to_id"])
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
