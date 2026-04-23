"""Learned policy driven by a transformer over tokenized state.

Registers in `policy.py`'s registry as ``--policy transformer``.

Phase 3.1 scope: a *plays-games-without-crashing* wrapper. Weights
are random; actions are essentially noise. Phase 3.2 adds the trainer
and makes this actually learn.

Design:
  - Owns its own device, encoder, and model (no sharing with the
    converter — encoder has an independent name→id dict).
  - select_action runs under torch.no_grad() for Phase 3.1. Phase 3.2
    will switch to full-grad mode when trajectories are being
    collected, or use a parallel "rollout" path that doesn't track
    grads and a separate "training" forward that does.
  - Stores nothing across calls yet. Phase 3.2 adds a rollout buffer
    of (state, action, log_prob, value) for policy-gradient updates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch

from action_sampler import sample_action
from classes import GameState
from device import describe, select_device
from encoder import GameStateEncoder
from model import WesnothModel


class TransformerPolicy:
    """Learned policy — conforms to the ``Policy`` Protocol."""

    # Flag for game_manager.run_training: "call train_step on me".
    # No trainer exists yet; the hook is a no-op until Phase 3.2.
    trainable = True

    def __init__(
        self,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        d_ff: int = 256,
        device=None,
    ):
        self._device = device if device is not None else select_device()
        self._encoder = GameStateEncoder(d_model=d_model).to(self._device)
        self._model = WesnothModel(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
        ).to(self._device)

        # Eval mode: we want no dropout during rollout. Phase 3.2 will
        # switch to .train() during training-phase forwards only.
        self._encoder.eval()
        self._model.eval()

    def select_action(self, game_state: GameState) -> Dict:
        with torch.no_grad():
            encoded = self._encoder.encode(game_state)
            output = self._model(encoded)
            sampled = sample_action(encoded, output, game_state)
        return sampled.action

    # ----- Phase 3.2 training hooks (stubs) ------------------------------

    def train_step(self) -> None:
        """No-op in Phase 3.1. Phase 3.2 will plug the trainer in here."""
        pass

    def save_checkpoint(self, path: Path) -> None:
        """Serialize model + encoder + unit_type_to_id + arch hash."""
        torch.save(
            {
                'model_state':       self._model.state_dict(),
                'encoder_state':     self._encoder.state_dict(),
                'unit_type_to_id':   dict(self._encoder.unit_type_to_id),
                'arch': {
                    'd_model': self._model.d_model,
                    # Phase 2c open question 5: architecture hash guard.
                    # Store the key shape dims so load can sanity-check.
                },
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self._device)
        if ckpt['arch']['d_model'] != self._model.d_model:
            raise RuntimeError(
                f"Checkpoint d_model={ckpt['arch']['d_model']} does not "
                f"match policy d_model={self._model.d_model}"
            )
        self._model.load_state_dict(ckpt['model_state'])
        self._encoder.load_state_dict(ckpt['encoder_state'])
        self._encoder.unit_type_to_id = dict(ckpt['unit_type_to_id'])


def _register() -> None:
    """Register in the Policy registry. Called at import time."""
    import policy
    policy.register('transformer', TransformerPolicy)


_register()
