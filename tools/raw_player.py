"""Raw-policy player with a joint sampling temperature (no search).

The legacy raw player (`TransformerPolicy.select_action`) samples the
factored chain actor -> type -> target -> weapon at temperature 1,
i.e. from the policy's full distribution over every legal action.
This player enumerates every legal action with its joint prior (the
same `enumerate_legal_actions_with_priors` search reads) and picks

    p(a) proportional to prior(a) ** (1 / temperature)   if temperature > 0
    a = argmax prior(a)                                   if temperature = 0

At temperature 1 it equals the legacy player in distribution (chain
rule). It exists for the raw-argmax control: every eval before
2026-09-04 compared search (argmax of visits after 30 decisions)
against the SAMPLED raw policy, so "search improves the seed" was
never separated from "argmax beats sampling".

Result provenance: `tools/eval_procedure.procedure_of` tags this
player "raw:t<temperature>"; the legacy sampler stays "raw".
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors


def pick_index(priors: np.ndarray, temperature: float,
               rng: np.random.Generator) -> int:
    """Index of the chosen action: argmax at temperature 0, else a
    draw proportional to prior ** (1 / temperature)."""
    if temperature <= 0.0:
        return int(priors.argmax())
    logp = np.log(np.maximum(priors, 1e-300)) / temperature
    logp -= logp.max()
    p = np.exp(logp)
    p /= p.sum()
    return int(rng.choice(len(priors), p=p))


class RawPolicyPlayer:
    """Same duck type the eval loop drives (`_PolicyPair`):
    select_action / drop_pending / drop_last_pending. Records no
    training targets."""

    trainable = False

    def __init__(self, base, temperature: float,
                 seed: Optional[int] = None, forbid_end_turn: bool = False):
        if temperature < 0.0:
            raise ValueError("temperature must be >= 0 (0 = argmax)")
        self._base = base
        self.temperature = float(temperature)
        self._rng = np.random.default_rng(seed)
        # The continue-edit proposer of tools/turn_gap.py: keep acting
        # while any non-end action is legal (end_turn only when nothing
        # else is).
        self.forbid_end_turn = bool(forbid_end_turn)

    def select_action(self, game_state, *, game_label: str = "default",
                      sim=None) -> Dict:
        base = self._base
        # The inference model is read per call: elo_eval_game swaps a
        # forward-counting proxy onto the base before play.
        with base._lock:
            decision_step = base._decision_step
            base._decision_step += 1
        with torch.no_grad():
            encoded = base._inference_encoder.encode(game_state)
            output = base._inference_model(encoded)
            legal = enumerate_legal_actions_with_priors(
                encoded, output, game_state, decision_step=decision_step)
        if not legal:
            return {"type": "end_turn"}
        if self.forbid_end_turn:
            acting = [la for la in legal if la.action.get("type") != "end_turn"]
            legal = acting or legal
        priors = np.array([la.prior for la in legal], dtype=np.float64)
        return legal[pick_index(priors, self.temperature, self._rng)].action

    def drop_pending(self, game_label: str) -> None:
        pass

    def drop_last_pending(self, game_label: str) -> bool:
        # Nothing recorded; the bounce retry just re-decides.
        return True
