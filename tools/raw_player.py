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


END_TURN_RULES = ("joint", "actor")


def shift_end_turn_prior(priors: np.ndarray, is_end: np.ndarray, offset: float) -> np.ndarray:
    """The joint priors after adding `offset` to the end_turn ACTOR
    logit, exactly: end_turn's prior is its actor mass p, and the other
    actors' masses scale by one common factor under a softmax, so
    p' = sigmoid(logit(p) + offset) and every non-end prior scales by
    (1 - p') / (1 - p). The attribution arm of the actor-level test
    (docs/training_signal_panel_20260905.md, test 1): the same nudge the
    mini-category gate applies on the server, applied on the client to
    whatever priors it was given."""
    if not offset or not is_end.any():
        return priors
    out = np.array(priors, dtype=np.float64, copy=True)
    p_end = float(out[is_end].sum())
    if p_end <= 0.0 or p_end >= 1.0:
        return out
    logit = np.log(p_end) - np.log1p(-p_end) + float(offset)
    p_new = 1.0 / (1.0 + np.exp(-logit))
    out[is_end] *= p_new / p_end
    out[~is_end] *= (1.0 - p_new) / (1.0 - p_end)
    return out


def actor_rule_index(priors: np.ndarray, actors: np.ndarray, is_end: np.ndarray,
                     temperature: float, rng: np.random.Generator) -> int:
    """end_turn decided at the ACTOR level: play it only when its actor
    mass is at least the largest actor marginal (the sum of the joint
    priors over one unit's or recruit slot's legal actions); otherwise
    choose among the non-end actions by `pick_index`. The joint argmax
    compares end_turn's whole actor mass against four-way products,
    so it ends the turn whenever the act mass is spread over several
    units (panel test 1)."""
    if not is_end.any():
        return pick_index(priors, temperature, rng)
    end_idx = int(np.flatnonzero(is_end)[0])
    acting = np.flatnonzero(~is_end)
    if len(acting) == 0:
        return end_idx
    marginal = {}
    for i in acting:
        a = int(actors[i])
        marginal[a] = marginal.get(a, 0.0) + float(priors[i])
    if float(priors[is_end].sum()) >= max(marginal.values()):
        return end_idx
    return int(acting[pick_index(priors[acting], temperature, rng)])


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
                 seed: Optional[int] = None, forbid_end_turn: bool = False,
                 compact_selection: bool = True, end_turn_rule: str = "joint",
                 end_turn_offset: float = 0.0):
        if temperature < 0.0:
            raise ValueError("temperature must be >= 0 (0 = argmax)")
        if end_turn_rule not in END_TURN_RULES:
            raise ValueError(f"end_turn_rule must be one of {END_TURN_RULES}")
        self._base = base
        self.temperature = float(temperature)
        self._rng = np.random.default_rng(seed)
        # How end_turn is decided (panel test 1): "joint" = the joint
        # argmax or sample over every legal action; "actor" = end_turn
        # only when its actor mass leads every actor marginal
        # (actor_rule_index). `end_turn_offset` shifts end_turn's actor
        # logit before either (shift_end_turn_prior).
        self.end_turn_rule = end_turn_rule
        self.end_turn_offset = float(end_turn_offset)
        # The continue-edit proposer of tools/turn_gap.py: keep acting
        # while any non-end action is legal (end_turn only when nothing
        # else is).
        self.forbid_end_turn = bool(forbid_end_turn)
        # Behind a shared inference server the model output carries the
        # legal actions as compact arrays (server-side priors); picking
        # on those and materializing one action skips building every
        # LegalActionPrior, a quarter of the worker's Python per
        # decision (2026-09-11 worker profile). Same choice, same rng
        # draws (tests/test_server_priors.py); False forces the list.
        self.compact_selection = bool(compact_selection)

    def _select_compact(self, compact, encoded, decision_step: int) -> Optional[Dict]:
        """The choice on the compact arrays, or None when the list path
        applies (an oracle anneal the server cannot carry, which the
        list path reports)."""
        from wesnoth_ai.action_sampler import combat_alphas_at
        from wesnoth_ai.server_priors import KIND_END_TURN, compact_action
        if any(combat_alphas_at(decision_step)) or any(combat_alphas_at(0)):
            return None
        n = len(compact.prior)
        if n == 0:
            return {"type": "end_turn"}
        idx = np.arange(n)
        if self.forbid_end_turn:
            acting = idx[compact.kind != KIND_END_TURN]
            if len(acting):
                idx = acting
        priors = np.asarray(compact.prior, dtype=np.float64)[idx]
        is_end = np.asarray(compact.kind)[idx] == KIND_END_TURN
        return compact_action(compact, int(idx[self._choose(
            priors, np.asarray(compact.actor)[idx], is_end)]), encoded)

    def _choose(self, priors: np.ndarray, actors: np.ndarray, is_end: np.ndarray) -> int:
        """The index chosen among `priors` under the player's rule."""
        priors = shift_end_turn_prior(priors, is_end, self.end_turn_offset)
        if self.end_turn_rule == "actor":
            return actor_rule_index(priors, actors, is_end, self.temperature, self._rng)
        return pick_index(priors, self.temperature, self._rng)

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
            compact = getattr(output, "legal_compact", None)
            if compact is not None and self.compact_selection:
                chosen = self._select_compact(compact, encoded, decision_step)
                if chosen is not None:
                    return chosen
            legal = enumerate_legal_actions_with_priors(
                encoded, output, game_state, decision_step=decision_step)
        if not legal:
            return {"type": "end_turn"}
        if self.forbid_end_turn:
            acting = [la for la in legal if la.action.get("type") != "end_turn"]
            legal = acting or legal
        priors = np.array([la.prior for la in legal], dtype=np.float64)
        actors = np.array([la.actor_idx for la in legal], dtype=np.int64)
        is_end = np.array([la.action.get("type") == "end_turn" for la in legal])
        return legal[self._choose(priors, actors, is_end)].action

    def drop_pending(self, game_label: str) -> None:
        pass

    def drop_last_pending(self, game_label: str) -> bool:
        # Nothing recorded; the bounce retry just re-decides.
        return True
