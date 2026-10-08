"""The look-ahead player's evaluators (tools/lookahead_player.py): a batch
of outcome states (`lookahead_world.Outcome`, each holding a core and its
side to move) and the deciding side -> one value per state in [-1, 1],
signed to that side.

  material  tanh(the side's HP margin over the two player sides / hp_scale)
            (`critic_data.player_hp_margin`'s margin); no forward.
  critic    a step-1 critic checkpoint (wesnoth_ai/critic.py) read on each
            state's core under its view ("obs": the side to move's
            observation; "true": fog off and every unit uncovered,
            `critic_data.encode_view`), batched per decision, its value
            for the side to move negated where the opponent moves.
  rollout   the interface for rollouts scored by an evaluator; not built.

Each evaluator counts the states it valued, its forwards (`forwards`:
batched model calls; 0 for material) and its seconds encoding the states
and running the model (`seconds_encode`, `seconds_forward`).
"""
from __future__ import annotations

import math
import time
from typing import Dict, Sequence

import numpy as np

from wesnoth_ai.classes import opponent_of


class Evaluator:
    """values(states, side) -> np.ndarray of len(states) in [-1, 1]."""
    name = "evaluator"

    def __init__(self):
        self.states = 0
        self.forwards = 0
        self.seconds_encode = 0.0
        self.seconds_forward = 0.0

    def values(self, states: Sequence, side: int) -> np.ndarray:
        raise NotImplementedError


def core_hp_margin(core, side: int) -> int:
    """The side's units' hit points less its opponent's, read off a
    `GameCore` (the margin of `critic_data.player_hp_margin`)."""
    them = opponent_of(side)
    margin = 0
    for d in core.units_export():
        s = int(d["side"])
        if s == side:
            margin += int(d["current_hp"])
        elif s == them:
            margin -= int(d["current_hp"])
    return margin


class MaterialEvaluator(Evaluator):
    name = "material"

    def __init__(self, hp_scale: float):
        super().__init__()
        self.hp_scale = float(hp_scale)

    def values(self, states: Sequence, side: int) -> np.ndarray:
        self.states += len(states)
        return np.array([math.tanh(core_hp_margin(s.core.core, side) / self.hp_scale) for s in states],
                        dtype=np.float64)


class CriticEvaluator(Evaluator):
    name = "critic"

    def __init__(self, checkpoint: str, view: str, device: str = "cpu", batch: int = 64):
        super().__init__()
        import torch
        from wesnoth_ai.critic import load_critic
        self.device = torch.device(device)
        self.encoder, self.model, self.meta = load_critic(checkpoint, self.device)
        trained = (self.meta.get("critic") or {}).get("view")
        if trained != view:
            raise ValueError(f"{checkpoint} is a critic of the {trained!r} view; the config reads it "
                             f"under {view!r}")
        self.view = view
        self.batch = int(batch)
        self.type_to_id = dict(self.encoder.unit_type_to_id)
        self.faction_to_id = dict(self.encoder.faction_to_id)

    def values(self, states: Sequence, side: int) -> np.ndarray:
        from wesnoth_ai.critic import critic_values
        from wesnoth_ai.critic_data import encode_view
        if not states:
            return np.zeros(0, dtype=np.float64)
        t0 = time.perf_counter()
        raws = [encode_view(s.core, self.view, self.type_to_id, self.faction_to_id) for s in states]
        t1 = time.perf_counter()
        mover = critic_values(self.encoder, self.model, raws, self.device, batch=self.batch)
        self.seconds_encode += t1 - t0
        self.seconds_forward += time.perf_counter() - t1
        self.states += len(states)
        self.forwards += -(-len(states) // self.batch)
        signs = np.array([1.0 if s.side_to_move == side else -1.0 for s in states])
        return np.clip(np.asarray(mover, dtype=np.float64) * signs, -1.0, 1.0)


class RolloutEvaluator(Evaluator):
    """Rollouts from each state by a player, scored by an evaluator at a
    horizon (docs/selfplay_program_20261008.md: the evaluator step 1's
    Kill reading falls back on). The interface only: building one
    raises until the rollouts are written."""
    name = "rollout"

    def __init__(self, **spec):
        raise NotImplementedError("the rollout evaluator is an interface only; rollouts are not "
                                  "implemented (docs/selfplay_program_20261008.md)")


def build_evaluator(spec: Dict) -> Evaluator:
    """The evaluator a look-ahead config's "evaluator" entry names."""
    name = spec.get("name")
    if name == "material":
        return MaterialEvaluator(float(spec["hp_scale"]))
    if name == "critic":
        return CriticEvaluator(str(spec["checkpoint"]), str(spec["view"]), str(spec.get("device", "cpu")),
                               int(spec.get("batch", 64)))
    if name == "rollout":
        return RolloutEvaluator(**{k: v for k, v in spec.items() if k != "name"})
    raise ValueError(f"unknown evaluator {name!r}")
