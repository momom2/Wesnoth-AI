"""The look-ahead player (docs/selfplay_program_20261008.md "The rounds",
step 3): the raw player's prior tilted by a one-step look-ahead with
exact chance.

At each decision:
  1. the raw player's forward and decode (`RawPolicyPlayer.decode`: the
     reference decode, the side's memory read and written once, as the raw
     player does), giving the offered actions and their priors pi;
  2. the candidates: the prior's top k (`kinds="attacks"`: its argmax and
     the attacks among the top k);
  3. each candidate's outcome states with their exact probabilities, on a
     fork of the simulator determinized for the side
     (wesnoth_ai/lookahead_world.py: the observed world by default);
  4. each state valued by the evaluator (wesnoth_ai/lookahead_evaluators.py)
     for the deciding side, one batch per decision; a state where a leader
     died takes the game's result instead;
     Q(a) = sum_o p_o v(s_o);
  5. the played action: argmax over the candidates of
     log pi(a) + clip((Q(a) - V) / sigma, -c, c), V the prior-weighted mean
     of Q over the candidates (Muesli's clipped target, Hessel et al. 2021;
     sigma a fixed scale). Ties go to the action the raw player lists first,
     so with c = 0 the player plays the raw player's argmax.

A candidate whose outcomes cannot be built (`ExpansionError`) keeps its
prior score (advantage 0) and is left out of V; every such failure is
counted by reason in the game's telemetry and the first of each reason is
logged. Without a simulator to fork (`sim=None`) the decision is the raw
player's, counted as "no_sim".

Result provenance: `wesnoth_ai.lookahead_config.procedure_tag`.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tools.raw_player import KIND_NAMES, Decoded, RawPolicyPlayer
from wesnoth_ai.lookahead_config import LookaheadConfig
from wesnoth_ai.lookahead_evaluators import Evaluator, build_evaluator
from wesnoth_ai.lookahead_world import ExpansionError, Outcome, build_world, expand

log = logging.getLogger("lookahead_player")

ATTACK = KIND_NAMES.index("attack")
# The evaluators a process keeps across games (a persistent eval worker),
# by their config entry.
_EVALUATORS: Dict[str, Evaluator] = {}


def lookahead_player(raw: RawPolicyPlayer, config: LookaheadConfig, keep: bool = False) -> "LookaheadPlayer":
    """The look-ahead player over `raw`. `keep`: build the evaluator once
    per configuration in this process and keep it for the next game."""
    key = json.dumps(config.evaluator, sort_keys=True)
    evaluator = _EVALUATORS.get(key) if keep else None
    if evaluator is None:
        evaluator = build_evaluator(config.evaluator)
        if keep:
            _EVALUATORS[key] = evaluator
    return LookaheadPlayer(raw, config, evaluator)


def candidate_indices(decoded: Decoded, base: int, k: int, kinds: str) -> List[int]:
    """The candidates among a decision's offered actions, ascending: the
    prior's top k (ties by listing order), the raw choice `base` always
    among them; with `kinds="attacks"`, `base` and the attacks of the top
    k."""
    n = len(decoded)
    order = np.lexsort((np.arange(n), -decoded.priors))[:max(0, int(k))]
    if kinds == "attacks":
        order = [i for i in order if decoded.kinds[i] == ATTACK]
    return sorted({int(base), *(int(i) for i in order)})


def tilted_choice(priors: np.ndarray, q: np.ndarray, sigma: float, c: float) -> Tuple[int, float]:
    """(index of the played candidate, V): the argmax of
    log pi + clip((Q - V) / sigma, -c, c) over candidates listed in the
    raw player's order, V the prior-weighted mean of Q over the candidates
    whose Q is known (a NaN Q gets advantage 0). The first maximum wins."""
    priors = np.asarray(priors, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    known = np.isfinite(q)
    if not known.any():
        return int(np.argmax(priors)), float("nan")
    weights = priors[known] / priors[known].sum()
    v = float(np.dot(weights, q[known]))
    advantage = np.where(known, (np.where(known, q, v) - v) / float(sigma), 0.0)
    scores = np.log(priors) + np.clip(advantage, -float(c), float(c))
    return int(np.argmax(scores)), v


@dataclass
class GameTelemetry:
    """One game's record of a look-ahead player's decisions."""
    decisions: int = 0
    operated: int = 0
    by_kind: Dict[str, int] = field(default_factory=lambda: dict.fromkeys(KIND_NAMES, 0))
    flips_by_kind: Dict[str, int] = field(default_factory=lambda: dict.fromkeys(KIND_NAMES, 0))
    flips_to_kind: Dict[str, int] = field(default_factory=lambda: dict.fromkeys(KIND_NAMES, 0))
    candidates_by_kind: Dict[str, int] = field(default_factory=lambda: dict.fromkeys(KIND_NAMES, 0))
    states: int = 0
    states_max: int = 0
    terminal_states: int = 0
    forwards: int = 0
    failed: Dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    seconds_max: float = 0.0
    seconds_prior: float = 0.0
    seconds_expand: float = 0.0
    seconds_evaluate: float = 0.0

    def as_dict(self) -> Dict:
        n = max(1, self.decisions)
        out = {k: dict(v) if isinstance(v, dict) else v for k, v in self.__dict__.items()}
        out["flips"] = sum(self.flips_by_kind.values())
        out["candidates"] = sum(self.candidates_by_kind.values())
        out["states_per_decision"] = self.states / n
        out["seconds_per_decision"] = self.seconds / n
        out["forwards_per_decision"] = self.forwards / n
        return out


class LookaheadPlayer:
    """The eval loop's duck type (`select_action`, `drop_pending`,
    `drop_last_pending`) over a raw player at temperature 0 with the joint
    end_turn rule."""

    trainable = False

    def __init__(self, raw: RawPolicyPlayer, config: LookaheadConfig, evaluator: Evaluator):
        if raw.temperature != 0.0 or raw.end_turn_rule != "joint" or raw.forbid_end_turn:
            raise ValueError("the look-ahead player tilts the raw player at temperature 0 with the joint "
                             "end_turn rule (an argmax procedure)")
        self._raw = raw
        self.config = config
        self.evaluator = evaluator
        self.memory_slots = raw.memory_slots
        self._games: Dict[str, GameTelemetry] = {}
        self._warned: set = set()

    # ---- the eval loop's duck type -----------------------------------

    def select_action(self, game_state, *, game_label: str = "default", sim=None) -> Dict:
        t0 = time.perf_counter()
        tel = self._games.setdefault(game_label, GameTelemetry())
        decoded = self._raw.decode(game_state, game_label=game_label)
        tel.seconds_prior += time.perf_counter() - t0
        if decoded is None:
            self._note(tel, KIND_NAMES.index("end_turn"), KIND_NAMES.index("end_turn"), t0)
            return {"type": "end_turn"}
        base = self._raw.choose(decoded)
        chosen = base
        if self.config.k > 0 and len(decoded) > 1:
            chosen = self._operate(decoded, base, game_label, sim, tel)
        self._note(tel, int(decoded.kinds[base]), int(decoded.kinds[chosen]), t0, flipped=chosen != base)
        return decoded.action(chosen)

    def drop_pending(self, game_label: str) -> None:
        self._raw.drop_pending(game_label)

    def drop_last_pending(self, game_label: str) -> bool:
        return self._raw.drop_last_pending(game_label)

    def telemetry(self, game_label: str) -> Dict:
        """The game's telemetry so far (GameTelemetry.as_dict)."""
        return self._games.setdefault(game_label, GameTelemetry()).as_dict()

    def pop_telemetry(self, game_label: str) -> Dict:
        return self._games.pop(game_label, GameTelemetry()).as_dict()

    # ---- the operator ----------------------------------------------------

    def _operate(self, decoded: Decoded, base: int, game_label: str, sim, tel: GameTelemetry) -> int:
        cfg = self.config
        cands = candidate_indices(decoded, base, cfg.k, cfg.kinds)
        if len(cands) < 2:
            return base
        if sim is None:
            self._fail(tel, "no_sim", "no simulator to expand on")
            return base
        t0 = time.perf_counter()
        side = int(sim.current_side)
        try:
            world = build_world(sim, side, cfg.determinization, f"lookahead:{game_label}:{tel.decisions}")
        except ExpansionError as e:
            self._fail(tel, e.reason, str(e))
            return base
        tel.operated += 1
        for i in cands:
            tel.candidates_by_kind[KIND_NAMES[int(decoded.kinds[i])]] += 1
        outcomes: List[Optional[List[Outcome]]] = []
        for i in cands:
            try:
                outcomes.append(expand(world, decoded.action(i), max_attack_leaves=cfg.max_attack_leaves))
            except ExpansionError as e:
                self._fail(tel, e.reason, str(e))
                outcomes.append(None)
            except AssertionError:
                raise
            except Exception as e:                       # noqa: BLE001 -- counted, logged, played on
                self._fail(tel, f"error:{type(e).__name__}", repr(e), exc_info=True)
                outcomes.append(None)
        t1 = time.perf_counter()
        q = self._values(outcomes, side, tel)
        tel.seconds_expand += t1 - t0
        tel.seconds_evaluate += time.perf_counter() - t1
        pick, _v = tilted_choice(decoded.priors[cands], q, cfg.sigma, cfg.c)
        return cands[pick]

    def _values(self, outcomes: Sequence[Optional[List[Outcome]]], side: int, tel: GameTelemetry) -> np.ndarray:
        """Q of each candidate (NaN where its expansion failed): one
        evaluator batch over every outcome state where the game goes on."""
        pending = [o for outs in outcomes if outs for o in outs if o.terminal is None]
        before = self.evaluator.forwards
        values = self.evaluator.values(pending, side) if pending else np.zeros(0)
        tel.forwards += self.evaluator.forwards - before
        tel.states += len(pending)
        tel.states_max = max(tel.states_max, len(pending))
        value_of = {id(o): float(v) for o, v in zip(pending, values)}
        q = np.full(len(outcomes), np.nan)
        for i, outs in enumerate(outcomes):
            if not outs:
                continue
            tel.terminal_states += sum(o.terminal is not None for o in outs)
            q[i] = sum(o.prob * (o.terminal if o.terminal is not None else value_of[id(o)]) for o in outs)
        return q

    # ---- bookkeeping -----------------------------------------------------

    def _note(self, tel: GameTelemetry, base_kind: int, played_kind: int, t0: float,
              flipped: bool = False) -> None:
        tel.decisions += 1
        tel.by_kind[KIND_NAMES[base_kind]] += 1
        if flipped:
            tel.flips_by_kind[KIND_NAMES[base_kind]] += 1
            tel.flips_to_kind[KIND_NAMES[played_kind]] += 1
        dt = time.perf_counter() - t0
        tel.seconds += dt
        tel.seconds_max = max(tel.seconds_max, dt)

    def _fail(self, tel: GameTelemetry, reason: str, detail: str, exc_info: bool = False) -> None:
        tel.failed[reason] = tel.failed.get(reason, 0) + 1
        if reason not in self._warned:
            self._warned.add(reason)
            log.warning("look-ahead: a candidate's outcomes could not be built (%s); it keeps its prior "
                        "score, counted per game under failed[%r]", detail, reason, exc_info=exc_info)
