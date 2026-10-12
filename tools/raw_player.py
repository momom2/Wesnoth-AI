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

The decode knobs (docs/decode_knobs_prereg_20261012.md): before the
choice, the end_turn offset shifts end_turn's logit
(`shift_end_turn_prior`), and the attack offset, the recruit offset and
the attack time-of-day slope shift the logit of every attack and recruit
(`knob_logits`, `shift_by_logits`), a move being the zero of the scale.
The slope multiplies the attacker's time-of-day damage modifier at the
hex it strikes from (`attack_tod_modifiers`). All of them at 0 leave the
priors untouched, the very array.

Result provenance: `tools/eval_procedure.procedure_of` tags this
player "raw:t<temperature>" with its decode ("+eo-1.5+ao0.4..."); the
legacy sampler stays "raw".
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors
from wesnoth_ai.memory import MemoryState
from wesnoth_ai.server_priors import KIND_ATTACK, KIND_END_TURN, KIND_MOVE, KIND_RECRUIT


END_TURN_RULES = ("joint", "actor")
# The list path's action types as the compact arrays' kind codes.
KIND_OF_TYPE = {"attack": KIND_ATTACK, "move": KIND_MOVE, "recruit": KIND_RECRUIT,
                "end_turn": KIND_END_TURN}
# Attacks whose time of day the slope could not read (no unit of the side
# on the attack's hex, no hex to strike from, a hex off the core's map):
# they get modifier 0 and are counted here, by reason, for the process (the
# first one logged) and on the player that asked (`RawPolicyPlayer.tod_unread`,
# which a match records per game).
TOD_UNREAD: Dict[str, int] = {}

log = logging.getLogger("raw_player")


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


def knob_logits(kinds: np.ndarray, attack_tod: Optional[np.ndarray], attack_offset: float,
                recruit_offset: float, attack_tod_slope: float) -> np.ndarray:
    """The decode knobs' logit offset of each legal action: `attack_offset`
    plus `attack_tod_slope` times the attack's entry of `attack_tod` on an
    attack, `recruit_offset` on a recruit, 0 on a move and on end_turn
    (whose offset `shift_end_turn_prior` applies)."""
    kinds = np.asarray(kinds)
    out = np.zeros(len(kinds), dtype=np.float64)
    attack = kinds == KIND_ATTACK
    out[attack] = float(attack_offset)
    if attack_tod_slope:
        out[attack] += float(attack_tod_slope) * np.asarray(attack_tod, dtype=np.float64)[attack]
    out[kinds == KIND_RECRUIT] = float(recruit_offset)
    return out


def shift_by_logits(priors: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """The joint priors after adding `logits` to the actions' logits: each
    prior times exp(its offset), renormalised (a common shift of every
    logit changes nothing, so a move at 0 is the scale's zero)."""
    out = np.asarray(priors, dtype=np.float64) * np.exp(np.asarray(logits, dtype=np.float64))
    total = float(out.sum())
    if not total > 0.0:
        return np.array(priors, dtype=np.float64, copy=True)
    return out / total


def _tod_unread(reason: str, detail: str, unread: Optional[Dict[str, int]]) -> None:
    if not TOD_UNREAD:
        log.warning("the attack time-of-day slope reads 0 for an attack: %s (%s); "
                    "counted in raw_player.TOD_UNREAD", reason, detail)
    for counts in (TOD_UNREAD, unread):
        if counts is not None:
            counts[reason] = counts.get(reason, 0) + 1


def attack_tod_modifiers(game_state, attacks: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]],
                         unread: Optional[Dict[str, int]] = None) -> np.ndarray:
    """For each attack (the attacker's hex, the target's hex), the
    attacker's time-of-day damage modifier as a fraction (-0.25 to +0.25
    under the pool's schedules) at the hex it strikes from: the hex the
    simulator moves it to before the fight (`pathfind_sim.attack_hexes`,
    the side's observable reach), read from the core that answers for the
    state (`GameCore.combat_tod_modifier`: its alignment and fearless trait
    under the illuminated time of day there, as the fight reads them).
    An attack it cannot read is counted in TOD_UNREAD and `unread`."""
    from tools.pathfind_sim import attack_hexes
    from wesnoth_ai.game_core import core_for
    out = np.zeros(len(attacks), dtype=np.float64)
    if not attacks:
        return out
    core = core_for(game_state).core
    side = int(game_state.global_info.current_side)
    units = {(u.position.x, u.position.y): u for u in game_state.map.units if u.side == side}
    by_start: Dict[Tuple[int, int], List[int]] = {}
    for k, (start, _target) in enumerate(attacks):
        by_start.setdefault(start, []).append(k)
    for start, ks in by_start.items():
        unit = units.get(start)
        if unit is None:
            for _ in ks:
                _tod_unread("no unit of the side on the attack's hex", str(start), unread)
            continue
        strike = attack_hexes(game_state, unit, [attacks[k][1] for k in ks])
        for k, hexx in zip(ks, strike):
            if hexx is None:
                _tod_unread("no hex to strike from", f"{start} -> {attacks[k][1]}", unread)
                continue
            modifier = core.combat_tod_modifier(start[0], start[1], hexx[0], hexx[1])
            if modifier is None:
                _tod_unread("a hex off the core's map", f"{start} -> {hexx}", unread)
                continue
            out[k] = modifier / 100.0
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


def _xy(position) -> Tuple[int, int]:
    return (int(position.x), int(position.y))


def model_memory_slots(base) -> Optional[int]:
    """The memory slots of the model a policy base runs: 0 for a model
    without a memory, None when it does not say (a served model whose
    transport carries no hello)."""
    model = getattr(base, "_inference_model", None)
    slots = getattr(model, "memory_slots", None)
    return None if slots is None else int(slots or 0)


class RawPolicyPlayer:
    """Same duck type the eval loop drives (`_PolicyPair`):
    select_action / drop_pending / drop_last_pending. Records no
    training targets."""

    trainable = False

    def __init__(self, base, temperature: float,
                 seed: Optional[int] = None, forbid_end_turn: bool = False,
                 compact_selection: bool = True, end_turn_rule: str = "joint",
                 end_turn_offset: float = 0.0, memory_slots: Optional[int] = None,
                 attack_offset: float = 0.0, recruit_offset: float = 0.0,
                 attack_tod_slope: float = 0.0):
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
        # The decode knobs on attacks and recruits (`knob_logits`); all 0
        # leaves the choice exactly today's.
        self.attack_offset = float(attack_offset)
        self.recruit_offset = float(recruit_offset)
        self.attack_tod_slope = float(attack_tod_slope)
        # The attacks whose time of day the slope could not read, by reason.
        self.tod_unread: Dict[str, int] = {}
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
        # A model with a memory (docs/parity_memory_design_20260929.md,
        # "Serving and play"): the active slots this player reads and
        # writes, and each side's state per game, from the side's previous
        # decision to its next (a bounced recruit decided again is one more
        # decision). None: a model without a memory.
        model_slots = model_memory_slots(base)
        if memory_slots is None and model_slots:
            raise ValueError(f"the model carries a memory of {model_slots} slots: name the slots this "
                             f"player uses (memory_slots=0 plays it without its memory)")
        self.memory_slots = None if memory_slots is None else int(memory_slots)
        self._memories: Dict[Tuple[str, int], object] = {}
        # Per game, the last decision's (side key, memory before it).
        self._undo: Dict[str, Tuple[Tuple[str, int], object]] = {}

    def _select_compact(self, compact, encoded, decision_step: int,
                        game_state=None) -> Optional[Dict]:
        """The choice on the compact arrays, or None when the list path
        applies (an oracle anneal the server cannot carry, which the
        list path reports)."""
        from wesnoth_ai.action_sampler import combat_alphas_at
        from wesnoth_ai.server_priors import compact_action
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
        kinds = np.asarray(compact.kind)[idx]
        is_end = kinds == KIND_END_TURN
        logits = None
        if self.knobs_on:
            attacks = [(_xy(encoded.unit_positions[int(compact.actor[i])]),
                        _xy(encoded.hex_positions[int(compact.target[i])]))
                       for i in idx[kinds == KIND_ATTACK]]
            logits = self._knob_logits(kinds, attacks, game_state)
        return compact_action(compact, int(idx[self._choose(
            priors, np.asarray(compact.actor)[idx], is_end, logits)]), encoded)

    @property
    def knobs_on(self) -> bool:
        """Whether a knob on attacks or recruits is set."""
        return bool(self.attack_offset or self.recruit_offset or self.attack_tod_slope)

    def _knob_logits(self, kinds: np.ndarray, attacks, game_state) -> np.ndarray:
        """`knob_logits` of the legal actions, `attacks` being the
        (attacker's hex, target's hex) of the attacks among them in order;
        their time of day is read only when the slope is set."""
        tod = None
        if self.attack_tod_slope:
            tod = np.zeros(len(kinds), dtype=np.float64)
            tod[np.asarray(kinds) == KIND_ATTACK] = attack_tod_modifiers(game_state, attacks,
                                                                           self.tod_unread)
        return knob_logits(kinds, tod, self.attack_offset, self.recruit_offset,
                           self.attack_tod_slope)

    def _choose(self, priors: np.ndarray, actors: np.ndarray, is_end: np.ndarray,
                knob_offsets: Optional[np.ndarray] = None) -> int:
        """The index chosen among `priors` under the player's rule, the
        knobs' logit offsets (`_knob_logits`) added first when given."""
        priors = shift_end_turn_prior(priors, is_end, self.end_turn_offset)
        if knob_offsets is not None:
            priors = shift_by_logits(priors, knob_offsets)
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
            output = self._forward(base, encoded, game_label, game_state)
            compact = getattr(output, "legal_compact", None)
            if compact is not None and self.compact_selection:
                chosen = self._select_compact(compact, encoded, decision_step, game_state)
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
        logits = None
        if self.knobs_on:
            kinds = np.array([KIND_OF_TYPE[la.action["type"]] for la in legal], dtype=np.int64)
            attacks = [(_xy(la.action["start_hex"]), _xy(la.action["target_hex"]))
                       for la in legal if la.action["type"] == "attack"]
            logits = self._knob_logits(kinds, attacks, game_state)
        return legal[self._choose(priors, actors, is_end, logits)].action

    def _forward(self, base, encoded, game_label: str, game_state):
        """The model's output, the side's memory read and written when the
        model has one."""
        if self.memory_slots is None:
            return base._inference_model(encoded)
        key = (game_label, int(game_state.global_info.current_side))
        output = base._inference_model(encoded, memory=MemoryState(self.memory_slots,
                                                                   self._memories.get(key)))
        if output.memory is None:
            raise RuntimeError("a player with a memory got no memory back from its model")
        self._undo[game_label] = (key, self._memories.get(key))
        self._memories[key] = output.memory
        return output

    def advance_memory(self, game_state, *, game_label: str = "default") -> None:
        """The side to move's memory read and written at a decision this
        player does not choose (a corpus game's history, a recorded turn
        replayed): the forward a decision runs, without the choice."""
        if self.memory_slots is None:
            return
        with torch.no_grad():
            encoded = self._base._inference_encoder.encode(game_state)
            self._forward(self._base, encoded, game_label, game_state)
        self._undo.pop(game_label, None)

    def memory_of(self, game_label: str, side: int):
        """The memory `side` holds in game `game_label`: what its last
        decision wrote, None before its first."""
        return self._memories.get((game_label, int(side)))

    def set_memory(self, game_label: str, side: int, state) -> None:
        """Start `side` of game `game_label` from `state` (a memory
        carried over from another player of the same game), None for the
        learned initial memory."""
        key = (game_label, int(side))
        if state is None:
            self._memories.pop(key, None)
        else:
            self._memories[key] = state

    def drop_pending(self, game_label: str) -> None:
        """The game is over: its sides' memories go."""
        for key in [k for k in self._memories if k[0] == game_label]:
            del self._memories[key]
        self._undo.pop(game_label, None)

    def drop_last_pending(self, game_label: str) -> bool:
        """The last decision was refused (a recruit the engine bounces) and
        is decided again: the side's memory goes back to what it was
        before it. The engine records nothing for a refused recruit
        (`menu_handler::do_recruit`, src/menu_events.cpp:350-370, 1.18.4),
        so the corpus holds no position there."""
        undo = self._undo.pop(game_label, None)
        if undo is not None:
            key, before = undo
            if before is None:
                self._memories.pop(key, None)
            else:
                self._memories[key] = before
        return True
