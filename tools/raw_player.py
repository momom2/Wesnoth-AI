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

Before the choice, the end_turn offset shifts end_turn's logit
(`shift_end_turn_prior`) and the attack offset every attack's
(`shift_attack_prior`). At argmax, the near-tie coins
(tools/coin_decode.py) play the runner-up at a logged coin where the
two best priors nearly tie; `pop_coin_log` hands a game's band
decisions to the caller, who records them.

Result provenance: `tools/eval_procedure.procedure_of` tags this
player "raw:t<temperature>"; the legacy sampler stays "raw".
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from tools.coin_decode import describe_action, near_tie, near_tie_coin
from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors
from wesnoth_ai.memory import MemoryState
from wesnoth_ai.stage_timing import stage


END_TURN_RULES = ("joint", "actor")
# A band decision as `_choose` returns it: (a0, a1, G, D), indices into the
# arrays it was given.
Band = Tuple[int, int, float, int]


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


def shift_attack_prior(priors: np.ndarray, is_attack: np.ndarray, offset: float) -> np.ndarray:
    """The joint priors after adding `offset` to every attack's logit: each
    attack's prior times exp(offset), renormalised, a move being the zero of
    the scale (the attack knob of tag archive/feature-decode-knobs,
    `knob_logits` and `shift_by_logits`, whose +1 reading is the attack
    known answer)."""
    if not offset or not is_attack.any():
        return priors
    out = np.asarray(priors, dtype=np.float64) * np.where(is_attack, np.exp(float(offset)), 1.0)
    total = float(out.sum())
    if not total > 0.0:
        return np.array(priors, dtype=np.float64, copy=True)
    return out / total


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
    # It reads the state only while it decides and keeps nothing of it,
    # so a game loop hands it the simulator's own view instead of a
    # snapshot (`eval_players._decision_state`).
    reads_state_only = True

    def __init__(self, base, temperature: float,
                 seed: Optional[int] = None, forbid_end_turn: bool = False,
                 compact_selection: bool = True, end_turn_rule: str = "joint",
                 end_turn_offset: float = 0.0, memory_slots: Optional[int] = None,
                 attack_offset: float = 0.0, coin_band: float = 0.0, coin_p: float = 0.25,
                 coin_seed: Optional[int] = None):
        if temperature < 0.0:
            raise ValueError("temperature must be >= 0 (0 = argmax)")
        if end_turn_rule not in END_TURN_RULES:
            raise ValueError(f"end_turn_rule must be one of {END_TURN_RULES}")
        if coin_band < 0.0:
            raise ValueError("coin_band must be >= 0 (0 = no coins)")
        if coin_band and (temperature != 0.0 or end_turn_rule != "joint"):
            raise ValueError("the near-tie coins apply at the joint argmax only (temperature 0, rule 'joint')")
        if coin_band and not 0.0 < coin_p < 1.0:
            raise ValueError("coin_p must lie strictly between 0 and 1")
        if coin_band and coin_seed is None:
            raise ValueError("the near-tie coins are keyed by the game seed: pass coin_seed")
        self._base = base
        self.temperature = float(temperature)
        self._rng = np.random.default_rng(seed)
        # How end_turn is decided (panel test 1): "joint" = the joint
        # argmax or sample over every legal action; "actor" = end_turn
        # only when its actor mass leads every actor marginal
        # (actor_rule_index). `end_turn_offset` shifts end_turn's actor
        # logit before either (shift_end_turn_prior), `attack_offset`
        # every attack's logit after it (shift_attack_prior).
        self.end_turn_rule = end_turn_rule
        self.end_turn_offset = float(end_turn_offset)
        self.attack_offset = float(attack_offset)
        # The near-tie coins (tools/coin_decode.py): the band width in nats
        # (0: off), the propensity, the game seed that keys them; per game,
        # each side's decisions so far, the last decision's (side, index),
        # and the band decisions logged.
        self.coin_band = float(coin_band)
        self.coin_p = float(coin_p)
        self.coin_seed = coin_seed
        self._decisions: Dict[Tuple[str, int], int] = {}
        self._last_decision: Dict[str, Tuple[int, int]] = {}
        self._coin_log: Dict[str, List[dict]] = {}
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
                        coin_key: Optional[Tuple[int, int]] = None,
                        game_label: str = "default", sim=None) -> Optional[Dict]:
        """The choice on the compact arrays, or None when the list path
        applies (an oracle anneal the server cannot carry, which the
        list path reports)."""
        from wesnoth_ai.action_sampler import combat_alphas_at
        from wesnoth_ai.server_priors import KIND_ATTACK, KIND_END_TURN, compact_action
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
        pick, band = self._choose(priors, np.asarray(compact.actor)[idx], kinds == KIND_END_TURN,
                                  kinds == KIND_ATTACK, coin_key)
        if band is not None:
            self._log_band(game_label, coin_key, band, sim,
                           compact_action(compact, int(idx[band[0]]), encoded),
                           compact_action(compact, int(idx[band[1]]), encoded))
        return compact_action(compact, int(idx[pick]), encoded)

    def _choose(self, priors: np.ndarray, actors: np.ndarray, is_end: np.ndarray,
                is_attack: np.ndarray, coin_key: Optional[Tuple[int, int]] = None
                ) -> Tuple[int, Optional[Band]]:
        """(the index chosen among `priors` under the player's rule, the
        band decision when the near-tie coin decided it). `coin_key` is the
        mover's (side, decision index) when the coins are on."""
        priors = shift_end_turn_prior(priors, is_end, self.end_turn_offset)
        priors = shift_attack_prior(priors, is_attack, self.attack_offset)
        if self.end_turn_rule == "actor":
            return actor_rule_index(priors, actors, is_end, self.temperature, self._rng), None
        if coin_key is not None:
            tie = near_tie(priors)
            if tie is not None and tie[2] < self.coin_band:
                a0, a1, gap = tie
                d = near_tie_coin(self.coin_seed, coin_key[0], coin_key[1], self.coin_p)
                return (a1 if d else a0), (a0, a1, gap, d)
        return pick_index(priors, self.temperature, self._rng), None

    def _coin_key(self, game_label: str, game_state) -> Optional[Tuple[int, int]]:
        """The mover's (side, decision index) in this game when the coins
        are on: its decisions are counted from 0, a refused recruit decided
        again counting as one more."""
        if not self.coin_band:
            return None
        side = int(game_state.global_info.current_side)
        index = self._decisions.get((game_label, side), 0)
        self._decisions[(game_label, side)] = index + 1
        self._last_decision[game_label] = (side, index)
        return side, index

    def _log_band(self, game_label: str, coin_key: Tuple[int, int], band: Band, sim,
                  action0: Dict, action1: Dict) -> None:
        """A band decision into the game's coin log: the mover's side and
        decision index, the command index it played at (the simulator's
        command count before the step), G, P, D, and a0 and a1."""
        a0, a1, gap, d = band
        self._coin_log.setdefault(game_label, []).append({
            "side": coin_key[0], "i": coin_key[1],
            "cmd": None if sim is None else len(sim.command_history),
            "gap": gap, "p": self.coin_p, "d": d,
            "a0": describe_action(action0), "a1": describe_action(action1)})

    def pop_coin_log(self, game_label: str) -> List[dict]:
        """The band decisions of game `game_label`, in the order they were
        made, handed over once."""
        return self._coin_log.pop(game_label, [])

    def select_action(self, game_state, *, game_label: str = "default",
                      sim=None) -> Dict:
        base = self._base
        # The inference model is read per call: elo_eval_game swaps a
        # forward-counting proxy onto the base before play.
        with base._lock:
            decision_step = base._decision_step
            base._decision_step += 1
        coin_key = self._coin_key(game_label, game_state)
        with torch.no_grad():
            encoded = base._inference_encoder.encode(game_state)
            output = self._forward(base, encoded, game_label, game_state)
            compact = getattr(output, "legal_compact", None)
            if compact is not None and self.compact_selection:
                with stage("select"):
                    chosen = self._select_compact(compact, encoded, decision_step, coin_key,
                                                  game_label, sim)
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
        types = [la.action.get("type") for la in legal]
        pick, band = self._choose(priors, actors, np.array([t == "end_turn" for t in types]),
                                  np.array([t == "attack" for t in types]), coin_key)
        if band is not None:
            self._log_band(game_label, coin_key, band, sim, legal[band[0]].action, legal[band[1]].action)
        return legal[pick].action

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
        """The game is over: its sides' memories and decision counts go
        (its coin log stays for `pop_coin_log`)."""
        for key in [k for k in self._memories if k[0] == game_label]:
            del self._memories[key]
        for key in [k for k in self._decisions if k[0] == game_label]:
            del self._decisions[key]
        self._undo.pop(game_label, None)
        self._last_decision.pop(game_label, None)

    def drop_last_pending(self, game_label: str) -> bool:
        """The last decision was refused (a recruit the engine bounces) and
        is decided again: the side's memory goes back to what it was
        before it. The engine records nothing for a refused recruit
        (`menu_handler::do_recruit`, src/menu_events.cpp:350-370, 1.18.4),
        so the corpus holds no position there. A band decision refused
        this way is marked `refused` in the coin log: it played nothing."""
        log = self._coin_log.get(game_label)
        last = self._last_decision.get(game_label)
        if log and last is not None and (log[-1]["side"], log[-1]["i"]) == last:
            log[-1]["refused"] = True
        undo = self._undo.pop(game_label, None)
        if undo is not None:
            key, before = undo
            if before is None:
                self._memories.pop(key, None)
            else:
                self._memories[key] = before
        return True
