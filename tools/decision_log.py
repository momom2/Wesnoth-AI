"""The decision log: one entry per decision of a game, in play order
(docs/training_design_20261019/build_contracts.md §1).

The raw player (`tools/raw_player.RawPolicyPlayer(log_decisions=True)`)
builds an entry at each decision from the legal list it chose from, as
compact arrays in the reference order (`server_priors.CompactActions`:
actor, kind, target, weapon, prior), and the game's record carries the
merged log as its `decisions` field (tools/game_record.py). An entry:

    side         the side that decided
    cmd          the index in the record's command list of the first
                 command the decision emitted (an approach move before its
                 attack belongs to the attack's decision); -1 when the
                 simulator refused it
    refused      the simulator refused the action; the decision counts
    action       [actor, kind, target, weapon], the keys of
                 wesnoth_ai/own_loss.py (-1 for an absent target or weapon)
    n            the number of legal actions
    mass         the sum of the network's joint priors over the legal list
    logp         log(prior / mass) of the sampled action: the network's raw
                 law, no temperature, no offset
    H            the entropy of that law, in nats
    p_end, p_att, p_rec
                 the law's mass on end_turn, on attacks, on recruits
    p_att_actor  p(attack | the sampled actor); NaN when the actor has no
                 legal attack
    cdf          [mass before, mass up to and with] the sampled action, in
                 the legal list's order
    v            the value head's expectation, in the deciding side's frame

While a game runs, each entry also carries `_at`, the length of the
command list when the decision was made: the decision emitted nothing
(it was refused) exactly when the next decision of the game starts at the
same length. `finish_log` turns `_at` into `cmd` and `refused`;
`merge_logs` puts the two players' entries in play order and drops `_at`.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from wesnoth_ai.server_priors import KIND_ATTACK, KIND_END_TURN, KIND_MOVE, KIND_RECRUIT

# The columns of an entry, in the contract's order.
ENTRY_KEYS = ("side", "cmd", "refused", "action", "n", "mass", "logp", "H", "p_end", "p_att",
              "p_rec", "p_att_actor", "cdf", "v")
KIND_OF_TYPE = {"attack": KIND_ATTACK, "move": KIND_MOVE, "recruit": KIND_RECRUIT,
                "end_turn": KIND_END_TURN}


def law_columns(actor: np.ndarray, kind: np.ndarray, prior: np.ndarray, pick: int) -> Dict:
    """The columns of an entry that read the law: n, mass, logp, H, the
    category masses, p_att_actor and cdf, for the action at index `pick`
    of the legal list (arrays in the list's order)."""
    prior = np.asarray(prior, dtype=np.float64)
    kind = np.asarray(kind)
    actor = np.asarray(actor)
    mass = float(prior.sum())
    q = prior / mass
    live = q > 0.0
    entropy = float(-(q[live] * np.log(q[live])).sum())
    cum = np.cumsum(q)
    before = float(cum[pick - 1]) if pick > 0 else 0.0
    mine = actor == actor[pick]
    attacks = mine & (kind == KIND_ATTACK)
    p_att_actor = float(q[attacks].sum() / q[mine].sum()) if attacks.any() and q[mine].sum() > 0 else math.nan
    with np.errstate(divide="ignore"):
        logp = float(np.log(q[pick]))
    return {"n": int(len(prior)), "mass": mass, "logp": logp, "H": entropy,
            "p_end": float(q[kind == KIND_END_TURN].sum()), "p_att": float(q[kind == KIND_ATTACK].sum()),
            "p_rec": float(q[kind == KIND_RECRUIT].sum()), "p_att_actor": p_att_actor,
            "cdf": [before, float(cum[pick])]}


def action_key(actor: int, kind: int, target: int, weapon: int) -> List[int]:
    """[actor, kind, target, weapon] as `own_loss._key` keys an action."""
    kind = int(kind)
    return [int(actor), kind, int(target) if kind != KIND_END_TURN else -1,
            int(weapon) if kind == KIND_ATTACK else -1]


def decision_entry(*, side: int, at: int, actor: np.ndarray, kind: np.ndarray, target: np.ndarray,
                   weapon: np.ndarray, prior: np.ndarray, pick: int, value: float) -> Dict:
    """The entry of a decision by `side` that chose the action at index
    `pick` of the legal list (compact arrays), made when the command list
    held `at` commands, with the value head's reading `value`."""
    entry = {"side": int(side), "cmd": int(at), "refused": False,
             "action": action_key(actor[pick], kind[pick], target[pick], weapon[pick])}
    entry.update(law_columns(actor, kind, prior, pick))
    entry["v"] = float(value)
    entry["_at"] = int(at)
    return entry


def finish_log(entries: Sequence[Dict], n_commands: int) -> List[Dict]:
    """One player's entries of a finished game with `cmd` and `refused`
    final: a decision is refused when the harness reported it
    (`refused` already set) or when it emitted no command, that is when
    the player's next decision, or the end of the game's `n_commands`
    commands, starts where it started. The entries keep `_at` for
    `merge_logs`."""
    out = [dict(e) for e in entries]
    for i, e in enumerate(out):
        nxt = out[i + 1]["_at"] if i + 1 < len(out) else int(n_commands)
        if e["refused"] or nxt == e["_at"]:
            e["refused"] = True
            e["cmd"] = -1
    return out


def merge_logs(logs: Sequence[Sequence[Dict]]) -> List[Dict]:
    """The finished logs of the players of one game as one log in play
    order, without `_at`. Decisions of different sides never share a
    start: a side's turn ends with its end_turn command."""
    tagged = [(e["_at"], p, i, e) for p, log in enumerate(logs) for i, e in enumerate(log)]
    tagged.sort(key=lambda t: (t[0], t[1], t[2]))
    for (at0, p0, _, e0), (at1, p1, _, e1) in zip(tagged, tagged[1:]):
        if at0 == at1 and (p0 != p1 or e0["side"] != e1["side"]):
            raise ValueError(f"two sides decided at command {at0}: the logs do not belong to one game")
    return [{k: e[k] for k in ENTRY_KEYS} for _, _, _, e in tagged]


def value_of(output) -> float:
    """The value head's expectation in a model output, NaN when it has
    none."""
    value = getattr(output, "value", None)
    if value is None:
        return math.nan
    return float(np.asarray(value.detach().float().cpu() if hasattr(value, "detach") else value).reshape(-1)[0])


def compact_columns(legal) -> Optional[Dict[str, np.ndarray]]:
    """A `LegalActionPrior` list as compact arrays (the list path of the
    raw player); None for an empty list."""
    if not legal:
        return None
    kinds = [KIND_OF_TYPE[la.action.get("type", "end_turn")] for la in legal]
    return {"actor": np.array([la.actor_idx for la in legal], dtype=np.int64),
            "kind": np.array(kinds, dtype=np.int64),
            "target": np.array([-1 if la.target_idx is None else la.target_idx for la in legal], dtype=np.int64),
            "weapon": np.array([-1 if la.weapon_idx is None else la.weapon_idx for la in legal], dtype=np.int64),
            "prior": np.array([la.prior for la in legal], dtype=np.float64)}
