"""Near-tie coins at the argmax decode, and the per-game attack coin
(docs/near_tie_coins_prereg_20261020.md; the design is
docs/training_design_20261019/current_design.md, sections 3.1 and 3.7).

A near-tie coin. At a decision of the raw player at argmax, the joint priors
the argmax compares (after the end_turn offset and any attack offset,
`RawPolicyPlayer._choose`) give the argmax a0, the runner-up a1 and their
log-prior gap G = log p(a0) - log p(a1) >= 0 (`near_tie`). When G < w, the
band width, the decision is a band decision: a coin D, 1 with probability
P, plays a1 when it lands 1 and a0 otherwise. end_turn is a candidate like
any other action.

The attack coin. One coin a game, 1 with probability 1/2, puts the attack
offset (every attack's logit raised by the offset) on player A when it lands
1 and on player B otherwise.

Each coin is a pure function of its key: the first 8 bytes of the SHA-256
of the key's text under the coin's own domain name, read as a fraction of
2**64 (`coin_uniform`). A near-tie coin's key is (game seed, the mover's
side, the mover's decision index in the game); the attack coin's is the
game seed. The dice are another SHA-256 stream (`wesnoth_sim.request_seed`,
salted with "elo:<seed>"), a coin never reads a prior, and distinct keys
give distinct inputs, so a coin is independent of the dice, of the
inference numerics and of every other coin, to the extent SHA-256's outputs
on distinct inputs are (tests/test_near_tie_coins.py measures the
frequencies and the correlations).
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, Optional, Tuple

import numpy as np

NEAR_TIE_DOMAIN = "near_tie_coin"
ATTACK_COIN_DOMAIN = "attack_coin"
# The attack coin's propensity: the hook lands on either player alike.
ATTACK_COIN_P = 0.5
# The action kinds as the compact arrays code them (wesnoth_ai.server_priors
# KIND_*), by name; a record logs the names.
KIND_CODES = {"attack": 0, "move": 1, "recruit": 2, "end_turn": 3}


def coin_uniform(domain: str, *key) -> float:
    """A draw in [0, 1) that is a pure function of (domain, key)."""
    text = ":".join([domain] + [str(k) for k in key])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2.0 ** 64


def near_tie_coin(seed: int, side: int, index: int, p: float) -> int:
    """The coin of decision `index` of `side` in the game on `seed`: 1 with
    probability `p`."""
    return int(coin_uniform(NEAR_TIE_DOMAIN, int(seed), int(side), int(index)) < p)


def attack_coin(seed: int) -> int:
    """The game's attack coin: 1 puts the attack offset on player A, 0 on
    player B."""
    return int(coin_uniform(ATTACK_COIN_DOMAIN, int(seed)) < ATTACK_COIN_P)


def near_tie(priors: np.ndarray) -> Optional[Tuple[int, int, float]]:
    """(a0, a1, G): the argmax of `priors` (the first maximum, as
    `raw_player.pick_index` takes it), the runner-up (the first maximum of
    the others) and G = log p(a0) - log p(a1), infinite when p(a1) is 0.
    None with fewer than two actions."""
    if len(priors) < 2:
        return None
    p = np.asarray(priors, dtype=np.float64)
    a0 = int(p.argmax())
    rest = p.copy()
    rest[a0] = -np.inf
    a1 = int(rest.argmax())
    if not p[a1] > 0.0:
        return a0, a1, math.inf
    return a0, a1, float(np.log(p[a0]) - np.log(p[a1]))


def _xy(position) -> list:
    return [int(position.x), int(position.y)]


def describe_action(action: Dict) -> Dict:
    """An action dict as JSON data: its kind, the acting unit's hex and the
    target hex (a move, an attack), the attack's weapon, the recruit's unit
    type and hex."""
    kind = action.get("type", "end_turn")
    out: Dict = {"kind": kind}
    if kind in ("move", "attack"):
        out["from"] = _xy(action["start_hex"])
        out["to"] = _xy(action["target_hex"])
        if kind == "attack":
            out["weapon"] = int(action.get("attack_index", 0))
    elif kind == "recruit":
        out["unit"] = str(action["unit_type"])
        out["to"] = _xy(action["target_hex"])
    return out
