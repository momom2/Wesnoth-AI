"""The frozen recruit rule at the decode, and the seat coin of match L
(docs/recruit_rule_l_prereg_20261021.md; tools/recruit_rule_fit.py fits and
freezes the rule, the design is
docs/training_design_20261019/current_design.md, section 4 rows "R0" and
"Box A: L").

The rule, Bd of round10_synthesis_checks.py. At a decision of the raw player
at argmax whose two best joint priors (after the end_turn offset) are
recruits of two different unit types u0 (the argmax) and u1 (the runner-up)
with a log-prior gap G under the rule's band, its predicted advantage of a1
over a0 for the mover is

    f = (w[k(u1)] - w[k(u0)]) + (m[k(u1)@e] - m[k(u0)@e]) + c (n0 - n1),

k(u) = "<mover faction>|<unit>", e the enemy faction, n0 and n1 how many
units of u0's and u1's types the mover recruited earlier in the game (the
recruit commands of its side so far), a key the fit never saw worth 0. The
player plays a1 iff f > 0 at its first `cap` such decisions of the side in
the game (each a switch) and a0 at every other.

The rule file holds the rule's content under "rule" and the SHA-256 of that
content's canonical JSON (`rule_hash`) under "hash": a file whose content
does not hash to its "hash" is refused.

The seat coin. One coin a game, 1 with probability 1/2, puts the rule on
player A when it lands 1 and on player B otherwise, keyed by the game seed
under its own domain (tools/coin_decode.coin_uniform), so it is independent
of the dice and of every other coin.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Tuple

RULE_SEAT_DOMAIN = "recruit_rule_seat"
# The seat coin's propensity: the rule lands on either player alike.
RULE_SEAT_P = 0.5
# The characters of the rule's hash its procedure tag and file name carry.
SHORT_HASH = 12


def rule_seat_coin(seed: int) -> int:
    """The game's seat coin: 1 puts the rule on player A, 0 on player B."""
    from tools.coin_decode import coin_uniform      # numpy: kept out of the torch-free driver's imports
    return int(coin_uniform(RULE_SEAT_DOMAIN, int(seed)) < RULE_SEAT_P)


def rule_hash(rule: Mapping) -> str:
    """The SHA-256 of a rule's content as canonical JSON (keys sorted, no
    spaces, floats as Python writes them, which read back exactly)."""
    text = json.dumps(rule, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def earlier_recruits(commands: Iterable, side: int, u0: str, u1: str) -> Tuple[int, int]:
    """(n0, n1): the recruits of types u0 and u1 that `side` made in
    `commands`, a game's commands so far (the core's lists: an
    ["init_side", side] opens each side's turn, ["recruit", unit, ...] is a
    recruit of the side to move)."""
    mover, n0, n1 = None, 0, 0
    for cmd in commands:
        if cmd[0] == "init_side":
            mover = int(cmd[1])
        elif cmd[0] == "recruit" and mover == side:
            n0 += cmd[1] == u0
            n1 += cmd[1] == u1
    return n0, n1


def live_features(game_state, commands: Iterable, u0: str, u1: str) -> Dict:
    """The rule's inputs at a decision, read off the live game: the mover
    (the side to move), its faction and the enemy's (the state's sides),
    the two types and the mover's earlier recruits of each (`commands`, the
    simulator's command history before the decision)."""
    side = int(game_state.global_info.current_side)
    if side not in (1, 2):
        raise ValueError(f"the recruit rule decides for the players' sides 1 and 2, not side {side}")
    n0, n1 = earlier_recruits(commands, side, u0, u1)
    return {"side": side, "fac": game_state.sides[side - 1].faction, "enemy": game_state.sides[2 - side].faction,
            "u0": u0, "u1": u1, "n0": n0, "n1": n1}


@dataclass(frozen=True)
class RecruitRule:
    """A frozen rule (see the module docstring)."""
    hash: str
    band: float
    cap: int
    units: Mapping[str, float]
    matchups: Mapping[str, float]
    count: float

    @property
    def short_hash(self) -> str:
        return self.hash[:SHORT_HASH]

    @classmethod
    def from_data(cls, data: Mapping) -> "RecruitRule":
        rule = data["rule"]
        got = rule_hash(rule)
        if got != data.get("hash"):
            raise ValueError(f"the rule's content hashes to {got}, its file says {data.get('hash')}: refused")
        if not 0.0 < float(rule["band"]) or int(rule["cap"]) < 1:
            raise ValueError(f"a rule needs a band above 0 and a cap of 1 or more, not {rule['band']}, {rule['cap']}")
        return cls(hash=got, band=float(rule["band"]), cap=int(rule["cap"]),
                   units={k: float(v) for k, v in rule["units"].items()},
                   matchups={k: float(v) for k, v in rule["matchups"].items()}, count=float(rule["count"]))

    @classmethod
    def load(cls, path) -> "RecruitRule":
        return cls.from_data(json.loads(Path(path).read_text(encoding="utf-8")))

    def value(self, fac: str, enemy: str, u0: str, u1: str, n0: int, n1: int) -> float:
        """f, the rule's predicted advantage of a1 over a0 for the mover."""
        k0, k1 = f"{fac}|{u0}", f"{fac}|{u1}"
        unit = self.units.get(k1, 0.0) - self.units.get(k0, 0.0)
        matchup = self.matchups.get(f"{k1}@{enemy}", 0.0) - self.matchups.get(f"{k0}@{enemy}", 0.0)
        return (unit + matchup) + self.count * float(n0 - n1)
