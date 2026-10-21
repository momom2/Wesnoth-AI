"""The policy-gradient pass's inputs as typed records
(docs/training_design_20261019/build_contracts.md §1 and §2): a game
record's decision log and the label rows of `labels.jsonl`.

A decision is one sampled action of either side, in play order, refused
ones included (the simulator refused the action and the side decided
again): its side, the index of the first command it emitted (-1 when
refused), the action's factored key, and what the network's raw law said
of it when it was sampled. A label row is one game-side's weight R, its
split and whether it is held out.

JSON has no NaN: a `p_att_actor` written as null reads as NaN.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# (actor, kind, target, weapon), kind as in `server_priors` (0 attack, 1
# move, 2 recruit, 3 end_turn), target -1 for end_turn, weapon -1 unless an
# attack: `own_loss.ActionKey`.
ActionKey = Tuple[int, int, int, int]
N_SPLITS = 8
N_LUCK_SUMS = 7
HALVES = ("A", "B")


class RecordFormatError(ValueError):
    """A decision log or a label row is not in the contract's format."""


def _float(d: dict, key: str) -> float:
    v = d[key]
    return math.nan if v is None else float(v)


@dataclass(frozen=True)
class Decision:
    """One logged decision (build_contracts.md §1)."""
    side: int
    cmd: int                        # the first command it emitted; -1 when refused
    refused: bool
    action: ActionKey
    n: int                          # legal actions
    mass: float                     # the joint priors' sum over the legal list
    logp: float                     # log of the sampled action's prior / mass
    H: float                        # the law's entropy, nats
    p_end: float
    p_att: float
    p_rec: float
    p_att_actor: float              # p(attack | the sampled actor); NaN when it has no attack
    cdf: Tuple[float, float]
    v: float                        # the value head's expectation, the deciding side's frame

    @classmethod
    def from_dict(cls, d: dict) -> "Decision":
        try:
            action = tuple(int(x) for x in d["action"])
            cdf = tuple(float(x) for x in d["cdf"])
            dec = cls(side=int(d["side"]), cmd=int(d["cmd"]), refused=bool(d["refused"]),
                      action=action, n=int(d["n"]), mass=_float(d, "mass"), logp=_float(d, "logp"),
                      H=_float(d, "H"), p_end=_float(d, "p_end"), p_att=_float(d, "p_att"),
                      p_rec=_float(d, "p_rec"), p_att_actor=_float(d, "p_att_actor"), cdf=cdf,
                      v=_float(d, "v"))
        except (KeyError, TypeError, ValueError) as e:
            raise RecordFormatError(f"a decision entry is not in the contract's format ({e!r}): {d!r}") from e
        if len(action) != 4 or len(cdf) != 2 or dec.side not in (1, 2):
            raise RecordFormatError(f"a decision entry is not in the contract's format: {d!r}")
        if dec.refused != (dec.cmd < 0):
            raise RecordFormatError(f"a decision is refused exactly when it names no command: {d!r}")
        return dec

    @property
    def kind(self) -> int:
        return self.action[1]

    def to_dict(self) -> dict:
        return {"side": self.side, "cmd": self.cmd, "refused": self.refused, "action": list(self.action),
                "n": self.n, "mass": self.mass, "logp": self.logp, "H": self.H, "p_end": self.p_end,
                "p_att": self.p_att, "p_rec": self.p_rec,
                "p_att_actor": None if math.isnan(self.p_att_actor) else self.p_att_actor,
                "cdf": list(self.cdf), "v": self.v}


def decisions_of(rec: dict) -> List[Decision]:
    """The record's decision log, in play order."""
    if "decisions" not in rec:
        raise RecordFormatError(f"{rec.get('game_label')}: the record carries no decision log")
    return [Decision.from_dict(d) for d in rec["decisions"]]


def side_counts(decisions: Iterable[Decision]) -> Dict[int, int]:
    """Decisions per side, refused ones included."""
    out = {1: 0, 2: 0}
    for d in decisions:
        out[d.side] += 1
    return out


@dataclass(frozen=True)
class LabelRow:
    """One game-side's label (build_contracts.md §2): R = z - b - beta_S in
    the side's frame."""
    game: str
    side: int
    split: int
    heldout: bool
    half: Optional[str]
    z: float
    capped: bool
    cell: str
    S: Tuple[float, ...]
    b: float
    beta_S: float
    R: float

    @classmethod
    def from_dict(cls, d: dict) -> "LabelRow":
        try:
            row = cls(game=str(d["game"]), side=int(d["side"]), split=int(d["split"]), heldout=bool(d["heldout"]),
                      half=None if d.get("half") is None else str(d["half"]), z=float(d["z"]),
                      capped=bool(d["capped"]), cell=str(d["cell"]), S=tuple(float(x) for x in d["S"]),
                      b=float(d["b"]), beta_S=float(d["beta_S"]), R=float(d["R"]))
        except (KeyError, TypeError, ValueError) as e:
            raise RecordFormatError(f"a label row is not in the contract's format ({e!r}): {d!r}") from e
        problems = []
        if row.side not in (1, 2):
            problems.append("side")
        if not 0 <= row.split < N_SPLITS:
            problems.append("split")
        if len(row.S) != N_LUCK_SUMS:
            problems.append("S")
        if (row.half is not None) and (row.half not in HALVES or not row.heldout):
            problems.append("half")
        if not math.isfinite(row.R):
            problems.append("R")
        if problems:
            raise RecordFormatError(f"a label row's {', '.join(problems)} is not the contract's: {d!r}")
        return row


def read_labels(path: Path) -> List[LabelRow]:
    """The rows of a labels.jsonl, each game-side once."""
    rows, seen = [], set()
    for k, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = LabelRow.from_dict(json.loads(line))
        if (row.game, row.side) in seen:
            raise RecordFormatError(f"{path}:{k}: game {row.game} side {row.side} is labelled twice")
        seen.add((row.game, row.side))
        rows.append(row)
    return rows


def labelled_games(rows: Iterable[LabelRow], heldout: bool) -> Dict[str, Dict[int, LabelRow]]:
    """The games whose two sides are both labelled, held out or not, by
    name: each side's row. A game labelled on one side only, or whose sides
    disagree on the split or the hold-out, is refused."""
    games: Dict[str, Dict[int, LabelRow]] = {}
    for r in rows:
        if r.heldout == heldout:
            games.setdefault(r.game, {})[r.side] = r
    for game, sides in games.items():
        if set(sides) != {1, 2}:
            raise RecordFormatError(f"game {game} is labelled on side(s) {sorted(sides)} only")
        if sides[1].split != sides[2].split:
            raise RecordFormatError(f"game {game}'s sides are labelled with splits {sides[1].split} and "
                                    f"{sides[2].split}")
    return games
