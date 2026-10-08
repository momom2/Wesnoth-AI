"""Training positions of the step-1 critics (docs/selfplay_program_20261008.md,
"Step 1").

A game gives its positions from every decision point and every side-turn
start of a player side (`eligible_points`), at most MAX_POSITIONS_PER_GAME
of them drawn with a generator seeded by the game's own seed
(`sample_points`). Each position carries

  z     the game's outcome signed to the side to move: +1 won, -1 lost;
  aux   the mover's HP margin (`hp_margin`) right after its next init_side,
        NaN when the game ends first;

and two encodings of the side to move, both through the Rust core with the
parity recipe's switches (`ENCODING`), neither carrying a memory:

  true  the position with fog off, on a fork of the core: every unit of
        both sides, the enemy's gold, income and upkeep;
  obs   the side's observation as the game was played (its fog, its
        sighting record).

The source is a whole-game record (tools/game_record.py: `walk_core`
verifies its fingerprints) or a corpus replay (`corpus_walk`). The caller
leaves out games whose outcome is not a player's win: a game stopped at the
turn cap has no label.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math
import pickle
import random
import zlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from wesnoth_ai.classes import PLAYER_SIDES

MAX_POSITIONS_PER_GAME = 24
VIEWS = ("true", "obs")
# The commands a player decides (tools/replay_dataset.PAIRED_KINDS, plus a
# recall, which a corpus replay can hold).
DECISION_KINDS = frozenset({"move", "attack", "recruit", "recall", "end_turn"})
# The recipe's encoding (tools/preencode_sequences.ENCODING, which parity3 reads).
ENCODING = {"relevant_set": True, "fog_hides_enemy_villages": True, "terrain_multi_hot": True,
            "observation_parity": True, "relevant_set_version": 2}
HOLDOUT_FRACTION = 0.05


@dataclass(frozen=True)
class Point:
    """A position a game offers: the one before command `index`, with
    `side` to move; `kind` is "turn_start" when an init_side of the side
    put it there, else "decision"."""
    index: int
    side: int
    kind: str


@dataclass
class CriticPosition:
    index: int
    side: int
    turn: int
    kind: str
    z: float
    aux: float = math.nan


@dataclass
class GamePositions:
    """One game's sampled positions and their encodings, packed
    (`pack_raw`), view by view in the positions' order."""
    positions: List[CriticPosition]
    raws: Dict[str, List[bytes]]
    eligible: int
    seed: int


class PositionError(Exception):
    """A game's command stream and its rebuilt core disagree on who moves."""


# ---------------------------------------------------------------------
# Which positions
# ---------------------------------------------------------------------

def eligible_points(commands: Sequence[list], engine_issued: Iterable[int] = ()) -> List[Point]:
    """Every position a player side faces in a command stream: before each
    command it decides (DECISION_KINDS, under a player side's init_side,
    not made by the engine), and right after each init_side of a player
    side. A position that is both is one point, a turn start."""
    engine = set(int(i) for i in engine_issued)
    points: Dict[int, Point] = {}
    side = 0
    n = len(commands)
    for k, cmd in enumerate(commands):
        kind = cmd[0] if cmd else ""
        if kind == "init_side":
            side = int(cmd[1])
            if side in PLAYER_SIDES and k + 1 < n:
                points[k + 1] = Point(k + 1, side, "turn_start")
            continue
        if side in PLAYER_SIDES and kind in DECISION_KINDS and k not in engine and k not in points:
            points[k] = Point(k, side, "decision")
    return [points[k] for k in sorted(points)]


def game_seed(key: str) -> int:
    """A seed for a game without one of its own (a corpus replay): 63 bits
    of the SHA-256 of its key."""
    return int.from_bytes(hashlib.sha256(f"critic_positions:{key}".encode("utf-8")).digest()[:8], "big") >> 1


def sample_points(points: Sequence[Point], seed: int,
                  cap: int = MAX_POSITIONS_PER_GAME) -> List[Point]:
    """At most `cap` of `points`, drawn without replacement by a generator
    seeded with `seed`, in game order."""
    if len(points) <= cap:
        return list(points)
    picks = random.Random(int(seed)).sample(range(len(points)), cap)
    return [points[i] for i in sorted(picks)]


def split_of(key: str) -> Tuple[str, float]:
    """("train" or "holdout", u): the game's side of the 95/5 split and a
    uniform draw in [0, 1) that orders the training games for the size
    curve (the first quarter of M's are the ones with u < 0.25), both from
    the SHA-256 of its key."""
    h = hashlib.sha256(f"critic_split:{key}".encode("utf-8")).digest()
    holdout = int.from_bytes(h[:8], "big") / 2.0 ** 64 < HOLDOUT_FRACTION
    return ("holdout" if holdout else "train"), int.from_bytes(h[8:16], "big") / 2.0 ** 64


# ---------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------

def hp_margin(gs, mover: int) -> int:
    """The mover's units' total hit points minus every other side's: the
    turn-value experiment's HP margin (tools/playout_reads.py on branch
    exp/turn-value), which its benchmark records as `hp_margin_post`."""
    return sum(u.current_hp if u.side == mover else -u.current_hp for u in gs.map.units)


def signed_outcome(winner: int, side: int) -> float:
    return 1.0 if int(winner) == int(side) else -1.0


# ---------------------------------------------------------------------
# Encodings
# ---------------------------------------------------------------------

def true_state(cs):
    """A fork of the core with fog off: what the side to move would see if
    nothing were hidden. The core it came from is left as it was."""
    fork = cs.fork()
    g = dict(fork.core.globals_export())
    g["fog_on"] = False
    fork.core.set_globals(g)
    return fork


def encode_view(cs, view: str, type_to_id: Dict[str, int], faction_to_id: Dict[str, int]):
    """The side to move's encoding under `view` ("true" or "obs"), with the
    observation record and the opponent's true faction left out, as the
    sequence pre-encoder stores them (the parity network reads the
    faction's posterior)."""
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}, got {view!r}")
    source = true_state(cs) if view == "true" else cs.fork()
    raw = source.encode_raw(type_to_id=type_to_id, faction_to_id=faction_to_id, **ENCODING)
    return dataclasses.replace(raw, observation=None, their_faction_id=0)


def pack_raw(raw) -> bytes:
    return zlib.compress(pickle.dumps(raw, protocol=pickle.HIGHEST_PROTOCOL), 1)


def unpack_raw(blob: bytes):
    from wesnoth_ai import unpickle
    return unpickle.loads(zlib.decompress(blob))


# ---------------------------------------------------------------------
# Walking a game
# ---------------------------------------------------------------------

def corpus_walk(data: dict, end: Optional[list] = None) -> Iterator[Tuple[int, Any, list]]:
    """(index, the core before the command, command) over a corpus
    replay's commands on the Rust core (`replay_dataset.record_core`), as
    tools/game_record.walk_core walks a record; the final core is appended
    to `end`."""
    from tools.replay_dataset import record_core
    cs = record_core(data)
    for k, cmd in enumerate(data.get("commands", [])):
        yield k, cs, cmd
        cs.apply_command(list(cmd))
    if end is not None:
        end.append(cs)


def collect_positions(steps: Iterator[Tuple[int, Any, list]], end: list, chosen: Sequence[Point],
                      winner: int, encode: Callable[[Any, str], bytes],
                      views: Sequence[str] = VIEWS) -> Tuple[List[CriticPosition], Dict[str, List[bytes]]]:
    """Walk `steps` (a `walk_core` or `corpus_walk` generator whose final
    core lands in `end`) and take the `chosen` points: each one's label
    and its encodings (`encode(core, view)`), and, once the mover's next
    turn has begun, its HP margin there."""
    wanted = {p.index: p for p in chosen}
    positions: List[CriticPosition] = []
    raws: Dict[str, List[bytes]] = {v: [] for v in views}
    pending: Dict[int, List[CriticPosition]] = {s: [] for s in PLAYER_SIDES}
    begun: Optional[int] = None                       # the side whose init_side was the last command

    def settle(cs, side: int) -> None:
        if pending[side]:
            margin = hp_margin(cs.to_state(), side)
            for pos in pending[side]:
                pos.aux = float(margin)
            pending[side] = []

    for k, cs, cmd in steps:
        if begun is not None:
            settle(cs, begun)
        point = wanted.get(k)
        if point is not None:
            side = int(cs.core.current_side)
            if side != point.side:
                raise PositionError(f"command {k}: the stream says side {point.side} moves, "
                                    f"the core says side {side}")
            pos = CriticPosition(index=k, side=side, turn=int(cs.core.turn_number), kind=point.kind,
                                 z=signed_outcome(winner, side))
            positions.append(pos)
            for view in views:
                raws[view].append(encode(cs, view))
            pending[side].append(pos)
        begun = int(cmd[1]) if cmd and cmd[0] == "init_side" and int(cmd[1]) in PLAYER_SIDES else None
    if begun is not None and end:
        settle(end[0], begun)
    if len(positions) != len(chosen):
        raise PositionError(f"{len(chosen)} points chosen, {len(positions)} reached")
    return positions, raws


def record_positions(rec: Dict[str, Any], *, type_to_id: Dict[str, int], faction_to_id: Dict[str, int],
                     cap: int = MAX_POSITIONS_PER_GAME, views: Sequence[str] = VIEWS) -> GamePositions:
    """The sampled positions of a whole-game record (a match game), its
    fingerprints verified on the way; the game's seed is its `seed`."""
    from tools.game_record import walk_core
    seed = int(rec["seed"]) if rec.get("seed") is not None else game_seed(str(rec.get("game_label")))
    points = eligible_points(rec["commands"])
    chosen = sample_points(points, seed, cap)
    end: list = []
    positions, raws = collect_positions(walk_core(rec, True, end), end, chosen, int(rec["winner"]),
                                        _encoder(type_to_id, faction_to_id), views)
    return GamePositions(positions=positions, raws=raws, eligible=len(points), seed=seed)


def corpus_positions(data: dict, file: str, winner: int, *, type_to_id: Dict[str, int],
                     faction_to_id: Dict[str, int], cap: int = MAX_POSITIONS_PER_GAME,
                     views: Sequence[str] = VIEWS) -> GamePositions:
    """The sampled positions of a corpus replay whose winner is `winner`;
    the commands the engine made under a player side are no decision
    (`replay_dataset.engine_issued_of`)."""
    from tools.replay_dataset import engine_issued_of
    seed = game_seed(file)
    points = eligible_points(data.get("commands", []), engine_issued_of(data))
    chosen = sample_points(points, seed, cap)
    end: list = []
    positions, raws = collect_positions(corpus_walk(data, end), end, chosen, int(winner),
                                        _encoder(type_to_id, faction_to_id), views)
    return GamePositions(positions=positions, raws=raws, eligible=len(points), seed=seed)


def _encoder(type_to_id: Dict[str, int], faction_to_id: Dict[str, int]) -> Callable[[Any, str], bytes]:
    return lambda cs, view: pack_raw(encode_view(cs, view, type_to_id, faction_to_id))
