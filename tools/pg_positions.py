"""The position of every logged decision of a game record, replayed through
the Rust core (docs/training_design_20261019/build_contracts.md §1), for
the policy-gradient pass (tools/pg_pass.py) and the network evaluator
(tools/pg_eval.py).

A decision's position is the core before the first command it emitted
(its `cmd`), with the recruit rejections that happened before it at that
point applied, encoded as the player encodes (the recipe's encoding, the
core's legality masks). The record keeps each rejection at the index of the
command it preceded (`WesnothSim.reject_recruit_hex`); the decisions at one
command index sit in log order:

- a refused decision emitted nothing: it sits where the next decision that
  emitted a command sits (the side decided again at once), or at the end of
  the commands. It is a shadow row: in play its memory write was undone, so
  the trainer reads the memory from before it and writes nothing;
- a refused recruit left one rejection at its index (the match loop rejects
  the ordered hex), so the decisions after it at that index see it;
- the decision that emitted the command decides last at its index; a
  recruit the engine relocated left one rejection more there, on the
  ordered hex, which only later decisions see. Its action is the log's,
  the ordered hex, not the command's.

So a decision sees, of its index's rejections, as many as the refused
recruits before it at that index. A record whose rejections do not fit
that count, whose decisions fall out of order, or whose logged action is
not legal at its replayed position is refused (`PositionError`); a decision
whose kind is not its command's (the simulator ended a turn for an
untranslatable action) or whose legal count is not the logged one is
counted.

    python tools/pg_positions.py --games DIR [DIR ...] --labels labels.jsonl --workers 30
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import multiprocessing as mp
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from wesnoth_ai.pg_records import ActionKey, Decision, decisions_of, side_counts  # noqa: E402
from wesnoth_ai.sequence_streams import GameSide  # noqa: E402
from wesnoth_ai.server_priors import KIND_ATTACK, KIND_END_TURN, KIND_RECRUIT, PackedMasks  # noqa: E402

log = logging.getLogger("pg_positions")

RECORD_SUFFIX = ".game.jsonl.gz"
# The command a decision of each kind emits first (an attack may emit the
# simulator's approach move before it).
KIND_COMMANDS = {KIND_ATTACK: ("attack", "move"), 1: ("move",), KIND_RECRUIT: ("recruit",),
                 KIND_END_TURN: ("end_turn",)}


class PositionError(ValueError):
    """A record's decision log does not replay onto its commands."""


@dataclass
class PGPosition:
    """One logged decision at its replayed position."""
    raw: object                     # RawEncoded: observation, true faction and masks left out
    masks: PackedMasks              # the legality masks the player decided under
    action: ActionKey
    shadow: bool                    # refused: reads the memory from before it, writes nothing
    index: int                      # its index in the record's decision log
    logp: float                     # the logged log probability of its action


@dataclass
class GamePositions:
    game: str
    sides: Dict[int, List[PGPosition]] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Where each decision sits
# ---------------------------------------------------------------------------

def decision_slots(decisions: Sequence[Decision], rejections: Sequence[Sequence[int]],
                   n_commands: int) -> List[Tuple[int, int]]:
    """Per decision, (the command index it sits at, how many of that
    index's rejections it sees) (module docstring)."""
    slots = [0] * len(decisions)
    following = n_commands
    for i in range(len(decisions) - 1, -1, -1):
        d = decisions[i]
        if not d.refused:
            if not d.cmd < following:
                raise PositionError(f"decision {i} names command {d.cmd}, not before the next decision's "
                                    f"{following} (of {n_commands} commands)")
            following = d.cmd
        slots[i] = following
    by_slot: Dict[int, int] = Counter(int(k) for k, _x, _y in rejections)
    seen: Counter = Counter()
    out = []
    refused_recruits: Counter = Counter()
    last_recruit: Dict[int, bool] = {}
    for d, k in zip(decisions, slots):
        out.append((k, seen[k]))
        if d.refused and d.kind == KIND_RECRUIT:
            seen[k] += 1
            refused_recruits[k] += 1
        elif not d.refused:
            last_recruit[k] = d.kind == KIND_RECRUIT
    for k, n in by_slot.items():
        extra = n - refused_recruits[k]
        if extra not in (0, 1) or (extra == 1 and not last_recruit.get(k, False)):
            raise PositionError(f"command index {k} holds {n} recruit rejections for {refused_recruits[k]} "
                                f"refused recruits and a {'recruit' if last_recruit.get(k) else 'non-recruit'} "
                                f"decision")
    return out


def _bit(packed_row, t: int) -> bool:
    return 0 <= t < 8 * len(packed_row) and bool((int(packed_row[t // 8]) >> (7 - t % 8)) & 1)


def action_is_legal(masks: PackedMasks, action: ActionKey) -> bool:
    """Whether the masks allow the action: its actor, its type, its target
    and its weapon."""
    a, kind, t, w = action
    U, R = masks.n_units, masks.n_recruits
    if not 0 <= a < U + R + 1 or not masks.actor_mask[a]:
        return False
    if kind == KIND_END_TURN:
        return a == U + R
    if kind == KIND_RECRUIT:
        return U <= a < U + R and _bit(masks.union_valid[a], t)
    if a >= U or not masks.type_valid[a, kind]:
        return False
    if kind == KIND_ATTACK:
        return _bit(masks.attack_valid[a], t) and 0 <= w < int(masks.n_attacks[a])
    return _bit(masks.move_valid[a], t)


# ---------------------------------------------------------------------------
# The replay
# ---------------------------------------------------------------------------

def positions_of(rec: dict, type_to_id: Dict[str, int], faction_to_id: Dict[str, int],
                 game: Optional[str] = None, verify: bool = True) -> GamePositions:
    """Every logged decision of `rec` at its position, by side, in log
    order. With `verify`, the record's fingerprints are checked as
    `game_record.walk` checks them."""
    from tools.game_record import _check, start_core
    decisions = decisions_of(rec)
    commands = rec["commands"]
    rejections = [(int(k), int(x), int(y)) for k, x, y in rec.get("rejections", ())]
    slots = decision_slots(decisions, rejections, len(commands))
    by_slot: Dict[int, List[Tuple[int, int]]] = {}
    for k, x, y in rejections:
        by_slot.setdefault(k, []).append((x, y))
    digests = {int(k): d for k, d in rec.get("turn_digests", ())} if verify else {}
    out = GamePositions(game=game or str(rec.get("game_label")), sides={1: [], 2: []})
    counts: Counter = Counter()
    cs = start_core(rec, verify=verify)
    applied = {"commands": 0}
    seen: Counter = Counter()

    def reject_up_to(k: int, n: int) -> None:
        for x, y in by_slot.get(k, [])[seen[k]:n]:
            cs.core.add_recruit_rejected(x, y)
        seen[k] = max(seen[k], n)

    def advance_to(k: int) -> None:
        while applied["commands"] < k:
            c = applied["commands"]
            reject_up_to(c, len(by_slot.get(c, ())))
            cs.apply_command(list(commands[c]))
            _check(rec, cs, digests.get(c), f"after command {c} ({commands[c][0]})")
            applied["commands"] += 1

    for i, (d, (k, r)) in enumerate(zip(decisions, slots)):
        advance_to(k)
        reject_up_to(k, r)
        out.sides[d.side].append(_position(cs, d, i, type_to_id, faction_to_id, counts, out.game))
        if not d.refused and commands[d.cmd][0] not in KIND_COMMANDS.get(d.kind, ()):
            counts["command_mismatch"] += 1
    advance_to(len(commands))
    reject_up_to(len(commands), len(by_slot.get(len(commands), ())))
    if "final_side" in rec:
        cs.core.set_global_int("current_side", int(rec["final_side"]))
    _check(rec, cs, rec.get("final_digest") if verify else None, "at the end")
    counts.update(decisions=len(decisions), shadow=sum(d.refused for d in decisions),
                  side1=len(out.sides[1]), side2=len(out.sides[2]))
    out.counts = dict(counts)
    return out


def _position(cs, d: Decision, i: int, type_to_id, faction_to_id, counts: Counter, game: str) -> PGPosition:
    """Decision i at the core's position, checked against the log."""
    from tools.preencode_sequences import ENCODING
    from wesnoth_ai.server_priors import _legal_capacity, core_packed_masks
    from wesnoth_ai.model_output import MAX_ATTACKS
    side = int(cs.core.current_side)
    if side != d.side:
        raise PositionError(f"{game}: decision {i} is side {d.side}'s, and side {side} is to move there")
    raw = cs.encode_raw(type_to_id=type_to_id, faction_to_id=faction_to_id, masks=True, **ENCODING)
    masks = core_packed_masks(raw)
    if not action_is_legal(masks, d.action):
        raise PositionError(f"{game}: decision {i}'s action {d.action} is not legal at its replayed position")
    if _legal_capacity(masks, MAX_ATTACKS) != d.n:
        counts["legal_count_mismatch"] += 1
    # What the trainer stages, as the sequence records store it
    # (preencode_sequences.sequence_position): the true state's observation
    # and the true enemy faction stay out.
    raw = dataclasses.replace(raw, observation=None, their_faction_id=0, core_masks=None)
    return PGPosition(raw=raw, masks=masks, action=d.action, shadow=d.refused, index=i, logp=d.logp)


# ---------------------------------------------------------------------------
# The records and their loader
# ---------------------------------------------------------------------------

def read_record(path: Path) -> dict:
    from tools.game_record import read_records
    records = list(read_records(path))
    if len(records) != 1:
        raise PositionError(f"{path} holds {len(records)} records, not one")
    return records[0]


class RecordIndex:
    """The game records of match directories, by game name: the record's
    stem (`<stem>.game.jsonl.gz`), or `<directory>__<stem>`
    (tools/own_trajectories.py's names). A stem two directories share
    answers only to the second form."""

    def __init__(self, dirs: Iterable[Path]):
        self.by_name: Dict[str, Path] = {}
        stems: Dict[str, List[Path]] = {}
        for d in dirs:
            for p in sorted(Path(d).glob("*" + RECORD_SUFFIX)):
                stem = p.name[:-len(RECORD_SUFFIX)]
                self.by_name[f"{Path(d).name}__{stem}"] = p
                stems.setdefault(stem, []).append(p)
        for stem, paths in stems.items():
            if len(paths) == 1:
                self.by_name.setdefault(stem, paths[0])

    def path(self, game: str) -> Path:
        p = self.by_name.get(game)
        if p is None:
            raise PositionError(f"no game record answers to {game!r}")
        return p


def _side_counts(path: str) -> Dict[int, int]:
    return side_counts(decisions_of(read_record(Path(path))))


def side_lengths(paths: Dict[str, Path], workers: int = 0) -> Dict[GameSide, int]:
    """Each game-side's logged decisions, refused ones included, the logs
    read on `workers` processes (0: here)."""
    games = list(paths)
    if workers > 0:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
            counts = list(pool.map(_side_counts, [str(paths[g]) for g in games], chunksize=16))
    else:
        counts = [_side_counts(str(paths[g])) for g in games]
    return {GameSide(g, side): n for g, c in zip(games, counts) for side, n in c.items()}


_W: Dict = {}


def _worker_init(type_to_id, faction_to_id, verify) -> None:
    _W.update(type_to_id=type_to_id, faction_to_id=faction_to_id, verify=verify)


def _build(game: str, path: str) -> GamePositions:
    return positions_of(read_record(Path(path)), _W["type_to_id"], _W["faction_to_id"], game=game,
                        verify=_W["verify"])


class PositionLoader:
    """Games' positions built ahead on `workers` processes (0: on the
    calling thread, when asked) and kept until `keep_only` drops them."""

    def __init__(self, paths: Dict[str, Path], type_to_id: Dict[str, int], faction_to_id: Dict[str, int],
                 workers: int = 0, verify: bool = True):
        self.paths = {g: Path(p) for g, p in paths.items()}
        self.vocab = (type_to_id, faction_to_id)
        self.verify = bool(verify)
        self._games: Dict[str, object] = {}
        # The replay's counts, summed over the games built (each once).
        self.counts: Counter = Counter()
        self._counted: set = set()
        self._pool = None
        if workers > 0:
            self._pool = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_worker_init,
                                             initargs=(type_to_id, faction_to_id, self.verify))
        self.wait_seconds = 0.0

    def prefetch(self, games: Iterable[str]) -> None:
        if self._pool is None:
            return
        for g in games:
            if g not in self._games:
                self._games[g] = self._pool.submit(_build, g, str(self.paths[g]))

    def game(self, game: str) -> GamePositions:
        got = self._games.get(game)
        if got is None:
            if self._pool is not None:
                self.prefetch([game])
                got = self._games[game]
            else:
                got = self._games[game] = positions_of(read_record(self.paths[game]), *self.vocab, game=game,
                                                       verify=self.verify)
        if not isinstance(got, GamePositions):
            t0 = time.perf_counter()
            got = self._games[game] = got.result()
            self.wait_seconds += time.perf_counter() - t0
        if game not in self._counted:
            self._counted.add(game)
            self.counts.update(got.counts)
        return got

    def side(self, game: str, side: int) -> List[PGPosition]:
        return self.game(game).sides[side]

    def keep_only(self, games: Iterable[str]) -> None:
        keep = set(games)
        for g in [g for g in self._games if g not in keep]:
            del self._games[g]

    def release(self, game: str) -> None:
        self._games.pop(game, None)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(cancel_futures=True)
            self._pool = None


# ---------------------------------------------------------------------------
# The command line: every labelled game replayed, its counts summed
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Replay every labelled game's decisions and count them (I-count).")
    ap.add_argument("--games", type=Path, nargs="+", required=True, help="match directories")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True, help="the network whose vocabularies encode")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from tools.preencode_corpus import vocab_from_checkpoint
    from wesnoth_ai.pg_records import read_labels
    games = sorted({r.game for r in read_labels(args.labels)})
    index = RecordIndex(args.games)
    loader = PositionLoader({g: index.path(g) for g in games}, *vocab_from_checkpoint(args.checkpoint),
                            workers=args.workers)
    totals: Counter = Counter()
    errors: Dict[str, str] = {}
    ahead = max(1, 4 * args.workers)
    for i, g in enumerate(games):
        loader.prefetch(games[i:i + ahead])
        try:
            totals.update(loader.game(g).counts)
        except Exception as e:                # noqa: BLE001 - every game's verdict is reported
            errors[g] = f"{type(e).__name__}: {e}"[:300]
        loader.release(g)
    loader.close()
    print(json.dumps({"games": len(games), "errors": errors, "counts": dict(totals)}, indent=1))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
