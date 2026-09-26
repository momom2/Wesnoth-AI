"""Game records: every generated game kept whole (user order 2026-09-24).

A record holds what rebuilds the game position by position: the
scenario setup (or, for a mid-game start, the corpus file, the digest
of its content and the cut), the build arguments, the simulator's
advancement channel, and the command list the simulator applied, in
the form `tools.replay_dataset._apply_command` replays, attack and
recruit seeds included. Recruit rejections, which change what the side
to move observes but apply no command, are kept beside it.

A record also carries fingerprints of the game it was written from:
the `state_digest` of the position each side's turn started from (the
neutral side's included) and of the final position. `walk` and
`rebuild` check them and raise `RecordMismatch` when the rebuild leaves
the played game.

Fight outcome distributions ride along as optional data, keyed by the
index of the attack command they describe (`outcomes`): training writes
those it computed while playing (the search's exact distribution of an
attack it played, the counter-weapon choice's strike tables), and an
analysis may add its own with `add_outcomes`. Nothing reads them to
rebuild the game.

    configure(directory, name)                            # once per process
    record_game(sim, setup, game_label=..., build={...}, players={...})
    rec = game_record(sim, setup, game_label=..., build={...}, players={...})
    GameRecordLog(path).write(rec)                        # one gzip member
    for rec in read_records(path): ...
    for k, gs, cmd in walk(rec): ...                      # the state before each command
    gs = rebuild(rec)                                     # the final state

A log is a sequence of gzip members, one record each, so a log grows as
games finish, a crash damages at most the member being written, and
any run of whole members is itself a valid log (the training box
uploads a log as such runs, scripts/hf_upload_loop.py).

Formats: 1 has no fingerprints and no corpus digest; it is read and
rebuilt without checks. 2 adds `turn_digests`, `final_digest` and the
mid-game `sha256`.
"""
from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import logging
import os
import sys
import threading
import zlib
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterator, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from wesnoth_ai.paths import REPO_ROOT  # noqa: E402

log = logging.getLogger("game_record")

FORMAT = 2


class RecordMismatch(Exception):
    """A record does not rebuild the game it was written from."""


# ---------------------------------------------------------------------
# Building a record
# ---------------------------------------------------------------------
def _setup_dict(setup) -> Dict[str, Any]:
    """The scenario setup as data: a ScenarioSetup's fields, or a
    mid-game start's corpus file and cut."""
    if isinstance(setup, tuple) and setup and setup[0] == "__midgame__":
        _, _gs, scenario_id, cut_turn, begin_side, provenance = setup
        return {"midgame": dict(provenance), "scenario_id": scenario_id,
                "cut_turn": int(cut_turn), "begin_side": int(begin_side)}
    return dataclasses.asdict(setup)


def game_record(sim, setup, *, game_label: str, build: Optional[Dict[str, Any]] = None,
                players: Optional[Dict[str, Any]] = None,
                extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The record of a finished (or abandoned) game played on `sim`.
    `build` holds the keyword arguments `build_scenario_gamestate` was
    given; `players` names who played each side and how."""
    from wesnoth_ai import __version__
    from wesnoth_ai.classes import state_digest
    from wesnoth_ai.constants import OBSERVATION_EPOCH
    history = sim.command_history
    outcomes = {str(k): rc.extras["outcomes"] for k, rc in enumerate(history)
                if rc.extras.get("outcomes")}
    gi = sim.gs.global_info
    rec = {
        "format": FORMAT,
        "code_version": __version__,
        "observation_epoch": OBSERVATION_EPOCH,
        "game_label": game_label,
        "setup": _setup_dict(setup),
        "build": dict(build or {}),
        "scenario_id": sim.scenario_id,
        "max_turns": int(sim.max_turns),
        "seed_salt": sim._seed_salt,
        "uniform_advancement": bool(getattr(gi, "_advance_uniform", False)),
        "players": dict(players or {}),
        "winner": int(sim.winner),
        "ended_by": sim.ended_by,
        "turns": int(gi.turn_number),
        # The side the finished game points at: the simulator points a
        # game a neutral side ended at the surviving player side, with
        # no command behind it.
        "final_side": int(gi.current_side),
        "commands": [list(rc.cmd) for rc in history],
        "rejections": [list(r) for r in getattr(sim, "recruit_rejections", ())],
        "turn_digests": [[int(k), d] for k, d in getattr(sim, "turn_digests", ())],
        "final_digest": state_digest(sim.gs),
        "outcomes": outcomes,
    }
    if extra:
        rec.update(extra)
    return rec


# ---------------------------------------------------------------------
# Outcome distributions as data
# ---------------------------------------------------------------------
def distribution_data(dist) -> Dict[str, Any]:
    """`combat_outcomes.OutcomeDistribution` as data: each outcome key
    (attacker hp, defender hp, slowed, poisoned and petrified flags of
    both, the unit types after any advancement) with its probability."""
    return {"attacker": dist.attacker_id, "defender": dist.defender_id,
            "outcomes": [[*key, p] for key, p in sorted(dist.probs.items(), key=lambda kv: -kv[1])]}


def strike_table_data(states: Dict[tuple, float]) -> List[list]:
    """A strike DP's final states (`combat_outcomes._strike_dp`) as
    data: each state key with its probability."""
    return [[*key, p] for key, p in sorted(states.items(), key=lambda kv: -kv[1])]


def note_search_outcomes(sim, policy, game_label: str, commands_before: int) -> None:
    """After `sim.step`: if `policy` searched and computed the exact
    outcome distribution of the attack it chose, put it into that
    attack's outcome data under "search".

    `commands_before` is the length of the command history before the
    step. The distribution is attached only when the step recorded that
    attack: one attack command, after the simulator's approach move if
    it made one. A refused step (nothing recorded), a truncated approach
    (a move alone) or an attack turned into end_turn (whose neutral turn
    may record attacks of its own) drops it. A policy without
    `pop_played_outcomes`, or whose decision procedure never fills it
    (TurnCommitPolicy, PlanTournamentPolicy), attaches nothing."""
    pop = getattr(policy, "pop_played_outcomes", None)
    if pop is None:
        return
    dist = pop(game_label)
    if dist is None:
        return
    added = sim.command_history[commands_before:]
    if [rc.kind for rc in added] not in (["attack"], ["move", "attack"]):
        return
    added[-1].extras.setdefault("outcomes", {})["search"] = distribution_data(dist)


def add_outcomes(rec: Dict[str, Any], command_index: int, source: str, data: Any) -> None:
    """Attach an outcome distribution to the attack at `command_index`
    under `source` (who computed it, e.g. "search", "counter_weapon",
    or an analysis's name)."""
    rec.setdefault("outcomes", {}).setdefault(str(command_index), {})[source] = data


# ---------------------------------------------------------------------
# Writing and reading
# ---------------------------------------------------------------------
class GameRecordLog:
    """Appends records to a log, each record one JSON line compressed
    as its own gzip member, so a crash damages at most the game being
    written."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, rec: Dict[str, Any]) -> None:
        line = (json.dumps(rec, separators=(",", ":")) + "\n").encode("utf-8")
        with open(self.path, "ab") as fh:
            fh.write(gzip.compress(line))
            fh.flush()
            os.fsync(fh.fileno())

    def __enter__(self) -> "GameRecordLog":
        return self

    def __exit__(self, *exc) -> None:
        return None


# ID1, ID2 and CM=deflate: the first bytes of every gzip member.
_GZIP_HEADER = b"\x1f\x8b\x08"
_READ_CHUNK = 1 << 16


def _find_header(fh: BinaryIO, pos: int) -> Optional[int]:
    """The offset of the first gzip header at or after `pos`, or None."""
    fh.seek(pos)
    tail = b""
    base = pos
    while True:
        chunk = fh.read(_READ_CHUNK)
        if not chunk:
            return None
        buf = tail + chunk
        i = buf.find(_GZIP_HEADER)
        if i >= 0:
            return base + i
        keep = len(_GZIP_HEADER) - 1
        tail = buf[-keep:]
        base += len(buf) - len(tail)


def _inflate_member(fh: BinaryIO, begin: int) -> Optional[Tuple[int, bytes]]:
    """(end offset, content) of the gzip member starting at `begin`, or
    None when it is damaged or cut short."""
    fh.seek(begin)
    inflater = zlib.decompressobj(wbits=31)            # one gzip member
    parts: List[bytes] = []
    fed = 0
    try:
        while not inflater.eof:
            chunk = fh.read(_READ_CHUNK)
            if not chunk:
                return None
            fed += len(chunk)
            parts.append(inflater.decompress(chunk))
    except zlib.error:
        return None
    return begin + fed - len(inflater.unused_data), b"".join(parts)


def _whole_members(fh: BinaryIO, start: int = 0) -> Iterator[Tuple[int, int, bytes]]:
    """(begin, end, content) of each whole gzip member at or after offset
    `start`, in order. Bytes outside whole members (a member damaged or
    cut short, anything else) are passed over: after a member that does
    not inflate, the scan resumes at the next gzip header past its
    start."""
    pos = start
    while True:
        begin = _find_header(fh, pos)
        if begin is None:
            return
        member = _inflate_member(fh, begin)
        if member is None:
            pos = begin + 1
            continue
        end, content = member
        yield begin, end, content
        pos = end


def complete_members_end(path: Path, start: int = 0) -> int:
    """The end offset of the last whole gzip member at or after `start`
    (`start` when there is none): the bytes [start, end) are a valid
    log whatever is still being written after them."""
    end = start
    with open(path, "rb") as fh:
        for _begin, end, _content in _whole_members(fh, start):
            pass
    return end


def read_records(path: Path) -> Iterator[Dict[str, Any]]:
    """The records of a log, in the order they were written. Bytes that
    hold no whole record (a record cut short by a crash, damage) are
    skipped with a warning naming them, and reading goes on with the
    next whole record."""
    path = Path(path)
    read_to = 0
    with open(path, "rb") as fh:
        for begin, end, content in _whole_members(fh):
            if begin > read_to:
                _warn_skipped(path, read_to, begin)
            read_to = end
            yield from _json_lines(path, begin, content)
        size = fh.seek(0, os.SEEK_END)
    if size > read_to:
        _warn_skipped(path, read_to, size)


def _warn_skipped(path: Path, begin: int, end: int) -> None:
    log.warning("%s: bytes %d to %d hold no whole record (cut short or damaged); skipped",
                path, begin, end)


def _json_lines(path: Path, begin: int, content: bytes) -> Iterator[Dict[str, Any]]:
    for line in content.decode("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as e:
            log.warning("%s: the record at offset %d is not JSON (%s); skipped", path, begin, e)


# ---------------------------------------------------------------------
# The process's sink: where the games this process plays are recorded
# ---------------------------------------------------------------------
_SINK: Optional[Path] = None
_SINK_LOCK = threading.Lock()


def configure(directory: Optional[Path], name: str) -> None:
    """Record every game this process finishes to `directory/name.jsonl.gz`
    (None: record nothing). Threads of one process share the file."""
    global _SINK
    _SINK = None if directory is None else Path(directory) / f"{name}.jsonl.gz"


def recording() -> bool:
    return _SINK is not None


def record_game(sim, setup, **kwargs) -> None:
    """`game_record(sim, setup, **kwargs)` written to the process's sink,
    if one is configured. A failure is logged, never raised: a game is
    not lost to its record."""
    if _SINK is None:
        return
    try:
        rec = game_record(sim, setup, **kwargs)
        with _SINK_LOCK:
            GameRecordLog(_SINK).write(rec)
    except Exception as e:                            # noqa: BLE001
        log.warning("game %s: record not written: %r", kwargs.get("game_label"), e)


# ---------------------------------------------------------------------
# Rebuilding the game
# ---------------------------------------------------------------------
def _corpus_game(rec: Dict[str, Any], verify: bool) -> dict:
    """The corpus game a mid-game start was cut from, checked against
    the digest of the content the game was played from."""
    prov = rec["setup"]["midgame"]
    path = Path(prov["dataset_dir"]) / prov["file"]
    if not path.is_absolute():
        path = REPO_ROOT / path
    content = gzip.decompress(path.read_bytes())
    want = prov.get("sha256")
    if verify and want and hashlib.sha256(content).hexdigest() != want:
        raise RecordMismatch(
            f"{rec.get('game_label')}: {path} is not the corpus file the game "
            f"started from (its content changed since)")
    return json.loads(content)


def start_state(rec: Dict[str, Any], *, verify: bool = True):
    """The position before the record's first command, built as the
    simulator built it."""
    from tools.replay_dataset import (_apply_command, _build_initial_gamestate,
                                      _setup_scenario_events)
    setup = rec["setup"]
    if "midgame" in setup:
        data = _corpus_game(rec, verify)
        gs = _build_initial_gamestate(data)
        _setup_scenario_events(gs, data.get("scenario_id", ""))
        for cmd in data["commands"][:int(setup["midgame"]["boundary_idx"])]:
            _apply_command(gs, cmd)
        gs.global_info._last_advance_events = []
        gs.global_info._last_checkup_strikes = None
    else:
        from wesnoth_ai.rules.scenario_pool import ScenarioSetup, build_scenario_gamestate
        gs = build_scenario_gamestate(ScenarioSetup(**setup), **rec.get("build", {}))
        _setup_scenario_events(gs, rec["scenario_id"])
    if rec.get("uniform_advancement"):
        gs.global_info._advance_uniform = True
        gs.global_info._advance_counter = 0
        gs.global_info._advance_salt = rec.get("seed_salt", "")
    return gs


def _check(rec: Dict[str, Any], gs, want: Optional[str], where: str) -> None:
    from wesnoth_ai.classes import state_digest
    if want is not None and state_digest(gs) != want:
        raise RecordMismatch(
            f"{rec.get('game_label')}: the rebuilt position {where} "
            f"(turn {gs.global_info.turn_number}, side {gs.global_info.current_side}) "
            f"is not the played game's")


def walk(rec: Dict[str, Any], gs=None, *, verify: bool = True) -> Iterator[Tuple[int, Any, list]]:
    """(index, state before the command, command) for every command of
    the record, the recruit rejections applied where they happened.
    The state is the one the walk mutates: copy it to keep it. When the
    walk is exhausted, `gs` (if given, the start state to mutate) holds
    the final position, pointed at the side the game ended on.

    With `verify`, each position a player side's turn started from and
    the final position are checked against the record's fingerprints
    (format 2 on), and a difference raises `RecordMismatch`."""
    from tools.replay_dataset import _apply_command
    if gs is None:
        gs = start_state(rec, verify=verify)
    rejections: Dict[int, List[Tuple[int, int]]] = {}
    for k, x, y in rec.get("rejections", ()):
        rejections.setdefault(int(k), []).append((int(x), int(y)))
    digests = {int(k): d for k, d in rec.get("turn_digests", ())} if verify else {}
    n = len(rec["commands"])
    for k, cmd in enumerate(rec["commands"]):
        for x, y in rejections.get(k, ()):
            _reject_recruit(gs, x, y)
        yield k, gs, cmd
        _apply_command(gs, cmd)
        _check(rec, gs, digests.get(k), f"after command {k} ({cmd[0]})")
    for x, y in rejections.get(n, ()):
        _reject_recruit(gs, x, y)
    if "final_side" in rec:
        gs.global_info.current_side = int(rec["final_side"])
    _check(rec, gs, rec.get("final_digest") if verify else None, "at the end")


def _reject_recruit(gs, x: int, y: int) -> None:
    rejected = set(getattr(gs.global_info, "_recruit_rejected_hexes", None) or ())
    rejected.add((x, y))
    gs.global_info._recruit_rejected_hexes = rejected


def rebuild(rec: Dict[str, Any], *, verify: bool = True):
    """The position the game ended in: after the record's last command,
    pointed at the side the simulator left it at. With `verify`, see
    `walk`."""
    gs = start_state(rec, verify=verify)
    for _step in walk(rec, gs, verify=verify):
        pass
    return gs
