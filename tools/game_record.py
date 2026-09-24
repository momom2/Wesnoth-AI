"""Game records: every generated game kept whole (user order 2026-09-24).

A record holds what rebuilds the game position by position: the
scenario setup (or, for a mid-game start, the corpus file and the cut),
the build arguments, the simulator's advancement channel, and the
command list the simulator applied, in the form
`tools.replay_dataset._apply_command` replays, attack and recruit seeds
included. Recruit rejections, which change what the side to move
observes but apply no command, are kept beside it.

Fight outcome distributions ride along as optional data, keyed by the
index of the attack command they describe (`outcomes`): training writes
those it computed while playing (the search's exact distribution of an
attack it played, the counter-weapon choice's strike tables), and an
analysis may add its own with `add_outcomes`. Nothing reads them to
rebuild the game.

    configure(directory, name)                            # once per process
    record_game(sim, setup, game_label=..., build={...}, players={...})
    rec = game_record(sim, setup, game_label=..., build={...}, players={...})
    GameRecordLog(path).write(rec)                        # one JSON line, gzip
    for rec in read_records(path): ...
    for k, gs, cmd in walk(rec): ...                      # the state before each command
    gs = rebuild(rec)                                     # the final state

One record per line, so a log grows as games finish and a truncated
file loses at most its last line.
"""
from __future__ import annotations

import dataclasses
import gzip
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

log = logging.getLogger("game_record")

FORMAT = 1


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


def note_search_outcomes(sim, policy, game_label: str) -> None:
    """After `sim.step`: if `policy` searched and computed the exact
    outcome distribution of the attack just recorded, put it into that
    command's outcome data under "search"."""
    pop = getattr(policy, "pop_played_outcomes", None)
    if pop is None:
        return
    dist = pop(game_label)
    if dist is None or not sim.command_history:
        return
    rc = sim.command_history[-1]
    if rc.kind == "attack":
        rc.extras.setdefault("outcomes", {})["search"] = distribution_data(dist)


def add_outcomes(rec: Dict[str, Any], command_index: int, source: str, data: Any) -> None:
    """Attach an outcome distribution to the attack at `command_index`
    under `source` (who computed it, e.g. "search", "counter_weapon",
    or an analysis's name)."""
    rec.setdefault("outcomes", {}).setdefault(str(command_index), {})[source] = data


# ---------------------------------------------------------------------
# Writing and reading
# ---------------------------------------------------------------------
class GameRecordLog:
    """Appends records to a gzip JSON-lines file, one line per game,
    each line compressed as its own gzip member so a crash loses at
    most the game being written."""

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


def read_records(path: Path) -> Iterator[Dict[str, Any]]:
    """The records of a log, in the order they were written. A final
    line cut short by a crash is skipped with a warning."""
    with gzip.open(Path(path), "rt", encoding="utf-8") as fh:
        try:
            for line in fh:
                if line.strip():
                    yield json.loads(line)
        except (EOFError, json.JSONDecodeError) as e:
            log.warning("%s: the last record is truncated (%s); skipped", path, e)


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
def start_state(rec: Dict[str, Any]):
    """The position before the record's first command, built as the
    simulator built it."""
    from tools.replay_dataset import (_apply_command, _build_initial_gamestate,
                                      _setup_scenario_events)
    setup = rec["setup"]
    if "midgame" in setup:
        prov = setup["midgame"]
        path = Path(prov["dataset_dir"]) / prov["file"]
        if not path.is_absolute():
            path = _ROOT / path
        data = json.load(gzip.open(str(path), "rt", encoding="utf-8"))
        gs = _build_initial_gamestate(data)
        _setup_scenario_events(gs, data.get("scenario_id", ""))
        for cmd in data["commands"][:int(prov["boundary_idx"])]:
            _apply_command(gs, cmd)
        gs.global_info._last_advance_events = []
        gs.global_info._last_checkup_strikes = None
    else:
        from tools.scenario_pool import ScenarioSetup, build_scenario_gamestate
        gs = build_scenario_gamestate(ScenarioSetup(**setup), **rec.get("build", {}))
        _setup_scenario_events(gs, rec["scenario_id"])
    if rec.get("uniform_advancement"):
        gs.global_info._advance_uniform = True
        gs.global_info._advance_counter = 0
        gs.global_info._advance_salt = rec.get("seed_salt", "")
    return gs


def walk(rec: Dict[str, Any], gs=None) -> Iterator[Tuple[int, Any, list]]:
    """(index, state before the command, command) for every command of
    the record, the recruit rejections applied where they happened.
    The state is the one the walk mutates: copy it to keep it. When the
    walk is exhausted, `gs` (if given, the start state to mutate) holds
    the final position."""
    from tools.replay_dataset import _apply_command
    if gs is None:
        gs = start_state(rec)
    rejections: Dict[int, List[Tuple[int, int]]] = {}
    for k, x, y in rec.get("rejections", ()):
        rejections.setdefault(int(k), []).append((int(x), int(y)))
    n = len(rec["commands"])
    for k, cmd in enumerate(rec["commands"]):
        for x, y in rejections.get(k, ()):
            _reject_recruit(gs, x, y)
        yield k, gs, cmd
        _apply_command(gs, cmd)
    for x, y in rejections.get(n, ()):
        _reject_recruit(gs, x, y)


def _reject_recruit(gs, x: int, y: int) -> None:
    rejected = set(getattr(gs.global_info, "_recruit_rejected_hexes", None) or ())
    rejected.add((x, y))
    gs.global_info._recruit_rejected_hexes = rejected


def rebuild(rec: Dict[str, Any]):
    """The position the game ended in: after the record's last command,
    pointed at the side the simulator left it at."""
    gs = start_state(rec)
    for _step in walk(rec, gs):
        pass
    if "final_side" in rec:
        gs.global_info.current_side = int(rec["final_side"])
    return gs
