"""A live Wesnoth game mirrored in the simulator from the engine's own log.

Wesnoth logs every synced command as it records it (`--log-info=replay`:
"add_synced_command", src/replay.cpp:253, 1.18.4), every number those
commands draw from the synced generator (`--log-info=random`:
"next_random_impl returned", src/random_synced.cpp:37) and every choice an
AI side makes between advancements (`--log-info=engine`: "chose
advancement number", src/actions/advancement.cpp:241). The commands of both
sides, replayed in a `WesnothSim` with the engine's own draws, give the
simulator the engine's game command for command, so a player observes it
exactly as in a simulated match (its sightings, seen sets and memory
included), and each decision can compare the two boards
(tools/live_vs_rca.py). The default AI's "stop unit" action, which
takes a unit's movement and attacks without a synced command, so neither
the replay nor the mirror sees it, is logged by `--log-info=ai/actions`
(ai/actions.cpp:876-926); its leader_shares_keep candidate action takes its
leaders' movement with no action at all (ai/default/ca.cpp:1688), and the
candidate action loop logs that it runs it at debug level
(`--log-debug=ai/stage/rca`, ai/default/stage_rca.cpp:124).

The log carries no `[init_side]` or `[end_turn]` (both recorded outside
add_synced_command, replay.cpp:219-226 and 305-311): the mirror ends a side's turn when
the next command comes from another side, or when the frame of a decision
names another turn.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ENGINE_LOG_DOMAINS = "replay,random,engine,ai/actions"
ENGINE_DEBUG_DOMAINS = "ai/stage/rca"
SYNC_MARKER = "WESNOTH_AI_SYNC"

_COMMAND_HEADER = "add_synced_command:"
_DRAW = re.compile(r"next_random_impl returned (\d+)\s*$")
_ADVANCEMENT = re.compile(r"chose advancement number (-?\d+)")
_STOP = re.compile(r"stopunit by side (\d+).* from unit on location (\d+),(\d+)")
_CANDIDATE_ACTION = re.compile(r"Executing best candidate action: candidate action with name \[([^\]]+)\]")
_SYNC = re.compile(SYNC_MARKER + r" (\d+)")
_TIMESTAMPED = re.compile(r"^\d{8} \d{2}:\d{2}:\d{2}")
_MESSAGE = re.compile(r"^\d{8} \d{2}:\d{2}:\d{2} (error|warning) ([^:]+): (.*)$")
# Domains whose messages come from loading the game's data, before play.
_LOADING_DOMAINS = {"config", "preprocessor", "general"}


class MirrorDivergence(RuntimeError):
    """The engine's game and the simulator's disagree: a fidelity defect
    to investigate at its root (CLAUDE.md, principle 4)."""


@dataclass
class WmlNode:
    """A WML block as `config::debug()` prints it: attributes, children."""
    attrs: Dict[str, str] = field(default_factory=dict)
    children: List[Tuple[str, "WmlNode"]] = field(default_factory=list)

    def child(self, tag: str) -> Optional["WmlNode"]:
        return next((node for name, node in self.children if name == tag), None)


@dataclass
class LoggedCommand:
    """One synced command: the side that issued it, its action tag and
    block, the numbers it drew and the advancement choices made during it."""
    from_side: int
    tag: str
    block: WmlNode
    draws: List[int] = field(default_factory=list)
    advancements: List[int] = field(default_factory=list)


@dataclass
class EngineMessage:
    """A warning or an error the engine logged during play (a WML or Lua
    error, an AI warning), first line only."""
    level: str
    domain: str
    text: str


@dataclass
class AiStop:
    """The default AI took the movement (and maybe the attacks) of its
    unit on (x, y), 1-based, outside any synced command."""
    side: int
    x: int
    y: int
    movement: bool
    attacks: bool


@dataclass
class AiCandidateAction:
    """The default AI's loop ran this candidate action."""
    name: str


@dataclass
class SyncMarker:
    """The live stage's marker, written to the engine log just before the
    frame of decision `seq`: every command before it has been logged."""
    seq: int


def parse_debug_wml(lines: List[str]) -> WmlNode:
    """`config::debug()` text (tab-indented `key = value` lines and
    `[tag]` / `[/tag]` lines) as a tree."""
    root = WmlNode()
    stack = [root]
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[/"):
            if len(stack) > 1:
                stack.pop()
            continue
        if line.startswith("[") and line.endswith("]"):
            node = WmlNode()
            stack[-1].children.append((line[1:-1], node))
            stack.append(node)
            continue
        key, sep, value = line.partition(" = ")
        if sep:
            stack[-1].attrs[key.strip()] = value.strip().strip('"')
    return root


class EngineLogReader:
    """Incremental reader of the engine log (`wesnoth-*.log`, the stderr
    stream): `feed` the new text, get back the complete events in order. A
    command is complete when the next command header or a sync marker
    arrives, since its draws and advancement choices follow its header
    (an attack's are logged strike by strike as the fight plays); until
    then it is the `open_command`."""

    def __init__(self):
        self._partial = ""
        self._body: Optional[List[str]] = None
        self._open: Optional[LoggedCommand] = None

    def feed(self, text: str) -> List[object]:
        events: List[object] = []
        lines = (self._partial + text).split("\n")
        self._partial = lines.pop()
        for line in lines:
            events.extend(self._line(line.rstrip("\r")))
        return events

    @property
    def open_command(self) -> Optional[LoggedCommand]:
        """The last command logged, its draws possibly still coming."""
        return self._open

    def take_open(self) -> Optional[LoggedCommand]:
        """The open command, taken as complete (it ended the game)."""
        cmd, self._open = self._open, None
        return cmd

    def close(self) -> List[object]:
        """Flush the open command (the log ended: the game is over)."""
        events = self.feed("\n") if self._partial else []
        events.extend(self._close_body())
        if self._open is not None:
            events.append(self._open)
            self._open = None
        return events

    def _line(self, line: str) -> List[object]:
        if self._body is not None:
            if line.strip() and not _TIMESTAMPED.match(line):
                self._body.append(line)
                return []
            return self._close_body() + self._line_outside(line)
        return self._line_outside(line)

    def _line_outside(self, line: str) -> List[object]:
        events: List[object] = []
        if line.rstrip().endswith(_COMMAND_HEADER):
            if self._open is not None:
                events.append(self._open)
                self._open = None
            self._body = []
            return events
        m = _DRAW.search(line)
        if m:
            if self._open is None:
                raise MirrorDivergence(f"a draw outside any command: {line!r}")
            self._open.draws.append(int(m.group(1)))
            return events
        m = _ADVANCEMENT.search(line)
        if m and self._open is not None:
            self._open.advancements.append(int(m.group(1)))
            return events
        m = _SYNC.search(line)
        if m:
            if self._open is not None:
                events.append(self._open)
                self._open = None
            events.append(SyncMarker(int(m.group(1))))
            return events
        m = _CANDIDATE_ACTION.search(line)
        if m:
            events.append(AiCandidateAction(m.group(1)))
            return events
        m = _STOP.search(line)
        if m:
            events.append(AiStop(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                 "remove movement" in line, "remove attacks" in line))
            return events
        m = _MESSAGE.match(line)
        if m and m.group(2) not in _LOADING_DOMAINS:
            events.append(EngineMessage(m.group(1), m.group(2), m.group(3).strip()))
        return events

    def _close_body(self) -> List[object]:
        if self._body is None:
            return []
        node = parse_debug_wml(self._body)
        self._body = None
        actions = [(tag, block) for tag, block in node.children]
        if not actions:
            raise MirrorDivergence(f"a synced command without an action: {node.attrs}")
        tag, block = actions[0]
        self._open = LoggedCommand(from_side=int(node.attrs.get("from_side", 0) or 0), tag=tag, block=block)
        return []


class EngineLog:
    """The engine log file of one Wesnoth process, read as it grows."""

    def __init__(self, path: Path):
        self.path = path
        self._offset = 0
        self.reader = EngineLogReader()

    def poll(self) -> List[object]:
        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(self._offset)
                text = f.read()
                self._offset = f.tell()
        except OSError:
            return []
        return self.reader.feed(text) if text else []


def _hex(node: WmlNode) -> Tuple[int, int]:
    """A WML location (1-based) as the simulator's (0-based)."""
    return int(node.attrs["x"]) - 1, int(node.attrs["y"]) - 1


def sim_command(cmd: LoggedCommand) -> list:
    """The command in `_apply_command`'s vocabulary, WML's 1-based hexes
    made 0-based (as tools/replay_extract.py makes them)."""
    b = cmd.block
    if cmd.tag == "move":
        xs = [int(v) - 1 for v in b.attrs["x"].split(",")]
        ys = [int(v) - 1 for v in b.attrs["y"].split(",")]
        return ["move", xs, ys, cmd.from_side]
    if cmd.tag == "attack":
        ax, ay = _hex(b.child("source"))
        dx, dy = _hex(b.child("destination"))
        return ["attack", ax, ay, dx, dy, int(b.attrs.get("weapon", 0)),
                int(b.attrs.get("defender_weapon", -1)), "", list(cmd.advancements)]
    if cmd.tag == "recruit":
        x, y = _hex(b)
        return ["recruit", b.attrs["type"], x, y, ""]
    raise MirrorDivergence(f"side {cmd.from_side} played a [{cmd.tag}] command the mirror does not model")


class LiveMirror:
    """Replays the engine's commands in `sim` and keeps its turn in step."""

    MAX_TURN_STEPS = 6     # end_turns to reach a side's turn: a round and a half

    def __init__(self, sim):
        self.sim = sim
        self.commands = 0

    def apply(self, cmd: LoggedCommand) -> None:
        self.reach_side(cmd.from_side)
        command = sim_command(cmd)
        used = self.sim.apply_engine_command(command, cmd.draws)
        if command[0] == "attack" and used is not None and used != len(cmd.draws):
            raise MirrorDivergence(f"command {self.commands}: the fight took {used} draws, the engine "
                                   f"{len(cmd.draws)} ({command})")
        if command[0] == "move" and cmd.draws:
            raise MirrorDivergence(f"command {self.commands}: a move that drew {len(cmd.draws)} numbers")
        self.commands += 1

    def reach_side(self, side: int) -> None:
        """End turns until `side` is to play (its turn may follow a side
        that issued no command)."""
        for _ in range(self.MAX_TURN_STEPS):
            if self.sim.done or self.sim.current_side == side:
                return
            self.sim.step({"type": "end_turn"})
        raise MirrorDivergence(f"side {side} never came to play (side {self.sim.current_side} "
                               f"on turn {self.sim.turn_number})")

    def reach_turn(self, turn: int, side: int) -> None:
        """End turns until (turn, side), the frame of a decision."""
        for _ in range(self.MAX_TURN_STEPS):
            if self.sim.done or (self.sim.turn_number, self.sim.current_side) == (turn, side):
                return
            self.sim.step({"type": "end_turn"})
        raise MirrorDivergence(f"the engine is at turn {turn} side {side}, the simulator at turn "
                               f"{self.sim.turn_number} side {self.sim.current_side}")


def engine_commands(fork_history: List[object]) -> List[dict]:
    """The commands a policy action became on a simulator fork, as the
    live stage executes them (1-based hexes, the Lua AI's 1-based weapon
    index, ai/lua/core.cpp:209-215): up to and including the end_turn."""
    out: List[dict] = []
    for rc in fork_history:
        cmd = rc.cmd
        if cmd[0] == "move":
            out.append({"type": "move", "from_x": cmd[1][0] + 1, "from_y": cmd[2][0] + 1,
                        "to_x": cmd[1][-1] + 1, "to_y": cmd[2][-1] + 1})
        elif cmd[0] == "attack":
            out.append({"type": "attack", "from_x": cmd[1] + 1, "from_y": cmd[2] + 1,
                        "to_x": cmd[3] + 1, "to_y": cmd[4] + 1, "weapon": int(cmd[5]) + 1})
        elif cmd[0] == "recruit":
            out.append({"type": "recruit", "unit_type": cmd[1], "x": cmd[2] + 1, "y": cmd[3] + 1})
        elif cmd[0] == "end_turn":
            out.append({"type": "end_turn"})
            break
    return out

