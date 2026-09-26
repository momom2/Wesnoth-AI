"""Who plays each side of a recorded game, and where the recorded game
stops being a game between its two players.

A multiplayer replay keeps recording after its game is decided. A player
who surrenders hands the side to another player, and one who leaves or
disconnects hands it to the host; either way the other player can end
up holding both sides and play on, and nothing after that is a decision
of the side's own player. The server announces each of these events as
a chat line from "server" (src/server/wesnothd/game.cpp, 1.18.4):

    "X takes control of side N."      change_controller, line 590
    "X has surrendered."              the [surrender] command, line 1054,
                                      after the side passed on (1034-1052)
    "X has left the game."            remove_player, line 1493; each side
    "X has disconnected."             X held passes to the host (1517)
    "X becomes an observer."          change_side_controller, line 558

and records the [surrender] command itself, `side_number` being the
surrendering client's viewing team, 0-based (quit_confirmation.cpp:78;
the server checks `sides_[side_number] == user`, game.cpp:927-935).

`find_game_end` cuts a game at the first action after a surrender, and at
the first action of a player side's turn taken while the side is held by
the other side's player. A player who disconnects and comes back before
their side's turn, or whose side passes to a third person who plays it
on, is not cut: the game is still one between two people. Measured on
300 corpus games (2026-09-26): 49 cut at a surrender, 9 at a side played
by its opponent, 3 from their first action (one name on both sides).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from wesnoth_ai.sim.classes import PLAYER_SIDES, opponent_of

TAKES_CONTROL_RE = re.compile(r"^(.+) takes control of side (\d+)\.$")
SURRENDERED_RE = re.compile(r"^(.+) has surrendered\.$")

# Tags of a [command] that advance the game: after the game's end, the
# first one of these is where the record is cut.
ACTION_TAGS = frozenset({"move", "attack", "recruit", "recall", "init_side",
                         "end_turn", "fire_event"})

# The [side] attributes naming the player who holds a side, in the
# precedence tools/player_ratings.player_id_of_side uses.
_PLAYER_KEYS = ("player_id", "current_player", "save_id", "name")
# An AI side's player_id is its host's name; the side is the AI's.
AI_CONTROLLER = "[ai]"

SURRENDER = "surrender"
PLAYED_BY_OPPONENT = "played_by_opponent"


@dataclass
class GameEnd:
    """How the recorded game ended, as the server recorded it.

    `reason`: SURRENDER (a player surrendered), PLAYED_BY_OPPONENT (a
    player side's turn was taken by the other side's player), or ""
    (neither). `cut_index`: the index, among the replay's [command]s, of
    the first one left out; None when nothing follows the end.
    `first_action`: the index of the replay's first action, so a cut
    there means no command of the game is its players'.
    `surrender_side`: the side the server names as surrendering: the
    [surrender] command's, else the side the surrendering player last
    held. `surrender_sides_agree` is False when both are known and
    differ. `taken_side`: the side whose turn its opponent took."""
    reason: str = ""
    cut_index: Optional[int] = None
    first_action: Optional[int] = None
    taken_side: Optional[int] = None
    surrender_side: Optional[int] = None
    surrender_player: str = ""
    surrender_sides_agree: bool = True
    owners: Dict[int, str] = field(default_factory=dict)

    @property
    def cut_before_play(self) -> bool:
        return self.cut_index is not None and self.cut_index == self.first_action

    def as_record(self) -> dict:
        """The extracted record's `game_end` field."""
        return {"reason": self.reason or "end",
                "surrender_side": self.surrender_side,
                "surrender_sides_agree": self.surrender_sides_agree,
                "played_by_opponent_side": self.taken_side,
                "cut_command": self.cut_index,
                "cut_before_play": self.cut_before_play}


def initial_controllers(snapshot) -> Dict[int, str]:
    """{side: player name} from the starting snapshot's [side] blocks,
    AI_CONTROLLER for an AI side; a side without a name is left out
    (never matches another)."""
    out: Dict[int, str] = {}
    if snapshot is None:
        return out
    for side in snapshot.all("side"):
        try:
            num = int(str(side.attrs.get("side", "0")).strip('"') or 0)
        except ValueError:
            continue
        if str(side.attrs.get("controller", "")).strip('"').lower() == "ai":
            out[num] = AI_CONTROLLER
            continue
        for key in _PLAYER_KEYS:
            name = str(side.attrs.get(key, "") or "").strip().strip('"')
            if name:
                out[num] = name
                break
    return out


def _server_message(sub) -> Optional[str]:
    if sub.tag != "speak":
        return None
    if str(sub.attrs.get("id", "")).strip('"') != "server":
        return None
    return str(sub.attrs.get("message", "")).strip('"')


def _int_attr(sub, key: str) -> Optional[int]:
    try:
        return int(str(sub.attrs.get(key, "")).strip('"'))
    except ValueError:
        return None


class _Control:
    """Who holds each side now, and whose side it is: its first player,
    or a third person who took it over (a replacement). A side held by
    the other side's player is not played by its own."""

    def __init__(self, snapshot):
        self.holder = initial_controllers(snapshot)
        self.owner = dict(self.holder)
        self.last_handed_on: Dict[str, int] = {}   # player -> side they last lost
        self.side_to_move: Optional[int] = None

    def take(self, player: str, side: int) -> None:
        previous = self.holder.get(side)
        if previous:
            self.last_handed_on[previous] = side
        self.holder[side] = player
        if player not in (self.owner.get(s) for s in PLAYER_SIDES):
            self.owner[side] = player

    def played_by_opponent(self, side: Optional[int]) -> bool:
        if side not in PLAYER_SIDES:
            return False
        holder = self.holder.get(side)
        opponent = self.owner.get(opponent_of(side))
        return bool(holder) and holder == opponent and holder != AI_CONTROLLER

    def side_of(self, player: str) -> Optional[int]:
        """The player side a surrendering player held. The server hands
        the side on before it announces the surrender (game.cpp:1052,
        1054), so it is the side the player last handed on, unless they
        still hold exactly one."""
        held = [s for s in PLAYER_SIDES if self.holder.get(s) == player]
        if len(held) == 1:
            return held[0]
        side = self.last_handed_on.get(player)
        return side if side in PLAYER_SIDES else None


def find_game_end(commands: List, snapshot) -> GameEnd:
    """Where the game ends in the replay's [command] list (all [replay]
    blocks in order), from the server's lines and [surrender] commands.
    See the module docstring for the rule."""
    control = _Control(snapshot)
    first_action: Optional[int] = None
    surrender_player = ""
    by_message: Optional[int] = None
    by_command: Optional[int] = None
    for i, cmd in enumerate(commands):
        action = next((sub for sub in cmd.children if sub.tag in ACTION_TAGS), None)
        if action is not None:
            first_action = i if first_action is None else first_action
            if surrender_player or by_command is not None:
                return _ended_by_surrender(control, i, first_action, surrender_player,
                                           by_message, by_command)
            if action.tag == "init_side":
                control.side_to_move = _int_attr(action, "side_number")
            if control.played_by_opponent(control.side_to_move):
                return GameEnd(reason=PLAYED_BY_OPPONENT, cut_index=i,
                               first_action=first_action,
                               taken_side=control.side_to_move,
                               owners=dict(control.owner))
        for sub in cmd.children:
            if sub.tag == "surrender" and by_command is None:
                side_number = _int_attr(sub, "side_number")
                by_command = None if side_number is None else side_number + 1
                continue
            message = _server_message(sub)
            if message is None:
                continue
            taken = TAKES_CONTROL_RE.match(message)
            if taken:
                control.take(taken.group(1), int(taken.group(2)))
                continue
            quit_ = SURRENDERED_RE.match(message)
            if quit_ and not surrender_player:
                surrender_player = quit_.group(1)
                by_message = control.side_of(surrender_player)
    if surrender_player or by_command is not None:
        return _ended_by_surrender(control, None, first_action, surrender_player,
                                   by_message, by_command)
    return GameEnd(first_action=first_action, owners=dict(control.owner))


def _ended_by_surrender(control: _Control, cut_index: Optional[int],
                        first_action: Optional[int], player: str,
                        by_message: Optional[int], by_command: Optional[int]) -> GameEnd:
    side = by_command if by_command in PLAYER_SIDES else by_message
    agree = by_command is None or by_message is None or by_command == by_message
    return GameEnd(reason=SURRENDER, cut_index=cut_index, first_action=first_action,
                   surrender_side=side, surrender_player=player,
                   surrender_sides_agree=agree, owners=dict(control.owner))
