"""The scenario events the engine fires around each side's turn: which
names, how often and in what order (docs/wesnoth_rules.md "Turn
events").

`play_controller::do_init_side` fires "turn N" and "new turn" once per
turn, at the first side's turn start, then "side turn", "side S turn",
"side turn N" and "side S turn N"; after the refresh, healing and
income, the four refresh forms. The end of a side's turn fires four end
forms and the end of the turn two more. Our applier used to fire "side S
turn N", "turn N", "new turn" and "side turn" at every side's
init_side, so a repeating "new turn" event ran once per side, and
"side S turn", "side turn N", three refresh forms and every end form
never ran. These tests drive the real applier (`_apply_command`) over a
unit-less two-side game whose events count and log themselves.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.replay_dataset import _apply_command, _build_initial_gamestate  # noqa: E402
from tools.replay_extract import parse_wml  # noqa: E402
from tools.scenario_events import collect_events  # noqa: E402

# Every name the engine fires around the turns of a two-side game's
# first two turns and the start of its third.
NAMES = [
    "turn 1", "turn 2", "turn 3", "new turn",
    "side turn", "side 1 turn", "side 2 turn",
    "side turn 1", "side turn 2", "side turn 3",
    "side 1 turn 1", "side 2 turn 1", "side 1 turn 2", "side 2 turn 2", "side 1 turn 3",
    "turn refresh", "side 1 turn refresh", "side 2 turn refresh",
    "turn 1 refresh", "turn 2 refresh", "turn 3 refresh",
    "side 1 turn 1 refresh", "side 2 turn 2 refresh",
    "side turn end", "side 1 turn end", "side 2 turn end",
    "side turn 1 end", "side turn 2 end", "side 1 turn 1 end", "side 2 turn 2 end",
    "turn end", "turn 1 end", "turn 2 end",
]


def _tag(name: str) -> str:
    return name.replace(" ", "_")


def _game(extra_events: str = ""):
    """A unit-less two-side game whose repeating event on each name adds
    one to `count_<name>` and appends the name to `log`."""
    blocks = "".join(
        f"[event]\nname={name}\nfirst_time_only=no\n"
        f"[set_variable]\nname=count_{_tag(name)}\nadd=1\n[/set_variable]\n"
        f"[set_variable]\nname=log\nvalue=$log;{_tag(name)}\n[/set_variable]\n"
        f"[/event]\n" for name in NAMES)
    root = parse_wml(f"[multiplayer]\n{blocks}{extra_events}[/multiplayer]\n")
    gs = _build_initial_gamestate({"map_data": "Gg, Gg\nGg, Gg",
                                   "starting_sides": [{"side": 1}, {"side": 2}]})
    gs.global_info._scenario_events = collect_events(root, "synthetic")
    gs.global_info._wml_variables = {}
    return gs


TWO_TURNS = [["init_side", 1], ["end_turn"], ["init_side", 2], ["end_turn"],
             ["init_side", 1], ["end_turn"], ["init_side", 2], ["end_turn"],
             ["init_side", 1]]


def _variables(gs):
    return gs.global_info._wml_variables


def test_each_turn_event_fires_as_often_as_the_engine_fires_it():
    gs = _game()
    for cmd in TWO_TURNS:
        _apply_command(gs, cmd)
    counts = {name: int(_variables(gs).get(f"count_{_tag(name)}", 0)) for name in NAMES}
    assert counts == {
        # Once per turn, at the side that opens it: turns 1, 2 and 3.
        "turn 1": 1, "turn 2": 1, "turn 3": 1, "new turn": 3,
        # Once per side turn: five side turns started.
        "side turn": 5, "side 1 turn": 3, "side 2 turn": 2,
        "side turn 1": 2, "side turn 2": 2, "side turn 3": 1,
        "side 1 turn 1": 1, "side 2 turn 1": 1, "side 1 turn 2": 1,
        "side 2 turn 2": 1, "side 1 turn 3": 1,
        "turn refresh": 5, "side 1 turn refresh": 3, "side 2 turn refresh": 2,
        "turn 1 refresh": 2, "turn 2 refresh": 2, "turn 3 refresh": 1,
        "side 1 turn 1 refresh": 1, "side 2 turn 2 refresh": 1,
        # Four side turns ended.
        "side turn end": 4, "side 1 turn end": 2, "side 2 turn end": 2,
        "side turn 1 end": 2, "side turn 2 end": 2,
        "side 1 turn 1 end": 1, "side 2 turn 2 end": 1,
        # Two turns ended, each when side 1 opened the next.
        "turn end": 2, "turn 1 end": 1, "turn 2 end": 1,
    }


def test_the_events_fire_in_the_engines_order():
    """Within side 1's turn-2 start: the end of turn 1, the turn's own
    two, the four side forms, then the four refresh forms
    (play_controller.cpp:597-604, :473-482, :519-522); at a side's
    turn end, the four end forms (:585-588)."""
    gs = _game()
    for cmd in TWO_TURNS[:4]:
        _apply_command(gs, cmd)
    _variables(gs)["log"] = ""
    _apply_command(gs, ["init_side", 1])
    assert _variables(gs)["log"].split(";")[1:] == [
        "turn_end", "turn_1_end",
        "turn_2", "new_turn",
        "side_turn", "side_1_turn", "side_turn_2", "side_1_turn_2",
        "turn_refresh", "side_1_turn_refresh", "turn_2_refresh"]
    _variables(gs)["log"] = ""
    _apply_command(gs, ["end_turn"])
    assert _variables(gs)["log"].split(";")[1:] == [
        "side_turn_end", "side_1_turn_end", "side_turn_2_end"]


def test_a_comma_separated_name_answers_to_each_name():
    """An [event]'s name= is a list (event_handler::names splits it on
    commas): this one runs at both sides' turn starts."""
    both = ("[event]\nname=side 1 turn 2, side_2_turn_2\nfirst_time_only=no\n"
            "[set_variable]\nname=both\nadd=1\n[/set_variable]\n[/event]\n")
    gs = _game(both)
    for cmd in TWO_TURNS:
        _apply_command(gs, cmd)
    assert _variables(gs)["both"] == "2"


def test_the_core_hands_a_command_to_python_only_when_it_would_fire_an_event():
    """GameCore runs no scenario events, so the wrapper routes an
    init_side or end_turn to the Python applier exactly when one of
    the names that command fires has an event that can still fire."""
    from wesnoth_ai.game_core import CoreState

    class Globals:
        def __init__(self, turn, side):
            self.g = {"turn_number": turn, "current_side": side}

        def globals_export(self):
            return self.g

    root = parse_wml("[multiplayer]\n"
                     "[event]\nname=turn 3 end\n[/event]\n"
                     "[event]\nname=side 2 turn 4 end\n[/event]\n"
                     "[event]\nname=side 3 turn\nfirst_time_only=no\n[/event]\n"
                     "[/multiplayer]\n")
    events = collect_events(root, "synthetic")
    core = CoreState(core=Globals(3, 2), game_id="", statics={"_scenario_events": events})
    assert core._events_pending(["init_side", 1])          # ends turn 3
    assert not core._events_pending(["init_side", 2])
    assert not core._events_pending(["end_turn"])          # side 2, turn 3
    assert core._events_pending(["init_side", 3])          # "side 3 turn"
    core.core = Globals(4, 2)
    assert core._events_pending(["end_turn"])              # side 2, turn 4
    events[0].fired = True
    core.core = Globals(3, 2)
    assert not core._events_pending(["init_side", 1])      # latched
