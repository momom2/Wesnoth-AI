"""wesnoth_ai/rules/wml_state: the one reader of the WML that describes a game's
starting state, shared by the replay path and the generation path.

The cases here are the ones the two former parsers each handled alone:
the percent form only the generation side knew, the concatenated
attribute only the replay side knew, and the two spellings of the
village economy that each side read half of.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from wesnoth_ai.rules import wml_state as ws  # noqa: E402
from tools.replay_extract import parse_wml  # noqa: E402


def node(text: str):
    """The scenario node of a WML fragment, through the real parser."""
    root = parse_wml(text)
    return root.first("scenario") or root.first("replay_start")


def test_the_integer_reader_accepts_what_either_pipeline_used_to():
    # Plain and signed.
    assert ws.wml_int("3") == 3
    assert ws.wml_int("-2") == -2
    # A real zero, not an absence: the mini maps' scenery sides use it.
    assert ws.wml_int("0") == 0
    # The percent form the add-on scenarios write.
    assert ws.wml_int("70%") == 70
    assert ws.wml_int('"70%"') == 70
    # Attributes that got concatenated in the wild (2p Evil Factory
    # saves): salvage the leading integer rather than drop the replay.
    assert ws.wml_int("1 controller=human") == 1
    # Absent or unusable.
    assert ws.wml_int("") is None
    assert ws.wml_int(None) is None
    assert ws.wml_int("yes") is None
    assert ws.wml_int("yes", 7) == 7


def test_the_boolean_and_list_readers():
    assert ws.wml_bool("yes", False) is True
    assert ws.wml_bool("no", True) is False
    assert ws.wml_bool('"true"', False) is True
    assert ws.wml_bool(None, True) is True
    assert ws.wml_bool("garbage", True) is True          # default on nonsense
    assert ws.wml_list("Elvish Fighter, Elvish Archer ,") == ["Elvish Fighter",
                                                              "Elvish Archer"]
    assert ws.wml_list(None) == []


@pytest.mark.parametrize("wml, expected", [
    # The per-side spelling: what a save carries and what the Mini Maps
    # Collection writes.
    ('[scenario]\n[side]\nside=1\nvillage_gold=3\nvillage_support=2\n[/side]\n[/scenario]\n',
     (3, 2, None)),
    # The game-creation spelling mainline .cfg files declare.
    ('[scenario]\nmp_village_gold=2\nmp_village_support=1\n[/scenario]\n',
     (2, 1, None)),
    # Both: the per-side form wins, because that is what the engine reads.
    ('[scenario]\nmp_village_gold=2\n[side]\nside=1\nvillage_gold=5\n[/side]\n[/scenario]\n',
     (5, None, None)),
    # The percent form of the experience modifier.
    ('[scenario]\nexperience_modifier="70%"\n[/scenario]\n', (None, None, 70)),
    # Nothing declared.
    ('[scenario]\n[/scenario]\n', (None, None, None)),
    # A declared zero survives: it is a setting, not an absence.
    ('[scenario]\n[side]\nside=1\nvillage_gold=0\n[/side]\n[/scenario]\n',
     (0, None, None)),
    # Side 2 answers when side 1 is absent.
    ('[scenario]\n[side]\nside=2\nvillage_gold=4\n[/side]\n[/scenario]\n',
     (4, None, None)),
])
def test_both_spellings_of_the_economy_are_read(wml, expected):
    assert ws.scenario_economy(node(wml)) == expected


def test_a_side_block_reads_the_same_whether_it_came_from_a_save_or_a_cfg():
    """A save's [side] carries everything; a .cfg's carries a subset
    and the caller supplies the rest. One reader, one record shape."""
    save = node('[replay_start]\n[side]\nside=2\nfaction="Undead"\ngold=125\n'
                'income=-1\nvillage_gold=3\nvillage_support=2\nfog=no\nshroud=yes\n'
                'recruit="Skeleton,Ghoul"\ntype="Dark Sorcerer"\ncolor="red"\n'
                'controller="human"\n[/side]\n[/replay_start]\n')
    got = ws.read_side(save.first("side"))
    assert got == {
        "side": 2, "faction": "Undead", "gold": 125,
        "base_income": ws.ENGINE_BASE_INCOME - 1,     # income= is an offset
        "village_income": 3, "village_support": 2,
        "fog": False, "shroud": True,
        "recruit": ["Skeleton", "Ghoul"],
        "leader_type": "Dark Sorcerer", "color": "red", "controller": "human",
    }
    # The .cfg form: no faction, no recruit list, no economy. The
    # defaults fill exactly those, and nothing the block declares.
    cfg = node('[scenario]\n[side]\nside=1\ngold=175\nfog=yes\n[/side]\n[/scenario]\n')
    got = ws.read_side(cfg.first("side"),
                       defaults={"faction": "Rebels", "recruit": ["Elvish Fighter"],
                                 "gold": 100, "village_income": 9})
    assert got["gold"] == 175, "the block's own value wins over the default"
    assert got["faction"] == "Rebels" and got["recruit"] == ["Elvish Fighter"]
    assert got["village_income"] == 9 and got["fog"] is True
    assert got["base_income"] == ws.ENGINE_BASE_INCOME
    # A [side] with no usable number is not a side.
    assert ws.read_side(node('[scenario]\n[side]\nteam_name=x\n[/side]\n'
                             '[/scenario]\n').first("side")) is None


def test_villages_come_back_zero_indexed_and_placeholders_are_dropped():
    side = node('[scenario]\n[side]\nside=2\n[village]\nx=7\ny=41\n[/village]\n'
                '[village]\nx=0\ny=0\n[/village]\n[/side]\n[/scenario]\n').first("side")
    assert ws.read_villages(side, 2) == [{"x": 6, "y": 40, "side": 2}]


def test_a_unit_reads_its_position_leader_flag_and_petrified_status():
    side = node('[scenario]\n[side]\nside=3\n'
                '[unit]\ntype="Giant Scorpion"\nx=5\ny=9\n'
                '[status]\npetrified=yes\n[/status]\n[/unit]\n'
                '[unit]\ntype="Lieutenant"\nx=2\ny=3\ncanrecruit=yes\n[/unit]\n'
                '[unit]\ntype="Ghoul"\nx=recall\ny=recall\n[/unit]\n'
                '[/side]\n[/scenario]\n').first("side")
    units = [ws.read_unit(u, 3, uid=i) for i, u in enumerate(side.all("unit"))]
    assert units[0] == {"uid": 0, "type": "Giant Scorpion", "side": 3,
                        "x": 4, "y": 8, "is_leader": False, "petrified": True}
    assert units[1] == {"uid": 1, "type": "Lieutenant", "side": 3,
                        "x": 1, "y": 2, "is_leader": True}
    # A recall-list unit has no board position: skipped, not an error.
    assert units[2] is None


def test_the_time_of_day_attributes_are_read_not_resolved():
    n = node('[scenario]\ncurrent_time=1\nrandom_start_time=yes\n'
             '[time]\nid=dawn\n[/time]\n[time]\nid=day\n[/time]\n[/scenario]\n')
    assert ws.read_tod(n) == (1, True, 2)
    # No schedule: the caller's slot count stands.
    assert ws.read_tod(node('[scenario]\n[/scenario]\n')) == (None, False, 6)


@pytest.mark.parametrize("declared, slots, start", [
    (-1, 6, 5), (7, 6, 1), (5, 2, 1), (0, 6, 0), (-13, 6, 5),
])
def test_the_start_slot_wraps_as_the_engine_wraps_it(declared, slots, start):
    """The tod_manager constructor wraps `current_time` into the
    schedule with a modulo that is never negative (`fix_time_index`,
    src/tod_manager.cpp:66, 1.18.4). The raw value reached the
    simulator unchecked: the default-cycle index clamped a negative
    one to dawn, the time-area index wrapped it, and the Rust core
    panicked on it (core_step.rs, a signed `%` cast to usize)."""
    times = "".join(f"[time]\nid=t{i}\n[/time]\n" for i in range(slots))
    n = node(f"[scenario]\ncurrent_time={declared}\n{times}[/scenario]\n")
    assert ws.read_tod(n) == (start, False, slots)


def test_a_record_start_slot_reaches_the_state_wrapped():
    """A replay record's `tod_start_index` is read the same way, and
    the board cycle and a time area then agree on the phase: before,
    -1 put the board at dawn (a clamp) and the area at second watch
    (a wrap)."""
    from tools.replay_dataset import _build_initial_gamestate, _lawful_bonus_at
    gs = _build_initial_gamestate({"map_data": "Gg, Gg\nGg, Gg", "tod_start_index": -1})
    assert gs.global_info._tod_start_offset == 5
    assert gs.global_info.time_of_day == "second_watch"
    gs.global_info._time_areas = {(0, 0): [0, 25, 25, 0, -25, -25]}
    assert _lawful_bonus_at(gs, 0, 0, 1) == _lawful_bonus_at(gs, 1, 1, 1) == -25


def test_a_time_area_does_not_contribute_its_own_schedule():
    """The engine reads `current_time` and the slot count from the
    scenario's TOP level; a [time_area]'s [time] children are that
    area's cycle. Generation used to strip [time_area] with a regex
    before searching the raw text; reading a parsed node makes the
    stripping unnecessary, and this pins that it really is."""
    n = node('[scenario]\n[time]\nid=dawn\n[/time]\n'
             '[time_area]\ncurrent_time=3\n'
             '[time]\nid=x\n[/time]\n[time]\nid=y\n[/time]\n'
             '[/time_area]\n[/scenario]\n')
    assert ws.read_tod(n) == (None, False, 1)


def test_a_value_that_is_neither_yes_nor_no_is_not_quietly_no():
    """`random_start_time` has a third form -- a value list such as
    `"2,4"` -- and folding it onto False reads as "no random start",
    which silently begins the game at dawn. The caller needs to be
    able to tell the two apart."""
    assert ws.wml_bool_or_none("yes") is True
    assert ws.wml_bool_or_none("no") is False
    assert ws.wml_bool_or_none('"2,4"') is None
    assert ws.wml_bool_or_none("") is None
    assert ws.wml_bool_or_none(None) is None
    # The defaulting form keeps its contract.
    assert ws.wml_bool('"2,4"', False) is False
    assert ws.wml_bool('"2,4"', True) is True


def test_both_pipelines_read_the_time_of_day_through_this_module():
    """Generation read the three keys with three private regexes over
    the raw template while reconstruction read them off a parsed node.
    One reader now; this fails if either grows its own again."""
    import inspect
    import re

    from tools import replay_extract, scenario_pool

    for module in (scenario_pool, replay_extract):
        src, name = inspect.getsource(module), module.__name__
        assert "read_tod(" in src, f"{name} no longer uses the shared reader"
        for key in ("current_time", "random_start_time"):
            bad = re.findall(r"re\.(?:search|findall|match)\([^)]*"
                             + key, src)
            assert not bad, f"{name}: a private regex over {key}: {bad}"


def test_the_board_schedule_is_read_and_compared():
    """Both engines hardcode the six-slot day, so the one thing that
    must not happen is a scenario declaring a different one and
    nothing noticing."""
    default = ('[scenario]\n'
               + "".join(f'[time]\nid={i}\nlawful_bonus={b}\n[/time]\n'
                         for i, b in (("dawn", 0), ("morning", 25),
                                      ("afternoon", 25), ("dusk", 0),
                                      ("first_watch", -25),
                                      ("second_watch", -25)))
               + '[/scenario]\n')
    assert ws.board_cycle(node(default)) == [0, 25, 25, 0, -25, -25]
    assert ws.board_cycle_is_default(node(default))
    # No schedule at all means the engine's default applies.
    assert ws.board_cycle(node('[scenario]\n[/scenario]\n')) == []
    assert ws.board_cycle_is_default(node('[scenario]\n[/scenario]\n'))
    # An hourly schedule (24 slots with intermediate bonuses) exists in
    # the wild; it is exactly what the constant cannot express.
    hourly = ('[scenario]\n'
              + "".join(f'[time]\nid=h{i}\nlawful_bonus={b}\n[/time]\n'
                        for i, b in enumerate([0, 5, 15, 25, 25, 25]))
              + '[/scenario]\n')
    assert not ws.board_cycle_is_default(node(hourly))


def test_an_unplayable_schedule_is_refused_under_strict(monkeypatch):
    hourly = ('[scenario]\n'
              + "".join(f'[time]\nid=h{i}\nlawful_bonus={b}\n[/time]\n'
                        for i, b in enumerate([0, 5, 15, 25, 25, 25]))
              + '[/scenario]\n')
    assert ws.check_board_cycle(node(hourly), "x") is False   # warns
    monkeypatch.setenv("WESNOTH_STRICT_WML", "1")
    with pytest.raises(ws.UnsupportedSchedule):
        ws.check_board_cycle(node(hourly), "x")
    # The default passes under strict too, or the switch is unusable.
    assert ws.check_board_cycle(node('[scenario]\n[/scenario]\n'), "x")


def test_a_time_area_schedule_is_not_the_board_schedule():
    """A zone's cycle is read per hex. If it leaked into the board
    read, Tombs of Kesorak would look like a non-default map and the
    guard would cry wolf on a scenario we handle correctly."""
    n = node('[scenario]\n[time_area]\n'
             + "".join(f'[time]\nid=z{i}\nlawful_bonus={b}\n[/time]\n'
                       for i, b in enumerate([-25, 0, 0, -25]))
             + '[/time_area]\n[/scenario]\n')
    assert ws.board_cycle(n) == []
    assert ws.board_cycle_is_default(n)


def test_the_quick_leader_gates_are_seen_even_though_they_are_not_modelled():
    """`tools/traits.py` gives every 4-MP leader the quick trait
    unconditionally. The era gates that on a WML variable and on a
    per-unit one (eras.lua:5-22), and real scenarios use the second --
    Dark Forecast and Isle of Mists set `dont_make_me_quick`. Neither
    is ours, which is WHY the unconditional rule is right; this reads
    the gates so that stays a checked fact."""
    clean = node('[scenario]\n[side]\n[unit]\ntype="Lieutenant"\n'
                 '[/unit]\n[/side]\n[/scenario]\n')
    assert ws.quick_leader_gates(clean) == []
    assert ws.check_quick_leader_gates(clean, "x")

    gated = node('[scenario]\n[side]\n[unit]\ntype="Lieutenant"\n'
                 '[variables]\ndont_make_me_quick=yes\n[/variables]\n'
                 '[/unit]\n[/side]\n[/scenario]\n')
    assert ws.quick_leader_gates(gated) == [
        "scenario/side/unit/variables.dont_make_me_quick"]
    assert not ws.check_quick_leader_gates(gated, "x")

    # The other gate, set the way WML sets a variable.
    via_set = node('[scenario]\n[event]\n[set_variable]\n'
                   'name=make_4mp_leaders_quick\nvalue=no\n'
                   '[/set_variable]\n[/event]\n[/scenario]\n')
    assert ws.quick_leader_gates(via_set)


def test_a_quick_leader_gate_is_refused_under_strict(monkeypatch):
    gated = node('[scenario]\n[side]\n[unit]\n'
                 '[variables]\ndont_make_me_quick=yes\n[/variables]\n'
                 '[/unit]\n[/side]\n[/scenario]\n')
    monkeypatch.setenv("WESNOTH_STRICT_WML", "1")
    with pytest.raises(ws.UnmodelledGate):
        ws.check_quick_leader_gates(gated, "x")


def test_nothing_we_build_touches_a_quick_leader_gate():
    """The precondition itself, over every scenario we build: the pool
    plus the two off-whitelist mainline maps in the corpus."""
    from tools.analysis.expansion_diff import POOL, _scenario_block
    from tools.scenario_events import load_scenario_wml

    for scenario_id in list(POOL) + ["multiplayer_Cynsaun_Battlefield",
                                     "multiplayer_Hornshark_Island"]:
        root = load_scenario_wml(scenario_id)
        assert root is not None, scenario_id
        block = _scenario_block(root)
        assert ws.quick_leader_gates(block) == [], scenario_id


def test_no_second_parser_of_the_side_block_survives():
    """The point of this module is that there is one reader. A new
    regex over `[side]` or its economy attributes in the pipeline
    files is the drift this replaced, so it fails here rather than in
    a replay sweep nobody runs locally.

    Scope: the files that build or emit a starting state. Diagnostics
    that scan raw text on purpose (`check_replay_consistency`,
    `filter_replays`) and the census are out of scope, and the census
    reads headers the pipelines never parse.
    """
    import inspect
    import re

    from tools import (dump_savestate, replay_builder, replay_extract, scenario_pool,
                       sim_to_replay)

    watched = [scenario_pool, replay_extract, sim_to_replay, replay_builder, dump_savestate]
    # A regex that reaches into a [side] block or its economy attrs.
    suspicious = re.compile(
        r"re\.(compile|search|match|finditer|findall)\([^)]*"
        r"(\[side\]|village_gold|village_support|experience_modifier"
        r"|canrecruit|\[village\])")
    offenders = {}
    for module in watched:
        text = inspect.getsource(module)
        hits = [m.group(0)[:70] for m in suspicious.finditer(text)]
        if hits:
            offenders[module.__name__] = hits
    assert offenders == {}, (
        f"a second parser of the side block appeared: {offenders}. "
        f"Read it through wesnoth_ai/rules/wml_state instead.")
