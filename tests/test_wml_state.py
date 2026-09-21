"""tools/wml_state: the one reader of the WML that describes a game's
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

from tools import wml_state as ws  # noqa: E402
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
    n = node('[scenario]\ncurrent_time=5\nrandom_start_time=yes\n'
             '[time]\nid=dawn\n[/time]\n[time]\nid=day\n[/time]\n[/scenario]\n')
    assert ws.read_tod(n) == (5, True, 2)
    # No schedule: the caller's slot count stands.
    assert ws.read_tod(node('[scenario]\n[/scenario]\n')) == (None, False, 6)
