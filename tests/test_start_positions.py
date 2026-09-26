#!/usr/bin/env python3
"""A map cell's starting-position label is a STRING, not one digit.

`string_to_number_` (wesnoth_src/src/terrain/translation.cpp:743-756,
1.18.4 tag) trims the cell, then cuts everything up to each remaining
space into `start_positions`:

    utils::trim(str);
    ...
    std::size_t offset = str.find(' ', 0);
    while(offset != std::string::npos) {
        start_positions.push_back(std::string(str.substr(0, offset)));
        str.remove_prefix(offset + 1);
        offset = str.find(' ', 0);
    }

`read_game_map` (translation.cpp:317-327) stores each label in a
string-keyed bimap, and `gamemap_base::starting_position`
(wesnoth_src/src/map/map.cpp:324-327) asks for side N by
`std::to_string(n)`. So a label may be a multi-digit side ("10 Wo") or
a name no side owns ("lake Gs^Vc" in data/test/maps/
simple_find_path.map, "book_start Isc^Ii" in
campaigns/Descent_Into_Darkness/maps/07c_A_Small_Favor3.map).

Until 2026-09-13 six places read exactly one digit. Two lost the
terrain (the whole "10 Wo" became the base code), two read side 1 off
"10 Kh", and the [terrain] event rewrite dropped the label from the
exported map_data. No tracked map or sampled replay carries such a
label today -- these tests are the guard for the day one does.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from wesnoth_ai.rules.terrain_resolver import (  # noqa: E402
    split_start_position, start_position_side, strip_start_position,
)

# Border ring + two labelled hexes: side 10 at playable (0, 0) and a
# named location at playable (1, 1) that belongs to no side.
MAP_DATA = "\n".join([
    "Gg, Gg, Gg, Gg",
    "Gg, 10 Kh, Gg, Gg",
    "Gg, Gg, lake Gs^Vc, Gg",
    "Gg, Gg, Gg, Gg",
])


def test_the_label_is_any_text_before_a_space():
    assert split_start_position("1 Gg^Fp") == ("1", "Gg^Fp")
    assert split_start_position("10 Wo") == ("10", "Wo")
    assert split_start_position("lake Gs^Vc") == ("lake", "Gs^Vc")
    assert split_start_position("Gg^Fp") == ("", "Gg^Fp")
    # Several labels on one hex: the engine pushes one per space.
    assert split_start_position("1 2 Gg^Vh") == ("1 2", "Gg^Vh")
    assert split_start_position("1  Gg") == ("1 ", "Gg")
    # The trim runs BEFORE the split, so a leading space is not a label
    # and a lone trailing space leaves the cell as its own terrain.
    assert split_start_position(" 1 Gg^Fp") == ("1", "Gg^Fp")
    assert split_start_position("  Gg  ") == ("", "Gg")
    assert split_start_position("1 ") == ("", "1")
    assert split_start_position("") == ("", "") and split_start_position(None) == ("", "")


def test_strip_returns_the_code_half():
    assert strip_start_position("10 Wo") == "Wo"
    assert strip_start_position("book_start Isc^Ii") == "Isc^Ii"
    assert strip_start_position("Gg") == "Gg"


def test_only_a_side_s_own_spelling_names_a_side():
    """`starting_position(n)` asks for `std::to_string(n)` (map.cpp:
    324-327), so "10" is side 10 while "01" and a name are special
    locations no side starts on."""
    assert start_position_side("1") == 1
    assert start_position_side("10") == 10
    assert start_position_side("01") is None
    assert start_position_side("lake") is None
    assert start_position_side("P1_Burner") is None
    assert start_position_side("") is None and start_position_side(None) is None


def test_scenario_pool_reads_a_two_digit_side_and_skips_a_name():
    from tools.scenario_pool import extract_player_starts
    from wesnoth_ai.classes import Position

    starts = extract_player_starts(MAP_DATA)
    assert starts == {10: Position(x=0, y=0)}, \
        "side 10 starts at the labelled hex; 'lake' is not a side"


def test_replay_extract_reads_the_same_labels():
    from tools.replay_extract import _parse_map_starting_positions

    assert _parse_map_starting_positions(MAP_DATA) == {10: (0, 0)}


def test_a_terrain_event_keeps_the_start_label():
    """A [terrain] event replaces the terrain, never the hex's label:
    the engine writes the cell back as label + " " + code
    (number_to_string_, translation.cpp:775-782). Splicing a fixed two
    characters recognized only a one-digit label, so rewriting "10 Kh"
    or "lake Gs^Vc" silently dropped the start position from the
    map_data we export."""
    from tools.replay_extract import WMLNode
    from tools.scenario_events import _terrain_action
    from wesnoth_ai.classes import GameState, GlobalInfo, Hex, Map, Position

    gi = GlobalInfo(current_side=1, turn_number=1, time_of_day="dawn",
                    village_gold=2, village_upkeep=1, base_income=2)
    setattr(gi, "_raw_map_data", MAP_DATA)
    setattr(gi, "_terrain_codes", {(0, 0): "Kh", (1, 1): "Gs^Vc"})
    hexes = {Hex(position=Position(x=x, y=y), terrain_types=set(), modifiers=set())
             for x in range(2) for y in range(2)}
    gs = GameState(
        game_id="t", global_info=gi, sides=[],
        map=Map(size_x=2, size_y=2, mask=set(), fog=set(), hexes=hexes, units=set()),
    )

    # WML (1,1) is the labelled keep; (2,2) is the named location.
    action = WMLNode("terrain")
    action.attrs = {"x": "1,2", "y": "1,2", "terrain": "Gg^Fp"}
    _terrain_action(gs, action)

    rows = [[c.strip() for c in row.split(",")]
            for row in getattr(gs.global_info, "_raw_map_data").splitlines()]
    assert rows[1][1] == "10 Gg^Fp", "side 10 still starts on that hex"
    assert rows[2][2] == "lake Gg^Fp", "the named location survives the morph"
    # The terrain code the resolvers read carries no label.
    codes = getattr(gs.global_info, "_terrain_codes")
    assert codes[(0, 0)] == "Gg^Fp" and codes[(1, 1)] == "Gg^Fp"
