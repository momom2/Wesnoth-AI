"""The hidden-units oracle's simulator side (tools/hidden_units_oracle.py).

The live half needs Wesnoth; this pins the half that does not: the
scripted board builds in the simulator, the hide-cover rule decides
the probe's outcome the way the engine did on 2026-09-20 (the record
under training/metrics/fidelity/), and the setup file the Lua side
reads is well formed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from tools.hidden_units_oracle import (  # noqa: E402
    HIDER, MOVER_START, build_cases, map_data, setup_lua, sim_predictions,
)


def _case(name: str):
    return next(c for c in build_cases() if c.name == name)


def test_the_board_and_the_setup_file_are_well_formed():
    case = _case("ambush+Gg^Fp")
    rows = map_data(case).split("\n")[3:]
    assert len(rows) == 16 and all(len(r.split(", ")) == 16 for r in rows)
    assert rows[HIDER[1]].split(", ")[HIDER[0]] == "Gg^Fp"        # Wesnoth coords index the bordered grid
    lua = setup_lua(case)
    assert lua.startswith("return {") and 'code="Gg^Fp"' in lua and "tod_index = 1" in lua
    assert f"x={MOVER_START[0]}, y={MOVER_START[1]}" in lua and "canrecruit=true" in lua


def test_the_simulator_stops_the_mover_next_to_a_covered_hider_and_not_otherwise():
    covered = sim_predictions(_case("ambush+Gg^Fp"))
    assert covered["visible_start"] == ["u1"]                      # the ranger is hidden
    assert covered["moves"][0]["landed"] == [8, 6] and covered["moves"][0]["stop_reason"] == "ambush"
    assert "u2" in covered["moves"][0]["visible_after"]            # uncovered by the ambush
    plain = sim_predictions(_case("ambush-Gg^Gvs"))                # farmland is not a village or a forest
    assert "u2" in plain["visible_start"]
    assert plain["moves"][0]["landed"] == [8, 9] and plain["moves"][0]["stop_reason"] != "ambush"


def test_nightstalk_follows_the_time_of_day_and_illumination():
    assert sim_predictions(_case("nightstalk@first_watch"))["visible_start"] == ["u1"]
    assert "u2" in sim_predictions(_case("nightstalk@morning"))["visible_start"]
    assert "u2" in sim_predictions(_case("nightstalk@first_watch+illuminated"))["visible_start"]
    assert "u2" not in sim_predictions(_case("nightstalk@first_watch+mage_two_away"))["visible_start"]
