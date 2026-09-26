"""Preprocessor conditionals are evaluated, the way Wesnoth does
(2026-09-23).

Until this landed the expander resolved no `#ifdef` at all: the
directive lines were kept, the WML parser skipped them, and the
content of every branch survived. That was right for the only
conditional in a scenario we build -- Hornshark Island's, which tests
its own `define=` -- and wrong for the core macros' `#ifdef EASY` /
`NORMAL` / `HARD`, `#ifdef __UNUSED` and `#ifndef MULTIPLAYER`: a
multiplayer game defines none of those, so their content must go.
Rules transcribed from 1.18.4 src/serialization/preprocessor.cpp
:1322-1400.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from wesnoth_ai.rules import scenario_cfg as cfg  # noqa: E402


def _eval(text, defines=("MULTIPLAYER",)):
    return cfg.evaluate_conditionals(text, set(defines)).splitlines()


def test_a_branch_is_live_only_when_its_symbol_is_defined():
    text = "#ifdef MULTIPLAYER\nmp\n#endif\n#ifdef EASY\neasy\n#endif\n"
    assert _eval(text) == ["mp"]


def test_ifndef_negates_and_else_flips():
    text = ("#ifndef MULTIPLAYER\ncampaign\n#else\nmp\n#endif\n"
            "#ifdef HARD\nhard\n#else\nnot hard\n#endif\n")
    assert _eval(text) == ["mp", "not hard"]


def test_conditionals_nest_and_a_dead_parent_wins():
    text = ("#ifdef EASY\n#ifdef MULTIPLAYER\nnever\n#endif\n#endif\n"
            "#ifdef MULTIPLAYER\n#ifndef EASY\nboth\n#endif\n#endif\n")
    assert _eval(text) == ["both"]


def test_a_macro_counts_as_defined_from_the_line_that_defines_it():
    """The engine's define set holds macros too (`parent_.defines_`),
    so `#ifdef SOME_MACRO` turns true once it is #defined -- and a
    #define inside a dead branch defines nothing."""
    text = ("#ifdef LATER\nbefore\n#endif\n"
            "#define LATER\nbody\n#enddef\n"
            "#ifdef LATER\nafter\n#endif\n"
            "#ifdef EASY\n#define GHOST\n#enddef\n#endif\n"
            "#ifdef GHOST\nghost\n#endif\n")
    lines = _eval(text)
    assert "before" not in lines and "after" in lines
    assert "ghost" not in lines


def test_the_define_set_carries_across_files():
    """One set passed through several files is one engine map."""
    defines = {"MULTIPLAYER"}
    cfg.evaluate_conditionals("#define FROM_FILE_A\nx\n#enddef\n", defines)
    assert cfg.evaluate_conditionals(
        "#ifdef FROM_FILE_A\nseen\n#endif\n", defines).splitlines() == ["seen"]


@pytest.mark.parametrize("text", [
    "#else\n", "#endif\n", "#ifdef MULTIPLAYER\nx\n",
    "#ifdef A\n#else\n#else\n#endif\n", "#ifver 1.18\nx\n#endif\n",
    "#ifdef\nx\n#endif\n",
])
def test_malformed_or_unsupported_conditionals_fail_loudly(text):
    with pytest.raises(cfg.PreprocessorError):
        cfg.evaluate_conditionals(text, {"MULTIPLAYER"})


def test_a_scenario_defines_its_own_symbol():
    raw = (Path(__file__).parent.parent / "wesnoth_src" / "data" / "multiplayer"
           / "scenarios" / "2p_Hornshark_Island.cfg").read_text(encoding="utf-8")
    assert cfg.scenario_defines(raw) == {"MULTIPLAYER_HORNSHARK_ISLAND_LOAD"}


def test_difficulty_branches_are_gone_from_a_multiplayer_expansion():
    """QUANTITY picks a value by difficulty with three #ifdef blocks. A
    multiplayer game defines no difficulty, so the engine's QUANTITY
    expands to nothing; ours used to keep all three assignments, the
    last of which won."""
    cfg._CORE_MACROS_CACHE = None
    body = cfg._load_core_macros()["QUANTITY"].body
    assert "{NAME}=" not in body, body


def test_hornshark_still_builds_what_it_did():
    """Its conditionals test its own define, which is set when the
    scenario loads, so every one of them is live: MODIFY_BOWMAN is
    still defined and the Loyalist Bowmen still get firststrike."""
    from wesnoth_ai.rules.scenario_pool import ScenarioSetup, build_scenario_gamestate
    from tools.wesnoth_sim import WesnothSim

    cfg._CORE_MACROS_CACHE = None
    gs = build_scenario_gamestate(ScenarioSetup(
        scenario_id="multiplayer_Hornshark_Island", faction1="Loyalists",
        leader1="Lieutenant", faction2="Loyalists", leader2="Lieutenant",
        fogless=False, tod_start=None))
    sim = WesnothSim(gs, scenario_id="multiplayer_Hornshark_Island")
    bowmen = [u for u in sim.gs.map.units if u.name == "Bowman"]
    assert len(bowmen) == 2
    assert all(any("firststrike" in (a.weapon_specials or set())
                   for a in u.attacks) for u in bowmen)


def test_no_directive_survives_into_a_parsed_scenario():
    """Nothing downstream should ever see a conditional again."""
    from wesnoth_ai.rules.scenario_surface import CORPUS_SCENARIOS

    for scenario_id in CORPUS_SCENARIOS:
        path = cfg.find_scenario_cfg_path(scenario_id)
        assert path is not None, scenario_id
        text = cfg._preprocess_text(
            path.read_text(encoding="utf-8", errors="replace"),
            {"MULTIPLAYER"} | cfg.scenario_defines(
                path.read_text(encoding="utf-8", errors="replace")))
        survivors = [ln for ln in text.splitlines() if cfg._COND_RE.match(ln)]
        assert not survivors, (scenario_id, survivors)
