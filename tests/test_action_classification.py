"""Every event-action tag is classified, and the default fails
(2026-09-22).

`_apply_action` used to fall back to a no-op, so a tag with no handler
was indistinguishable from one we had decided to ignore. `[foreach]`
and `[unstore_unit]` were dropped that way -- the second puts back a
unit that `[store_unit kill=yes]` has just removed, so dropping it
deletes units -- and nothing said so. The fallback is now an error
path: warn by default, raise under `WESNOTH_STRICT_WML`.

These tests are about the CLASSIFICATION, not about any one tag: they
fail when a new tag appears in the pool unclassified, and when an entry
is added without a reason.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import tools.scenario_events as se  # noqa: E402
from wesnoth_ai.rules.scenario_cfg import UnmodelledWML  # noqa: E402
from tools.replay_extract import parse_wml  # noqa: E402
from wesnoth_ai.rules.scenario_pool import (LADDER_SCENARIO_IDS,  # noqa: E402
                                 MINI_MAP_SCENARIO_IDS, ScenarioSetup,
                                 build_scenario_gamestate)

POOL = list(LADDER_SCENARIO_IDS) + list(MINI_MAP_SCENARIO_IDS)
TRIGGERS = ("prestart", "start", "turn refresh", "turn 1",
            "side 1 turn", "side 2 turn", "side 3 turn")


def _build(scenario_id: str):
    return build_scenario_gamestate(ScenarioSetup(
        scenario_id=scenario_id, faction1="Rebels", leader1="Elvish Captain",
        faction2="Loyalists", leader2="Lieutenant", fogless=False,
        tod_start=None))


@pytest.fixture(scope="module")
def fallthrough_census():
    """Fire every scenario's events and collect what had no handler."""
    se.reset_unmodelled_actions()
    for scenario_id in POOL:
        gs = _build(scenario_id)
        events = getattr(gs.global_info, "_scenario_events", []) or []
        for trigger in TRIGGERS:
            se.fire_event(gs, events, trigger)
    counts = se.unmodelled_action_counts()
    se.reset_unmodelled_actions()
    return counts


def test_the_whole_pool_is_classified(fallthrough_census):
    assert fallthrough_census == {}, (
        "these tags fell through on the generation path; classify them or "
        f"write handlers: {sorted(fallthrough_census)}")


def test_an_unknown_tag_is_reported_not_swallowed():
    """The property that was missing. A tag nobody has classified must
    leave a trace."""
    gs = _build("multiplayer_Hamlets")
    se.reset_unmodelled_actions()
    se._apply_action(gs, parse_wml("[teleport_everyone]\n[/teleport_everyone]\n")
                     .first("teleport_everyone"), "multiplayer_Hamlets")
    assert se.unmodelled_action_counts() == {"teleport_everyone": 1}
    se.reset_unmodelled_actions()


def test_a_nested_unknown_tag_names_its_scenario(caplog):
    """Tags inside an [if] body dispatch from within a handler that is
    never told the scenario. Two scenarios hitting the same nested tag
    must each be reported: reports dedupe on (tag, scenario), so a
    nameless report would silence the second one."""
    import logging

    event_wml = ('[event]\nname=prestart\n[if]\n[variable]\nname=x\n'
                 'equals=\n[/variable]\n[then]\n[teleport_everyone]\n'
                 '[/teleport_everyone]\n[/then]\n[/if]\n[/event]\n')
    se.reset_unmodelled_actions()
    with caplog.at_level(logging.WARNING, logger="scenario_events"):
        for scenario_id in ("scenario_a", "scenario_b"):
            gs = _build("multiplayer_Hamlets")
            event = se.ScenarioEvent(
                name="prestart",
                actions=list(parse_wml(event_wml).first("event").children),
                scenario_id=scenario_id)
            se.fire_event(gs, [event], "prestart")
    reported = [r.getMessage() for r in caplog.records
                if "teleport_everyone" in r.getMessage()]
    assert any("scenario_a" in m for m in reported), reported
    assert any("scenario_b" in m for m in reported), reported
    assert se.unmodelled_action_counts() == {"teleport_everyone": 2}
    se.reset_unmodelled_actions()


def _heals(body: str):
    return parse_wml(f"[heals]\n{body}[/heals]\n").first("heals")


def test_a_heal_amount_is_read_not_assumed():
    """The two amounts the sim models come from `value=`."""
    assert se._heals_ability(_heals("value=4\n")) == "heals_4"
    assert se._heals_ability(_heals("value=8\n")) == "heals_8"


def test_an_empty_heals_block_heals_nothing_and_says_so(caplog, monkeypatch):
    """The shape our expander produced for Hornshark Island's
    {ABILITY_HEALS} until 2026-09-22: an empty [heals]. The engine heals
    0 for it (heal.cpp:211 builds the effect with a default of 0); this
    reader used to assume 4, which matched the macro by luck and would
    have halved an {ABILITY_HEALS_8}."""
    import logging

    with caplog.at_level(logging.WARNING, logger="scenario_events"):
        assert se._heals_ability(_heals("")) is None
    assert any("no value" in r.getMessage() for r in caplog.records)
    monkeypatch.setenv("WESNOTH_STRICT_WML", "1")
    with pytest.raises(UnmodelledWML):
        se._heals_ability(_heals(""))


def test_hornshark_mermaids_still_heal_four():
    """The fix must not move a real game: Hornshark Island's preplaced
    Mermaid Initiates are granted {ABILITY_HEALS} and must still carry
    heals_4. Built through the production path, both sides Rebels (the
    faction whose [case] places them)."""
    from tools.wesnoth_sim import WesnothSim

    gs = build_scenario_gamestate(ScenarioSetup(
        scenario_id="multiplayer_Hornshark_Island", faction1="Rebels",
        leader1="Elvish Captain", faction2="Rebels",
        leader2="Elvish Captain", fogless=False, tod_start=None))
    sim = WesnothSim(gs, scenario_id="multiplayer_Hornshark_Island")
    mermaids = [u for u in sim.gs.map.units if u.name == "Mermaid Initiate"]
    assert len(mermaids) == 2
    assert all("heals_4" in u.abilities for u in mermaids)


def test_strict_mode_refuses_an_unknown_tag(monkeypatch):
    """What a box run or a sweep can demand: no unmodelled WML at all."""
    monkeypatch.setenv("WESNOTH_STRICT_WML", "1")
    gs = _build("multiplayer_Hamlets")
    node = parse_wml("[teleport_everyone]\n[/teleport_everyone]\n").first(
        "teleport_everyone")
    with pytest.raises(UnmodelledWML):
        se._apply_action(gs, node, "multiplayer_Hamlets")
    se.reset_unmodelled_actions()


def test_a_classified_tag_is_silent():
    """The other half: classifying a tag must actually suppress the
    report, or the strict mode is unusable."""
    gs = _build("multiplayer_Hamlets")
    se.reset_unmodelled_actions()
    for tag in sorted(set(se._IGNORED_ACTIONS) | set(se._SUBSTITUTED_ACTIONS)):
        se._apply_action(gs, parse_wml(f"[{tag}]\n[/{tag}]\n").first(tag), "x")
    assert se.unmodelled_action_counts() == {}


def test_every_classified_tag_carries_a_reason():
    for table in (se._IGNORED_ACTIONS, se._SUBSTITUTED_ACTIONS):
        for tag, reason in table.items():
            assert reason.strip(), f"[{tag}] is classified with no reason"


def test_the_three_tables_are_disjoint():
    """A tag in two tables means two people decided differently and one
    of them is dead code."""
    handled = set(se._ACTION_HANDLERS)
    ignored = set(se._IGNORED_ACTIONS)
    substituted = set(se._SUBSTITUTED_ACTIONS)
    assert not handled & ignored, handled & ignored
    assert not handled & substituted, handled & substituted
    assert not ignored & substituted, ignored & substituted


def test_the_end_turn_substitution_precondition_holds():
    """`end_turn` is classified SUBSTITUTED on the grounds that the
    pool's only use sits on a `controller=null` side, which never takes
    a turn. That is a claim about the scenarios, so it is checked
    against them: if a scenario ever ends the turn of a side we DO run,
    the classification is wrong and this fails."""
    from tools.analysis.expansion_diff import TEMPLATES, _scenario_block

    checked = 0
    for scenario_id in POOL:
        path = TEMPLATES / f"{scenario_id}.wml"
        block = _scenario_block(parse_wml(
            path.read_text(encoding="utf-8", errors="replace")))
        controllers = {
            (s.attrs.get("side", "") or "").strip().strip('"'):
                (s.attrs.get("controller", "") or "").strip().strip('"')
            for s in block.all("side")}
        for event in block.all("event"):
            if not event.all("end_turn"):
                continue
            name = (event.attrs.get("name", "") or "").strip().strip('"')
            assert name.startswith("side ") and name.endswith(" turn"), name
            side = name.split()[1]
            assert controllers.get(side) == "null", (
                f"{scenario_id}: [end_turn] on side {side}, whose controller "
                f"is {controllers.get(side)!r} -- it is not inert")
            checked += 1
    assert checked == 1, f"expected the one known [end_turn], found {checked}"
