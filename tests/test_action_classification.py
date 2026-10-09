"""Every event-action tag is classified, and the default fails
(2026-09-22).

A tag the core's event dispatch (rust/wesnoth_core/src/events.rs
`apply_action`) has no handler for is counted and warned about by
default, and raises `UnmodelledWML` under `WESNOTH_STRICT_WML`; the
tags that do nothing for a recorded reason are `NO_OP_ACTIONS`.
`[foreach]` and `[unstore_unit]` were once dropped silently -- the
second puts back a unit that `[store_unit kill=yes]` has just removed,
so dropping it deletes units.

These tests are about the CLASSIFICATION, not about any one tag: they
fail when a new tag appears in the pool unclassified.
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.replay_extract import parse_wml  # noqa: E402
from tools.scenario_events import collect_events  # noqa: E402
from wesnoth_ai import game_core as gc  # noqa: E402
from wesnoth_ai.paths import RUST_CORE_SRC_DIR  # noqa: E402
from wesnoth_ai.rules.scenario_cfg import UnmodelledWML  # noqa: E402
from wesnoth_ai.rules.scenario_pool import (LADDER_SCENARIO_IDS,  # noqa: E402
                                            MINI_MAP_SCENARIO_IDS, ScenarioSetup,
                                            build_scenario_gamestate)

pytestmark = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")

POOL = list(LADDER_SCENARIO_IDS) + list(MINI_MAP_SCENARIO_IDS)
TRIGGERS = ["prestart", "start", "turn refresh", "turn 1",
            "side 1 turn", "side 2 turn", "side 3 turn"]


def _build(scenario_id: str):
    return build_scenario_gamestate(ScenarioSetup(
        scenario_id=scenario_id, faction1="Rebels", leader1="Elvish Captain",
        faction2="Loyalists", leader2="Lieutenant", fogless=False,
        tod_start=None))


def _counts():
    import wesnoth_core
    return wesnoth_core.unmodelled_action_counts()


def _reset():
    import wesnoth_core
    wesnoth_core.reset_unmodelled_actions()


def _core_with(event_wml: str, scenario_id: str = "probe"):
    """A Hamlets core whose only events are `event_wml`'s, collected as
    `scenario_id`'s."""
    gs = _build("multiplayer_Hamlets")
    root = parse_wml(f"[multiplayer]\n{event_wml}[/multiplayer]\n")
    gs.global_info._scenario_events = collect_events(root, scenario_id)
    gs.global_info._wml_variables = {}
    return gc.CoreState.from_state(gs)


def _fire(cs, trigger: str) -> None:
    cs.core.fire_events([trigger])
    gc._log_core_warnings()


def _action_event(body: str, name: str = "test") -> str:
    return f"[event]\nname={name}\n{body}[/event]\n"


@pytest.fixture(scope="module")
def fallthrough_census():
    """Set up every scenario on the core, fire its events, and collect
    what had no handler."""
    _reset()
    for scenario_id in POOL:
        cs = gc.CoreState.from_state(_build(scenario_id))
        cs.setup_scenario(scenario_id)
        cs.core.fire_events(TRIGGERS)
    counts = _counts()
    _reset()
    return counts


def test_the_whole_pool_is_classified(fallthrough_census):
    assert fallthrough_census == {}, (
        "these tags fell through on the generation path; classify them or "
        f"write handlers: {sorted(fallthrough_census)}")


def test_an_unknown_tag_is_reported_not_swallowed():
    """The property that was missing. A tag nobody has classified must
    leave a trace."""
    cs = _core_with(_action_event("[teleport_everyone]\n[/teleport_everyone]\n"))
    _reset()
    _fire(cs, "test")
    assert _counts() == {"teleport_everyone": 1}
    _reset()


def test_a_nested_unknown_tag_names_its_scenario(caplog):
    """Tags inside an [if] body dispatch from within a handler. Two
    scenarios hitting the same nested tag must each be reported: reports
    dedupe on (tag, scenario), so a nameless report would silence the
    second one."""
    body = ('[if]\n[variable]\nname=x\nequals=\n[/variable]\n[then]\n'
            '[teleport_everyone]\n[/teleport_everyone]\n[/then]\n[/if]\n')
    _reset()
    with caplog.at_level(logging.WARNING, logger="game_core"):
        for scenario_id in ("scenario_a", "scenario_b"):
            _fire(_core_with(_action_event(body, "prestart"), scenario_id), "prestart")
    reported = [r.getMessage() for r in caplog.records
                if "teleport_everyone" in r.getMessage()]
    assert any("scenario_a" in m for m in reported), reported
    assert any("scenario_b" in m for m in reported), reported
    assert _counts() == {"teleport_everyone": 2}
    _reset()


def _placed_abilities(heals_body: str, scenario_id: str):
    """The abilities of a unit an event places with `[heals]heals_body`."""
    cs = _core_with(_action_event(
        "[unit]\nside=1\ntype=Mermaid Initiate\nx=2\ny=2\n"
        f"[abilities]\n[heals]\n{heals_body}[/heals]\n[/abilities]\n[/unit]\n"), scenario_id)
    before = set(cs.core.unit_ids())
    _fire(cs, "test")
    (placed,) = set(cs.core.unit_ids()) - before
    return cs.core.unit_export(placed)["abilities"]


def test_a_heal_amount_is_read_not_assumed():
    """The two amounts the sim models come from `value=`."""
    assert "heals_4" in _placed_abilities("value=4\n", "heals_four")
    assert "heals_8" in _placed_abilities("value=8\n", "heals_eight")


def test_an_empty_heals_block_heals_nothing_and_says_so(caplog, monkeypatch):
    """The shape our expander produced for Hornshark Island's
    {ABILITY_HEALS} until 2026-09-22: an empty [heals]. The engine heals
    0 for it (heal.cpp:211 builds the effect with a default of 0)."""
    with caplog.at_level(logging.WARNING, logger="game_core"):
        abilities = _placed_abilities("", "empty_heals")
    assert not any(a.startswith("heals") for a in abilities), abilities
    assert any("no value" in r.getMessage() for r in caplog.records)
    monkeypatch.setenv("WESNOTH_STRICT_WML", "1")
    with pytest.raises(UnmodelledWML):
        _placed_abilities("", "empty_heals_strict")


def test_hornshark_mermaids_still_heal_four():
    """Hornshark Island's preplaced Mermaid Initiates are granted
    {ABILITY_HEALS} and carry heals_4. Built through the production
    path, both sides Rebels (the faction whose [case] places them)."""
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
    cs = _core_with(_action_event("[teleport_everyone]\n[/teleport_everyone]\n"))
    with pytest.raises(UnmodelledWML):
        cs.core.fire_events(["test"])
    _reset()


def _no_op_actions():
    text = (RUST_CORE_SRC_DIR / "events.rs").read_text(encoding="utf-8")
    block = re.search(r"const NO_OP_ACTIONS: \[&str; \d+\] = \[(.*?)\];", text, re.S).group(1)
    return re.findall(r'"([a-z_]+)"', block)


def test_a_classified_tag_is_silent():
    """The other half: classifying a tag must actually suppress the
    report, or the strict mode is unusable."""
    tags = _no_op_actions()
    assert "end_turn" in tags and "message" in tags
    cs = _core_with(_action_event("".join(f"[{t}]\n[/{t}]\n" for t in tags)))
    _reset()
    _fire(cs, "test")
    assert _counts() == {}


def test_the_end_turn_substitution_precondition_holds():
    """`end_turn` is classified as doing nothing on the grounds that the
    pool's only use sits on a `controller=null` side, which never takes
    a turn. That is a claim about the scenarios, so it is checked
    against them: if a scenario ever ends the turn of a side we DO run,
    the classification is wrong and this fails."""
    from wesnoth_ai.rules.expansion_diff import TEMPLATES, _scenario_block

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
