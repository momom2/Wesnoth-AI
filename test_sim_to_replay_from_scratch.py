"""From-scratch save emitter (no source bz2): tests.

`tools.sim_to_replay.export_replay_from_scratch` composes a complete
Wesnoth-loadable `.bz2` from:
  - tools/templates/wesnoth_save_scaffold.wml  (era + outer scaffolding)
  - tools/templates/wesnoth_scenario_tail.wml  (ToD + music + events)
  - wesnoth_src/data/multiplayer/scenarios/2p_<MapName>.cfg
  - wesnoth_src/data/multiplayer/maps/2p_<MapName>.map
  - sim state (factions / leaders / starting gold) + sim.command_history

The contract:
  1. All 21 Ladder Era scenarios resolve to a .cfg + .map (the id-to-
     filename mapping isn't fully mechanical; Caves of the Basilisk's
     id is `multiplayer_Basilisk` but its file is
     `2p_Caves_of_the_Basilisk.cfg`).
  2. The emitted bz2 round-trips through our own WML parser
     (`tools.replay_extract.parse_replay_file`): every top-level
     block we emit is present + well-formed.
  3. Per-side faction / leader / recruit / starting-gold matches the
     sim state (not the source scenario's defaults).
  4. The [replay] section preserves the sim's commands.

Wesnoth-side loadability is not exercised by pytest (would need a
Wesnoth subprocess), but the user-verification path is documented:
generate a fresh export, copy to the saves dir, load via
File -> Load Game -> Replays.

Dependencies: tools.sim_to_replay, tools.scenario_pool,
              tools.wesnoth_sim, tools.replay_extract.
Dependents: regression CI.
"""
from __future__ import annotations

import bz2
from pathlib import Path

import pytest

from tools.scenario_pool import (
    LADDER_SCENARIO_IDS, ScenarioSetup,
    build_scenario_gamestate, load_factions,
)
from tools.sim_to_replay import (
    _scenario_cfg_path, _scrape_scenario_metadata, _load_map_data,
    _scrape_map_keep_positions, _scrape_scenario_starting_gold,
    build_save_wml, export_replay_from_scratch,
)
from tools.wesnoth_sim import PvPDefaults, WesnothSim


# ---------------------------------------------------------------------
# Scenario-template lookup
# ---------------------------------------------------------------------

def test_every_ladder_scenario_resolves_to_a_cfg():
    """All 21 ladder maps must have a .cfg in wesnoth_src/. Failure
    means either the scenario_id table drifted from wesnoth_src or
    a .cfg got renamed -- either way we can't emit for that map."""
    missing = []
    for sid in LADDER_SCENARIO_IDS:
        if _scenario_cfg_path(sid) is None:
            missing.append(sid)
    assert not missing, f"no .cfg for scenarios: {missing}"


def test_every_ladder_scenario_has_a_map_file():
    """Each .cfg must reference a .map that loads."""
    failures = []
    for sid in LADDER_SCENARIO_IDS:
        cfg = _scenario_cfg_path(sid)
        meta = _scrape_scenario_metadata(cfg)
        mf = meta.get("map_file")
        if not mf:
            failures.append(f"{sid}: no map_file attr")
            continue
        try:
            _load_map_data(mf)
        except Exception as e:
            failures.append(f"{sid}: map load failed: {e}")
    assert not failures, "\n".join(failures)


def test_every_ladder_scenario_has_two_keep_markers():
    """Every map must have `1 K*` and `2 K*` keep markers (side 1
    and side 2 starting positions). Without these, Wesnoth can't
    place the leaders at game start."""
    failures = []
    for sid in LADDER_SCENARIO_IDS:
        cfg = _scenario_cfg_path(sid)
        meta = _scrape_scenario_metadata(cfg)
        md = _load_map_data(meta["map_file"])
        keeps = _scrape_map_keep_positions(md)
        if 1 not in keeps or 2 not in keeps:
            failures.append(f"{sid}: keeps={keeps}")
    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------
# Build a sim + emit + parse
# ---------------------------------------------------------------------

def _build_sim_for(scenario_id: str, faction1="Drakes", faction2="Rebels",
                   max_turns: int = 2) -> WesnothSim:
    """Construct a `WesnothSim` for the given scenario + factions.
    Doesn't run the sim -- just builds the initial state. Tests can
    `sim.step` if they want to populate command_history."""
    factions = load_factions()
    setup = ScenarioSetup(
        scenario_id=scenario_id,
        faction1=faction1,
        leader1=factions[faction1].random_leader_pool[0],
        faction2=faction2,
        leader2=factions[faction2].random_leader_pool[0],
    )
    gs = build_scenario_gamestate(setup, experience_modifier=70)
    return WesnothSim(gs, scenario_id=scenario_id, max_turns=max_turns)


def test_build_save_wml_for_hamlets_parses_back():
    """Smoke: build a save for Hamlets, parse it with our own
    parser, verify all the expected top-level blocks are present."""
    sim = _build_sim_for("multiplayer_Hamlets")
    save_wml = build_save_wml(sim)
    # Top-level structure check via substring matching (cheap; the
    # parser test below covers detailed structure).
    for block in (
        "[scenario]", "[/scenario]",
        "[carryover_sides_start]", "[/carryover_sides_start]",
        "[multiplayer]", "[/multiplayer]",
        "[era]", "[/era]",
        "[replay]", "[/replay]",
    ):
        assert block in save_wml, (
            f"missing block {block!r} from from-scratch save WML"
        )
    # Scenario id propagated.
    assert 'id="multiplayer_Hamlets"' in save_wml


def test_emitted_save_parses_with_our_wml_parser(tmp_path):
    """Generated `.bz2` must round-trip through
    `tools.replay_extract.parse_replay_file`. If it doesn't, our
    own pipeline (extract -> recon -> diff) couldn't process the
    sim's exports."""
    from tools.replay_extract import parse_replay_file
    sim = _build_sim_for("multiplayer_Fallenstar_Lake")
    out_path = tmp_path / "test.bz2"
    export_replay_from_scratch(sim, out_path)
    root = parse_replay_file(out_path)
    top_tags = [c.tag for c in root.children]
    # Order: initial empty [replay], [scenario], [carryover_sides_start],
    # [multiplayer], [statistics], [era], [replay] (commands).
    assert top_tags == [
        "replay", "scenario", "carryover_sides_start",
        "multiplayer", "statistics", "era", "replay",
    ], f"unexpected top-level structure: {top_tags}"


def test_sides_reflect_sim_factions_and_leaders(tmp_path):
    """The per-side `[side]` blocks must use the sim's actual
    faction/leader/recruit, not the scenario template's defaults."""
    from tools.replay_extract import parse_replay_file
    sim = _build_sim_for(
        "multiplayer_Den_of_Onis",
        faction1="Northerners",
        faction2="Loyalists",
    )
    out_path = tmp_path / "test.bz2"
    export_replay_from_scratch(sim, out_path)
    root = parse_replay_file(out_path)
    scn = root.first("scenario")
    assert scn is not None
    sides = scn.all("side")
    assert len(sides) == 2
    s1, s2 = sides
    assert s1.attrs.get("faction") == "Northerners"
    assert s2.attrs.get("faction") == "Loyalists"
    # The leader type comes from the random_leader_pool[0]; just
    # check it's non-empty and matches the sim's leader.
    sim_leaders = {u.side: u.name for u in sim.gs.map.units if u.is_leader}
    assert s1.attrs.get("type") == sim_leaders[1]
    assert s2.attrs.get("type") == sim_leaders[2]


def test_starting_gold_from_cfg_overrides_pvp_default(tmp_path):
    """Arcanclave Citadel's .cfg sets `gold=175`. The emitted save
    must reflect that, not the PvP default 100."""
    from tools.replay_extract import parse_replay_file
    sim = _build_sim_for("multiplayer_Arcanclave_Citadel")
    out_path = tmp_path / "test.bz2"
    export_replay_from_scratch(sim, out_path)
    root = parse_replay_file(out_path)
    scn = root.first("scenario")
    for side in scn.all("side"):
        assert side.attrs.get("gold") == "175", (
            f"Arcanclave side gold should be 175 (from .cfg), got "
            f"{side.attrs.get('gold')}"
        )


def test_all_21_ladder_maps_emit_without_error(tmp_path):
    """Generate a tiny save for every ladder map -- catches per-map
    regressions in the scenario-template walking (case-insensitive
    id matches, weird map_file paths, missing keep markers, etc.)."""
    factions = load_factions()
    failures = []
    for sid in LADDER_SCENARIO_IDS:
        try:
            sim = _build_sim_for(sid)
            out = tmp_path / f"{sid}.bz2"
            export_replay_from_scratch(sim, out)
            assert out.exists() and out.stat().st_size > 1000
        except Exception as e:
            failures.append(f"{sid}: {type(e).__name__}: {e}")
    assert not failures, "\n".join(failures)


def test_scrape_starting_gold_arcanclave():
    """Direct unit test for the .cfg gold scraper. Arcanclave
    specifies 175 per side; Hamlets specifies neither."""
    arc = _scrape_scenario_starting_gold(
        _scenario_cfg_path("multiplayer_Arcanclave_Citadel"))
    assert arc == {1: 175, 2: 175}, arc
    ham = _scrape_scenario_starting_gold(
        _scenario_cfg_path("multiplayer_Hamlets"))
    assert ham == {}, ham
