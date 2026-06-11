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


def test_aethermaw_emit_carries_morph_events(tmp_path):
    """Aethermaw morphs terrain on turns 4-6 via [event] name='side N
    turn M' blocks defined in the scenario .cfg. The per-scenario
    template extraction must propagate those events into our emitted
    save -- otherwise Wesnoth playback wouldn't mutate the map and
    the recorded sim commands (which moved through morphed hexes)
    would src_missing or final_occupied.

    Concrete check: count [event] sub-blocks in the emitted
    [scenario]. Aethermaw has 13 events total in the post-expansion
    save (3 prestart + 10 turn-keyed). Below 10 means morph events
    didn't propagate.
    """
    from tools.replay_extract import parse_replay_file
    sim = _build_sim_for("multiplayer_Aethermaw")
    out_path = tmp_path / "test.bz2"
    export_replay_from_scratch(sim, out_path)
    root = parse_replay_file(out_path)
    scn = root.first("scenario")
    events = scn.all("event")
    assert len(events) >= 10, (
        f"Aethermaw save missing morph events: only {len(events)} "
        f"[event] blocks in [scenario]; expected at least 10 "
        f"(3 prestart + 7 side-N-turn-M morph triggers)"
    )
    # At least one of the events must contain a [terrain] action
    # (the actual morph instruction).
    terrain_events = [e for e in events if e.first("terrain")]
    assert terrain_events, (
        "no [event] block carries a [terrain] mutation -- Aethermaw "
        "morph won't fire during Wesnoth playback"
    )


def test_emit_side_order_is_ascending(tmp_path):
    """Wesnoth processes [side] blocks in order and expects them
    in ascending side-number order (side=1, side=2, side=3, ...).
    If we splice our player sides AFTER the template's scenery side
    (which was the bug user reported -- CoB and TSG side 2 leader
    failing to spawn), the order becomes (3, 1, 2) and Wesnoth
    drops side 2.

    Regression: for every map with a scenery side, verify the
    emitted side order is monotonically ascending."""
    from tools.replay_extract import parse_replay_file
    for sid in ("multiplayer_Basilisk",
                "multiplayer_thousand_stings_garrison",
                "multiplayer_Sullas_Ruins"):
        sim = _build_sim_for(sid)
        out = tmp_path / f"{sid}.bz2"
        export_replay_from_scratch(sim, out)
        root = parse_replay_file(out)
        scn = root.first("scenario")
        side_nums = [int(s.attrs.get("side", 0)) for s in scn.all("side")]
        assert side_nums == sorted(side_nums), (
            f"{sid}: [side] blocks out of order: {side_nums}; "
            f"Wesnoth requires ascending side numbers"
        )


def test_emit_preserves_scenery_side_units(tmp_path):
    """Scenery sides (side 3+) carry pre-placed statues / decorations
    that several ladder maps depend on: Caves of the Basilisk's 15
    petrified victims, Thousand Stings Garrison's 66 frozen
    scorpions, Sullas Ruins' 5 stone-mage statues. The
    `_strip_player_side_blocks` extractor must KEEP these (it strips
    only sides 1+2 -- the player sides we replace at emit time).
    Without the scenery side, Wesnoth playback renders the maps
    without their iconic statues and lets units walk through hexes
    that should be blocked.
    """
    from tools.replay_extract import parse_replay_file
    cases = {
        "multiplayer_Basilisk":                  ("Petrified Basilisk's victims", 10),
        "multiplayer_thousand_stings_garrison":  ("TSG statues",                  30),
        "multiplayer_Sullas_Ruins":              ("Sulla's stone mages",          3),
    }
    for sid, (label, min_units) in cases.items():
        sim = _build_sim_for(sid)
        out = tmp_path / f"{sid}.bz2"
        export_replay_from_scratch(sim, out)
        root = parse_replay_file(out)
        scn = root.first("scenario")
        sides = scn.all("side")
        # Player sides 1+2 (rendered by our emitter) + at least one
        # scenery side 3+.
        scenery_sides = [s for s in sides
                         if int(s.attrs.get("side", 0)) >= 3]
        assert scenery_sides, (
            f"{sid} ({label}): no scenery side 3+ in emitted save -- "
            f"statues won't render in Wesnoth playback"
        )
        scenery_units = sum(
            len(s.all("unit")) for s in scenery_sides)
        assert scenery_units >= min_units, (
            f"{sid} ({label}): scenery side has only {scenery_units} "
            f"units, expected at least {min_units}"
        )


def test_emit_preserves_top_level_block_structure(tmp_path):
    """Splicing must not run two top-level blocks onto one line:
    `[/scenario][carryover_sides_start]` would be unparseable.
    Verify the emitted save has exactly the 7 top-level blocks we
    expect, in order."""
    from tools.replay_extract import parse_replay_file
    sim = _build_sim_for("multiplayer_Aethermaw")
    out_path = tmp_path / "test.bz2"
    export_replay_from_scratch(sim, out_path)
    root = parse_replay_file(out_path)
    top = [c.tag for c in root.children]
    assert top == [
        "replay", "scenario", "carryover_sides_start",
        "multiplayer", "statistics", "era", "replay",
    ], f"top-level structure regressed: {top}"


def test_scrape_starting_gold_arcanclave():
    """Direct unit test for the .cfg gold scraper. Arcanclave
    specifies 175 per side; Hamlets specifies neither."""
    arc = _scrape_scenario_starting_gold(
        _scenario_cfg_path("multiplayer_Arcanclave_Citadel"))
    assert arc == {1: 175, 2: 175}, arc
    ham = _scrape_scenario_starting_gold(
        _scenario_cfg_path("multiplayer_Hamlets"))
    assert ham == {}, ham
