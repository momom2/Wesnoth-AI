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


def test_export_works_when_a_leader_is_dead(tmp_path):
    """DECISIVE games must export. Regression (2026-07-03): the
    [side] renderer read leaders from the FINAL state, so the first
    ever export of a decisive game crashed with "sim has no leader
    for side 1" — the loser's leader is dead at export time. [side]
    describes the STARTING setup; it now uses the sim's
    initial_leaders snapshot."""
    from tools.replay_extract import parse_replay_file
    sim = _build_sim_for("multiplayer_Den_of_Onis")
    dead = next(u for u in sim.gs.map.units if u.is_leader and u.side == 1)
    start_type = dead.name
    sim.gs.map.units.discard(dead)      # simulate a leader kill
    sim.winner, sim.ended_by, sim.done = 2, "leader_killed", True
    out_path = tmp_path / "decisive.bz2"
    export_replay_from_scratch(sim, out_path)
    root = parse_replay_file(out_path)
    sides = root.first("scenario").all("side")
    assert len(sides) == 2
    assert sides[0].attrs.get("type") == start_type, (
        "side 1 must be emitted with its STARTING leader type even "
        "though that leader died")


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


def test_all_mini_maps_emit_without_error(tmp_path):
    """Same guarantee as the 21-ladder-map test, for the tactical-
    training mini pool (Mini Maps Collection add-on). Mini templates
    can ONLY come from tools/build_scenario_templates.py (the game's
    own cfg + the game's own preprocessor) -- no human replays exist
    for these scenarios, so the retired replay-extraction path never
    covered them."""
    from tools.scenario_pool import MINI_MAP_SCENARIO_IDS
    load_factions()
    failures = []
    for sid in MINI_MAP_SCENARIO_IDS:
        try:
            sim = _build_sim_for(sid)
            out = tmp_path / f"{sid}.bz2"
            export_replay_from_scratch(sim, out)
            assert out.exists() and out.stat().st_size > 500
        except Exception as e:
            failures.append(f"{sid}: {type(e).__name__}: {e}")
    assert not failures, "\n".join(failures)


def test_exported_save_pins_tod_start_slot():
    """The emitted [scenario] must carry `current_time=<the slot the
    sim played>`. 1.18.4 tod_manager.cpp:51-55: a present
    current_time disables random_start_time entirely, so playback
    neither re-draws the start slot (Mini_Maps have
    random_start_time=yes) nor consumes an extra synced-RNG draw.
    Without the pin, Wesnoth drew its own slot and every
    ToD-sensitive strike diverged (2026-07-15: tentacle retaliation
    4 sim-dawn vs 3 engine-day on enclave_micro_isar).
    """
    import re
    # Default-schedule map: sim plays slot 0 -> pin 0.
    sim = _build_sim_for("enclave_micro_isar",
                         faction1="Knalgan Alliance", faction2="Rebels")
    wml = build_save_wml(sim)
    m = re.search(r"^\s*current_time=(\d+)", wml, re.MULTILINE)
    assert m and m.group(1) == "0", m
    # Second-watch map ({DEFAULT_SCHEDULE_SECOND_WATCH} emits
    # current_time=5): the fresh build must START there and the
    # export must pin the same slot.
    from tools.scenario_pool import _scenario_tod_start
    assert _scenario_tod_start("multiplayer_Fallenstar_Lake") == 5
    assert _scenario_tod_start("multiplayer_Hamlets") == 0
    sim2 = _build_sim_for("multiplayer_Fallenstar_Lake")
    assert getattr(sim2.gs.global_info, "_tod_start_offset", 0) == 5
    assert sim2.gs.global_info.time_of_day == "second_watch"
    wml2 = build_save_wml(sim2)
    m2 = re.search(r"^\s*current_time=(\d+)", wml2, re.MULTILINE)
    assert m2 and m2.group(1) == "5", m2


def test_tod_start_policy_mirrors_engine():
    """User policy 2026-07-15 == engine behavior: set-time maps
    (ladder) start at their scenario `current_time` (Fallenstar
    Lake / Ruined Passage = 5, everything else 0);
    random_start_time=yes maps (all minis) draw a uniform slot per
    game; a ScenarioSetup.tod_start override forces any slot on any
    map (general capability). Human-derived midgame starts read the
    replay's own current_time via _build_initial_gamestate."""
    import random as _random
    from tools.scenario_pool import sample_tod_start

    rng = _random.Random(123)
    # Set-time maps: constant, whatever the rng says.
    assert all(sample_tod_start("multiplayer_Fallenstar_Lake", rng) == 5
               for _ in range(10))
    assert all(sample_tod_start("multiplayer_Hamlets", rng) == 0
               for _ in range(10))
    # random_start_time=yes minis: uniform draw -- across 60 draws
    # all 6 slots should appear (P(miss one) < 1e-4).
    draws = {sample_tod_start("enclave_micro_isar", rng)
             for _ in range(60)}
    assert draws == {0, 1, 2, 3, 4, 5}, draws
    # random_setup stamps the draw on the setup, and the built game
    # carries it end to end (offset + global time_of_day + export).
    from tools.scenario_pool import random_setup
    setup = None
    rng2 = _random.Random(7)
    while setup is None or setup.tod_start in (None, 0):
        setup = random_setup(rng2, mini_maps=True, forced_faction=None)
    gs = build_scenario_gamestate(setup, experience_modifier=70)
    assert getattr(gs.global_info, "_tod_start_offset", 0) == setup.tod_start
    # Override capability: force second watch on a dawn map.
    factions = load_factions()
    forced = ScenarioSetup(
        scenario_id="multiplayer_Hamlets",
        faction1="Drakes",
        leader1=factions["Drakes"].random_leader_pool[0],
        faction2="Rebels",
        leader2=factions["Rebels"].random_leader_pool[0],
        tod_start=5,
    )
    gs2 = build_scenario_gamestate(forced, experience_modifier=70)
    assert getattr(gs2.global_info, "_tod_start_offset", 0) == 5
    assert gs2.global_info.time_of_day == "second_watch"
    import re
    sim = WesnothSim(gs2, scenario_id="multiplayer_Hamlets", max_turns=2)
    m = re.search(r"^\s*current_time=(\d+)", build_save_wml(sim),
                  re.MULTILINE)
    assert m and m.group(1) == "5", m
