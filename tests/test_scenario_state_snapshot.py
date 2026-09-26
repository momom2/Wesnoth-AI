"""A characterization pin on the initial state every pool scenario
builds, so that deduplicating the WML readers cannot change a game
without saying so.

The generation path (`scenario_pool.build_scenario_gamestate`, .cfg +
.map) and the reconstruction path (`replay_extract` -> a record ->
`replay_dataset._build_initial_gamestate`) parse the same WML tags with
separate code. A bit-exact replay sweep proves the reconstruction
path; it says nothing about generation, which is how a hardcoded
village gold survived four months after the rule was pinned in the
catalog (BACKLOG.md "The scenario's economy is read from the
scenario"). This test is the other half: it fingerprints the built
state of all 28 pool scenarios under fixed setups.

When a fingerprint changes, the diff of `scenario_state_snapshot.json`
names the scenario and the field. Regenerate deliberately with

    python tests/test_scenario_state_snapshot.py --update

and put the reason in the commit message.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from wesnoth_ai.rules.scenario_pool import (LADDER_SCENARIO_IDS,  # noqa: E402
                                 MINI_MAP_SCENARIO_IDS, ScenarioSetup,
                                 build_scenario_gamestate)

SNAPSHOT = Path(__file__).parent / "data" / "scenario_state_snapshot.json"
# Fixed on purpose: the factions and leaders a scenario is built with
# are the lobby's choice, not the scenario's, so they are held still to
# leave the scenario's own parameters as the only moving part.
SETUP = dict(faction1="Rebels", leader1="Elvish Captain",
             faction2="Loyalists", leader2="Lieutenant",
             fogless=False, tod_start=1)
SCENARIOS = list(LADDER_SCENARIO_IDS) + list(MINI_MAP_SCENARIO_IDS)


def fingerprint(scenario_id: str) -> dict:
    """Every scenario-derived field of the built state, canonically
    ordered. Faction-derived fields (recruit lists) are counted rather
    than listed: they come from the era, not the scenario."""
    gs = build_scenario_gamestate(ScenarioSetup(scenario_id=scenario_id, **SETUP))
    gi = gs.global_info
    units = sorted(
        (u.side, u.position.x, u.position.y, u.name, bool(u.is_leader),
         int(u.max_hp), int(u.max_exp))
        for u in gs.map.units)
    owner = getattr(gi, "_village_owner", None) or {}
    return {
        "sides": [{"gold": s.current_gold, "base_income": s.base_income,
                   "villages": s.nb_villages_controlled,
                   "faction": s.faction, "n_recruits": len(s.recruits)}
                  for s in gs.sides],
        "village_gold": gi.village_gold,
        "village_upkeep": gi.village_upkeep,
        "global_base_income": gi.base_income,
        "experience_modifier": getattr(gi, "_experience_modifier", None),
        "fog": bool(getattr(gi, "_fog", True)),
        "tod_start_offset": getattr(gi, "_tod_start_offset", None),
        "time_of_day": str(gi.time_of_day),
        "map": [gs.map.size_x, gs.map.size_y, len(gs.map.hexes)],
        "units": units,
        "n_units": len(units),
        "pre_owned_villages": sorted((f"{x},{y}", side) for (x, y), side in owner.items()),
        "terrain_digest": hashlib.sha1(
            "".join(sorted(f"{h.position.x},{h.position.y}:"
                           f"{getattr(h, 'terrain_types', '')}"
                           for h in gs.map.hexes)).encode()).hexdigest()[:12],
    }


def _load() -> dict:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


@pytest.mark.parametrize("scenario_id", SCENARIOS)
def test_the_built_state_matches_the_snapshot(scenario_id):
    expected = _load()
    assert scenario_id in expected, (
        f"{scenario_id} has no snapshot; regenerate with --update")
    got = json.loads(json.dumps(fingerprint(scenario_id)))   # tuples -> lists
    assert got == expected[scenario_id], (
        f"{scenario_id}'s built state changed. If that is the point of the "
        f"change, regenerate the snapshot and say why in the commit message.")


def test_the_snapshot_covers_every_pool_scenario():
    """A scenario added to a pool without a snapshot would be
    refactored blind."""
    assert sorted(_load()) == sorted(SCENARIOS)


def test_the_fingerprint_notices_an_economy_change(monkeypatch):
    """The pin has to be able to fail: a village-gold change of the
    kind this file exists to catch must move the fingerprint."""
    import wesnoth_ai.rules.scenario_pool as sp
    before = fingerprint("2p_mini_edited")
    monkeypatch.setattr(sp, "scenario_economy", lambda sid: (None, None, None))
    assert fingerprint("2p_mini_edited") != before


def main() -> int:
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    data = {sid: fingerprint(sid) for sid in SCENARIOS}
    SNAPSHOT.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {SNAPSHOT} ({len(data)} scenarios)")
    return 0


if __name__ == "__main__":
    sys.exit(main() if "--update" in sys.argv else pytest.main([__file__, "-q"]))
