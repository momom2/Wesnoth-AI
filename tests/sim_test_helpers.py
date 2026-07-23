"""From-scratch sim builders for the test suite.

Replaces the retired corpus bootstrap (tests used to seed sims from
`replays_dataset/*.json.gz`; the corpus is permanently retired, user
decision 2026-06-12). Sims are built the same way production
self-play builds them: `tools.scenario_pool` setup + gamestate from
the scenario .cfg / .map under `wesnoth_src/data/`.

Not a test module (name deliberately not `test_*`); imported by
test_sim_determinism / test_sim_advance / test_parallel_rollouts /
test_recruit_rejection / test_sim_self_play_smoke.
"""
from __future__ import annotations

import copy
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from tools.scenario_pool import (   # noqa: E402
    ScenarioSetup, build_scenario_gamestate, load_factions,
    random_setup,
)
from tools.wesnoth_sim import WesnothSim   # noqa: E402

_FACTIONS_DIR = (Path(__file__).parent.parent / "wesnoth_src" / "data"
                 / "multiplayer" / "factions")


def require_scenario_data() -> None:
    """Skip the calling test when the vendored Wesnoth data tree is
    absent (wesnoth_src/ is environment, not repo content — see
    CLAUDE.md "Wesnoth data provenance" for the refresh command)."""
    if not _FACTIONS_DIR.is_dir():
        pytest.skip("wesnoth_src/data missing — refresh from the "
                    "Steam install (see CLAUDE.md)")


def scenario_setup(seed: int = 0, *, mini: bool = False,
                   scenario_id: Optional[str] = None) -> ScenarioSetup:
    """Deterministic-for-a-seed scenario + faction/leader draw, with
    an optional scenario override."""
    require_scenario_data()
    load_factions()
    setup = random_setup(random.Random(seed), forced_faction=None,
                         mini_maps=mini)
    if scenario_id is not None:
        setup = ScenarioSetup(
            scenario_id=scenario_id,
            faction1=setup.faction1, leader1=setup.leader1,
            faction2=setup.faction2, leader2=setup.leader2,
        )
    return setup


def fresh_scenario_sim(seed: int = 0, *, max_turns: int = 6,
                       mini: bool = False,
                       scenario_id: Optional[str] = None) -> WesnothSim:
    """One from-scratch sim, the production way."""
    setup = scenario_setup(seed, mini=mini, scenario_id=scenario_id)
    gs = build_scenario_gamestate(setup)
    return WesnothSim(gs, scenario_id=setup.scenario_id,
                      max_turns=max_turns)


def twin_scenario_sims(seed: int = 0, *, max_turns: int = 6,
                       mini: bool = False,
                       scenario_id: Optional[str] = None,
                       ) -> Tuple[WesnothSim, WesnothSim]:
    """Two sims guaranteed to share an IDENTICAL starting state (one
    gamestate build, deep-copied into each). Determinism tests
    compare runs from these twins so they measure the SIM's
    determinism, not the builder's."""
    setup = scenario_setup(seed, mini=mini, scenario_id=scenario_id)
    gs = build_scenario_gamestate(setup)
    return (
        WesnothSim(copy.deepcopy(gs), scenario_id=setup.scenario_id,
                   max_turns=max_turns),
        WesnothSim(copy.deepcopy(gs), scenario_id=setup.scenario_id,
                   max_turns=max_turns),
    )
