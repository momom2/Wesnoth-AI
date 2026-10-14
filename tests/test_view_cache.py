"""The simulator's live view refreshed from its cache (`wesnoth_core.
ViewCache`, `CoreState.view_state`) holds what a whole rebuild of the core
holds after every command, and keeps the objects of the units a command
left alone."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from wesnoth_ai import game_core as gc  # noqa: E402

pytestmark = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")


def _play(seed: int, decisions: int, on_step) -> int:
    """A random legal game; `on_step(sim, before)` after each step, with
    the unit objects of the view before it by id."""
    from tools.inference_seam import RemoteEncoder
    from tools.random_positions import random_legal_action
    from tools.selfplay_game import _would_recruit_bounce
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate, random_setup
    rng = random.Random(seed)
    setup = random_setup(random.Random(seed))
    sim = WesnothSim(build_scenario_gamestate(setup), scenario_id=setup.scenario_id, max_turns=30)
    enc = RemoteEncoder({}, {}, relevant_set=True, fog_hides_enemy_villages=True, server_priors=True,
                        terrain_multi_hot=True, observation_parity=True, relevant_set_version=2)
    steps = 0
    for _ in range(decisions):
        if sim.done:
            break
        gs = sim.gs
        side = gs.global_info.current_side
        e = enc.encode(gs)
        leaders = {u.side: (u.position.x, u.position.y) for u in gs.map.units if u.is_leader}
        action = random_legal_action(e, rng, aim=leaders.get(3 - side), keep=leaders.get(side))
        before = {u.id: u for u in gs.map.units}
        if _would_recruit_bounce(action, sim.gs):
            tgt = action["target_hex"]
            sim.reject_recruit_hex(tgt.x, tgt.y)
        else:
            sim.step(action)
        on_step(sim, before, action)
        steps += 1
    return steps


@pytest.mark.parametrize("seed", [3, 8])
def test_the_live_view_is_the_cores_state(seed):
    """After every command of a random legal game (moves, attacks with
    their strike records cleared, recruits, rejections, turn ends): the
    live view's units, sides, globals and map are a whole rebuild's, and
    a unit the step did not touch keeps its object."""
    from wesnoth_ai.game_core import view_content
    kept = []

    def check(sim, before, action):
        assert view_content(sim.gs) == view_content(sim.core.to_state())
        if action["type"] == "move" and sim.gs.global_info.current_side in (1, 2):
            moved = {(action["start_hex"].x, action["start_hex"].y)}
            same = [u for u in sim.gs.map.units if u.id in before and (u.position.x, u.position.y) not in moved
                    and vars(u) == vars(before[u.id])]
            kept.append(sum(u is before[u.id] for u in same) / max(1, len(same)))

    assert _play(seed, 160, check) > 100
    assert kept and min(kept) == 1.0


def test_a_full_rebuild_and_the_cache_give_the_same_views():
    """The same game with every refresh a whole rebuild: the same view
    after every step."""
    import tools.wesnoth_sim as ws
    from wesnoth_ai.game_core import view_content
    views = {}
    for incremental in (True, False):
        out = views[incremental] = []
        old = ws.WesnothSim.incremental_view
        ws.WesnothSim.incremental_view = incremental
        try:
            _play(5, 120, lambda sim, before, action: out.append(
                (view_content(sim.gs)[0], view_content(sim.gs)[1:2], sorted(sim.gs.global_info.__dict__))))
        finally:
            ws.WesnothSim.incremental_view = old
    assert len(views[True]) > 40 and views[True] == views[False]
