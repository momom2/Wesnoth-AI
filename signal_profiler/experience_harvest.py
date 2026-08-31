"""Play N games through the PRODUCTION pipeline and return the
drained experience batch (signal_profiler v1). Mirrors the arm-V3
generation config by default (TCS, mover frame, gate projection);
--no-turn-search yields the plain-Gumbel teacher instead.
"""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

log = logging.getLogger("signal_profiler")


def harvest_experiences(policy, n_games: int, seed: int,
                        max_turns: int = 60) -> Tuple[List, List]:
    """(experiences, outcomes). `policy` is a fully-configured
    MCTSPolicy/TurnCommitPolicy; games run through
    tools.sim_self_play.play_one_game — the production loop
    (fork-guard decisions, bounce retries, finalize_game with GBC
    labels and value weights)."""
    from tools.scenario_pool import build_scenario_gamestate, random_setup
    from tools.sim_self_play import _recruit_cost_lookup, play_one_game
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.rewards import WeightedReward

    cost_lookup = _recruit_cost_lookup()
    reward_fn = WeightedReward()
    outcomes = []
    for g in range(n_games):
        rng = random.Random(seed + g)
        setup = random_setup(rng)
        gs = build_scenario_gamestate(setup)
        sim = WesnothSim(gs, scenario_id=setup.scenario_id,
                         max_turns=max_turns)
        out = play_one_game(sim, policy, reward_fn,
                            game_label=f"profile_g{g}",
                            cost_lookup=cost_lookup)
        outcomes.append(out)
        log.info("game %d/%d done: winner=%s turns=%s", g + 1,
                 n_games, getattr(out, "winner", "?"),
                 getattr(out, "turns", "?"))
    with policy._lock:
        batch = list(policy._queue)
        policy._queue = []
    log.info("harvested %d experiences from %d games",
             len(batch), n_games)
    return batch, outcomes


def make_policy(checkpoint: Path, device, *, turn_search: bool = True,
                value_memory_iters: int = 20,
                games_per_iter: int = 24):
    """Arm-V3-config policy factory ingredients: TCS + mover frame +
    gate projection + head-only value memory. Returns a zero-arg
    factory (fresh policy per call — gradient_tree needs isolation
    across term variants)."""
    from tools.eval_sim import _load_policy
    from tools.mcts import MCTSConfig
    from tools.mcts_policy import MCTSPolicy, ReplayConfig
    from tools.turn_policy import TurnCommitPolicy
    from tools.turn_search_config import TurnSearchConfig

    def factory():
        base = _load_policy(checkpoint, device, label="profile")
        mc = MCTSConfig(n_simulations=32)
        kw = dict(
            replay_config=ReplayConfig(enabled=False),
            gbc_labels=True,
            value_memory_games=value_memory_iters * games_per_iter,
        )
        if turn_search:
            cfg = TurnSearchConfig(boundary_frame="mover",
                                   project="reval",
                                   project_halfturns=1)
            return TurnCommitPolicy(base, mc, turn_config=cfg, **kw)
        return MCTSPolicy(base, mc, **kw)

    return factory
