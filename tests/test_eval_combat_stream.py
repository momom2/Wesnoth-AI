#!/usr/bin/env python3
"""Per-game combat luck in eval games (2026-09-13).

Every eval game used to run on ONE luck vector: an unsalted
`WesnothSim` draws its k-th synced-RNG seed from
`request_seed(k) = sha256("sim_to_replay:k")[:8]`, a pure function of
the request counter, so the k-th combat roll of every game was the
same. The scenario and factions varied per game, the dice did not --
an N-game match was N draws against one vector while the standard
error assumed N independent ones.

These tests drive the production salt (`elo_eval_game.combat_salt`) on
real `WesnothSim` instances built the way `elo_eval_game.main` builds
them, and keep the old regime as the control: with the shared stream,
two different games must still produce IDENTICAL dice, so a test that
passed vacuously would fail there.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.elo_eval_game import combat_salt          # noqa: E402
from tools.scenario_pool import (                    # noqa: E402
    build_scenario_gamestate, random_setup,
)
from tools.wesnoth_sim import WesnothSim             # noqa: E402

N_ROLLS = 12


def _luck_stream(seed: int, shared_stream: bool = False,
                 n: int = N_ROLLS):
    """The first `n` synced-RNG seeds an eval game on `seed` would
    hand to combat, from a sim built exactly as elo_eval_game does."""
    setup = random_setup(random.Random(seed))
    sim = WesnothSim(build_scenario_gamestate(setup),
                     scenario_id=setup.scenario_id, max_turns=200)
    sim._seed_salt = combat_salt(seed, shared_stream)
    return [sim._next_seed() for _ in range(n)]


def test_salt_is_per_game_and_clearable():
    assert combat_salt(10_000) == combat_salt(10_000)
    assert combat_salt(10_000) != combat_salt(10_001)
    assert combat_salt(10_000, shared_stream=True) == "", \
        "the compat flag must restore the unsalted stream"


def test_different_games_roll_different_dice():
    a = _luck_stream(10_000)
    b = _luck_stream(10_001)
    assert len(set(a)) == N_ROLLS, "a stream must not repeat itself"
    assert not set(a) & set(b), (
        "two eval games must draw independent combat luck; any shared "
        "seed means the dice correlate across the match")


def test_same_seed_reproduces_the_same_dice():
    assert _luck_stream(10_000) == _luck_stream(10_000), \
        "a game must stay reproducible from its slot"


def test_shared_stream_control_still_shares_everything():
    """The defect, kept as the control: under --shared-combat-stream
    two DIFFERENT games roll the identical vector. If this ever fails,
    the test above proves nothing."""
    assert (_luck_stream(10_000, shared_stream=True)
            == _luck_stream(10_001, shared_stream=True))


def test_result_file_carries_the_regime_to_the_catalog(tmp_path):
    """End to end on real games: elo_eval_game records the regime and
    the slot, elo_collect lifts them onto the catalog edge, and a
    shared-stream dir is then refused against the per-game edge."""
    import json

    from tools import elo_collect
    from tools.elo_catalog import decode_game_ids, load_catalog
    from tools.elo_eval_game import main as game_main

    out = tmp_path / "per_game"
    for seed in (10_000, 10_001):
        assert game_main(["x", "A", "dummy", "B", "dummy",
                          "1" if seed % 2 == 0 else "2", str(seed),
                          str(out), "--mcts-sims", "0",
                          "--max-turns", "2", "--device", "cpu"]) == 0
    played = [json.loads(f.read_text(encoding="utf-8"))
              for f in sorted(out.glob("game_*.json"))]
    assert {g["combat_stream"] for g in played} == {"per_game"}

    cat_path = tmp_path / "cat.json"
    assert elo_collect.main(["x", str(out), "--catalog-path",
                             str(cat_path)]) == 0
    edge, = load_catalog(cat_path)["edges"].values()
    assert edge["protocol"]["estimands"]["combat_stream"] == "per_game"
    assert decode_game_ids(edge["games"]) == {(1, 10_000), (2, 10_001)}

    # The compat flag round-trips the other way, so the two regimes
    # are distinguishable on the edge (test_elo_catalog covers the
    # refusal that distinction buys).
    old = tmp_path / "shared"
    for seed in (20_000, 20_001):
        assert game_main(["x", "A", "dummy", "B", "dummy",
                          "1" if seed % 2 == 0 else "2", str(seed),
                          str(old), "--mcts-sims", "0",
                          "--max-turns", "2", "--device", "cpu",
                          "--shared-combat-stream"]) == 0
    old_cat = tmp_path / "cat_old.json"
    assert elo_collect.main(["x", str(old), "--catalog-path",
                             str(old_cat)]) == 0
    old_edge, = load_catalog(old_cat)["edges"].values()
    assert old_edge["protocol"]["estimands"]["combat_stream"] == "shared"


def test_a_salted_eval_sim_is_not_a_search_fork():
    """Salting eval games per game must not silence the live-sim
    contract warnings.

    Three checks in `WesnothSim` (a move target the mask offered that
    the sim cannot land on, a recruit outside the leader's castle
    network, a recruit off the side's list) are mask-contract
    VIOLATIONS on a live sim and worth a WARNING, but ordinary chance
    divergence inside a search fork. They used to decide which by
    asking whether `_seed_salt` was set — so the day eval games got a
    per-game salt, the verdict path silently dropped all three to
    DEBUG, and one of them even relabelled itself "search-fork chance
    divergence" on a live game.

    `_seed_salt` is an RNG input; `_is_search_fork` is the question.
    """
    import inspect

    from tools import elo_eval_game, wesnoth_sim

    src = inspect.getsource(wesnoth_sim)
    assert "log.debug if self._seed_salt" not in src, \
        "the log sites must branch on _is_search_fork, not on the salt"
    assert src.count("log.debug if self._is_search_fork") == 3, \
        "all three live-sim contract checks branch on the fork flag"

    # The eval path sets the salt and must NOT set the flag.
    eval_src = inspect.getsource(elo_eval_game)
    assert "_seed_salt" in eval_src, "eval games are salted"
    assert "_is_search_fork" not in eval_src, \
        "an eval game is a live game of record, not a search fork"

    # Every producer that DOES fork for search sets the flag.
    for mod in ("tools.mcts", "tools.turn_search", "tools.turn_gap",
                "tools.plan_tournament"):
        m = __import__(mod, fromlist=["x"])
        assert "_is_search_fork = True" in inspect.getsource(m), mod


def test_the_fork_flag_defaults_off_and_survives_a_fork():
    from sim_test_helpers import fresh_scenario_sim

    sim = fresh_scenario_sim(0, max_turns=4, use_core=False)
    assert sim._is_search_fork is False, "a live sim is not a fork"
    sim._seed_salt = "elo:7"
    assert sim._is_search_fork is False, "a salt alone must not make it one"

    sim._is_search_fork = True
    assert sim.fork()._is_search_fork is True, \
        "a fork of a fork is still a fork"
