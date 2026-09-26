"""The game states the benchmarks and the turn-level gap measure on.

`load_states` rebuilds the pinned positions of configs/bench_states.json:
holdout ladder games of the imitation corpus at a side-turn boundary
(`reconstruct_boundary`), bit-exact, so every box measures the same
positions. tools/bench_pipeline.py picks them (--build-states) and
copies the game files they need to a box (--pack-states).
`harvest_states` collects states from dummy-vs-dummy simulator games
instead: the simulator's own stream of board shapes, no network needed.
"""
from __future__ import annotations

import copy
import gzip
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from wesnoth_ai.paths import CONFIGS_DIR, IMITATION_DATASET_DIR

log = logging.getLogger("bench_states")

DEFAULT_MANIFEST = CONFIGS_DIR / "bench_states.json"
DEFAULT_DATASET = IMITATION_DATASET_DIR


def reconstruct_boundary(data: dict, cut_turn: int):
    """Walk a replay's commands to the first init_side of a player
    side with turn_number >= cut_turn and return (state, begin_side)
    with the side's turn begun (income, healing applied), i.e. the
    position the side to move faces. None when the game ends first
    or a leader is dead."""
    from tools.replay_dataset import (_apply_command, _build_initial_gamestate,
                                      _setup_scenario_events)
    from tools.wesnoth_sim import WesnothSim
    gs = _build_initial_gamestate(data)
    scenario_id = data.get("scenario_id", "")
    _setup_scenario_events(gs, scenario_id)
    for cmd in data.get("commands", []):
        if (cmd and cmd[0] == "init_side"
                and gs.global_info.turn_number >= cut_turn
                and gs.global_info.current_side in (1, 2)
                and len(cmd) > 1 and cmd[1] in (1, 2)):
            if not {1, 2} <= {u.side for u in gs.map.units if u.is_leader}:
                return None
            for attr in ("_last_advance_events", "_last_checkup_strikes"):
                if hasattr(gs.global_info, attr):
                    setattr(gs.global_info, attr, [] if attr.endswith("events") else None)
            begin_side = int(cmd[1])
            sim = WesnothSim(gs, scenario_id, max_turns=200,
                             apply_scenario_events=False, begin_side=begin_side)
            return sim.gs, begin_side
        _apply_command(gs, cmd)
    return None


def load_states(manifest_path: Path, dataset_dir: Path,
                limit: Optional[int] = None) -> List[Tuple[object, str]]:
    """(GameState, scenario_id) for every manifest entry, rebuilt from
    the dataset (bit-exact reconstruction, so every box sees the same
    positions)."""
    man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    out = []
    for e in man["states"][:limit]:
        with gzip.open(dataset_dir / e["file"], "rt", encoding="utf-8") as f:
            data = json.load(f)
        res = reconstruct_boundary(data, e["cut_turn"])
        if res is None or res[1] != e["begin_side"]:
            raise RuntimeError(f"benchmark state failed to reconstruct: {e}")
        out.append((res[0], e["scenario_id"]))
    return out


def harvest_states(n: int, seed: int):
    """Deep-copied GameStates from dummy-vs-dummy sim games -- the
    sim's real shape stream, no network involved."""
    from tools.elo_ladder import _ScriptedAdapter
    from tools.eval_players import _PolicyPair, _play_one_eval_game
    from tools.scenario_pool import build_scenario_gamestate, random_setup
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.dummy_policy import DummyPolicy
    import random as _random

    sink: list = []
    cap_per_game = max(4, n // 8)   # force >=8 games' worth of maps
    stride = 3                      # skip adjacent near-identical shapes

    class _Recorder:
        def __init__(self, inner):
            self._inner = inner
            self._seen = 0
            self._taken = 0

        def select_action(self, gs, **kw):
            self._seen += 1
            if (len(sink) < n and self._taken < cap_per_game
                    and self._seen % stride == 0):
                sink.append(copy.deepcopy(gs))
                self._taken += 1
            return self._inner.select_action(gs, **kw)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    g = 0
    while len(sink) < n and g < 50:
        rng = _random.Random(seed + g)
        g += 1
        setup = random_setup(rng)
        gs = build_scenario_gamestate(setup)
        sim = WesnothSim(gs, scenario_id=setup.scenario_id,
                         max_turns=30)
        _play_one_eval_game(
            sim,
            _PolicyPair(policy=_Recorder(_ScriptedAdapter(DummyPolicy())),
                        label="a", side=1),
            _PolicyPair(policy=_Recorder(_ScriptedAdapter(DummyPolicy())),
                        label="b", side=2),
            game_label=f"bench{g}")
    log.info("harvested %d states from %d dummy games", len(sink), g)
    return sink[:n]
