#!/usr/bin/env python3
"""Per-call cost of the Rust-owned state against the Python state
(docs/plan_20260904.md step 1.2's acceptance: step and fork under
0.1 ms, encode under 0.2 ms).

Measures, on real midgame states, the three operations a search makes
per node:

  fork    `CoreState.fork()`            vs `copy.deepcopy(gs)`
  step    one command through the core   vs `replay_dataset._apply_command`
  encode  `CoreState.encode_raw()`       vs `encoder.encode_raw(gs)`

Each is timed over `--repeat` calls on each of `--states` states; the
reported number is the median over states of the per-call mean. The
states come from replayed human games (a real unit count and map
size), so the numbers are the ones a search would pay.

    python tools/bench_core.py --corpus replays_dataset_imitation --states 8

Prints one JSON record; `--out` writes it.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("bench_core")


def _states_from_corpus(corpus: Path, n_states: int, skip: int) -> List[tuple]:
    """(state, next command) pairs from replayed games, one per game,
    taken `skip` commands in so the board is populated."""
    from tools.replay_dataset import (_apply_command, _build_initial_gamestate,
                                      _setup_scenario_events, filter_competitive_2p)
    out = []
    for gz in filter_competitive_2p(corpus)[:n_states * 3]:
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            data = json.load(f)
        gs = _build_initial_gamestate(data)
        _setup_scenario_events(gs, data.get("scenario_id", ""))
        cmds = data.get("commands", [])
        if len(cmds) < skip + 4:
            continue
        for cmd in cmds[:skip]:
            _apply_command(gs, list(cmd))
        nxt = next((c for c in cmds[skip:] if c and c[0] == "move"), None)
        if nxt is None:
            continue
        out.append((gs, list(nxt)))
        if len(out) >= n_states:
            break
    return out


def _time(fn, repeat: int) -> float:
    """Mean seconds per call over `repeat` calls (one warm call first)."""
    fn()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t0) / repeat


def bench(states, repeat: int, vocab) -> dict:
    from wesnoth_ai.encoder import encode_raw
    from wesnoth_ai.game_core import CoreState
    from tools.replay_dataset import _apply_command
    type_to_id, faction_to_id = vocab
    rows = {k: [] for k in ("fork_py", "fork_core", "step_py", "step_core",
                            "encode_py", "encode_core", "units", "hexes")}
    for gs, cmd in states:
        cs = CoreState.from_state(gs)
        rows["units"].append(len(gs.map.units))
        rows["hexes"].append(len(gs.map.hexes))
        rows["fork_py"].append(_time(lambda: copy.deepcopy(gs), repeat))
        rows["fork_core"].append(_time(lambda: cs.fork(), repeat))
        # The step is timed on a fresh copy per call so every call
        # applies the same command to the same state.
        rows["step_py"].append(_time(lambda: _apply_command(copy.deepcopy(gs), list(cmd)), repeat)
                               - rows["fork_py"][-1])
        rows["step_core"].append(_time(lambda: cs.fork().apply_command(list(cmd)), repeat)
                                 - rows["fork_core"][-1])
        kw = dict(type_to_id=type_to_id, faction_to_id=faction_to_id,
                  relevant_set=True, fog_hides_enemy_villages=True)
        rows["encode_py"].append(_time(lambda: encode_raw(gs, **kw), repeat))
        rows["encode_core"].append(_time(lambda: cs.encode_raw(**kw), repeat))
    out = {"states": len(states), "repeat": repeat}
    for k, v in rows.items():
        if not v:
            continue
        out[k + ("" if k in ("units", "hexes") else "_ms")] = (
            round(statistics.median(v), 1) if k in ("units", "hexes")
            else round(statistics.median(v) * 1e3, 4))
    return out


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("replays_dataset_imitation"))
    ap.add_argument("--states", type=int, default=8)
    ap.add_argument("--skip", type=int, default=120, help="commands applied before the state is taken")
    ap.add_argument("--repeat", type=int, default=40)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING))
    from wesnoth_ai.game_core import game_core_class
    if game_core_class() is None:
        print(json.dumps({"error": "wesnoth_core.GameCore unavailable (phase 8 wheel)"}))
        return 1
    states = _states_from_corpus(args.corpus, args.states, args.skip)
    if not states:
        print(json.dumps({"error": f"no states from {args.corpus}"}))
        return 1
    names = sorted({u.name for gs, _ in states for u in gs.map.units})
    factions = sorted({s.faction for gs, _ in states for s in gs.sides if s.faction})
    vocab = ({n: i for i, n in enumerate(names)}, {f: i for i, f in enumerate(factions)})
    res = bench(states, args.repeat, vocab)
    print(json.dumps(res, indent=1))
    if args.out:
        args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
