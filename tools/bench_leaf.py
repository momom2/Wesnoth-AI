#!/usr/bin/env python3
"""Where the MCTS actor's per-leaf Python goes (docs/box_specs.md
"The actor's per-leaf Python").

The pool's throughput is set by the actor, not by the server: an
iteration achieves 422-691 leaf evaluations per second against a
saturated server rate of about 1,050, so every actor millisecond per
leaf is the product. This times the components of ONE leaf expansion
on a real midgame state, in the basis the reference player plays:

  encode_raw            the numpy/Rust arrays for the leaf's state
  encode_from_raw       those arrays as torch tensors
  enumerate_legal_...   the legality masks and one prior per legal action
  sort + MCTSEdge       materializing the edges the search will index

Run it where the phase-8 wheel is installed; on a machine whose wheel
predates the kernels the numbers are the old Python path's, not the
production path's (the laptop cannot build the wheel).

    python tools/bench_leaf.py --repeat 20 --out leaf.json
"""
from __future__ import annotations

import argparse
import cProfile
import gzip
import io
import json
import logging
import pstats
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("bench_leaf")


def midgame_state(corpus: Path, index: int, skip: int):
    from tools.replay_dataset import (_apply_command, _build_initial_gamestate,
                                      _setup_scenario_events, filter_competitive_2p)
    files = filter_competitive_2p(corpus)
    gz = files[index % len(files)]
    with gzip.open(gz, "rt", encoding="utf-8") as f:
        data = json.load(f)
    gs = _build_initial_gamestate(data)
    _setup_scenario_events(gs, data.get("scenario_id", ""))
    for cmd in data.get("commands", [])[:skip]:
        _apply_command(gs, list(cmd))
    return gs


def _timed(fn, n: int):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn()
    return (time.perf_counter() - t0) / n * 1e3, out


def bench(gs, repeat: int, device: str) -> dict:
    import torch
    from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors
    from wesnoth_ai.encoder import GameStateEncoder, encode_raw
    from wesnoth_ai.model import WesnothModel
    from tools.mcts import MCTSEdge, _leaf_to_cpu, _packed_masks_of

    enc = GameStateEncoder(d_model=384, relevant_set_hexes=True).to(device)
    enc.register_names(gs)
    model = WesnothModel(d_model=384, num_layers=8, num_heads=12, d_ff=1536).eval().to(device)
    out: dict = {"units": len(gs.map.units), "hexes": len(gs.map.hexes),
                 "turn": gs.global_info.turn_number, "device": device}
    kw = dict(type_to_id=enc.unit_type_to_id, faction_to_id=enc.faction_to_id,
              relevant_set=True, fog_hides_enemy_villages=enc.fog_hides_enemy_villages)
    with torch.no_grad():
        out["encode_raw_ms"], raw = _timed(lambda: encode_raw(gs, **kw), repeat)
        out["encode_from_raw_ms"], _ = _timed(lambda: enc.encode_from_raw(raw), repeat)
        out["encode_total_ms"], encoded = _timed(lambda: enc.encode(gs), repeat)
        encoded = enc.encode(gs)
        out["forward_ms"], output = _timed(lambda: model(encoded), max(3, repeat // 4))
        out["packed_masks_ms"], _ = _timed(lambda: _packed_masks_of(encoded), repeat)
        out["leaf_to_cpu_ms"], _ = _timed(lambda: _leaf_to_cpu(encoded, output), repeat)
        enc_cpu, out_cpu = _leaf_to_cpu(encoded, output)
        out["enumerate_ms"], priors = _timed(
            lambda: enumerate_legal_actions_with_priors(enc_cpu, out_cpu, gs, decision_step=0),
            max(3, repeat // 2))
        out["n_legal_actions"] = len(priors)
        out["sort_ms"], _ = _timed(lambda: sorted(priors, key=lambda p: -p.prior), repeat)
        out["edges_ms"], _ = _timed(lambda: [MCTSEdge(p) for p in priors], repeat)
    out["actor_python_per_leaf_ms"] = round(
        out["encode_total_ms"] + out["packed_masks_ms"] + out["leaf_to_cpu_ms"]
        + out["enumerate_ms"] + out["sort_ms"] + out["edges_ms"], 4)
    # The core's own encode, when the wheel carries it.
    try:
        from wesnoth_ai.game_core import CoreState, game_core_class
        if game_core_class() is not None:
            cs = CoreState.from_state(gs)
            out["core_encode_raw_ms"], _ = _timed(lambda: cs.encode_raw(**kw), repeat)
    except Exception as e:  # noqa: BLE001 - an optional column
        out["core_encode_raw_error"] = repr(e)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def profile_enumeration(gs, n: int = 5) -> str:
    import torch
    from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.model import WesnothModel
    from tools.mcts import _leaf_to_cpu
    enc = GameStateEncoder(d_model=384, relevant_set_hexes=True)
    enc.register_names(gs)
    model = WesnothModel(d_model=384, num_layers=8, num_heads=12, d_ff=1536).eval()
    with torch.no_grad():
        encoded = enc.encode(gs)
        output = model(encoded)
        enc_cpu, out_cpu = _leaf_to_cpu(encoded, output)
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(n):
            enumerate_legal_actions_with_priors(enc_cpu, out_cpu, gs, decision_step=0)
        pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(25)
    return s.getvalue()


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("replays_dataset_imitation"))
    ap.add_argument("--states", type=int, default=3)
    ap.add_argument("--skip", type=int, default=140)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--profile", action="store_true", help="also print a cProfile of the enumeration")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    try:
        import wesnoth_core  # noqa: F401 -- the kernels the leaf path runs on
    except ImportError:
        print("wesnoth_core is not installed: build the wheel first (rust/wesnoth_core)")
        return 1
    rows = []
    for i in range(args.states):
        gs = midgame_state(args.corpus, i * 3, args.skip)
        rows.append(bench(gs, args.repeat, args.device))
        print(json.dumps(rows[-1], indent=1))
    res = {"wheel_phase": getattr(wesnoth_core, "__phase__", None), "states": rows}
    if args.profile:
        res["profile"] = profile_enumeration(midgame_state(args.corpus, 0, args.skip))
        print(res["profile"])
    if args.out:
        args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
