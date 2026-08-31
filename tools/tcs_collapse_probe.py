#!/usr/bin/env python3
"""TCS accept/reject stream probe (2026-08-31, user order: find the
collapse mechanism).

Runs plan_turn OFFLINE on the SAME harvested states under several
checkpoints (seed vs the collapsed value-memory arms), tracing every
gate decision via turn_search.TRACE. If the collapsed checkpoints
commit short turns here too, the collapse reproduces in vitro and
the stream shows where the actions go: short spines (policy-side),
end_turn alternatives winning the climb (grading prefers early
stops), or accept-starvation (deltas below the gate).

Usage (GPU box):
    python tools/tcs_collapse_probe.py \
        --ckpt seed=training/checkpoints/seed_imit_tierb_start.pt \
        --ckpt armV2=training/checkpoints/armV2_final.pt \
        --n-states 40 --out /workspace/tcs_probe.json
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("tcs_collapse_probe")


def harvest(n_states: int, seed: int):
    """(gs, scenario_id) midstates from dummy games -- real armies,
    fog, contact -- so plan_turn has actual turns to build."""
    from tools.elo_ladder import _ScriptedAdapter
    from tools.eval_sim import _PolicyPair, _play_one_eval_game
    from tools.scenario_pool import build_scenario_gamestate, random_setup
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.dummy_policy import DummyPolicy
    import random as _r

    out = []

    class _Rec:
        def __init__(self, inner, sid):
            self._i = inner
            self._sid = sid
            self._seen = 0

        def select_action(self, gs, **kw):
            self._seen += 1
            if self._seen % 9 == 0 and len(out) < n_states:
                out.append((copy.deepcopy(gs), self._sid))
            return self._i.select_action(gs, **kw)

        def __getattr__(self, name):
            return getattr(self._i, name)

    g = 0
    while len(out) < n_states and g < 30:
        rng = _r.Random(seed + g)
        g += 1
        setup = random_setup(rng)
        gs = build_scenario_gamestate(setup)
        sim = WesnothSim(gs, scenario_id=setup.scenario_id,
                         max_turns=16)
        _play_one_eval_game(
            sim,
            _PolicyPair(policy=_Rec(_ScriptedAdapter(DummyPolicy()),
                                    setup.scenario_id),
                        label="a", side=1),
            _PolicyPair(policy=_Rec(_ScriptedAdapter(DummyPolicy()),
                                    setup.scenario_id),
                        label="b", side=2),
            game_label=f"probe{g}")
    return out[:n_states]


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", action="append", required=True,
                    help="label=path; repeatable")
    ap.add_argument("--n-states", type=int, default=40)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level))

    import numpy as np
    import torch
    from tools import turn_search as ts
    from tools.eval_sim import _load_policy
    from tools.mcts import MCTSConfig
    from tools.turn_search_config import TurnSearchConfig
    from tools.wesnoth_sim import WesnothSim

    device = (torch.device("cuda")
              if args.device == "cuda" and torch.cuda.is_available()
              else None)
    states = harvest(args.n_states, args.seed)
    log.info("harvested %d states", len(states))

    cfg = TurnSearchConfig(boundary_frame="mover")
    mc = MCTSConfig(n_simulations=32)
    results = {}
    for spec in args.ckpt:
        label, path = spec.split("=", 1)
        policy = _load_policy(Path(path), device, label=label)
        events = []
        ts.TRACE = events.append
        try:
            for k, (gs, scen) in enumerate(states):
                sim = WesnothSim(copy.deepcopy(gs), scenario_id=scen,
                                 max_turns=200)
                side = sim.gs.global_info.current_side
                rng = np.random.default_rng(1000 + k)
                events.append({"ev": "state", "k": k})
                try:
                    ts.plan_turn(policy, sim, side, 0, cfg, mc, rng,
                                 salt_ns=f"probe{k}", full=True)
                except Exception as e:  # noqa: BLE001
                    events.append({"ev": "error", "k": k,
                                   "err": repr(e)[:200]})
        finally:
            ts.TRACE = None
        results[label] = events

        # Aggregates, printed as we go.
        spines = [e["n"] for e in events if e["ev"] == "spine"]
        finals = [e for e in events if e["ev"] == "final"]
        gates = [e for e in events if e["ev"] == "gate"]
        acc = [e for e in gates if e["accept"]]
        et_acc = [e for e in acc if e["best_alt_type"] == "end_turn"]
        short_acc = [e for e in acc if e["best_len"] < e["inc_len"]]
        print(f"\n=== {label} ===")
        print(f"states {len(spines)}  spine_len mean "
              f"{np.mean(spines):.1f} med {np.median(spines):.0f}")
        if finals:
            print(f"committed len mean "
                  f"{np.mean([f['committed'] for f in finals]):.1f} med "
                  f"{np.median([f['committed'] for f in finals]):.0f}  "
                  f"accepts/plan {np.mean([f['accepts'] for f in finals]):.2f}")
        if gates:
            print(f"gates {len(gates)}  accept_rate "
                  f"{len(acc)/len(gates):.2f}  "
                  f"end_turn-accepts {len(et_acc)}  "
                  f"shortening-accepts {len(short_acc)}")
            print(f"inc_val mean {np.mean([g['inc_val'] for g in gates]):.3f}  "
                  f"best_delta med "
                  f"{np.median([g['best_delta'] for g in gates]):.4f}  "
                  f"reval_mean med "
                  f"{np.median([g['reval_mean'] for g in gates]):.4f}")

    args.out.write_text(json.dumps(results), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
