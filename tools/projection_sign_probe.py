"""Q7 projection sign test (approved 2026-08-17; run 2026-08-21).

Question: does boundary-only scoring systematically OVERRATE passive
turns? A correct value head already prices "the enemy moves next", in
which case projection adds nothing (user 2026-08-21). Projection is a
bet on a specific head defect: immediate threats are under-priced at
the boundary and become visible after the enemy's actual reply.

Per sampled state, candidate turns spanning the passivity axis:
  full     -- the policy's own complete turn (spine)
  half     -- first half of the spine, then end_turn
  pass     -- immediate end_turn
Each candidate is scored twice at the same salt:
  V_b -- at the boundary (production estimand)
  V_p -- project_value: after `--halfturns` closed-loop enemy
         half-turns (the projection estimand)
Delta = V_b - V_p. The passivity-overrated signature is
Delta(pass) > Delta(full), paired per state.

Shard on a box (states are independent):
    python tools/projection_sign_probe.py --checkpoint CKPT \
        --states 17 --seed N --out shard_N.jsonl
Collate: --collate 'shards/*.jsonl' (paired t on pass-vs-full delta).
"""
from __future__ import annotations

import argparse
import copy
import glob as _glob
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

from tools.turn_counterfactual_probe import load_policy  # noqa: E402
from tools.turn_search import (  # noqa: E402
    materialize, project_value, record_spine,
)
from tools.wesnoth_sim import WesnothSim  # noqa: E402

log = logging.getLogger("proj_sign")


def probe_state(policy, sim0, side, ds, rng, sid, halfturns,
                max_actions):
    spine, _ = record_spine(policy, sim0, side, ds, rng, max_spine=40)
    if len(spine) < 4:          # too short to distinguish half/full
        return None
    commands = [s.action for s in spine]
    cands = {
        "full": commands,
        "half": commands[:len(commands) // 2],
        "pass": [],
    }
    rec = {"state": sid, "K": len(spine),
           "turn": sim0.gs.global_info.turn_number}
    from tools.turn_search import boundary_value
    for name, cmds in cands.items():
        # Both boundary frames: opponent (leg-4 production estimand,
        # fog-blind to the mover's turn) and mover (the leg-5
        # estimand). Projection always starts from the post-flip sim.
        m = materialize(policy, sim0, side, cmds, f"q7:{sid}", ds,
                        skip_value=True)
        m2 = materialize(policy, sim0, side, cmds, f"q7:{sid}", ds,
                         skip_value=True, mover_frame=True)
        if m.invalid or m2.invalid or m.boundary_sim is None \
                or m2.boundary_sim is None:
            return None
        vb_opp = boundary_value(policy, m.boundary_sim, side, ds)
        vb_mov = boundary_value(policy, m2.boundary_sim, side, ds)
        v_p = project_value(policy, m.boundary_sim, side, ds,
                            halfturns, max_actions, rng)
        if any(math.isnan(v) for v in (vb_opp, vb_mov, v_p)):
            return None
        rec[f"vb_opp_{name}"] = vb_opp
        rec[f"vb_mover_{name}"] = vb_mov
        rec[f"vp_{name}"] = v_p
        rec[f"delta_opp_{name}"] = vb_opp - v_p
        rec[f"delta_mover_{name}"] = vb_mov - v_p
    return rec


def collate(patterns):
    rows = []
    for pat in patterns:
        for p in sorted(_glob.glob(pat)):
            with open(p, encoding="utf-8") as f:
                rows += [json.loads(ln) for ln in f if ln.strip()]
    rows = [r for r in rows if "delta_opp_pass" in r]
    n = len(rows)
    if n < 3:
        print(f"only {n} rows; nothing to collate")
        return 1
    for frame in ("opp", "mover"):
        print(f"--- frame: {frame}")
        for name in ("pass", "half", "full"):
            d = np.array([r[f"delta_{frame}_{name}"] for r in rows])
            print(f"  delta_{name}: mean {d.mean():+.4f} +- "
                  f"{d.std(ddof=1)/math.sqrt(n):.4f} (n={n})")
        # THE test: paired excess of pass-delta over full-delta.
        x = np.array([r[f"delta_{frame}_pass"]
                      - r[f"delta_{frame}_full"] for r in rows])
        se = x.std(ddof=1) / math.sqrt(n)
        t = x.mean() / se if se > 0 else float("nan")
        print(f"  SIGNATURE (delta_pass - delta_full, paired): "
              f"mean {x.mean():+.4f} +- {se:.4f}  t={t:.2f}")
    print("(>0 with |t|>2 => boundary scoring overrates passing in "
          "that frame; projection earns its cost. C51 atom = 0.04.)")
    return 0


def main(argv) -> int:
    from tools.scenario_pool import build_scenario_gamestate, random_setup

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--states", type=int, default=17)
    ap.add_argument("--halfturns", type=int, default=1)
    ap.add_argument("--max-actions", type=int, default=40)
    ap.add_argument("--category", default="ladder")
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--turn-sample-prob", type=float, default=0.25)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--collate", nargs="+", default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=args.log_level)
    if args.collate:
        return collate(args.collate)
    if not args.checkpoint:
        ap.error("--checkpoint required unless --collate")

    rng_py = random.Random(args.seed)
    rng = np.random.default_rng(args.seed)
    policy = load_policy(args.checkpoint, device=args.device)
    ds = policy._decision_step

    f_out = args.out.open("w", encoding="utf-8") if args.out else None
    done = 0
    t0 = time.time()
    for g in range(args.games):
        if done >= args.states:
            break
        setup = random_setup(rng_py, category=args.category)
        gs = build_scenario_gamestate(setup)
        sim = WesnothSim(gs, scenario_id=setup.scenario_id,
                         max_turns=args.max_turns)
        label = f"q7_g{g}"
        prev = None
        while not sim.done:
            gi = sim.gs.global_info
            ts = (gi.turn_number, gi.current_side)
            if ts != prev:
                prev = ts
                if (done < args.states
                        and rng.random() < args.turn_sample_prob):
                    sid = f"s{args.seed}g{g}t{gi.turn_number}" \
                          f"s{gi.current_side}"
                    rec = probe_state(policy, sim.fork(),
                                      gi.current_side, ds, rng, sid,
                                      args.halfturns, args.max_actions)
                    if rec is not None:
                        done += 1
                        line = json.dumps(rec)
                        if f_out:
                            f_out.write(line + "\n")
                            f_out.flush()   # incremental: partial
                            #                 shards survive kills
                        log.info(f"{sid} ({done}/{args.states}, "
                                 f"{time.time()-t0:.0f}s)")
            if done >= args.states:
                break
            pre = copy.deepcopy(sim.gs)
            action = policy.select_action(pre, game_label=label,
                                          sim=sim)
            sim.step(action)
        policy.drop_pending(label)
    if f_out:
        f_out.close()
    log.info(f"done: {done} states in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
