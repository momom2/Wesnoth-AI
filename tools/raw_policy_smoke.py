#!/usr/bin/env python3
"""Raw-policy self-play smoke: does the policy's own play reach
conclusions?

The acceptance test for the supervised (behavior-cloning) pass
(2026-07-16): play N fresh games per pool with the RAW policy (no
MCTS, no crutches) and report decisive rate + basic engagement.
Baseline for scale: pre-SL fresh-ladder self-play was ~0-decisive.

--idle-gate on|off toggles constants.FORBID_IDLE_END_TURN at
runtime: the gate masks end_turns that human play (and thus the
cloned policy) uses deliberately, so its fate is decided on this
harness's evidence.

Usage:
    python tools/raw_policy_smoke.py CKPT [--games 12] [--pools
        ladder,mini] [--idle-gate off] [--max-turns 60] [--seed 5]
"""

from __future__ import annotations

import argparse
import copy
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("raw_policy_smoke")


def main(argv) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--games", type=int, default=12,
                    help="Games per pool.")
    ap.add_argument("--pools", default="ladder,mini",
                    help="Comma list: ladder | ladder_fogless | mini")
    ap.add_argument("--idle-gate", choices=("on", "off"), default="on")
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv[1:])

    import constants
    constants.FORBID_IDLE_END_TURN = (args.idle_gate == "on")
    log.info(f"idle end_turn gate: {args.idle_gate}")

    import torch
    from tools.eval_sim import _load_policy
    from tools.scenario_pool import random_setup, build_scenario_gamestate
    from tools.wesnoth_sim import WesnothSim

    policy = _load_policy(args.checkpoint, torch.device(args.device),
                          label="smoke")
    rng = random.Random(args.seed)

    for pool in args.pools.split(","):
        pool = pool.strip()
        mini = pool == "mini"
        fogless = pool == "ladder_fogless"
        n_dec = 0
        turns_list, attacks, recruits = [], 0, 0
        t0 = time.time()
        for g in range(args.games):
            setup = random_setup(
                rng, forced_faction=None, mini_maps=mini,
                category="fogless" if fogless else "ladder")
            gs = build_scenario_gamestate(setup)
            sim = WesnothSim(gs, scenario_id=setup.scenario_id,
                             max_turns=args.max_turns)
            label = f"{pool}{g}"
            while not sim.done:
                pre = copy.deepcopy(sim.gs)
                action = policy.select_action(pre, game_label=label)
                sim.step(action)
            policy.drop_pending(label)
            decisive = sim.winner in (1, 2)
            n_dec += int(decisive)
            turns_list.append(sim.gs.global_info.turn_number)
            attacks += sum(1 for rc in sim.command_history
                           if rc.kind == "attack" and rc.side in (1, 2))
            recruits += sum(1 for rc in sim.command_history
                            if rc.kind == "recruit")
            log.info(f"  {pool} g{g}: winner={sim.winner} "
                     f"ended_by={sim.ended_by} "
                     f"turns={sim.gs.global_info.turn_number} "
                     f"({setup.scenario_id})")
        n = args.games
        log.info(
            f"== {pool}: decisive {n_dec}/{n}, "
            f"attacks/game {attacks / n:.1f}, "
            f"recruits/game {recruits / n:.1f}, "
            f"mean turns {sum(turns_list) / n:.1f}, "
            f"{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
