#!/usr/bin/env python3
"""Generation throughput through the real actor pool (plan step 1.3):
leaves per second per box, games per hour, at a fixed search budget,
with or without server-side priors.

Runs one ActorPool iteration exactly as the az legs did (plain PUCT,
no Gumbel root, no tree reuse, one ladder game per actor) and reports
the pool's own telemetry: served forwards, decisions, seconds, tokens
per leaf, padding ratio, per-game finish times, plus the games'
outcomes. Run on a GPU box; never the laptop.

Usage:
  python tools/bench_pool.py --checkpoint training/checkpoints/seed.pt \\
      --actors 19 --games 19 --sims 32 --leaf-batch 16 --server-priors \\
      --out pool_on.json
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

log = logging.getLogger("bench_pool")


def run_pool(policy, *, actors: int, games: int, sims: int, leaf_batch: int,
             server_priors: bool, max_turns: int, device, seed: int,
             iteration_timeout: float, log_level: int = logging.WARNING,
             max_batch: int = 16, serve_threads: int = 2) -> dict:
    from tools.actor_pool import ActorPool
    from tools.mcts import MCTSConfig
    from tools.mcts_policy import MCTSPolicy, ReplayConfig
    from tools.sim_self_play import k_median_of
    from tools.wesnoth_sim import PvPDefaults
    if games < actors:
        raise ValueError(f"games ({games}) < actors ({actors}): the surplus actors "
                         f"would play nothing and the record would misstate the "
                         f"actor count")
    mcts_cfg = MCTSConfig(n_simulations=sims, gumbel_root=False, tree_reuse=False,
                          playout_cap_randomization=False, draw_tiebreak=None,
                          batch_size=leaf_batch)
    mpolicy = MCTSPolicy(policy, mcts_cfg, replay_config=ReplayConfig(enabled=False),
                         holdout_size=0, gbc_labels=False)
    pool = ActorPool(mpolicy, actors, mcts_cfg, turn_cfg=None, pt_cfg=None,
                     gbc_labels=False, train_kwargs={},
                     scenario_opts=dict(forced_faction=None, mini_maps=None,
                                        mini_ratio=0.0, fogless_ratio=0.0,
                                        ladder_ratio=1.0, midgame_ratio=0.0,
                                        midgame_dataset=None),
                     max_turns=max_turns, max_turns_min=max_turns,
                     pvp_defaults=PvPDefaults(), device=device, max_batch=max_batch,
                     log_level=log_level, iteration_timeout=iteration_timeout,
                     drain_grace=120.0, server_priors=bool(server_priors),
                     serve_threads=serve_threads)
    pool.start()
    t0 = time.monotonic()
    try:
        outcomes, exps = pool.run_iteration(0, games, seed)
    finally:
        pool.shutdown()
    wall = time.monotonic() - t0
    gen = getattr(pool, "last_iteration_seconds", None) or wall
    served = getattr(pool, "last_served_forwards", 0) or 0
    decided = sum(1 for o in outcomes if o.winner != 0)
    # A run that hit the iteration timeout spends its tail with most
    # actors idle; its wall-clock rates are lower bounds (2026-09-04
    # review: the two truncated baseline rows are under-reported).
    truncated = len(outcomes) < games
    res = {
        "truncated": truncated,
        "server_priors": bool(server_priors), "actors": actors, "games_requested": games,
        "infer_bf16": bool(getattr(policy._inference_model, "infer_autocast_bf16", False)
                           or getattr(policy, "_infer_bf16", False)),
        "sims": sims, "leaf_batch": leaf_batch, "max_turns": max_turns,
        "max_batch": max_batch, "serve_threads": serve_threads,
        "games_completed": len(outcomes), "decisive": decided,
        "abandoned": getattr(pool, "_last_abandoned", None),
        "experiences": len(exps),
        "gen_seconds": gen, "wall_seconds": wall,
        "forwards": served, "decisions": getattr(pool, "last_decisions", None),
        "leaves_per_s": served / gen if gen else None,
        # The iteration average includes the tail where most actors
        # have finished; this is the best 60-s window (the fed rate).
        "saturated_leaves_per_s": getattr(pool, "last_saturated_leaves_per_s", None),
        "leaf_timeline": getattr(pool, "last_leaf_timeline", None),
        "tokens_per_leaf": getattr(pool, "last_tokens_per_leaf", None),
        "pad_ratio": getattr(pool, "last_pad_ratio", None),
        "game_finish_p50_s": getattr(pool, "last_game_finish_p50", None),
        "game_finish_max_s": getattr(pool, "last_game_finish_max", None),
        "mean_turns": (statistics.fmean(o.turns for o in outcomes) if outcomes else None),
        "k_median": k_median_of(outcomes) if outcomes else None,
        "games_per_hour": (len(outcomes) / (gen / 3600.0)) if gen else None,
    }
    return res


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--actors", type=int, default=19)
    ap.add_argument("--games", type=int, default=19)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--leaf-batch", type=int, default=16)
    ap.add_argument("--max-batch", type=int, default=16,
                    help="Leaves the server coalesces per batch across actors "
                         "(ActorPool max_batch; the az legs used 16).")
    ap.add_argument("--serve-threads", type=int, default=2,
                    help="Serving threads in the pool process (2 in the az legs); "
                         "more overlap the per-batch Python with the GPU wait.")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--server-priors", action="store_true")
    ap.add_argument("--iteration-timeout", type=float, default=1500.0)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    ap.add_argument("--infer-bf16", action=argparse.BooleanOptionalAction, default=False,
                    help="bf16 autocast on the inference server's batched path "
                         "(the az legs ran fp32 eager; the eval harness "
                         "defaults to bf16 on cuda).")
    ap.add_argument("--infer-compile", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--dollars-per-hour", type=float, default=0.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    import torch
    from tools.eval_sim import _load_policy
    # The server process runs two serve threads and a GPU; torch's
    # default intra-op pool (64 threads on the 128-thread Vast hosts,
    # against a ~17-core cgroup quota) only burns quota. 2026-09-04:
    # 358 leaves/s capped against 279-320 uncapped (docs/box_specs.md).
    torch.set_num_threads(4)
    device = torch.device("cuda") if args.device == "cuda" and torch.cuda.is_available() \
        else torch.device("cpu")
    if (args.infer_bf16 or args.infer_compile) and device.type != "cuda":
        raise SystemExit("--infer-bf16/--infer-compile need a cuda device")
    policy = _load_policy(args.checkpoint, device, label="pool",
                          infer_bf16=args.infer_bf16, infer_compile=args.infer_compile)
    res = run_pool(policy, actors=args.actors, games=args.games, sims=args.sims,
                   leaf_batch=args.leaf_batch, server_priors=args.server_priors,
                   max_turns=args.max_turns, device=device, seed=args.seed,
                   iteration_timeout=args.iteration_timeout, max_batch=args.max_batch,
                   serve_threads=args.serve_threads)
    if args.dollars_per_hour and res["games_per_hour"]:
        res["games_per_dollar"] = res["games_per_hour"] / args.dollars_per_hour
    print(json.dumps(res, indent=1))
    if args.out:
        args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
