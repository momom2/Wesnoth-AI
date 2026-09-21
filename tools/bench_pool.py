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


def merged_packed_compile(per_server) -> dict:
    """One record over every server's compiled packed trunk (the
    certification row of design note section 13): active only when
    every server's is, recompiles summed, the first fallback reason,
    and the per-server records. A server whose stats never arrived
    (None) leaves the row uncertified."""
    known = [s for s in per_server if s is not None]
    out = dict(known[0]) if known else {}
    out["active"] = (bool(known) and len(known) == len(per_server)
                     and all(bool(s.get("active")) for s in known))
    out["recompiles"] = sum(int(s.get("recompiles") or 0) for s in known)
    out["fallback_reason"] = next(
        (s.get("fallback_reason") for s in known if s.get("fallback_reason")), None)
    out["per_server"] = list(per_server)
    return out


def _run_stream(pool, games: int, seed: int, rounds: int, step_seconds: float,
                timeout: float):
    """`rounds` windows of `games` completed games each, a publication
    (unchanged weights: the sync path under load) and `step_seconds`
    of idling between them, then the drain. Returns the outcomes and
    experiences of every window, the drain's included, and one record
    per window. The pool's `last_*` readbacks describe the whole run
    afterwards (leaves served over the run, saturated rate over it)."""
    stream = pool.stream(seed)
    stream.start()
    records = []
    outcomes: list = []
    n_exps = 0
    t_run = time.monotonic()
    served_before = stream.leaves_served()
    try:
        for r in range(rounds):
            window = stream.collect(games, timeout=timeout)
            outcomes.extend(window.outcomes)
            n_exps += len(window.experiences)
            records.append({
                "round": r, "games": len(window.games), "seconds": window.seconds,
                "timed_out": window.timed_out,
                "leaves_per_s": (pool.last_served_forwards or 0) / max(1e-9, window.seconds),
                "saturated_leaves_per_s": pool.last_saturated_leaves_per_s,
                "queue_depth": pool.last_queue_depth,
                "game_seconds_p50": window.game_seconds_p50,
                "straddle_mean": window.straddle_mean, "straddle_max": window.straddle_max,
                "straddled_share": window.straddled_share,
                "decisions": window.decisions})
            # Counted, not kept: a window's experiences are the size of
            # an iteration's, and a run of several windows hoarding them
            # all in the learner pushed a 32 GB host into swap next to
            # 48 actors (2026-09-18, the box's first stream arm died
            # there). The record above has what the run wants.
            del window.games[:]
            if step_seconds:
                time.sleep(step_seconds)
            stream.publish()
    finally:
        tail = stream.stop(grace=120.0)
        outcomes.extend(tail.outcomes)
        n_exps += len(tail.experiences)
    # The run's own totals: leaves the in-process server served from
    # the first window to the end of the drain, over that span.
    pool.last_served_forwards = stream.leaves_served() - served_before
    pool.last_iteration_seconds = time.monotonic() - t_run
    pool.last_saturated_leaves_per_s = max(
        (r["saturated_leaves_per_s"] or 0 for r in records), default=None)
    p50s = sorted(r["game_seconds_p50"] for r in records if r["game_seconds_p50"] is not None)
    pool.last_game_finish_p50 = p50s[len(p50s) // 2] if p50s else None
    pool.last_game_finish_max = None
    pool.last_decisions = sum(r["decisions"] for r in records)
    return outcomes, n_exps, records


def run_pool(policy, *, actors: int, games: int, sims: int, leaf_batch: int,
             server_priors: bool, max_turns: int, device, seed: int,
             iteration_timeout: float, log_level: int = logging.WARNING,
             max_batch: int = 64, serve_threads: int = 2, packed_embed: bool = False,
             coalesce: str = "fifo", coalesce_gap: int = 0,
             serve_processes: int = 1, graphed_serve: bool = False,
             stream_rounds: int = 0, step_seconds: float = 0.0) -> dict:
    from tools.actor_pool import ActorPool
    from tools.mcts import MCTSConfig
    from tools.mcts_policy import MCTSPolicy, ReplayConfig
    from tools.sim_self_play import k_median_of
    from tools.wesnoth_sim import PvPDefaults
    if games < actors and not stream_rounds:
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
                     serve_threads=serve_threads, packed_embed=packed_embed,
                     coalesce=coalesce, coalesce_gap=coalesce_gap,
                     serve_processes=serve_processes, graphed_serve=graphed_serve)
    pool.start()
    t0 = time.monotonic()
    rounds: list = []
    try:
        parity = _server_parity(pool, seed) if serve_processes > 1 else None
        if parity:
            log.info("serve-process parity on one leaf: %s", parity)
        if stream_rounds:
            outcomes, n_exps, rounds = _run_stream(pool, games, seed, stream_rounds,
                                                   step_seconds, iteration_timeout)
        else:
            outcomes, exps = pool.run_iteration(0, games, seed)
            n_exps = len(exps)
    finally:
        pool.shutdown()
    wall = time.monotonic() - t0
    gen = getattr(pool, "last_iteration_seconds", None) or wall
    served = getattr(pool, "last_served_forwards", 0) or 0
    decided = sum(1 for o in outcomes if o.winner != 0)
    # A run that hit the iteration timeout spends its tail with most
    # actors idle; its wall-clock rates are lower bounds (2026-09-04
    # review: the two truncated baseline rows are under-reported).
    truncated = (len(outcomes) < games) if not stream_rounds else any(
        r["timed_out"] for r in rounds)
    base = getattr(policy, "_inference_base", policy._inference_model)
    res = {
        "truncated": truncated,
        "server_priors": bool(server_priors), "actors": actors, "games_requested": games,
        "infer_bf16": bool(getattr(policy._inference_model, "infer_autocast_bf16", False)
                           or getattr(policy, "_infer_bf16", False)),
        "sims": sims, "leaf_batch": leaf_batch, "max_turns": max_turns,
        "max_batch": max_batch, "serve_threads": serve_threads,
        # Servers in total (the learner process plus N-1 serve processes),
        # the leaves each served, and how far each serve process's copy
        # of the model is from the learner's on one leaf (bf16 noise on
        # cuda, exact on cpu).
        "serve_processes": serve_processes,
        "leaves_per_server": getattr(pool, "last_leaves_per_server", None),
        "server_parity": parity,
        "packed_trunk": bool(getattr(base, "infer_packed_trunk", False)),
        "packed_embed": bool(packed_embed),
        "graphed_serve": bool(graphed_serve),
        "graphed_serve_summary": pool._server.graphed_summary(),
        "coalesce": coalesce, "coalesce_gap": coalesce_gap,
        # Continuous generation (tools/actor_stream.py): one row per
        # window, and the whole run's rate below (games per hour over
        # every window and the drain; the barrier's tail is gone, so
        # the iteration rate and the saturated rate should meet).
        "stream": bool(stream_rounds), "stream_rounds": rounds,
        "step_seconds": step_seconds,
        # The stream's steady state: the windows' games over their
        # seconds plus the idle after each (a step's stand-in), the
        # drain left out. games_per_hour above spans the whole run,
        # drain included, which a campaign pays once.
        "window_games_per_hour": (
            3600.0 * sum(r["games"] for r in rounds)
            / max(1e-9, sum(r["seconds"] + step_seconds for r in rounds))
            if rounds else None),
        # Warmup seconds, recompiles and any eager fallback of the
        # compiled packed trunk, over every server (design note
        # section 13; each serve process compiles its own copy).
        "packed_compile": merged_packed_compile(
            getattr(pool, "last_packed_compile_per_server", None)
            or [base.packed_compile_stats()]),
        "games_completed": len(outcomes), "decisive": decided,
        "abandoned": getattr(pool, "_last_abandoned", None),
        "experiences": n_exps,
        "gen_seconds": gen, "wall_seconds": wall,
        "forwards": served, "decisions": getattr(pool, "last_decisions", None),
        "leaves_per_s": served / gen if gen else None,
        # The iteration average includes the tail where most actors
        # have finished; this is the best 60-s window (the fed rate).
        "saturated_leaves_per_s": getattr(pool, "last_saturated_leaves_per_s", None),
        "leaf_timeline": getattr(pool, "last_leaf_timeline", None),
        "tokens_per_leaf": getattr(pool, "last_tokens_per_leaf", None),
        "pad_ratio": getattr(pool, "last_pad_ratio", None),
        # Host milliseconds per batch by stage (design note section 14)
        # and what the batch picker saw.
        "host_ms_per_batch": getattr(pool, "last_host_ms", None),
        "queue_depth": getattr(pool, "last_queue_depth", None),
        "skipped_requests": getattr(pool, "last_skipped_requests", None),
        "game_finish_p50_s": getattr(pool, "last_game_finish_p50", None),
        "game_finish_max_s": getattr(pool, "last_game_finish_max", None),
        "mean_turns": (statistics.fmean(o.turns for o in outcomes) if outcomes else None),
        "k_median": k_median_of(outcomes) if outcomes else None,
        "games_per_hour": (len(outcomes) / (gen / 3600.0)) if gen else None,
    }
    return res


def _server_parity(pool, seed: int) -> list:
    """One ladder leaf through every server: the largest absolute
    difference of each serve process's value and value logits from
    the learner process's."""
    import random
    from tools.scenario_pool import build_scenario_gamestate, load_factions, random_setup
    load_factions()
    gs = build_scenario_gamestate(random_setup(random.Random(seed), forced_faction=None,
                                               category="ladder"))
    outs = pool.probe([gs])
    ref = outs[0][0]
    return [{"value_abs_diff": float((o[0].value - ref.value).abs().max()),
             "value_logits_abs_diff": float((o[0].value_logits - ref.value_logits).abs().max())}
            for o in outs[1:]]


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--actors", type=int, default=19)
    ap.add_argument("--games", type=int, default=19)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--leaf-batch", type=int, default=16)
    ap.add_argument("--max-batch", type=int, default=64,
                    help="Leaves the server coalesces per batch across actors "
                         "(ActorPool max_batch). 64 since 2026-09-21: 1.34x the "
                         "saturated rate of the az legs' 16 on a quiet 4090 host "
                         "(docs/serve_batch_prereg_20260920.md).")
    ap.add_argument("--serve-threads", type=int, default=2,
                    help="Serving threads per server (2 in the az legs); "
                         "more overlap the per-batch Python with the GPU wait.")
    ap.add_argument("--serve-processes", type=int, default=1,
                    help="Servers in total: the pool process plus N-1 serve processes, "
                         "each with its own copy of the model on the same device, its "
                         "own request queue and serve threads; actors are assigned "
                         "round-robin (docs/box_specs.md, 'Serve processes: how to run').")
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
    ap.add_argument("--packed-trunk", action="store_true",
                    help="Run the server's trunk on the packed sequence (flash "
                         "varlen, wesnoth_ai/packed_trunk.py; needs cuda + bf16).")
    ap.add_argument("--compile-packed", action="store_true",
                    help="torch.compile the packed layer loop over native bf16 weights "
                         "(design note section 13; needs --packed-trunk).")
    ap.add_argument("--compile-packed-mode", default="default",
                    choices=("default", "max-autotune-no-cudagraphs"))
    ap.add_argument("--packed-embed", action="store_true",
                    help="Embed each batch from one pinned buffer straight into the "
                         "trunk's layout (design note section 14; any device, any trunk).")
    ap.add_argument("--graphed-serve", action="store_true",
                    help="Every server replays its priors batches from per-bucket CUDA "
                         "graphs (wesnoth_ai/graphed_serve.py; needs --infer-bf16 "
                         "--packed-trunk on cuda).")
    ap.add_argument("--coalesce", default="fifo", choices=("fifo", "length"),
                    help="How a serve thread picks a batch from the queued requests: "
                         "arrival order, or the requests nearest in token count to the "
                         "oldest one when more wait than one batch takes (section 7).")
    ap.add_argument("--coalesce-gap", type=int, default=0,
                    help="With --coalesce length: refuse a request further than this many "
                         "tokens from the batch's anchor even if the batch is not full "
                         "(0: no gap rule).")
    ap.add_argument("--stream", action="store_true",
                    help="Continuous generation (tools/actor_stream.py): --rounds "
                         "windows of --games completed games with every actor playing "
                         "throughout, instead of one barrier iteration.")
    ap.add_argument("--rounds", type=int, default=3,
                    help="Windows to collect under --stream.")
    ap.add_argument("--step-seconds", type=float, default=0.0,
                    help="Idle seconds between windows under --stream, standing in "
                         "for the learner's step (serving goes on meanwhile).")
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
    if args.packed_trunk:
        if device.type != "cuda" or not args.infer_bf16:
            raise SystemExit("--packed-trunk needs --device cuda and --infer-bf16")
        getattr(policy, "_inference_base", policy._inference_model).infer_packed_trunk = True
    if args.compile_packed:
        if not args.packed_trunk:
            raise SystemExit("--compile-packed needs --packed-trunk")
        base = getattr(policy, "_inference_base", policy._inference_model)
        mode = None if args.compile_packed_mode == "default" else args.compile_packed_mode
        base.configure_packed_compile(mode=mode)
        log.info("packed compile warmup: %s", base.warmup_packed_compile())
    res = run_pool(policy, actors=args.actors, games=args.games, sims=args.sims,
                   leaf_batch=args.leaf_batch, server_priors=args.server_priors,
                   max_turns=args.max_turns, device=device, seed=args.seed,
                   iteration_timeout=args.iteration_timeout, max_batch=args.max_batch,
                   serve_threads=args.serve_threads, packed_embed=args.packed_embed,
                   coalesce=args.coalesce, coalesce_gap=args.coalesce_gap,
                   serve_processes=args.serve_processes, graphed_serve=args.graphed_serve,
                   stream_rounds=(int(args.rounds) if args.stream else 0),
                   step_seconds=float(args.step_seconds))
    if args.dollars_per_hour and res["games_per_hour"]:
        res["games_per_dollar"] = res["games_per_hour"] / args.dollars_per_hour
    print(json.dumps(res, indent=1))
    if args.out:
        args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
