"""Independent self-play worker: plays full games IN-PROCESS (own
encoder/model, own GPU forwards — no inference server, no IPC) and
spools each finished game to a file for the learner to consume.

Why this exists (measured, 2026-07-03/06): the central-server actor
pool is latency-bound at ~200 req/s with the GPU ~idle during
rollout, while N independent processes doing in-process forwards
saturated a 4090 at 99%. This worker IS that winning pattern, made
production: the spool directory is the seam between rollout and
learning.

Protocol:
  - One game per loop: fresh random setup (mini/drill ratios from
    args), MCTS self-play via the production play_one_game, then the
    game's MCTSExperiences + GameOutcome are pickled ATOMICALLY
    (tmp + os.replace) to <spool>/game_<worker>_<n>.pkl.
  - Between games the worker re-checks the checkpoint file's mtime
    and hot-reloads weights when the learner saved a newer one. The
    combat-oracle anneal uses the checkpoint's decision_step (same
    staleness bound as the old pool's per-iteration broadcast).
  - Holdout diversion does NOT happen here (worker holdout_size=0);
    the learner offers each game file to its holdout, preserving
    whole-game granularity.
  - CUDA OOM at init (too many sibling workers for the card) falls
    back to CPU forwards for this worker rather than dying.

Usage (normally spawned by sim_self_play --spool-workers):
    python tools/selfplay_worker.py --worker-id 0 \
        --checkpoint training/checkpoints/tier_a_campaign.pt \
        --spool-dir /workspace/spool --mcts-sims 32 \
        --mini-ratio 0.5 --drill-ratio 0.3 --max-turns 200 \
        --moves-left-utility 0.2 --seed 1234
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("selfplay_worker")


def _build_policy(ckpt: Path, device, args):
    import torch
    from tools.draw_tiebreak import DrawTiebreakConfig
    from tools.mcts import MCTSConfig
    from tools.mcts_policy import MCTSPolicy
    from transformer_policy import TransformerPolicy

    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    arch = raw.get("arch", {})
    base = TransformerPolicy(
        d_model=arch.get("d_model", 128),
        num_layers=arch.get("num_layers", 3),
        num_heads=arch.get("num_heads", 4),
        d_ff=arch.get("d_ff", 256),
        device=device,
        aux_score=bool(raw.get("aux_score", False)),
        moves_left=bool(raw.get("moves_left", False)),
    )
    base.load_checkpoint(ckpt)
    cfg = MCTSConfig(
        n_simulations=args.mcts_sims,
        draw_tiebreak=DrawTiebreakConfig(cap=args.draw_tiebreak_cap),
        moves_left_utility=args.moves_left_utility,
    )
    return MCTSPolicy(base, cfg), base


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker-id", type=int, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--spool-dir", type=Path, required=True)
    ap.add_argument("--mcts-sims", type=int, default=32)
    ap.add_argument("--mini-ratio", type=float, default=0.5)
    ap.add_argument("--drill-ratio", type=float, default=0.3)
    ap.add_argument("--max-turns", type=int, default=200)
    ap.add_argument("--draw-tiebreak-cap", type=float, default=0.3)
    ap.add_argument("--moves-left-utility", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args(argv[1:])
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=f"%(asctime)s w{args.worker_id} %(levelname)s %(message)s")

    import torch
    torch.set_num_threads(max(1, args.torch_threads))

    # Heavy imports AFTER thread cap.
    from tools.actor_pool import _zero_reward, _set_fd_safe_sharing
    from tools.scenario_pool import build_scenario_gamestate, random_setup
    from tools.sim_self_play import _recruit_cost_lookup, play_one_game
    from wesnoth_sim import PvPDefaults, WesnothSim

    _set_fd_safe_sharing()

    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    try:
        policy, base = _build_policy(args.checkpoint, device, args)
    except torch.cuda.OutOfMemoryError:
        log.warning("CUDA OOM at init (too many workers for the "
                    "card?); falling back to CPU forwards.")
        device = None
        policy, base = _build_policy(args.checkpoint, None, args)

    spool = args.spool_dir / "games"
    spool.mkdir(parents=True, exist_ok=True)
    pvp = PvPDefaults()
    cost = _recruit_cost_lookup()
    rng = random.Random(args.seed)
    ckpt_mtime = args.checkpoint.stat().st_mtime
    n = 0
    while True:
        # Hot-reload the learner's latest weights between games.
        try:
            m = args.checkpoint.stat().st_mtime
            if m != ckpt_mtime:
                base.load_checkpoint(args.checkpoint)
                ckpt_mtime = m
                log.info("reloaded checkpoint (decision_step="
                         f"{base._decision_step})")
        except (OSError, RuntimeError, EOFError) as e:
            # Mid-save read or transient FS error: play on with the
            # current weights, retry next game.
            log.warning(f"checkpoint reload skipped: {e}")

        setup = random_setup(rng, mini_ratio=args.mini_ratio,
                             drill_ratio=args.drill_ratio)
        gs = build_scenario_gamestate(
            setup, base_income=pvp.base_income,
            village_gold=pvp.village_gold,
            village_upkeep=pvp.village_support,
            experience_modifier=pvp.experience_modifier)
        sim = WesnothSim(gs, scenario_id=setup.scenario_id,
                         max_turns=args.max_turns)
        label = f"w{args.worker_id}g{n}"
        ds_before = base._decision_step
        outcome = play_one_game(sim, policy, _zero_reward,
                                game_label=label, cost_lookup=cost)
        with policy._lock:
            exps = policy._queue
            policy._queue = []
        payload = {
            "experiences": exps,
            "outcome": outcome,
            "n_decisions": base._decision_step - ds_before,
            "worker": args.worker_id,
            "ckpt_step": ds_before,
        }
        tmp = spool / f".tmp_w{args.worker_id}_{n}"
        final = spool / f"game_w{args.worker_id}_{n:06d}.pkl"
        with tmp.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, final)
        n += 1
        log.info(f"spooled {final.name}: {len(exps)} exps, "
                 f"winner={outcome.winner} map={outcome.map_class}")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
