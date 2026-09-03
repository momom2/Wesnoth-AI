#!/usr/bin/env python3
"""Step-scale measurement: how far along one real update direction
can the policy move before it breaks?

Reproduces the minimal loop's first update from the seed (fresh
Adam moments => the update is lr * sign(g) on every parameter),
then walks theta0 + alpha * delta for a grid of alpha and reads, at
each scale:

  * held-out loss (policy CE + value loss on a fifth of the games,
    never stepped on) -- what the backtracking rule sees;
  * KL(pi_old || pi_new) on real held-out states -- what a KL-budget
    rule would see;
  * searched self-play games -- K median, decisive rate, end_turn
    share: the thing that actually collapsed.

    python signal_profiler/run_step_scale.py \
        --checkpoint training/checkpoints/seed_imit_tierb_start.pt \
        --games 24 --probe-games 8 --actors 19 --out step_scale.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.step_control import (  # noqa: E402
    action_priors, held_loss, policy_shift, publish_weights, split_holdout,
)

log = logging.getLogger("step_scale")


def game_stats(outcomes) -> Dict:
    from tools.sim_self_play import k_median_of
    tot = sum(sum(o.action_counts.values()) for o in outcomes) or 1
    return {"n_games": len(outcomes),
            "decisive": sum(1 for o in outcomes if o.winner != 0),
            "k_median": k_median_of(outcomes),
            "mean_turns": statistics.fmean(o.turns for o in outcomes),
            "end_turn_pct": 100.0 * sum(o.action_counts.get("end_turn", 0)
                                        for o in outcomes) / tot,
            "attack_pct": 100.0 * sum(o.action_counts.get("attack", 0)
                                      for o in outcomes) / tot}


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--probe-games", type=int, default=8)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--scales", default="1,0.3,0.1,0.03")
    ap.add_argument("--kl-states", type=int, default=200)
    ap.add_argument("--actors", type=int, default=8)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    import torch
    from tools.actor_pool import ActorPool
    from tools.eval_sim import _load_policy
    from tools.mcts import MCTSConfig
    from tools.mcts_policy import MCTSPolicy, ReplayConfig
    from tools.wesnoth_sim import PvPDefaults

    device = (torch.device("cuda")
              if args.device == "cuda" and torch.cuda.is_available()
              else torch.device("cpu"))
    dev_str = "cuda" if device.type == "cuda" else "cpu"
    base = _load_policy(args.checkpoint, device, label="step_scale")
    cfg = base._trainer.config
    cfg.value_loss_form = "mse_mean"
    cfg.value_coef = 1.0
    cfg.learning_rate = args.lr
    for g in base._trainer.optimizer.param_groups:
        g["lr"] = args.lr
    cfg.aux_coef = cfg.gbc_coef = cfg.moves_left_coef = 0.0
    cfg.value_label_smoothing = 0.0
    cfg.trust_lambda = 0.0
    cfg.grad_clip = 1.0
    log.info(f"optimizer state entries at load: {len(base._trainer.optimizer.state)}")

    mcts_cfg = MCTSConfig(
        n_simulations=args.sims, gumbel_root=False, tree_reuse=False,
        playout_cap_randomization=False, draw_tiebreak=None,
        batch_size=16 if dev_str == "cuda" else 1)
    policy = MCTSPolicy(base, mcts_cfg,
                        replay_config=ReplayConfig(enabled=False),
                        holdout_size=0, gbc_labels=False)
    scenario_opts = dict(forced_faction=None, mini_maps=None,
                         mini_ratio=0.0, fogless_ratio=0.0,
                         ladder_ratio=1.0, midgame_ratio=0.0,
                         midgame_dataset=None)
    pool = ActorPool(policy, args.actors, mcts_cfg, turn_cfg=None,
                     pt_cfg=None, gbc_labels=False, train_kwargs={},
                     scenario_opts=scenario_opts, max_turns=args.max_turns,
                     max_turns_min=args.max_turns,
                     pvp_defaults=PvPDefaults(), device=device,
                     max_batch=16, log_level=logging.WARNING)
    pool.start()
    rng = random.Random(args.seed)
    report: Dict = {"checkpoint": str(args.checkpoint), "lr": args.lr,
                    "sims": args.sims, "scales": {}}
    try:
        # ---- harvest = the alpha-0 game probe ------------------------
        t0 = time.monotonic()
        outcomes, exps = pool.run_iteration(0, args.games, rng.randint(0, 2**31 - 1))
        capped = {o.game_label for o in outcomes if o.winner == 0}
        kept = [e for e in exps if getattr(e, "game_id", "") not in capped]
        train_exps, held_exps = split_holdout(kept, args.holdout_frac, rng)
        log.info(f"harvest {time.monotonic() - t0:.0f}s: games {len(outcomes)} "
                 f"capped {len(capped)} exps train {len(train_exps)} "
                 f"held {len(held_exps)}")
        report["harvest"] = dict(game_stats(outcomes), seconds=time.monotonic() - t0,
                                 n_train=len(train_exps), n_held=len(held_exps))

        theta0 = {k: v.detach().clone() for k, v in base._model.state_dict().items()}
        kl_states = (held_exps if len(held_exps) <= args.kl_states
                     else rng.sample(held_exps, args.kl_states))
        old_priors = [action_priors(base, e) for e in kl_states]
        loss0 = held_loss(base, held_exps)
        log.info(f"alpha=0 held loss {loss0}")

        # ---- the real first step (fresh moments) --------------------
        st = base._trainer.step_mcts(list(train_exps))
        theta1 = {k: v.detach().clone() for k, v in base._model.state_dict().items()}
        delta = {}
        tot = near = 0
        for k, v in theta0.items():
            if torch.is_floating_point(v):
                d = theta1[k] - v
                delta[k] = d
                r = (d.abs() / args.lr).flatten()
                tot += r.numel()
                near += int(((r > 0.9) & (r < 1.1)).sum())
        report["step"] = {"policy_loss": float(st.policy_loss),
                          "value_loss": float(st.value_loss),
                          "grad_norm": float(st.grad_norm),
                          "frac_params_moved_by_lr": near / max(tot, 1)}
        log.info(f"step: {report['step']}")

        # ---- walk the scales ----------------------------------------
        scales = [0.0] + [float(s) for s in args.scales.split(",") if s]
        for i, alpha in enumerate(scales):
            publish_weights(base, {k: (v + alpha * delta[k] if k in delta else v)
                                   for k, v in theta0.items()})
            row: Dict = {"alpha": alpha}
            row["held_loss"] = held_loss(base, held_exps) if alpha else loss0
            row["shift"] = policy_shift(base, kl_states, old_priors)
            if alpha:
                t1 = time.monotonic()
                oc, _ = pool.run_iteration(i, args.probe_games,
                                           rng.randint(0, 2**31 - 1))
                row["games"] = dict(game_stats(oc), seconds=time.monotonic() - t1)
            else:
                row["games"] = report["harvest"]
            report["scales"][str(alpha)] = row
            log.info(f"alpha={alpha}: held {row['held_loss']} | shift "
                     f"{row['shift']} | games {row['games']}")
            args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    finally:
        pool.stop()
    print("\n alpha   held_CE  held_val   KL_med  end_turn_prior   K  decisive  end_turn%  turns")
    for a, r in report["scales"].items():
        g, s, h = r["games"], r["shift"], r["held_loss"]
        print(f"{float(a):6.2f}  {h['policy_ce']:8.4f} {h['value_loss']:8.4f} "
              f"{s['kl_median']:8.4f} {s['end_turn_prior_mean']:14.4f} "
              f"{g['k_median']:>4} {g['decisive']:>3}/{g['n_games']:<3} "
              f"{g['end_turn_pct']:8.1f} {g['mean_turns']:6.1f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
