#!/usr/bin/env python3
"""Minimal self-play loop -- docs/az_minimal_spec.md, nothing else.

    play N games with plain MCTS  ->  targets = visit counts,
    value = game result           ->  ONE gradient step  ->  repeat

Imports only the simulator/actor pool, encoder, model, the core
trainer step, MCTS, and the measurement instruments. Nothing from
the quarantined pile (docs/../quarantine/README.md): no TCS, replay
buffer, anchors, GBC/aux heads, memory, grounding, trust region.

Three data streams, every iteration, to az_history.csv:
  performance : decisive rate, actions/side-turn median (K), action
                mix, and -- every --pin-every iterations -- raw net
                vs raw seed and net+search vs seed+search matches;
  signal      : policy vs value gradient norm and share (unclipped,
                on a 128-state subsample), target-vs-prior KL/TV,
                value CE / floor / AUC by turn decade, plus the deep
                profile on every pin (signal_profiler v2);
  time        : generation seconds, forwards served, decisions,
                forwards/s, train seconds, probe/profile seconds.

Kill: raw pin <= seed at the end of the budget (read by hand), or
K median < 10 for 3 consecutive iterations (exit 7 + ABORTED_7).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

log = logging.getLogger("az_loop")

COLUMNS = [
    "iter", "decision_step", "n_games", "decisive", "s1_wins", "s2_wins",
    "capped_discarded", "k_median", "action_attack_pct",
    "action_end_turn_pct", "action_move_pct", "action_recruit_pct",
    "mean_turns", "n_experiences",
    # signal
    "policy_loss", "value_loss", "grad_norm",
    "sig_policy_norm", "sig_value_norm", "sig_value_share",
    "target_kl_median", "target_kl_mean", "target_tv_mean",
    "target_end_turn_delta", "target_attack_delta",
    "fresh_value_ce", "fresh_ce_floor", "fresh_value_auc",
    "fresh_auc_d1_10", "fresh_auc_d11_20", "fresh_auc_d21_30",
    "fresh_ce_d1_10", "fresh_ce_d11_20", "fresh_ce_d21_30",
    # time
    "gen_seconds", "forwards", "decisions", "forwards_per_s",
    "decisions_per_s", "train_seconds", "telemetry_seconds",
    "probe_seconds", "profile_seconds", "iter_seconds",
    # pins
    "pin_step", "raw_vs_seed_wdl", "search_vs_seed_wdl",
]


def _wdl(games_dir: Path) -> str:
    """W-D-L of A over the game jsons run_elo_batch wrote."""
    w = d = l_ = 0
    for f in games_dir.glob("game_*.json"):
        try:
            r = json.load(open(f)).get("outcome_a")
        except Exception:  # noqa: BLE001
            continue
        if r is None:
            continue
        if r > 0:
            w += 1
        elif r < 0:
            l_ += 1
        else:
            d += 1
    return f"{w}-{d}-{l_}"


def _probe(pin: Path, seed: Path, outdir: Path, games: int, sims: int,
           device: str) -> str:
    cmd = [sys.executable, str(ROOT / "tools" / "run_elo_batch.py"),
           "--label-a", pin.stem, "--spec-a", str(pin),
           "--label-b", "seed", "--spec-b", str(seed),
           "--games", str(games), "--mcts-sims", str(sims),
           "--no-turn-search", "--device", device,
           "--outdir", str(outdir), "--time-budget-min", "150",
           "--min-free-mb", "500"]
    subprocess.run(cmd, cwd=str(ROOT), check=False)
    return _wdl(outdir)


def _profile(pin: Path, out: Path, device: str) -> None:
    cmd = [sys.executable, str(ROOT / "signal_profiler" / "run_profile_v2.py"),
           "--checkpoint", str(pin), "--games", "4", "--consult-cap", "0",
           "--no-turn-search", "--seed", "31337", "--device", device,
           "--out", str(out)]
    with open(out.with_suffix(".out"), "w") as f:
        subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT,
                       check=False)


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-checkpoint", type=Path, required=True)
    ap.add_argument("--campaign", type=Path, required=True,
                    help="rolling checkpoint (resumed if it exists)")
    ap.add_argument("--workdir", type=Path, default=Path("/workspace"))
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--games-per-iter", type=int, default=24)
    ap.add_argument("--actors", type=int, default=8)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--value-coef", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--pin-every", type=int, default=10)
    ap.add_argument("--probe-games", type=int, default=40)
    ap.add_argument("--search-probe-every-pins", type=int, default=2)
    ap.add_argument("--abort-k-median", type=float, default=10.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rng-seed", type=int, default=20260903)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv[1:])
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S")

    import torch
    from tools.actor_pool import ActorPool
    from tools.eval_sim import _load_policy
    from tools.mcts import MCTSConfig
    from tools.mcts_policy import MCTSPolicy, ReplayConfig
    from tools.sim_self_play import k_median_of
    from tools.wesnoth_sim import PvPDefaults
    from tools.signal_telemetry import signal_grad_norms
    from signal_profiler.target_amplitude import target_amplitude

    device = (torch.device("cuda")
              if args.device == "cuda" and torch.cuda.is_available()
              else torch.device("cpu"))
    dev_str = "cuda" if device.type == "cuda" else "cpu"
    ckpt_in = args.campaign if args.campaign.exists() else args.seed_checkpoint
    base = _load_policy(ckpt_in, device, label="az")
    cfg = base._trainer.config
    cfg.value_loss_form = "mse_mean"
    cfg.value_coef = float(args.value_coef)
    cfg.learning_rate = float(args.lr)
    for g in base._trainer.optimizer.param_groups:
        g["lr"] = float(args.lr)
    cfg.aux_coef = 0.0
    cfg.gbc_coef = 0.0
    cfg.moves_left_coef = 0.0
    cfg.value_label_smoothing = 0.0
    cfg.trust_lambda = 0.0
    cfg.grad_clip = 1.0
    log.info(f"loaded {ckpt_in.name} decision_step={base._decision_step} "
             f"| value_loss={cfg.value_loss_form} c={cfg.value_coef} "
             f"lr={cfg.learning_rate} clip={cfg.grad_clip}")

    # Plain PUCT: no Gumbel root, no tree reuse, no playout caps, no
    # tiebreak labels, no auxiliary utilities.
    mcts_cfg = MCTSConfig(
        n_simulations=args.sims, gumbel_root=False, tree_reuse=False,
        playout_cap_randomization=False, draw_tiebreak=None,
        batch_size=16 if dev_str == "cuda" else 1)
    policy = MCTSPolicy(base, mcts_cfg,
                        replay_config=ReplayConfig(enabled=False),
                        holdout_size=0, gbc_labels=False,
                        signal_telemetry=True)
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

    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    csv_path = workdir / "az_history.csv"
    new_csv = not csv_path.exists()
    fh = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
    if new_csv:
        writer.writeheader()
    rng = random.Random(args.rng_seed + int(base._decision_step))
    k_low = 0
    pins_done = 0
    try:
        for it in range(args.iterations):
            t_it = time.monotonic()
            row: Dict = {"iter": it, "decision_step": base._decision_step}

            # ---- generation --------------------------------------
            outcomes, exps = pool.run_iteration(it, args.games_per_iter,
                                                rng.randint(0, 2**31 - 1))
            capped = {o.game_label for o in outcomes if o.winner == 0}
            kept = [e for e in exps if getattr(e, "game_id", "") not in capped]
            row.update(
                n_games=len(outcomes),
                decisive=sum(1 for o in outcomes if o.winner != 0),
                s1_wins=sum(1 for o in outcomes if o.winner == 1),
                s2_wins=sum(1 for o in outcomes if o.winner == 2),
                capped_discarded=len(capped),
                k_median=k_median_of(outcomes),
                mean_turns=(sum(o.turns for o in outcomes) / len(outcomes)
                            if outcomes else None),
                n_experiences=len(kept),
                gen_seconds=getattr(pool, "last_iteration_seconds", None),
                forwards=getattr(pool, "last_served_forwards", None),
                decisions=getattr(pool, "last_decisions", None))
            tot_actions = sum(sum(o.action_counts.values()) for o in outcomes) or 1
            for k in ("attack", "end_turn", "move", "recruit"):
                row[f"action_{k}_pct"] = 100.0 * sum(
                    o.action_counts.get(k, 0) for o in outcomes) / tot_actions
            if row["gen_seconds"]:
                row["forwards_per_s"] = (row["forwards"] or 0) / row["gen_seconds"]
                row["decisions_per_s"] = (row["decisions"] or 0) / row["gen_seconds"]

            # ---- signal: pre-step value probe + target amplitude --
            t_tel = time.monotonic()
            if kept:
                sample = kept if len(kept) <= 256 else rng.sample(kept, 256)
                fm = base._trainer.eval_value_metrics(sample)
                row.update(fresh_value_ce=fm["ce"],
                           fresh_ce_floor=fm["marginal_ce_floor"],
                           fresh_value_auc=fm["value_auc"])
                for dkey in ("d1_10", "d11_20", "d21_30"):
                    bd = fm.get("by_decade", {}).get(dkey)
                    if bd:
                        row[f"fresh_auc_{dkey}"] = bd["auc"]
                        row[f"fresh_ce_{dkey}"] = bd["ce"]
                ta = target_amplitude(policy, sample[:96])
                if ta.get("n"):
                    cats = ta.get("category_mass_delta_mean", {})
                    row.update(target_kl_median=ta["kl_median"],
                               target_kl_mean=ta["kl_mean"],
                               target_tv_mean=ta["tv_mean"],
                               target_end_turn_delta=cats.get("end_turn"),
                               target_attack_delta=cats.get("attack"))
            # ---- ONE gradient step --------------------------------
            t_tr = time.monotonic()
            with policy._lock:
                policy._queue = list(kept)
            stats = policy.train_step()
            row.update(policy_loss=stats.policy_loss,
                       value_loss=stats.value_loss,
                       grad_norm=stats.grad_norm,
                       train_seconds=time.monotonic() - t_tr)
            # per-source gradient norms (unclipped, optimizer stubbed)
            norms = signal_grad_norms(base._trainer, kept, rng) if kept else {}
            pn = norms.get("sig_policy_norm")
            vn = norms.get("sig_value_game_norm")
            row.update(sig_policy_norm=pn, sig_value_norm=vn,
                       sig_value_share=((vn ** 2) / (pn ** 2 + vn ** 2)
                                        if pn is not None and vn is not None
                                        and (pn or vn) else None))
            row["telemetry_seconds"] = (time.monotonic() - t_tel) - row["train_seconds"]
            base.save_checkpoint(args.campaign)

            # ---- tripwire ------------------------------------------
            km = row["k_median"]
            k_low = k_low + 1 if (km is not None and km < args.abort_k_median) else 0
            if k_low >= 3:
                log.error(f"ABORT: K median {km} < {args.abort_k_median} "
                          f"for 3 consecutive iterations (iter {it})")
                writer.writerow(row)
                fh.flush()
                (workdir / "ABORTED_7").touch()
                return 7

            # ---- pins: probes + deep profile ---------------------
            if (it + 1) % args.pin_every == 0:
                pins_done += 1
                step = int(base._decision_step)
                pin = workdir / "pins" / f"pin_{step}.pt"
                pin.parent.mkdir(exist_ok=True)
                shutil.copy(args.campaign, pin)
                t_pr = time.monotonic()
                row["pin_step"] = step
                row["raw_vs_seed_wdl"] = _probe(
                    pin, args.seed_checkpoint,
                    workdir / "probes" / f"raw_{step}", args.probe_games, 0,
                    dev_str)
                if pins_done % args.search_probe_every_pins == 0:
                    row["search_vs_seed_wdl"] = _probe(
                        pin, args.seed_checkpoint,
                        workdir / "probes" / f"search_{step}",
                        args.probe_games, args.sims, dev_str)
                row["probe_seconds"] = time.monotonic() - t_pr
                t_pf = time.monotonic()
                (workdir / "profiles").mkdir(exist_ok=True)
                _profile(pin, workdir / "profiles" / f"pin_{step}.json", dev_str)
                row["profile_seconds"] = time.monotonic() - t_pf
                log.info(f"PIN {step}: raw {row['raw_vs_seed_wdl']} "
                         f"search {row.get('search_vs_seed_wdl', '-')}")

            row["iter_seconds"] = time.monotonic() - t_it
            writer.writerow(row)
            fh.flush()
            log.info(
                f"iter {it}: games {row['n_games']} dec {row['decisive']} "
                f"K {km} atk% {row['action_attack_pct']:.1f} | "
                f"loss p {stats.policy_loss:.4f} v {stats.value_loss:.4f} "
                f"| sig p/v {pn} {vn} share_v {row['sig_value_share']} "
                f"| kl {row.get('target_kl_median')} "
                f"| gen {row['gen_seconds']:.0f}s train {row['train_seconds']:.1f}s")
    finally:
        pool.shutdown()
        fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
