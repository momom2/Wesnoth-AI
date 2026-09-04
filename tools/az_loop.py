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
import collections
import csv
import gc
import json
import logging
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

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
    "fresh_value_mean", "fresh_label_mean",
    "fresh_auc_d1_10", "fresh_auc_d11_20", "fresh_auc_d21_30",
    "fresh_ce_d1_10", "fresh_ce_d11_20", "fresh_ce_d21_30",
    # time
    "gen_seconds", "forwards", "decisions", "forwards_per_s",
    "decisions_per_s", "tokens_per_leaf", "pad_ratio",
    "game_finish_p50", "game_finish_max", "gpu_reserved_mb", "rss_mb",
    "gc_seconds", "gc_gen2_seconds", "gc_gen2_count", "live_objects",
    "train_seconds", "telemetry_seconds",
    "probe_seconds", "profile_seconds", "iter_seconds",
    # step control (tools/step_control.py)
    "step_alpha", "step_trials", "held_before", "held_after",
    "held_delta_mean", "held_delta_se",
    "step_kl_median", "step_kl_mean", "step_tv_mean", "end_turn_prior",
    "step_dv_mean", "step_dv_abs_mean", "value_center", "value_level",
    # pins
    "pin_step", "raw_vs_seed_wdl", "search_vs_seed_wdl",
]


def _migrate_history_columns(csv_path: Path) -> None:
    """Rewrite an existing history CSV whose header differs from
    COLUMNS (columns were added mid-leg). Rows are mapped by their
    own width: header-width rows by the old header, COLUMNS-width
    rows by COLUMNS (rows appended after a column change but before
    a migration); anything else is dropped with a warning."""
    if not csv_path.exists():
        return
    with open(csv_path, newline="", encoding="utf-8") as f:
        raw = list(csv.reader(f))
    if not raw or raw[0] == COLUMNS:
        return
    old_header = raw[0]
    # Columns added since the header was written, in COLUMNS order;
    # a row of width len(old_header) + k was written by an interim
    # column list holding the first k of them at their positions.
    added = [c for c in COLUMNS if c not in old_header]
    widths = {}
    for k in range(len(added) + 1):
        interim = [c for c in COLUMNS if c in old_header or c in added[:k]]
        widths[len(interim)] = interim
    rows, dropped = [], 0
    for r in raw[1:]:
        cols = widths.get(len(r))
        if cols is None:
            dropped += 1
            continue
        rows.append(dict(zip(cols, r)))
    backup = csv_path.with_suffix(f".pre_migration_{int(time.time())}.csv")
    shutil.copy2(csv_path, backup)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    log.warning(f"history CSV migrated to {len(COLUMNS)} columns "
                f"({len(rows)} rows kept, {dropped} dropped; backup {backup.name})")


class _GcMeter:
    """Wall time spent in the cyclic garbage collector, by generation.
    Every collection stops all threads, including the two that serve
    the actors' leaves; a large live heap makes gen-2 sweeps long.
    Generation throughput decayed 2x over a process's life (az3);
    this says whether GC is the reason."""

    def __init__(self):
        self.seconds = [0.0, 0.0, 0.0]
        self.counts = [0, 0, 0]
        self._t0 = None
        gc.callbacks.append(self._cb)

    def _cb(self, phase, info):
        if phase == "start":
            self._t0 = time.monotonic()
        elif self._t0 is not None:
            g = int(info.get("generation", 0))
            self.seconds[g] += time.monotonic() - self._t0
            self.counts[g] += 1
            self._t0 = None

    def take(self):
        s, c = self.seconds, self.counts
        self.seconds, self.counts = [0.0, 0.0, 0.0], [0, 0, 0]
        return sum(s), s[2], c[2]


def _rss_mb() -> Optional[float]:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


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
           device: str, value_center: float = 0.0) -> str:
    cmd = [sys.executable, str(ROOT / "tools" / "run_elo_batch.py"),
           "--label-a", pin.stem, "--spec-a", str(pin),
           "--label-b", "seed", "--spec-b", str(seed),
           "--value-center-a", str(value_center),
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
    ap.add_argument("--holdout-frac", type=float, default=0.2,
                    help="Fraction of each iteration's GAMES held out "
                         "of the step; the applied update is shrunk "
                         "(1, 1/2, 1/4, ...) until their loss falls "
                         "(tools/step_control.py). 0 = fixed step.")
    ap.add_argument("--max-level-shift", type=float, default=0.08,
                    help="Largest mean value-head shift one step may "
                         "apply on the held-out states; the step is "
                         "shrunk until it fits. 0.08 = two C51 atoms, "
                         "the trust-region delta of "
                         "docs/design_constants.md. Negative = off.")
    ap.add_argument("--step-select", choices=("first", "best"), default="first",
                    help="first = largest fraction passing the held-out "
                         "test (Armijo); best = the passing fraction with "
                         "the lowest held-out loss among --step-trials "
                         "(exact line search on the held-out games).")
    ap.add_argument("--step-trials", type=int, default=7,
                    help="Fractions tried: 1, 1/2, ... 1/2^(n-1).")
    ap.add_argument("--value-center", action="store_true",
                    help="Search subtracts the value head's mean on the "
                         "latest batch from every value it reads "
                         "(MCTSConfig.value_center), so the head's level "
                         "cannot decide act-vs-end_turn. Off = plain.")
    ap.add_argument("--tempo-bonus", type=float, default=0.0,
                    help="With --value-center: search sees the mover's "
                         "positions this much above the head's level, "
                         "pricing the tempo that end_turn hands over. "
                         "0.44 = the seed's measured level on its own "
                         "self-play states, the act/end balance that "
                         "plays K 10-12 (docs/design_constants.md).")
    ap.add_argument("--iteration-timeout", type=float, default=1800.0,
                    help="Wall-clock seconds after which the pool drains; "
                         "in-flight games are abandoned 300 s later. One "
                         "game per actor makes the iteration as long as "
                         "its slowest game (az3 iteration 8: 45+ min on "
                         "one game).")
    ap.add_argument("--kl-states", type=int, default=100,
                    help="Held-out states on which the per-step policy "
                         "movement (KL, TV, end_turn mass) is measured.")
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--pin-every", type=int, default=10)
    ap.add_argument("--probe-games", type=int, default=40)
    ap.add_argument("--search-probe-every-pins", type=int, default=2)
    ap.add_argument("--start-iter", type=int, default=0,
                    help="First iteration index (resume: keeps the pin "
                         "cadence and CSV numbering of the running leg).")
    ap.add_argument("--abort-k-median", type=float, default=3.0,
                    help="Abort when K median stays below this for 3 "
                         "iterations. 3 = degenerate play (K 1-2, most "
                         "games undecided); K 5-8 is NOT collapse under "
                         "plain search with a level-correct value head "
                         "(step-scale measurement 2026-09-03: K 7 with "
                         "6/8 decisive at the value-loss optimum).")
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
    from tools.step_control import (
        action_priors, backtracking_step, split_holdout,
    )
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
                     max_batch=16, log_level=logging.WARNING,
                     iteration_timeout=args.iteration_timeout,
                     drain_grace=300.0)
    pool.start()

    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    csv_path = workdir / "az_history.csv"
    _migrate_history_columns(csv_path)
    new_csv = not csv_path.exists()
    fh = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
    if new_csv:
        writer.writeheader()
    rng = random.Random(args.rng_seed + int(base._decision_step))
    gc_meter = _GcMeter()
    k_low = 0
    pins_done = 0
    try:
        for it in range(args.start_iter, args.iterations):
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
                decisions=getattr(pool, "last_decisions", None),
                tokens_per_leaf=getattr(pool, "last_tokens_per_leaf", None),
                pad_ratio=getattr(pool, "last_pad_ratio", None),
                game_finish_p50=getattr(pool, "last_game_finish_p50", None),
                game_finish_max=getattr(pool, "last_game_finish_max", None))
            # Per-process accumulation watch: generation throughput
            # decayed 2x over iterations 2-5 and a process restart
            # restored it (2026-09-03); these say whether memory grows.
            gc_total, gc_gen2, gc_n2 = gc_meter.take()
            live = gc.get_objects()
            row.update(
                gpu_reserved_mb=(torch.cuda.memory_reserved() / 2**20
                                 if device.type == "cuda" else None),
                rss_mb=_rss_mb(), gc_seconds=gc_total,
                gc_gen2_seconds=gc_gen2, gc_gen2_count=gc_n2,
                live_objects=len(live))
            # Who holds the heap: the live heap grew ~0.5M objects and
            # RSS ~1.7 GB per iteration (az5); the top types name the
            # retainer.
            top = collections.Counter(type(o).__name__ for o in live).most_common(8)
            del live
            log.info(f"iter {it}: live objects by type: "
                     + ", ".join(f"{n}={c}" for n, c in top))
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
                # Value LEVEL in the mover frame: mean prediction vs
                # mean label. Search's act-vs-end_turn gap moves by
                # twice the level error (step-scale measurement,
                # 2026-09-03), so this is the number to watch.
                lvl = sample[:96]
                row.update(fresh_value_mean=sum(action_priors(base, e)[2]
                                                for e in lvl) / len(lvl),
                           fresh_label_mean=sum(float(e.z) for e in lvl) / len(lvl))
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
            # ---- ONE gradient step, backtracked on held-out games --
            t_tr = time.monotonic()
            train_exps, held_exps = split_holdout(kept, args.holdout_frac, rng)
            kl_states = (held_exps if len(held_exps) <= args.kl_states
                         else rng.sample(held_exps, args.kl_states))
            captured = {}

            def _take_step():
                with policy._lock:
                    policy._queue = list(train_exps)
                captured["stats"] = policy.train_step()
                return captured["stats"]

            res = backtracking_step(base, _take_step, train_exps, held_exps,
                                    kl_states,
                                    max_level_shift=(None if args.max_level_shift < 0
                                                     else args.max_level_shift),
                                    max_trials=args.step_trials,
                                    select=args.step_select)
            stats = captured["stats"]
            row.update(policy_loss=stats.policy_loss,
                       value_loss=stats.value_loss,
                       grad_norm=stats.grad_norm,
                       step_alpha=res.alpha, step_trials=res.trials,
                       held_before=res.held_before.get("total"),
                       held_after=res.held_after.get("total"),
                       held_delta_mean=res.held_after.get("delta_mean"),
                       held_delta_se=res.held_after.get("delta_se"),
                       step_kl_median=res.shift.get("kl_median"),
                       step_kl_mean=res.shift.get("kl_mean"),
                       step_tv_mean=res.shift.get("tv_mean"),
                       end_turn_prior=res.shift.get("end_turn_prior_mean"),
                       step_dv_mean=res.shift.get("dv_mean"),
                       step_dv_abs_mean=res.shift.get("dv_abs_mean"),
                       train_seconds=time.monotonic() - t_tr)
            # Search value centering for the NEXT iteration: the head's
            # mean value on this iteration's held-out states under the
            # weights just published (see MCTSConfig.value_center).
            if args.value_center and kl_states:
                center = statistics.fmean(action_priors(base, e)[2]
                                          for e in kl_states)
                # Tempo bonus: search sees the mover's positions as
                # `tempo_bonus` better than the head's level, i.e. it
                # prices handing the turn over. 0.44 = the seed's own
                # level on its self-play states (design_constants.md).
                pool.value_center = center - args.tempo_bonus
                row["value_center"] = pool.value_center
                row["value_level"] = center
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
                        args.probe_games, args.sims, dev_str,
                        value_center=float(row.get("value_center") or 0.0))
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
