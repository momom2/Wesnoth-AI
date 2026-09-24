#!/usr/bin/env python3
"""Minimal self-play loop -- docs/archive/az_minimal_spec.md, nothing else.

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

`--stream` (2026-09-18, tools/actor_stream.py): the actors never wait
at the iteration barrier. An "iteration" is then a WINDOW of
--games-per-iter completed games, collected while every actor keeps
playing, and the step's weights publish into the running servers. The
CSV keeps its columns; gen_seconds is the window's span, game_finish_*
are per-game durations, and four columns describe what the barrier
never had: straddle_mean / straddle_max / straddled_share (weight
publications a game lived through) and step_leaves_per_s (what the
in-process server served while the learner stepped). Opt-in until a
learner that improves on the prior has been run both ways.
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
    "value_level_ref",
    # serve processes (--serve-processes > 1): the weights version the
    # serve processes acknowledged after the step
    "server_weights_version",
    # --stream (tools/actor_stream.py): publications a game of the
    # window lived through, the window's median game duration, the
    # in-process serve rate during the learner's step, a window cut
    # short by the timeout
    "straddle_mean", "straddle_max", "straddled_share", "game_seconds_p50",
    "step_leaves_per_s", "window_timed_out",
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
        # elo_eval_game writes outcome_a as "win" / "loss" / "draw"
        # (or "timeout..." for unfinished games, which do not count).
        if r == "win":
            w += 1
        elif r == "loss":
            l_ += 1
        elif r == "draw":
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
    if sims <= 0:
        # The reference player is raw:t0 (argmax); the legacy sampler
        # ("raw") is 400 Elo weaker and was what every earlier pin
        # compared (docs/raw_argmax_control_20260904.md).
        cmd += ["--raw-temperature-a", "0", "--raw-temperature-b", "0"]
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


def _game_records_dir(args) -> Optional[Path]:
    """Where this run's actors record their games: a per-run
    subdirectory of --game-record-dir (WORKDIR/game_records unless
    given), or None when recording is disabled."""
    root = args.game_record_dir
    if root is not None and str(root) in ("", "."):
        return None
    from tools.validation_exports import run_tag
    return (root if root is not None else args.workdir / "game_records") / run_tag()


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed-checkpoint", type=Path, required=True)
    ap.add_argument("--campaign", type=Path, required=True,
                    help="rolling checkpoint (resumed if it exists)")
    ap.add_argument("--workdir", type=Path, default=Path("/workspace"))
    ap.add_argument("--game-record-dir", type=Path, default=None,
                    help="Every game the actors finish is recorded whole here, "
                         "one file per actor under a per-run subdirectory "
                         "(tools/game_record.py); default WORKDIR/game_records. "
                         "Pass an empty string to disable.")
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--games-per-iter", type=int, default=24)
    ap.add_argument("--actors", type=int, default=0,
                    help="Actor processes; 0 (the default) means as many as the "
                         "box and the iteration allow. An actor is BLOCKED on the "
                         "inference server for nine tenths of its cycle (it holds "
                         "one request in flight), so the count buys in-flight "
                         "leaves, not CPU. Two ceilings apply: --games-per-iter, "
                         "since an iteration hands out one game per actor and the "
                         "surplus idles; and the container's pids limit, which "
                         "counts THREADS and which host_resources.max_actors now "
                         "READS rather than guesses (exceeding it produces 0 "
                         "leaves/s, as a 2026-09-04 run at 38 actors did). "
                         "Measured 2026-09-13 on a 24-core 4090 box: 19 -> 665, "
                         "32 -> 914, 48 -> 1,006, 64 -> 1,116 leaf evaluations/s "
                         "against a server saturating near 1,500 "
                         "(docs/box_specs.md \"Actors buy in-flight leaves\"). "
                         "The curve was still rising at 64, so --games-per-iter "
                         "is the knob worth raising on a bigger box.")
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
    ap.add_argument("--mini-ratio", type=float, default=0.0,
                    help="Share of games on the mini maps instead of the Ladder "
                         "pool (the smoke test's knob; a campaign leaves it 0).")
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
    ap.add_argument("--server-priors", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Server-side priors in the actor pool "
                         "(plan 1.3; 141 -> 172 leaves/s alone, 320 "
                         "with bf16, docs/box_specs.md).")
    ap.add_argument("--infer-bf16", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="bf16 autocast on the pool's inference server "
                         "(cuda only; the learner's own probes stay in "
                         "the model's precision).")
    ap.add_argument("--train-bf16", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="bf16 autocast around the trainer's forward "
                         "(TrainerConfig.train_autocast_bf16; cuda only). "
                         "The backward replays the forward's dtypes, so "
                         "the trunk's matmuls run bf16 both ways; the "
                         "losses, the master weights and AdamW stay fp32. "
                         "Batch 16 on a 4090: 45.0 against 57.1 ms per "
                         "experience, loss within 3e-4 of fp32, gradient "
                         "cosine 0.9994, norm within 0.3%% on one batch of "
                         "64 (docs/box_specs.md 'Training path cost "
                         "(2026-09-05)'). Default on.")
    ap.add_argument("--tf32", action="store_true",
                    help="The learner's fp32 matmuls on the tensor cores (TF32), the "
                         "in-process serving untouched. A recipe change, so off by "
                         "default and one factor of its own.")
    ap.add_argument("--packed-trunk", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Serve the pool's forwards through the packed varlen "
                         "trunk (flash attention, no padding; cuda + bf16 only, "
                         "so the learner's fp32 probes keep the padded path). "
                         "2026-09-05: 652 -> 833 leaves/s saturated.")
    ap.add_argument("--compile-packed", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Compile the packed layer loop (one inductor graph); "
                         "measured separately, off until its row is in.")
    ap.add_argument("--serve-processes", type=int, default=1,
                    help="Serving processes for the pool (1 = the learner's "
                         "own serve threads). Each extra process holds a model "
                         "copy and receives the weights after every step; the "
                         "serve threads' host work (~33 ms per batch) is the "
                         "measured ceiling (docs/box_specs.md).")
    ap.add_argument("--graphed-serve", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Every server replays its priors batches from per-bucket "
                         "CUDA graphs (wesnoth_ai/graphed_serve.py): one launch per "
                         "batch instead of the ~150 the host pays today. cuda + bf16 "
                         "+ the packed trunk; off until its box row is in.")
    ap.add_argument("--stream", action=argparse.BooleanOptionalAction, default=False,
                    help="Continuous generation (tools/actor_stream.py): no iteration "
                         "barrier; each step takes the next --games-per-iter completed "
                         "games while every actor keeps playing, and publishes into the "
                         "running servers. Games straddle publications (recorded per "
                         "window). Off until measured against the barrier both ways.")
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
    from wesnoth_ai.trainer import STEP_MCTS_STAGES
    from signal_profiler.target_amplitude import target_amplitude

    device = (torch.device("cuda")
              if args.device == "cuda" and torch.cuda.is_available()
              else torch.device("cpu"))
    # Learner + inference server share this process; the actors do the
    # CPU work. Cap torch's intra-op pool (default: every hardware
    # thread of the host, far beyond the cgroup quota). See bench_pool.
    torch.set_num_threads(4)
    dev_str = "cuda" if device.type == "cuda" else "cpu"
    ckpt_in = args.campaign if args.campaign.exists() else args.seed_checkpoint
    base = _load_policy(ckpt_in, device, label="az")
    if device.type == "cuda" and args.packed_trunk:
        inference_base = getattr(base, "_inference_base", base._inference_model)
        inference_base.infer_packed_trunk = True
        if args.compile_packed:
            inference_base.infer_compile_packed = True
            inference_base.warmup_packed_compile()
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
    # Sixteen experiences per forward+backward: docs/box_specs.md
    # "Training path cost (2026-09-05)" -- fp32 batch 16 matched batch
    # 1 exactly (gradient within 1e-4) on one batch of 64, at 57 vs 68
    # ms per experience before the batched policy loss.
    cfg.train_batch_size = 16
    cfg.train_autocast_bf16 = bool(args.train_bf16)
    log.info(f"loaded {ckpt_in.name} decision_step={base._decision_step} "
             f"| value_loss={cfg.value_loss_form} c={cfg.value_coef} "
             f"lr={cfg.learning_rate} clip={cfg.grad_clip} "
             f"batch={cfg.train_batch_size} "
             f"train_bf16={cfg.train_autocast_bf16}")

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
    mini_ratio = min(1.0, max(0.0, float(args.mini_ratio)))
    scenario_opts = dict(forced_faction=None, mini_maps=(True if mini_ratio > 0 else None),
                         mini_ratio=mini_ratio, fogless_ratio=0.0,
                         ladder_ratio=1.0 - mini_ratio, midgame_ratio=0.0,
                         midgame_dataset=None)
    # Two ceilings, both real, applied in order.
    #
    # 1. An iteration posts one ticket per game and then one end
    #    marker per actor, so an actor beyond the game count takes an
    #    end marker immediately and idles for the whole iteration.
    # 2. A container's pids controller counts THREADS, and an actor is
    #    several. Exceeding it does not degrade, it produces ZERO
    #    leaves per second -- it cost a whole rental on 2026-09-04,
    #    and holding the default low "to be safe" has cost throughput
    #    on every box since. host_resources reads the actual limit.
    #
    # --actors 0 (the default) means "as many as the box and the
    # iteration allow", so raising --games-per-iter on a bigger box
    # picks up the actors automatically. The measured curve was still
    # rising at 64 (docs/box_specs.md "Actors buy in-flight leaves"),
    # so the game count is the binding knob, not this one.
    from tools.host_resources import (PIDS_PER_ACTOR_ESTIMATE, max_actors,
                                      pids_current, pids_per_actor)
    if args.actors <= 0:
        n_actors = args.games_per_iter
        log.info("--actors auto: %d, matching --games-per-iter", n_actors)
    else:
        n_actors = args.actors
        if n_actors > args.games_per_iter and not args.stream:
            log.warning("--actors %d exceeds --games-per-iter %d; running %d actors "
                        "(the surplus would idle). Raise --games-per-iter to use them.",
                        n_actors, args.games_per_iter, args.games_per_iter)
            n_actors = args.games_per_iter
    if args.stream:
        # A stream keeps every actor playing whatever the window size;
        # actors over games per window means a game straddles about
        # that many publications on average.
        log.info("--stream: %d actors per %d-game window, about %.2f publications "
                 "per game", n_actors, args.games_per_iter,
                 n_actors / max(1, args.games_per_iter))
    fits, why = max_actors(n_actors)
    log.info("actor budget: %s", why)
    if fits < n_actors:
        log.warning("clamping actors %d -> %d: %s", n_actors, fits, why)
        n_actors = fits
    if n_actors < 1:
        # A pool of zero actors posts tickets nobody takes and the
        # iteration blocks until its timeout. Fail at the argument, not
        # an hour into a rental.
        raise SystemExit(
            f"--games-per-iter {args.games_per_iter} leaves no actors to run "
            f"(--actors {args.actors}); both must be at least 1")
    _pids_before = pids_current()
    pool = ActorPool(policy, n_actors, mcts_cfg, turn_cfg=None,
                     pt_cfg=None, gbc_labels=False, train_kwargs={},
                     scenario_opts=scenario_opts, max_turns=args.max_turns,
                     max_turns_min=args.max_turns,
                     pvp_defaults=PvPDefaults(), device=device,
                     # 64 leaves per serve batch (four actor requests
                     # coalesced): 1.34x the saturated rate of the 16-leaf
                     # cap the az legs used, measured 2026-09-21 in two
                     # interleaved pairs on a quiet 4090 host
                     # (docs/serve_batch_prereg_20260920.md).
                     max_batch=64, log_level=logging.WARNING,
                     iteration_timeout=args.iteration_timeout,
                     drain_grace=300.0,
                     server_priors=bool(args.server_priors),
                     infer_bf16=bool(args.infer_bf16 and device.type == "cuda"),
                     # 2026-09-05: the padded encode was 8.5 ms of the ~36 ms
                     # host work per batch; one pinned copy straight into
                     # the packed layout is 3.7 (docs/box_specs.md).
                     packed_embed=bool(args.packed_trunk and device.type == "cuda"),
                     serve_processes=max(1, int(args.serve_processes)),
                     graphed_serve=bool(args.graphed_serve),
                     game_records_dir=_game_records_dir(args))
    pool.start()
    # Calibration happens after the FIRST ITERATION, not here.
    # `pool.start()` is a loop of `p.start()` with no barrier, so at
    # this point a spawned actor is one task that has not yet imported
    # torch; sampling now reads about 1 task per actor and would tell
    # the reader to lower an estimate that exists to keep a box alive.
    # The only per-actor figure this repo has ever measured is 38
    # actors exhausting a 4,352 pids limit -- about 114 tasks each --
    # from before the OMP=1 cap, so the true number today is genuinely
    # unknown and worth measuring properly.
    _pids_calibrated = False

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
    # Fixed reference states for the search value center: the level
    # measured on each iteration's own held-out states moved +-0.3
    # between batches (state mix), as much as the tempo bonus itself
    # (az5 iteration 4: K 6 from a stale center). The first held-out
    # sample of the run is kept and the center is read on it every
    # iteration, so only weight changes move it.
    ref_states = None
    k_low = 0
    pins_done = 0
    stream = None
    if args.stream:
        stream = pool.stream(rng.randint(0, 2**31 - 1), tag=args.start_iter)
        stream.start()
    try:
        for it in range(args.start_iter, args.iterations):
            t_it = time.monotonic()
            row: Dict = {"iter": it, "decision_step": base._decision_step}

            # ---- generation --------------------------------------
            if stream is not None:
                window = stream.collect(args.games_per_iter,
                                        timeout=args.iteration_timeout)
                outcomes, exps = window.outcomes, window.experiences
                row.update(straddle_mean=window.straddle_mean,
                           straddle_max=window.straddle_max,
                           straddled_share=window.straddled_share,
                           game_seconds_p50=window.game_seconds_p50,
                           window_timed_out=int(window.timed_out))
            else:
                outcomes, exps = pool.run_iteration(it, args.games_per_iter,
                                                    rng.randint(0, 2**31 - 1))
            if not _pids_calibrated:
                _pids_calibrated = True
                _per_actor = (pids_per_actor(n_actors, _pids_before)
                              if _pids_before else None)
                if _per_actor is not None:
                    log.info(
                        "pids: %d actors cost %.1f tasks each after one "
                        "iteration (budgeting estimate %d). If these differ, "
                        "update host_resources.PIDS_PER_ACTOR_ESTIMATE -- it "
                        "is what keeps the actor count under the limit that "
                        "produces zero leaves/s when exceeded.",
                        n_actors, _per_actor, PIDS_PER_ACTOR_ESTIMATE)
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
            # Every step_mcts call of the block (the held-out probes
            # and the step itself) adds its stage seconds here.
            train_stages = base._trainer.stage_timings = {}
            t_tr = time.monotonic()
            leaves_before_step = stream.leaves_served() if stream is not None else 0
            train_exps, held_exps = split_holdout(kept, args.holdout_frac, rng)
            kl_states = (held_exps if len(held_exps) <= args.kl_states
                         else rng.sample(held_exps, args.kl_states))
            captured = {}

            def _take_step():
                with policy._lock:
                    policy._queue = list(train_exps)
                # TF32 (when asked for) for the learner's matmuls only:
                # this process also serves the actors' inference, and
                # that must keep the numerics the spawned serve
                # processes use.
                from wesnoth_ai.train_perf import tf32_training
                with tf32_training(bool(getattr(args, "tf32", False))):
                    captured["stats"] = policy.train_step()
                return captured["stats"]

            res = backtracking_step(base, _take_step, train_exps, held_exps,
                                    kl_states,
                                    max_level_shift=(None if args.max_level_shift < 0
                                                     else args.max_level_shift),
                                    max_trials=args.step_trials,
                                    select=args.step_select)
            stats = captured["stats"]
            if args.serve_processes > 1 and stream is None:
                # The step published new inference weights (through the
                # backtracking's final publish); the serve processes must
                # hold them before the next iteration's PLAY.
                row["server_weights_version"] = pool.sync_servers()
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
            base._trainer.stage_timings = None
            # Search value centering for the NEXT iteration: the head's
            # mean value on this iteration's held-out states under the
            # weights just published (see MCTSConfig.value_center).
            if args.value_center and kl_states:
                if ref_states is None:
                    ref_states = list(kl_states)
                center_batch = statistics.fmean(action_priors(base, e)[2]
                                                for e in kl_states)
                center = statistics.fmean(action_priors(base, e)[2]
                                          for e in ref_states)
                # Tempo bonus: search sees the mover's positions as
                # `tempo_bonus` better than the head's level, i.e. it
                # prices handing the turn over. 0.44 = the seed's own
                # level on its self-play states (design_constants.md).
                pool.value_center = center - args.tempo_bonus
                row["value_center"] = pool.value_center
                row["value_level"] = center_batch
                row["value_level_ref"] = center
            if stream is not None:
                # The servers take the step's weights now, the actors
                # the new center and anneal counter; the publication
                # is dated for the straddle count of the games in
                # flight.
                row["server_weights_version"] = stream.publish(
                    value_center=pool.value_center,
                    decision_step=int(base._decision_step))
                step_s = row["train_seconds"]
                row["step_leaves_per_s"] = ((stream.leaves_served() - leaves_before_step)
                                            / step_s if step_s else None)
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
            # "other" = the inference snapshot, the policy-shift
            # forwards and the block's own Python.
            other = row["train_seconds"] - sum(train_stages.values())
            log.info("train path (s): "
                     + " ".join(f"{s} {train_stages.get(s, 0.0):.1f}"
                                for s in STEP_MCTS_STAGES)
                     + f" | other {other:.1f}")
    finally:
        if stream is not None:
            stream.stop(grace=300.0)
        pool.shutdown()
        fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
