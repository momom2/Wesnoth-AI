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
  - One game per loop: fresh random setup (absolute mix ratios from
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
        --mini-ratio 0.2 --ladder-ratio 0.8 --max-turns 200 \
        --moves-left-utility 0.2 --seed 1234
"""

from __future__ import annotations

import argparse
import json
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
        aux_value_bonus=getattr(args, "aux_value_bonus", 0.0),
    )
    return MCTSPolicy(
        base, cfg,
        train_draw_tiebreak=getattr(args, "train_draw_tiebreak",
                                    False)), base


def _ctl_wants_exit(ctl_path: Path, current_device: str) -> bool:
    """True when the learner's device-ctl file exists and names a
    device other than the one this worker runs on (the graceful-
    demotion signal; see SpoolWorkers.demote_one_cuda_worker).
    Unreadable/garbage ctl content is ignored -- never kill a
    healthy worker on a torn read."""
    try:
        if not ctl_path.exists():
            return False
        want = ctl_path.read_text(encoding="ascii").strip()
    except OSError:
        return False
    return want in ("cpu", "cuda") and want != current_device


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker-id", type=int, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--spool-dir", type=Path, required=True)
    ap.add_argument("--mcts-sims", type=int, default=32)
    # Mix ratios: ABSOLUTE fractions of all games; the five must sum
    # to 1 (roll_mix validates). The parent passes all of them.
    ap.add_argument("--mini-ratio", type=float, default=0.0)
    ap.add_argument("--drill-ratio", type=float, default=0.0)
    ap.add_argument("--max-turns", type=int, default=200)
    ap.add_argument("--draw-tiebreak-cap", type=float, default=0.3)
    ap.add_argument("--train-draw-tiebreak", action="store_true",
                    help="LEGACY: material-tiebreak z on drawn games'"
                         " training labels (see sim_self_play).")
    ap.add_argument("--moves-left-utility", type=float, default=0.0)
    ap.add_argument("--aux-value-bonus", type=float, default=0.0)
    ap.add_argument("--fogless-ratio", type=float, default=0.0)
    ap.add_argument("--midgame-ratio", type=float, default=0.0)
    ap.add_argument("--ladder-ratio", type=float, default=1.0)
    ap.add_argument("--midgame-dataset", type=Path,
                    default=Path("replays_dataset"))
    ap.add_argument("--validate-export-every", type=int, default=100)
    ap.add_argument("--validate-export-dir", type=Path,
                    default=Path("training/validate_exports"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=2)
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"),
                    default="auto",
                    help="Forward device for this worker. The learner "
                         "assigns it per-worker from the VRAM budget "
                         "(sim_self_play._assign_spool_devices): each "
                         "cuda worker costs ~600MB of the card "
                         "(CUDA context + model), and 56 auto-cuda "
                         "workers starved the trainer's backward into "
                         "an OOM crash-loop on 2026-07-18. 'auto' "
                         "keeps the legacy cuda-if-available behavior "
                         "with the init-time OOM fallback to cpu.")
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args(argv[1:])
    if int(args.validate_export_every) > 0:
        import tools.sim_self_play as _ssp
        from tools.validation_exports import ValidationExporter
        # play_one_game (imported from sim_self_play) reads the
        # module global -- set it in THIS worker process too.
        _ssp.VALIDATION_EXPORTER = ValidationExporter(
            args.validate_export_dir,
            every=args.validate_export_every)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=f"%(asctime)s w{args.worker_id} %(levelname)s %(message)s")

    import torch
    torch.set_num_threads(max(1, args.torch_threads))

    # Heavy imports AFTER thread cap.
    from tools.actor_pool import _zero_reward, _set_fd_safe_sharing
    from tools.scenario_pool import (build_scenario_gamestate,
                                     random_setup, roll_mix)
    from tools.sim_self_play import _recruit_cost_lookup, play_one_game
    from wesnoth_sim import PvPDefaults, WesnothSim

    _set_fd_safe_sharing()

    device = None
    if args.device == "cuda":
        device = torch.device("cuda")     # explicit: fail loud if absent
    elif args.device == "auto" and torch.cuda.is_available():
        device = torch.device("cuda")
    # args.device == "cpu": stay None -- zero VRAM footprint; the
    # learner keeps the whole card for its backward pass.
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
    hb_decisions = 0
    hb_started = time.time()
    ctl_path = args.spool_dir / "ctl" / f"w{args.worker_id}.device"
    while True:
        # Graceful demotion seam (2026-07-20): the learner flips
        # spool/ctl/w<id>.device when the VRAM headroom guard wants
        # this slot off the card. Checked BETWEEN games, so nothing
        # in flight is lost -- exit cleanly and let ensure_alive
        # respawn us on the slot's new device.
        if _ctl_wants_exit(ctl_path,
                           "cuda" if device is not None else "cpu"):
            log.info("device ctl disagrees with our device; exiting "
                     "for respawn")
            return 0
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

        setup = None
        midgame_cut = None
        cat = roll_mix(rng, midgame=args.midgame_ratio,
                       mini=args.mini_ratio, drill=args.drill_ratio,
                       fogless=args.fogless_ratio,
                       ladder=args.ladder_ratio)
        if cat == "midgame":
            from tools.midgame_starts import sample_midgame_start
            mg = sample_midgame_start(rng, args.midgame_dataset)
            if mg is not None:
                setup = ("__midgame__",) + mg
            else:
                cat = "ladder"  # degraded sample -> regular game
        if setup is None:
            setup = random_setup(rng, category=cat)
        if isinstance(setup, tuple) and setup[0] == "__midgame__":
            _, gs, scen_id, midgame_cut, begin_side, mg_prov = setup
            sim = WesnothSim(gs, scenario_id=scen_id,
                             max_turns=args.max_turns,
                             apply_scenario_events=False,
                             begin_side=begin_side)
            sim._midgame_start = True
            sim._midgame_provenance = mg_prov
        else:
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

        # Heartbeat: cumulative per-worker throughput + device tag,
        # overwritten atomically after each game. Consumed by
        # tools/profile_worker_split.py to measure cuda-vs-cpu
        # per-worker rates from the LIVE fleet (and handy for
        # at-a-glance worker liveness). Losing it is harmless.
        try:
            hb_decisions += int(payload["n_decisions"])
            stats_dir = args.spool_dir / "stats"
            stats_dir.mkdir(parents=True, exist_ok=True)
            hb = {
                "worker": args.worker_id,
                "device": (device.type if device is not None else "cpu"),
                "games": n,
                "decisions": hb_decisions,
                "started": hb_started,
                "updated": time.time(),
            }
            hb_tmp = stats_dir / f".tmp_w{args.worker_id}.json"
            hb_tmp.write_text(json.dumps(hb), encoding="utf-8")
            os.replace(hb_tmp, stats_dir / f"w{args.worker_id}.json")
        except OSError as e:
            log.warning(f"heartbeat write failed: {e}")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
