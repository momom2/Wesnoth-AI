"""One-game demo via the simulator + bz2 replay export.

Replaces the old "watch one Wesnoth window" path. The sim is headless
-- there's no live game window -- but we play one full game with the
loaded checkpoint vs itself, then export a Wesnoth-loadable .bz2
replay so the user can open it in Wesnoth's replay viewer (File ->
Load Game -> pick the .bz2). That's the same end goal as the old
--display flow (see one model game, animations on, no training)
without spinning up the broken Wesnoth subprocess.

Why a separate script (vs adding `--games 1` + a flag to
sim_self_play): sim_self_play's loop is built around train_step.
Doing inference-only with replay export is enough of a
different shape -- no reward bookkeeping, no gradient step, no
multi-iteration loop -- that splitting it keeps each tool's job
crisp.

CLI:

    python tools/sim_demo_game.py
        --checkpoint training/checkpoints/supervised_epoch3.pt
        --out demo.bz2
        --max-turns 40

Auto-pick checkpoint: if --checkpoint is omitted, picks the most
recently modified `supervised*.pt` in `training/checkpoints/`. Same
behavior the old run_self_play.ps1 had so the GUI's Display button
keeps its zero-config feel.

The exported .bz2 lands wherever --out points (default:
`logs/sim_demo_<timestamp>.bz2`). The script logs the full path so
the GUI can print it and the user can double-click straight from
the log line.
"""

from __future__ import annotations

import argparse
import copy
import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional

# Make project root importable when run as a script.
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

from sim_to_replay import export_replay, find_source_bz2
from transformer_policy import TransformerPolicy
from wesnoth_sim import PvPDefaults, WesnothSim


log = logging.getLogger("sim_demo_game")


def _autoselect_checkpoint(root: Path) -> Optional[Path]:
    """Return the freshest `supervised*.pt` (or any .pt) in `root` by
    modification time. Mirrors the .ps1 auto-pick logic so users
    coming from `cluster/gui.pyw` get the same convenience."""
    if not root.is_dir():
        return None
    pts = sorted(root.glob("supervised*.pt"),
                 key=lambda p: p.stat().st_mtime, reverse=True)
    if pts:
        return pts[0]
    pts = sorted(root.glob("*.pt"),
                 key=lambda p: p.stat().st_mtime, reverse=True)
    return pts[0] if pts else None


def _pick_replay_seed(pool: Path, rng: random.Random) -> Optional[Path]:
    """Random `replays_dataset/*.json.gz` to bootstrap the sim from.

    Filters to 2p game_ids via index.jsonl when present (mirrors
    `sim_self_play._gather_replay_pool`). Without the filter, the
    demo can land on a campaign-scenario replay whose runtime state
    used Lua / WML paths the sim doesn't model -- e.g. a
    `[modify_unit]` that pushed hp past max_hp via `violate_maximum=yes`,
    which trips `_assert_invariants` on the first sim step.
    """
    if not pool.is_dir():
        return None
    files = list(pool.glob("*.json.gz"))
    if not files:
        return None
    # 2p filter (matches sim_self_play._gather_replay_pool).
    idx_path = pool / "index.jsonl"
    if idx_path.exists():
        import json as _json
        keep_names: set = set()
        with idx_path.open() as f:
            for line in f:
                try:
                    e = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                if e.get("game_id", "").startswith("2p"):
                    keep_names.add(e.get("file", ""))
        if keep_names:
            files = [f for f in files if f.name in keep_names]
    if not files:
        return None
    return rng.choice(files)


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Model checkpoint .pt. Default: freshest "
                         "supervised*.pt under training/checkpoints/.")
    ap.add_argument("--replay-pool", type=Path,
                    default=Path("replays_dataset"),
                    help="Pool of .json.gz replays to seed the initial "
                         "state from. Default: replays_dataset/.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output .bz2 path. Default: "
                         "logs/sim_demo_<UTC>.bz2.")
    ap.add_argument("--max-turns", type=int, default=40,
                    help="Per-game turn cap.")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed for replay sampling. Default: time-based.")
    ap.add_argument("--starting-gold", type=int, default=100)
    ap.add_argument("--village-gold", type=int, default=2)
    ap.add_argument("--village-support", type=int, default=1)
    ap.add_argument("--exp-modifier", type=int, default=70)
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING"])
    args = ap.parse_args(argv[1:])

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # 1. Pick checkpoint
    ckpt = args.checkpoint
    if ckpt is None:
        ckpt = _autoselect_checkpoint(Path("training/checkpoints"))
        if ckpt is None:
            log.error(
                "no checkpoint passed and none found under "
                "training/checkpoints/. Pull one from the cluster first "
                "(cluster/pull_checkpoint.ps1) or pass --checkpoint.")
            return 2
        log.info(f"auto-picked checkpoint: {ckpt}")
    if not ckpt.exists():
        log.error(f"checkpoint not found: {ckpt}")
        return 2

    # 2. Pick a seed replay to bootstrap from
    rng = random.Random(args.seed if args.seed is not None
                        else int(time.time()))
    seed_replay = _pick_replay_seed(args.replay_pool, rng)
    if seed_replay is None:
        log.error(f"no .json.gz replays under {args.replay_pool}")
        return 2
    log.info(f"seed replay: {seed_replay.name}")

    # 3. Locate the matching .bz2 (sim_to_replay needs the source bz2
    #    to splice the demo replay's commands into the original
    #    [scenario] block).
    src_bz2 = find_source_bz2(seed_replay)
    if src_bz2 is None:
        log.error(
            f"could not auto-locate the .bz2 matching {seed_replay}. "
            f"Pick a different --replay-pool entry or check that "
            f"replays_raw/ has the matching source.")
        return 2
    log.info(f"source bz2:  {src_bz2}")

    # 4. Build the sim with PvP defaults so the playback uses standard
    #    2p economy / experience rules regardless of the source replay's
    #    host-customized settings.
    pvp = PvPDefaults(
        starting_gold=args.starting_gold,
        village_gold=args.village_gold,
        village_support=args.village_support,
        experience_modifier=args.exp_modifier,
    )
    sim = WesnothSim.from_replay(
        seed_replay, max_turns=args.max_turns, pvp_defaults=pvp,
    )

    # 5. Load the policy + drive the game. We use TransformerPolicy
    #    with training off (no gradient updates, no replay buffer
    #    growth) so this is pure inference.
    policy = TransformerPolicy()
    policy.load_checkpoint(ckpt)
    log.info("running one game (this is headless -- progress in stderr)...")
    t0 = time.perf_counter()
    game_label = "demo"
    while not sim.done:
        # Deepcopy the state before each select_action: the policy
        # stores the GameState reference in a Transition (it's a
        # trainable policy class even though we discard pending here),
        # and `sim.step` mutates `sim.gs` in place. Without the copy
        # the stored state would diverge from what the trainer's
        # reforward sees later. See the contract in
        # `transformer_policy.select_action`'s docstring.
        pre_state = copy.deepcopy(sim.gs)
        action = policy.select_action(pre_state, game_label=game_label)
        sim.step(action)
    # Drop any pending trajectory transitions so policy state stays
    # clean. observe(done=True) would normally close them; we don't
    # want to and we have no reward to attach.
    policy.drop_pending(game_label)
    dt = time.perf_counter() - t0

    log.info(
        f"game over in {dt:.1f}s: winner={sim.winner} "
        f"turns={sim.gs.global_info.turn_number} "
        f"ended_by={sim.ended_by} "
        f"actions={len(sim.command_history)}")

    # 6. Export the replay
    if args.out is None:
        Path("logs").mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = Path("logs") / f"sim_demo_{ts}.bz2"
    else:
        out_path = args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)

    export_replay(sim, source_bz2=src_bz2, out_path=out_path,
                  pvp_defaults=pvp)
    log.info(f"wrote {out_path}")
    log.info(
        f"To watch: open {out_path} in Wesnoth (File -> Load Game -> "
        f"pick this file).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
