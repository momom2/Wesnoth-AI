#!/usr/bin/env python3
"""Run N self-play sim games with DummyPolicy. Reports crash rate,
speed, and game-length stats.

Pre-flight check before sending self-play to the cluster: validates
the simulator + rollout pipeline + reward computation end-to-end on
just the rule-based DummyPolicy (no model, no training, no GPU).

What this catches:
  - Sim invariant bugs that fire mid-game (e.g. the known
    `hex (0,0) occupied by both 'u_' and 'u_'` crash).
  - Replay-load failures on the 2p-filtered pool.
  - Reward computation crashes on edge cases.
  - Memory leaks across hundreds of games.

What this does NOT catch:
  - Anything model-specific (predict_priors, policy gradient,
    encoder vocab overflow). Use `tools/sim_self_play.py` with a
    real checkpoint for that.

Usage:
    python tools/sim_dummy_smoke.py
    python tools/sim_dummy_smoke.py --n-games 200 --max-turns 40
    python tools/sim_dummy_smoke.py --n-games 500 --workers 4
    python tools/sim_dummy_smoke.py --csv logs/dummy_smoke.csv

Exit 0 if every game completed cleanly; exit 1 if ANY crashed.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Project root on sys.path so absolute imports work whether this
# is run as `python tools/sim_dummy_smoke.py` or `python -m
# tools.sim_dummy_smoke`.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tools"))

from dummy_policy import DummyPolicy
from rewards import WeightedReward
from sim_self_play import GameOutcome, _recruit_cost_lookup, play_one_game
from wesnoth_sim import PvPDefaults, WesnothSim


log = logging.getLogger("sim_dummy_smoke")


class _PolicyAdapter:
    """Wrap a stateless scripted policy (e.g. DummyPolicy) so it
    looks like a trainable policy to play_one_game.

    `play_one_game` calls `observe(label, side, reward, done)` per
    step and `drop_pending(label)` on crash; DummyPolicy has neither
    because it doesn't accumulate trajectories. The adapter no-ops
    both, plus `reset_game`.
    """

    def __init__(self, base):
        self.base = base

    def select_action(self, game_state, *, game_label="default"):
        return self.base.select_action(game_state, game_label=game_label)

    def observe(self, *_args, **_kwargs) -> None:
        pass

    def drop_pending(self, *_args, **_kwargs) -> None:
        pass

    def reset_game(self, *_args, **_kwargs) -> None:
        pass


def _pick_replay_pool(pool_dir: Path) -> List[Path]:
    """2p-filtered replay pool, mirroring sim_demo_game and
    sim_self_play. Falls back to all .json.gz when there's no
    index.jsonl."""
    import json as _json
    files = list(pool_dir.glob("*.json.gz"))
    if not files:
        return []
    idx = pool_dir / "index.jsonl"
    if idx.exists():
        keep: set = set()
        with idx.open() as f:
            for line in f:
                try:
                    e = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                if e.get("game_id", "").startswith("2p"):
                    keep.add(e.get("file", ""))
        if keep:
            files = [f for f in files if f.name in keep]
    return files


def _run_one_game(
    *,
    replay: Path,
    max_turns: int,
    pvp: PvPDefaults,
    cost_lookup: Dict[str, int],
    game_label: str,
) -> Tuple[Optional[GameOutcome], Optional[str]]:
    """Build a sim from `replay`, run one game with DummyPolicy.
    Returns (outcome, crash_message). Exactly one of the two is
    None: outcome on success, crash message on failure."""
    try:
        sim = WesnothSim.from_replay(replay, max_turns=max_turns,
                                     pvp_defaults=pvp)
    except Exception as e:
        return None, f"replay-load: {e}"
    policy = _PolicyAdapter(DummyPolicy())
    reward_fn = WeightedReward()
    try:
        outcome = play_one_game(
            sim, policy, reward_fn,
            game_label=game_label, cost_lookup=cost_lookup,
        )
        return outcome, None
    except Exception as e:
        # Distinguish sim invariant violations from other crashes;
        # invariant text starts with "sim invariant:" by convention
        # so we can taxonomize on it later.
        return None, f"sim: {e}"


def _summarize(
    outcomes: List[GameOutcome],
    crashes: List[Tuple[str, str]],   # (game_label, crash_msg)
    wall_seconds: float,
) -> int:
    n_total = len(outcomes) + len(crashes)
    n_ok    = len(outcomes)
    n_bad   = len(crashes)
    crash_rate = n_bad / n_total if n_total else 0.0

    print()
    print("=" * 72)
    print(f"DummyPolicy smoke: {n_total} games in {wall_seconds:.1f}s")
    print("=" * 72)
    print(f"  succeeded:   {n_ok:4d}  ({100*(1 - crash_rate):5.1f}%)")
    print(f"  crashed:     {n_bad:4d}  ({100*crash_rate:5.1f}%)")
    print()

    if outcomes:
        turns = sorted(o.turns for o in outcomes)
        actions = sorted(o.side1_actions + o.side2_actions
                         for o in outcomes)
        ended = Counter(o.ended_by for o in outcomes)
        winners = Counter(o.winner for o in outcomes)
        print(f"  turns       median={turns[len(turns)//2]:3d}  "
              f"mean={sum(turns)/len(turns):5.1f}  "
              f"max={max(turns)}")
        print(f"  actions     median={actions[len(actions)//2]:3d}  "
              f"mean={sum(actions)/len(actions):5.1f}  "
              f"max={max(actions)}")
        print(f"  ended_by    " + ", ".join(
            f"{k}={v}" for k, v in sorted(ended.items())))
        print(f"  winners     " + ", ".join(
            f"side{k}={v}" for k, v in sorted(winners.items())))
        rate = sum(actions) / wall_seconds if wall_seconds else 0
        print(f"  speed       {rate:7.1f} actions/s  "
              f"({n_ok / wall_seconds:.2f} games/s)")
        print()

    if crashes:
        # Group by crash-message prefix so we see how many distinct
        # failure modes there are.
        by_kind: Dict[str, int] = Counter()
        for _, msg in crashes:
            # First line, first 80 chars; collapse hex coordinates
            # so crashes that differ only by hex still group.
            first = msg.splitlines()[0][:120]
            by_kind[first] += 1
        print(f"  crash distribution ({len(by_kind)} distinct):")
        for msg, count in by_kind.most_common(10):
            print(f"    {count:4d}x  {msg}")
        if len(by_kind) > 10:
            print(f"    ... ({len(by_kind) - 10} more)")
        print()

    return 0 if n_bad == 0 else 1


def _write_csv(
    path: Path,
    outcomes: List[GameOutcome],
    crashes: List[Tuple[str, str]],
) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["game_label", "status", "winner", "ended_by",
                    "turns", "side1_actions", "side2_actions",
                    "side1_reward", "side2_reward", "crash_msg"])
        for o in outcomes:
            w.writerow([o.game_label, "ok", o.winner, o.ended_by,
                        o.turns, o.side1_actions, o.side2_actions,
                        o.side1_reward, o.side2_reward, ""])
        for label, msg in crashes:
            w.writerow([label, "crash", "", "", "", "", "", "", "", msg])


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay-pool", type=Path,
                    default=Path("replays_dataset"),
                    help="2p replay pool. Default: replays_dataset/.")
    ap.add_argument("--n-games", type=int, default=100,
                    help="Number of games to run. Default: 100.")
    ap.add_argument("--max-turns", type=int, default=40,
                    help="Per-game turn cap. Default: 40.")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel game workers. 1 = serial. "
                         "Each worker holds an independent sim + policy "
                         "instance; DummyPolicy is stateless so workers "
                         "don't share anything mutable.")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for replay sampling. Default: 0.")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Optional path to dump per-game CSV.")
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
    # Suppress per-game crash tracebacks from sim_self_play's logger;
    # we collect them into the summary instead.
    logging.getLogger("sim_self_play").setLevel(logging.CRITICAL)

    pool = _pick_replay_pool(args.replay_pool)
    if not pool:
        log.error(f"no 2p replays found under {args.replay_pool}")
        return 2
    log.info(f"replay pool: {len(pool)} 2p files in {args.replay_pool}")

    pvp = PvPDefaults(
        starting_gold=args.starting_gold,
        village_gold=args.village_gold,
        village_support=args.village_support,
        experience_modifier=args.exp_modifier,
    )
    cost_lookup = _recruit_cost_lookup()
    rng = random.Random(args.seed)

    outcomes: List[GameOutcome] = []
    crashes: List[Tuple[str, str]] = []
    t0 = time.perf_counter()

    if args.workers <= 1:
        for i in range(args.n_games):
            replay = rng.choice(pool)
            label = f"smoke{i:04d}"
            outcome, crash = _run_one_game(
                replay=replay, max_turns=args.max_turns,
                pvp=pvp, cost_lookup=cost_lookup, game_label=label,
            )
            if outcome is not None:
                outcomes.append(outcome)
            else:
                crashes.append((label, crash or "unknown"))
            if (i + 1) % max(1, args.n_games // 10) == 0:
                done = i + 1
                rate = done / (time.perf_counter() - t0)
                log.info(f"  {done}/{args.n_games} games "
                         f"({rate:.1f} games/s)")
    else:
        # Workers each pull from a shared atomic counter, identical
        # pattern to sim_self_play._worker_loop. DummyPolicy is
        # stateless so we can share it across threads safely.
        import threading
        shared = {
            "lock": threading.Lock(),
            "next_idx": 0,
        }

        def _worker(worker_rng: random.Random):
            while True:
                with shared["lock"]:
                    if shared["next_idx"] >= args.n_games:
                        return
                    i = shared["next_idx"]
                    shared["next_idx"] += 1
                replay = worker_rng.choice(pool)
                label = f"smoke{i:04d}"
                outcome, crash = _run_one_game(
                    replay=replay, max_turns=args.max_turns,
                    pvp=pvp, cost_lookup=cost_lookup, game_label=label,
                )
                with shared["lock"]:
                    if outcome is not None:
                        outcomes.append(outcome)
                    else:
                        crashes.append((label, crash or "unknown"))

        threads = []
        for w in range(args.workers):
            wrng = random.Random(rng.randint(0, 2**32 - 1))
            t = threading.Thread(target=_worker, args=(wrng,),
                                 daemon=True, name=f"smoke-w{w}")
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

    wall = time.perf_counter() - t0
    rc = _summarize(outcomes, crashes, wall_seconds=wall)
    if args.csv is not None:
        _write_csv(args.csv, outcomes, crashes)
        log.info(f"wrote {args.csv}")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
