#!/usr/bin/env python3
"""Corpus proof for the reachability mask: every HUMAN action must be
inside the legality mask.

Rationale (2026-07-17 mask/sim contract): the mask defines what the
policy may ATTEMPT -- exactly what a human could order through the
same fog. Humans, by construction, only ordered things their client
allowed, so every recorded human command must land on a mask-legal
(actor, type, target) triple. Any miss is a mask bug (too tight = the
model is forbidden something humans can do) or a planner bug (our
reachability diverges from Wesnoth's).

BC-label caveat: pairs whose actor slot resolves but whose target is
mask-illegal would train the policy toward actions it can never
sample -- so this doubles as a training-data health check.

Usage:
    python tools/check_mask_coverage.py [--games 100] [--seed 0]
        [--dataset replays_dataset]
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

log = logging.getLogger("check_mask_coverage")


def main(argv) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path,
                    default=Path("replays_dataset"))
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose-misses", type=int, default=10,
                    help="Print detail for the first N misses.")
    args = ap.parse_args(argv[1:])

    from encoder import GameStateEncoder
    from model import UnitActionType
    from action_sampler import _build_legality_masks
    from tools.replay_dataset import iter_replay_pairs

    files = sorted(args.dataset.glob("*.json.gz"))
    if not files:
        log.error(f"no replays under {args.dataset}")
        return 2
    rng = random.Random(args.seed)
    rng.shuffle(files)
    files = files[:args.games]

    enc = GameStateEncoder()
    stats = Counter()
    misses_shown = 0

    for gz in files:
        try:
            for gs, ai in iter_replay_pairs(gz):
                stats[f"pairs_{ai.action_type}"] += 1
                if ai.actor_idx is None:
                    continue
                encoded = enc.encode(gs)
                masks = _build_legality_masks(encoded, gs)
                A = masks.actor_valid.size(-1)
                ok = True
                why = ""
                if ai.actor_idx >= A or not bool(
                        masks.actor_valid[0, ai.actor_idx].item()):
                    ok, why = False, "actor_invalid"
                elif ai.action_type == "move":
                    row = masks.target_valid_move[ai.actor_idx]
                    if (ai.target_idx is None
                            or not bool(row[ai.target_idx].item())):
                        ok, why = False, "move_target_invalid"
                elif ai.action_type == "attack":
                    row = masks.target_valid_attack[ai.actor_idx]
                    if (ai.target_idx is None
                            or not bool(row[ai.target_idx].item())):
                        ok, why = False, "attack_target_invalid"
                elif ai.action_type in ("recruit", "recall"):
                    row = masks.target_valid[ai.actor_idx]
                    if (ai.target_idx is None
                            or not bool(row[ai.target_idx].item())):
                        ok, why = False, "recruit_target_invalid"
                if ok:
                    stats[f"ok_{ai.action_type}"] += 1
                else:
                    stats[f"MISS_{ai.action_type}_{why}"] += 1
                    if misses_shown < args.verbose_misses:
                        misses_shown += 1
                        side = gs.global_info.current_side
                        log.info(
                            f"MISS [{gz.name}] turn="
                            f"{gs.global_info.turn_number} side={side} "
                            f"{ai.action_type} actor={ai.actor_idx} "
                            f"target={ai.target_idx} ({why})")
        except Exception as e:  # noqa: BLE001 -- keep sweeping
            stats["replay_error"] += 1
            log.warning(f"{gz.name}: {type(e).__name__}: {e}")

    log.info("=== mask coverage over %d games ===", len(files))
    for k in sorted(stats):
        log.info(f"  {k}: {stats[k]}")
    total_checked = sum(v for k, v in stats.items()
                        if k.startswith(("ok_", "MISS_")))
    n_miss = sum(v for k, v in stats.items() if k.startswith("MISS_"))
    if total_checked:
        log.info(f"  coverage: {total_checked - n_miss}/{total_checked}")
    return 1 if n_miss else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
