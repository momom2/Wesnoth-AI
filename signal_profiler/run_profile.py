#!/usr/bin/env python3
"""Signal-profiling round driver (signal_profiler v1).

Plays N production-config games, then builds the gradient-amplitude
tree (see README.md). Everything runs on fresh policy copies; the
checkpoint on disk is never touched.

    python signal_profiler/run_profile.py \
        --checkpoint training/checkpoints/seed_imit_tierb_start.pt \
        --games 8 --out profile.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_profiler.experience_harvest import (  # noqa: E402
    harvest_experiences, make_policy,
)
from signal_profiler.gradient_tree import build_tree  # noqa: E402
from signal_profiler.render import render_tree  # noqa: E402


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--no-turn-search", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level))

    import torch
    device = (torch.device("cuda")
              if args.device == "cuda" and torch.cuda.is_available()
              else None)

    factory = make_policy(args.checkpoint, device,
                          turn_search=not args.no_turn_search)
    gen_policy = factory()
    batch, outcomes = harvest_experiences(
        gen_policy, args.games, args.seed, max_turns=args.max_turns)
    del gen_policy
    if not batch:
        print("no experiences harvested; nothing to profile")
        return 1
    n_gbc = sum(1 for e in batch if getattr(e, "gbc_labels", None))
    n_aux = sum(1 for e in batch
                if getattr(e, "aux_target", None) is not None)
    print(f"label coverage: gbc={n_gbc}/{len(batch)} "
          f"aux={n_aux}/{len(batch)}")

    tree = build_tree(factory, batch)
    print(f"linearity residual: "
          f"{tree.get('linearity_residual_frac', float('nan')):.4f} "
          f"(should be ~0 unclipped; larger = normalization "
          f"coupling)")
    tree["outcomes"] = [
        {"winner": getattr(o, "winner", None),
         "turns": getattr(o, "turns", None)} for o in outcomes]
    args.out.write_text(json.dumps(tree), encoding="utf-8")
    print(render_tree(tree))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
