#!/usr/bin/env python3
"""Round-6 driver: per-decade aleatoric-label probe.

    python signal_profiler/run_aleatoric.py \
        --checkpoint training/checkpoints/armV3_cliff_2951396.pt \
        --games 8 --out aleatoric_cliff.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_profiler.aleatoric_probe import (  # noqa: E402
    DECADE_KEYS, aleatoric_probe,
)
from signal_profiler.experience_harvest import (  # noqa: E402
    harvest_experiences, make_policy,
)


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level))

    import torch
    device = (torch.device("cuda")
              if args.device == "cuda" and torch.cuda.is_available()
              else None)
    factory = make_policy(args.checkpoint, device)
    gen = factory()
    batch, outcomes = harvest_experiences(
        gen, args.games, args.seed, max_turns=args.max_turns)
    del gen
    if not batch:
        print("no experiences harvested")
        return 1

    result = aleatoric_probe(factory, batch)
    result["outcomes"] = [
        {"winner": getattr(o, "winner", None),
         "turns": getattr(o, "turns", None)} for o in outcomes]
    args.out.write_text(json.dumps(result), encoding="utf-8")

    print(f"\npooled: {result['pooled']}")
    print(f"{'decade':8s} {'n':>5s} {'auc':>6s} {'ce':>6s} "
          f"{'floor':>6s} {'cos(w,l)':>9s} {'|gw|':>6s} {'|gl|':>6s}")
    for key in DECADE_KEYS:
        md = result["by_decade"].get(key)
        gd = result["grad_by_decade"].get(key, {})
        if not md:
            continue
        auc = md["auc"]
        print(f"{key:8s} {md['n']:5d} "
              f"{auc if auc == auc else float('nan'):6.3f} "
              f"{md['ce']:6.3f} {md['floor']:6.3f} "
              f"{gd.get('cos_win_lose', float('nan')):9.3f} "
              f"{gd.get('norm_win', float('nan')):6.2f} "
              f"{gd.get('norm_lose', float('nan')):6.2f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
