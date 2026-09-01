#!/usr/bin/env python3
"""Signal-profiling v2 driver: the complete decomposition.

Adds to the v1 gradient tree: consultation capture during harvest,
target category split, value-gradient provenance splits, and the
update-space (post-Adam, production clip) tree with per-term value
movement on imagined vs real states.

    python signal_profiler/run_profile_v2.py \
        --checkpoint training/checkpoints/seed_imit_tierb_start.pt \
        --games 8 --out profile_v2.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_profiler.consult_capture import (  # noqa: E402
    capture_consultations,
)
from signal_profiler.experience_harvest import (  # noqa: E402
    harvest_experiences, make_policy,
)
from signal_profiler.gradient_tree import build_tree  # noqa: E402
from signal_profiler.render import render_tree  # noqa: E402
from signal_profiler.target_amplitude import (  # noqa: E402
    target_amplitude,
)
from signal_profiler.update_tree import build_update_tree  # noqa: E402
from signal_profiler.value_splits import value_grad_splits  # noqa: E402


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed", type=int, default=31337)
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--consult-cap", type=int, default=400)
    ap.add_argument("--real-probe", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vg", action="store_true",
                    help="Arm-VG provenance split: value term "
                         "decomposed into game/ground/consist "
                         "sub-terms (requires a grounding-enabled "
                         "harvest to be meaningful).")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level))

    import torch
    device = (torch.device("cuda")
              if args.device == "cuda" and torch.cuda.is_available()
              else None)

    factory = make_policy(args.checkpoint, device,
                          grounding=args.vg)
    surgeries = None
    if args.vg:
        from signal_profiler.gradient_tree import TERM_SURGERY_VG
        surgeries = TERM_SURGERY_VG
    gen_policy = factory()
    with capture_consultations(cap=args.consult_cap,
                               seed=args.seed) as res:
        batch, outcomes = harvest_experiences(
            gen_policy, args.games, args.seed,
            max_turns=args.max_turns)
    del gen_policy
    if not batch:
        print("no experiences harvested; nothing to profile")
        return 1
    consult_states = res.states
    rng = random.Random(args.seed)
    real_states = [e.game_state for e in
                   rng.sample(batch, min(args.real_probe, len(batch)))]
    n_gbc = sum(1 for e in batch if getattr(e, "gbc_labels", None))
    n_aux = sum(1 for e in batch
                if getattr(e, "aux_target", None) is not None)
    print(f"label coverage: gbc={n_gbc}/{len(batch)} "
          f"aux={n_aux}/{len(batch)}")
    print(f"consult states captured: {len(consult_states)} "
          f"(of {res.seen} seen); real probe: {len(real_states)}")

    ta_policy = factory()
    targets = target_amplitude(ta_policy, batch)
    del ta_policy
    print(f"target amplitude: {targets}")

    tree = build_tree(factory, batch, surgeries=surgeries)
    tree["target_amplitude"] = targets
    print(f"linearity residual: "
          f"{tree.get('linearity_residual_frac', float('nan')):.4f}")

    tree["value_grad_splits"] = value_grad_splits(factory, batch)
    print(f"value splits: {tree['value_grad_splits']}")

    tree["update_tree"] = build_update_tree(
        factory, batch, consult_states, real_states,
        surgeries=surgeries)
    tree["outcomes"] = [
        {"winner": getattr(o, "winner", None),
         "turns": getattr(o, "turns", None)} for o in outcomes]
    args.out.write_text(json.dumps(tree), encoding="utf-8")
    print(render_tree(tree))
    ut = tree["update_tree"]
    print(f"\nUPDATE tree (post-Adam, clip 1.0; opt_state="
          f"{ut['has_optimizer_state']}):")
    for t, node in ut["terms"].items():
        vm = node["value_movement"]
        print(f"  {t:15s} |du|={node['update_norm']:.4f} "
              f"proj={node['proj_frac']:+.3f} "
              f"dv_consult={vm['consult'].get('mean_abs', 0):.5f} "
              f"dv_real={vm['real'].get('mean_abs', 0):.5f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
