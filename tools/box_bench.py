#!/usr/bin/env python3
"""2-minute box-shape microbenchmark (user directive 2026-08-05:
"we can't A/B every time we switch box -- it's too long").

Measures the numbers the actor pool's cost model needs ON THIS BOX
and prints its projected games per hour instead of an hour-scale run:

  - model forward latency, GPU, batch 1/8/16/32/64/128 (mixed-length
    states, so padding waste is priced in)
  - encode_raw + legal-action enumeration per state (CPU, serial)
  - core count

Cost model (see BACKLOG "Throughput program"):

  pool  games/hr  ~ min( GPU ceiling:   3600 * B_eff / (D * S * fwd_B),
                         CPU ceiling:   cores * 3600 / (D * S * (enc + enum)) )

with D = decisions/game (default 500), S = mean sims/decision after
playout-cap (default 14). The projection is a PREDICTION, not a
measurement of the pool.

Usage:  python tools/box_bench.py [--checkpoint CKPT] [--decisions 500]
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Checkpoint for arch+weights (default: a "
                         "fresh 15M Tier-b-shaped net).")
    ap.add_argument("--decisions", type=int, default=500,
                    help="Assumed decisions/game for the projection.")
    ap.add_argument("--mean-sims", type=float, default=14.0,
                    help="Mean sims/decision (32-sim budget under "
                         "default playout-cap = 14).")
    ap.add_argument("--states", type=int, default=8,
                    help="Distinct states per batch (padding realism).")
    args = ap.parse_args(argv[1:])

    import os
    import torch
    from wesnoth_ai.transformer_policy import TransformerPolicy
    from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors
    from wesnoth_ai.rules.scenario_pool import random_setup, build_scenario_gamestate

    n_cores = os.cpu_count() or 1
    # A bench on a busy box measures contention, not the box (the
    # 2026-08-05 calibration attempt read cpu-forward 5487ms vs the
    # true ~60-100ms because 76 workers were running). Warn
    # loudly; results under load are NOT calibration-grade.
    try:
        # getloadavg is absent on Windows (AttributeError, not OSError).
        load1 = os.getloadavg()[0]
        if load1 > n_cores * 0.25:
            print(f"WARNING: loadavg {load1:.0f} on {n_cores} cores "
                  f"-- box is busy; numbers below measure CONTENTION, "
                  f"re-run on an idle box before trusting them")
    except (OSError, AttributeError):
        pass
    has_cuda = torch.cuda.is_available()
    gpu = (torch.cuda.get_device_name(0) if has_cuda else "none")

    if args.checkpoint:
        raw = torch.load(args.checkpoint, map_location="cpu",
                         weights_only=False)
        a = raw["arch"]
        kw = dict(aux_score=bool(raw.get("aux_score")),
                  moves_left=bool(raw.get("moves_left")),
                  relevant_set_hexes=bool(raw.get("relevant_set_hexes")))
    else:
        a = dict(d_model=384, num_layers=8, num_heads=12, d_ff=1536)
        kw = {}

    def build(device):
        pol = TransformerPolicy(device=torch.device(device), **a, **kw)
        if args.checkpoint:
            pol.load_checkpoint(args.checkpoint)
        return pol

    # ---- states (mixed maps for realistic sequence-length spread) ----
    cpu_pol = build("cpu")
    enc = cpu_pol._inference_encoder
    states = []
    for i in range(args.states):
        setup = random_setup(random.Random(100 + i),
                             category="mini" if i % 3 == 0 else "ladder")
        gs = build_scenario_gamestate(
            setup, starting_gold=None, base_income=2, village_gold=2,
            village_upkeep=1, experience_modifier=70)
        enc.register_names(gs)
        states.append(gs)

    def timeit(fn, n, warmup=3):
        for _ in range(warmup):
            fn()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n * 1000.0   # ms

    # ---- CPU-side per-state costs ----
    # A worker's thread budget, not the idle machine's: the 2026-08-05
    # idle calibration measured 25 ms with all cores against the
    # ~60-100 ms a real worker paid.
    torch.set_num_threads(2)
    encoded = [enc.encode(gs) for gs in states]
    t_enc = timeit(lambda: enc.encode(states[0]), 10)
    with torch.no_grad():
        out0 = cpu_pol._inference_model(encoded[0])
    t_enum = timeit(lambda: enumerate_legal_actions_with_priors(
        encoded[0], out0, states[0]), 10)

    print(f"box: {n_cores} cores, gpu={gpu}")
    print(f"arch: {a}")
    print(f"encode          {t_enc:8.2f} ms/state (cpu, serial)")
    print(f"enumerate       {t_enum:8.2f} ms/state (cpu, serial)")

    t_gpu = {}
    if has_cuda:
        gpu_pol = build("cuda")
        genc = gpu_pol._inference_encoder
        for gs in states:
            genc.register_names(gs)
        gencoded = [genc.encode(gs) for gs in states]
        m = gpu_pol._inference_model
        with torch.no_grad():
            for B in (1, 8, 16, 32, 64, 128):
                batch = [gencoded[i % len(gencoded)] for i in range(B)]

                def fwd():
                    if B == 1:
                        m(batch[0])
                    else:
                        m.forward_batch(batch)
                    torch.cuda.synchronize()

                t_gpu[B] = timeit(fwd, 6)
                print(f"forward gpu b{B:<3} {t_gpu[B]:8.2f} ms "
                      f"({t_gpu[B] / B:6.2f} ms/state)")

    # ---- projection ----
    D, S = args.decisions, args.mean_sims
    print()
    print(f"PROJECTION (D={D} decisions/game, mean sims {S}):")
    if t_gpu:
        best_B, best = min(
            ((B, t / B) for B, t in t_gpu.items() if B > 1),
            key=lambda kv: kv[1])
        gpu_ceiling = 3600e3 / (D * S * best)
        cpu_ceiling = n_cores * 3600e3 / (D * S * (t_enc + t_enum))
        pool = min(gpu_ceiling, cpu_ceiling)
        lim = "GPU" if gpu_ceiling < cpu_ceiling else "CPU(enc+enum)"
        print(f"  actor-pool (batch {best_B}):    {pool:7.1f} games/hr "
              f"[{lim}-limited; gpu ceiling {gpu_ceiling:.1f}, "
              f"cpu ceiling {cpu_ceiling:.1f}]")
    else:
        print("  no CUDA: no pool projection.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
