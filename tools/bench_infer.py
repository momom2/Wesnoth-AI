"""Pointed inference benchmark: eager vs bf16 vs torch.compile vs
compile+shape-bucketing, over the REAL per-decision shape stream.

Answers ONE question (user, 2026-08-28): does compiled inference pay
on our workload -- where nearly every decision presents a new token-
sequence shape (unit/hex counts change as the game evolves) -- and
does padding shapes up to bucket edges rescue it from recompiles?

Run on a rented GPU box, never the laptop. States come from
dummy-policy games (no net needed to generate them), so the shape
distribution is the sim's own, then each arm forwards the SAME
encoded states through the checkpoint model.

Caveat printed with results: the padded arm zero-pads without a
key-padding mask (the single-sample forward has none), so its
OUTPUTS are not production-correct -- FLOPs and shapes are, which
is what a timing needs. Adopting padding for real would need mask
support in the single-sample path (forward_batch already has it).

Usage (GPU box):
    python tools/bench_infer.py --checkpoint training/checkpoints/seed_imit_tierb_start.pt \
        --n-states 150 --pad-multiple 64
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import logging
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger("bench_infer")


def harvest_states(n: int, seed: int):
    """Deep-copied GameStates from dummy-vs-dummy sim games -- the
    sim's real shape stream, no network involved."""
    from tools.elo_ladder import _ScriptedAdapter
    from tools.eval_sim import _PolicyPair, _play_one_eval_game
    from tools.scenario_pool import build_scenario_gamestate, random_setup
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.dummy_policy import DummyPolicy
    import random as _random

    sink: list = []
    cap_per_game = max(4, n // 8)   # force >=8 games' worth of maps
    stride = 3                      # skip adjacent near-identical shapes

    class _Recorder:
        def __init__(self, inner):
            self._inner = inner
            self._seen = 0
            self._taken = 0

        def select_action(self, gs, **kw):
            self._seen += 1
            if (len(sink) < n and self._taken < cap_per_game
                    and self._seen % stride == 0):
                sink.append(copy.deepcopy(gs))
                self._taken += 1
            return self._inner.select_action(gs, **kw)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    g = 0
    while len(sink) < n and g < 50:
        rng = _random.Random(seed + g)
        g += 1
        setup = random_setup(rng)
        gs = build_scenario_gamestate(setup)
        sim = WesnothSim(gs, scenario_id=setup.scenario_id,
                         max_turns=30)
        _play_one_eval_game(
            sim,
            _PolicyPair(policy=_Recorder(_ScriptedAdapter(DummyPolicy())),
                        label="a", side=1),
            _PolicyPair(policy=_Recorder(_ScriptedAdapter(DummyPolicy())),
                        label="b", side=2),
            game_label=f"bench{g}")
    log.info("harvested %d states from %d dummy games", len(sink), g)
    return sink[:n]


def pad_to_buckets(enc, mult: int):
    """Zero-pad the three variable blocks up to multiples of `mult`
    so the compiler sees few distinct shapes. Timing-faithful only
    (no mask; see module docstring)."""
    import torch

    def _pad(t):
        length = t.size(1)
        target = max(mult, -(-length // mult) * mult)
        if target == length:
            return t
        return torch.cat(
            [t, torch.zeros(t.size(0), target - length, t.size(2),
                            device=t.device, dtype=t.dtype)], dim=1)

    return dataclasses.replace(
        enc, hex_tokens=_pad(enc.hex_tokens),
        unit_tokens=_pad(enc.unit_tokens),
        recruit_tokens=_pad(enc.recruit_tokens))


def time_arm(name: str, model, encs, sync) -> dict:
    """Forward every enc once; per-call wall times. The first call
    (and any compile-triggered slow call) is reported separately
    from the steady state."""
    times = []
    t_first = None
    for i, e in enumerate(encs):
        t0 = time.perf_counter()
        model(e)
        sync()
        dt = (time.perf_counter() - t0) * 1000.0
        if i == 0:
            t_first = dt
        else:
            times.append(dt)
    med = statistics.median(times)
    slow = sum(1 for t in times if t > 5 * med)
    return {"arm": name, "first_ms": t_first,
            "median_ms": med,
            "p90_ms": sorted(times)[int(0.9 * len(times))],
            "mean_ms": statistics.fmean(times),
            "slow_calls_gt5xmed": slow}


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-states", type=int, default=150)
    ap.add_argument("--seed", type=int, default=97)
    ap.add_argument("--pad-multiple", type=int, default=64)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level))

    import torch
    from tools.eval_sim import _load_policy

    device = (torch.device("cuda") if args.device == "cuda"
              else None)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")
    sync = (torch.cuda.synchronize if args.device == "cuda"
            else (lambda: None))

    states = harvest_states(args.n_states, args.seed)
    policy = _load_policy(Path(args.checkpoint), device, label="bench")
    enc = policy._inference_encoder
    with torch.no_grad():
        encs = [enc.encode(gs) for gs in states]
    shapes = sorted({(e.hex_tokens.size(1), e.unit_tokens.size(1),
                      e.recruit_tokens.size(1)) for e in encs})
    log.info("%d states, %d distinct (H,U,R) shapes", len(encs),
             len(shapes))
    padded = [pad_to_buckets(e, args.pad_multiple) for e in encs]
    pshapes = {(e.hex_tokens.size(1), e.unit_tokens.size(1),
                e.recruit_tokens.size(1)) for e in padded}
    log.info("padded to multiples of %d: %d distinct shapes",
             args.pad_multiple, len(pshapes))

    base = policy._inference_model
    results = []

    def run(name, model, inputs, bf16=False):
        prev = getattr(base, "infer_autocast_bf16", False)
        base.infer_autocast_bf16 = bf16
        try:
            with torch.no_grad():
                results.append(time_arm(name, model, inputs, sync))
        finally:
            base.infer_autocast_bf16 = prev
        log.info("%s: %s", name, results[-1])

    run("eager-fp32", base, encs)
    run("eager-bf16", base, encs, bf16=True)
    run("eager-fp32-padded", base, padded)

    for bf16 in (False, True):
        torch._dynamo.reset()
        cm = torch.compile(base)
        tag = "bf16" if bf16 else "fp32"
        run(f"compile-{tag}", cm, encs, bf16=bf16)
        try:
            from torch._dynamo.utils import counters
            log.info("dynamo stats after compile-%s: %s", tag,
                     dict(counters["stats"]))
        except Exception:  # noqa: BLE001
            pass

    torch._dynamo.reset()
    cm = torch.compile(base)
    run("compile-fp32-padded", cm, padded)
    torch._dynamo.reset()
    cm = torch.compile(base)
    run("compile-bf16-padded", cm, padded, bf16=True)

    print("\narm                    first_ms  median_ms  p90_ms  "
          "mean_ms  slow>5xmed")
    for r in results:
        print(f"{r['arm']:22s} {r['first_ms']:8.1f} {r['median_ms']:9.2f}"
              f" {r['p90_ms']:7.2f} {r['mean_ms']:8.2f}"
              f" {r['slow_calls_gt5xmed']:6d}")
    print("\nNOTE: padded arms are timing-faithful but mask-less; "
          "production padding needs single-path mask support.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
