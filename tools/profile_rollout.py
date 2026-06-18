"""Profile an MCTS self-play rollout by component, to decide where
native-code / optimization effort would actually pay off.

Times the real production rollout (`play_one_game` via an MCTSPolicy)
broken into: game logic (`sim.step`, `sim.fork`), `encode`, the legal-
action sampler (`enumerate_legal_actions_with_priors`), the model
`forward`, and the remainder (MCTS tree bookkeeping + the per-action
deepcopy snapshot). Reports two views:

  * CPU rollout  — as measured on this box.
  * GPU/actor-pool throughput regime — the forward term dropped, because
    on a batched inference server it is sub-ms AND overlapped (with the
    actor pool oversubscribed, a blocked actor's core runs another), so
    per-actor THROUGHPUT is bounded by local CPU work. This view is what
    decides whether the simulator (or anything else) is the bottleneck.

Why it exists: a one-off profile (2026-06-17; laptop CPU, 0.47M net,
mini maps) found the game-logic sim is only ~¼ of the throughput-
relevant cost, the sampler is the single biggest Python component, and
no single component dominates — so a full Rust simulator rewrite is poor
ROI (Amdahl-capped, plus the bit-exact-parity re-validation cost). RE-RUN
THIS on the rented GPU box during Phase 0 with the TARGET model size
(`--d-model ...` or `--checkpoint-in`) and FULL maps (`--mini-ratio 0`):
a bigger forward + bigger boards shift the breakdown, and the decision
should rest on real-hardware numbers, not the laptop estimate.

CUDA note: GPU kernels are async, so the profiler SYNCHRONIZES after each
GPU-touching component for accurate per-component attribution. That
serializes GPU work, so the reported games/sec is a LOWER bound under
profiling — measure true throughput with the actor pool, not here.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent.parent))
sys.path.insert(0, str(_THIS.parent))

log = logging.getLogger("profile_rollout")

# Components timed individually; everything else falls into "remainder".
_COMPONENTS = ["sim.step", "sim.fork", "encode", "enumerate", "forward"]


def _build_policy(args, device):
    """An MCTSPolicy at the requested arch (checkpoint or fresh init)."""
    from transformer_policy import TransformerPolicy
    from tools.mcts import MCTSConfig
    from tools.mcts_policy import MCTSPolicy

    arch: Dict[str, int] = {}
    if args.checkpoint_in and args.checkpoint_in.exists():
        raw = torch.load(args.checkpoint_in, map_location="cpu",
                         weights_only=False)
        for k in ("d_model", "num_layers", "num_heads", "d_ff"):
            v = (raw.get("arch") or {}).get(k)
            if v is not None:
                arch[k] = int(v)
    else:
        arch = dict(d_model=args.d_model, num_layers=args.num_layers,
                    num_heads=args.num_heads, d_ff=args.d_ff)
    base = TransformerPolicy(device=device, **arch)
    if args.checkpoint_in and args.checkpoint_in.exists():
        try:
            base.load_checkpoint(args.checkpoint_in)
        except RuntimeError as e:
            log.warning(f"checkpoint load failed ({e}); profiling fresh init")
    cfg = MCTSConfig(
        n_simulations=args.mcts_sims, gumbel_root=True,
        gumbel_m=args.mcts_gumbel_m, chance_nodes=True,
        exact_outcome_enumeration=True, batch_size=1, add_root_noise=False)
    return MCTSPolicy(base, cfg), base, arch


def profile_rollout(
    base, mcts_policy, *, device, n_games: int, max_turns: int,
    mini_ratio: float, forced_faction, seed: int, warmup_turns: int,
) -> Dict:
    """Run `n_games` profiled games and return the timing breakdown."""
    import tools.mcts as mcts_mod
    from tools.scenario_pool import random_setup
    from tools.sim_self_play import _play_one_game_safe, _recruit_cost_lookup
    from wesnoth_sim import PvPDefaults

    sync = (device.type == "cuda")
    acc: Dict[str, List[float]] = {c: [0, 0.0] for c in _COMPONENTS}

    def _timed(orig, label, gpu_touch):
        def timed(*a, **k):
            t = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                if sync and gpu_touch:
                    torch.cuda.synchronize()
                acc[label][0] += 1
                acc[label][1] += time.perf_counter() - t
        return timed

    # Build one sim up front so we can patch the ACTUAL sim class (a
    # top-level `import wesnoth_sim` can resolve to a different module
    # object than the one mcts uses -- patch via type(sim) to be safe).
    rng = random.Random(seed)
    enc, mdl = base._inference_encoder, base._inference_model
    cost = _recruit_cost_lookup()
    pvp = PvPDefaults()

    probe = random_setup(rng, forced_faction=forced_faction,
                         mini_ratio=mini_ratio)
    from tools.scenario_pool import build_scenario_gamestate
    from wesnoth_sim import WesnothSim
    SimCls = type(WesnothSim(build_scenario_gamestate(probe),
                             scenario_id=probe.scenario_id, max_turns=2))

    originals = {
        ("sim", "step"): SimCls.step,
        ("sim", "fork"): SimCls.fork,
        ("enc",): enc.encode,
        ("mdl", "forward"): mdl.forward,
        ("mdl", "forward_batch"): mdl.forward_batch,
        ("mcts",): mcts_mod.enumerate_legal_actions_with_priors,
    }
    SimCls.step = _timed(SimCls.step, "sim.step", False)
    SimCls.fork = _timed(SimCls.fork, "sim.fork", False)
    enc.encode = _timed(enc.encode, "encode", True)
    mdl.forward = _timed(mdl.forward, "forward", True)
    mdl.forward_batch = _timed(mdl.forward_batch, "forward", True)
    mcts_mod.enumerate_legal_actions_with_priors = _timed(
        mcts_mod.enumerate_legal_actions_with_priors, "enumerate", sync)

    try:
        # Warmup (clears CUDA context / cuDNN autotune / first-call JIT);
        # timers run but are zeroed before the measured games.
        if warmup_turns > 0:
            setup = random_setup(rng, forced_faction=forced_faction,
                                 mini_ratio=mini_ratio)
            _play_one_game_safe(setup=setup, max_turns=warmup_turns,
                                pvp_defaults=pvp, policy=mcts_policy,
                                reward_fn=lambda d: 0.0, cost_lookup=cost,
                                game_label="warmup")
            for c in acc:
                acc[c] = [0, 0.0]

        t0 = time.perf_counter()
        n_actions = 0
        for g in range(n_games):
            setup = random_setup(rng, forced_faction=forced_faction,
                                 mini_ratio=mini_ratio)
            out = _play_one_game_safe(
                setup=setup, max_turns=max_turns, pvp_defaults=pvp,
                policy=mcts_policy, reward_fn=lambda d: 0.0,
                cost_lookup=cost, game_label=f"prof{g}")
            if out is not None:
                n_actions += out.side1_actions + out.side2_actions
        total = time.perf_counter() - t0
    finally:
        SimCls.step = originals[("sim", "step")]
        SimCls.fork = originals[("sim", "fork")]
        enc.encode = originals[("enc",)]
        mdl.forward = originals[("mdl", "forward")]
        mdl.forward_batch = originals[("mdl", "forward_batch")]
        mcts_mod.enumerate_legal_actions_with_priors = originals[("mcts",)]

    named = sum(acc[c][1] for c in _COMPONENTS)
    return {
        "components": {c: {"calls": acc[c][0], "seconds": acc[c][1]}
                       for c in _COMPONENTS},
        "remainder_seconds": max(0.0, total - named),
        "total_seconds": total, "n_games": n_games, "n_actions": n_actions,
    }


def _print_report(rep: Dict, device) -> None:
    total = rep["total_seconds"]
    comp = rep["components"]
    rem = rep["remainder_seconds"]
    fwd = comp["forward"]["seconds"]
    gpu_total = max(1e-9, total - fwd)   # forward -> ~0 regime

    print()
    print("=" * 60)
    print(f"Rollout profile  ({rep['n_games']} games, {rep['n_actions']} "
          f"actions, {total:.1f}s, device={device})")
    print("=" * 60)
    print(f"{'component':<14}{'calls':>9}{'sec':>8}{'us/call':>9}"
          f"{'%CPU':>7}{'%GPUreg':>8}")
    print("-" * 60)
    rows = [(c, comp[c]["calls"], comp[c]["seconds"]) for c in _COMPONENTS]
    rows.append(("remainder", 0, rem))
    for name, calls, sec in sorted(rows, key=lambda r: -r[2]):
        calls_str = str(calls) if calls else ""
        us_str = f"{sec / calls * 1e6:.0f}" if calls else ""
        gpu_pct = "" if name == "forward" else f"{100 * sec / gpu_total:.1f}%"
        print(f"{name:<14}{calls_str:>9}{sec:>8.2f}{us_str:>9}"
              f"{100 * sec / total:>6.1f}%{gpu_pct:>8}")
    print("-" * 60)
    print(f"{'TOTAL':<14}{'':>9}{total:>8.2f}{'':>9}{100.0:>6.1f}%")
    if rep["n_actions"]:
        print(f"\nthroughput (profiled, GPU-serialized): "
              f"{rep['n_actions']/total:.1f} actions/s, "
              f"{rep['n_games']/total*3600:.0f} games/hr  (1 actor; the "
              f"actor pool scales ~linearly with cores)")
    print(f"'%GPUreg' = share once the forward is batched+overlapped away "
          f"(forward was {100 * fwd / total:.0f}% of CPU rollout).")


def main(argv: List[str]) -> int:
    from tools.device_select import select_inference_device, describe_device
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-in", type=Path, default=None,
                    help="Load arch+weights from this checkpoint "
                         "(else fresh init at the --d-model flags).")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=256)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--max-turns", type=int, default=24)
    ap.add_argument("--mcts-sims", type=int, default=50)
    ap.add_argument("--mcts-gumbel-m", type=int, default=16)
    ap.add_argument("--mini-ratio", type=float, default=0.0,
                    help="0 = full ladder maps (representative); 1 = "
                         "mini maps (faster, understates sim/encode).")
    ap.add_argument("--forced-faction", default=None)
    ap.add_argument("--warmup-turns", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-json", type=Path, default=None)
    ap.add_argument("--log-level", default="WARNING",
                    choices=["DEBUG", "INFO", "WARNING"])
    args = ap.parse_args(argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    device = select_inference_device(args.device)
    log.info(f"device: {describe_device(device)}")
    forced = (None if (args.forced_faction or "").lower() == "none"
              else args.forced_faction) if args.forced_faction else ...

    mcts_policy, base, arch = _build_policy(args, device)
    log.info(f"arch={arch} sims={args.mcts_sims} mini_ratio={args.mini_ratio}")
    rep = profile_rollout(
        base, mcts_policy, device=device, n_games=args.games,
        max_turns=args.max_turns, mini_ratio=args.mini_ratio,
        forced_faction=forced, seed=args.seed, warmup_turns=args.warmup_turns)
    rep["arch"] = arch
    rep["device"] = str(device)
    _print_report(rep, device)
    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(rep, indent=2, default=str),
                                  encoding="utf-8")
        log.info(f"wrote {args.save_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
