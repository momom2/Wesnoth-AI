"""Whether the memory carries a full-network step's divergence (CPU, float32;
Idle priority).

Stage C's evaluator replays each stepped network on its own memory along a
game-side (tools/pg_eval.py _walk), and a full-network step's KL sits in a few
game-sides, from some 20 to 40 decisions in and on to most of the side
(trunk_step_tail.py). This script walks chosen game-sides decision by decision
through lanes, each one network reading one memory, theta' = theta_k + s d
with d stage C's reduced_adamw direction (its pass.pt) and s its step:

  base         theta_k on its own memory: the reference law.
  step         theta' on its own memory: stage C's reading.
  step_fed     theta' reading theta_k's memory at every decision, its own
               write dropped: the step with the memory held.
  base_fed     theta_k reading theta''s memory, its own write dropped: the
               memory's part alone.
  transplant   theta_k on its own memory, which at decision --at is set once
               to theta''s: whether theta_k's own recurrence keeps the gap.
  noise        the same from a random gap of the same norm.
  half_step    theta_k + (s / 2) d on its own memory: how the divergence
               scales with the step.
  memory_only  theta_k + s d on the memory's own parameters
               (model.slot_memory.*) alone, on its own memory.
  rest_only    theta_k + s d on every other parameter, on its own memory.

Per decision and lane it keeps Delta = log pi_base(a) - log pi_lane(a) at the
sampled action, the exact KL(pi_base || pi_lane), and for each carried memory
its distance from the base's over the base's norm, at the read. If the memory
carries the divergence, base_fed reads about what step reads and step_fed
stays small.

With --scales the lanes are the base and, for each multiple m, theta_k + m s d
on its own memory (step_x<m>): how small a step, and of which sign, still
sets the divergence off.

    python scripts/window_20261029/trunk_memory_lane.py --sides game_own2a_own2b_s2_827637:2 game_own2a_own2b_s2_827637:1 --out training/metrics/trunk_memory_lane_20261110.json
    python scripts/window_20261029/trunk_memory_lane.py --sides game_own2a_own2b_s2_827637:2 --scales 0.25 0.0625 0.015625 -1 --out training/metrics/trunk_memory_lane_scales_20261110.json
"""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# (lane, its network, the memory it reads, whether its write is carried on)
LANES = (
    ("base", "k", "base", True),
    ("base_fed", "k", "step", False),
    ("transplant", "k", "transplant", True),
    ("noise", "k", "noise", True),
    ("step", "step", "step", True),
    ("step_fed", "step", "base", False),
    ("half_step", "half", "half_step", True),
    ("memory_only", "memory", "memory_only", True),
    ("rest_only", "rest", "rest_only", True),
)
LATE = ("transplant", "noise")                 # the memories that start at --at
MEMORY_PREFIX = "model.slot_memory."
PARTIAL_EVERY = 50                             # decisions between two writes of a side's partial reading


def scale_lanes(scales) -> tuple:
    """The base lane and, per multiple m of stage C's step, theta_k + m s d on
    its own memory."""
    return (LANES[0], *((f"step_x{m:g}", f"x{m!r}", f"step_x{m:g}", True) for m in scales))


def walk(context, nets, lanes, gs, at: int, generator, partial):
    """The readings of `lanes` along game-side `gs` (module docstring), a dict
    of per-decision lists; `partial(reading)` every PARTIAL_EVERY decisions."""
    import torch
    from tools.pg_pass import stage_step
    from wesnoth_ai.memory import MemoryBatch
    from wesnoth_ai.pg_loss import LAW_KEYS, action_log_probs, factor_laws, legal_kl_laws

    positions = context.loader.side(gs.game, gs.side)
    if len(positions) != gs.stop - gs.start:
        raise ValueError(f"{gs.game} side {gs.side}: {len(positions)} positions for {gs.stop - gs.start} decisions")
    k, device = context.config.slots, context.device
    state = {lane: nets[net][1].initial_memory(k).to(device) for lane, net, _, carried in lanes
             if carried and lane not in LATE}
    late = any(lane in LATE for lane, *_ in lanes)
    out = {"decisions": len(positions), "logged": [], "base_logp": [], "base_write": [],
           "delta": {lane: [] for lane, *_ in lanes[1:]}, "kl": {lane: [] for lane, *_ in lanes[1:]},
           "gap": {lane: [] for lane, _, _, carried in lanes[1:] if carried}}
    with torch.no_grad():
        for t, p in enumerate(positions):
            if late and t == at:
                gap = state["step"] - state["base"]
                u = torch.randn(gap.shape, generator=generator).to(device)
                state["transplant"] = state["step"].clone()
                state["noise"] = state["base"] + u * (gap.norm() / u.norm())
            live = [lane for lane in lanes if lane[2] in state]
            base_norm = state["base"].norm()
            for lane in out["gap"]:
                out["gap"][lane].append(float((state[lane] - state["base"]).norm() / base_norm)
                                        if lane in state else None)
            logp, kl, wrote, ref = {}, {}, {}, None
            for net in dict.fromkeys(lane[1] for lane in live):          # theta_k first: the reference law
                mine = [lane for lane in live if lane[1] == net]
                encoder, model = nets[net]
                staged = context.encoder.stage_raws([p.raw] * len(mine), device=device)
                batch = stage_step([p] * len(mine), staged, context.model, device)
                memory = MemoryBatch(padded=torch.stack([state[lane[2]] for lane in mine]), counts=[k] * len(mine))
                padded = model.forward_embedded(encoder.embed_staged(staged), memory=memory,
                                                packed=context.config.packed)
                laws = factor_laws(padded, batch.views, batch.rows)
                if ref is None:                                          # the base lane is theta_k's row 0
                    ref = {key: getattr(laws, key)[:1] for key in LAW_KEYS}
                base_laws = dataclasses.replace(laws, **{key: ref[key].expand_as(getattr(laws, key))
                                                         for key in LAW_KEYS})
                row_kl = legal_kl_laws(base_laws, laws)
                row_logp = action_log_probs(padded, batch)
                for b, (lane, _, _, carried) in enumerate(mine):
                    logp[lane], kl[lane] = float(row_logp[b]), float(row_kl[b])
                    if carried:
                        wrote[lane] = padded.memory_padded[b, :k].clone()
            out["logged"].append(float(p.logp))
            out["base_logp"].append(logp["base"])
            out["base_write"].append(float((wrote["base"] - state["base"]).norm() / base_norm))
            for lane, *_ in lanes[1:]:
                out["delta"][lane].append(logp["base"] - logp[lane] if lane in logp else None)
                out["kl"][lane].append(kl.get(lane))
            if not p.shadow:                                             # a refused row writes nothing
                state.update(wrote)
            if (t + 1) % PARTIAL_EVERY == 0:
                partial(out)
    return out


def summary(reading) -> dict:
    """Each lane's mean KL, half mean Delta^2, summed Delta and largest
    |Delta| over the decisions it read, and each memory's gap along the side."""
    import numpy as np
    lanes = {}
    for lane, deltas in reading["delta"].items():
        d = np.array([x for x in deltas if x is not None])
        q = np.array([x for x in reading["kl"][lane] if x is not None])
        if len(d):
            lanes[lane] = {"decisions": len(d), "kl": float(q.mean()), "half_mean_sq": float(0.5 * (d * d).mean()),
                           "sum_delta": float(d.sum()), "max_abs": float(np.abs(d).max())}
    marks = [m for m in (1, 5, 10, 20, 40, 80, 160, 320) if m < reading["decisions"]] + [reading["decisions"] - 1]
    gaps = {lane: {str(m): g[m] for m in marks if g[m] is not None} for lane, g in reading["gap"].items()}
    replay = np.array(reading["logged"]) - np.array(reading["base_logp"])
    return {"lanes": lanes, "gap": gaps, "replay_half_mean_sq": float(0.5 * (replay * replay).mean()),
            "base_write_median": float(np.median(reading["base_write"]))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sides", nargs="+", required=True, help="held-out game-sides, each <game>:<side>")
    ap.add_argument("--at", type=int, default=10, help="the decision where transplant and noise start")
    ap.add_argument("--seed", type=int, default=20261110, help="the noise lane's")
    ap.add_argument("--scales", type=float, nargs="+",
                    help="read these multiples of stage C's step, each on its own memory, in place of the lanes")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    try:
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    except Exception:                                   # a reading, not a run: priority is best effort
        pass
    import torch
    torch.set_num_threads(a.threads)
    from tools.pg_stage_c import load_pass
    from trunk_kl_by_module import STAGE_C, context_on
    from wesnoth_ai.pg_step import direction

    t0 = time.time()
    wanted = [(text.rsplit(":", 1)[0], int(text.rsplit(":", 1)[1])) for text in a.sides]
    p = load_pass(STAGE_C / "pass.pt")
    s = json.loads((STAGE_C / "stage_c.json").read_text(encoding="utf-8"))["arms"]["reduced_adamw"]["s"]
    d = direction(p, "reduced_adamw")
    memory_names = sorted(n for n in d if n.startswith(MEMORY_PREFIX))
    if not memory_names:
        raise ValueError(f"the direction moves no {MEMORY_PREFIX}* parameter")
    context = context_on(list(dict.fromkeys(game for game, _ in wanted)))
    lanes = scale_lanes(a.scales) if a.scales else LANES

    def network(net: str):
        if net == "memory":
            return context.network({n: d[n] for n in memory_names}, s)
        if net == "rest":
            return context.network({n: t for n, t in d.items() if not n.startswith(MEMORY_PREFIX)}, s)
        multiple = {"k": 0.0, "step": 1.0, "half": 0.5}.get(net)
        return context.network(d, s * (float(net[1:]) if multiple is None else multiple))    # "x<m>": m times the step

    nets = {net: network(net) for net in dict.fromkeys(lane[1] for lane in lanes)}
    out = {"s": s, "at": a.at, "seed": a.seed, "memory_parameters": memory_names,
           "lanes": [list(lane) for lane in lanes], "sides": {}}
    if a.out.exists():                                  # a stopped reading goes on from the sides it finished
        before = json.loads(a.out.read_text(encoding="utf-8"))
        if all(before.get(key) == out[key] for key in ("s", "at", "seed", "lanes")):
            out["sides"] = {name: side for name, side in before["sides"].items() if "summary" in side}

    def write():
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")

    by_name = {(gs.game, gs.side): gs for gs in context.sides}
    for game, side in wanted:
        name = f"{game}:{side}"
        if name in out["sides"]:
            continue
        gs = by_name[(game, side)]
        generator = torch.Generator().manual_seed(a.seed)

        def partial(reading, name=name):
            out["sides"][name] = {"R": gs.R, "reading": reading}
            write()
            print(f"{time.time() - t0:7.0f} s  {name} at decision {len(reading['logged'])}", flush=True)

        reading = walk(context, nets, lanes, gs, a.at, generator, partial)
        out["sides"][name] = {"R": gs.R, "summary": summary(reading), "reading": reading}
        write()
        print(f"{time.time() - t0:7.0f} s  {name}", json.dumps(out["sides"][name]["summary"], indent=1), flush=True)
    context.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
