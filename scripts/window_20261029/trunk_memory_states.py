"""Whose divergence a trunk step's KL is: the weights' or the memory's (CPU,
float32; Idle priority, 4 threads beside a run).

trunk_memory_mode.py found that on a diverging game-side theta_k's own write
grows a small difference of the memory into a second state. This reading
splits the step's divergence at each decision of a side, against theta_k on
its own memory M (the reference):

  full     theta' on its own memory: what stage C's evaluator charges the step
  weights  theta' on M: the step's own change of the policy, memory given
  memory   theta_k on theta''s memory: the memory's difference alone
  free     theta_k on a memory seeded from theta''s at decision --seed-at and
           written by theta_k since: what the reference does from there alone

and each divergence by the law that moved (pg_loss.LAW_KEYS; the chain rule's
terms, which sum to the divergence), with the lanes' agreement on the most
likely actor, the actor law's entropy, the value, and the log probability each
lane gives the logged action. --arrays saves each lane's memory along the side,
as its mean over the slots.

    python scripts/window_20261029/trunk_memory_states.py --sides game_own2a_own2b_s2_827637:2 --out training/metrics/trunk_memory_states_20261110.json
"""
import argparse
import collections
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

LANES = ("full", "weights", "memory", "free")


def walk(context, nets, gs, partial, seed_at, limit=None):
    """The per-decision columns along game-side `gs` (module docstring), and
    the lanes' memories as their means over the slots."""
    import torch
    from tools.pg_pass import stage_step
    from trunk_memory_loop import PARTIAL_EVERY
    from wesnoth_ai.constants import TURN_NORM
    from wesnoth_ai.memory import MemoryBatch
    from wesnoth_ai.pg_loss import LAW_KEYS, action_log_probs, factor_laws, legal_kl_laws

    positions = context.loader.side(gs.game, gs.side)
    if len(positions) != gs.stop - gs.start:
        raise ValueError(f"{gs.game} side {gs.side}: {len(positions)} positions for {gs.stop - gs.start} decisions")
    k, device = context.config.slots, context.device
    M, full = (nets[net][1].initial_memory(k).to(device) for net in ("k", "step"))
    free = None
    cols, arrays = collections.defaultdict(list), collections.defaultdict(list)

    def forward(net, memories, p):
        encoder, model = nets[net]
        n = memories.size(0)
        staged = context.encoder.stage_raws([p.raw] * n, device=device)
        batch = stage_step([p] * n, staged, context.model, device)
        padded = model.forward_embedded(encoder.embed_staged(staged), packed=context.config.packed,
                                        memory=MemoryBatch(padded=memories, counts=[k] * n))
        return (padded.memory_padded[:, :k].clone(), factor_laws(padded, batch.views, batch.rows),
                padded.value.float()[:, 0], action_log_probs(padded, batch))

    def row(laws, i):
        return {key: getattr(laws, key)[i:i + 1] for key in LAW_KEYS}

    def entropy(laws, i):
        return float(-torch.where(laws.actor_ok[i], laws.actor[i].exp() * laws.actor[i],
                                  torch.zeros_like(laws.actor[i])).sum())

    def top(laws, i):
        return int(laws.actor[i].masked_fill(~laws.actor_ok[i], float("-inf")).argmax())

    with torch.no_grad():
        for t, p in enumerate(positions[:limit]):
            if t == seed_at:
                free = full.clone()
            norm = M.norm()
            read = torch.stack([M, full] + ([free] if free is not None else []))
            wrote_k, laws_k, value_k, logp_k = forward("k", read, p)
            wrote_s, laws_s, value_s, logp_s = forward("step", torch.stack([full, M]), p)
            P = dataclasses.replace(laws_k, **{key: getattr(laws_k, key)[:1] for key in LAW_KEYS},
                                    **{key + "_ok": getattr(laws_k, key + "_ok")[:1] for key in LAW_KEYS},
                                    slot=laws_k.slot[:1])
            lanes = {"full": (laws_s, 0, value_s, logp_s, full), "weights": (laws_s, 1, value_s, logp_s, M),
                     "memory": (laws_k, 1, value_k, logp_k, full)}
            if free is not None:
                lanes["free"] = (laws_k, 2, value_k, logp_k, free)

            for name, memory in (("memory_mean", M), ("full_mean", full), ("free_mean", free)):
                arrays[name].append((M if memory is None else memory).mean(0).cpu().numpy())
            cols["shadow"].append(bool(p.shadow))
            cols["kind"].append(int(p.action[1]))
            cols["turn"].append(round(float(p.raw.global_feats[0]) * TURN_NORM, 2))
            cols["value"].append(float(value_k[0]))
            cols["logp"].append(float(logp_k[0]))
            cols["entropy"].append(entropy(laws_k, 0))
            cols["top_slot"].append(int(laws_k.slot[0, top(laws_k, 0)]))
            for name in LANES:
                if name not in lanes:
                    for key in ("kl", "by_law", "size", "value", "logp", "entropy", "same_top"):
                        cols[f"{name}_{key}"].append(None)
                    continue
                laws, i, value, logp, memory = lanes[name]
                Q = row(laws, i)
                cols[f"{name}_kl"].append(float(legal_kl_laws(P, dataclasses.replace(P, **Q))[0]))
                cols[f"{name}_by_law"].append([float(legal_kl_laws(P, dataclasses.replace(P, **{key: Q[key]}))[0])
                                               for key in LAW_KEYS])
                cols[f"{name}_size"].append(float((memory - M).norm() / norm))
                cols[f"{name}_value"].append(float(value[i] - value_k[0]))
                cols[f"{name}_logp"].append(float(logp[i] - logp_k[0]))
                cols[f"{name}_entropy"].append(entropy(laws, i))
                cols[f"{name}_same_top"].append(top(laws, i) == top(laws_k, 0))
            if not p.shadow:                                              # a refused row writes nothing
                M, full = wrote_k[0], wrote_s[0]
                if free is not None:
                    free = wrote_k[2]
            if (t + 1) % PARTIAL_EVERY == 0:
                partial(cols)
    return dict(cols), dict(arrays)


def summary(cols) -> dict:
    """Each lane's means by stretch of the side (trunk_memory_loop.SEGMENTS)."""
    import numpy as np
    from trunk_memory_loop import SEGMENTS
    live = ~np.array(cols["shadow"])
    T = len(live)
    out = {"decisions": T, "refused": int((~live).sum()), "segments": {}}
    for a, b in SEGMENTS:
        seg = np.zeros(T, bool)
        seg[a:b] = True
        seg &= live
        if not seg.any():
            continue

        def mean(name, seg=seg):
            rows = [x for x, keep in zip(cols[name], seg) if keep and x is not None]
            return np.mean(np.array(rows, float), axis=0).tolist() if rows else None

        out["segments"][f"[{a},{min(b, T)})"] = {
            "decisions": int(seg.sum()), **{name: mean(name) for name in ("value", "logp", "entropy")},
            **{lane: {key: mean(f"{lane}_{key}") for key in ("kl", "by_law", "size", "value", "logp", "entropy",
                                                             "same_top")} for lane in LANES}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sides", nargs="+", required=True, help="held-out game-sides, each <game>:<side>")
    ap.add_argument("--direction", choices=("pass", "noise", "random"), default="pass")
    ap.add_argument("--direction-file", type=Path,
                    help="a saved direction in --direction's place (trunk_memory_project.py --save)")
    ap.add_argument("--scale", type=float, default=1.0, help="the step, in multiples of stage C's")
    ap.add_argument("--seed-at", type=int, default=10, help="the decision that seeds the free lane")
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--limit", type=int, help="read only the first decisions of each side")
    ap.add_argument("--arrays", type=Path, help="a directory for the lanes' memories, one .npz a side")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    try:
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    except Exception:                                   # a reading, not a run: priority is best effort
        pass
    import numpy as np
    import torch
    torch.set_num_threads(a.threads)
    from tools.pg_stage_c import load_pass
    from trunk_kl_by_module import STAGE_C, context_on
    from trunk_memory_loop import step_direction
    from wesnoth_ai.pg_loss import LAW_KEYS

    t0 = time.time()
    wanted = [(text.rsplit(":", 1)[0], int(text.rsplit(":", 1)[1])) for text in a.sides]
    s = a.scale * json.loads((STAGE_C / "stage_c.json").read_text(encoding="utf-8"))["arms"]["reduced_adamw"]["s"]
    if a.direction_file:
        d = torch.load(a.direction_file, map_location="cpu", weights_only=True)
    else:
        d = step_direction(load_pass(STAGE_C / "pass.pt"), a.direction, torch.Generator().manual_seed(a.seed))
    context = context_on(list(dict.fromkeys(game for game, _ in wanted)))
    nets = {"k": context.network(d, 0.0), "step": context.network(d, s)}
    out = {"s": s, "direction": a.direction_file.name if a.direction_file else a.direction, "seed_at": a.seed_at, "seed": a.seed, "limit": a.limit,
           "laws": list(LAW_KEYS), "sides": {}}
    if a.out.exists():                                  # a stopped reading goes on from the sides it finished
        before = json.loads(a.out.read_text(encoding="utf-8"))
        if all(before.get(key) == value for key, value in out.items() if key != "sides"):
            out["sides"] = {name: side for name, side in before["sides"].items() if "summary" in side}

    def write():
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out), encoding="utf-8")

    by_name = {(gs.game, gs.side): gs for gs in context.sides}
    for game, side in wanted:
        name = f"{game}:{side}"
        if name in out["sides"]:
            continue
        gs = by_name[(game, side)]

        def partial(cols, name=name, gs=gs):
            out["sides"][name] = {"R": gs.R, "reading": cols}
            write()
            print(f"{time.time() - t0:7.0f} s  {name} at decision {len(cols['shadow'])}", flush=True)

        cols, arrays = walk(context, nets, gs, partial, a.seed_at, a.limit)
        if a.arrays:
            a.arrays.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(a.arrays / f"{name.replace(':', '_side')}.npz",
                                **{key: np.stack(rows) for key, rows in arrays.items()})
        out["sides"][name] = {"R": gs.R, "summary": summary(cols), "reading": cols}
        write()
        last = list(out["sides"][name]["summary"]["segments"].values())[-1]
        print(f"{time.time() - t0:7.0f} s  {name}", json.dumps({lane: last[lane]["kl"] for lane in LANES}), flush=True)
    context.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
