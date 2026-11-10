"""Which weights of a trunk step open the memory's gap (CPU, float32; Idle
priority, 4 threads beside a run).

trunk_memory_mode.py found that the pass's direction opens a gap along one
pattern of the memory, 68 times its own noise's, and that theta_k's write then
grows it. This reading takes the step one group of weights at a time
(trunk_kl_by_module.group_of, the memory's own tensors apart): theta_k + s d
with d kept on the group alone, each network on its own memory along a side.
At each mark it reads the group's gap over |M|, its cosine with the whole
step's gap, its part of the whole step's gap (the parts sum to 1 while the
response is linear) and, with --pattern, the cosine of its mean over the slots
with that vector.

    python scripts/window_20261029/trunk_memory_groups.py --sides game_own2a_own2b_s2_827637:2 --out training/metrics/trunk_memory_groups_20261110.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

MARKS = (0, 1, 5, 10, 20, 40)    # decisions written before a reading
WHOLE = "all"


def memory_group(name: str, head_of) -> str:
    """trunk_kl_by_module's group of parameter `name`, the memory's tensors
    (initial, slot_embed, gate, candidate) each apart."""
    from trunk_kl_by_module import group_of
    group = group_of(name, head_of)
    if group != "slot_memory":
        return group
    bare = name[len("model."):] if name.startswith("model.") else name
    return "memory." + bare.split(".")[1]


def walk(context, nets, gs, pattern=None) -> dict:
    """Per mark, each network's gap from theta_k's memory along game-side `gs`."""
    import torch
    from trunk_memory_loop import common_share
    from wesnoth_ai.memory import MemoryBatch

    positions = context.loader.side(gs.game, gs.side)
    k, device = context.config.slots, context.device
    memories = {name: model.initial_memory(k).detach().to(device) for name, (_, model) in nets.items()}
    out = {}

    def read():
        base = memories["k"]
        whole = memories[WHOLE] - base
        total = torch.zeros_like(whole)
        row = {}
        for name in nets:
            if name == "k":
                continue
            gap = memories[name] - base
            if name != WHOLE:
                total += gap
            size = float(gap.norm())
            row[name] = {"size": size / float(base.norm()),
                         "cos": float((gap * whole).sum() / (gap.norm() * whole.norm())) if size else 0.0,
                         "part": float((gap * whole).sum() / (whole * whole).sum()),
                         "common": common_share(gap) if size else 0.0}
            if pattern is not None and size:
                mean = gap.mean(0)
                row[name]["on_pattern"] = float((mean * pattern).sum() / mean.norm())
        row["sum_off_whole"] = float((total - whole).norm() / whole.norm())
        return row

    with torch.no_grad():
        for t, p in enumerate(positions[:max(MARKS)]):
            if t in MARKS and not torch.equal(memories[WHOLE], memories["k"]):
                out[str(t)] = read()
            if p.shadow:                                                  # a refused row writes nothing
                continue
            staged = context.encoder.stage_raws([p.raw], device=device)
            for name, (encoder, model) in nets.items():
                padded = model.forward_embedded(encoder.embed_staged(staged), packed=context.config.packed,
                                                memory=MemoryBatch(padded=memories[name][None], counts=[k]))
                memories[name] = padded.memory_padded[0, :k].clone()
        if len(positions) >= max(MARKS):
            out[str(max(MARKS))] = read()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sides", nargs="+", required=True, help="held-out game-sides, each <game>:<side>")
    ap.add_argument("--scale", type=float, default=1.0, help="the step, in multiples of stage C's")
    ap.add_argument("--pattern", type=Path, help="a .npy unit vector of the memory's width")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    try:
        import psutil
        psutil.Process().nice(psutil.IDLE_PRIORITY_CLASS)
    except Exception as error:                          # a reading, not a run: priority is best effort
        print(f"WARNING: not at Idle priority ({error!r})", flush=True)
    import numpy as np
    import torch
    torch.set_num_threads(a.threads)
    from tools.pg_stage_c import load_pass
    from trunk_kl_by_module import STAGE_C, context_on
    from wesnoth_ai.pg_step import direction, head_of

    t0 = time.time()
    wanted = [(text.rsplit(":", 1)[0], int(text.rsplit(":", 1)[1])) for text in a.sides]
    s = a.scale * json.loads((STAGE_C / "stage_c.json").read_text(encoding="utf-8"))["arms"]["reduced_adamw"]["s"]
    d = direction(load_pass(STAGE_C / "pass.pt"), "reduced_adamw")
    groups = sorted({memory_group(n, head_of) for n, t in d.items() if float(t.norm()) > 0})
    context = context_on(list(dict.fromkeys(game for game, _ in wanted)))
    nets = {"k": context.network(d, 0.0), WHOLE: context.network(d, s)}
    for group in groups:
        nets[group] = context.network({n: t for n, t in d.items() if memory_group(n, head_of) == group}, s)
    pattern = torch.from_numpy(np.load(a.pattern)).float() if a.pattern else None
    out = {"s": s, "marks": list(MARKS), "groups": groups,
           "norms": {group: float(sum(float(t.norm()) ** 2 for n, t in d.items()
                                      if memory_group(n, head_of) == group) ** 0.5) for group in groups},
           "sides": {}}
    by_name = {(gs.game, gs.side): gs for gs in context.sides}
    for game, side in wanted:
        out["sides"][f"{game}:{side}"] = walk(context, nets, by_name[(game, side)], pattern)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"{time.time() - t0:7.0f} s  {game}:{side}", flush=True)
    context.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
