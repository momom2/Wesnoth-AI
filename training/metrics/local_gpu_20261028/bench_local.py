"""Forward and train-step rates of `own2` on one device, for the local
machine's record (README.md beside this file).

The 200 positions of configs/bench_states.json (rebuilt from the imitation
dataset), each with the 64-slot initial memory, through
`WesnothModel.forward_padded`:
  forward  eval mode, no grad, batch FWD_B, per precision;
  train    train mode, forward + backward + AdamW step, batch TRAIN_B, a
           surrogate loss over every head (the squared finite logits), so
           the backward runs through the trunk and all heads (the states are
           encoded once, outside the graph: the embedding tables' gradient
           is not timed).
Precisions: fp32 everywhere; fp16 and bf16 autocast on cuda. Each rate is
read after --warmup batches over --reps passes of the positions (the first
--states of them; all 200 by default).

    python bench_local.py --root <repo> --dataset <replays_dataset_imitation> \
        --device cuda|cpu --out rates_<device>.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

FWD_B = 64
TRAIN_B = 32


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=0, help="torch threads on cpu (0: torch's default)")
    ap.add_argument("--states", type=int, default=None, help="the first N bench positions (default all)")
    ap.add_argument("--warmup", type=int, default=3, help="untimed batches before each rate")
    ap.add_argument("--reps", type=int, default=3, help="timed passes over the positions")
    args = ap.parse_args()
    sys.path.insert(0, str(args.root))
    import torch
    from tools.bench_states import load_states
    from tools.eval_players import _load_policy

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    cuda = device.type == "cuda"
    states = [gs for gs, _ in load_states(args.root / "configs" / "bench_states.json", args.dataset,
                                          args.states)]
    policy = _load_policy(args.root / "training" / "checkpoints" / "own2.pt", device, label="own2")
    encoder, model = policy._inference_encoder, policy._inference_model
    with torch.no_grad():        # the encoder's embedding lookups stay out of the timed graph
        encs = [encoder.encode(gs) for gs in states]
    k = model.memory_slots

    def sync():
        if cuda:
            torch.cuda.synchronize()

    def batches(b):
        for i in range(0, len(encs) - b + 1, b):
            chunk = encs[i:i + b]
            yield chunk, [model.initial_memory(k).to(device) for _ in chunk]

    def ctx(precision):
        if precision == "fp32":
            return torch.autocast(device.type, enabled=False)
        return torch.autocast(device.type, dtype=getattr(torch, {"fp16": "float16", "bf16": "bfloat16"}[precision]))

    def timed(step, b):
        n = 0
        for _ in range(args.warmup):
            chunk, mem = next(batches(b))
            step(chunk, mem)
        sync()
        t0 = time.perf_counter()
        for _ in range(args.reps):
            for chunk, mem in batches(b):
                step(chunk, mem)
                n += len(chunk)
        sync()
        return n / (time.perf_counter() - t0)

    precisions = ["fp32", "fp16", "bf16"] if cuda else ["fp32"]
    rows = {}
    model.eval()
    for p in precisions:
        def fwd(chunk, mem, p=p):
            with torch.no_grad(), ctx(p):
                model.forward_padded(chunk, autocast_bf16=False, memory=mem)
        rows[f"forward_{p}"] = timed(fwd, FWD_B)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-9)
    scaler = torch.amp.GradScaler("cuda") if cuda else None
    for p in precisions:
        def train(chunk, mem, p=p):
            with ctx(p):
                out = model.forward_padded(chunk, autocast_bf16=False, memory=mem)
                loss = 0.0
                for t in (out.actor_logits, out.type_logits, out.target_logits, out.weapon_logits,
                          out.value_logits):
                    t = t.float()
                    loss = loss + torch.where(torch.isfinite(t), t, torch.zeros_like(t)).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            if p == "fp16":
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
        rows[f"train_{p}"] = timed(train, TRAIN_B)
    rec = {"device": torch.cuda.get_device_name(0) if cuda else platform.processor(),
           "torch": torch.__version__, "hip": getattr(torch.version, "hip", None),
           "threads": torch.get_num_threads(), "positions": len(encs), "warmup_batches": args.warmup,
           "reps": args.reps, "forward_batch": FWD_B, "train_batch": TRAIN_B,
           "positions_per_s": {r: round(v, 2) for r, v in rows.items()}}
    args.out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
