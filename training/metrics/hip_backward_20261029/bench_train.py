"""`own2`'s train step on the local RX 6600: the padded fp32 trunk (the
reading of training/metrics/local_gpu_20261028) against the packed trunk in
fp16 with the HIP attention backward (wesnoth_ai/hip_attention_grad.py);
the record is README.md beside this file. Adapted from
training/metrics/local_gpu_20261028/bench_local.py.

The positions of configs/bench_states.json (rebuilt from the imitation
dataset), each with the 64-slot initial memory, through
`WesnothModel.forward_padded` in train mode: forward, backward and AdamW,
batch TRAIN_B, bench_local's surrogate loss (the squared finite logits of
every head), the states encoded once outside the graph. Modes:

  fp32_padded   no autocast, the padded trunk (math SDPA): the baseline
  fp16_packed   fp16 autocast, the packed trunk (the HIP attention forward
                and backward), a GradScaler

Each rate is read after --warmup batches over --reps passes of the first
--states positions (all 200 by default), without a synchronization inside
the pass; then --phase-batches more batches with one between the forward,
the backward and the step give each one's mean time.

`--check` also compares parameter gradients on the first CHECK_B positions
in eval mode (no dropout) with the gradient wanted, of a bounded loss (each
head's log-sum-exp over each sample's real slots: the trunks fill the
padded slots differently, and the squared logits overflow fp16), against
the fp32 padded trunk's: the fp16 packed trunk with the HIP backward, the
same with fp16 SDPA in its place, the fp16 padded trunk, and each path run
twice. Per comparison the cosine over all parameters, the relative norm of
the difference, and per tensor the largest error over the reference's
largest magnitude.

`--attention` times one layer's attention alone at the 2026-10-28 reference
problem (48 segments of 141-692 tokens, 12 heads of 32): the inference
kernel, the forward with its log-sum-exp, and the backward, each the best
of ATTENTION_ROUNDS interleaved rounds of ten calls.

    set HIP_VISIBLE_DEVICES=1
    python bench_train.py --root <repo> --dataset <replays_dataset_imitation> \
        --modes fp32_padded fp16_packed --out rates.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

TRAIN_B = 32
CHECK_B = 8
# Powers of two, so unscaling is exact; the largest that does not overflow
# keeps the fp16 backward's small gradients above fp16's subnormals (the
# check runs without a GradScaler).
CHECK_LOSS_SCALES = (1024.0, 128.0, 16.0, 1.0)
ATTENTION_ROUNDS = 5
FIELDS = ("actor_logits", "type_logits", "target_logits", "weapon_logits", "value_logits")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--modes", nargs="+", default=["fp32_padded", "fp16_packed"],
                    choices=("fp32_padded", "fp16_packed"))
    ap.add_argument("--states", type=int, default=None, help="the first N bench positions (default all)")
    ap.add_argument("--warmup", type=int, default=3, help="untimed batches before each rate")
    ap.add_argument("--reps", type=int, default=3, help="timed passes over the positions")
    ap.add_argument("--phase-batches", type=int, default=3, help="batches timed phase by phase")
    ap.add_argument("--check", action="store_true", help="compare the two modes' gradients")
    ap.add_argument("--attention", action="store_true", help="time one layer's attention alone")
    args = ap.parse_args()
    sys.path.insert(0, str(args.root))
    import torch
    from tools.bench_states import load_states
    from tools.eval_players import _load_policy

    device = torch.device("cuda")
    states = [gs for gs, _ in load_states(args.root / "configs" / "bench_states.json", args.dataset,
                                          args.states)]
    policy = _load_policy(args.root / "training" / "checkpoints" / "own2.pt", device, label="own2")
    encoder, model = policy._inference_encoder, policy._inference_model
    with torch.no_grad():        # the encoder's embedding lookups stay out of the timed graph
        encs = [encoder.encode(gs) for gs in states]
    k = model.memory_slots
    rec = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "hip": getattr(torch.version, "hip", None), "positions": len(encs), "train_batch": TRAIN_B,
           "warmup_batches": args.warmup, "reps": args.reps, "memory_slots": k}

    def sync():
        torch.cuda.synchronize()

    def batches(b):
        for i in range(0, len(encs) - b + 1, b):
            chunk = encs[i:i + b]
            yield chunk, [model.initial_memory(k).to(device) for _ in chunk]

    def forward(mode, chunk, mem):
        if mode == "fp32_padded":
            with torch.autocast("cuda", enabled=False):
                return surrogate_of(torch, model.forward_padded(chunk, autocast_bf16=False, memory=mem,
                                                                packed=False))
        with torch.autocast("cuda", dtype=torch.float16):
            return surrogate_of(torch, model.forward_padded(chunk, autocast_bf16=False, memory=mem, packed=True))

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-9)
    scaler = torch.amp.GradScaler("cuda")

    def step(mode, chunk, mem, phases=None):
        t0 = time.perf_counter()
        loss = forward(mode, chunk, mem)
        if phases is not None:
            sync()
            t1 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        if mode == "fp16_packed":
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if phases is not None:
            sync()
            t2 = time.perf_counter()
        if mode == "fp16_packed":
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        if phases is not None:
            sync()
            t3 = time.perf_counter()
            phases.append((t1 - t0, t2 - t1, t3 - t2))

    rates, phase_ms = {}, {}
    for mode in args.modes:
        for _ in range(args.warmup):
            step(mode, *next(batches(TRAIN_B)))
        sync()
        n, t0 = 0, time.perf_counter()
        for _ in range(args.reps):
            for chunk, mem in batches(TRAIN_B):
                step(mode, chunk, mem)
                n += len(chunk)
        sync()
        rates[mode] = round(n / (time.perf_counter() - t0), 2)
        phases = []
        for (chunk, mem), _ in zip(batches(TRAIN_B), range(args.phase_batches)):
            step(mode, chunk, mem, phases)
        phase_ms[mode] = {name: round(1000 * sum(p[i] for p in phases) / len(phases), 1)
                          for i, name in enumerate(("forward", "backward", "optimizer"))}
        print(mode, rates[mode], phase_ms[mode], flush=True)
    rec["positions_per_s"] = rates
    rec["phase_ms_per_batch"] = phase_ms
    rec["grad_scaler_scale"] = float(scaler.get_scale()) if "fp16_packed" in args.modes else None

    if args.check:
        rec["gradient_check"] = gradient_check(torch, model, encs[:CHECK_B], k, device)
    if args.attention:
        rec["attention_one_layer"] = attention_times(torch, device)
    args.out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rec, indent=1))
    return 0


def gradient_check(torch, model, encs, k, device):
    """Parameter gradients on `encs` in eval mode (no dropout), the gradient
    wanted, against the fp32 padded trunk's: the fp16 packed trunk with the
    HIP backward; to tell its attention's rounding from fp16 autocast's,
    the same packed trunk with the per-segment fp16 SDPA (math) in its
    place, and the padded trunk under the same fp16 autocast; and the HIP
    trunk against the SDPA one."""
    import wesnoth_ai.packed_trunk as pt
    model.eval()
    mem = [model.initial_memory(k).to(device) for _ in encs]

    scales = {}

    def grads(mode, hip=True):
        """The gradients of a cross-entropy-like loss, each field's
        log-sum-exp over each sample's real slots (a softmax for gradient;
        the two trunks fill the padded slots differently), at the largest
        of CHECK_LOSS_SCALES that does not overflow."""
        saved = pt._hip_trains
        if not hip:
            pt._hip_trains = lambda q, k, v: False
        try:
            for scale in CHECK_LOSS_SCALES:
                model.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=mode != "fp32_padded"):
                    out = model.forward_padded(encs, autocast_bf16=False, memory=mem,
                                               packed=mode == "fp16_packed")
                loss = 0.0
                for x in out.float32().samples():
                    for f in FIELDS:
                        t = getattr(x, f)
                        loss = loss + torch.logsumexp(t[torch.isfinite(t)], dim=0)
                (loss * scale).backward()
                grads = {n: p.grad.detach().float() / scale for n, p in model.named_parameters()
                         if p.grad is not None}
                if all(torch.isfinite(t).all() for t in grads.values()):
                    scales[f"{mode}{'' if hip else '_sdpa'}"] = scale
                    return grads
        finally:
            pt._hip_trains = saved
        raise RuntimeError(f"{mode}: a gradient overflows at every scale of {CHECK_LOSS_SCALES}")

    def compare(got, ref):
        rel = {n: ((got[n] - ref[n]).abs().max() / ref[n].abs().max().clamp_min(1e-30)).item() for n in ref}
        flat_r = torch.cat([ref[n].flatten() for n in ref])
        flat_g = torch.cat([got[n].flatten() for n in ref])
        trunk = [v for n, v in rel.items() if n.startswith("encoder.layers")]
        worst = sorted(rel.items(), key=lambda kv: -kv[1])[:3]
        return {"cosine": round(torch.nn.functional.cosine_similarity(flat_r, flat_g, dim=0).item(), 6),
                "relative_norm_error": round(((flat_g - flat_r).norm() / flat_r.norm()).item(), 5),
                "trunk_rel_max": round(max(trunk), 5) if trunk else None,
                "median_rel": round(sorted(rel.values())[len(rel) // 2], 5),
                "worst_rel": {n: round(v, 5) for n, v in worst}}
    ref = grads("fp32_padded")
    hip = grads("fp16_packed")
    hip_again = grads("fp16_packed")
    sdpa = grads("fp16_packed", hip=False)
    padded16 = grads("fp16_padded")
    ref_again = grads("fp32_padded")
    model.train()
    model.zero_grad(set_to_none=True)
    return {"positions": len(encs), "tensors": len(ref), "loss_scales": scales,
            "fp16_packed_hip_vs_fp32_padded": compare(hip, ref),
            "fp16_packed_sdpa_vs_fp32_padded": compare(sdpa, ref),
            "fp16_padded_vs_fp32_padded": compare(padded16, ref),
            "fp16_packed_hip_vs_fp16_packed_sdpa": compare(hip, sdpa),
            "fp16_packed_hip_vs_itself_again": compare(hip_again, hip),
            "fp32_padded_vs_itself_again": compare(ref_again, ref)}


def surrogate_of(torch, out):
    loss = 0.0
    for t in (out.actor_logits, out.type_logits, out.target_logits, out.weapon_logits, out.value_logits):
        t = t.float()
        loss = loss + torch.where(torch.isfinite(t), t, torch.zeros_like(t)).pow(2).mean()
    return loss


def attention_times(torch, device):
    """One layer's attention, forward with its log-sum-exp and backward, at
    the 2026-10-28 reference problem; mean ms over 10 runs after 3."""
    from wesnoth_ai.hip_attention_grad import hip_varlen_attention_trainable
    rng = random.Random(0)
    lens = [rng.randint(141, 692) for _ in range(48)]
    cu_host = [0]
    for n in lens:
        cu_host.append(cu_host[-1] + n)
    cu = torch.tensor(cu_host, dtype=torch.int32, device=device)
    qkv = torch.randn(cu_host[-1], 3, 12, 32, device=device, dtype=torch.float16)
    q, k, v = (t.detach().requires_grad_(True) for t in qkv.unbind(1))
    g = torch.randn(q.shape, device=device, dtype=torch.float16)

    def timed(fn, reps=10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return 1000 * (time.perf_counter() - t0) / reps
    from wesnoth_ai.hip_kernels import hip_varlen_attention

    def inference_call():
        with torch.no_grad():
            hip_varlen_attention(q, k, v, cu, max(lens))
    calls = {"inference": inference_call,
             "forward": lambda: hip_varlen_attention_trainable(q, k, v, cu, max(lens), rows=cu_host[-1]),
             "both": lambda: torch.autograd.backward(
                 hip_varlen_attention_trainable(q, k, v, cu, max(lens), rows=cu_host[-1]), g)}
    for fn in calls.values():
        fn()
    # The generation run shares the GPU, and one reading moves by 3x from
    # round to round (2026-10-29): the best of ATTENTION_ROUNDS interleaved
    # rounds of ten calls each.
    best = {name: float("inf") for name in calls}
    for _ in range(ATTENTION_ROUNDS):
        for name, fn in calls.items():
            best[name] = min(best[name], timed(fn))
    inference, fwd, both = best["inference"], best["forward"], best["both"]
    flops = sum(4 * n * n * 32 * 12 for n in lens)        # QK^T and PV, two flops a multiply-add
    # The backward's useful work is 2.5 forwards (FlashAttention-2's count: dV, dP, dQ, dK and
    # the recomputed S, over the forward's S and PV); these kernels compute S and dP twice.
    return {"segments": len(lens), "tokens": cu_host[-1], "heads": 12, "head_dim": 32,
            "inference_forward_ms": round(inference, 3),
            "forward_lse_ms": round(fwd, 3), "backward_ms": round(both - fwd, 3),
            "forward_tflops": round(flops / fwd / 1e9, 2), "backward_tflops": round(2.5 * flops / (both - fwd) / 1e9, 2)}


if __name__ == "__main__":
    sys.exit(main())
