"""The bias fold of WesnothModel._run_hip_fp16_trunk (README.md beside this
file): the loop before it against the loop after it, on `own2` and the 200
bench positions, in one process.

The old loop is 041bd7ca's _run_hip_fp16_trunk, verbatim, on that commit's
kernels (wesnoth_ai/hip_kernels.py read from git at --old-ref and built
beside the new one); the new loop is the checked-out model's. Three parts:

  accuracy  each loop's logits against the padded fp32 trunk over the real
            slots (the loops in chunks of 40, the reference in chunks of
            --ref-chunk): the largest difference per head, actor and target
            argmax agreement; and old against new.
  forward   forward_padded at fp16 for each loop on the same batch,
            interleaved batch by batch (the order swapped every round),
            batches of --batches sizes from the positions: per batch and
            round the new time over the old, their median; the trunk alone
            the same way, on the trunk input captured from each batch.
  kernels   one layer's calls at layer 0's shapes on the first batch's
            tokens, interleaved call by call (CUDA events, order reversed
            every other round): the four GEMMs with and without their bias,
            the attention (old, new without a bias, new with one), the add
            and norm (old, new without, new with), F.relu against the bias
            and ReLU; the medians, and one layer's sum for each loop.

Every forward ends with a sync, as the server's batches do, and the
process's GPU memory is capped (--memory-fraction): built for a GPU that
a serving run shares (README, "Two display-driver resets").

    set HIP_VISIBLE_DEVICES=1
    python bias_fold_ab.py --root <repo> --checkpoint <own2.pt> \
        --dataset <replays_dataset_imitation> --out ab.json [--rounds 12] [--kernel-rounds 40]
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

FIELDS = ("actor_logits", "type_logits", "target_logits", "weapon_logits", "value_logits")


def load_old_kernels(root: Path, ref: str):
    """wesnoth_ai/hip_kernels.py at `ref`, imported as its own module (it
    builds its own code object on the device)."""
    src = subprocess.run(["git", "-C", str(root), "show", f"{ref}:wesnoth_ai/hip_kernels.py"],
                         check=True, capture_output=True, text=True).stdout
    mod = types.ModuleType(f"hip_kernels_{ref}")
    mod.__file__ = f"<git {ref}:wesnoth_ai/hip_kernels.py>"
    sys.modules[mod.__name__] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def make_old_trunk(old):
    """041bd7ca's WesnothModel._run_hip_fp16_trunk, on `old`'s kernels."""
    import torch
    import torch.nn.functional as F

    def old_trunk(self, x, index):
        w = self._weight_copy(torch.float16, torch.float32, x.device)
        total = x.shape[0]
        with torch.autocast("cuda", enabled=False):
            h32 = x.float()
            h16 = h32.half()
            for layer, (qkv_w, qkv_b, out_w, out_b, ln1_w, ln1_b, ff1_w, ff1_b, ff2_w, ff2_b,
                        ln2_w, ln2_b) in zip(self.encoder.layers, w.layers):
                _, heads, head_dim, E = qkv_w.shape
                qkv = F.linear(h16, qkv_w.view(3 * E, E), None if qkv_b is None else qkv_b.view(3 * E))
                q, k, v = qkv.view(total, 3, heads, head_dim).unbind(1)
                ctx = old.hip_varlen_attention(q, k, v, index.cu_seqlens, index.max_len)
                sa = F.linear(ctx.view(total, E), out_w, out_b)
                h32, h16 = old.hip_add_layer_norm(h32, sa, ln1_w, ln1_b, layer.norm1.eps)
                ff = F.linear(layer.activation(F.linear(h16, ff1_w, ff1_b)), ff2_w, ff2_b)
                h32, h16 = old.hip_add_layer_norm(h32, ff, ln2_w, ln2_b, layer.norm2.eps)
            if w.final_norm is not None:
                h32 = F.layer_norm(h32, (h32.shape[-1],), w.final_norm[0], w.final_norm[1],
                                   self.encoder.norm.eps)
        return h32
    return old_trunk


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--old-ref", default="041bd7ca")
    ap.add_argument("--batches", default="48,24")
    ap.add_argument("--rounds", type=int, default=12, help="forward rounds (each: every batch, both loops)")
    ap.add_argument("--kernel-rounds", type=int, default=40)
    ap.add_argument("--parts", default="accuracy,forward,kernels")
    ap.add_argument("--ref-chunk", type=int, default=4, help="positions a padded fp32 reference forward")
    ap.add_argument("--memory-fraction", type=float, default=0.25,
                    help="cap on this process's share of the GPU memory (another process may be serving)")
    args = ap.parse_args()
    sys.path.insert(0, str(args.root))
    import torch
    import torch.nn.functional as F

    import wesnoth_ai.hip_elementwise as he
    import wesnoth_ai.hip_kernels as hk
    from tools.bench_states import load_states
    from tools.eval_players import _load_policy

    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, torch.cuda.current_device())
    parts = set(args.parts.split(","))
    old = load_old_kernels(args.root, args.old_ref)
    old_trunk = make_old_trunk(old)
    states = [gs for gs, _ in load_states(args.root / "configs" / "bench_states.json", args.dataset, None)]
    policy = _load_policy(args.checkpoint, dev, label="own2", infer_bf16=True)
    model, enc = policy._inference_model, policy._inference_encoder
    model.infer_autocast_dtype = torch.float16
    model.infer_packed_trunk = True
    model.eval()
    layer0 = model.encoder.layers[0]
    rec = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "hip": torch.version.hip,
           "old_ref": args.old_ref, "positions": len(states), "memory_slots": model.memory_slots,
           "activation_is_relu": layer0.activation is F.relu,
           "bias_fold_served": hk.hip_varlen_bias_applies(dev, model.d_model // layer0.self_attn.num_heads)}
    with torch.no_grad():
        encs = [enc.encode(gs) for gs in states]
    k = model.memory_slots
    mems = [model.initial_memory(k).to(dev) for _ in encs]
    new_trunk = type(model)._run_hip_fp16_trunk

    def use(trunk):
        model._run_hip_fp16_trunk = types.MethodType(trunk, model)

    def fwd(lo, hi, low=True):
        """One forward, then a sync, as the server syncs each batch. The
        packed forward issues no host sync: forwards queued back to back
        built up GPU work that, beside the stage 1 server, reset the display
        driver twice (2026-10-29 14:06 and 14:18, README)."""
        with torch.no_grad():
            if low:
                out = model.forward_padded(encs[lo:hi], memory=mems[lo:hi])
            else:
                out = model.forward_padded(encs[lo:hi], autocast_bf16=False, packed=False, memory=mems[lo:hi])
        torch.cuda.synchronize()
        return out

    if "accuracy" in parts:
        def samples(runs):
            """Each run's per-sample heads, float32 on the host."""
            out = {}
            for name, lo_hi, low in runs:
                if low:
                    use(name_to_trunk[name])
                rows = []
                for lo, hi in lo_hi:
                    rows += [{f: getattr(s, f).float().cpu() for f in FIELDS} for s in fwd(lo, hi, low).samples()]
                out[name] = rows
            return out

        def compare(got, ref):
            worst = {f: 0.0 for f in FIELDS}
            agree_a = agree_t = n_t = 0
            for x, y in zip(got, ref):
                for f in FIELDS:
                    a, b = x[f], y[f]
                    fin = torch.isfinite(b)
                    if not torch.equal(fin, torch.isfinite(a)):
                        raise RuntimeError(f"finiteness differs on {f}")
                    if fin.any():
                        worst[f] = max(worst[f], (a - b).abs()[fin].max().item())
                agree_a += int(x["actor_logits"].argmax() == y["actor_logits"].argmax())
                ta = x["target_logits"].masked_fill(~torch.isfinite(x["target_logits"]), -1e30)
                tb = y["target_logits"].masked_fill(~torch.isfinite(y["target_logits"]), -1e30)
                if tb.numel():           # one argmax per actor row
                    am_a, am_b = ta.argmax(-1), tb.argmax(-1)
                    agree_t += int((am_a == am_b).sum())
                    n_t += am_b.numel()
            return {"max_abs": {f: round(v, 5) for f, v in worst.items()}, "actor_argmax": agree_a,
                    "target_argmax": agree_t, "targets": n_t}

        name_to_trunk = {"old": old_trunk, "new": new_trunk}
        n = len(encs)
        # The padded fp32 reference in small chunks: its [B, heads, L, L] scores
        # at 40 positions held GBs beside a running server (see README).
        outs = samples([("ref", [(lo, min(lo + args.ref_chunk, n)) for lo in range(0, n, args.ref_chunk)], False),
                        ("old", [(lo, min(lo + 40, n)) for lo in range(0, n, 40)], True),
                        ("new", [(lo, min(lo + 40, n)) for lo in range(0, n, 40)], True)])
        use(new_trunk)
        torch.cuda.empty_cache()
        acc = {name: compare(outs[a], outs[b]) for name, (a, b) in
               (("old_vs_fp32", ("old", "ref")), ("new_vs_fp32", ("new", "ref")), ("new_vs_old", ("new", "old")))}
        rec["accuracy"] = acc
        print(json.dumps(acc, indent=1), "max reserved MB", torch.cuda.max_memory_reserved() >> 20, flush=True)

    sizes = [int(b) for b in args.batches.split(",")]
    if "forward" in parts:
        captured = {}

        def capture(self, x, index):
            captured["x"], captured["index"] = x, index
            return new_trunk(self, x, index)

        rec["forward"] = {}
        for B in sizes:
            spans = [(lo, lo + B) for lo in range(0, len(encs) - B + 1, B)]
            inputs = []
            for lo, hi in spans:          # the trunk's input per batch, and a warm-up of both loops
                use(capture)
                fwd(lo, hi)
                inputs.append((captured["x"], captured["index"]))
                for trunk in (old_trunk, new_trunk):
                    use(trunk)
                    fwd(lo, hi)
            whole, trunk_only = [], []
            for r in range(args.rounds):
                for i, (lo, hi) in enumerate(spans):
                    order = (old_trunk, new_trunk) if (r + i) % 2 == 0 else (new_trunk, old_trunk)
                    t = {}
                    for trunk in order:
                        use(trunk)
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        fwd(lo, hi)
                        torch.cuda.synchronize()
                        t[trunk] = time.perf_counter() - t0
                    whole.append((t[old_trunk], t[new_trunk]))
                    x, index = inputs[i]
                    t = {}
                    for trunk in order:
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                            trunk(model, x, index)
                        torch.cuda.synchronize()
                        t[trunk] = time.perf_counter() - t0
                    trunk_only.append((t[old_trunk], t[new_trunk]))

            def summary(pairs):
                return {"pairs": len(pairs),
                        "old_ms_median": round(statistics.median(p[0] for p in pairs) * 1e3, 2),
                        "new_ms_median": round(statistics.median(p[1] for p in pairs) * 1e3, 2),
                        "new_over_old_median": round(statistics.median(p[1] / p[0] for p in pairs), 4),
                        "new_over_old_q1_q3": [round(q, 4) for q in
                                               statistics.quantiles([p[1] / p[0] for p in pairs], n=4)[::2]],
                        "old_positions_per_s": round(B / statistics.median(p[0] for p in pairs), 1),
                        "new_positions_per_s": round(B / statistics.median(p[1] for p in pairs), 1)}
            rec["forward"][f"B{B}"] = {"batches": len(spans), "tokens_first_batch": int(inputs[0][0].shape[0]),
                                       "forward": summary(whole), "trunk": summary(trunk_only)}
            print(B, json.dumps(rec["forward"][f"B{B}"]), "max reserved MB",
                  torch.cuda.max_memory_reserved() >> 20, flush=True)
        use(new_trunk)

    if "kernels" in parts:
        torch.cuda.empty_cache()     # the parts before held blocks near the memory cap
        captured = {}

        def grab(self, x, index):
            captured["x"], captured["index"] = x, index
            return new_trunk(self, x, index)
        use(grab)
        fwd(0, sizes[0])
        use(new_trunk)
        x, index = captured["x"], captured["index"]
        T = x.shape[0]
        w = model._weight_copy(torch.float16, torch.float32, dev)
        (qkv_w, qkv_b, out_w, out_b, ln1_w, ln1_b, ff1_w, ff1_b, ff2_w, ff2_b, ln2_w, ln2_b) = w.layers[0]
        _, heads, head_dim, E = qkv_w.shape
        g = torch.Generator(device=dev).manual_seed(0)
        h16 = torch.randn(T, E, generator=g, device=dev).half()
        h32 = torch.randn(T, E, generator=g, device=dev)
        qkv = torch.nn.functional.linear(h16, qkv_w.view(3 * E, E))
        q, kk, v = qkv.view(T, 3, heads, head_dim).unbind(1)
        f16 = torch.randn(T, ff1_w.shape[0], generator=g, device=dev).half()
        eps = layer0.norm1.eps
        cu, ml = index.cu_seqlens, index.max_len
        calls = {
            "gemm_qkv_bias": lambda: F.linear(h16, qkv_w.view(3 * E, E), qkv_b.view(3 * E)),
            "gemm_qkv": lambda: F.linear(h16, qkv_w.view(3 * E, E)),
            "gemm_out_bias": lambda: F.linear(h16, out_w, out_b),
            "gemm_out": lambda: F.linear(h16, out_w),
            "gemm_ff1_bias": lambda: F.linear(h16, ff1_w, ff1_b),
            "gemm_ff1": lambda: F.linear(h16, ff1_w),
            "gemm_ff2_bias": lambda: F.linear(f16, ff2_w, ff2_b),
            "gemm_ff2": lambda: F.linear(f16, ff2_w),
            "attn_old": lambda: old.hip_varlen_attention(q, kk, v, cu, ml),
            "attn_new": lambda: hk.hip_varlen_attention(q, kk, v, cu, ml),
            "attn_new_bias": lambda: hk.hip_varlen_attention(q, kk, v, cu, ml, bias=qkv_b),
            "add_ln_old": lambda: old.hip_add_layer_norm(h32, h16, ln1_w, ln1_b, eps),
            "add_ln_new": lambda: he.hip_add_layer_norm(h32, h16, ln1_w, ln1_b, eps),
            "add_ln_new_bias": lambda: he.hip_add_layer_norm(h32, h16, ln1_w, ln1_b, eps, y_bias=out_b),
            "relu": lambda: layer0.activation(f16),
            "bias_relu": lambda: he.hip_bias_relu_(f16, ff1_b),
        }
        names = list(calls)
        times = {n: [] for n in names}
        with torch.no_grad():
            for n in names:           # warm-up
                calls[n]()
                torch.cuda.synchronize()
            for r in range(args.kernel_rounds):
                for n in (names if r % 2 == 0 else names[::-1]):
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    calls[n]()
                    b.record()
                    b.synchronize()
                    times[n].append(a.elapsed_time(b))
        med = {n: round(statistics.median(times[n]), 4) for n in names}
        old_layer = (med["gemm_qkv_bias"] + med["gemm_out_bias"] + med["gemm_ff1_bias"] + med["gemm_ff2_bias"]
                     + med["attn_old"] + 2 * med["add_ln_old"] + med["relu"])
        new_layer = (med["gemm_qkv"] + med["gemm_out"] + med["gemm_ff1"] + med["gemm_ff2"]
                     + med["attn_new_bias"] + 2 * med["add_ln_new_bias"] + med["bias_relu"])
        rec["kernels"] = {"tokens": int(T), "rounds": args.kernel_rounds, "median_ms": med,
                          "min_ms": {n: round(min(times[n]), 4) for n in names},
                          "layer_old_ms": round(old_layer, 3), "layer_new_ms": round(new_layer, 3),
                          "layer_new_over_old": round(new_layer / old_layer, 4)}
        print(json.dumps(rec["kernels"], indent=1), flush=True)
    rec["max_memory_reserved_mb"] = torch.cuda.max_memory_reserved() >> 20
    args.out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
