# GPU side of one inference-server batch: what it costs, what each lever buys

Design note, 2026-09-04. Read-only analysis of the code and the recorded
benchmarks; no new measurement was possible (no GPU here, no torch runs
allowed). Every number below is either quoted from a record, derived from
one with the derivation shown, or marked as an estimate to be confirmed by
the protocol in section 9.

Model under discussion: the 15M "tier B" net, `d_model=384, layers=8,
heads=12 (head_dim 32), d_ff=1536` (docs/archive/tier_b_runbook.md:29;
14.2M in the trunk + heads/embeddings ≈ 15M), `nn.TransformerEncoderLayer`
post-norm, ReLU FFN (no `activation=` passed at
wesnoth_ai/model.py:262-266, so the default), `enable_nested_tensor=False`
(model.py:273-275), dropout 1e-4 (identity in eval mode).

## 0. Summary

1. **GPU time per 16-leaf batch at production sizes (~1,270 tokens) is
   about 24 ms; the serve thread's own CPU work is 10-12 ms.** Two
   independent derivations agree (section 1). The server is therefore
   GPU-bound once fed: ceiling ≈ 16/0.024 = **670 leaves/s per 4090 with
   today's kernels**, and the measured 49% GPU utilization at 320 leaves/s
   is exactly this ceiling times the feeding fraction.
2. **In the current pool configuration (19 actors, one 16-leaf request in
   flight each), GPU-side work buys almost nothing**: the actor's cycle
   per request is ~0.95 s of which the server's response is ~0.1 s
   (section 2). Halving the GPU side moves throughput by ~5%. The levers
   here raise the ceiling that the feeding fix (more in-flight requests)
   will hit; measure them by GPU-ms per leaf, not by pool leaves/s.
3. **Physical floor**: 56 GFLOP per leaf at 1,270 tokens (28.3 MFLOP per
   token in the linears + 8 × 4·L²·d attention). At the 4090's 165 TFLOPS
   dense bf16 peak that is 2,950 leaves/s at 100% utilization. **The plan's
   3,000 leaves/s target (docs/plan_20260904.md:94) is not reachable by
   the GPU-side levers at this token count**; today's path runs at 21-27%
   of peak, and everything in this note together reaches ~50-60%, i.e.
   **1,300-1,800 leaves/s**. The rest needs fewer tokens (plan step 1.4)
   or fp8.
4. Ranking by GPU-ms saved per implementation day (section 8):
   (a) packed varlen trunk (no padding, no mask, flash kernel) −20-25%;
   (b) inductor compile of a tensor-only `forward_streams` with native
   bf16 weights, dynamic shapes, no CUDA graphs −15-20% GPU and −50% CPU;
   (c) sync/copy consolidation in `batched_priors` −8-12%;
   (d) length-bucketed batch composition −8-10% at max_batch 16,
   −30-35% at max_batch 64 (fed regime only);
   (e) CUDA graphs with shape buckets: ≈0 on GPU time (padding to buckets
   cancels the launch-gap gain), −60-70% serve-thread CPU; worth it only
   if CPU becomes the limit again after (b), or for the batch-1 eval path.

## 1. What the GPU side costs today

### 1.1 The path (server priors protocol, the production path)

`actor_pool.py:764` → `InferenceServer.infer_batch` → `_infer_with_priors`
(tools/inference_seam.py:253-285):

- `encode_from_raw_padded` (wesnoth_ai/encoder.py:786-817) runs OUTSIDE
  the autocast block (inference_seam.py:261 vs 263): `_embed_streams`
  (encoder.py:819-879) does 15 `_cat_to_dev` calls, each
  `from_numpy → pin_memory → to(device, non_blocking)` (encoder.py:831-838),
  15 embedding/linear kernels, then three `pad_sequence` calls over
  Python lists of `split` views (encoder.py:805-813). Output: fp32
  `[B, H_max, d]`, `[B, U_max, d]`, `[B, R_max, d]`, `[B,1,d]` ×2.
- `forward_streams` (model.py:543-625) under `torch.autocast(bf16)`
  (inference_seam.py:263-264): token-kind adds (560-567), `cat` (568),
  a **host-side numpy loop over B** building the key-padding mask, the
  actor gather index and actor kinds (574-586), two pageable H2D copies
  (587-588), `self.encoder(x, src_key_padding_mask=pad_mask)` (591), one
  `gather` (594), four head linears, one `bmm` for target logits (604),
  value head (607-613). Returns `PaddedOutput`; `.float32()` casts every
  bf16 output back (inference_seam.py:264; model.py:660-667), including
  `target_logits [B, A_max, H_max]`.
- `batched_priors` (wesnoth_ai/server_priors.py:103-222): numpy
  allocation of three `[B, A_max, H_max]` bool masks + optional
  `[B, A_max, H_max]` float32 attack bias (119-131), a per-leaf numpy loop
  with three `np.unpackbits` per leaf (132-146), then **twelve** separate
  `torch.from_numpy(a).to(device)` pageable copies (148-149, 152-175),
  five `masked_fill` + five softmaxes, one boolean-mask `index_put_`
  (`al[is_end] += ...`, line 160 — this is a hidden `nonzero`, i.e. a
  host sync), four explicit `torch.nonzero` (184-187, each a host sync),
  four `stack`s of fancy-index gathers, **eight** `.cpu()` calls (198),
  then numpy lexsort and per-leaf slicing (200-221).
- Back in `_infer_with_priors`: 3-5 more `.cpu()` (268-271), then per
  leaf four `torch.zeros` + a `ModelOutput` + `move_model_output`
  (274-285).

The padding layout is three interior pad runs per row (hex pad, unit pad,
recruit pad; model.py:580-582), not a trailing run. This matters for
sections 4 and 5.

### 1.2 GPU time per batch: two derivations

**From the pool run** (`pool_on_bf16`, docs/box_specs.md:249-278): 320
leaves/s at 49% GPU utilization → GPU busy 0.49 / 320 = **1.53 ms per
leaf = 24.5 ms per 16-leaf batch**. Cross-check with the serve-thread
timers: infer 1,088 s over 287,483 leaves = 60.5 ms per 16-leaf batch per
thread, wait 623 s. Two threads share one GPU and one stream, so a
thread's `infer` stage contains its own CPU (~10-12 ms), its own GPU
(24 ms) and, most of the time, the other thread's GPU work it waits
behind at its first sync (24 ms): 10 + 24 + 24 ≈ 58 ms. GPU busy fraction
= 2 threads × (1,088/1,796 of the time inferring) × (24/60) = 0.49. The
three measured numbers (leaves/s, GPU util, per-thread infer time) are
mutually consistent with 24 ms of GPU per batch and ~10 ms of CPU.

**From the idle-box forward bench** (`fwdbatch_bf16`, box_specs.md:216;
records training/metrics/bench_pipeline/fwdbatch_bf16.md): the batch-4
row is 1.67 ms/sample = 6.7 ms per call at every token bucket, i.e. the
CPU floor of one `forward_batch` call is ~6-7 ms (pad_sequence ×3, the
numpy mask loop, ~130 kernel launches, `samples()` views). Subtracting
it: batch 16 at ≤1,018 tokens (13.8 ms) → ~0.45-0.55 ms GPU per sample;
batch 16 at ≤2,200 tokens (33.4 ms) → ~1.7-1.8 ms; interpolating to
1,270 tokens → ~1.0-1.2 ms per sample for the forward, plus the priors
kernels, the ~9 MB of pageable mask/bias H2D per batch and the small D2H
(~0.2-0.3 ms per leaf) → **1.2-1.5 ms per leaf**. Same answer.

### 1.3 Where the 24 ms go (estimate; the profiler run in section 9 pins it)

At B=16, L=1,300, M = B·L = 20,800 tokens:

| component | FLOP or bytes per batch | estimated ms | basis |
|---|---|---|---|
| linears (QKV 384→1152, out 384→384, FFN 384→1536→384), 8 layers | 2·M·1.77M·8 = 589 GFLOP | 6.5-10 | cuBLAS bf16 on K=384 shapes, 60-90 TFLOPS effective |
| attention, mem-efficient kernel with bias, head_dim 32 | 8 × 16 × 4·L²·d = 332 GFLOP | 5.5-11 | cutlass FMHA at head_dim 32 with an additive bias: 30-60 TFLOPS |
| elementwise on the fp32 residual stream: 4 autocast input casts per layer, 2 residual adds, 2 LayerNorms, ReLU on [M,1536], weight casts | ≈ 0.5 GB read+write per layer → 4 GB | 4.5-5.5 | ~800 GB/s effective |
| heads + priors chain: `bmm` [16,73,384]×[16,384,1300], 5 masked_fill + 5 softmax over [16,73,1300] fp32, `.float32()` of target logits, 5 syncs' pipeline bubbles, ~9 MB pageable H2D, 8+5 small D2H | | 2.5-3.5 | memory-bound kernels on 5 MB tensors; pageable copies do not overlap compute |
| total | | 19-30 | measured 24.5 |

Effective throughput today: 56 GFLOP per leaf / 1.5 ms = 37 TFLOPS,
22% of peak. The bench rows say the same: batch 64 at ≤714 tokens, 0.58
ms/sample for 25.8 GFLOP = 44 TFLOPS; at the 2,200 bucket 2.17 ms for
~75-80 GFLOP = 35-37 TFLOPS.

## 2. What a GPU-side gain buys, by regime

Closed loop: N = 19 actors, throughput X = 320/16 = 20 requests/s.
Cycle per request = N/X = 0.95 s. Server response per request ≈ 60 ms
own batch (with the two-thread sharing of section 1.2) + queueing at
server utilization ~0.6 (~20-40 ms) ≈ 0.08-0.10 s. So the actor's own
part of the cycle is ~0.85 s per 16-leaf request (~50 ms per leaf of
actor wall, at ~50% CPU per actor in the `top` snapshot,
pool_on_bf16_rerun_top.txt). The bench's per-leaf actor components sum
to ~5 ms on an idle box (encode_raw 1.35, pack_masks 1.14 incl. masks,
unpack_compact 0.49, fork+step 0.7; box_specs.md:222-223, 172-173), so
either contention inflates them ~5x or the actor loop carries cost the
bench does not time (node expansion into ~350 `LegalActionPrior`
objects, tree selection, GC). That gap is outside this note's scope but
decides the regime:

- **Under-fed regime (now):** throughput ≈ N·16 / (Z + R) with Z ≈ 0.85,
  R ≈ 0.1. Halving the server's per-batch time: R 0.10 → 0.06, throughput
  +4-5%. Zeroing it: +10%. The pool benchmark will not show GPU-side
  gains until Z shrinks or requests-in-flight per actor rises.
- **Fed regime (after 2 in-flight requests per actor, or 28-38 actors,
  the next planned runs):** throughput → 16 / (GPU ms per batch), i.e.
  670 leaves/s today, and every GPU-ms saved converts 1:1.

Recommendation for measurement: report **GPU-ms per leaf** (`nvidia-smi`
utilization ÷ leaves/s, or better `torch.cuda.Event` pairs around
forward+priors in `_serve_worker`, accumulated into the serve stats
dict at actor_pool.py:738) alongside leaves/s. That isolates the GPU
side from feeding.

## 3. Option 1: torch.compile on `forward_streams`

### 3.1 What compile does today

`--infer-compile` wraps the whole module (`torch.compile(model,
mode="reduce-overhead")`, wesnoth_ai/transformer_policy.py:245-246) with
`torch._dynamo.config.suppress_errors = True` (line 237). The server
receives that wrapper (actor_pool.py:491) and calls
`model.forward_streams(...)` (inference_seam.py:264): `OptimizedModule`
only compiles `forward`; other attributes delegate to `_orig_mod`, so
**the padded server path runs eager even with `--infer-compile`**. The
bench's `forward_costs` calls `forward_batch` (bench_pipeline.py:284,
289), also eager. The compiled batch-1 path measures 5.2-5.4 ms flat
from 714 to 2,200 tokens (fwdbatch_bf16.md batch-1 rows), i.e. it is
overhead-bound, not GPU-bound — consistent with the compile being
ineffective there (recompiles or eager fallback under `suppress_errors`).
Not this note's subject, but it means "compile" has never been shown to
work on this model; treat the first compile of `forward_streams` as an
experiment, not a port.

### 3.2 Recompile hazards in the current function

Dynamo specializes on Python values it reads. In `forward_streams`:

- `sizes: List[Tuple[int,int,int]]` (model.py:544, 555-557, 578-586):
  the loop reads every `U_b, R_b, H_b` as constants → one guard set per
  distinct list of sizes → a recompile per batch composition →
  `cache_size_limit` (default 8) exhausted within the first eight batches
  → dynamo gives up on the frame and runs eager, silently under
  `suppress_errors`. This alone makes compiling the function as written
  worthless.
- The numpy mask/index construction (574-586) and `torch.from_numpy`
  (587-589): traceable in principle (`trace_numpy`), but slice
  assignment in a Python loop over B graph-breaks; each break adds a
  guard evaluation and a sub-graph.
- `if H_max:` / `if U_max:` / `if R_max:` (560-565, 599): fine as
  symbolic guards, but 0 and 1 are specialized values in dynamo, so
  R_max = 0 (no recruits) and U_max = 1 produce separate compiles.
- `hex_batch.size(1) * 0 + unit_batch.size(1)` (558): harmless, but
  remove it.
- Returning a `PaddedOutput` dataclass with CPU `actor_kind` (589, 619):
  return a tuple of device tensors from the compiled region and build
  the dataclass outside.

### 3.3 What must change (tensor-only graph)

Split the function:

```
# Python, outside the graph (numpy, ~0.2 ms):
sizes_np  -> pad_mask [B, L] bool, actor_idx [B, A_max] int64 (as today)
# compiled, tensors only:
_trunk_and_heads(hex_b, unit_b, recruit_b, global_b, end_b, pad_mask, actor_idx)
    -> (actor_logits, type_logits, weapon_logits, target_logits,
        value, value_logits, cliffness, aux, ml)
```

Dynamic dims: `B`, `H_max`, `U_max`, `R_max`. Either
`torch.compile(fn, dynamic=True, fullgraph=True)` or `mark_dynamic` on
dims 0 and 1 of the three streams each call. `fullgraph=True` so a graph
break is an error at warmup, not a silent 3x slowdown; do not inherit
the global `suppress_errors=True` for this path (assert
`torch._dynamo.utils.counters["graph_break"]` is empty after warmup).
Put the `torch.autocast` block inside the compiled function (dynamo
records the context) or, better, drop autocast: convert the inference
copy to bf16 natively (`self._inference_model.to(torch.bfloat16)`;
the trainer's copy is a separate module, transformer_policy.py:185-187,
and `load_state_dict` casts on the snapshot copy), keep LayerNorm inputs
fp32 by explicit `.float()` where the numerics need it. Native bf16
removes the 4 per-layer input casts and the per-batch weight casts
(autocast's cast cache is cleared at every context exit, so all ~50
weight tensors are re-cast every batch: ~50 kernels and ~90 MB of
traffic).

Pre-warm at startup with (B, H, U, R) ∈ {(16, 800, 20, 7),
(16, 1300, 30, 7), (64, 2200, 60, 7), (8, 700, 2, 0)} and assert the
frame count stops growing (`counters["frames"]["ok"]`).

### 3.4 Modes

- `dynamic=True`, default mode (inductor, no cudagraphs): one compile,
  40-90 s per server process (the seed measured 10-14 s per static shape
  on the smaller batch-1 graph, transformer_policy.py:238-240; dynamic
  is 2-3x that). FX-graph cache in `TORCHINDUCTOR_CACHE_DIR`
  (transformer_policy.py:242-244) brings later processes to 5-10 s.
- `mode="max-autotune-no-cudagraphs"`: adds Triton GEMM templates with
  epilogue fusion (bias+ReLU into linear1, residual into linear2/out) —
  the only way to shave the memory-bound layer glue further than plain
  fusion. Autotune runs at the first shape hint and is reused. Cost:
  2-5 min extra warmup per process; gain on the K=384 GEMMs is uncertain
  (cuBLAS is already decent), 0-15% of the 7-10 ms of linears. Try after
  the default mode has a number.
- `mode="reduce-overhead"` with `dynamic=True`: cudagraph trees record a
  new graph per distinct input shape; with hundreds of distinct
  (B, H_max, U_max, R_max) combinations this never stops recording
  (memory and warmup unbounded). Only usable after bucketing (section 4).

### 3.5 Expected gain

GPU: elementwise glue 5 → ~2 ms (fused residual+LN, fused ReLU, no
casts with native bf16); priors chain 5 masked_fill + 5 softmax → 2-3
fused kernels, ~1 → 0.5 ms; attention and GEMMs unchanged (SDPA lowers
to the same `_scaled_dot_product_efficient_attention` extern call).
**−3.5 to −5 ms of 24 (−15-20%)**. CPU: ~130 trunk launches + ~35 priors
launches at ~25 µs eager dispatch (≈4 ms) → ~85 launches through
inductor's wrapper at ~10-15 µs (≈1.1 ms): **−3 ms of the thread's
10-12 ms**, and −3 ms of the ~13 ms that the same launches cost under
the pool's CPU contention (10% of serve-thread samples at 132 ms per
batch per thread in `prof_idle`).

Size: ~120 lines (split function, on-device or numpy mask builder,
native-bf16 conversion, warmup + guards check, a CLI flag on
`InferenceServer`). Risks: silent eager fallback (mitigated by
`fullgraph=True` and the counters check); bf16 LayerNorm numerics if
converted natively (parity protocol, section 9); compile time added to
every server start (acceptable: the server lives for a campaign; not
acceptable for one-process-per-game eval workers, which are already
being replaced by persistent workers, box_specs.md:320-344).

## 4. Option 2: CUDA graphs with length buckets

### 4.1 Shapes and capture count

A CUDA graph fixes every tensor shape. Bucket L = H+U+R+2 to
{768, 1024, 1280, 1536, 2048, 2304} and B to {8, 16, 32, 64}; fix
A_max = 73 (U ≤ 64, the bench maximum is 63; R ≤ 8) so the actor
dimension is not a bucket axis (eager fallback for U > 64). That is 24
graphs. Bucketing L to 128 alone costs +6-7% tokens and +8-9% GPU cost
(simulation, section 10: `L → multiple of 128: cost overhead 1.079-1.087`);
the six coarse buckets above cost ~+12%. Bucketing B: real requests are
5-16 leaves (the last chunk of a sequential-halving phase,
tools/mcts.py:1453-1457, minus terminal/expanded leaves,
mcts.py:1216-1228), so with max_batch 16 the batch is padded from 5-16 to
8 or 16: +10-25% on those batches. Net padding overhead of graphs at
max_batch 16: **+15-25% GPU time**.

### 4.2 What graphs remove

The launch gaps. On an idle box the two serve threads overlap CPU with
GPU and the GPU is rarely starved, so the gain in GPU-ms is only the
inter-kernel bubbles (~180 kernels × ~5 µs ≈ 1 ms) plus the five sync
bubbles in `batched_priors` (~0.3 ms) — and the syncs must be removed
before capture anyway (`nonzero` and boolean-mask indexing are illegal
during capture; the pageable copies too). Under CPU contention the
launch stream itself stalls and the GPU idles behind it; graphs make
per-batch GPU time independent of CPU speed. So: **GPU time ≈ +15-25%
(padding) −5% (gaps) = net worse on an idle box; neutral-to-positive
only when the CPU is starved. CPU per batch 10-12 ms → 3-4 ms** (mask
pack, staging copies, wire remain).

### 4.3 Expressing the masks and the compaction gather in-graph

With static shapes the host must not build per-batch index arrays.
Switch to an **end-aligned layout**: row b = [hex(H_b) | unit(U_b) |
recruit(R_b) | global | end_turn | pad]. Then everything derives from one
static `[B, 3]` int32 input `sizes_t = (H_b, U_b, R_b)`:

- `n_b = H_b + U_b + R_b + 2`; `pad_mask = arange(L)[None] >= n_b[:, None]`
  (the TransformerEncoder converts it to the float mask in-graph).
- `actor_idx[b, a] = where(a < U_b + R_b, H_b + a, H_b + U_b + R_b + 1)`
  for a in [0, 73); padded actor slots point at end_turn and are
  masked out by `actor_m` in `batched_priors` as today.
- `global_idx[b] = H_b + U_b + R_b` → one gather for the value head.
- `hex_ctx = x` (all L positions); target logits `[B, 73, L]`; positions
  ≥ H_b in row b hold unit/recruit/pad tokens and are already excluded by
  the priors masks (`atk_m[b, :A, :H]` etc., server_priors.py:136-138).
  Extra cost: the bmm grows from H_max to L columns (~+8% of a 1 GFLOP op).
- Kind embedding: one per-token kind index `[B, L]` (derived from
  `sizes_t`) → `kind_embed[kind_idx]` instead of three broadcast adds.
- Priors masks: ship the bit-packed `[A, ceil(H/8)]` arrays (already the
  wire format, server_priors.py:39-41) into a static `[B, 73, ceil(L/8)]`
  uint8 buffer and unpack on device (`(u8[..., None] >> shifts) & 1`);
  the numpy `_unpack_bits` loop (132-146) disappears.
- Compaction: replace the four `nonzero` with a fixed-capacity compaction
  (flat mask → exclusive cumsum → scatter of indices into a `[C]` buffer,
  `count` read back with the results; overflow flag → eager fallback).
  Capacity: legal actions per leaf are ~300-1,500; C = 64 × 4,096.

The encoder side changes accordingly: `_embed_streams` already produces
the packed `[total, d]` embeddings (encoder.py:848-877); instead of three
`pad_sequence`, one `index_copy_` into the `[B, L, d]` static buffer with
a per-token destination index built by `np.repeat`/`np.arange` host-side
(one small H2D). This removes three Python `split`/`pad_sequence` loops
(~1 ms CPU).

### 4.4 Memory

Per graph, private pool = peak live intermediates: at B=64, L=2304:
residual fp32 226 MB, QKV bf16 340 MB, FFN hidden bf16 453 MB, two LN
temporaries 450 MB, target logits + 3 probability tables
`[64,73,2304]` fp32 170 MB → **1.2-1.6 GB for the largest bucket**, ~0.2
GB at (16, 1280). Manual `torch.cuda.CUDAGraph` with one pool per graph:
sum over 24 buckets ≈ 8-12 GB — not affordable next to the learner's
12.6 GB backward peak on a 24 GB card (box_specs.md:29-31). Options:
share one pool across captures (safe only if every graph's static
outputs stay referenced for life and replays never run concurrently —
which also means **one GPU-submission lock across the two serve
threads**), or let cudagraph trees (`mode="reduce-overhead"` on the
bucketed, dynamic-compiled function) manage a shared pool — their
state is per-device global and not thread-safe, so the same lock
applies. Static inputs: one max-size buffer `[64, 2304, 384]` fp32
(226 MB) with per-bucket views, not one buffer per bucket. Budget:
~2 GB total. Capture: 24 × (1-2 warmup runs + capture ≈ 0.3-0.5 s) ≈
10 s per process, on first use per bucket.

### 4.5 Verdict

Graphs are a CPU-side lever. They cost 15-25% GPU time in padding and
recover ~5%. Do them last, only if after options 1, 3, 4 the serve
thread's CPU per batch (then ~5-6 ms) is again comparable to GPU per
batch (then ~13-15 ms) under the pool's contention, or use them on the
eval harness's batch-1 path where the 5.3 ms is pure overhead (2-3x
there, out of scope). The end-aligned layout and the sync-free priors
they need are worth building regardless (sections 5, 6).

## 5. Option 3: the attention kernel

### 5.1 What runs today (pinned from the 2.5.1 source, section 11)

`nn.TransformerEncoderLayer.forward` skips its fused fast path because
"autocast is enabled" (transformer.py, both Encoder and Layer); with
`enable_nested_tensor=False` the Encoder's nested-tensor path is off
too (it would also require a left-aligned mask, which ours is not). So
each layer runs `F.multi_head_attention_forward` with
`need_weights=False` — the profile's leaf frames
`multi_head_attention_forward (functional.py:6278, 6285)` and
`_in_projection_packed (functional.py:5501)` (prof_idle_summary.txt)
confirm the Python path. Inside it:

1. `TransformerEncoder.forward` turns the bool `src_key_padding_mask`
   into a float mask once (`_canonical_mask`, `target_type=src.dtype`
   = fp32: 0 / −inf).
2. `multi_head_attention_forward` reshapes it to `[B·heads, 1, S]`
   (`view(bsz,1,1,S).expand(-1,heads,-1,-1).reshape(B·heads,1,S)` — a
   0.5 MB copy per layer), then `view(bsz, num_heads, -1, src_len)` →
   `[B, 12, 1, S]`, `dropout_p = 0` (not training), and calls
   `scaled_dot_product_attention(q, k, v, attn_mask)` with q,k,v bf16
   from the autocast'd in-projection.
3. `scaled_dot_product_attention` is in CUDA autocast's
   lower-precision list (autocast_mode.h `AT_FORALL_LOWER_PRECISION_FP`),
   and the policy casts every floating tensor argument including the
   optional mask, so the mask arrives bf16.
4. Backend selection (sdp_utils.cpp): flash is rejected by
   `check_for_attn_mask` (non-null mask); mem-efficient accepts bf16 on
   sm80+ and head_dim % 8 == 0 (32) and has no mask constraint; order is
   flash > efficient > math. **The cutlass memory-efficient kernel with
   an additive bias runs.** `preprocess_mask` (attention.cpp) checks
   8-element stride alignment of the `[B,12,1,S]` mask; S = H+U+R+2 is
   rarely a multiple of 8, so `pad_bias` copies and re-slices it (a
   `[B,12,1,S]` copy, ~0.5 MB, trivial) and then expands it to
   `[B,12,L,S]` with stride 0 on L — no materialization (the kernel
   takes `bias_strideM` directly and only requires the last dim
   contiguous).

So the attention is not the `nn.MultiheadAttention` math fallback; it
is the right kernel family, but a slow member of it: head_dim 32 with a
bias read per key, no block skipping for padded keys, and per layer two
mask kernels (canonicalize/merge, pad).

### 5.2 What a bucketed or end-aligned mask allows

- S a multiple of 8 (any bucketing): the per-layer `pad_bias` copy
  disappears. Negligible (~0.05 ms per batch).
- **End-aligned padding + per-row lengths: drop the mask entirely and use
  the varlen flash kernel.** `torch.ops.aten._flash_attention_forward`
  is callable with `cum_seq_q, cum_seq_k` int32 offsets and
  `max_q, max_k` (its call in torch 2.5.1's nested SDPA glue, section
  11). Two ways to feed it:
  (a) packed q,k,v `[total_tokens, 12, 32]` with true offsets — no
      padded keys, no padded queries;
  (b) the dense padded `[B·L, 12, 32]` view with offsets
      `[0, L, 2L, ...]` for q and the same for k plus per-row `seqused_k`
      — dense layout kept (graph-friendly), padded keys skipped, padded
      queries still computed. `seqused_k` exists on the 2.5 op in my
      reading but the yaml fetch truncated; **verify on the box with
      `torch.ops.aten._flash_attention_forward._schema` before choosing
      (b)**; (a) needs only the arguments quoted in section 11.
  Flash needs sm80+; the box wheel carries sm_86/sm_80 SASS
  (box_specs.md:154-156), which the sm_89 card runs natively (same major
  compute capability), so this is not a PTX-JIT path.
- Public-API alternative: `torch.nested.nested_tensor(..., layout=jagged)`
  + `F.scaled_dot_product_attention` dispatches to the same varlen
  flash kernel and avoids the private op, but the surrounding
  `nn.TransformerEncoderLayer` does not accept NJTs, so either way the
  layer loop is re-implemented over the layer's own parameters
  (`self_attn.in_proj_weight/bias`, `out_proj`, `linear1/2`, `norm1/2`;
  ~40 lines, ReLU, post-norm).

### 5.3 The packed trunk (this is where options 2 and 3 meet)

Run the whole trunk on the packed `[total, d]` tensor
`_embed_streams` already produces: linears and LayerNorms on
`[total, d]` (no padding at all), attention via (a). Only the heads pay
padding: gather actor rows into `[B, 73, d]` and hex rows into
`[B, H_max, d]` (two `index_select`, indices host-built as today) for
the `bmm`; value head reads `global` rows by index. The padding mask,
the kind-add broadcast, `cat`, the three `pad_sequence`, the
`[B,12,1,S]` mask kernels all disappear.

Expected GPU gain at production sizes: attention 5.5-11 → 3.5-6 ms
(flash on sm8x at head_dim 32 is ~40-50% of peak; no bias reads;
−8% padded keys at pad 1.11, −30% at pad 1.44), linears and glue −8%
(pad 1.11) to −30% (pad 1.44), mask kernels −0.3 ms. **−5 to −7 ms of
24 at max_batch 16 (−20-25%); −8 to −10 ms relative to unbucketed
max_batch 64 batches.** With option 1 (compile the packed layer loop —
it is tensor-only by construction, with `total` as the single dynamic
dim) the two compose: ~13-15 ms per 16 leaves.

Size: ~150 lines (layer loop, packed encode assembly, head gathers, a
flag) + parity test. Risks: private-op signature drift across torch
versions (pin 2.5.1, assert the schema at import); kernel numerics
(flash vs cutlass vs math differ at bf16 rounding; parity protocol,
section 7); zero-length rows (a request with U=0 is impossible — the
own leader is always visible — but R=0 happens; a segment length of 0
inside the packed layout is fine because each sample is one segment of
H+U+R+2 ≥ 4 tokens).

## 6. Option 4: sync and copy points in `batched_priors`

### 6.1 What the 12% actually is

`dev (server_priors.py:149)` at 12% of serve-thread samples is the first
pageable `.to(device)` after the forward (line 152). A pageable H2D is a
synchronous `cudaMemcpy`; it waits for everything queued on the stream —
the whole forward. **The 12% is GPU time being waited for, not copy
time**, and reordering does not remove it: the numpy mask loop (132-146)
already runs before that line, so there is no CPU work left to overlap.
What is avoidable:

- ~9 MB of pageable copies per 16-leaf batch (three `[16,73,1300]` bool
  masks 4.5 MB; the float32 attack bias 6 MB when the combat-oracle bias
  is active — it is present whenever `attack_bias` is non-zero,
  server_priors.py:94, i.e. throughout the anneal horizon of
  action_sampler.py:95-116): pageable copies stage through a driver
  bounce buffer at ~5-8 GB/s and do not overlap compute on the stream:
  **~1.2-1.8 ms of GPU-stream time per batch**, ~1-1.5 ms of thread wall.
- Twelve separate `dev()` calls (~25-40 µs each of Python+launch): ~0.4 ms.
- Five host syncs inside the priors chain (`al[is_end] += ...` at 160,
  four `nonzero` at 184-187): each drains the pipeline and leaves the GPU
  idle while the host launches the next kernels: ~0.3-0.5 ms of bubbles.
- Eight `.cpu()` at 198 plus 3-5 at inference_seam.py:268-271: each is a
  synchronous small D2H (~30-50 µs): ~0.5 ms.

### 6.2 Minimum-sync arrangement

1. **One pinned H2D staging buffer per serve thread.** The actor already
   ships one contiguous uint8 buffer per request (`leaf_wire.pack_request`,
   wesnoth_ai/leaf_wire.py:66-98) holding every numeric stream and every
   packed mask. Copy the request buffers into a preallocated pinned
   buffer (numpy memcpy, ~2 MB per batch), one `copy_(non_blocking=True)`
   to a device buffer, and build the device-side views from the headers
   (`unpack_request` logic, leaf_wire.py:101-135, applied to a device
   tensor with `.view(dtype)` slices). The 15 `_cat_to_dev` pin+copy
   pairs in `_embed_streams` and the 12 `dev()` calls become one copy;
   the bool masks travel bit-packed (8x smaller) and are unpacked on
   device; the float attack bias travels as is (or as bf16, 2x).
2. No boolean-mask indexing, no `nonzero`: `al + where(is_end,
   et_bias[:, None], 0)`; fixed-capacity compaction (section 4.3).
3. **One pinned D2H staging buffer**: concatenate `[counts, ia, im, ir,
   ie, fa, fm, fr, fe, value, value_logits, cliffness, aux, ml]` on
   device into one flat buffer (one `cat` kernel), one
   `copy_(non_blocking=True)` into pinned host memory, one
   `stream.synchronize()` (or an event the thread waits on). Then numpy
   as today.

Gain: **−2 to −3 ms of GPU-stream time per batch (−8-12%)** and −1.5 to
−2.5 ms of thread CPU/wall. It is also the prerequisite for any graph
capture (section 4) and for a single serve thread ever being enough.
Size: ~120 lines. Risks: pinned buffer sizing (bounded by max_batch ×
max tokens; ~30 MB each), the compaction capacity fallback, and one
more place where the wire header must agree with the server (already
the case for `leaf_wire`).

## 7. Option 5: length-aware batch composition in the serve thread

### 7.1 Where the padding comes from

One request = the leaves of one search tree = one game = one map
(`_run_sim_batch`, tools/mcts.py:1229-1237, one `forward_batch` per
chunk, `_IPCInferenceClient.infer_batch`, actor_pool.py:108-131, one
message). Within a request H is constant with the seed served today:
its encoding is full-board — the bench's largest token bucket, 2,200,
is Arcanclave's 2,162 hexes plus units and recruits, and the seam rows'
663-token mean is Sablestone's 588 plus a small army (fwdbatch_bf16.md,
seam_sorted.md) — and U varies by a few units. So intra-request padding
is ~1.003 (simulation, section 10). The `relevant_set_hexes` flag is
carried by the checkpoint (`CHECKPOINT_STRUCT_FLAGS`, tools/eval_sim.py:
253-254; the pool reads it from the encoder, actor_pool.py:533-535); a
relevant-set checkpoint would make H vary per leaf inside a request,
raising intra-request padding, and the packed trunk of section 5.3 is
then the only layout without waste. The measured pool ratios — 1.08-1.11 at
max_batch 16 and 1.44 at max_batch 64 (pool_*.json) — come entirely from
**coalescing requests of different maps** (actor_pool.py:749-755
coalesces while `n_leaves < max_batch`; requests are 5-16 leaves, so
even at max_batch 16 two or three requests often share a batch). The
ladder-map hex counts are bimodal: 588-1,188 for 19 maps, 1,960 and
2,162 for Aethermaw and Arcanclave (from the map files under
`wesnoth_src/data/multiplayer/maps/`, (rows−2)×(cols−2), matching every
`n_hexes` in configs/bench_states.json). One Arcanclave request in a
batch of Sablestone requests pads 588 → 2,162.

`bench_pipeline.seam_costs` sorts the 200 states by tokens before
cutting batches (bench_pipeline.py:327) and so measures ~663-token
near-homogeneous batches — it reports the bucketed case, not
production (1,150-1,300 tokens, ratio 1.11-1.44).

### 7.2 Sorting versus bucketing

Sorting the requests that happen to be in the queue does nothing for a
batch that will contain all of them anyway: the batch's cost is set by
its longest row. The lever is **which requests go together**: cut the
sorted queue where the token count jumps (e.g. > 128 tokens between
neighbours), and never put a 2,000-token request in a batch with
600-token ones even if the batch stays under max_batch. When the queue
holds a single request (the under-fed regime, most of the time today)
there is nothing to choose; the option pays only when the queue has
depth, i.e. exactly when the feeding fix lands.

Simulation (section 10; ladder maps uniform or hex-weighted, requests of
5-16 leaves, batch cost = linear + quadratic terms):

| max_batch | unbucketed pad / cost ratio | bucket 128 | bucket 256 |
|---|---|---|---|
| 16 | 1.17-1.20 / 1.24-1.28 (measured 1.08-1.11) | 1.01-1.02 / 1.01-1.02 | 1.03-1.04 / 1.04-1.05 |
| 32 | 1.37-1.42 / 1.53-1.60 | 1.03-1.04 / 1.03-1.04 | 1.06-1.08 / 1.08-1.10 |
| 64 | 1.60-1.64 / 1.88-1.92 (measured 1.44 at ~30 leaves) | 1.04-1.06 / 1.05-1.07 | 1.10-1.13 / 1.13-1.17 |

The simulation over-estimates the unbucketed ratio (it always fills to
max_batch; the live queue often cannot), so scale by the measured
values: at max_batch 16 the GPU cost per leaf drops ~8-10%; at
max_batch 64, fed, ~30-35% relative to the unbucketed 1.44 (cost ratio
≈ pad^1.3 for this mix). Bucketing shrinks batches (fewer rows per
batch) — with ~30 queued leaves in three maps the batch may become 10
leaves; batch efficiency between 8 and 32 rows is flat within 5-7% in
the bench (fwdbatch_bf16.md rows), so the loss is small, and the serve
thread can take the next bucket immediately.

Size: ~30 lines in `_serve_worker` (a `PackedRequest` already carries
`n_hexes`/`n_units` in its headers, leaf_wire.py:52-60, so the token
count is free). Risk: starvation of the odd-sized request — bound the
wait (a request older than one batch time goes out alone).

## 8. Ranking

Baseline per 16-leaf batch at ~1,270 tokens: GPU 24 ms, CPU 10-12 ms,
fed ceiling 670 leaves/s.

| rank | option | GPU ms/batch after | fed ceiling leaves/s | CPU ms/batch | size | main risk |
|---|---|---|---|---|---|---|
| 1 | packed varlen trunk (sections 5.3, 4.3 layout) | 17-19 | 840-940 | 8-9 | ~150 lines + parity | private-op schema; bf16 kernel numerics |
| 2 | inductor compile of a tensor-only `forward_streams`, native bf16, dynamic shapes, no graphs (section 3) | 19-20.5 (24 base) | 780-840 | 6-7 | ~120 lines | silent eager fallback; compile time per process |
| 3 | one staging buffer each way, no `nonzero`, one sync (section 6) | 21-22 | 730-760 | 8-9 | ~120 lines | compaction capacity; buffer sizing |
| 4 | bucketed batch composition (section 7) | 22 at mb16; 24 → 16 per 16 leaves at mb64 | 730 / 1,000 | same | ~30 lines | starvation; zero effect until fed |
| 5 | CUDA graphs on buckets (section 4) | 25-29 (padding) | 550-640 | 3-4 | ~250 lines incl. 3 | memory pool sharing; thread safety; capture legality |
| 1+2+3+4 | | 13-15 at mb16; ~11-12 per 16 leaves at mb32-64 bucketed | 1,070-1,230 / 1,300-1,450 | 5-6 | | |

Effective TFLOPS at the combined row: 56 GFLOP / 0.8 ms = 70 TFLOPS,
42% of peak; 1,500+ leaves/s would need the GEMM epilogue fusion
(max-autotune) and a better head_dim (a 1.4 decision). 3,000 needs
~2x fewer tokens.

Order of work: 3 first (cheap, needed by everything else, gives the
profiler a clean picture), then 1, then 2 on top of 1's layer loop
(one dynamic dim, `total`, so compile is trivial there), 4 alongside,
5 never unless the CPU column is again the bottleneck in the fed pool.

## 9. Benchmark protocol on the existing tools

Each option lands behind a flag on `InferenceServer` (and `--infer-*`
on `bench_pipeline.py`/`bench_pool.py`), measured against the previous
row on the same box in one session.

1. **Kernel-level truth first** (30 s, run before touching anything):
   add `--profile` to `bench_pipeline.seam_costs` (bench_pipeline.py:304):
   wrap one 16-leaf priors-protocol batch at production sizes in
   `torch.profiler.profile(activities=[CUDA, CPU])` and print
   `key_averages().table(sort_by="cuda_time_total", row_limit=30)`. This
   gives the section 1.3 decomposition directly (matmul vs
   `fmha_cutlassF` vs elementwise vs `Memcpy HtoD/DtoH` vs `nonzero`)
   and settles the attention backend (`fmha` kernel name vs
   `flash_fwd`). Kill any option whose target component is under 5% of
   the table.
2. **Production-size seam rows**: `seam_costs` currently sorts the
   states (bench_pipeline.py:327) and measures 663-token batches. Add
   `--seam-mix {sorted,manifest,pool}`: `manifest` order gives mixed
   maps (pad ~1.4); `pool` composes batches as `_serve_worker` does
   (requests of 5-16 same-map leaves, coalesced to max_batch). Report
   ms/batch, leaves/s and GPU-ms/batch (`torch.cuda.Event`
   `elapsed_time` around forward+priors) for batch 16 and 64. Expected
   after each option: the GPU-ms column moves by the section 8 numbers;
   if the seam leaves/s moves < 5% at batch 16 on `pool` mix, kill.
3. **Numerics**: (i) `tests/test_server_priors.py` differential
   (compact vs reference enumeration) on the box with the new path;
   (ii) a 200-state parity run in `bench_pipeline.component_costs`
   (bench_pipeline.py:181-238 already computes `batched_priors` on
   `forward_padded([encoded])` per state): compare old vs new path
   priors — max |Δprior|, and the argmax action per state. Calibrate
   first with today's bf16 path against fp32 eager (that difference is
   already accepted in production); a new kernel must stay within it:
   argmax identical on ≥ 198/200 and max |Δprior| ≤ the bf16-vs-fp32
   figure. The raw:t0 eval harness is deterministic run to run
   (box_specs.md:341-344), so also replay 20 argmax games through
   `run_elo_batch.py --persistent-workers` and diff outcomes and turn
   counts against the current path; up to 3/20 differing is the level
   bf16 itself produced (box_specs.md:337-341); more is a bug.
4. **Pool**: `tools/bench_pool.py --checkpoint <seed> --actors 19
   --games 16 --server-priors --infer-bf16 --max-batch 16|64 <new flag>`
   with `nvidia-smi dmon -s u -d 5` in the background; report leaves/s,
   GPU util, and GPU-ms/leaf = util ÷ leaves/s (add the Event timer to
   the serve stats so it is logged directly). In the under-fed regime
   expect leaves/s to move ≤ 5% and GPU-ms/leaf to move by the section 8
   numbers; that is the pass criterion. Repeat at `--actors 28` and
   `38` (the planned feeding runs) where leaves/s should follow
   GPU-ms/leaf. Run-to-run noise on shared hosts is ~13% (box_specs.md
   270-274): two runs per row, same seed.
5. **Compile-specific**: log `torch._dynamo.utils.counters` after
   warmup and after the first 200 batches; any growth in
   `frames`/`graph_break`/`recompiles` fails the row. Log warmup
   seconds per process.
6. **Graph-specific** (if ever): replay all 24 buckets in random order
   100 times against eager outputs (aliasing corruption shows up as
   mismatches), `torch.cuda.memory_reserved()` before/after capture must
   stay under 2.5 GB.

## 10. Simulation used for the padding numbers (numpy, no torch)

Inputs: ladder-map hex counts {588, 638, 696, 725, 744, 756, 806, 810,
837, 851, 870, 888, 902, 936, 984, 1050, 1160, 1175, 1188, 1960, 2162}
(map files, matching `n_hexes` in configs/bench_states.json for the 19
maps present there), unit counts from the 200 bench states (5-63,
median 19, mean 21.5), R = 7 recruit tokens (own faction's list,
encoder.py:1204-1216), +2 for global and end_turn; requests of 5-16
leaves of one map with U jittered ±3 per leaf; batches coalesced to
max_batch as actor_pool.py:749-755; cost per row = 28.3 MFLOP × L +
8 × 4·L²·384 (linears + attention). Mean tokens: 1,017 with uniform
maps, 1,168 weighting maps by hex count (the pool measures 1,150-1,300,
so big maps contribute more leaves — more decisions per game). Results
are in section 7.2 and 4.1. Attention share of FLOPs: 23% at L=700,
30% at 1,000, 36% at 1,300, 46% at 2,000.

## 11. Facts pinned from torch 2.5.1 source (fetched from the v2.5.1 tag)

- `torch/nn/modules/transformer.py`: `TransformerEncoderLayer.forward`
  and `TransformerEncoder.forward` skip the fused path with
  "autocast is enabled"; the Encoder's nested path additionally needs
  `use_nested_tensor` and "src and src_key_padding_mask ... left
  aligned"; both call `F._canonical_mask(..., target_type=src.dtype)`;
  `_sa_block` passes `need_weights=False`.
- `torch/nn/functional.py` `multi_head_attention_forward` (quoted from
  the local 2.10 copy, lines 6555-6570 and 6614-6631; the 2.5.1 profile
  leaf lines are 6278/6285 — same function, the code is unchanged in
  substance): bool mask → float(−inf) via `_canonical_mask`; key padding
  mask merged as `view(bsz,1,1,S).expand(-1,H,-1,-1).reshape(B·H,1,S)`;
  `dropout_p = 0.0` when not training; `attn_mask.view(bsz, num_heads,
  -1, src_len)` then `scaled_dot_product_attention(q, k, v, attn_mask,
  dropout_p, is_causal)`.
- `aten/src/ATen/autocast_mode.h`: `scaled_dot_product_attention` is in
  `AT_FORALL_LOWER_PRECISION_FP`; `layer_norm` in `AT_FORALL_FP32`;
  `softmax` in `AT_FORALL_FP32_SET_OPT_DTYPE`; the lower-precision policy
  applies `cached_cast` to every argument including `optional<Tensor>`.
- `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp`: flash
  constraints include `check_for_attn_mask` (any mask → rejected);
  mem-efficient dtypes on sm80+ = {half, float, bfloat16}, head_dim must
  be a multiple of 8, no mask constraint; default order flash,
  efficient, math, cudnn.
- `aten/src/ATen/native/transformers/attention.cpp`: `preprocess_mask`
  uses `mem_eff_alignment = 8`, pads via `pad_bias` only when
  `aligned_tensor<8>` fails, then `expand`s to `[B, H, L, S]`;
  `validate_sdpa_input` accepts mask dtype bool, float, or the query's.
- `aten/src/ATen/native/transformers/cuda/attention.cu`
  `_efficient_attention_forward`: bias dtype must equal the query's
  ("invalid dtype for bias - should match query's dtype"), bias must be
  4-D with `stride(3) == 1`; `bias_strideB/H/M` are taken as given
  (stride 0 accepted).
- `torch/nested/_internal/sdpa.py`: calls
  `torch.ops.aten._flash_attention_forward(q, k, v, cum_seq_q,
  cum_seq_k, max_q, max_k, dropout_p, is_causal, False, scale=...)` with
  int32 cumulative offsets, and `_efficient_attention_forward(q, k, v,
  None, cum_seq_q, cum_seq_k, max_q, max_k, dropout_p, int(is_causal),
  compute_logsumexp, scale=...)` — both varlen entry points exist in
  2.5.1 as private ops.
