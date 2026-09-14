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

## 12. Option 1 implemented (2026-09-05): the packed varlen trunk

Written without a GPU; every CUDA-side claim below is pinned from the
torch 2.5.1 source (the box wheel) and waits for the commands at the end
of this section. The local wheel is 2.10.0+cpu, so the 2.5.1 files were
fetched from the v2.5.1 tag and read, not executed.

### What is implemented

- `wesnoth_ai/packed_trunk.py` (new): the packed layout, its one-copy
  device transfer, the attention wrapper and the layer loop.
  `build_packed_layout(sizes, H_max, U_max, R_max, TokenKind, ActorKind)`
  builds, in numpy, the index arrays of section 4.3 for the padded row
  order hex | unit | recruit | global | end_turn: `src` (padded flat row
  of every packed token), `kind` (TokenKind per packed token), `actor`,
  `hex`, `unit`, `glob` (packed row of every head slot; padded slots
  point at the row's end_turn token, a finite value no consumer reads),
  `cu_seqlens` and `actor_kind`. `PackedLayout.to_device` concatenates
  them into one int64 buffer, pins it and copies it non-blocking (one
  transfer, no host synchronization), then splits it into views and
  casts the offsets to int32 on the device. `packed_trunk` runs the
  layers' own parameters over the `[total, d]` tensor with the post-norm
  math of `nn.TransformerEncoderLayer.forward` (transformer.py:902-906,
  `_sa_block` 911-927, `_ff_block` 930-932 at v2.5.1); `_self_attention`
  is `F.multi_head_attention_forward`'s self-attention path
  (`_in_projection_packed` functional.py:5501-5510, head split
  6274-6276, out projection 6285): one `F.linear` with `in_proj_weight`
  viewed `[total, 3, heads, head_dim]` and unbound into q, k, v
  (unit-stride last dim; the kernel reads the row and head strides from
  the tensors, flash_api.cpp `set_params_fprop` lines 85-88), then
  `packed_attention`, then `out_proj`.
- `packed_attention` calls
  `torch.ops.aten._flash_attention_forward(q, k, v, cu, cu, max_len,
  max_len, 0.0, False, False)[0]` when q is CUDA fp16/bf16 (the
  positional arguments of the 2.5.1 schema, native_functions.yaml:14814,
  quoted in the module docstring; the same call the nested-tensor glue
  makes, torch/nested/_internal/sdpa.py:752-764). Elsewhere (CPU, fp32)
  it runs `F.scaled_dot_product_attention` once per segment over the
  same packed tensors, so the bookkeeping is testable on the laptop and
  the fp32 CUDA comparison isolates layout from kernel rounding.
- `wesnoth_ai/model.py`: `WesnothModel.infer_packed_trunk = False` (the
  switch; a plain attribute, state_dict unchanged);
  `forward_streams(..., packed=None)`, threaded through `forward_padded`
  and `forward_batch`. `packed=None` selects the packed trunk only when
  `_packed_trunk_applies`: switch on, eval mode, CUDA, and the dtype the
  in-projections will produce is bf16/fp16 (bf16 inputs, or fp32 inputs
  under a bf16 autocast: `torch.is_autocast_enabled("cuda")` /
  `torch.get_autocast_dtype("cuda")`, torch/csrc/autograd/init.cpp:559-592
  at v2.5.1). Every other call keeps the padded trunk unchanged; `True`
  and `False` force a path (tests). `_forward_streams_packed` packs the
  five padded streams with one `index_select` on the concatenated
  `[B*L, d]` tensor, adds the token-kind embedding by one lookup on
  `kind` (the same fp32 add as the padded path's broadcast adds), runs
  the trunk, and lays actor / hex / global (and unit, for GBC) contexts
  out in the padded shapes with `index_select`. The heads moved verbatim
  into `_heads`, shared by both paths, so `PaddedOutput`, its views,
  `start_priors` and `batched_priors` are untouched. Nothing between the
  index copy and the head outputs synchronizes with the host (the padded
  path's own `pad_mask` / `actor_idx` `.to(device)` copies are blocking:
  `memcpy_and_sync`, c10/cuda/CUDAFunctions.h:77-86, Copy.cu:384; the
  packed path's pinned non-blocking copy goes through Copy.cu:362-381).
- `tools/inference_seam.py` needs no change: `_infer_with_priors` calls
  `forward_streams` under the bf16 autocast, so the model attribute
  alone selects the trunk. `tools/bench_pipeline.py --packed-trunk` sets
  the attribute on the uncompiled inference model (a compiled wrapper
  forwards attribute access, `OptimizedModule.__getattr__/__setattr__`)
  and records `packed_trunk` in the JSON; sections B and D then measure
  the packed path, and the flag refuses CPU or fp32 runs.
- No CLI flag on the actor pool or the policy loader yet: the production
  switch is `policy._inference_base.infer_packed_trunk = True`, to be
  plumbed once the box numbers justify it.

### Facts pinned from the 2.5.1 source while implementing

- The open question of section 5.2 is settled: `seqused_k` is a
  keyword argument of the 2.5.1 op (`Tensor? seqused_k=None`,
  native_functions.yaml:14814; flash_api.cpp:652-657 checks it int32,
  CUDA, contiguous, `[B]`). Route (a) does not use it.
- `mha_varlen_fwd` (flash_api.cpp:543) requires: fp16/bf16 (572), same
  dtype for k, v (577-578), int32 cu_seqlens (579-580), CUDA tensors
  (582-584), unit-stride last dim (595-597), contiguous cu_seqlens
  (598-599), q `[total_q, heads, head_dim]` and k, v
  `[total_k, heads_k, head_dim]` (640-644), cu_seqlens `[B+1]` (650-651),
  head_dim at most 256 and a multiple of 8 (633-635; tier B's 32
  passes), sm80+ (567). Output `empty_like(q_padded)` (678):
  `[total, heads, head_dim]`. With dropout 0 the seed/offset tensors are
  `at::empty` (no RNG state, no host work). The varlen branch of
  `_flash_attention_forward` is attention.cu:935-964.
- The 2.5.1 encoder-layer fused fast path is gated by autocast
  (transformer.py:821), training, odd heads, hooks and grad
  (793-853), not by dropout, so the `dropout=1e-4` rationale in
  `WesnothModel.__init__` gates nothing on this torch. The packed loop
  is eval-only and omits the dropouts.
- Under autocast, `F.linear` (in/out projections, MLP) runs bf16 and
  `layer_norm` fp32, in both paths; the residual stream is fp32 in both;
  the attention kernel receives bf16 q, k, v in both. The only
  numerical difference between the paths is that kernel: flash varlen
  (packed) against cutlass mem-efficient with an additive bf16 bias
  (padded). Expected: bf16 rounding differences of a few 1e-2 of each
  field's scale after 8 layers. The CUDA test asserts packed-vs-padded
  below 0.1 of the scale and packed-vs-fp32 within twice the padded
  path's own bf16 error plus 1% of scale, and prints max abs and
  scale-relative differences for actor, type, target and weapon logits,
  value, value_logits and cliffness.

### Tests

- `tests/test_packed_trunk.py` (CPU, 7 tests, 4 s): the layout
  round-trips the padded streams (pack, kinds, offsets, every head
  gather, including rows without recruits and batches without hexes);
  packed and padded `forward_streams` agree on random streams whose pad
  positions hold random values; on real encoded states,
  `forward_batch(packed=True)` equals the single-sample forward and the
  compact priors from the packed output equal those from the padded
  output; the switch leaves CPU and training-mode calls on the padded
  trunk; unsupported layer options are refused.
- `tests/test_packed_trunk_cuda.py` (skipped without CUDA): fp32
  packed (segment SDPA) vs fp32 padded on the device (bookkeeping only);
  bf16 packed vs bf16 padded vs fp32 with the differences printed;
  `torch.cuda.set_sync_debug_mode("error")` around the packed forward;
  GPU ms per batch of 16 for both trunks, near-homogeneous
  (1,150-1,300 hexes, padding ratio about 1.1, the production mix) and
  mixed (600-2,200 tokens, about 1.5), with stream-event timing. Random
  weights at the tier-B shape; no checkpoint needed.

### Not verifiable here; the box must confirm

1. The op runs on the box wheel: `USE_FLASH_ATTENTION` compiled in
   (attention.cu:1000 raises otherwise) and the sm_86 SASS / PTX-JIT
   path on the sm_89 card serving the flash kernels.
2. The parity numbers: that the bf16 differences land where this
   section expects (a few 1e-2 of scale) and that the fp32 device
   comparison sits at 1e-5..1e-4 (TF32 is off by default for matmul; if
   the fp32 mem-efficient kernel rounds more, the 5e-3 bound in the test
   says so).
3. The sync check: `pin_memory()` on the cached host allocator and the
   `torch.split` views issue no flagged operation on the second call.
4. The GPU time: section 5.3 predicts -20-25% per batch at padding 1.1
   (attention 5.5-11 to 3.5-6 ms, mask kernels gone, -8% on the linears).
5. `torch.is_autocast_enabled("cuda")` accepts the device string in
   2.5.1 (init.cpp:565 parses that form), read, not executed.

### Commands on the box

    pytest -s tests/test_packed_trunk_cuda.py -p no:cacheprovider
    pytest tests/test_packed_trunk.py tests/test_forward_batch_padded.py tests/test_server_priors.py tests/test_server_priors_cuda.py
    python tools/bench_pipeline.py --checkpoint training/checkpoints/seed_imit_tierb_start.pt \
        --device cuda --batch-sizes 16,64 --label padded --out training/metrics/bench_pipeline/seam_padded.json
    python tools/bench_pipeline.py --checkpoint training/checkpoints/seed_imit_tierb_start.pt \
        --device cuda --batch-sizes 16,64 --packed-trunk --label packed --out training/metrics/bench_pipeline/seam_packed.json

Compare the `seam` rows (protocol priors, batch 16 and 64: ms per batch,
leaves per second) and the `forwards` rows between the two JSONs; the
GPU-only number is the CUDA test's. No strength check is needed for
this change (same weights, same math up to bf16 rounding); the `raw:t0`
self-match of docs/plan_20260904.md remains the gate before the packed
trunk becomes the pool default.

## 13. Option 2 implemented (2026-09-05): inductor compile of the packed layer loop

Written without a GPU, on top of section 12's packed trunk. The local
wheel is 2.10.0+cpu; every torch 2.5.1 (box wheel) fact below was read
from the v2.5.1 tag, and the two wheels' differences that matter are
listed. Only the CPU tier ran here (inductor's CPU backend, MSVC).

### What is implemented

- `wesnoth_ai/packed_trunk.py`: `packed_trunk_layers` (built by
  `make_packed_trunk_layers(activation, eps)`) is the section 12 layer
  loop as a pure function of tensors: `(x [total, d] fp32, cu_seqlens
  [B+1] int32, max_len, layers, final_norm) -> x`, post-norm math of
  `nn.TransformerEncoderLayer` with explicit casts (`x.to(weight
  dtype)` before each linear, fp32 residual stream and LayerNorms);
  `PackedTrunkWeights`, the encoder's parameters copied in the compute
  dtype (linear weights bf16 on CUDA, LayerNorm weights fp32, the
  in-projection shaped `[3, heads, head_dim, E]`), with `refresh` for
  in-place updates; `CompiledPackedTrunk`, the `torch.compile` wrapper
  with warmup, recompile and fallback accounting; the custom op
  `wesnoth_ai::packed_attention` (below).
- `wesnoth_ai/model.py`: `WesnothModel.infer_compile_packed` (default
  off, requires `infer_packed_trunk`: `forward_streams` raises
  otherwise), `configure_packed_compile(backend="inductor",
  mode=None)`, `warmup_packed_compile(shapes)` (synthetic batches under
  the server's bf16 autocast on CUDA), `packed_compile_active`,
  `packed_compile_stats()` (active, backend, mode, warmup_seconds,
  recompiles, fallback_reason, cache_entries, weights_dtype). The
  weight copy is versioned: `WesnothModel.load_state_dict` bumps
  `_weights_version` (every publication goes through it: the policy's
  `_snapshot_inference_weights`, checkpoint loads), and the next packed
  forward refreshes the copy in place. The trainer's fp32 module and
  the padded path are untouched; the eager packed path of section 12 is
  unchanged (it stays the "eager packed" reference in the benchmarks).
- `tools/bench_pool.py --compile-packed [--compile-packed-mode
  default|max-autotune-no-cudagraphs]` and `tools/bench_pipeline.py
  --compile-packed`: require `--packed-trunk` (hence cuda + bf16), run
  `warmup_packed_compile` before serving, and record
  `packed_compile_stats()` after the run in the JSON (recompiles and any
  fallback during the run show there).

### The compile boundary: the attention as an opaque custom op

The whole loop (8 layers) is one dynamo frame and one inductor graph;
the attention inside it is `torch.ops.wesnoth_ai.packed_attention`,
defined with `torch.library.define` (schema `(Tensor q, Tensor k,
Tensor v, Tensor cu_seqlens, SymInt max_len) -> Tensor`, tag
`needs_fixed_stride_order`), an eager kernel registered under
`CompositeExplicitAutograd` and a fake kernel (`q.new_empty(q.shape)`;
torch 2.5.1 torch/library.py:421, 502, 684). Dynamo records the call,
inductor emits it as an extern kernel and fuses everything around it;
the eager kernel picks the flash varlen op or the per-segment SDPA at
run time, exactly as section 12's `packed_attention`. Reasons for the
op rather than tracing the flash op or splitting the graph at the
attention:

- The per-segment SDPA fallback (CPU, fp32) is a Python loop over host
  offsets; traced, it would specialize the graph on every batch
  composition. Behind the op it is invisible to dynamo, so the CPU
  tests exercise the same graph structure as the box.
- The private `_flash_attention_forward` never enters a graph (its
  2.5.1 fallback at torch/_inductor/lowering.py:2380 exists, but its
  meta kernel and `sdpa_constraint` would be one more private surface
  to pin).
- One compiled call per batch instead of two per layer (16 guard
  evaluations and wrapper entries).
- `SymInt max_len` lets the per-batch longest segment flow through
  as a symbol; an `int` schema would guard on its value and recompile
  per batch. The tag makes inductor pass q, k, v with the eager strides
  (the unbind views of the in-projection output, `(3E, head_dim, 1)`;
  the 2.5.1 default for custom ops is `flexible_layout`,
  torch/_inductor/config.py:73, lowering.py:106-127); the kernel reads
  those strides directly and the eager kernel makes the last dim
  unit-stride if inductor ever hands it something else.

### Dynamic shapes and what stays static

`torch.compile(dynamic=True, fullgraph=True)`, plus per call
`mark_dynamic(x, 0)`, `mark_dynamic(cu_seqlens, 0)` and
`mark_static(x, 1)` (torch/_dynamo/variables/builder.py:2560-2660 at
v2.5.1: marked dims win over the default, and a marked-dynamic dim that
dynamo would specialize raises a constraint violation at compile time
instead of recompiling later). Under `dynamic=True`
(`assume_static_by_default=False`, eval_frame.py:286-293) every int
that reaches the function as an argument, a closure cell or an
attribute becomes a symbol (builder.py:1416-1445, wrap_symint
1710-1800; observed on 2.10 as well), so `heads` and `head_dim` are
read from the static shape of the in-projection copy: nn.Parameters
keep static shapes (`force_parameter_static_shapes`, utils.py:2307-2311
and the placeholders observed locally). `eps` is a closure float,
specialized in 2.5.1 (`specialize_float=True`, config.py:64) and a 0-d
tensor input in 2.10 (`specialize_float=False`); either way no
recompile. Observed graph on the local wheel: three symbols (`total`,
`B+1`, `max_len`), every weight static, one cache entry across three
batch compositions.

The call runs under `torch.no_grad()` with `torch.autocast(device,
enabled=False)`: grad mode and autocast state are part of dynamo's
GLOBAL_STATE guard (observed: a call under autocast or with grad on
recompiles), so the wrapper fixes both regardless of the caller. The
nested autocast exit does not clear the cast cache
(torch/amp/autocast_mode.py:363-364 at v2.5.1: only the outermost exit
does), so the heads' autocast casts after the trunk cost what they
cost today.

### Mode

Default inductor mode: no CUDA graphs (`triton.cudagraphs` is off
unless `TORCHINDUCTOR_CUDAGRAPHS=1`, torch/_inductor/config.py:833;
cudagraph trees would record one graph per distinct shape, section
3.4). `max-autotune-no-cudagraphs` is one flag away
(`--compile-packed-mode`) for the GEMM epilogue templates; try it only
after the default mode has a number.

### Fallback detection

- With `fullgraph=True`, graph breaks, the cache-size limit and backend
  failures raise to the caller in 2.5.1: `torch.compile(fullgraph)` is
  `optimize_assert` (eval_frame.py:1602) over `convert_frame_assert`,
  which bypasses the `suppress_errors` swallow of `ConvertFrame`
  (convert_frame.py:1111); the cache-limit path is
  `unimplemented("cache_size_limit reached")` (convert_frame.py:862)
  re-raised from `_compile`. So `--infer-compile`'s global
  `suppress_errors=True` in the same process cannot make this path
  silent. In 2.10 the limit raises `FailOnRecompileLimitHit` (observed).
  `CompiledPackedTrunk.run` catches any exception from the compiled
  call, logs one WARNING and serves its own eager loop from then on.
- Belt and braces after every call: dynamo appends to
  `torch._dynamo.utils.guard_failures[code]` on every guard miss before
  deciding to recompile or give up (guards.py:2754 via
  convert_frame.py:828-833), and `_debug_get_cache_entry_list(code)`
  (eval_frame.py:130-139) counts the compiled variants. A miss with a
  new entry is a recompile (WARNING, `recompiles += 1`, still active);
  a miss without one means dynamo gave up (WARNING once, `active`
  False, eager loop). Both APIs exist on 2.10 and behave the same
  (observed).
- Warmup compiles on the first batch and runs a second, distinct shape
  (one segment cut from the batch: other `total`, offsets of length 2,
  other `max_len`); a specialized graph shows as a second cache entry
  and is logged. `warmup_seconds` is the wall time of both calls.

### Tests

- `tests/test_packed_compile.py` (CPU, 7 tests): compiled equals eager
  packed on three batch compositions with one cache entry and no
  recompile, for `aot_eager` (dynamo side, no compiler needed) and
  `inductor` (skipped without a C++ compiler on PATH); the warmup
  helper; the weight copy follows `load_state_dict`; a failing backend
  falls back to the eager loop with exactly one WARNING; a second
  architecture recompiles (counted, one WARNING) and a third over
  `cache_size_limit` drops to eager (one WARNING); the switch requires
  the packed trunk.
- `tests/test_packed_compile_cuda.py` (skipped without CUDA, one
  tier-B compile shared by the module): compiled bf16 vs eager packed
  bf16 with the differences printed (bound 0.05 of scale, and no
  further from fp32 than the eager packed path plus 1% of scale);
  `torch.cuda.set_sync_debug_mode("error")` around a compiled forward
  after warmup; warmup seconds; GPU and wall ms per batch of 16 for
  eager padded, eager packed, compiled packed (a small table).

### Where the two wheels differ

- Cache-limit failure: `Unsupported` (2.5.1) vs `FailOnRecompileLimitHit`
  (2.10); both are caught by the same `except`.
- `torch._dynamo.utils.counters["frames"]` is empty on 2.10; not used.
- `specialize_float`: True on 2.5.1 (eps constant), False on 2.10 (eps
  a tensor input); no recompile either way.
- `torch.library.register_fake` grew an `allow_override` keyword on
  2.10; the positional form used here is common to both.
- Inductor's CPU backend on this Windows laptop needs MSVC's `cl` on
  PATH (`vcvars64.bat`) and a console code page torch can decode
  (`chcp 1252` or `65001`; the French `cl /help` banner breaks torch
  2.10's cp1252 decode otherwise). Laptop only; the box has gcc.

### Not verifiable here; the box must confirm

1. Warmup time on the tier-B trunk: the dynamic compile of ~100 ops
   with a symbolic token count; section 3.4's 40-90 s was for the full
   forward. Expect 20-60 s cold, 5-10 s with the FX graph cache
   (`TORCHINDUCTOR_CACHE_DIR`, set by `--infer-compile`; set it for the
   pool process too). `warmup_seconds` in the stats says.
2. One cache entry and zero recompiles after warmup and after a pool
   run (`packed_compile.recompiles == 0`, `cache_entries == 1` in the
   JSON): the serve threads share the compiled function; the first
   compile holds dynamo's lock and the second thread waits on it.
3. Inductor emits the custom op as an extern kernel with the unbind
   strides. If it copies q, k, v to contiguous buffers instead, that is
   three extra copy kernels (~0.1 ms at 20k tokens), visible with
   `TORCH_LOGS=output_code`.
4. Parity: compiled vs eager packed bf16 a few 1e-3 of scale (same
   kernel, same cast points; inductor fuses the residual adds and
   LayerNorms), and no further from fp32 than the eager packed path.
5. No implicit sync in a compiled forward after warmup (inductor's
   wrapper reads sizes from the tensors; `assert_size_stride` and the
   custom op do not synchronize).
6. GPU ms per batch of 16 at ~1,270 tokens: section 3.5 predicts -3.5
   to -5 ms of 24 for compiling the whole forward, of which the trunk's
   elementwise glue (5 -> ~2 ms) is the largest part; the heads and the
   priors chain stay eager here. Expect compiled packed = eager packed
   minus 2.5-4 ms, and the serve thread's CPU per batch down by ~2 ms.
   Kill: under 1 ms of GPU gain at batch 16.
7. The weight refresh: after the learner's first `train_step`, the
   compiled path's outputs move with the published weights (the CPU
   test covers the mechanism; the box run covers the threading).
8. `torch.library.define` with a tag, `impl("CompositeExplicitAutograd")`
   and `register_fake` on 2.5.1 (read at the cited lines, not executed).

### Commands on the box

    pytest -s tests/test_packed_compile_cuda.py -p no:cacheprovider
    pytest tests/test_packed_compile.py tests/test_packed_trunk.py tests/test_forward_batch_padded.py
    python tools/bench_pipeline.py --checkpoint training/checkpoints/seed_imit_tierb_start.pt \
        --device cuda --batch-sizes 16,64 --packed-trunk --compile-packed \
        --label packed_compiled --out training/metrics/bench_pipeline/seam_packed_compiled.json
    python tools/bench_pool.py --checkpoint training/checkpoints/seed_imit_tierb_start.pt \
        --actors 19 --games 16 --sims 32 --leaf-batch 16 --server-priors --infer-bf16 \
        --packed-trunk --compile-packed --max-batch 16 --out pool_packed_compiled.json
    python tools/bench_pool.py ... --packed-trunk --compile-packed \
        --compile-packed-mode max-autotune-no-cudagraphs --out pool_packed_autotune.json

Compare the `seam` and `forwards` rows with `seam_packed.json` (section
12), and read `packed_compile` in every JSON: `warmup_seconds`,
`recompiles` (must be 0), `fallback_reason` (must be null),
`cache_entries` (must be 1). GPU-ms per leaf from the pool run as in
section 9.4. Same weights and math up to bf16 rounding, so no strength
check; the `raw:t0` self-match remains the gate before any of this
becomes the pool default.

## 14. Serve-thread host work (2026-09-05): packed embed, batched capacity, length-aware coalescing

Written on the laptop without a GPU, on top of sections 12 and 13. With
the packed trunk the serve threads' own CPU per batch is the larger
part of a batch (docs/box_specs.md, "Pool runs with the saturated
rate"); this section cuts it in three independent pieces, each behind
its own switch or as a pure refactor with a parity test, and adds the
host milliseconds per batch by stage to the pool's log so the next
profile is a log line. Host costs below are CPU microbenchmarks at
production sizes (16 leaves, 1,300 hexes, 30 units, 8 recruits, 4
weapon slots), minimum over repeats on the shared laptop; the box
numbers wait for the commands at the end.

### Where the host milliseconds were (per 16-leaf batch, laptop)

- `_stage_masks` 3.1-4.6 ms, of which the per-leaf `_legal_capacity`
  loop 2.3-4.5 ms (about fifteen small numpy calls per leaf; the
  popcount itself is 0.3 ms either way); the slice writes 0.15 ms
  (0.6 with the combat-oracle attack bias, which is present throughout
  the anneal and after it, `combat_alphas_at` floors at 0.1 of the
  configured alpha); zeroing the buffer under 0.15 ms.
- `unpack_request` (wesnoth_ai/leaf_wire.py) 5.9-6.3 ms: 390 us per
  leaf over 26 fields, mostly `np.prod` on a shape tuple (2 us a call
  against 0.1 for `math.prod`) plus the view and the two dataclass
  constructions. Not touched here (the module is owned elsewhere); it
  is now the `unpack` column of the stats line, and it is the largest
  host item left.
- The padded embed's host side: 17 `np.concatenate` + `from_numpy`
  0.36-0.44 ms, and on CUDA 17 `pin_memory` allocations, 17 copies and
  three pageable copies (the global features and the two faction-id
  tensors), each of which synchronizes the stream (the CUDA runtime
  syncs the stream before a pageable host-to-device copy); the padded
  `build_packed_layout` loop 0.9-1.5 ms; three `pad_sequence` calls,
  3B copy launches.

### What is implemented

1. **Packed embed** (`InferenceServer(packed_embed=True)`,
   `ActorPool(packed_embed=True)`, `bench_pool.py --packed-embed`).
   `GameStateEncoder.encode_from_raw_embedded` writes every numeric
   field of every RawEncoded into ONE pinned host buffer
   (`packed_trunk.FlatLayout`, `np.concatenate(out=)` per field), moves
   it with one non-blocking copy, runs the same embedding expressions as
   `_embed_streams` (shared `_hex_embedding` / `_unit_embedding` /
   `_global_embedding`) on device views of it, and returns
   `packed_trunk.EmbeddedStreams`: the tokens in stream order (every
   sample's hexes, then units, recruits, one global row per sample, one
   end_turn row). `WesnothModel.forward_embedded` orders them on the
   device: with the packed trunk, `build_packed_layout(source="streams")`
   and one `index_select` build the packed [total, d] tensor directly,
   and no padded tensor exists anywhere; with the padded trunk, one
   gather through `padded_gather_index` (pad slots read an appended
   zero row) rebuilds the padded streams exactly as `pad_sequence`
   fills them and `forward_streams` runs unchanged. The padded-stream
   `_forward_streams_packed` and this path share `_packed_trunk_heads`.
   Nothing between the buffer write and the heads waits for the device.
   Host work removed per batch on CUDA: 16 pinned allocations and
   copies, three stream synchronizations, 3B `pad_sequence` launches and
   the `torch.cat` of five padded streams; on CPU the staging alone
   measures 2.4 -> 2.1 ms at a token width of 8 (the CPU has no pins,
   copies or syncs to remove).
2. **Batched capacity** (pure refactor, wesnoth_ai/server_priors.py).
   `_stage_masks` counts the compaction capacity once over the staged
   batch views (`_staged_capacity`: pad slots hold actor_mask 0 and zero
   bits, so the count equals the per-leaf sum exactly) instead of
   calling `_legal_capacity` per leaf: 2.34 -> 0.51 ms, `_stage_masks`
   whole 3.1-4.6 -> 1.5 ms. The popcount uses `np.bitwise_count` where
   numpy 2 provides it and the table otherwise. `_legal_capacity` stays
   as the per-leaf reference (tests). The flat compaction is unchanged.
3. **Vectorized packed layout** (pure refactor, `build_packed_layout`):
   the per-sample loop is now numpy over the whole batch (a ragged
   arange for `src` and `kind`, `np.where` over [B, A_max] / [B, H_max]
   / [B, U_max] for the head arrays): 0.94 -> 0.2-0.4 ms. Applies to both
   embed paths, so the control row of the A/B below already carries it.
4. **Length-aware coalescing** (`ActorPool(coalesce="length",
   coalesce_gap=N)`, `bench_pool.py --coalesce length [--coalesce-gap
   N]`; section 7). `_BatchPicker`, shared by the serve threads, moves
   everything queued into a waiting list and picks each batch under a
   lock. `fifo` is the arrival-order rule the pool always had (the
   control; identical batches). `length`: when everything waiting fits
   one batch, the same; otherwise the oldest request anchors the batch
   and the requests nearest to it in token count fill it (one request is
   one tree on one map, so its token count is its longest leaf, read
   from the PackedRequest headers at intake); `gap` > 0 refuses a
   request further than that many tokens from the anchor even if the
   batch is not full. A request left behind once goes into the next
   batch ahead of the anchor rule, so no request is delayed by more than
   one batch; the count of deferrals (`skipped_requests`) and the mean
   number of waiting requests at a pick (`queue_depth`) are recorded, and
   the policy is inert when the depth is under 2. The failure-reply path
   and the stats are as before.
5. **Stats**: the serve dict carries `unpack` (request views) from the
   pool and, on CUDA, `t_encode`, `t_forward` (launches), `t_priors`
   (launches), `t_finish` (the one wait for the device) and `t_reply`
   (building the ModelOutputs) from `_infer_with_priors`; `run_iteration`
   logs `host ms per batch: unpack= encode= forward= priors= wait=
   reply= wire= put=` plus `requests/batch`, `queue depth` and
   `skipped`, and `bench_pool.py` records them as `host_ms_per_batch`,
   `queue_depth`, `skipped_requests`. Off the device the t_* stages read
   0 (launching and waiting are not separable there).

### Tests

- `tests/test_packed_embed.py` (CPU, 5 tests): the vectorized layout
  equals the per-sample reference (kept in the test) on four size sets
  including rows without recruits, without hexes and without actors; the
  streams source gathers the same tokens as the padded source and
  `padded_gather_index` reproduces `pad_sequence`'s zero-padded streams;
  on real scenario states `forward_embedded` equals `forward_streams` on
  the padded encode TO THE BIT (`torch.equal`) for both trunks; the
  server's `packed_embed` switch leaves every reply identical (values,
  compact actions, priors); the batched capacity equals the per-leaf sum
  on harvested and synthetic packs with pad rows and pad columns.
- `tests/test_batch_picker.py` (7 tests): the fifo rule including the
  overshoot semantics, the length grouping and the one-batch delay
  bound, the gap rule, blocking only on an empty queue, legacy payloads,
  policy validation.
- Existing: tests/test_forward_batch_padded.py, test_packed_trunk.py,
  test_server_priors.py, test_server_priors_staging.py, test_leaf_wire.py
  (22 passed, 4 CUDA skipped) and the slow-tier
  tests/test_actor_pool_smoke.py (both protocols through the real pool
  with the picker in the loop).

### Not verifiable here; the box must confirm

1. The packed embed's CUDA parity: the trunk sees the same bf16 inputs
   (same expressions, same concatenation order, so the same kernels per
   stream), so expect bit-equal outputs or at most the bf16 noise of
   section 12; and `torch.cuda.set_sync_debug_mode("error")` around
   `encode_from_raw_embedded` + `forward_embedded` (one pinned copy, no
   pageable copy).
2. The host milliseconds: the stats line should show `encode` down by
   1-2 ms per batch against the control and `priors` down by ~2 ms
   (capacity), with `unpack` at 4-6 ms the largest remaining item.
3. The saturated rate: each piece against its control on the same box in
   one session, 32 games, 2 serve threads, packed trunk on (the current
   default); a piece is kept when it moves `saturated_leaves_per_s`
   beyond the run-to-run noise of that box (the 833/863 pair of section
   13 puts the noise near 30-40 leaves/s).
4. The coalescing has an effect only when `queue_depth` exceeds 1 (the
   fed regime); read `pad_ratio` and `skipped_requests` next to the rate.
   With the packed trunk the padding costs only in the heads and the
   priors' mask kernels, so a small `pad_ratio` gain may not move the
   rate; that is a valid null result, recorded like the others.

### Commands on the box

    pytest tests/test_packed_embed.py tests/test_batch_picker.py tests/test_packed_trunk.py \
        tests/test_server_priors.py tests/test_server_priors_staging.py tests/test_server_priors_cuda.py
    # control (the vectorized layout and the batched capacity are in every row)
    python tools/bench_pool.py --checkpoint training/checkpoints/seed_imit_tierb_start.pt \
        --actors 32 --games 32 --sims 32 --leaf-batch 16 --server-priors --infer-bf16 \
        --packed-trunk --max-batch 16 --serve-threads 2 --out pool_control.json
    # piece 1
    python tools/bench_pool.py ... --packed-trunk --packed-embed --out pool_packed_embed.json
    # piece 3, on top of whichever embed won
    python tools/bench_pool.py ... --packed-trunk --coalesce length --out pool_coalesce.json
    python tools/bench_pool.py ... --packed-trunk --coalesce length --coalesce-gap 256 \
        --out pool_coalesce_gap256.json

Compare `saturated_leaves_per_s`, `host_ms_per_batch`, `pad_ratio`,
`queue_depth` and `skipped_requests` across the JSONs; the `host ms per
batch` log line gives the same split per iteration. Same weights and
math, so no strength check; the `raw:t0` self-match of
docs/plan_20260904.md remains the gate before any of this becomes the
pool default.

## 15. Option 2 of section 4 implemented (2026-09-14): CUDA graphs over static buckets

Section 4 priced graphs last because the GPU owned the batch at 1,270
tokens per leaf and the padding to buckets cost 15-25% of it. The
relevant-set basis (1.4, 2026-09-11) cut the tokens to ~300, and the
profile of one 16-leaf batch on a 4090 then read 406 kernel launches
for 3.0 ms of device time inside 8.6 ms of host wall (docs/box_specs.md
"The serve batch is launch-bound"): the host feeding the stream is the
cost, exactly the regime section 4.5 reserved graphs for.

`wesnoth_ai/graphed_serve.py` implements section 4.3's static layout
with one difference: the packed varlen trunk of section 12 is kept
(pad segments of one token each, spare rows past the last offset that
no head reads) instead of the end-aligned padded rows, so the trunk's
kernels are section 12's. Per bucket (segments, actor slots, hex slots,
token rows) one `torch.cuda.CUDAGraph` covers the two host->device
copies, the layer loop over a bf16 copy of the weights (section 13's
`PackedTrunkWeights`, refreshed in place on publication), the heads,
`server_priors.priors_outputs` at a fixed compaction capacity and the
device->host copy; the packed embed, the gather of the real tokens, the
numpy staging of the index and mask arrays and the unpacking stay
outside. Batches past a cap take the eager path. Behind
`--graphed-serve` (az_loop, bench_pool, run_elo_batch) and `--graphed`
(eval_inference_server); the box rows are in docs/box_specs.md.
