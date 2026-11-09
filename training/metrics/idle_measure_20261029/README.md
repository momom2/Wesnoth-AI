# The RX 6600 on an idle GPU (2026-10-29, 12:32-12:36)

Every rate on 2026-10-29 before this was measured beside the stage 1
generation run. The run was paused at a step boundary, after D0 (its STOP
file, no game lost), for these four readings on an idle GPU. It resumed at
12:34. The code is `feature/local-gpu-stage1` at edd2e903.

| Reading | Value | Contended reading of the day |
|---|---|---|
| `own2`'s forward, the fused fp16 packed path, 200 bench positions | 539.2 positions/s at batch 24, 535.3 at 48 | 535-539 (2026-10-28, idle) |
| Its logits against the padded fp32 trunk, real slots | at most 0.038 (target), actor argmax 200/200 | the same |
| `own2`'s train step, fp16 packed with the HIP backward, batch 32, AdamW | **150.5 positions/s**: forward 67.1, backward 143.5, optimizer 9.1 ms a batch | 88.5-148 (training/metrics/hip_backward_20261029) |
| The four GEMMs at M = 20,480, rocBLAS's default pick | 6.773 ms a layer (qkv 12.1, out 11.0, ff1 9.8, ff2 10.7 TFLOP/s); repeated, 6.765 | |
| The same, PyTorch TunableOp over rocBLAS's solutions | 6.513 ms (-3.8%): qkv slower (1.64 against 1.50 ms), ff1 and ff2 about 8% faster | |

**TunableOp.**
- With its defaults it fails on this GPU: it asks hipBLASLt first, which
  has no gfx1032 algorithms (`HIPBLAS_STATUS_INVALID_VALUE` from
  `getAllAlgos`).
- Restricted to rocBLAS (`PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0`), it finds
  3.8% on the GEMMs, about 2.4% of a forward.
- It tunes each exact (M, N, K), and the serving batches' token count M
  changes every batch, so using it would need M padded to buckets.
- Not worth it; dropped.

So the forward's GEMMs sit at what rocBLAS offers here. An int8 GEMM ran
at twice their rate and failed its accuracy gate
(training/metrics/int8_linear_20261029).

Commands: `measure.log` holds the run.
- The forward: `e2e2.py`, a scratch harness like
  `training/metrics/local_gpu_20261028/bench_local.py`, over the 200
  bench positions and the dataset.
- The train step: `training/metrics/hip_backward_20261029/bench_train.py
  --modes fp16_packed`.
- The GEMMs: `gemm_tune2.py default|tunable <outdir>`, with
  `HIP_VISIBLE_DEVICES=1`.
