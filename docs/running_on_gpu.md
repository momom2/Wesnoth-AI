# Running training on a GPU / cloud compute

The self-play trainer (`tools/sim_self_play.py`) is device-aware and
runs on a CUDA GPU. This doc is how to launch it on GPU compute and
the gotchas that differ from the local CPU laptop. It is **not** a
cluster-orchestration setup — the old ENSTA SLURM/sync infra was
decommissioned (2026-06); bring your own node/scheduler and just
invoke the entrypoint.

## Why the laptop's tuning does NOT transfer

The dev laptop is **CPU**, where the model forward dominates and
batching is expensive (128 samples ≈ 128× work). A **CUDA GPU** is
the opposite: forwards are cheap and batching is nearly free
(128 samples ≈ 1 batched call). That inverts the rollout:training
cost ratio, so several laptop settings are wrong for GPU and a couple
of defaults are made device-aware automatically:

| Knob | CPU (laptop) | CUDA (GPU) | Notes |
|---|---|---|---|
| `--train-batch-size` | 1 (auto) | **128 (auto)** | THE key GPU knob — batches the forward. Raise on a big GPU, lower on OOM. |
| `--torch-threads` | capped to ~4 (auto) | **uncapped (auto)** | Capping throttles the host-side MCTS/encoding that feeds the GPU. Don't set it on GPU. |
| compute-optimal `--replay-updates` | (laptop sweep said ~16) | **re-tune on GPU** | The optimum depends on the rollout:train cost ratio, which is hardware-specific — the laptop sweep result does NOT carry over. |
| wall-clock numbers | laptop CPU | irrelevant | Compare GPU runs to each other, not to laptop logs. |

The auto-defaults key off `--device`, so the minimal correct GPU
launch is just `--device cuda` (it picks `train_batch_size=128` and
skips the thread cap). Everything below is tuning on top.

## Launch

```
python tools/sim_self_play.py \
  --device cuda \
  --checkpoint-in training/checkpoints/sim_selfplay.pt \
  --checkpoint-out training/checkpoints/sim_selfplay.pt \
  --mcts --mcts-sims 32 \
  --replay-buffer --replay-updates 16 --value-coef 1.0 \
    --replay-minibatch 128 --replay-capacity 6000 \
  --train-batch-size 128 \
  --mini-ratio 0.5 --drill-ratio 0.3 \
  --time-budget HH:MM:SS --iterations 100000 --save-every 2
```

- **Replay buffer** is the validated training-efficiency lever
  (held-out value loss 3.44→3.04 in 20 iters vs flat for one-pass;
  see BACKLOG "Experience replay"). `enabled` only via
  `--replay-buffer`; off = exact legacy one-pass.
- **`--replay-updates`**: gradient steps per iteration. More = more
  value-head convergence but fewer fresh games per unit compute.
  Re-tune on the GPU (an equal-wall-clock sweep over {8,16,32}).
- **`--value-coef 1.0`**: the value head is the diagnosed bottleneck;
  1.0 weighted it slightly better than 0.5 in the laptop sweep.
- Checkpoints are device-portable (`load_checkpoint` loads on CPU via
  `map_location` then transfers to the model's device), so a
  laptop-warm-started checkpoint resumes fine on GPU and vice versa.

## REQUIRED first-run smoke on the GPU

The CUDA path could not be exercised on the dev laptop (no NVIDIA
GPU). On the first GPU node, run a 1-2 iteration smoke BEFORE a long
run and confirm:

```
python tools/sim_self_play.py --device cuda --mcts --mcts-sims 8 \
  --iterations 2 --games-per-iter 2 --max-turns 12 --mini-ratio 1.0 \
  --replay-buffer --replay-updates 2 --replay-minibatch 16 \
  --replay-min-size 1 --train-batch-size 8 \
  --checkpoint-out /tmp/gpu_smoke.pt --log-level INFO
```

Check: (1) exit 0, no `device`-mismatch errors; (2) `nvidia-smi`
shows GPU utilization during the train_step; (3) the log shows
`train_batch_size = 8` and NO thread-cap line; (4) the checkpoint
saves and re-loads. The batched forward path (B>1) + replay + the
vectorized policy loss were all verified together on CPU at B=8, so
the same code at B=128 on CUDA should just work — but confirm device
placement once on real hardware.

## Tuning / OOM

- **OOM**: lower `--train-batch-size` first, then `--replay-capacity`
  (each buffered experience holds a deepcopied game state).
- **Underutilized GPU**: raise `--train-batch-size` (bigger batched
  forward) and/or `--replay-updates`; tune `--mcts-batch-size` (B=8-32)
  so each Gumbel sequential-halving phase evaluates its leaves through one
  batched forward instead of B=1-per-sim (intra-search batching; was a
  no-op before the 2026-06-29 fix). **As of 2026-07-01 `--mcts-batch-size`
  defaults device-aware (B=16 on CUDA, B=1 on CPU)** so leaf batching is on
  by default on GPU; pass an explicit value to tune. And/or `--workers N`
  for parallel rollout (cross-game batching). The batching levers compose;
  profile to pick B.
- **First-GPU-node profiling TODO (see BACKLOG §2026-07-01 "DEFERRED"):**
  several CUDA-only D2H-sync stalls could not be measured on the CPU
  laptop and were recorded, not applied — the in-process rollout's
  per-leaf `.item()`/`.tolist()` on GPU tensors (biggest; adopt the
  actor-pool's sampler-on-CPU split), B2 (per-leaf value/cliffness batch
  read), B3 (pinned H2D). Profile with `tools/profile_rollout.py` on the
  node and fix the stalls that actually show before the long campaign.
- **Fresh campaign vs. resume**: start a NEW training run (treating an old
  checkpoint as weights-only) with `--reset-decision-step` so the
  combat-oracle anneal restarts from full strength; OMIT it when resuming
  an in-progress run (it would restart the anneal mid-training).
- **Throughput metric**: held-out value loss + attack%/decisive vs
  wall-clock *on that GPU*. There's a held-out eval recipe in the
  BACKLOG / git history (load checkpoints, score on a fixed set).

## What is CPU-laptop-specific (guard / ignore on GPU)

- `--torch-threads` (optimization #1) — CPU-only; auto-skipped on GPU.
- The `dropout=1e-4` fast-path note in `model.py` is about
  torch-directml, harmless on CUDA.
- Any wall-clock or iters/hour figure in the logs or BACKLOG.
