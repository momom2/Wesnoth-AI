# Tier-a calibration runbook (Kaggle → Vast.ai)

Turnkey command sequence for the first rented/free GPU run. Objective is
**calibration, not a strong model**: prove the CUDA path runs and is fed
(not GPU-starved), fix the D2H stalls that show, and get ONE Elo-vs-compute
point at the scaled net before committing to a paid Tier-b campaign.

Decisions are locked in `docs/superhuman_training_plan.md` §10. Everything
below was validated end-to-end on the CPU laptop except the CUDA-specific
throughput (that's what Phase 1 measures).

- **Model:** `--d-model 256 --num-layers 6 --num-heads 8 --d-ff 1024` (5.0M).
- **Warm-start:** Net2Net-grow the current 471K checkpoint (preserves the
  trained value head; measured value MAE ~0.017 — see plan §10).
- **Fresh campaign:** first launch uses `--reset-decision-step` (weights-
  only; anneal restarts at full strength). **Omit it on a resume.**

---

## Phase 0 — grow the checkpoint (do once, anywhere incl. locally)

```
python tools/net2net.py \
  --in  training/checkpoints/sim_selfplay.pt \
  --out training/checkpoints/tier_a_5m.pt \
  --d-model 256 --num-layers 6 --num-heads 8 --d-ff 1024
```
Commit `tier_a_5m.pt` (or upload it as a Kaggle Dataset) so both phases
start from the same warm net. It carries the vocab + decision_step.

---

## Phase 1 — Kaggle (free): pipeline test + profiling

**Why Kaggle first:** free T4×2, real background execution (Save & Run All /
Commit survives closing the tab), ~30 GPU-h/week. Caveat: only ~4 vCPU (the
rollout will be somewhat starved) and 9–12h session cap → this phase is for
*validating the pipeline and profiling*, not the long run.

Setup: verify phone (unlocks accelerators), enable **GPU T4×2** and
Internet, then `git clone` the repo in the notebook. **Do NOT `pip
install -r requirements.txt`** — it pins `torch-directml` (Windows-only,
would fail/mis-resolve on Kaggle's Linux image); torch, numpy and pytest
are all preinstalled. The clone is self-contained: the pinned scrapes
`unit_stats.json`/`terrain_db.json` AND the sim's runtime WML subset
(`wesnoth_src/data/multiplayer/*`, Mini Maps) are committed, and
`tier_a_5m.pt` is committed under `training/checkpoints/`.
A ready-to-run notebook lives at `kaggle/tier_a_phase1.ipynb`.

**1a. REQUIRED first-run CUDA smoke** (adapted from `running_on_gpu.md`, at
the real 5M arch):
```
python tools/sim_self_play.py --device cuda \
  --mcts --mcts-sims 8 --iterations 2 --games-per-iter 2 --max-turns 12 \
  --mini-ratio 1.0 --replay-buffer --replay-updates 2 --replay-minibatch 16 \
  --replay-min-size 1 --train-batch-size 8 \
  --d-model 256 --num-layers 6 --num-heads 8 --d-ff 1024 \
  --checkpoint-in training/checkpoints/tier_a_5m.pt \
  --checkpoint-out /kaggle/working/gpu_smoke.pt --log-level INFO
```
Confirm: exit 0, no `device`-mismatch errors; the log shows
`mcts_batch_size = 16 (... device=cuda)` and `train_batch_size = 8` with NO
thread-cap line; `nvidia-smi` shows utilization during `train_step`; the
checkpoint saves and re-loads (and writes a `.bak` on the 2nd save).

**1b. Profile the rollout** (find the D2H stalls the CPU laptop couldn't
measure):
```
python tools/profile_rollout.py --device cuda \
  --checkpoint-in training/checkpoints/tier_a_5m.pt \
  --d-model 256 --num-layers 6 --num-heads 8 --d-ff 1024 \
  --games 2 --mcts-sims 32 --mini-ratio 0.5 \
  --save-json /kaggle/working/profile.json --log-level INFO
```
Read the per-stage breakdown. If the in-process rollout's per-leaf
`.item()`/`.tolist()` on GPU tensors dominates (expected), apply the
prepped perf fixes (branch `gpu-perf-fixes`, see BACKLOG §2026-07-01) and
re-profile. Also profile `--mcts-batch-size 8/16/32` to pick B.

**1c. Short end-to-end pipeline run** (a few iterations; confirm the loop,
replay buffer, checkpoint/resume, and the WHR/Elo logging all work on CUDA).
**T4-sized batches** — measured 2026-07-02: minibatch/chunk 128/128 OOM'd
the 15GB T4 at iter 1 (12.7GB allocated, +818MB refused); 64/32 +
`expandable_segments` is the T4 setting. Phase 2's 24GB 4090 keeps 128/128:
```
PYTORCH_ALLOC_CONF=expandable_segments:True \
python tools/sim_self_play.py --device cuda \
  --mcts --mcts-sims 32 --iterations 20 --games-per-iter 8 --max-turns 24 \
  --mini-ratio 0.5 --drill-ratio 0.3 \
  --replay-buffer --replay-updates 16 --value-coef 1.0 \
  --replay-minibatch 64 --replay-capacity 6000 --train-batch-size 32 \
  --d-model 256 --num-layers 6 --num-heads 8 --d-ff 1024 \
  --reset-decision-step \
  --checkpoint-in training/checkpoints/tier_a_5m.pt \
  --checkpoint-out /kaggle/working/tier_a_5m.pt \
  --save-every 2 --log-level INFO
```
**Gate to Phase 2 (all green):** CUDA smoke passes, profiling shows the GPU
is fed (not idling on D2H syncs) after the perf fixes, and held-out value
loss trends down over the 20 iters. If the rollout is hopelessly vCPU-
starved on Kaggle's 4 vCPU, that's expected — Phase 2's high-vCPU host fixes
it; the point of Phase 1 is the pipeline + profile, not throughput.

---

## Phase 2 — Vast.ai spot 4090 (~$30): the calibration run

Rent an **interruptible RTX 4090 on a high-vCPU host** — in the Vast search,
filter `cpu_cores >= 32` (the rollout is CPU-bound; a 6-vCPU host starves the
GPU). Attach a persistent disk for checkpoints. Interruptible is safe: the
checkpoint save is atomic (`.tmp` + `os.replace`) and keeps a `.bak`, and the
resume path falls back to `.bak` if the primary is truncated by a preemption.

**Concrete Vast.ai steps (2026-07-02):**
1. Search: GPU = RTX 4090, **Interruptible**, filters `cpu_cores >= 32`,
   RAM ≥ 32 GB (the replay buffer holds 6000 deep-copied GameStates),
   disk slider ~40 GB, reliability ≥ 99%, inet ≥ 100 Mbps.
2. Template: an official PyTorch CUDA template with **Python ≥ 3.11**
   (the onstart script hard-checks and refuses otherwise).
3. Paste `scripts/vast_onstart.sh` into the template's **On-start
   Script** box. It runs at every (re)start and encodes everything:
   env check, clone, FIRST-launch-vs-resume (`--reset-decision-step`
   only when `tier_a_campaign.pt` doesn't exist yet), the full
   runbook command with all three tripwires, logging to
   `/workspace/train.log`, and an `ABORTED_<rc>` marker that BLOCKS
   auto-relaunch after a tripwire (codes 4/5 need a human).
4. Bid: set ~20–30% above the current interruptible price (~$0.15/h →
   bid ~$0.20/h) so routine outbids are rare; a preemption STOPS the
   instance (disk persists, onstart auto-resumes on restart) — only
   DESTROY loses the disk. Download the checkpoint + CSV before
   destroying.
5. Monitor (from the local machine): `pip install vastai`, set the API
   key, then `vastai show instances`, `vastai logs <id>`, or ssh +
   `tail -f /workspace/train.log`. Pull artifacts with `vastai copy`
   / scp: `training/logs/trainer_history_local.csv` and
   `training/checkpoints/tier_a_campaign.pt`.
6. Watch, in order: `holdout value CE` falling (the ONLY value-learning
   signal), decisive-game % rising, games/hr. Tripwire exits leave
   `ABORTED_4` (all-draws) / `ABORTED_5` (holdout stall) in
   `/workspace` — diagnose from the CSV, remove the marker, restart.

**First launch (fresh campaign — note `--reset-decision-step`):**
```
python tools/sim_self_play.py --device cuda \
  --mcts --mcts-sims 32 \
  --d-model 256 --num-layers 6 --num-heads 8 --d-ff 1024 \
  --replay-buffer --replay-updates 16 --value-coef 1.0 \
    --replay-minibatch 128 --replay-capacity 6000 \
  --train-batch-size 128 --mcts-batch-size 16 \
  --mini-ratio 0.5 --drill-ratio 0.3 \
  --holdout-size 512 \
  --abort-decisive-rate 0.05 --abort-window 40 \
  --abort-holdout-stall 150 \
  --reward-config configs/reward_selfplay.json \
  --reset-decision-step \
  --checkpoint-in  training/checkpoints/tier_a_5m.pt \
  --checkpoint-out training/checkpoints/tier_a_5m.pt \
  --time-budget HH:MM:SS --iterations 100000 --save-every 2 --log-level INFO
```

**Safeguards in that command (added 2026-07-02):**
- `--holdout-size 512` — the first ~512 experiences (whole games) are
  held out of training and the net's value CE on them is logged each
  iter (`holdout value CE=` line + `holdout_value_loss` CSV column).
  THIS is the gate's "held-out value loss trending down" — the train
  value loss is measured on replay samples the net already fit, so it
  can fall by memorization. Watch the two curves together: holdout
  flat while train falls = memorizing, not learning. Caveat: the
  holdout is NOT persisted — a preemption-resume re-collects it from
  post-resume games, restarting that curve's baseline.
- `--abort-holdout-stall 150` — memorization tripwire: if the holdout
  CE makes no new best (min delta 0.01) for 150 consecutive iters
  (several hours at expected iteration times), save + flush + exit
  **code 5**. Motivated by the measured 2026-07-02 signature (train
  value loss 3.8→1.15 with holdout flat at ~3.1). The window is
  deliberately generous: the frozen holdout goes stale as the policy
  improves, so short windows false-trip on long runs.
- `--abort-decisive-rate 0.05 --abort-window 40` — predefined abort:
  if fewer than 5% of games over the trailing 40 iters are decisive
  (non-draw), the run saves a final checkpoint, flushes the CSV, and
  exits with **code 4**. This is the known failure shape (the old
  iter-168 baseline had ZERO leaderkills on full maps); deciding the
  threshold now beats rationalizing at hour two with money burning.
  On a trip: nothing is lost — diagnose (closest_approach / attack%
  trends in the CSV; consider a higher --draw-tiebreak-cap, more
  --mini-ratio, longer --max-turns), then resume from the same
  checkpoint WITHOUT `--reset-decision-step`.

**On a spot preemption / restart — SAME command but DROP `--reset-decision-
step`** (it would restart the anneal mid-run). `--checkpoint-in` ==
`--checkpoint-out` is intentional and now safe (atomic + `.bak`).

Re-tune on the node (hardware-specific — the laptop sweep does NOT carry
over): `--replay-updates` {8,16,32} equal-wall-clock, `--mcts-batch-size`
{8,16,32}, and add `--workers N` (cross-game batching) once single-game
throughput is understood.

**What to measure (the Tier-a deliverable = an Elo-vs-compute point):**
- Strength: `tools/elo_ladder.py` / `tools/whr.py` over periodic checkpoints
  (+ the scripted `dummy` and random floor). Watch the curve MOVE.
- Value learning: held-out value loss trending down.
- Behavior: attack% / decisive-game% / leader-threat on ladder maps (the
  BACKLOG's iter-168 baseline had *zero* leaderkills on full maps).
- Throughput: games/sec and GPU utilization *on that GPU* (compare GPU runs
  to each other, never to laptop logs).

**Gate to Tier-b:** re-estimate Tier-b cost from YOUR measured Elo-vs-compute
slope, not the plan's a-priori numbers. If the curve is flat, diagnose
(capacity? signal? throughput?) before spending Tier-b money.

**Value-learning gate = the HOLDOUT curve, not the train loss.** The
2026-07-02 Kaggle runs measured it directly: train value loss fell
3.77 → ~1.15 over 13 iters while holdout CE plateaued at ~3.1 — the
gap is replay-buffer memorization. On this run, watch
`holdout_value_loss` in the CSV: falling = real value learning;
flat-while-train-falls after the first GPU-hours = stop and diagnose
(signal? capacity?) before burning the rest of the budget.

---

## Cost sanity

- Phase 1 (Kaggle): **$0** (+ free Modal/RunPod credit if you smoke there).
- Phase 2 (Vast spot 4090, ~$0.20–0.35/hr, ~100–150 GPU-h): **~$30**, plus a
  few $ for persistent disk. Delete the disk when done.
