# Migrating supervised training to the ENSTA Mesogip cluster

You handle the password-prompted SSH/SCP steps; the cluster scripts do
the heavy lifting (env install, training, monitoring).

## What's in the bundle

`wai_cluster_bundle.tar.gz` (built by `cluster/build_bundle.ps1`) contains:

- All project Python (root `*.py` + `tools/`)
- `unit_stats.json` (already-scraped unit data)
- `replays_dataset/` (391 MB, 53k replays — the corpus dominates the bundle size)
- The slice of `wesnoth_src/` that `scenario_events.py` reads at runtime
  (just `data/multiplayer/scenarios/` + `data/core/macros/`, ~1 MB)
- `cluster/setup.sh` — venv + torch+CUDA
- `cluster/run.sh` — start/stop/status/tail

## Step-by-step

```powershell
# 1. Verify bundle (~390 MB).
Get-Item wai_cluster_bundle.tar.gz

# 2. Upload (CASCAD password ×2 from the off-site network, ×1 on ENSTA wired).
scp wai_cluster_bundle.tar.gz mesogip_outside:~/

# 3. SSH in and bootstrap.
ssh mesogip_outside
```

On the cluster (one big block — copy/paste-able):

```bash
mkdir -p ~/wesnoth-ai && cd ~/wesnoth-ai
tar -xzf ~/wai_cluster_bundle.tar.gz
bash cluster/setup.sh          # ~2 min: venv, pip, torch+CUDA, sanity check
bash cluster/run.sh             # forks training in background, returns immediately
bash cluster/run.sh status      # confirm it's running and the GPU is hot
```

You can `exit` the SSH session — `nohup` + `disown` keeps the job alive.

## What to look for in `run.sh status`

Healthy:

```
training: RUNNING (pid 12345)
--- last 15 log lines ---
  epoch=0 step=200 avg_loss=8.x pairs=6400 rate=200/s wall=0.5m eta=...
--- GPU utilization ---
0, NVIDIA RTX A4000, 92 %, 8000 MiB, 16 GB
```

If `rate=` is in the hundreds of pairs/sec and the GPU is showing
non-zero utilization, you're winning. CPU pace was ~25 pairs/sec; CUDA
should be ~10× that on any modern card.

If `cuda not available` shows during setup, the wheel index didn't match
the driver — re-run `bash cluster/setup.sh` after editing
`CUDA_TAG=cu118` (older driver) or `cu124` (newer) at the top of the
script.

## Pull the trained model back

After epoch 0 finishes (~5 hr if a 3090-class GPU, ~2 hr on an A100):

```powershell
scp mesogip_outside:~/wesnoth-ai/training/checkpoints/supervised_epoch0.pt training/checkpoints/
```

Then test it locally with whatever in-situ evaluation harness you've
been using.

## Things you can do mid-training

```bash
bash cluster/run.sh status   # how's it going
bash cluster/run.sh tail     # follow the log in real time (Ctrl-C to stop following)
bash cluster/run.sh stop     # SIGTERM the job (saves a checkpoint on the way out via the trainer's signal handler? — actually no, but the most recent periodic ckpt is at most 500 steps old)
```

## If the cluster session times out / disconnect

Training is detached from the shell, so closing the laptop doesn't
matter. Re-SSH and `bash cluster/run.sh status`.

## If you need to resume after a stop

```bash
bash cluster/run.sh stop                              # if a stale job is still around
sed -i 's|--device cuda|--resume training/checkpoints/supervised.pt --device cuda|' cluster/run.sh
bash cluster/run.sh
```

(Or just edit `cluster/run.sh` to add `--resume training/checkpoints/supervised.pt`.)

## If `nvidia-smi` shows no GPU

The cluster's compute nodes vs login node: if Mesogip uses SLURM you'd
need `srun --gres=gpu:1` or similar; the runbook above assumes the GPU
is on the box you ssh into. If not, that's the one place I had to guess
— let me know and I'll write a SLURM `sbatch` wrapper.
