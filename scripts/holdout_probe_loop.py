"""Periodic human-holdout CE probe for self-play campaign legs.

THE observable for the imitation->self-play handoff (BACKLOG item 3;
A1/F1 rulings 2026-08-10): does self-play keep the imitation prior's
human-play CE? Every PROBE_EVERY seconds this loop:

  1. peeks the rolling campaign checkpoint's decision_step;
  2. if it advanced since the last probe, snapshots the checkpoint
     (copy, so the trainer's atomic replace can't race the read);
  3. runs `tools/supervised_train.py <dataset> --resume <snap>
     --imitation-config configs/imitation.json --eval-only
     --eval-json ...` in a subprocess (crash-isolated from the
     learner) at the checkpoint's own arch;
  4. appends (utc, decision_step, ce, per-head top1, value_auc, n)
     to training/logs/holdout_probe.csv -- which hf_upload_loop
     escrows to HF, so the CE-vs-step CURVE survives the box.

Reference point: imit_tierb_start.pt = holdout CE 3.102 at handoff
t0. The pre-registered A1 observable is this curve staying flat.

Env knobs: CAMPAIGN_CKPT (default training/checkpoints/$CAMPAIGN_FILE
or tier_b_15m.pt), PROBE_EVERY (s, default 3600), PROBE_PAIRS
(default 1200), PROBE_DEVICE (default cpu -- keep the GPU for the
learner), IMITATION_DATASET (default from configs/imitation.json).

Run from the repo root (vast_onstart launches it next to
hf_upload_loop):
    nohup python scripts/holdout_probe_loop.py >> probe.log 2>&1 &
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PROBE_EVERY = int(os.environ.get("PROBE_EVERY", "3600"))
PROBE_PAIRS = int(os.environ.get("PROBE_PAIRS", "1200"))
PROBE_DEVICE = os.environ.get("PROBE_DEVICE", "cpu")
OUT_CSV = Path(os.environ.get(
    "PROBE_CSV", "training/logs/holdout_probe.csv"))

# Abort tripwire (2026-08-12 diagnosis, finding F4): the human-holdout
# CE is the only observable that showed the leg regressing, and it was
# wired to nothing -- the leg ran for days getting worse with every
# guard green. When PROBE_T0 is set and PROBE_ABORT_N consecutive
# probes read CE > PROBE_T0 + PROBE_ABORT_DELTA, this loop writes
# $WORKDIR/ABORTED_probe (evidence) and SIGKILLs the training
# processes; the onstart supervisor treats a signal exit without a
# watchdog marker as an operator stop and stands down. Off unless
# PROBE_T0 is set (the baseline is leg-specific: measure the seed
# checkpoint under this probe's own protocol first).
PROBE_T0 = os.environ.get("PROBE_T0")
PROBE_ABORT_DELTA = float(os.environ.get("PROBE_ABORT_DELTA", "0.5"))
PROBE_ABORT_N = int(os.environ.get("PROBE_ABORT_N", "3"))

_COLS = ["timestamp", "decision_step", "ce", "actor_top1", "type_top1",
         "target_top1", "weapon_top1", "value_auc", "n", "n_value",
         "probe_seconds"]


def _campaign_ckpt() -> Path:
    p = os.environ.get("CAMPAIGN_CKPT")
    if p:
        return Path(p)
    name = os.environ.get("CAMPAIGN_FILE", "tier_b_15m.pt")
    return Path("training/checkpoints") / name


def _peek(ckpt: Path):
    """(decision_step, arch dict) or None if unreadable."""
    import torch
    try:
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    except Exception as e:                          # noqa: BLE001
        print(f"probe: ckpt unreadable ({e!r})", flush=True)
        return None
    return int(raw.get("decision_step", 0)), dict(raw.get("arch") or {})


def _append(row: dict) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUT_CSV.exists() or OUT_CSV.stat().st_size == 0
    with OUT_CSV.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def probe_once(ckpt: Path, step: int, arch: dict) -> bool:
    """Snapshot + eval-only subprocess + CSV append. True on success."""
    dataset = os.environ.get("IMITATION_DATASET")
    if not dataset:
        cfg = json.loads(Path("configs/imitation.json")
                         .read_text(encoding="utf-8"))
        dataset = cfg.get("dataset_dir", "replays_dataset_imitation")
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="holdout_probe_") as td:
        snap = Path(td) / ckpt.name
        shutil.copy2(ckpt, snap)
        out_json = Path(td) / "eval.json"
        cmd = [sys.executable, "tools/supervised_train.py", dataset,
               "--resume", str(snap),
               "--imitation-config", "configs/imitation.json",
               "--eval-only", "--eval-json", str(out_json),
               "--eval-pairs", str(PROBE_PAIRS),
               "--device", PROBE_DEVICE]
        for k, flag in (("d_model", "--d-model"),
                        ("num_layers", "--num-layers"),
                        ("num_heads", "--num-heads"),
                        ("d_ff", "--d-ff")):
            if k in arch:
                cmd += [flag, str(arch[k])]
        r = subprocess.run(cmd, cwd=str(REPO), capture_output=True,
                           text=True, timeout=7200)
        if r.returncode != 0 or not out_json.exists():
            print(f"probe: eval failed rc={r.returncode}; stderr tail: "
                  f"{r.stderr[-500:]}", flush=True)
            return False
        stats = json.loads(out_json.read_text(encoding="utf-8"))
    row = {"timestamp": time.strftime("%FT%TZ", time.gmtime()),
           "decision_step": step,
           "probe_seconds": round(time.time() - t0, 1), **stats}
    _append(row)
    print(f"probe: step={step} ce={stats.get('ce'):.4f} "
          f"value_auc={stats.get('value_auc')} "
          f"({row['probe_seconds']}s)", flush=True)
    return True


def _abort_check() -> None:
    """Kill training when the last PROBE_ABORT_N probe CEs all exceed
    PROBE_T0 + PROBE_ABORT_DELTA. No-op unless PROBE_T0 is set."""
    if PROBE_T0 is None or not OUT_CSV.exists():
        return
    import csv as _csv
    import signal
    rows = list(_csv.DictReader(OUT_CSV.open(encoding="utf-8")))
    if len(rows) < PROBE_ABORT_N:
        return
    bar = float(PROBE_T0) + PROBE_ABORT_DELTA
    tail = rows[-PROBE_ABORT_N:]
    if not all(float(r["ce"]) > bar for r in tail):
        return
    workdir = Path(os.environ.get("WORKDIR", "/workspace"))
    marker = workdir / "ABORTED_probe"
    marker.write_text(
        f"{time.strftime('%FT%TZ', time.gmtime())} human-holdout CE > "
        f"{bar:.3f} (t0 {PROBE_T0} + {PROBE_ABORT_DELTA}) on "
        f"{PROBE_ABORT_N} consecutive probes: "
        f"{[round(float(r['ce']), 4) for r in tail]} at steps "
        f"{[r['decision_step'] for r in tail]}\n", encoding="utf-8")
    print(f"probe: ABORT TRIPWIRE -- {marker.read_text().strip()}",
          flush=True)
    for pd in Path("/proc").iterdir():
        if not pd.name.isdigit():
            continue
        try:
            if b"tools/sim_self_play.py" in (pd / "cmdline").read_bytes() \
                    and (pd / "comm").read_text().strip() == "python":
                os.kill(int(pd.name), signal.SIGKILL)
        except OSError:
            continue


def main() -> int:
    ckpt = _campaign_ckpt()
    print(f"holdout_probe_loop: watching {ckpt}, every {PROBE_EVERY}s, "
          f"{PROBE_PAIRS} pairs on {PROBE_DEVICE}"
          + (f"; abort at t0+{PROBE_ABORT_DELTA} x{PROBE_ABORT_N} "
             f"(t0={PROBE_T0})" if PROBE_T0 else "; abort OFF (no "
             "PROBE_T0)"), flush=True)
    last_step = None
    while True:
        try:
            if ckpt.exists():
                peek = _peek(ckpt)
                if peek is not None and peek[0] != last_step:
                    if probe_once(ckpt, *peek):
                        last_step = peek[0]
                        _abort_check()
        except Exception as e:                      # noqa: BLE001
            print(f"probe: cycle failed: {e!r}", flush=True)
        time.sleep(PROBE_EVERY)


if __name__ == "__main__":
    sys.exit(main())
