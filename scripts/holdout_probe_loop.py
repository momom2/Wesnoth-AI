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
# Stratified probe (2026-08-25 instrument repair): the pooled
# first-N-pairs stream ran ~3 games deep (60% of AUC comparisons
# inside ONE game; Hanley-McNeil CI invalid). Cap pairs per game so
# PROBE_PAIRS spans ~PROBE_PAIRS/cap games; statistics become
# per-game with between-game SE. Load-bearing default lives here in
# code. NOTE: changing the sampling changes the CE/AUC BASELINE --
# PROBE_T0 and floors must be re-derived under this instrument
# (leg-5 resume did; see docs/leg5_value_inversion_20260825.md).
PROBE_PAIRS_PER_GAME = int(os.environ.get("PROBE_PAIRS_PER_GAME", "8"))
PROBE_DEVICE = os.environ.get("PROBE_DEVICE", "cpu")
OUT_CSV = Path(os.environ.get(
    "PROBE_CSV", "training/logs/holdout_probe.csv"))

# CE ABORT REMOVED (user ruling 2026-08-25): "learning to play
# better does not necessarily mean learning how humans play." The
# human-play CE stays in the CSV as telemetry; it no longer kills
# training. PROBE_T0 is kept only as a logged reference level.
PROBE_T0 = os.environ.get("PROBE_T0")
PROBE_ABORT_N = int(os.environ.get("PROBE_ABORT_N", "3"))

# Value-accuracy alarm (A1, credit-assignment review 2026-08-17).
# Value tripwire (user ruling 2026-08-25, replacing the
# consecutive-hourly-probes design): each probe reads the model's
# winner-recognition over ~150 held-out human games (stratified,
# between-game SE). If a reading is below PROBE_AUC_FLOOR (0.60),
# the probe IMMEDIATELY redraws with different sampling RNG, up to
# PROBE_ABORT_N (3) total independent draws; only if ALL are below
# the floor does it abort training (marker + SIGKILL). The
# entry-qualification gate (--qualify CKPT) is unchanged: one
# probe, exit 0 iff value_auc >= QUALIFY_AUC_MIN.
PROBE_AUC_FLOOR = float(os.environ.get("PROBE_AUC_FLOOR", "0.60"))
QUALIFY_AUC_MIN = float(os.environ.get("QUALIFY_AUC_MIN", "0.60"))

_COLS = ["timestamp", "decision_step", "ce", "ce_se", "actor_top1",
         "type_top1", "target_top1", "weapon_top1", "value_auc",
         "value_auc_se", "n", "n_value", "n_auc_games",
         "n_ce_games", "probe_seconds"]


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


def probe_once(ckpt: Path, step: int, arch: dict,
               sample_seed: int = None):
    """Snapshot + eval-only subprocess + CSV append. Returns the
    stats dict on success, None on failure. `sample_seed` selects
    WHICH pairs each holdout game contributes (the independent-
    redraw mechanism)."""
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
               "--eval-pairs-per-game", str(PROBE_PAIRS_PER_GAME),
               "--device", PROBE_DEVICE]
        if sample_seed is not None:
            cmd += ["--eval-sample-seed", str(sample_seed)]
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
            return None
        stats = json.loads(out_json.read_text(encoding="utf-8"))
    row = {"timestamp": time.strftime("%FT%TZ", time.gmtime()),
           "decision_step": step,
           "probe_seconds": round(time.time() - t0, 1), **stats}
    _append(row)
    def _pm(v, se, fmt=".4f"):
        if v is None:
            return "n/a"
        s = f"{v:{fmt}}"
        return s + (f"±{se:{fmt}}" if se is not None else "")
    print(f"probe: step={step} "
          f"ce={_pm(stats.get('ce'), stats.get('ce_se'))} "
          f"value_auc={_pm(stats.get('value_auc'), stats.get('value_auc_se'), '.3f')} "
          f"seed={sample_seed} ({row['probe_seconds']}s)", flush=True)
    return stats


def redraw_verdict(readings, floor, n) -> bool:
    """Abort predicate (user ruling 2026-08-25): True iff ALL n
    INDEPENDENT redraws read a value_auc below `floor`. A missing
    reading (probe failure / no value pairs) can never abort."""
    if len(readings) < n:
        return False
    clean = [r for r in readings if r is not None]
    return len(clean) >= n and all(r < floor for r in clean[:n])


def _value_check(ckpt: Path, step: int, arch: dict,
                 first: dict) -> None:
    """The value tripwire (user ruling 2026-08-25): if this probe's
    value_auc reads below PROBE_AUC_FLOOR, immediately REDRAW the
    sample with different RNG, up to PROBE_ABORT_N total
    independent draws; abort training only if every draw is below
    the floor. (The CE abort was REMOVED the same day: playing
    better does not necessarily mean playing like humans --
    human-similarity is telemetry, not a kill-switch.)"""
    import signal

    def _auc(stats):
        v = stats.get("value_auc") if stats else None
        return float(v) if v not in (None, "") else None

    readings = [_auc(first)]
    if readings[0] is None or readings[0] >= PROBE_AUC_FLOOR:
        return
    for k in range(1, PROBE_ABORT_N):
        print(f"probe: value_auc {readings[-1]:.3f} < "
              f"{PROBE_AUC_FLOOR} -- independent redraw "
              f"{k}/{PROBE_ABORT_N - 1}", flush=True)
        stats = probe_once(ckpt, step, arch, sample_seed=step + k)
        readings.append(_auc(stats))
        if readings[-1] is not None and readings[-1] >= PROBE_AUC_FLOOR:
            return
    if not redraw_verdict(readings, PROBE_AUC_FLOOR, PROBE_ABORT_N):
        return
    reason = (f"value_auc < {PROBE_AUC_FLOOR} on {PROBE_ABORT_N} "
              f"INDEPENDENT sample redraws: "
              f"{[round(r, 4) for r in readings]}")
    workdir = Path(os.environ.get("WORKDIR", "/workspace"))
    marker = workdir / "ABORTED_probe"
    marker.write_text(
        f"{time.strftime('%FT%TZ', time.gmtime())} {reason} at step "
        f"{step}\n", encoding="utf-8")
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


def qualify_verdict(stats: dict, auc_min: float) -> tuple:
    """(passed, reason) for the entry-qualification gate. A missing
    value_auc REFUSES (an unmeasured judge is an unproven judge)."""
    v = stats.get("value_auc")
    if v in (None, ""):
        return False, "value_auc missing from probe output"
    v = float(v)
    if v < auc_min:
        return False, f"value_auc {v:.3f} < required {auc_min:.2f}"
    return True, f"value_auc {v:.3f} >= {auc_min:.2f}"


def qualify(ckpt: Path) -> int:
    """Entry-qualification gate: probe `ckpt` once; exit 0 iff its
    value_auc clears QUALIFY_AUC_MIN. The launch script gates on
    this exit code (A1: leg 3 launched with a 0.309-AUC judge and
    nobody looked). Exit 3 = refused, 2 = probe failed."""
    peek = _peek(ckpt)
    if peek is None:
        print(f"qualify: {ckpt} unreadable", flush=True)
        return 2
    step, arch = peek
    # probe_once appends the row (audit trail) and prints the stats.
    # Same seeded-random sampling as the hourly tripwire, so the
    # gate and the tripwire read the same instrument.
    if not probe_once(ckpt, step, arch, sample_seed=step):
        print("qualify: probe run failed", flush=True)
        return 2
    import csv as _csv
    rows = list(_csv.DictReader(OUT_CSV.open(encoding="utf-8")))
    passed, reason = qualify_verdict(rows[-1], QUALIFY_AUC_MIN)
    print(f"qualify: {'PASS' if passed else 'REFUSE'} -- {reason} "
          f"(step {step})", flush=True)
    return 0 if passed else 3


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--qualify":
        return qualify(Path(sys.argv[2]))
    ckpt = _campaign_ckpt()
    print(f"holdout_probe_loop: watching {ckpt}, every {PROBE_EVERY}s, "
          f"{PROBE_PAIRS} pairs (cap {PROBE_PAIRS_PER_GAME}/game) on "
          f"{PROBE_DEVICE}; CE = telemetry only (t0 ref {PROBE_T0})"
          + f"; value_auc floor {PROBE_AUC_FLOOR} x{PROBE_ABORT_N}",
          flush=True)
    last_step = None
    while True:
        try:
            if ckpt.exists():
                peek = _peek(ckpt)
                if peek is not None and peek[0] != last_step:
                    stats = probe_once(ckpt, *peek,
                                       sample_seed=peek[0])
                    if stats is not None:
                        last_step = peek[0]
                        _value_check(ckpt, *peek, first=stats)
        except Exception as e:                      # noqa: BLE001
            print(f"probe: cycle failed: {e!r}", flush=True)
        time.sleep(PROBE_EVERY)


if __name__ == "__main__":
    sys.exit(main())
