"""Prereg departure 33's watcher: if a design stop ends the rounds, the GPU
does not idle to the window's end. Every step is logged to
C:/Users/momom2/wesnoth_runs/after_stop.log. Run detached on the
repository's .venv:

    python C:/Users/momom2/wesnoth_runs/tools/after_stop.py --head <commit with chain_local --after-stop>

1. Wait for the chain (chain_local.py without --after-stop) to exit.
2. Act only on a design stop: rounds_local/STOP names "STOP after round K0"
   or "PAUSE after round K0", and the chain's log shows the final match (and
   the tail) done after it. Anything else is logged and left to the lead.
3. The paired arm reading on round K0's own data, when it and one trunk round
   fit before --rounds-end-by: a full-network pass in one window per
   game-side (pg_pass --window 10^7) at the policy that played round K0's
   games, then pg_round --arm reduced_adamw at round K0's radius, beside
   round K0's files (trunk_*). The arm the rounds go on with is
   reduced_adamw (in one window) iff its best half-A reading is at least the
   heads-only step's on the same half; heads_only otherwise, or when the
   reading fails (then a GPU health check first, and nothing more on a
   failure).
4. Re-arm the chain at --head with --after-stop K0 (arm_after_stop.ps1).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RUNS = Path("C:/Users/momom2/wesnoth_runs")
ROUNDS = RUNS / "rounds_local"
RUN_TREE = Path("C:/Users/momom2/Desktop/perso/projects/wesnoth_ai-run")
GPU_PY = "C:/Users/momom2/venvs/wesnoth_rocm/Scripts/python.exe"
LOG = RUNS / "after_stop.log"
FMT = "%Y-%m-%d %H:%M"
WINDOW = 10 ** 7                         # one window per game-side
PAIRED_H, TRUNK_ROUND_H, HEADS_ROUND_H = 13.0, 19.0, 12.0     # the walls planned (the 2026-11-02 21:45 review)
PASS_FP16 = ["--fp16", "--packed-trunk", "--floor-chunk", "8", "--floor-sides", "100"]   # rounds_local's
GPU_ENV = {"HIP_VISIBLE_DEVICES": "1", "AMD_SERIALIZE_KERNEL": "3"}


def log(msg: str) -> None:
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def chain_alive() -> bool:
    q = ("@(Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and "
         "$_.CommandLine -like '*chain_local.py*' -and $_.CommandLine -notlike '*--after-stop*' }).Count")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", q], capture_output=True, text=True, timeout=120)
        return r.returncode != 0 or not r.stdout.strip() or int(r.stdout.strip()) > 0
    except (subprocess.TimeoutExpired, ValueError, OSError):
        return True


def design_stop() -> int:
    """K0 when rounds_local's STOP is a design stop, else 0."""
    stop = ROUNDS / "STOP"
    why = stop.read_text(encoding="utf-8") if stop.exists() else ""
    m = re.search(r"(?:STOP|PAUSE) after round (\d+)", why)
    return int(m.group(1)) if m else 0


def round_json(k: int) -> dict:
    return json.loads((ROUNDS / f"round_{k:02d}" / "round.json").read_text(encoding="utf-8"))


def policy_after(k: int, pi0: str) -> str:
    while k >= 1:
        if (round_json(k).get("candidate") or {}).get("path"):
            return str(ROUNDS / f"round_{k:02d}" / "pi.pt")
        k -= 1
    return pi0


def best_half_a(rj: dict) -> float:
    vals = [r["halves"]["A"]["exact"]["estimate"] for r in rj["decision"].get("rungs") or []
            if (r.get("halves") or {}).get("A", {}).get("exact")]
    return max(vals) if vals else float("-inf")


def gpu_tool(cmd, log_name: str, cut_min: float) -> int:
    env = dict(os.environ, **GPU_ENV)
    with open(RUNS / log_name, "a", encoding="utf-8") as f:
        f.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        f.flush()
        p = subprocess.Popen([GPU_PY, "-u", *cmd], cwd=RUN_TREE, env=env, stdout=f, stderr=subprocess.STDOUT)
        try:
            return p.wait(timeout=cut_min * 60)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
            return 124


def gpu_healthy() -> bool:
    code = "import torch; x = torch.ones(1024, device='cuda'); torch.cuda.synchronize(); print(float(x.sum()))"
    try:
        r = subprocess.run([GPU_PY, "-c", code], env=dict(os.environ, **GPU_ENV), capture_output=True, text=True,
                           timeout=300)
        return r.returncode == 0 and r.stdout.strip() == "1024.0"
    except subprocess.TimeoutExpired:
        return False


def paired_reading(k0: int, pi0: str) -> str:
    """Step 3: the arm the rounds go on with ("reduced_adamw" or
    "heads_only"), or "" when the GPU fails its health check."""
    d, prev = ROUNDS / f"round_{k0:02d}", ROUNDS / f"round_{k0 - 1:02d}"
    pi = policy_after(k0 - 1, pi0)
    dirs = [str(p) for p in sorted((d / "play").glob("games_s*")) if p.is_dir()]
    labels = d / "play" / "labels.jsonl"
    log(f"PAIRED: round {k0}'s games ({len(dirs)} dirs) at {pi}: the full-network pass in one window")
    rc = gpu_tool(["tools/pg_pass.py", "--checkpoint", pi, "--games", *dirs, "--labels", str(labels), "--out",
                   str(d / "trunk_pass.pt"), "--device", "cuda", "--workers", "6", *PASS_FP16, "--window", str(WINDOW),
                   "--corrections", str(d / "corrections.json")], "after_stop_pass.log", 12 * 60)
    if rc == 0:
        factory = []
        for key, value in (("checkpoint", pi), ("games", ",".join(dirs)), ("labels", labels), ("device", "cuda"),
                           ("workers", "6"), ("fp16", "1"), ("cache", d / "trunk_eval_cache")):
            factory += ["--factory-arg", f"{key}={value}"]
        previous = ["--previous", str(prev / "direction.pt")] if (prev / "direction.pt").exists() else []
        rc = gpu_tool(["tools/pg_round.py", "--pass", str(d / "trunk_pass.pt"), "--factory",
                       "tools.pg_eval:network_factory", *factory, "--eps-from", str(prev / "round.json"), "--arm",
                       "reduced_adamw", *previous, "--out", str(d / "trunk_round.json"), "--candidate",
                       str(d / "trunk_pi.pt"), "--direction-out", str(d / "trunk_direction.pt"), "--chosen-network",
                       str(d / "trunk_rung.pt")], "after_stop_step.log", 16 * 60)
    if rc != 0 or not (d / "trunk_round.json").exists():
        log(f"PAIRED_FAILED rc={rc} (after_stop_pass.log, after_stop_step.log)")
        if not gpu_healthy():
            log("AFTER_STOP_HELD: the GPU fails its health check after the paired reading; nothing more runs")
            return ""
        return "heads_only"
    trunk, heads = best_half_a(json.loads((d / "trunk_round.json").read_text(encoding="utf-8"))), \
        best_half_a(round_json(k0))
    arm = "reduced_adamw" if trunk >= heads else "heads_only"
    log(f"PAIRED: half A of round {k0}: reduced_adamw in one window {trunk:+.4f}, heads_only {heads:+.4f}: "
        f"the rounds go on with {arm}")
    return arm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True, help="the commit with chain_local --after-stop (prereg departure 33)")
    ap.add_argument("--rounds-end-by", default="2026-11-08 02:00")
    ap.add_argument("--pi0", default=str(RUN_TREE / "training" / "checkpoints" / "own2.pt"))
    a = ap.parse_args()
    log(f"AFTER_STOP waiting for the chain to exit (head {a.head}, rounds end by {a.rounds_end_by})")
    while chain_alive():
        time.sleep(300)
    k0 = design_stop()
    chain_log = (RUNS / "chain.log").read_text(encoding="utf-8") if (RUNS / "chain.log").exists() else ""
    tail = chain_log[chain_log.rfind("CHAIN_DONE"):] if "CHAIN_DONE" in chain_log else ""
    log(f"the chain exited; design stop at round {k0 or 'none'}; its log ends: "
        f"{chain_log.strip().splitlines()[-1] if chain_log.strip() else '(empty)'}")
    if not k0:
        log("AFTER_STOP_IDLE: no design stop ended the rounds; nothing to do")
        return 0
    if "CHAIN_FINAL_DONE: match_local returned 0" not in tail or "CHAIN_TAIL_DONE: match_local returned 0" not in tail:
        log("AFTER_STOP_HELD: the final match or the tail did not return 0 after the stop; the lead looks")
        return 1
    left_h = (datetime.strptime(a.rounds_end_by, FMT).timestamp() - time.time()) / 3600
    subprocess.run(["git", "checkout", "-q", "--detach", a.head], cwd=RUN_TREE, check=True)
    if left_h >= PAIRED_H + TRUNK_ROUND_H:
        arm = paired_reading(k0, a.pi0)
        if not arm:
            return 1
    elif left_h >= HEADS_ROUND_H:
        arm = "heads_only"
        log(f"{left_h:.1f} h left: no room for the paired reading and a trunk round; heads_only")
    else:
        log(f"AFTER_STOP_IDLE: {left_h:.1f} h left before {a.rounds_end_by}: no round fits")
        return 0
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                        str(RUNS / "tools" / "arm_after_stop.ps1"), "-Head", a.head, "-K0", str(k0), "-Arm", arm,
                        "-EndBy", a.rounds_end_by], capture_output=True, text=True)
    log(f"AFTER_STOP_ARMED at {a.head}, round {k0}, arm {arm}: "
        f"{r.stdout.strip().splitlines()[0] if r.stdout.strip() else r.stderr[-300:]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
