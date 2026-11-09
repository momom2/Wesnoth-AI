"""After round 1 alone (prereg departure 30): wait for its rounds_local to
exit, apply departure 31's kill and departure 32's GPU check, then re-arm
the chain with --skip-p at the commit the check allows. Every step is logged
to C:/Users/momom2/wesnoth_runs/after_round1.log. Run detached on the
repository's .venv:

    python C:/Users/momom2/wesnoth_runs/tools/after_round1.py --fixed 8eb24002 --plain 6f86c363
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

RUNS = Path("C:/Users/momom2/wesnoth_runs")
RUN_TREE = Path("C:/Users/momom2/Desktop/perso/projects/wesnoth_ai-run")
STAGE1_TREE = Path("C:/Users/momom2/Desktop/perso/projects/wesnoth_ai-stage1")    # the code stage C's plant ran
GPU_PY = "C:/Users/momom2/venvs/wesnoth_rocm/Scripts/python.exe"
LOG = RUNS / "after_round1.log"
TOLERANCE = 1e-3          # departure 32: log probability and class mass
KILL_SE = 3.0             # departure 31: the match below its forecast by more than this


def log(msg: str) -> None:
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def rounds_alive() -> bool:
    q = ("@(Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and "
         "$_.CommandLine -like '*rounds_local.py*' }).Count")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", q], capture_output=True, text=True, timeout=120)
        return r.returncode != 0 or not r.stdout.strip() or int(r.stdout.strip()) > 0
    except (subprocess.TimeoutExpired, ValueError, OSError):
        return True


def last_line(p: Path) -> str:
    lines = p.read_text(encoding="utf-8").strip().splitlines() if p.exists() else []
    return lines[-1] if lines else ""


def kill_reason() -> str:
    """Departure 31's kill, or "" when round 1's match is within 3 combined
    SE of its forecast (half A's first order on the chosen rung)."""
    r = json.loads((RUNS / "rounds_local" / "round_01" / "round.json").read_text(encoding="utf-8"))
    xp = r["decision"].get("cross_play")
    if xp is None:
        return "round 1 holds no cross-play reading"
    ladder = json.loads((RUNS / "stage_c_local" / "stage_c.json").read_text(encoding="utf-8"))["ladder"]
    fo = ladder["rungs"][ladder["chosen"]]["halves"]["A"]["first_order"]
    se = math.hypot(xp["se"], fo["se"])
    z = (xp["estimate"] - fo["estimate"]) / se
    log(f"round 1's match {xp['estimate']:+.4f} +- {xp['se']:.4f} against the forecast {fo['estimate']:+.4f} +- "
        f"{fo['se']:.4f}: {z:+.2f} combined SE")
    return f"the match is {z:+.2f} SE from its forecast (departure 31's kill)" if z < -KILL_SE else ""


def gpu_check(head: str) -> str:
    """"pass", "fail" (it ran and the readings differ) or "error" (it did
    not finish: a possible GPU fault, so nothing more runs on the GPU)."""
    subprocess.run(["git", "checkout", "-q", "--detach", head], cwd=RUN_TREE, check=True)
    env = dict(os.environ, HIP_VISIBLE_DEVICES="1", AMD_SERIALIZE_KERNEL="3")
    try:
        r = subprocess.run([GPU_PY, "-u", str(RUNS / "tools" / "shared_trunk_gpu_check.py"), "--games", "24"],
                           cwd=RUN_TREE, env=env, capture_output=True, text=True, timeout=45 * 60)
    except subprocess.TimeoutExpired:
        log("GPU_CHECK timed out after 45 min")
        return "error"
    (RUNS / "shared_trunk_gpu_check.out").write_text(r.stdout + "\n--- stderr\n" + r.stderr[-4000:], encoding="utf-8")
    if r.returncode != 0:
        log(f"GPU_CHECK exited {r.returncode} (shared_trunk_gpu_check.out)")
        return "error"
    out = json.loads(r.stdout[r.stdout.index("{"):])
    worst = max(max(v[0], v[1]) for v in out["worst_logp_mass_decisions"].values())
    ok = worst <= TOLERANCE and out["reads_shared"] == {"full": 1} and out["rung_moves_logp_by"] > 10 * TOLERANCE
    log(f"GPU_CHECK {'pass' if ok else 'fail'}: worst difference {worst:.3g}, shared reads {out['reads_shared']}, "
        f"walls {out['wall_shared_s']} s shared against {out['wall_alone_s']} s alone")
    return "pass" if ok else "fail"


def plant_readings() -> bool:
    """The plant's discriminator (tools/plant_discriminate.py) read on the GPU
    as the readout ran (fp16, serialized) on the subsets the CPU reads in
    float32: 1 and 8 readout games. False when one does not finish."""
    env = dict(os.environ, HIP_VISIBLE_DEVICES="1", AMD_SERIALIZE_KERNEL="3")
    for games in (1, 8):
        out = RUNS / f"plant_disc_gpu{games}.json"
        try:
            r = subprocess.run([GPU_PY, "-u", str(RUNS / "tools" / "plant_discriminate.py"), "--games", str(games),
                                "--device", "cuda", "--fp16", "--skip-grads", "--out", str(out)],
                               cwd=STAGE1_TREE, env=env, capture_output=True, text=True, timeout=30 * 60)
        except subprocess.TimeoutExpired:
            log(f"PLANT_GPU {games} games timed out after 30 min")
            return False
        (RUNS / f"plant_disc_gpu{games}.log").write_text(r.stdout + "\n--- stderr\n" + r.stderr[-4000:],
                                                         encoding="utf-8")
        if r.returncode != 0:
            log(f"PLANT_GPU {games} games exited {r.returncode} (plant_disc_gpu{games}.log)")
            return False
        log(f"PLANT_GPU {games} games: {out.read_text(encoding='utf-8').replace(chr(10), ' ')[:900]}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", required=True, help="the commit with departure 32's fix")
    ap.add_argument("--plain", required=True, help="the commit without it")
    a = ap.parse_args()
    log(f"AFTER_ROUND1 waiting for rounds_local to exit (fixed {a.fixed}, plain {a.plain})")
    while rounds_alive():
        time.sleep(120)
    end = last_line(RUNS / "rounds_local" / "runner.log")
    log(f"rounds_local exited: {end}")
    if "ROUNDS_DONE" not in end and "ROUNDS_WAIT after round 1" not in end:
        log("AFTER_ROUND1_HELD: round 1 did not end as expected; nothing armed")
        return 1
    why = kill_reason()
    if why:
        log(f"AFTER_ROUND1_HELD: {why}; nothing armed")
        return 1
    check = gpu_check(a.fixed)
    if check == "error":
        log("AFTER_ROUND1_HELD: the GPU check did not finish; nothing more runs on the GPU until the lead looks")
        return 1
    head = a.fixed if check == "pass" else a.plain
    if not plant_readings():
        log("AFTER_ROUND1_HELD: a plant reading on the GPU did not finish; nothing more runs on the GPU until the "
            "lead looks")
        return 1
    subprocess.run(["git", "checkout", "-q", "--detach", head], cwd=RUN_TREE, check=True)
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                        str(RUNS / "tools" / "arm_chain.ps1"), "-Head", head], capture_output=True, text=True)
    log(f"AFTER_ROUND1_ARMED at {head}: {r.stdout.strip().splitlines()[0] if r.stdout.strip() else r.stderr[-300:]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
