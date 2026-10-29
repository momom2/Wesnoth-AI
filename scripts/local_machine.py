"""What the runs on the local Windows machine share (the RX 6600;
docs/new_machine.md step 8): scripts/selfplay_t1_local.py and
scripts/match_local.py import it.

- `below_normal_priority`, `keep_awake`: how a run holds the machine;
- `display_resets_since`: the display-driver resets (Windows System event
  4101) logged since a time;
- `require_visible_gpu`: HIP_VISIBLE_DEVICES (or CUDA_VISIBLE_DEVICES) is
  set;
- `default_tools_python`: the interpreter of the CPU tools;
- `reference_flags`: the reference player's run_elo_batch flags for a side;
- `run_tool`: one tool to its end or its cut, and, given a poll, stopped
  when the poll names a reason; `kill_tree` ends it with every process it
  started.

Off Windows the priority is `os.nice(5)`, keep-awake does nothing and no
reset is ever logged, so the scripts and their tests run on Linux.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

REPO = Path(__file__).resolve().parent.parent


def log(out: Path, msg: str, name: str = "runner.log") -> None:
    """`msg` stamped, printed and appended to out/name."""
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(out / name, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def below_normal_priority() -> None:
    """The process at below-normal priority; Windows gives the processes it
    starts the same class (CreateProcess: a child of a below-normal or idle
    process inherits its class)."""
    if sys.platform != "win32":
        os.nice(5)
        return
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE          # the pseudo-handle -1, 64 bits wide
    k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.SetPriorityClass.restype = wintypes.BOOL
    if not k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000):     # BELOW_NORMAL_PRIORITY_CLASS
        raise ctypes.WinError(ctypes.get_last_error())


def keep_awake() -> None:
    """Asks Windows not to sleep while the process lives (SetThreadExecutionState
    with ES_CONTINUOUS | ES_SYSTEM_REQUIRED, which a media player or a backup
    makes too): the request ends with the process, and no power setting
    changes. The display may still turn off. On 2026-10-28 the machine slept
    30 minutes after an update's restart. A restart itself is not prevented."""
    if sys.platform != "win32":
        return
    import ctypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
    k32.SetThreadExecutionState.restype = ctypes.c_uint32
    if not k32.SetThreadExecutionState(0x80000000 | 0x00000001):     # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        raise ctypes.WinError(ctypes.get_last_error())


def display_resets_since(t0: float) -> int:
    """Display-driver resets (Windows System event 4101, a TDR) logged since
    t0 (epoch seconds); 0 off Windows. The RX 6600 drives the display, so
    a reset means a kernel stalled the GPU past 2 s: the run stops."""
    if sys.platform != "win32":
        return 0
    start = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0))
    cmd = (f"(Get-WinEvent -FilterHashtable @{{LogName='System'; Id=4101; StartTime=[datetime]'{start}'}} "
           f"-ErrorAction SilentlyContinue | Measure-Object).Count")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=120)
    try:
        return int(r.stdout.strip() or 0)
    except ValueError:
        return 0


def require_visible_gpu() -> None:
    """Refuses to go on unless the GPU is picked: on this machine device 0
    is the iGPU, which the gfx1032 wheel has no kernels for."""
    if "HIP_VISIBLE_DEVICES" not in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise SystemExit("set HIP_VISIBLE_DEVICES to the GPU's index (1 for the RX 6600 here: device 0 is the iGPU)")


def default_tools_python(repo: Path = REPO) -> str:
    """The repository's .venv, whose CPU torch is faster than the ROCm
    build's; this interpreter when there is none."""
    venv = repo / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    return str(venv) if venv.exists() else sys.executable


def reference_flags(side: str, python: str, repo: Path = REPO) -> List[str]:
    """`tools/reference_player.py --flags side`, split: the reference's
    label, checkpoint, memory slots and decode in run_elo_batch's flags."""
    r = subprocess.run([python, "tools/reference_player.py", "--flags", side], cwd=repo,
                       capture_output=True, text=True, check=True)
    return r.stdout.split()


def kill_tree(p: subprocess.Popen, grace_s: float = 10.0) -> None:
    """Ends `p` and every process under it. On Windows a venv's python.exe is
    a launcher that starts the interpreter as its child, so killing `p` alone
    would leave the tool (run_elo_batch, its servers and its workers)
    running. Each process is terminated, and what is left after grace_s is
    killed; without psutil, `taskkill /T /F` on Windows, `p` alone elsewhere."""
    try:
        import psutil
    except ImportError:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True, timeout=120)
        p.kill()
        p.wait()
        return
    try:
        root = psutil.Process(p.pid)
        procs = [root] + root.children(recursive=True)     # listed before the root goes
    except psutil.NoSuchProcess:
        procs = []
    for q in procs:
        try:
            q.terminate()
        except psutil.NoSuchProcess:
            pass
    _gone, alive = psutil.wait_procs(procs, timeout=grace_s)
    for q in alive:
        try:
            q.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=grace_s)
    p.wait()


class ToolStopped(Exception):
    """run_tool's poll named a reason to stop: the tool was killed."""


def run_tool(out: Path, name: str, cmd: List[str], log_name: str, timeout_min: float, python: str, *,
             poll: Optional[Callable[[], Optional[str]]] = None, poll_s: float = 60.0,
             cwd: Path = REPO) -> int:
    """One tool to its end or its cut, its output appended to out/log_name:
    its exit status, 124 when cut at timeout_min. Given `poll`, it is asked
    every poll_s seconds while the tool runs; when it returns a reason, the
    tool is killed and ToolStopped(reason) raised. A cut or a stop kills the
    tool's whole process tree (kill_tree)."""
    full = [python, "-u"] + cmd
    log(out, f"{name}: {' '.join(cmd[:2])} ... (cut at {timeout_min:.0f} min, {log_name})")
    with open(out / log_name, "a", encoding="utf-8") as f:
        f.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(full)}\n")
        f.flush()
        p = subprocess.Popen(full, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + timeout_min * 60
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                log(out, f"{name}: past {timeout_min:.0f} min, cut")
                kill_tree(p)
                return 124
            try:
                return p.wait(timeout=left if poll is None else min(left, poll_s))
            except subprocess.TimeoutExpired:
                why = poll() if poll is not None and time.monotonic() < deadline else None
                if why:
                    log(out, f"{name}: {why}: stopped")
                    kill_tree(p)
                    raise ToolStopped(why)
