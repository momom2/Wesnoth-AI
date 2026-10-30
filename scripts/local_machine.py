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
  started. Its kernels run serialized except a match driver's (`tool_env`),
  and its log line names the commit it runs (`code_version`);
- `run_match`: run_tool for a match, which may step aside while the user's
  own programs use the GPU (scripts/gpu_yield.py) and resume after;
- `Steps`: a resumable run's parts (scripts/stage_c_local.py,
  scripts/rounds_local.py): `<part>.done` / `<part>.failed` markers, a STOP
  file between steps, and a display reset during a GPU step writing STOP
  and ending the run (`Stopped`).

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


RESET_CHECK_TIMED_OUT = -1         # display_resets_since's answer when its query hung twice: the run stops


def display_resets_since(t0: float) -> int:
    """Display-driver resets (Windows System event 4101, a TDR) logged since
    t0 (epoch seconds); 0 off Windows. The RX 6600 drives the display, so
    a reset means a kernel stalled the GPU past 2 s: the run stops.
    RESET_CHECK_TIMED_OUT (nonzero, so it stops a run too) when the query
    hung twice."""
    if sys.platform != "win32":
        return 0
    start = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0))
    cmd = (f"(Get-WinEvent -FilterHashtable @{{LogName='System'; Id=4101; StartTime=[datetime]'{start}'}} "
           f"-ErrorAction SilentlyContinue | Measure-Object).Count")
    for _ in range(2):                          # a query that hangs once is asked again
        try:
            r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True,
                               timeout=120)
        except subprocess.TimeoutExpired:
            continue
        try:
            return int(r.stdout.strip() or 0)
        except ValueError:
            return 0
    return RESET_CHECK_TIMED_OUT


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


# The RX 6600 faults (a launch failure and a display reset) when the pg
# tools' kernels run unserialized, and runs them clean serialized
# (training/metrics/pg_fp16_20261030, "After the reboot"). So every tool
# runs with SERIALIZE_ENV, except the match drivers below: their inference
# server ran stage 1's 13 h unserialized without a fault.
SERIALIZE_ENV = {"AMD_SERIALIZE_KERNEL": "3"}
UNSERIALIZED_TOOLS = frozenset({"tools/run_elo_batch.py", "scripts/selfplay_t1_local.py", "scripts/match_local.py"})


def tool_env(cmd: List[str], serialize: Optional[bool] = None) -> dict:
    """The environment a tool runs in: this process's, with SERIALIZE_ENV
    set when the tool is serialized and removed when it is not. `serialize`
    None: serialized unless cmd[0] is in UNSERIALIZED_TOOLS."""
    if serialize is None:
        serialize = not cmd or cmd[0] not in UNSERIALIZED_TOOLS
    env = dict(os.environ)
    for k, v in SERIALIZE_ENV.items():
        if serialize:
            env[k] = v
        else:
            env.pop(k, None)
    return env


def code_version(cwd: Path = REPO) -> str:
    """The commit a tool runs from cwd, "+dirty" when tracked files differ
    from it; "unknown" without git."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True,
                              text=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "-uno"], cwd=cwd, capture_output=True,
                               text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return (head or "unknown") + ("+dirty" if dirty else "")


def run_tool(out: Path, name: str, cmd: List[str], log_name: str, timeout_min: float, python: str, *,
             poll: Optional[Callable[[], Optional[str]]] = None, poll_s: float = 60.0,
             cwd: Path = REPO, serialize: Optional[bool] = None) -> int:
    """One tool to its end or its cut, its output appended to out/log_name:
    its exit status, 124 when cut at timeout_min. Given `poll`, it is asked
    every poll_s seconds while the tool runs; when it returns a reason, the
    tool is killed and ToolStopped(reason) raised. A cut or a stop kills the
    tool's whole process tree (kill_tree). The tool's environment is
    `tool_env(cmd, serialize)`; the step's log line names the commit it runs
    and whether its kernels are serialized."""
    full = [python, "-u"] + cmd
    env = tool_env(cmd, serialize)
    serialized = all(env.get(k) == v for k, v in SERIALIZE_ENV.items())
    log(out, f"{name}: {' '.join(cmd[:2])} ... (cut at {timeout_min:.0f} min, {log_name}; "
             f"code {code_version(cwd)}, {'serialized' if serialized else 'unserialized'})")
    with open(out / log_name, "a", encoding="utf-8") as f:
        f.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(full)}\n")
        f.flush()
        p = subprocess.Popen(full, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env)
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


def run_match(out: Path, name: str, cmd: List[str], log_name: str, timeout_min: float, python: str, *,
              poll: Optional[Callable[[], Optional[str]]] = None, gpu_yield=None,
              poll_s: float = 60.0, cwd: Path = REPO, runner: Callable[..., int] = None) -> int:
    """run_tool for a match, which resumes in its directory when run again.
    With `gpu_yield` (gpu_yield.GpuYield), the GPU use of every process
    outside this one's tree is sampled at each poll. When the user's
    programs take the GPU, the match is stopped, the run waits until they
    have been quiet for the policy's minutes, and the same command runs
    again, losing only the games in flight. A stop for `poll`'s reason
    raises ToolStopped as run_tool does. `runner` stands in for run_tool
    (the scripts pass their own name for it, which their tests replace)."""
    runner = runner or run_tool
    if gpu_yield is None:
        return runner(out, name, cmd, log_name, timeout_min, python, poll=poll, poll_s=poll_s, cwd=cwd)
    from scripts.gpu_yield import tree_pids
    yielded: List[str] = []

    def watch() -> Optional[str]:
        why = poll() if poll is not None else None
        if why:
            return why
        busy = gpu_yield.step_aside(tree_pids(os.getpid()))
        if busy:
            yielded.append(busy)
            return f"the GPU yielded to the user ({busy})"
        return None

    while True:
        yielded.clear()
        try:
            return runner(out, name, cmd, log_name, timeout_min, python, poll=watch, poll_s=poll_s, cwd=cwd)
        except ToolStopped:
            if not yielded:
                raise
        t0 = time.monotonic()
        gpu_yield.wait_until_quiet(lambda: tree_pids(os.getpid()), every_s=poll_s,
                                   log=lambda m: log(out, f"{name}: {m}"))
        log(out, f"{name}: the GPU has been quiet for {gpu_yield.quiet_s / 60:.0f} min after "
                 f"{(time.monotonic() - t0) / 60:.0f} min of waiting: the match resumes")


class Stopped(Exception):
    """A run ends here: a STOP file, or a display reset during a step."""


class Steps:
    """A resumable run's parts in `out` (module docstring). `runner` stands
    in for run_tool and `resets` for display_resets_since (their tests
    replace both); GPU steps run on `py`, CPU steps on `tools_py`."""

    def __init__(self, out: Path, py: str, tools_py: str, runner: Callable[..., int] = None,
                 resets: Callable[[float], int] = None):
        self.out, self.py, self.tools_py = Path(out), py, tools_py
        self.runner = runner or run_tool
        self.resets = resets or display_resets_since
        self.state: List[str] = []

    def done(self, part: str) -> bool:
        return (self.out / f"{part}.done").exists()

    def failed_before(self, part: str) -> Optional[str]:
        f = self.out / f"{part}.failed"
        return f.read_text(encoding="utf-8").strip() if f.exists() else None

    def mark(self, part: str, why: Optional[str] = None) -> None:
        name = f"{part}.failed" if why else f"{part}.done"
        (self.out / name).write_text((why or time.strftime("%Y-%m-%d %H:%M:%S")) + "\n", encoding="utf-8")
        self.state.append(f"{part}: {'FAILED, ' + why if why else 'done'};")

    def skip(self, part: str) -> bool:
        """True when `part` is done or failed at an earlier start."""
        if self.done(part):
            self.state.append(f"{part}: done;")
            return True
        why = self.failed_before(part)
        if why:
            self.state.append(f"{part}: FAILED at an earlier start, {why};")
            return True
        return False

    def check_stop(self, before: str) -> None:
        if (self.out / "STOP").exists():
            raise Stopped(f"STOP found before {before}")

    def step(self, name: str, cmd: List[str], log_name: str, cut_min: float, gpu: bool = True,
             out: Optional[Path] = None) -> int:
        """One tool to its end or its cut, its output in `out`/log_name (out:
        the run's by default); after a GPU step, a display reset writes STOP
        in the run's directory and raises Stopped."""
        t0 = time.time()
        rc = self.runner(out or self.out, name, cmd, log_name, cut_min, self.py if gpu else self.tools_py)
        if gpu:
            n = self.resets(t0)
            if n:
                why = (f"the display-reset check timed out twice after {name}" if n == RESET_CHECK_TIMED_OUT
                       else f"{n} display-driver reset(s) (System event 4101) during {name}")
                if rc == 0:                     # its product may be whole: the lead decides, never a plain rerun
                    why += f"; {name} returned 0, so check its product before running it again"
                (self.out / "STOP").write_text(why + "\n", encoding="utf-8")
                raise Stopped(why)
        return rc
