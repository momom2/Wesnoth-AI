"""Wait for a runner process to end; if it ended on our planned STOP (the
file's text is exactly MARK) and the part it must have finished is done,
delete STOP and start the runner again, detached, on the code now in its
worktree. Anything else (a display reset's STOP, a failed part, no STOP) is
logged and left for the lead.

    python relaunch_after_stop.py --pid 17096 --out C:/Users/momom2/wesnoth_runs/stage_c_local \
        --needs pass.done --mark "..." -- <command line of the runner>
"""
import argparse
import ctypes
import os
import subprocess
import time
from pathlib import Path


def log(out: Path, msg: str) -> None:
    with open(out / "relaunch.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def wait_for(pid: int) -> None:
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = ctypes.c_void_p
    h = k32.OpenProcess(0x00100000, False, pid)            # SYNCHRONIZE
    if not h:
        return                                              # already gone
    k32.WaitForSingleObject(ctypes.c_void_p(h), 0xFFFFFFFF)
    k32.CloseHandle(ctypes.c_void_p(h))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--needs", required=True)
    ap.add_argument("--mark", required=True)
    ap.add_argument("--cwd", type=Path, required=True)
    ap.add_argument("--console", type=Path, required=True)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    log(a.out, f"watching pid {a.pid}")
    wait_for(a.pid)
    stop = a.out / "STOP"
    text = stop.read_text(encoding="utf-8").strip() if stop.exists() else None
    if text != a.mark or not (a.out / a.needs).exists():
        log(a.out, f"pid {a.pid} ended; not relaunched (STOP {text!r}, {a.needs} "
                   f"{'present' if (a.out / a.needs).exists() else 'missing'})")
        return 1
    stop.unlink()
    env = dict(os.environ, HIP_VISIBLE_DEVICES="1", AMD_SERIALIZE_KERNEL="3")
    with open(a.console, "a", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, cwd=a.cwd, env=env, stdout=f, stderr=subprocess.STDOUT,
                             creationflags=0x00000008 | 0x00000200)      # DETACHED_PROCESS | NEW_PROCESS_GROUP
    log(a.out, f"pid {a.pid} ended on the planned STOP; STOP removed, runner relaunched as pid {p.pid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
