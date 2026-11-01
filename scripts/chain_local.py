"""Starts stage P and then the rounds once the local stage C has finished,
so the GPU does not idle until the lead's next look (the 2026-10-31 review).
It imports no torch; run it detached on the repository's .venv, from the
worktree P and the rounds should run (pinned at a tested commit):

    python scripts/chain_local.py --stage-c C:/Users/momom2/wesnoth_runs/stage_c_local \
        --stage1 C:/Users/momom2/wesnoth_runs/selfplay_t1_local \
        --p-out C:/Users/momom2/wesnoth_runs/p_local --rounds-out C:/Users/momom2/wesnoth_runs/rounds_local \
        --e 0.02 --rounds 3 --checkpoint training/checkpoints/own2.pt --checkpoint-sha256 <hex> \
        --expect-head <sha> --gpu-python C:/Users/momom2/venvs/wesnoth_rocm/Scripts/python.exe

1. It waits for stage C's ALL_DONE. When no stage C runner is alive at two
   looks in a row and there is still no ALL_DONE, it stops: a STOP, a failed
   part or a crash is the lead's to read.
2. It waits for the runner to exit. Then it checks:
   - that stage C finished (`stage_c_unfinished`: ALL_DONE, no STOP, no
     failed part);
   - that this worktree's HEAD is --expect-head and has no tracked change;
   - that the checkpoint's SHA-256 is --checkpoint-sha256.
3. It reads stage C's verdict again with this worktree's code
   (stage_c_local.refresh_verdict: the stage C runner's code may predate a
   rule, prereg departure 23), then runs scripts/p_local.py, then
   scripts/rounds_local.py with --p-out, but only when P returns 0 (P
   passes and the link holds). A step that fails is never run again.
4. With --final-out, once the rounds end (0, or 3 when a design stop or a
   wait ended them), it plays the window's final match
   (docs/final_match_local_prereg_20261031.md) with scripts/match_local.py
   on the last round that adopted a candidate, after checking that
   candidate's SHA-256 against its round.json. No adopted round, no match.

Every decision is logged to --log (default: chain.log beside --rounds-out).
Exit: 0 when the rounds returned 0 and the final match (if asked) returned
0, 1 when the chain refused or a step failed (that step's code is in the
log).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.local_machine import log, stage_c_unfinished  # noqa: E402
from scripts.stage_c_local import refresh_verdict  # noqa: E402

RUNNER = "stage_c_local.py"              # the stage C runner's script, as its command line names it
POLL_S = 300.0                           # seconds between looks at stage C
EXIT_WAIT_LOOKS = 12                     # looks (an hour) for the runner to exit after its ALL_DONE
FINAL_GAMES = 12800                      # the final match's decisive games (its prereg)
FINAL_SEED_BASE = 900000                 # the final match's seeds (its prereg)
ROUNDS_ENDED = (0, 3)                    # rounds_local's codes after which the final match plays


def runner_alive(script: str = RUNNER) -> bool:
    """Whether a python process's command line names `script`. True when the
    process list cannot be read, so the chain looks again rather than
    deciding that the runner has ended."""
    if sys.platform != "win32":
        return subprocess.run(["pgrep", "-f", script], capture_output=True).returncode == 0
    query = (f"@(Get-CimInstance Win32_Process | Where-Object {{ $_.Name -like 'python*' -and "
             f"$_.CommandLine -like '*{script}*' }}).Count")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", query], capture_output=True, text=True,
                           timeout=120)
        out = r.stdout.strip()
        if r.returncode != 0 or not out:       # a query that failed or said nothing has not seen the runner end
            return True
        return int(out) > 0
    except (subprocess.TimeoutExpired, ValueError, OSError):    # OSError: powershell could not be started
        return True


def git_head(repo: Path = REPO) -> Tuple[str, bool]:
    """(HEAD's full SHA, whether tracked files differ from it)."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "-uno"], cwd=repo, capture_output=True,
                           text=True).stdout.strip()
    return head, bool(dirty)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_logged(cmd: List[str], console: Path, hip_devices: str) -> int:
    """`cmd` from the repository to its end, its output appended to `console`."""
    console.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HIP_VISIBLE_DEVICES=hip_devices)
    with open(console, "a", encoding="utf-8") as f:
        return subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def last_adopted(rounds_out: Path) -> Optional[Tuple[int, dict]]:
    """(k, round k's candidate) for the last round whose round.json names a
    candidate, as rounds_local's policy_after reads it; None when none did."""
    adopted = []
    for rj in rounds_out.glob("round_*/round.json"):
        cand = (json.loads(rj.read_text(encoding="utf-8")).get("candidate") or {})
        if cand.get("path"):
            adopted.append((int(rj.parent.name.split("_")[1]), cand))
    return max(adopted, key=lambda kc: kc[0]) if adopted else None


def last_line(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").strip().splitlines() if path.exists() else []
    return lines[-1] if lines else "(empty)"


class Chain:
    """The chain's steps; `alive`, `run`, `sleep`, `head` and `sha` are
    runner_alive, run_logged, time.sleep, git_head and file_sha256 (their
    tests pass fakes)."""

    def __init__(self, args, alive: Callable[[], bool] = runner_alive,
                 run: Callable[[List[str], Path, str], int] = run_logged,
                 sleep: Callable[[float], None] = time.sleep,
                 head: Callable[[], Tuple[str, bool]] = git_head, sha: Callable[[Path], str] = file_sha256):
        self.a, self.alive, self.run, self.sleep, self.head, self.sha = args, alive, run, sleep, head, sha
        self.log_dir = args.log.parent if args.log else args.rounds_out.parent
        self.log_name = args.log.name if args.log else "chain.log"

    def note(self, msg: str) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log(self.log_dir, msg, self.log_name)

    def wait_for_stage_c(self) -> Optional[str]:
        """None once ALL_DONE is there and the runner has exited; else why
        the chain stops."""
        sc, gone = self.a.stage_c, 0
        while not (sc / "ALL_DONE").exists():
            gone = 0 if self.alive() else gone + 1
            if gone >= 2:
                return f"stage C's runner ended without ALL_DONE (runner.log: {last_line(sc / 'runner.log')})"
            self.sleep(POLL_S)
        for _ in range(EXIT_WAIT_LOOKS):
            if not self.alive():
                return None
            self.sleep(POLL_S)
        return "stage C wrote ALL_DONE but its runner is still alive an hour later"

    def preflight(self) -> Optional[str]:
        """Why P must not start from this worktree, or None."""
        why = stage_c_unfinished(self.a.stage_c)
        if why:
            return why
        head, dirty = self.head()
        if not head.startswith(self.a.expect_head) or dirty:
            return f"this worktree is at {head}{' with tracked changes' if dirty else ''}, not {self.a.expect_head}"
        ckpt = self.a.checkpoint if self.a.checkpoint.is_absolute() else REPO / self.a.checkpoint
        try:                                   # resolved as P and the rounds resolve it (cwd=REPO)
            digest = self.sha(ckpt)
        except OSError as e:
            return f"{ckpt} cannot be read: {e}"
        if digest != self.a.checkpoint_sha256:
            return f"{self.a.checkpoint}'s SHA-256 is {digest}, not {self.a.checkpoint_sha256}"
        return None

    def common(self) -> List[str]:
        a = self.a
        return ["--stage1", str(a.stage1), "--stage-c", str(a.stage_c), "--checkpoint", str(a.checkpoint),
                "--tools-python", a.tools_python, "--keep-awake"]

    def __call__(self) -> int:
        a = self.a
        self.note(f"CHAIN_WAIT: stage C {a.stage_c}; then P ({a.p_out}, e {a.e}) and {a.rounds} rounds "
                  f"({a.rounds_out}) at {a.expect_head}")
        why = self.wait_for_stage_c() or self.preflight()
        if why:
            self.note(f"CHAIN_REFUSED: {why}")
            return 1
        self.note(f"CHAIN_VERDICT: stage C's verdict read again at {a.expect_head}: {refresh_verdict(a.stage_c)}")
        self.note("CHAIN_P: stage C finished; P starts")
        rc = self.run([a.gpu_python, "-u", "scripts/p_local.py", *self.common(), "--out", str(a.p_out),
                       "--e", repr(a.e)], a.p_out / "console.log", a.hip_visible_devices)
        if rc != 0:
            self.note(f"CHAIN_STOPPED: P returned {rc} ({a.p_out / 'runner.log'}: "
                      f"{last_line(a.p_out / 'runner.log')}); the rounds do not start")
            return 1
        self.note("CHAIN_ROUNDS: P passed and the link holds; the rounds start")
        heads = ["--heads-only-pass"] if a.heads_only_pass else []
        heads += ["--end-by", a.rounds_end_by] if a.rounds_end_by else []
        rc = self.run([a.gpu_python, "-u", "scripts/rounds_local.py", *self.common(), "--out", str(a.rounds_out),
                       "--rounds", str(a.rounds), "--p-out", str(a.p_out), *heads], a.rounds_out / "console.log",
                      a.hip_visible_devices)
        self.note(f"CHAIN_DONE: the rounds returned {rc} ({a.rounds_out / 'runner.log'}: "
                  f"{last_line(a.rounds_out / 'runner.log')})")
        if a.final_out is None or rc not in ROUNDS_ENDED:
            return 0 if rc == 0 else 1
        final = self.final_match()
        return 0 if rc == 0 and final == 0 else 1

    def final_match(self) -> int:
        """The window's final match on the last adopted round's policy; its
        code, or 1 when it does not start."""
        a = self.a
        last = last_adopted(a.rounds_out)
        if last is None:
            self.note("CHAIN_NO_FINAL: no round adopted a candidate, so nothing is measured (the final match's prereg)")
            return 1
        k, cand = last
        pi = a.rounds_out / f"round_{k:02d}" / "pi.pt"
        try:
            digest = self.sha(pi)
        except OSError as e:
            self.note(f"CHAIN_NO_FINAL: round {k}'s candidate {pi} cannot be read: {e}")
            return 1
        if digest != cand.get("sha256"):
            self.note(f"CHAIN_NO_FINAL: {pi}'s SHA-256 is {digest}, not round {k}'s {cand.get('sha256')}")
            return 1
        self.note(f"CHAIN_FINAL: the final match on round {k}'s candidate {pi} ({digest[:12]})")
        rc = self.run([a.gpu_python, "-u", "scripts/match_local.py", "--label", f"final_round{k}", "--checkpoint",
                       str(pi), "--out", str(a.final_out), "--games", str(a.final_games), "--seed-base",
                       str(a.final_seed_base), "--tools-python", a.tools_python, "--keep-awake"],
                      a.final_out / "console.log", a.hip_visible_devices)
        self.note(f"CHAIN_FINAL_DONE: match_local returned {rc} ({a.final_out})")
        return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage-c", type=Path, required=True)
    ap.add_argument("--stage1", type=Path, required=True)
    ap.add_argument("--p-out", type=Path, required=True)
    ap.add_argument("--rounds-out", type=Path, required=True)
    ap.add_argument("--e", type=float, required=True, help="the link match's temperature spread, as sealed")
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--checkpoint-sha256", required=True)
    ap.add_argument("--expect-head", required=True, help="the commit this worktree must be at (a prefix is enough)")
    ap.add_argument("--gpu-python", required=True, help="the GPU tools' interpreter (the ROCm venv)")
    ap.add_argument("--tools-python", default=sys.executable, help="the CPU tools' (default: this interpreter)")
    ap.add_argument("--hip-visible-devices", default="1", help="the GPU's index (1 here: device 0 is the iGPU)")
    ap.add_argument("--heads-only-pass", action="store_true",
                    help="passed to rounds_local: each round's pass takes the head maps' gradient alone when "
                         "stage C's arm reads no other")
    ap.add_argument("--rounds-end-by", default=None, metavar="'YYYY-MM-DD HH:MM'",
                    help="passed to rounds_local --end-by: the rounds end in time for the final match")
    ap.add_argument("--final-out", type=Path, default=None,
                    help="play the window's final match into this directory once the rounds end")
    ap.add_argument("--final-games", type=int, default=FINAL_GAMES)
    ap.add_argument("--final-seed-base", type=int, default=FINAL_SEED_BASE)
    ap.add_argument("--log", type=Path, default=None)
    return Chain(ap.parse_args())()


if __name__ == "__main__":
    sys.exit(main())
