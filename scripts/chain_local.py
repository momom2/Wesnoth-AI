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
   With --skip-p, P is not run and the rounds run with --no-p-gate
   (prereg departure 31: P's played check cannot resolve at the plant's
   size). --rounds-n-train, --rounds-held, --rounds-average and
   --rounds-lean are rounds_local's --n-train, --held, --average and
   --lean (prereg departures 25, 34, 37 and 43).
4. With --final-out, once the rounds end (0, or 3 when a design stop or a
   wait ended them, or 1 too with --final-after-failure: prereg departure
   43, since a failed part taints only its own round), it plays the
   window's final match
   (docs/final_match_local_prereg_20261031.md) with scripts/match_local.py
   on the last round that adopted a candidate, after checking that
   candidate's SHA-256 against its round.json. No adopted round, no match.
5. With --tail-out, once the final match returns 0, it plays the tail match
   (docs/tail_local_prereg_20261102.md): the same candidate against the
   reference, both sampling at temperature 1 (match_local --t1).
6. With --continue-rounds and --continue-end-by, once the rounds returned 0
   and the final match (and the tail, if asked) returned 0, the rounds go
   on: rounds_local again, resumed, to the later deadline. With
   --continue-final-out, a round adopted after the final match's gets its
   own final match there.

7. With --after-stop K0 (prereg departure 33), the rounds go on past round
   K0's design stop instead: its STOP (which must name round K0) is moved
   aside to STOP_roundK0, and rounds_local resumes with --rebase-after K0,
   --rounds-arm as its --arm and --rounds-pass-window as its --pass-window.
   No P, no tail and no continuation: with --final-out, the standard match
   on a round adopted after K0, or none.

Every decision is logged to --log (default: chain.log beside --rounds-out).
Exit: 0 when the rounds returned 0 and every match asked for returned 0, 1
when the chain refused or a step failed (that step's code is in the log).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
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
TAIL_GAMES = 12800                       # the tail match's decisive games (its prereg)
TAIL_SEED_BASE = 950000                  # the tail match's seeds (its prereg): no record names 950000-964299
END_BY_FORMAT = "%Y-%m-%d %H:%M"         # rounds_local's --end-by
MATCHES = {"FINAL":"the final match", "TAIL": "the tail match (raw:t1)",
           "CONTINUE_FINAL": "the continuation's final match"}
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
        p = "no P (--skip-p)" if a.skip_p else f"P ({a.p_out}, e {a.e})"
        self.note(f"CHAIN_WAIT: stage C {a.stage_c}; then {p} and {a.rounds} rounds ({a.rounds_out}) at "
                  f"{a.expect_head}")
        why = self.wait_for_stage_c() or self.preflight()
        if why:
            self.note(f"CHAIN_REFUSED: {why}")
            return 1
        self.note(f"CHAIN_VERDICT: stage C's verdict read again at {a.expect_head}: {refresh_verdict(a.stage_c)}")
        if a.skip_p:
            self.note("CHAIN_ROUNDS: stage C finished; P is skipped (--skip-p, prereg departure 31): the rounds "
                      "start with --no-p-gate")
        else:
            self.note("CHAIN_P: stage C finished; P starts")
            rc = self.run([a.gpu_python, "-u", "scripts/p_local.py", *self.common(), "--out", str(a.p_out),
                           "--e", repr(a.e)], a.p_out / "console.log", a.hip_visible_devices)
            if rc != 0:
                self.note(f"CHAIN_STOPPED: P returned {rc} ({a.p_out / 'runner.log'}: "
                          f"{last_line(a.p_out / 'runner.log')}); the rounds do not start")
                return 1
            self.note("CHAIN_ROUNDS: P passed and the link holds; the rounds start")
        if a.after_stop is not None:
            return self.after_stop()
        rc = self.rounds(a.rounds, a.rounds_end_by)
        self.note(f"CHAIN_DONE: the rounds returned {rc} ({a.rounds_out / 'runner.log'}: "
                  f"{last_line(a.rounds_out / 'runner.log')})")
        if a.final_out is None or rc not in self.ended():
            return 0 if rc == 0 else 1
        final = self.final_match("FINAL", a.final_out)
        if final[0] == 0 and a.tail_out is not None:
            final = (self.tail_match(), final[1])
        if final[0] != 0 or rc != 0 or a.continue_end_by is None:
            return 0 if rc == 0 and final[0] == 0 else 1
        return self.continuation(final[1])

    def after_stop(self) -> int:
        """Item 7 of the module docstring (prereg departure 33)."""
        a, k0 = self.a, self.a.after_stop
        stop = a.rounds_out / "STOP"
        why = stop.read_text(encoding="utf-8").strip() if stop.exists() else ""
        if f"after round {k0}" not in why:
            self.note(f"CHAIN_REFUSED: --after-stop {k0} needs round {k0}'s design stop in {stop} "
                      f"({why or 'no STOP'})")
            return 1
        aside, n = a.rounds_out / f"STOP_round{k0:02d}", 1
        while aside.exists():
            n += 1
            aside = a.rounds_out / f"STOP_round{k0:02d}_{n}"
        stop.rename(aside)
        self.note(f"CHAIN_AFTER_STOP: round {k0}'s stop moved to {aside.name} ({why}); the rounds go on to round "
                  f"{a.rounds} by {a.rounds_end_by}, arm {a.rounds_arm or 'from stage C'}, pass window "
                  f"{a.rounds_pass_window or 'the default'}, rebased on round {k0} (prereg departure 33)")
        rc = self.rounds(a.rounds, a.rounds_end_by)
        self.note(f"CHAIN_DONE: the rounds returned {rc} ({a.rounds_out / 'runner.log'}: "
                  f"{last_line(a.rounds_out / 'runner.log')})")
        if a.final_out is None or rc not in self.ended():
            return 0 if rc == 0 else 1
        last = last_adopted(a.rounds_out)
        if last is None or last[0] <= k0:
            self.note(f"CHAIN_NO_FINAL: no round after round {k0} adopted a candidate")
            return 0 if rc == 0 else 1
        final, _k = self.final_match("FINAL", a.final_out)
        return 0 if rc == 0 and final == 0 else 1

    def ended(self) -> Tuple[int, ...]:
        """rounds_local's codes after which a final match plays: ROUNDS_ENDED,
        and 1 with --final-after-failure (module docstring item 4)."""
        return ROUNDS_ENDED + ((1,) if getattr(self.a, "final_after_failure", False) else ())

    def rounds(self, n: int, end_by: Optional[str]) -> int:
        """rounds_local to round `n` by `end_by`, resumed from its markers."""
        a = self.a
        heads = ["--heads-only-pass"] if a.heads_only_pass else []
        heads += ["--end-by", end_by] if end_by else []
        heads += ["--n-train", str(a.rounds_n_train)] if a.rounds_n_train else []
        heads += ["--held", str(a.rounds_held)] if getattr(a, "rounds_held", None) else []
        heads += ["--arm", a.rounds_arm] if getattr(a, "rounds_arm", None) else []
        heads += ["--pass-window", str(a.rounds_pass_window)] if getattr(a, "rounds_pass_window", None) else []
        heads += ["--rebase-after", str(a.after_stop)] if getattr(a, "after_stop", None) is not None else []
        heads += ["--average", str(a.rounds_average)] if getattr(a, "rounds_average", None) else []
        heads += ["--lean"] if getattr(a, "rounds_lean", False) else []
        p = ["--no-p-gate"] if a.skip_p else ["--p-out", str(a.p_out)]
        return self.run([a.gpu_python, "-u", "scripts/rounds_local.py", *self.common(), "--out", str(a.rounds_out),
                         "--rounds", str(n), *p, *heads], a.rounds_out / "console.log", a.hip_visible_devices)

    def candidate(self, what: str) -> Optional[Tuple[int, Path, str]]:
        """(k, its pi.pt, its SHA-256) for the last round that adopted a
        candidate, once the file's SHA-256 is its round.json's; None, noted
        as CHAIN_NO_<what>, otherwise."""
        last = last_adopted(self.a.rounds_out)
        if last is None:
            self.note(f"CHAIN_NO_{what}: no round adopted a candidate, so nothing is measured (the final match's "
                      f"prereg)")
            return None
        k, cand = last
        pi = self.a.rounds_out / f"round_{k:02d}" / "pi.pt"
        try:
            digest = self.sha(pi)
        except OSError as e:
            self.note(f"CHAIN_NO_{what}: round {k}'s candidate {pi} cannot be read: {e}")
            return None
        if digest != cand.get("sha256"):
            self.note(f"CHAIN_NO_{what}: {pi}'s SHA-256 is {digest}, not round {k}'s {cand.get('sha256')}")
            return None
        return k, pi, digest

    def match(self, what: str, k: int, pi: Path, digest: str, out: Path, label: str, extra: List[str]) -> int:
        """match_local on round k's candidate into `out`; its code."""
        a = self.a
        self.note(f"CHAIN_{what}: {MATCHES[what]} on round {k}'s candidate {pi} ({digest[:12]})")
        rc = self.run([a.gpu_python, "-u", "scripts/match_local.py", "--label", label, "--checkpoint", str(pi),
                       "--out", str(out), *extra, "--tools-python", a.tools_python, "--keep-awake"],
                      out / "console.log", a.hip_visible_devices)
        self.note(f"CHAIN_{what}_DONE: match_local returned {rc} ({out})")
        return rc

    def final_match(self, what: str, out: Path) -> Tuple[int, Optional[int]]:
        """The standard match on the last adopted round's policy: (its code,
        or 1 when it does not start; that round, None without one)."""
        cand = self.candidate(what)
        if cand is None:
            return 1, None
        k, pi, digest = cand
        return self.match(what, k, pi, digest, out, f"final_round{k}",
                          ["--games", str(self.a.final_games), "--seed-base", str(self.a.final_seed_base)]), k

    def tail_match(self) -> int:
        """The tail match (raw:t1) on the last adopted round's policy; its
        code, or 1 when it does not start."""
        cand = self.candidate("TAIL")
        if cand is None:
            return 1
        k, pi, digest = cand
        return self.match("TAIL", k, pi, digest, self.a.tail_out, f"final_round{k}t1",
                          ["--t1", "--games", str(self.a.tail_games), "--seed-base", str(self.a.tail_seed_base)])

    def continuation(self, measured: int) -> int:
        """The rounds resumed to --continue-rounds by --continue-end-by, then
        the standard match on a round adopted after round `measured`."""
        a = self.a
        self.note(f"CHAIN_CONTINUE: the rounds go on to round {a.continue_rounds}, ending by {a.continue_end_by}")
        rc = self.rounds(a.continue_rounds, a.continue_end_by)
        self.note(f"CHAIN_CONTINUE_DONE: the rounds returned {rc} ({a.rounds_out / 'runner.log'}: "
                  f"{last_line(a.rounds_out / 'runner.log')})")
        if a.continue_final_out is None or rc not in self.ended():
            return 0 if rc == 0 else 1
        last = last_adopted(a.rounds_out)
        if last is None or last[0] <= measured:
            self.note(f"CHAIN_NO_CONTINUE_FINAL: no round after round {measured} adopted a candidate")
            return 0 if rc == 0 else 1
        final, _k = self.final_match("CONTINUE_FINAL", a.continue_final_out)
        return 0 if rc == 0 and final == 0 else 1


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
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
    ap.add_argument("--skip-p", action="store_true",
                    help="run no P: the rounds start with rounds_local --no-p-gate (prereg departure 31)")
    ap.add_argument("--heads-only-pass", action="store_true",
                    help="passed to rounds_local: each round's pass takes the head maps' gradient alone when "
                         "stage C's arm reads no other")
    ap.add_argument("--rounds-end-by", default=None, metavar="'YYYY-MM-DD HH:MM'",
                    help="passed to rounds_local --end-by: the rounds end in time for the final match")
    ap.add_argument("--rounds-n-train", type=int, default=None,
                    help="passed to rounds_local --n-train: a round's training games (prereg departure 25)")
    ap.add_argument("--rounds-held", type=int, default=None,
                    help="passed to rounds_local --held: |H|, a round's held-out games, for rounds whose games "
                         "start from now (prereg departure 34)")
    ap.add_argument("--rounds-average", type=int, default=None,
                    help="passed to rounds_local --average: a step reads its direction averaged with the N "
                         "previous rounds' beside it (prereg departure 37)")
    ap.add_argument("--rounds-lean", action="store_true",
                    help="passed to rounds_local --lean (prereg departure 43)")
    ap.add_argument("--final-after-failure", action="store_true",
                    help="play the final match after a failed round too (module docstring item 4, prereg "
                         "departure 43)")
    ap.add_argument("--final-out", type=Path, default=None,
                    help="play the window's final match into this directory once the rounds end")
    ap.add_argument("--final-games", type=int, default=FINAL_GAMES)
    ap.add_argument("--final-seed-base", type=int, default=FINAL_SEED_BASE)
    ap.add_argument("--tail-out", type=Path, default=None,
                    help="play the tail match (raw:t1) into this directory once the final match returns 0")
    ap.add_argument("--tail-games", type=int, default=TAIL_GAMES)
    ap.add_argument("--tail-seed-base", type=int, default=TAIL_SEED_BASE)
    ap.add_argument("--continue-rounds", type=int, default=None,
                    help="after the final match (and the tail), resume the rounds to this round")
    ap.add_argument("--continue-end-by", default=None, metavar="'YYYY-MM-DD HH:MM'",
                    help="the continued rounds' --end-by")
    ap.add_argument("--continue-final-out", type=Path, default=None,
                    help="the standard match on a round the continuation adopts, into this directory")
    ap.add_argument("--after-stop", type=int, default=None, metavar="K0",
                    help="go on past round K0's design stop (module docstring item 7, prereg departure 33)")
    ap.add_argument("--rounds-arm", default=None, help="passed to rounds_local --arm (prereg departure 33)")
    ap.add_argument("--rounds-pass-window", type=int, default=None,
                    help="passed to rounds_local --pass-window (prereg departure 33)")
    ap.add_argument("--log", type=Path, default=None)
    args = ap.parse_args(argv)
    if (args.continue_rounds is None) != (args.continue_end_by is None):
        ap.error("--continue-rounds and --continue-end-by go together")
    for flag, when in (("--rounds-end-by", args.rounds_end_by), ("--continue-end-by", args.continue_end_by)):
        if when is not None:
            try:
                datetime.strptime(when, END_BY_FORMAT)
            except ValueError:
                ap.error(f"{flag} {when!r} is not 'YYYY-MM-DD HH:MM'")
    if args.continue_end_by is not None and args.rounds_end_by is not None and \
            datetime.strptime(args.continue_end_by, END_BY_FORMAT) <= datetime.strptime(args.rounds_end_by,
                                                                                       END_BY_FORMAT):
        ap.error("--continue-end-by must come after --rounds-end-by")
    if args.continue_rounds is not None and (args.final_out is None or args.continue_rounds <= args.rounds):
        ap.error("the continuation follows the final match (--final-out) and goes past --rounds")
    if (args.tail_out is not None or args.continue_final_out is not None) and args.final_out is None:
        ap.error("--tail-out and --continue-final-out follow the final match (--final-out)")
    if args.continue_final_out is not None and args.continue_rounds is None:
        ap.error("--continue-final-out needs the continuation (--continue-rounds)")
    if args.after_stop is not None:
        if not args.skip_p:
            ap.error("--after-stop runs no P: give --skip-p")
        if args.tail_out is not None or args.continue_rounds is not None:
            ap.error("--after-stop plays no tail and no continuation")
        if args.after_stop < 1 or args.rounds <= args.after_stop:
            ap.error(f"--after-stop {args.after_stop}: the rounds must go past it (--rounds {args.rounds})")
    for out in ("tail_out", "continue_final_out"):
        if getattr(args, out) is not None and getattr(args, out) == args.final_out:
            ap.error(f"--{out.replace('_', '-')} must differ from --final-out (match_local refuses a changed match)")
    return args


def main(argv: Optional[List[str]] = None) -> int:
    return Chain(parse_args(argv))()


if __name__ == "__main__":
    sys.exit(main())
