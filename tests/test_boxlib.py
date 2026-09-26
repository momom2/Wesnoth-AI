"""scripts/box/boxlib.sh and box_onstart.sh, driven through bash with a stub
`python` first on PATH: no box, no network, no Vast or HF call. The stub
records every call (the uploader, the stop helper, the stage fetch) and
answers from STUB_* variables.

What is pinned: every exit of a script past box_init (an error, an unbound
variable, a signal, a clean finish) records its reason, writes ALL_DONE
before the final upload round and stops the instance after it; the
dead-man's switch does the same while the script is busy in a foreground
command; a refused stop sends its outcome and leaves the switch armed; a
bounded step is cut at its deadline, ended when its log stops growing or
once a line appears in what the log gains; a new code stage replaces the
old tree whole; the wheel's phase must equal the source's exactly; the
onstart runs the script only when every file arrived, and otherwise stops
the instance; every box script that sources the library parses."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "scripts" / "box"
STAGE = "tier-b/staging/stage_test.tar.gz"

STUB_PYTHON = r"""#!/usr/bin/env bash
# A stand-in for python: records the call, answers from STUB_* variables.
printf '%s\t%s\n' "$(date +%s)" "$*" >> "$STUB_DIR/calls.log"
if [ "${1:-}" = - ] && [ -n "${STUB_HF:-}" ]; then     # box_onstart.sh's fetch: - REPO SOURCE TARGET
    cat > /dev/null
    [ -f "$STUB_HF/$3" ] && cp "$STUB_HF/$3" "$4" && exit 0
    exit 1
fi
case " $* " in
    *" -m pip "*) exit 0 ;;
    *box_upload.py*--final*)
        ls "$BOX_OUT" > "$STUB_DIR/final_upload.$$.seen"
        exit "${STUB_UPLOAD_RC:-0}" ;;
    *box_upload.py*--clear*) exit "${STUB_CLEAR_RC:-0}" ;;
    *box_upload.py*--restore*) exit "${STUB_RESTORE_RC:-0}" ;;
    *box_upload.py*) exit "${STUB_UPLOAD_RC:-0}" ;;
    *box_stop.py*) exit "${STUB_STOP_RC:-0}" ;;
    *box_stage.py*fetch*)
        [ "${STUB_STAGE_RC:-0}" = 0 ] || exit "$STUB_STAGE_RC"
        into=${!#}
        mkdir -p "$into" && cp -R "$STUB_STAGE_TREE"/. "$into"/
        exit $? ;;
    *"import wesnoth_core"*) echo "${STUB_PHASE-0}"; exit 0 ;;
esac
cat > /dev/null
exit 0
"""


def find_bash() -> str | None:
    """A POSIX bash: Git's on Windows, never the WSL launcher."""
    if sys.platform == "win32":
        for candidate in (r"C:\Program Files\Git\usr\bin\bash.exe", r"C:\Program Files\Git\bin\bash.exe"):
            if os.path.exists(candidate):
                return candidate
        found = shutil.which("bash")
        if found and "system32" not in found.lower() and "windowsapps" not in found.lower():
            return found
        return None
    return shutil.which("bash")


BASH = find_bash()
# Slow: about 60 s on the laptop, where Git bash starts processes slowly;
# the library runs on Linux boxes, and CI runs the slow tier on Linux.
pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(BASH is None, reason="no POSIX bash on this machine")]


def bash_path(path: Path) -> str:
    """`path` as bash sees it (Git's bash wants /c/... on Windows)."""
    path = Path(path).resolve()
    if sys.platform == "win32" and path.drive:
        return f"/{path.drive[0].lower()}{path.as_posix()[2:]}"
    return str(path)


def run_bash(script: str, cwd: Path, timeout: float = 90) -> int:
    script_path = cwd / f"driver_{time.monotonic_ns()}.sh"
    script_path.write_bytes(script.encode())
    return subprocess.run([BASH, bash_path(script_path)], cwd=cwd, timeout=timeout,
                          stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode


def alive(pid: int | None) -> bool:
    """Whether bash can signal `pid` (on Windows, Git's bash pids are its own)."""
    return pid is not None and subprocess.run([BASH, "-c", f"kill -0 {pid} 2>/dev/null"]).returncode == 0


def eventually_dead(pid: int | None, seconds: float = 5.0) -> bool:
    end = time.monotonic() + seconds
    while alive(pid):
        if time.monotonic() > end:
            return False
        time.sleep(0.1)
    return True


class Box:
    """A work directory, the stub, and a driver that sources the library."""

    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.work = tmp / "work"
        self.out = self.work / "run"
        self.repo = self.work / "repo"
        stubs = tmp / "stubs"
        stubs.mkdir()
        (stubs / "python").write_bytes(STUB_PYTHON.encode())
        (stubs / "python").chmod(0o755)
        self.work.mkdir()
        (self.work / ".hf_token").write_bytes(b"hf_stubtoken0for0the0library0tests\r\n")  # not-a-secret
        self.stubs = stubs
        self.env = {"BOX_MAX_H": "1"}

    def prologue(self, env: dict) -> str:
        exports = "\n".join(f"export {k}='{v}'" for k, v in env.items())
        return f"""set -uo pipefail
exec > "{bash_path(self.tmp)}/driver.out" 2>&1
export PATH="{bash_path(self.stubs)}:$PATH"
export STUB_DIR="{bash_path(self.tmp)}"
{exports}
WORKDIR="{bash_path(self.work)}"
BOX_OUT="$WORKDIR/run"
export HF_DIR=tier-b/boxlib_test
BOX_MAX_H=${{BOX_MAX_H:-1}}
STAGE="${{STAGE:-{STAGE}}}"
BOX_REPO="$WORKDIR/repo"
BOX_STATE="$WORKDIR/state"
BOX_KILL_GRACE=1s
BOX_WATCH_POLL_S=0.2
. "{bash_path(LIB)}/boxlib.sh"
"""

    def run(self, body: str, init: bool = True, timeout: float = 90, **env) -> int:
        merged = {**self.env, **{k: str(v) for k, v in env.items()}}
        script = self.prologue(merged) + ("box_init\n" if init else "") + body + "\n"
        return run_bash(script, self.tmp, timeout)

    def calls(self) -> list[tuple[int, str]]:
        path = self.tmp / "calls.log"
        if not path.exists():
            return []
        rows = []
        for line in path.read_text().splitlines():
            stamp, _, args = line.partition("\t")
            rows.append((int(stamp), args))
        return rows

    def index(self, pattern: str, start: int = 0) -> int:
        """Position of the first call from `start` on matching `pattern`."""
        for i, (_t, args) in enumerate(self.calls()[start:], start):
            if re.search(pattern, args):
                return i
        raise AssertionError(f"no call matches {pattern!r}: {self.calls()}")

    def text(self, name: str) -> str:
        path = self.out / name
        return path.read_text() if path.exists() else ""

    def switch_pid(self) -> int | None:
        path = self.out / "tmp" / "deadman.pid"
        return int(path.read_text()) if path.exists() else None

    def output(self) -> str:
        path = self.tmp / "driver.out"
        return path.read_text() if path.exists() else ""


@pytest.fixture
def box(tmp_path):
    b = Box(tmp_path)
    yield b
    pid = b.switch_pid()
    if pid:
        subprocess.run([BASH, "-c", f"pkill -P {pid} 2>/dev/null; kill {pid} 2>/dev/null"])


FINAL = r"box_upload\.py .*--final"
STOP = r"box_stop\.py"


def assert_finished(box: Box, reason: str) -> None:
    """ALL_DONE and the reason are written, the final round ran with ALL_DONE
    already in place, and the stop came after it."""
    assert reason in box.text("status.txt").splitlines()[-1]
    assert reason in box.text("ALL_DONE")
    final = box.index(FINAL)
    assert box.index(STOP, final) > final
    seen = sorted(box.tmp.glob("final_upload.*.seen"))
    assert seen and "ALL_DONE" in seen[0].read_text().split()


def test_an_error_exit_uploads_with_all_done_then_stops(box):
    (box.out).mkdir(parents=True)
    (box.out / "ALL_DONE").write_text("the previous entry's\n")
    (box.out / "FAILED").write_text("the previous entry's\n")
    after_init = bash_path(box.tmp / "after_init.txt")
    assert box.run(f'ls "$BOX_OUT" > "{after_init}"\nfalse\nexit 3') == 3
    listing = (box.tmp / "after_init.txt").read_text().split()
    assert "ALL_DONE" not in listing and "FAILED" not in listing
    assert box.index(r"box_upload\.py .*--clear ALL_DONE FAILED") < box.index(FINAL)
    assert_finished(box, "UNEXPECTED_EXIT rc=3")
    assert "UNEXPECTED_EXIT rc=3" in box.text("FAILED")
    assert "entry: stage=" + STAGE in box.text("stages.txt")


def test_an_unbound_variable_finishes_the_entry(box):
    assert box.run('echo "$SURELY_NOT_SET_ANYWHERE"\necho "not reached" > "$BOX_OUT/reached"') != 0
    assert not (box.out / "reached").exists()
    assert_finished(box, "UNEXPECTED_EXIT rc=")
    assert box.text("FAILED")


def test_a_signal_during_a_step_finishes_at_once_and_ends_the_step(box):
    body = r"""( sleep 1; kill -TERM $$ ) &
box_bounded nap 1 nap.log bash -c 'echo $$ > "$BOX_OUT/nap.pid"; exec sleep 60'
echo "not reached" > "$BOX_OUT/reached"
"""
    start = time.monotonic()
    assert box.run(body) == 143
    assert time.monotonic() - start < 45
    assert not (box.out / "reached").exists()
    assert_finished(box, "UNEXPECTED_EXIT rc=143")
    assert eventually_dead(int(box.text("nap.pid")))


def test_a_clean_finish_writes_no_failed_and_disarms_the_switch(box):
    assert box.run('box_finish "RUN_DONE all good"') == 0
    assert_finished(box, "RUN_DONE all good")
    assert not (box.out / "FAILED").exists()
    assert box.switch_pid() and eventually_dead(box.switch_pid())


def test_a_refused_stop_sends_its_outcome_and_keeps_the_switch_armed(box):
    assert box.run('box_finish "RUN_DONE"', STUB_STOP_RC=1) == 0
    stop = box.index(STOP)
    box.index(r"box_upload\.py (?!.*--final)(?!.*--clear)(?!.*--restore)", stop + 1)
    assert "NOT stopped" in box.text("stop.log")
    assert alive(box.switch_pid())


def test_the_switch_finishes_the_entry_while_the_script_is_busy(box):
    body = 'sleep 5\ndate +%s > "$BOX_OUT/woke"\nbox_finish "MAIN_DONE"'
    assert box.run(body, BOX_MAX_H="0.0003") == 0
    lines = box.text("status.txt").splitlines()
    assert "DEADMAN" in lines[0] and "MAIN_DONE" in lines[-1]
    woke = int(box.text("woke"))
    stop_time = box.calls()[box.index(STOP)][0]
    assert stop_time < woke
    assert box.index(FINAL) < box.index(STOP)


REPORT = ('export BOX_OUT\nmkdir -p "$BOX_OUT/tmp"\n'
          'report() { echo "$1 $BOX_RC $BOX_WHY" >> "$BOX_OUT/report"; }\n')


def reported(box: Box) -> dict[str, str]:
    return dict(line.split(" ", 1) for line in box.text("report").splitlines())


def test_a_step_is_cut_at_its_deadline(box):
    body = REPORT + """box_bounded slow 0.02 slow.log sleep 30; report slow
box_bounded stubborn 0.02 stubborn.log bash -c 'trap "" TERM; sleep 30'; report stubborn
box_bounded quick 1 quick.log true; report quick
box_bounded broken 1 broken.log bash -c 'exit 7'; report broken
"""
    start = time.monotonic()
    box.run(body, init=False)
    assert time.monotonic() - start < 30
    # The stubborn step ignored its TERM, so the KILL ended it (137): at its
    # deadline, that is a cut too.
    assert reported(box) == {"slow": "124 cut", "stubborn": "124 cut", "quick": "0 ok",
                             "broken": "7 failed"}
    assert len(box.text("walls.txt").splitlines()) == 4


def test_a_watched_step_ends_when_silent_or_once_its_line_appears(box):
    grow = "$BOX_OUT/grow.log"
    body = REPORT + f"""echo 'EVAL[epoch0-end] from an earlier entry' > "{grow}"
box_bounded --until "{grow}" 'EVAL[epoch0-end]' until 1 until.log \\
    bash -c 'sleep 1.5; echo "EVAL[epoch0-end] step=9" >> "{grow}"; sleep 30'; report until
box_bounded --stall "{grow}" 0.03 stall 1 stall.log bash -c 'echo more >> "{grow}"; sleep 30'; report stall
"""
    start = time.monotonic()
    box.run(body, init=False)
    assert time.monotonic() - start < 30
    assert reported(box) == {"until": "143 until", "stall": "143 stalled"}
    # The earlier entry's line did not end the step: it lived to write its own.
    assert "EVAL[epoch0-end] step=9" in box.text("grow.log")
    assert "has not grown" in box.text("watchdog.log")


def test_an_unreachable_hf_at_the_entry_finishes_it(box):
    assert box.run('echo "not reached" > "$BOX_OUT/reached"', STUB_CLEAR_RC=1) == 1
    assert not (box.out / "reached").exists()
    assert_finished(box, "HF_UNREACHABLE")


def test_a_second_entry_leaves_the_running_one_alone(box, tmp_path):
    if subprocess.run([BASH, "-c", "command -v flock"], capture_output=True).returncode != 0:
        pytest.skip("flock is not installed (the box has it)")
    script = box.prologue(box.env) + "box_init\nsleep 6\nbox_finish FIRST_DONE\n"
    path = tmp_path / "first.sh"
    path.write_bytes(script.encode())
    first = subprocess.Popen([BASH, bash_path(path)], stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        if "entry:" in box.text("stages.txt"):
            break
        time.sleep(0.1)
    second = box.run('box_finish "SECOND_DONE"')
    assert first.wait(timeout=60) == 0
    assert second == 1
    assert "SECOND_DONE" not in box.text("status.txt")
    assert len([args for _t, args in box.calls() if re.search(STOP, args)]) == 1


def make_stage_tree(box: Box, library: Path = LIB) -> Path:
    tree = box.tmp / "stage_tree"
    (tree / "scripts" / "box").mkdir(parents=True)
    for f in library.iterdir():
        if f.is_file():
            shutil.copyfile(f, tree / "scripts" / "box" / f.name)
    (tree / "new_code.py").write_text("print('new')\n")
    return tree


def test_a_new_stage_replaces_the_old_tree_whole(box):
    tree = make_stage_tree(box)
    box.repo.mkdir(parents=True)
    (box.repo / "old_code.py").write_text("print('old')\n")
    (box.repo / ".staged_from").write_text("tier-b/staging/stage_old.tar.gz\n")
    body = """mkdir -p "$BOX_OUT"
staged() { echo "$1 rc=$? $(cat "$BOX_REPO/.staged_from")" >> "$BOX_OUT/staged"; }
box_stage_code; staged new
box_stage_code; staged same
STAGE=tier-b/staging/stage_next.tar.gz
export STUB_STAGE_RC=1
box_stage_code; staged missing
"""
    box.run(body, init=False, STUB_STAGE_TREE=bash_path(tree))
    assert box.text("staged").splitlines() == [f"new rc=0 {STAGE}", f"same rc=0 {STAGE}",
                                               f"missing rc=1 {STAGE}"]
    # The new tree replaced the old one; the same stage was not fetched
    # again; the stage that did not arrive left the tree in place whole.
    assert (box.repo / "new_code.py").exists() and not (box.repo / "old_code.py").exists()
    assert len([a for _t, a in box.calls() if "box_stage.py fetch" in a]) == 2


def test_a_stage_whose_library_differs_is_refused(box):
    other = box.tmp / "other_library"
    other.mkdir()
    for f in LIB.iterdir():
        if f.is_file():
            shutil.copyfile(f, other / f.name)
    (other / "boxlib.sh").write_bytes((LIB / "boxlib.sh").read_bytes() + b"# changed\n")
    tree = make_stage_tree(box, other)
    body = 'mkdir -p "$BOX_OUT"\nbox_stage_code; echo "rc=$?" > "$BOX_OUT/staged"'
    box.run(body, init=False, STUB_STAGE_TREE=bash_path(tree))
    assert box.text("staged").strip() == "rc=1"
    assert "differs from the stage's" in box.text("staging.log") and "boxlib.sh" in box.text("staging.log")


@pytest.mark.parametrize("wheel, ok", [("14", True), ("1", False), ("141", False), ("", False)])
def test_the_wheel_must_be_the_phase_the_source_declares(box, wheel, ok):
    src = box.repo / "rust" / "wesnoth_core" / "src"
    src.mkdir(parents=True)
    (src / "lib.rs").write_text('    m.add("__version__", "0.1.0")?;\n    m.add("__phase__", 14)?;\n')
    box.out.mkdir(parents=True)
    body = 'box_wheel_phase_ok > "$BOX_OUT/phase"; echo "rc=$?" >> "$BOX_OUT/phase"'
    box.run(body, init=False, STUB_PHASE=wheel)
    lines = box.text("phase").splitlines()
    assert lines == [f"wheel phase {wheel or 'none'}, source phase 14", "rc=0" if ok else "rc=1"]


def fake_hf(box: Box, files: dict[str, bytes]) -> Path:
    hf = box.tmp / "hf"
    for name, data in files.items():
        (hf / name).parent.mkdir(parents=True, exist_ok=True)
        (hf / name).write_bytes(data)
    return hf


def run_onstart(box: Box, hf: Path) -> int:
    onstart = bash_path(LIB / "box_onstart.sh")
    library = "tier-b/staging/stage_test.box"
    script = f"""exec > "{bash_path(box.tmp)}/driver.out" 2>&1
export PATH="{bash_path(box.stubs)}:$PATH" STUB_DIR="{bash_path(box.tmp)}" STUB_HF="{bash_path(hf)}"
export BOX_WORKDIR="{bash_path(box.work)}"
bash "{onstart}" demo_box.sh {library}
"""
    return run_bash(script, box.tmp)


def library_on_hf() -> dict[str, bytes]:
    return {f"tier-b/staging/stage_test.box/{f}": (LIB / f).read_bytes()
            for f in ("box_stop.py", "box_onstart.sh", "boxlib.sh", "box_upload.py", "box_stage.py")}


DEMO_SCRIPT = b'#!/usr/bin/env bash\necho "ran with $BOX_LIB" > "$BOX_LIB/../ran"\n'


def test_the_onstart_fetches_the_library_then_runs_the_script(box):
    hf = fake_hf(box, {**library_on_hf(), "tier-b/staging/demo_box.sh": DEMO_SCRIPT})
    assert run_onstart(box, hf) == 0
    assert (box.work / "ran").read_text().strip() == f"ran with {bash_path(box.work)}/box"
    for f in ("box_stop.py", "boxlib.sh", "box_upload.py", "box_stage.py"):
        assert (box.work / "box" / f).read_bytes() == (LIB / f).read_bytes()
    assert not [a for _t, a in box.calls() if re.search(r"box_stop\.py --outcome", a)]


def test_the_onstart_stops_the_instance_when_a_file_does_not_arrive(box):
    files = library_on_hf()
    del files["tier-b/staging/stage_test.box/boxlib.sh"]
    hf = fake_hf(box, {**files, "tier-b/staging/demo_box.sh": DEMO_SCRIPT})
    run_onstart(box, hf)
    assert not (box.work / "ran").exists()
    assert [a for _t, a in box.calls() if re.search(r"box_stop\.py --outcome", a)]
    assert "not fetched: boxlib.sh" in box.output()


def test_the_onstart_fetches_the_files_the_stage_carries():
    text = (LIB / "box_onstart.sh").read_text()
    listed = re.search(r"^LIBRARY_FILES=\(([^)]*)\)", text, re.M).group(1).split()
    from scripts.box.box_stage import LIBRARY_FILES
    assert tuple(listed) == LIBRARY_FILES
    assert set(LIBRARY_FILES) == {f.name for f in LIB.iterdir() if f.is_file() and f.suffix in (".sh", ".py")}


def sources_library(text: str) -> bool:
    return re.search(r"^\s*(?:\.|source)\s+\S*boxlib\.sh", text, re.M) is not None


def test_every_box_script_on_the_library_parses():
    scripts = [p for p in sorted((ROOT / "scripts").glob("*_box.sh")) if sources_library(p.read_text())]
    assert ROOT / "scripts" / "unit_vocab_retrain_box.sh" in scripts
    for path in scripts + [LIB / "boxlib.sh", LIB / "box_onstart.sh"]:
        result = subprocess.run([BASH, "-n", bash_path(path)], capture_output=True, text=True)
        assert result.returncode == 0, f"{path.name}: {result.stderr}"
