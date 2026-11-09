"""scripts/match_local.py, the strength match on the local machine, and
scripts/local_machine.py, the helpers it shares with
scripts/selfplay_t1_local.py: what a CPU can check.

- the command lines: the candidate at the reference's decode, both sides at
  argmax, the local pool, every flag one its tool declares;
- the refusal of a candidate whose decode or memory differs, unless asked;
- the resume: a complete match only fitted again, a short one played again
  once, another configuration refused, a STOP file, a display-driver reset;
- the helpers: a tool's exit status, its cut and its poll, the reset count,
  the priority and keep-awake off Windows, the GPU choice, the reference's
  flags.

No GPU and no Windows API: run_elo_batch is faked (it writes result files),
elo_collect runs for real on them, and the Windows calls are reached only
with sys.platform patched and their system call faked.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.local_machine as lm  # noqa: E402
import scripts.match_local as ml  # noqa: E402
from tools.reference_player import batch_flags, load as load_reference  # noqa: E402

REF = load_reference()
REF_LABEL = REF["label"]
REF_SLOTS = int(REF["memory_slots"])
REF_EO = float(REF["decode"]["raw_end_turn_offset"])


def _ref() -> ml.Reference:
    return ml.parse_reference(batch_flags("b"))


# ---- the command lines

def test_the_match_puts_the_candidate_at_the_reference_decode_on_the_local_pool(tmp_path):
    ref = _ref()
    assert (ref.label, ref.memory, ref.end_turn_offset) == (REF_LABEL, REF_SLOTS, REF_EO)
    memory, plays, eo, diffs = ml.candidate_decode(ref, REF_SLOTS, None, None, False)
    assert (memory, plays, eo, diffs) == (REF_SLOTS, REF_SLOTS, REF_EO, [])
    cand = ml.Candidate("cand", str(tmp_path / "cand.pt"), "0" * 64, memory, plays, eo, diffs)
    argv = ml.match_argv(cand, ref, tmp_path / "games", 800, 900000, 1500, 96, 8, 38)
    line = " ".join(argv)
    assert argv[0] == "tools/run_elo_batch.py"
    assert (f"--label-a cand --spec-a {tmp_path / 'cand.pt'} --memory-a {REF_SLOTS} "
            f"--raw-end-turn-offset-a {REF_EO}") in line
    assert " ".join(batch_flags("b")) in line                  # the reference's flags, verbatim
    for part in ("--mcts-sims 0 --raw-temperature-a 0 --raw-temperature-b 0",
                 "--persistent-workers --shared-inference --no-infer-compile --device cuda --infer-fp16",
                 "--games 800 --max-extra-games 1500 --seed-base 900000",
                 "--jobs 96 --worker-games 8 --time-budget-min 38", f"--outdir {tmp_path / 'games'}"):
        assert part in line, part
    assert ml.fit_argv(tmp_path / "games", REF_LABEL, tmp_path / "fit.json") == [
        "tools/elo_collect.py", str(tmp_path / "games"), "--no-catalog", "--anchor", REF_LABEL,
        "--save-json", str(tmp_path / "fit.json")]


def test_t1_samples_both_sides_without_the_offset_and_relabels_the_reference(tmp_path):
    ref = ml.sampled(_ref())
    assert (ref.label, ref.spec, ref.memory, ref.end_turn_offset) == (f"{REF_LABEL}t1", _ref().spec, REF_SLOTS, 0.0)
    assert "--raw-end-turn-offset-b" not in ref.flags and ref.flags[ref.flags.index("--label-b") + 1] == ref.label
    memory, plays, eo, diffs = ml.candidate_decode(ref, REF_SLOTS, None, None, False)
    assert (memory, plays, eo, diffs) == (REF_SLOTS, REF_SLOTS, 0.0, [])
    cand = ml.Candidate("cand", "c.pt", "0" * 64, memory, plays, eo, diffs)
    line = " ".join(ml.match_argv(cand, ref, tmp_path, 800, 950000, 1500, 96, 8, 38, ml.DECODE_T1))
    assert "--mcts-sims 0 --raw-temperature-a 1 --raw-temperature-b 1" in line
    assert "--raw-end-turn-offset" not in line and "--raw-temperature-a 0" not in line
    assert f"--label-a cand --spec-a c.pt --memory-a {REF_SLOTS} --label-b {REF_LABEL}t1" in line


def _declared(tool: str) -> set:
    """The options a tool's parser declares, with the --no- form of a
    BooleanOptionalAction."""
    src = (ROOT / tool).read_text(encoding="utf-8")
    flags = set(re.findall(r"add_argument\(\s*f?\"(--[a-z0-9-]+)\"", src))
    for side in ("a", "b"):          # run_elo_batch declares its fp8 switches per side in a loop
        flags |= {f.replace("{side}", side) for f in re.findall(r"add_argument\(\s*f\"(--[a-z0-9{}-]+)\"", src)}
    return flags


@pytest.mark.parametrize("tool, argv", [
    ("tools/run_elo_batch.py", lambda tmp: ml.match_argv(
        ml.Candidate("cand", "c.pt", "0" * 64, 64, 64, -1.5, []), _ref(), tmp, 800, 900000, 1500, 96, 8, 38)),
    ("tools/elo_collect.py", lambda tmp: ml.fit_argv(tmp, REF_LABEL, tmp / "fit.json")),
])
def test_every_flag_the_script_passes_is_one_its_tool_declares(tool, argv, tmp_path):
    declared = _declared(tool)
    for tok in argv(tmp_path)[1:]:
        if tok.startswith("--"):
            assert tok in declared or (tok.startswith("--no-") and "--" + tok[5:] in declared), tok


def test_the_driver_imports_no_torch():
    code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); import scripts.match_local; "
            f"print('torch' in sys.modules)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip() == "False"


# ---- the candidate's decode

@pytest.mark.parametrize("slots, memory, eo, passed, plays, flag", [
    (16, None, None, None, 16, None),                 # a smaller memory: all of its own slots
    (0, None, None, None, None, None),                # no memory: no --memory-a
    (REF_SLOTS, 0, None, 0, 0, "0"),                  # the memory switched off
    (REF_SLOTS, None, -2.5, REF_SLOTS, REF_SLOTS, str(REF_SLOTS)),   # another end_turn offset
])
def test_a_candidate_whose_decode_or_memory_differs_is_refused_unless_asked(slots, memory, eo, passed, plays,
                                                                           flag):
    ref = _ref()
    with pytest.raises(SystemExit, match="DECODE_DIFFERS.*--allow-decode-change"):
        ml.candidate_decode(ref, slots, memory, eo, False)
    got_memory, got_plays, got_eo, diffs = ml.candidate_decode(ref, slots, memory, eo, True)
    assert (got_memory, got_plays) == (passed, plays) and diffs
    assert got_eo == (REF_EO if eo is None else eo)
    flags = ml.Candidate("cand", "c.pt", "0" * 64, got_memory, got_plays, got_eo, diffs).flags()
    assert (flags[flags.index("--memory-a") + 1] if "--memory-a" in flags else None) == flag


def test_a_larger_memory_plays_at_the_reference_slots():
    assert ml.candidate_decode(_ref(), 2 * REF_SLOTS, None, None, False) == (REF_SLOTS, REF_SLOTS, REF_EO, [])


def test_a_memory_the_checkpoint_does_not_have_is_refused_even_when_asked():
    with pytest.raises(SystemExit, match="--memory"):
        ml.candidate_decode(_ref(), 16, 32, None, True)
    with pytest.raises(SystemExit, match="--memory"):
        ml.candidate_decode(_ref(), 0, 8, None, True)


def test_a_reference_option_the_candidate_would_not_take_is_refused():
    with pytest.raises(SystemExit, match="REFERENCE_FLAGS"):
        ml.parse_reference(batch_flags("b") + ["--plan-b", "1"])
    with pytest.raises(SystemExit, match="REFERENCE_FLAGS"):
        ml.parse_reference(["--label-b", REF_LABEL])


def test_seeds_past_a_replacement_offset_are_refused(tmp_path):
    base = ["--label", "cand", "--checkpoint", "c.pt", "--out", str(tmp_path)]
    assert ml.parse_args(base).seed_base == ml.SEED_BASE
    with pytest.raises(SystemExit):
        ml.parse_args(base + ["--seed-base", str(ml.REPL_SEED_OFFSET - 100)])


def test_the_checkpoints_memory_slots_are_read_in_a_child(tmp_path):
    torch = pytest.importorskip("torch")
    with_memory, without, junk = tmp_path / "m.pt", tmp_path / "n.pt", tmp_path / "junk.pt"
    torch.save({"memory_slots": 16, "arch": {"d_model": 8}}, with_memory)
    torch.save({"arch": {"d_model": 8}}, without)
    junk.write_bytes(b"not a checkpoint")
    assert ml.memory_slots_of(with_memory, sys.executable) == 16
    assert ml.memory_slots_of(without, sys.executable) == 0
    with pytest.raises(SystemExit, match="could not read the memory slots"):
        ml.memory_slots_of(junk, sys.executable)


# ---- the run: run_elo_batch faked, elo_collect real

def _write_games(games_dir: Path, n: int, start: int, seed_base: int, wins_every: int = 2,
                 label_b: str = REF_LABEL, procedure: str = f"raw:t0+eo{REF_EO:g}") -> None:
    games_dir.mkdir(parents=True, exist_ok=True)
    for i in range(start, start + n):
        side = 1 if i % 2 == 0 else 2
        rec = {"label_a": "cand", "label_b": label_b, "side_a": side, "seed": seed_base + i,
               "outcome_a": "win" if i % wins_every == 0 else "loss",
               "procedure_a": procedure, "procedure_b": procedure, "infer_fp16": True}
        (games_dir / f"game_cand_{label_b}_s{side}_{seed_base + i}.json").write_text(json.dumps(rec))


class FakeTools:
    """run_tool with run_elo_batch faked: each call plays the next (games
    written, exit status) of `plan` into its --outdir; the reference's
    --ensure answers 0; elo_collect runs."""

    def __init__(self, plan):
        self.plan = list(plan)
        self.calls = []
        self.real = lm.run_tool

    def __call__(self, out, name, cmd, log_name, timeout_min, python, **kw):
        self.calls.append(cmd)
        if cmd[0] == "tools/run_elo_batch.py":
            games_dir = Path(cmd[cmd.index("--outdir") + 1])
            n, rc = self.plan.pop(0)
            have = len(list(games_dir.glob("game_*.json"))) if games_dir.is_dir() else 0
            t1 = cmd[cmd.index("--raw-temperature-a") + 1] == "1"
            _write_games(games_dir, n, have, int(cmd[cmd.index("--seed-base") + 1]),
                         label_b=cmd[cmd.index("--label-b") + 1],
                         procedure="raw:t1" if t1 else f"raw:t0+eo{REF_EO:g}")
            return rc
        if cmd[0] == "tools/reference_player.py":
            return 0
        return self.real(out, name, cmd, log_name, timeout_min, python, **kw)

    def matches(self):
        return [c for c in self.calls if c[0] == "tools/run_elo_batch.py"]


@pytest.fixture
def match_env(tmp_path, monkeypatch):
    """A candidate file, the GPU picked, no priority change, no reset, the
    checkpoint read as the reference's memory size, and the reference's
    flags read in-process (their subprocess has its own test below)."""
    ckpt = tmp_path / "cand.pt"
    ckpt.write_bytes(b"candidate weights")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(ml, "below_normal_priority", lambda: None)
    monkeypatch.setattr(ml, "keep_awake", lambda: None)
    monkeypatch.setattr(ml, "display_resets_since", lambda t0: 0)
    monkeypatch.setattr(ml, "memory_slots_of", lambda spec, python: REF_SLOTS)
    monkeypatch.setattr(ml, "reference_flags", lambda side, python: batch_flags(side))
    out = tmp_path / "match"

    def argv(*extra):
        return ["--label", "cand", "--checkpoint", str(ckpt), "--out", str(out), "--games", "8",
                "--seed-base", "874000", "--tools-python", sys.executable, "--python", sys.executable,
                "--normal-priority"] + list(extra)

    def run(plan, *extra):
        fake = FakeTools(plan)
        monkeypatch.setattr(ml, "run_tool", fake)
        return ml.main(argv(*extra)), fake

    return out, run


def test_a_short_match_is_played_again_once_then_fitted(match_env):
    out, run = match_env
    rc, fake = run([(5, 3), (3, 0)])
    assert rc == 0
    assert len(fake.matches()) == 2 and fake.matches()[0] == fake.matches()[1]
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["complete"] and result["decisive"] == 8 and (result["wins"], result["losses"]) == (4, 4)
    assert isinstance(result["elo"], float) and result["se"] > 0
    assert result["memory_a"] == REF_SLOTS and result["memory_b"] == REF_SLOTS and not result["decode_differs"]
    assert result["procedures"] == [f"raw:t0+eo{REF_EO:g} against raw:t0+eo{REF_EO:g}"]
    fit = json.loads((out / "fit.json").read_text(encoding="utf-8"))
    assert fit["tables"][ml.PURE][REF_LABEL]["elo"] == 0.0
    assert len((out / "matches.txt").read_text(encoding="utf-8").splitlines()) == 2


def test_a_complete_match_is_only_fitted_again_without_the_gpu(match_env, monkeypatch):
    out, run = match_env
    assert run([(8, 0)])[0] == 0
    first = json.loads((out / "result.json").read_text(encoding="utf-8"))
    monkeypatch.delenv("HIP_VISIBLE_DEVICES")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    rc, fake = run([])
    assert rc == 0 and fake.matches() == []
    assert [c[0] for c in fake.calls] == ["tools/elo_collect.py"]
    again = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert (again["elo"], again["se"], again["decisive"]) == (first["elo"], first["se"], first["decisive"])


def test_a_partial_match_resumes_with_the_same_command(match_env):
    out, run = match_env
    rc, fake = run([(3, 4)])                # the replacement guard spent: no second attempt
    assert rc == 1 and len(fake.matches()) == 1
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert not result["complete"] and result["decisive"] == 3
    assert "SHORT" in ml.summary(result)
    rc, fake2 = run([(5, 0)])
    assert rc == 0 and fake2.matches() == fake.matches()
    assert json.loads((out / "result.json").read_text(encoding="utf-8"))["decisive"] == 8


@pytest.mark.parametrize("extra", [["--seed-base", "875000"], ["--games", "16"], ["--max-extra-games", "0"],
                                   ["--end-turn-offset", "-2.5", "--allow-decode-change"], ["--t1"]])
def test_a_resume_with_another_configuration_is_refused(match_env, extra):
    out, run = match_env
    assert run([(0, 4)])[0] == 1            # a first start that played nothing: its record stands
    with pytest.raises(SystemExit, match="MATCH_CHANGED"):
        run([(8, 0)], *extra)
    assert run([(8, 0)])[0] == 0            # the recorded configuration still resumes


def test_a_t1_match_samples_both_sides_and_fits_against_the_relabelled_reference(match_env):
    out, run = match_env
    rc, fake = run([(8, 0)], "--t1")
    assert rc == 0
    (argv,) = fake.matches()
    line = " ".join(argv)
    assert "--raw-temperature-a 1 --raw-temperature-b 1" in line and "--raw-end-turn-offset" not in line
    assert f"--memory-a {REF_SLOTS}" in line and f"--label-b {REF_LABEL}t1" in line
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["complete"] and result["reference"] == f"{REF_LABEL}t1" and result["decode"] == ml.DECODE_T1
    assert (result["end_turn_offset_a"], result["end_turn_offset_b"]) == (0.0, 0.0)
    assert result["procedures"] == ["raw:t1 against raw:t1"] and isinstance(result["elo"], float)
    assert json.loads((out / "fit.json").read_text(encoding="utf-8"))["tables"][ml.PURE][f"{REF_LABEL}t1"]["elo"] == 0.0
    assert json.loads((out / "match.json").read_text(encoding="utf-8"))["config"]["decode"] == ml.DECODE_T1
    with pytest.raises(SystemExit, match="MATCH_CHANGED"):
        run([(8, 0)])                       # the same directory at argmax is another match


def test_games_without_the_scripts_record_are_refused(match_env):
    out, run = match_env
    _write_games(out / "games", 2, 0, 874000)
    with pytest.raises(SystemExit, match="no match.json"):
        run([(8, 0)])


def test_a_decode_difference_is_refused_before_anything_runs(match_env):
    out, run = match_env
    with pytest.raises(SystemExit, match="DECODE_DIFFERS"):
        run([(8, 0)], "--memory", "0")
    assert not (out / "match.json").exists()


def test_the_reference_label_is_not_a_candidate_label(match_env, tmp_path):
    out, _run = match_env
    with pytest.raises(SystemExit, match="reference's label"):
        ml.main(["--label", REF_LABEL, "--checkpoint", str(tmp_path / "cand.pt"), "--out", str(out),
                 "--tools-python", sys.executable, "--normal-priority"])
    assert not (out / "match.json").exists()


def test_a_stop_file_stops_the_start(match_env):
    out, run = match_env
    out.mkdir(parents=True)
    (out / "STOP").write_text("by hand\n", encoding="utf-8")
    rc, fake = run([(8, 0)])
    assert rc == 2 and fake.calls == []


@pytest.mark.parametrize("how", ["after", "during"])
def test_a_display_reset_stops_the_match_and_writes_stop(match_env, monkeypatch, how):
    out, run = match_env
    if how == "after":
        monkeypatch.setattr(ml, "display_resets_since", lambda t0: 1)
        rc, fake = run([(3, 1)])
    else:
        def stopped(out_, name, cmd, *a, **kw):
            if cmd[0] == "tools/run_elo_batch.py":
                assert kw["poll"]() is None               # the poll asks the event log
                raise lm.ToolStopped("1 display-driver reset(s) (System event 4101) during the match")
            return 0
        monkeypatch.setattr(ml, "run_tool", stopped)
        rc = ml.main(["--label", "cand", "--checkpoint", str(out.parent / "cand.pt"), "--out", str(out),
                      "--games", "8", "--seed-base", "874000", "--tools-python", sys.executable,
                      "--normal-priority"])
    assert rc == 5
    assert "4101" in (out / "STOP").read_text(encoding="utf-8")
    assert not (out / "result.json").exists()
    assert run([(5, 0)])[0] == 2                   # the next start waits for the STOP file to go


# ---- the helpers

def test_run_tool_returns_the_exit_status_and_keeps_the_output(tmp_path):
    rc = lm.run_tool(tmp_path, "probe", ["-c", "import sys; print('hello'); sys.exit(3)"], "probe.log", 1,
                     sys.executable)
    assert rc == 3
    assert "hello" in (tmp_path / "probe.log").read_text(encoding="utf-8")
    assert "probe: -c" in (tmp_path / "runner.log").read_text(encoding="utf-8")


def test_tools_run_serialized_and_match_drivers_do_not(monkeypatch):
    """A pg tool gets SERIALIZE_ENV even from a runner started without it;
    a match driver loses it even from a runner started with it."""
    monkeypatch.delenv("AMD_SERIALIZE_KERNEL", raising=False)
    assert lm.tool_env(["tools/pg_pass.py", "--fp16"])["AMD_SERIALIZE_KERNEL"] == "3"
    monkeypatch.setenv("AMD_SERIALIZE_KERNEL", "3")
    for driver in lm.UNSERIALIZED_TOOLS:
        assert "AMD_SERIALIZE_KERNEL" not in lm.tool_env([driver, "--out", "x"])
    assert "AMD_SERIALIZE_KERNEL" not in lm.tool_env(["tools/pg_eval.py"], serialize=False)


def test_run_tool_passes_the_environment_and_logs_the_commit(tmp_path, monkeypatch):
    monkeypatch.delenv("AMD_SERIALIZE_KERNEL", raising=False)
    show = "import os; print('serial=' + os.environ.get('AMD_SERIALIZE_KERNEL', 'none'))"
    assert lm.run_tool(tmp_path, "env", ["-c", show], "env.log", 1, sys.executable) == 0
    assert "serial=3" in (tmp_path / "env.log").read_text(encoding="utf-8")
    line = (tmp_path / "runner.log").read_text(encoding="utf-8")
    assert "serialized" in line and f"code {lm.code_version()}" in line
    assert lm.code_version() != "unknown"


def test_run_tool_cuts_a_tool_past_its_time(tmp_path):
    t0 = time.monotonic()
    rc = lm.run_tool(tmp_path, "sleeper", ["-c", "import time; time.sleep(60)"], "sleeper.log", 0.02,
                     sys.executable)
    assert rc == 124 and time.monotonic() - t0 < 30
    assert "past 0 min, cut" in (tmp_path / "runner.log").read_text(encoding="utf-8")


# A tool that starts a grandchild and prints its pid, then sleeps: a cut must
# end both (on Windows the venv's python.exe is a launcher, so even the tool
# is a grandchild of the process run_tool starts).
_PARENT = ("import subprocess, sys, time; "
           "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
           "print('grandchild', c.pid, flush=True); time.sleep(120)")


def _gone(psutil, pid: int, seconds: float) -> bool:
    """The process has exited (a zombie counts: reaping it is its new parent's business)."""
    deadline = time.monotonic() + seconds
    while True:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)


def _grandchild(log: Path, seconds: float = 60) -> int:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        m = re.search(r"grandchild (\d+)", log.read_text(encoding="utf-8", errors="replace"))
        if m:
            return int(m.group(1))
        time.sleep(0.1)
    raise AssertionError(f"the tool never started its grandchild: {log.read_text(errors='replace')[-500:]}")


@pytest.mark.parametrize("with_psutil", [True, False])
def test_a_cut_ends_the_tools_whole_process_tree(tmp_path, monkeypatch, with_psutil):
    psutil = pytest.importorskip("psutil")
    if not with_psutil:
        if sys.platform != "win32":
            pytest.skip("without psutil only Windows' taskkill /T reaches the tree")
        monkeypatch.setitem(sys.modules, "psutil", None)          # kill_tree's import fails
    real_wait = subprocess.Popen.wait
    pid = []

    def wait_for_the_grandchild(self, timeout=None):
        # The cut comes once the grandchild runs, however slow the machine.
        if not pid:
            pid.append(_grandchild(tmp_path / "tree.log"))
            raise subprocess.TimeoutExpired(self.args, timeout)
        return real_wait(self, timeout=timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", wait_for_the_grandchild)
    try:
        rc = lm.run_tool(tmp_path, "tree", ["-c", _PARENT], "tree.log", 0.05, sys.executable)
        assert rc == 124
        assert _gone(psutil, pid[0], 30), "the cut left the tool's grandchild running"
    finally:
        monkeypatch.undo()
        for p in pid:
            try:
                psutil.Process(p).kill()
            except psutil.NoSuchProcess:
                pass


def test_run_tool_stops_a_tool_when_its_poll_names_a_reason(tmp_path):
    answers = iter([None, "the display reset"])
    t0 = time.monotonic()
    with pytest.raises(lm.ToolStopped, match="the display reset"):
        lm.run_tool(tmp_path, "sleeper", ["-c", "import time; time.sleep(60)"], "sleeper.log", 5,
                    sys.executable, poll=lambda: next(answers), poll_s=0.2)
    assert time.monotonic() - t0 < 30
    assert "the display reset: stopped" in (tmp_path / "runner.log").read_text(encoding="utf-8")


def test_display_resets_are_read_from_the_system_log(monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=answer, stderr="")

    monkeypatch.setattr(lm.subprocess, "run", fake_run)
    monkeypatch.setattr(lm.sys, "platform", "linux")
    answer = "7\n"
    assert lm.display_resets_since(time.time()) == 0 and calls == []      # off Windows: none, nothing asked
    monkeypatch.setattr(lm.sys, "platform", "win32")
    for answer, n in (("2\r\n", 2), ("", 0), ("not a number", 0)):
        assert lm.display_resets_since(time.time() - 60) == n
    assert calls[0][0] == "powershell" and "Id=4101" in calls[0][-1] and "LogName='System'" in calls[0][-1]

    hangs = []

    def hung_run(cmd, **kw):                             # hangs `len(hangs)` < limit times, then answers
        hangs.append(cmd)
        if len(hangs) <= limit:
            raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))
        return subprocess.CompletedProcess(cmd, 0, stdout="3\n", stderr="")

    monkeypatch.setattr(lm.subprocess, "run", hung_run)
    limit = 1
    assert lm.display_resets_since(time.time() - 60) == 3 and len(hangs) == 2      # asked again once
    hangs.clear()
    limit = 2
    assert lm.display_resets_since(time.time() - 60) == lm.RESET_CHECK_TIMED_OUT and len(hangs) == 2


def test_priority_and_keep_awake_off_windows(monkeypatch):
    niced = []
    monkeypatch.setattr(lm.sys, "platform", "linux")
    monkeypatch.setattr(lm.os, "nice", niced.append, raising=False)
    lm.below_normal_priority()
    assert niced == [5]
    assert lm.keep_awake() is None


def test_the_gpu_must_be_picked(monkeypatch):
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(SystemExit, match="HIP_VISIBLE_DEVICES"):
        lm.require_visible_gpu()
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    lm.require_visible_gpu()


def test_the_tools_python_is_the_repositorys_venv_when_there_is_one(tmp_path):
    assert lm.default_tools_python(tmp_path) == sys.executable
    venv = tmp_path / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    venv.parent.mkdir(parents=True)
    venv.write_bytes(b"")
    assert lm.default_tools_python(tmp_path) == str(venv)


def test_the_reference_flags_are_the_reference_players():
    assert lm.reference_flags("b", sys.executable) == batch_flags("b")
    assert lm.reference_flags("a", sys.executable) == batch_flags("a")
