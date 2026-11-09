"""scripts/gpu_yield.py: the counter parsing and the step-aside policy on
synthetic samples (no Windows counter, no GPU)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from gpu_yield import GpuYield, busiest_other, parse_samples  # noqa: E402


def test_parse_keeps_each_process_busiest_engine():
    text = ("pid_2796_luid_0x00000000_0x0000cb34_phys_0_eng_3_engtype_compute 1=88.4\n"
            "pid_2796_luid_0x00000000_0x0000cb34_phys_0_eng_0_engtype_3d=2.0\n"
            "pid_9000_luid_0x00000000_0x0000cb34_phys_0_eng_0_engtype_3d=41.5\n"
            "garbage line\n"
            "pid_9000_luid_0x00000000_0x0000cb34_phys_0_eng_5_engtype_copy=notanumber\n")
    assert parse_samples(text) == {2796: 88.4, 9000: 41.5}


def test_busiest_other_skips_the_run():
    use = {2796: 88.0, 9000: 41.0, 9100: 3.0}
    assert busiest_other(use, {2796}) == (9000, 41.0)
    assert busiest_other(use, {2796, 9000, 9100}) == (None, 0.0)


def test_steps_aside_after_consecutive_busy_samples_only():
    samples = iter([{9000: 50.0}, {9000: 2.0}, {9000: 50.0}, {9000: 60.0}])
    y = GpuYield(threshold=15, busy_samples=2, sample=lambda: next(samples))
    assert y.step_aside({1}) is None          # one busy sample
    assert y.step_aside({1}) is None          # quiet again: the run resets
    assert y.step_aside({1}) is None
    assert "pid 9000" in y.step_aside({1})    # two in a row


def test_the_run_own_use_never_triggers():
    y = GpuYield(threshold=15, busy_samples=1, sample=lambda: {2796: 99.0})
    assert y.step_aside({2796}) is None


def test_waits_for_a_quiet_spell_of_the_full_length():
    """Busy at the second sample: the quiet clock restarts there, so the
    wait ends quiet_minutes after it."""
    now = [0.0]
    samples = iter([{9000: 1.0}, {9000: 80.0}] + [{9000: 1.0}] * 100)
    y = GpuYield(threshold=15, quiet_minutes=2, sample=lambda: next(samples))

    def sleep(s):
        now[0] += s
    logs = []
    y.wait_until_quiet(lambda: {1}, every_s=30, log=logs.append, clock=lambda: now[0], sleep=sleep)
    assert now[0] == 60 + 120                 # busy at 60 s, then 120 s of quiet
    assert logs and "pid 9000" in logs[0]


def test_run_match_steps_aside_and_runs_the_match_again(tmp_path):
    """A tool that runs until stopped: two busy samples stop it, a quiet
    spell lets it run again, and the second run ends on its own."""
    from scripts.local_machine import run_match
    starts = tmp_path / "starts.txt"
    code = ("import pathlib, sys, time; p = pathlib.Path(sys.argv[1]); "
            "n = len(p.read_text().split()) if p.exists() else 0; "
            "p.write_text('start ' * (n + 1)); time.sleep(30 if n == 0 else 0)")
    busy = iter([{99999: 80.0}, {99999: 80.0}])

    def sample():
        """Quiet until the first run has written its start (a loaded machine
        can take seconds to start Python), then busy twice, then quiet."""
        if starts.exists():
            return next(busy, {99999: 0.0})
        return {99999: 0.0}
    y = GpuYield(threshold=15, busy_samples=2, quiet_minutes=0.01, sample=sample)
    rc = run_match(tmp_path, "toy", ["-c", code, str(starts)], "toy.log", 5, sys.executable,
                   gpu_yield=y, poll_s=0.3)
    assert rc == 0
    assert starts.read_text().split() == ["start", "start"]
    assert "yielded to the user" in (tmp_path / "runner.log").read_text(encoding="utf-8")
