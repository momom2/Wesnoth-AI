"""Regression tests for the hf_upload_loop hardening (2026-08-15):
on the tcs2 leg a hung `upload_file` call silently disabled escrow
for ~15 hours. These pin the two defenses -- the hard subprocess
timeout (exercised with a REAL hanging child, not a mock of the
timeout itself) and the cycle semantics (heartbeat every cycle;
change-signature only advances on a fully successful sweep, so a
timed-out file retries).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.hf_upload_loop as hul  # noqa: E402


def test_upload_timeout_kills_hanging_child(monkeypatch):
    """A child that hangs forever must be killed at the deadline and
    reported as a loud failure, not waited on. Uses the real
    subprocess kill path via the module's swappable child body."""
    monkeypatch.setattr(hul, "_CHILD_CODE",
                        "import time\ntime.sleep(60)\n")
    t0 = time.monotonic()
    ok = hul.upload_with_timeout("some.pt", "tier-b/some.pt",
                                 "repo/x", "tok", timeout_s=2)
    dt = time.monotonic() - t0
    assert ok is False
    assert dt < 15, f"kill took {dt:.1f}s -- timeout not enforced"


def test_upload_nonzero_exit_is_loud_failure(monkeypatch, capsys):
    monkeypatch.setattr(hul, "_CHILD_CODE",
                        "import sys\nsys.exit(3)\n")
    ok = hul.upload_with_timeout("some.pt", "tier-b/some.pt",
                                 "repo/x", "tok", timeout_s=10)
    assert ok is False
    assert "FAILED" in capsys.readouterr().out


def test_cycle_retries_after_failed_upload(tmp_path, monkeypatch,
                                           capsys):
    """Signature must NOT advance when any campaign upload fails, so
    the next cycle retries; and every cycle prints a heartbeat."""
    ckpt = tmp_path / "c.pt"
    ckpt.write_bytes(b"x" * 128)
    monkeypatch.setattr(hul, "FILES", [(str(ckpt), "c.pt")])
    monkeypatch.chdir(tmp_path)     # no validate_exports/games dirs

    calls = []
    fail_first = {"n": 0}

    def uploader(src, dst):
        calls.append((src, dst))
        fail_first["n"] += 1
        return fail_first["n"] > 1      # first call times out

    state = {}
    hul.run_cycle(uploader, state)      # attempt 1: fails
    assert "last_sig" not in state
    out1 = capsys.readouterr().out
    assert "cycle at" in out1 and "INCOMPLETE" in out1

    hul.run_cycle(uploader, state)      # attempt 2: same file retried
    assert state.get("last_sig") is not None
    assert len(calls) == 2 and calls[0] == calls[1]
    assert "campaign set" in capsys.readouterr().out

    hul.run_cycle(uploader, state)      # attempt 3: unchanged -> skip
    assert len(calls) == 2
    assert "nothing to upload" in capsys.readouterr().out


def test_heartbeat_prints_even_when_idle(tmp_path, monkeypatch,
                                         capsys):
    monkeypatch.setattr(hul, "FILES",
                        [(str(tmp_path / "absent.pt"), "absent.pt")])
    monkeypatch.chdir(tmp_path)
    hul.run_cycle(lambda s, d: True, {})
    out = capsys.readouterr().out
    assert out.count("hf_upload_loop: cycle") == 1
    assert "nothing to upload" in out
