"""scripts/selfplay_t1_local.py: a failing readout gate does not read as done.

tools/selfplay_readout.py exits 1 when a gate fails, and the box script
(scripts/selfplay_t1_box.sh, `gates()`) then ends SELFPLAY_T1_INCOMPLETE.
The local runner read the step as "read" and exited 0; it now returns the
gates' words and exits 1 unless every gate passed. Its process helpers are
tested with scripts/match_local.py's (tests/test_match_local.py).
"""
from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.selfplay_t1_local as st  # noqa: E402

PASSING = {k: {"pass": True} for k in st.GATES}


def _readout(path: Path, **failing) -> Path:
    r = {k: {"pass": not failing.get(k, False)} for k in st.GATES}
    path.write_text(json.dumps(r), encoding="utf-8")
    return path


def test_the_gates_read_as_the_box_scripts_words(tmp_path):
    assert st.gates_of(_readout(tmp_path / "r.json")) == " ".join(f"{k}=pass" for k in st.GATES)
    words = st.gates_of(_readout(tmp_path / "r.json", dice_control=True))
    assert "dice_control=FAIL" in words and words.count("=pass") == 4
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert st.gates_of(tmp_path / "bad.json") == "unreadable"
    assert st.gates_of(tmp_path / "absent.json") == "unreadable"


@pytest.mark.parametrize("words, ok", [
    (" ".join(f"{k}=pass" for k in st.GATES), True),
    (" ".join(f"{k}={'FAIL' if k == 'i_v' else 'pass'}" for k in st.GATES), False),
    ("i_sample=pass legal_mass=pass i_count=pass", False),      # under 4 labelled games: no dice control, no I-V
    ("unreadable", False), ("not read", False), ("FAILED: the labels, rc=1 (labels.log)", False), ("", False),
])
def test_a_run_is_done_only_when_every_gate_passed(words, ok):
    assert st.gates_passed(words) is ok
    assert st.verdict("played", words, "done", "done") == (0 if ok else 1)


def test_the_other_steps_still_decide_the_exit():
    words = " ".join(f"{k}=pass" for k in st.GATES)
    assert st.verdict("played", words, "done", "done") == 0
    assert st.verdict("FAILED at s03", words, "done", "done") == 1
    assert st.verdict("played", words, "FAILED: its fit failed (fit_d0.log)", "done") == 1
    assert st.verdict("played", words, "done", "FAILED: x") == 1


def test_a_readout_whose_gate_fails_is_marked_read_with_its_failure(tmp_path, monkeypatch):
    run = st.Run(Namespace(out=tmp_path, python=sys.executable, tools_python=sys.executable))
    for k in range(run.n_chunks()):
        (tmp_path / f"rows_s{k:02d}.jsonl").write_text("{}\n", encoding="utf-8")
    calls = []

    def fake_run(name, cmd, log_name, timeout_min, python):
        calls.append(name)
        if name == "labels":
            (tmp_path / "labels.jsonl").write_text("", encoding="utf-8")
            return 0
        _readout(tmp_path / "readout.json", i_sample=True)
        return 1                                           # selfplay_readout: a gate failed

    monkeypatch.setattr(run, "run", fake_run)
    words = run.read_selfplay()
    assert calls == ["labels", "readout"]
    assert "i_sample=FAIL" in words and not st.gates_passed(words)
    assert run.marked("readout")                           # the step ran to its end: not run again
    assert run.read_selfplay() == words and calls == ["labels", "readout"]
    assert st.verdict("played", words, "done", "done") == 1


def test_a_rounds_player_replaces_the_checkpoint_and_its_measures_may_be_skipped():
    """A round's games (scripts/rounds_local.py): the reference's flags with
    its label and checkpoint replaced and its end_turn offset dropped; a run
    without D0 and the memory report passes on its self-play and gates."""
    r = st.Run.__new__(st.Run)
    r.reference_flags = lambda side: [f"--label-{side}", "own2", f"--spec-{side}", "training/checkpoints/own2.pt",
                                      f"--memory-{side}", "64", f"--raw-end-turn-offset-{side}", "-1.5"]
    assert r.side_flags("b", "pi2b", spec="C:/rounds/pi_2.pt") == \
        ["--label-b", "pi2b", "--spec-b", "C:/rounds/pi_2.pt", "--memory-b", "64"]
    assert r.side_flags("a", "own2a") == ["--label-a", "own2a", "--spec-a", "training/checkpoints/own2.pt",
                                          "--memory-a", "64"]
    words = " ".join(f"{k}=pass" for k in st.GATES)
    assert st.verdict("played", words, "skipped", "skipped") == 0
    assert st.verdict("played", words, "skipped", "FAILED: x") == 1
