"""tools/analysis/endturn_readout.py on synthetic match records: the
scores, the fire ratio, and the exit codes the box script keys on."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "analysis"))

from endturn_readout import main, read_dir, verdict  # noqa: E402


def _write(path: Path, outcomes, fwd_a=12, fwd_b=10, turns=2, pa="raw:t0+endm", pb="raw:t0"):
    path.mkdir(parents=True, exist_ok=True)
    for i, o in enumerate(outcomes):
        (path / f"game_a_b_s1_{i}.json").write_text(json.dumps({
            "label_a": "a", "label_b": "b", "procedure_a": pa, "procedure_b": pb,
            "outcome_a": o, "turns": turns, "forwards_a": fwd_a, "forwards_b": fwd_b,
            "side_a": 1 + i % 2, "seed": i}), encoding="utf-8")


def test_readout_scores_and_fire_ratio(tmp_path):
    d = tmp_path / "games_screen_endm"
    _write(d, ["win", "win", "loss", "timeout", "draw"])
    r = read_dir(d)
    assert (r.wins, r.losses, r.draws, r.capped, r.games) == (2, 1, 1, 1, 5)
    assert r.p == 2 / 3 and r.decisive == 3
    assert r.p_half == (2 + 0.5 * 2) / 5
    assert r.capped_frac == 0.2
    assert r.dps_a == 6.0 and r.dps_b == 5.0 and abs(r.fire_ratio - 1.2) < 1e-12


def test_exit_codes_key_the_box_script(tmp_path, capsys):
    d = tmp_path / "games_screen_endm"
    _write(d, ["win"] * 6 + ["loss"] * 4, fwd_a=10, fwd_b=10)
    assert main([str(d), "--require-fire", "1.03"]) == 1      # does not fire
    _write(tmp_path / "fires", ["win"] * 6 + ["loss"] * 4, fwd_a=11, fwd_b=10)
    assert main([str(tmp_path / "fires"), "--require-fire", "1.03"]) == 0
    assert main([str(d), "--require-pass", "0.535"]) == 0        # p 0.6
    _write(tmp_path / "weak", ["win"] * 5 + ["loss"] * 5)
    assert main([str(tmp_path / "weak"), "--require-pass", "0.535"]) == 1
    _write(tmp_path / "games_screen_eo075", ["win"] * 7 + ["loss"] * 3, pa="raw:t0+eo-0.75")
    _write(tmp_path / "games_screen_eo150", ["win"] * 4 + ["loss"] * 6, pa="raw:t0+eo-1.5")
    capsys.readouterr()                                          # the lines above
    assert main([str(tmp_path / "games_screen_eo075"), str(tmp_path / "games_screen_eo150"),
                 "--best-p"]) == 0
    assert capsys.readouterr().out.strip() == "games_screen_eo075"


def test_verdict_applies_the_bars(tmp_path):
    _write(tmp_path / "games_screen_endm", ["win"] * 20 + ["loss"] * 20, fwd_a=11, fwd_b=10)
    _write(tmp_path / "games_endm", ["win"] * 440 + ["loss"] * 360 + ["timeout"] * 300)
    _write(tmp_path / "games_eo-0.75", ["win"] * 430 + ["loss"] * 370, pa="raw:t0+eo-0.75")
    text = verdict([read_dir(tmp_path / n) for n in ("games_screen_endm", "games_endm", "games_eo-0.75")],
                   fire=1.03, pass_p=0.535)
    assert "kill 1 (screen fire >= 1.03x): pass" in text
    assert "PASS (p 0.550)" in text
    assert "within 1 SE, the lever is act more" in text
    assert "barrier" in text and "clear" in text
    # An offset that BEATS the rule reads as "act more" too (2026-09-19:
    # -1.5 read 0.789 against the rule's 0.752); only one well below
    # the rule leaves something rule-specific.
    _write(tmp_path / "games_eo-1.5", ["win"] * 500 + ["loss"] * 300, pa="raw:t0+eo-1.5")
    _write(tmp_path / "games_eo-0.25", ["win"] * 380 + ["loss"] * 420, pa="raw:t0+eo-0.25")
    text = verdict([read_dir(tmp_path / n) for n in ("games_screen_endm", "games_endm",
                                                       "games_eo-1.5", "games_eo-0.25")],
                   fire=1.03, pass_p=0.535)
    assert "games_eo-1.5: p 0.625 against the rule's 0.550: above the rule by" in text
    assert "the lever is act more; the config scalar is the adopted form" in text
    assert "games_eo-0.25: p 0.475 against the rule's 0.550: below the rule by" in text
    assert "rule-specific" in text


def test_mixed_procedures_are_refused(tmp_path):
    import pytest
    d = tmp_path / "mixed"
    _write(d, ["win"], pa="raw:t0+endm")
    (d / "game_a_b_s1_9.json").write_text(json.dumps({
        "procedure_a": "raw:t0", "procedure_b": "raw:t0", "outcome_a": "win",
        "turns": 2, "forwards_a": 4, "forwards_b": 4}), encoding="utf-8")
    with pytest.raises(SystemExit, match="mixed procedures"):
        read_dir(d)
