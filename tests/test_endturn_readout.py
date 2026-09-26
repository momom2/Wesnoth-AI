"""tools/analysis/endturn_readout.py on synthetic match records: the
scores, the fire ratio, and the exit codes the box script keys on."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "analysis"))

from endturn_readout import Readout, main, read_dir, verdict  # noqa: E402


def _write(path: Path, outcomes, fwd_a=12, fwd_b=10, turns=2, pa="raw:t0+endm", pb="raw:t0",
           seed_base=0):
    path.mkdir(parents=True, exist_ok=True)
    for i, o in enumerate(outcomes):
        (path / f"game_a_b_s1_{i}.json").write_text(json.dumps({
            "label_a": "a", "label_b": "b", "procedure_a": pa, "procedure_b": pb,
            "outcome_a": o, "turns": turns, "forwards_a": fwd_a, "forwards_b": fwd_b,
            "side_a": 1 + i % 2, "seed": seed_base + i}), encoding="utf-8")


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
    _write(tmp_path / "games_endm", ["win"] * 440 + ["loss"] * 360 + ["timeout"] * 300,
           seed_base=42000)
    _write(tmp_path / "games_eo-0.75", ["win"] * 430 + ["loss"] * 370, pa="raw:t0+eo-0.75",
           seed_base=44000)
    text = verdict([read_dir(tmp_path / n) for n in ("games_screen_endm", "games_endm", "games_eo-0.75")],
                   fire=1.03, pass_p=0.535)
    assert "kill 1 (screen fire >= 1.03x): pass" in text
    assert "PASS (p 0.550)" in text
    assert "within 1 SE, the lever is act more" in text
    assert "barrier" in text and "clear" in text


def _readout(name: str, wins: int, losses: int, procedure_a: str) -> Readout:
    return Readout(name=name, games=wins + losses, wins=wins, losses=losses, draws=0,
                   capped=0, dps_a=9.0, dps_b=6.0, procedure_a=procedure_a,
                   procedure_b="raw:t0")


def test_attribution_reads_the_se_of_the_difference():
    """The rule's match and an offset's match are independent (disjoint
    seed bases), so their gap is read in SE of the difference,
    sqrt(se_rule^2 + se_offset^2). On the recorded 2026-09-19 pair (rule
    602-198, offset -1.5 631-169) that is 1.7 SE; the rule's SE alone
    reads 2.4. An offset 1.1 rule-SEs below the rule is within 1 SE of
    the difference. Outside 1 SE, either way, the pre-registered reading
    is "differs, rule-specific"."""
    rule = _readout("games_endm", 602, 198, "raw:t0+endm")
    arms = [_readout("games_eo-1.5", 631, 169, "raw:t0+eo-1.5"),
            _readout("games_eo-0.5", 588, 212, "raw:t0+eo-0.5"),
            _readout("games_eo-4", 680, 120, "raw:t0+eo-4"),
            _readout("games_eo-0.25", 540, 260, "raw:t0+eo-0.25")]
    text = verdict([rule] + arms, fire=1.03, pass_p=0.535)
    lines = {line.split(":")[0]: line for line in text.splitlines()
             if line.startswith("attribution ")}
    recorded = lines["attribution games_eo-1.5"]
    assert "+1.7 SE of the difference" in recorded
    assert recorded.endswith("differs, rule-specific")
    assert lines["attribution games_eo-0.5"].endswith("within 1 SE, the lever is act more")
    assert lines["attribution games_eo-4"].endswith("differs, rule-specific")
    assert lines["attribution games_eo-0.25"].endswith("differs, rule-specific")


def test_mixed_procedures_are_refused(tmp_path):
    import pytest
    d = tmp_path / "mixed"
    _write(d, ["win"], pa="raw:t0+endm")
    (d / "game_a_b_s1_9.json").write_text(json.dumps({
        "procedure_a": "raw:t0", "procedure_b": "raw:t0", "outcome_a": "win",
        "turns": 2, "forwards_a": 4, "forwards_b": 4}), encoding="utf-8")
    with pytest.raises(SystemExit, match="mixed procedures"):
        read_dir(d)


def test_arms_that_share_game_slots_get_no_independent_se(tmp_path):
    """Two arms that replayed the same (side, seed) slots are correlated:
    the readout says so instead of quoting sqrt(se_a^2 + se_b^2)."""
    from endturn_readout import diff_se
    _write(tmp_path / "games_endm", ["win"] * 30 + ["loss"] * 20)
    _write(tmp_path / "games_eo-1.5", ["win"] * 28 + ["loss"] * 22, pa="raw:t0+eo-1.5")
    rule, offset = read_dir(tmp_path / "games_endm"), read_dir(tmp_path / "games_eo-1.5")
    assert diff_se(offset, rule) is None
    text = verdict([rule, offset], fire=1.03, pass_p=0.535)
    assert "share 50 game slots" in text and "within 1 SE" not in text
