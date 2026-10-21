"""The rounds of training on own trajectories, one box script for all
(scripts/memory_own_trajectories_box.sh): each round's preset names the
pre-registered init, teacher and player, sources and matches; a round-2
configuration selects only `own1`'s 64-slot game-sides; every match side
plays its role's checkpoint under the reference decode; and the barriers
read the init checkpoint."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

from helpers.posix_bash import bash_path, find_bash  # noqa: E402

BASH = find_bash()
SCRIPT = ROOT / "scripts" / "memory_own_trajectories_box.sh"
needs_bash = pytest.mark.skipif(BASH is None, reason="no POSIX bash")
PARITY3 = "efba0a78518fe3c1273b408f7471f457353598482cb66e977f55a8db47f8d5ec"
OWN1 = "70925e770d43496e3ea8f9523656f201b71ada228a0362b0eb0a600c8c107f80"
OWN1_HF = "tier-b/memory_own_trajectories_20261011/own.pt"
PRESET_VARS = ("HF_DIR", "INIT_HF", "INIT_SHA256", "TEACHER_HF", "TEACHER_SHA256", "PLAYER_SHA256",
               "PLAYER_SLOTS", "SOURCE_TARS", "GEN_MATCH", "EVAL_MATCHES", "RUN_SEED", "BOX_MAX_H")


def _block(begin: str, end: str) -> str:
    text = SCRIPT.read_text(encoding="utf-8")
    return text[text.index(begin):text.index(end)]


def _bash(script: str) -> str:
    """`script` run by bash in the repository, this interpreter first on its
    PATH and the POSIX tools after it (Git's bash on Windows starts without
    them)."""
    script = f'export PATH="{bash_path(Path(sys.executable).parent)}:/usr/bin:$PATH"\n{script}'
    # The round variables are left out of the environment: on a box the running
    # script has exported its own, which a preset keeps over its defaults.
    env = {k: v for k, v in os.environ.items() if k not in PRESET_VARS + ("ROUND",)}
    out = subprocess.run([BASH, "-c", script], capture_output=True, text=True, cwd=ROOT, timeout=120, env=env)
    assert out.returncode == 0, out.stderr
    return out.stdout


def preset(round_: int) -> dict:
    """The round's preset as the box script sets it."""
    block = _block("# ---- round presets (begin)", "# ---- round presets (end)")
    lines = "".join(f'printf "%s\\t%s\\n" {v} "${{{v}}}"\n' for v in PRESET_VARS)
    out = _bash(f"set -u\nROUND={round_}\n{block}\n{lines}")
    return dict(line.split("\t", 1) for line in out.splitlines())


def _specs(text: str):
    return [spec.split() for spec in text.split(";") if spec.strip()]


@needs_bash
def test_the_round_presets_are_the_pre_registered_runs():
    one, two = preset(1), preset(2)
    assert one["INIT_SHA256"] == one["TEACHER_SHA256"] == one["PLAYER_SHA256"] == PARITY3
    assert one["INIT_HF"] == "tier-b/imitation_anneal_20261003/lower.pt" and one["GEN_MATCH"] == ""
    assert len(one["SOURCE_TARS"].split()) == 5 and one["HF_DIR"] == "tier-b/memory_own_trajectories_20261011"
    assert _specs(one["EVAL_MATCHES"]) == [
        ["m1_own64", "107000", "tuned", "64", "own64", "init", "0", "parity3_slots0"],
        ["m2_own0", "108000", "tuned", "0", "own0", "init", "0", "parity3_slots0"]]
    assert (one["RUN_SEED"], one["BOX_MAX_H"], one["PLAYER_SLOTS"]) == ("20261011", "7", "64")
    # Round 2: own1, the reference when it ran, at 64 slots throughout.
    assert two["INIT_SHA256"] == two["TEACHER_SHA256"] == two["PLAYER_SHA256"] == OWN1
    assert two["INIT_HF"] == OWN1_HF and two["PLAYER_SLOTS"] == "64"
    assert two["SOURCE_TARS"].split() == [f"tier-b/own_confirm_20261011/games_{m}.tar.gz"
                                         for m in ("own64_vs_p3", "own64_vs_own0", "own64_selfplay")]
    assert _specs(two["GEN_MATCH"]) == [["gen_own1_selfplay", "113000", "init", "64", "own1a", "init", "64", "own1b"]]
    assert [(s[1], s[2], s[3], s[5], s[6]) for s in _specs(two["EVAL_MATCHES"])] == [
        ("114000", "tuned", "64", "init", "64"), ("115000", "tuned", "0", "init", "0"),
        ("116000", "tuned", "64", "tuned", "0")]
    assert (two["HF_DIR"], two["BOX_MAX_H"]) == ("tier-b/memory_own_round2_20261011", "8")


def _result(d: Path, stem: str, a: dict, b: dict) -> None:
    res = {"side_a": 1, "outcome_a": "win"}
    for x, fields in (("a", a), ("b", b)):
        res.update({f"{k}_{x}": v for k, v in fields.items()})
    (d / f"{stem}.json").write_text(json.dumps(res), encoding="utf-8")
    (d / f"{stem}.game.jsonl.gz").write_bytes(b"")


@needs_bash
def test_a_round_2_configuration_selects_only_own1s_64_slot_sides(tmp_path):
    """The confirmation's matches and a parity3 self-pin: only the sides
    own1 played at 64 slots are own, whatever their labels."""
    from tools.own_trajectories import select_own_sides
    two = preset(2)
    own1 = {"checkpoint_sha256": two["PLAYER_SHA256"], "memory": 64, "forwards": 9}
    p3 = {"checkpoint_sha256": PARITY3, "memory": 64, "forwards": 9}
    _result(tmp_path, "game_own64_p3_s1_1", own1, p3)
    _result(tmp_path, "game_own64_own0_s1_2", own1, {**own1, "memory": 0})
    _result(tmp_path, "game_own64a_own64b_s1_3", own1, own1)
    _result(tmp_path, "game_p3a_p3b_s1_4", {**p3, "label": "own1"}, p3)
    games, counts = select_own_sides([tmp_path], two["PLAYER_SHA256"], int(two["PLAYER_SLOTS"]))
    picked = sorted((name.split("__")[1], s.side) for name, sides in games.items() for s in sides)
    assert picked == [("game_own64_own0_s1_2", 1), ("game_own64_p3_s1_1", 1), ("game_own64a_own64b_s1_3", 1),
                      ("game_own64a_own64b_s1_3", 2)]
    assert counts["own"] == 4 and counts["not_own: another checkpoint"] == 3


@needs_bash
def test_each_match_side_plays_its_roles_checkpoint_under_the_reference_decode():
    """The generation match plays the init at 64 slots on both sides; R3
    plays the tuned checkpoint at 64 against itself at 0; each side keeps
    the reference's decode and takes the label its spec gives."""
    two = preset(2)
    block = _block("# ---- match sides (begin)", "# ---- match sides (end)")
    r3 = _specs(two["EVAL_MATCHES"])[2]
    out = _bash(f"set -u\nEO=-1.5\nARM=/w/arm.pt\nINIT_PT=/w/init.pt\nTEACHER_PT=/w/init.pt\n{block}\n"
                f'match_args "{two["GEN_MATCH"]}" && echo "$M_NAME $M_SEED ${{MATCH_ARGS[*]}}"\n'
                f'match_args "{" ".join(r3)}" && echo "$M_NAME $M_SEED ${{MATCH_ARGS[*]}}"\n'
                'match_args "x 1 nobody 64 a init 0 b" || echo refused\n')
    gen, third, refused = out.splitlines()
    assert gen.split()[:2] == ["gen_own1_selfplay", "113000"]
    for side, label in (("a", "own1a"), ("b", "own1b")):
        assert f"--spec-{side} /w/init.pt" in gen and f"--memory-{side} 64" in gen
        assert f"--label-{side} {label}" in gen and f"--raw-end-turn-offset-{side} -1.5" in gen
    assert "--spec-a /w/arm.pt" in third and "--spec-b /w/arm.pt" in third
    assert "--memory-a 64" in third and "--memory-b 0" in third and third.split()[1] == "116000"
    assert refused == "refused", "an unknown role plays nothing"


def test_the_barriers_read_the_init_checkpoint(tmp_path, monkeypatch):
    """The agreement barrier reads the student and the init checkpoint it
    started from (`--init`), and passes when the student agrees more."""
    from tools import own_barriers
    probes = tmp_path / "own.probe.jsonl"
    probes.write_text("\n".join(json.dumps(r) for r in (
        {"positions": 0, "final": False, "k0": {"ce_all": 2.80}, "k64": {"ce_all": 2.70}},
        {"positions": 9, "final": True, "k0": {"ce_all": 2.81}, "k64": {"ce_all": 2.72}})), encoding="utf-8")
    read = []

    def server(spec, device):
        read.append(Path(spec).name)
        return Path(spec).name, own_barriers.SLOTS

    agreements = {"student.pt": 0.86, "own1.pt": 0.70}
    monkeypatch.setattr(own_barriers, "teacher_server", server)
    monkeypatch.setattr(own_barriers, "own_holdout_sides", lambda directory: [])
    monkeypatch.setattr(own_barriers, "agreement", lambda srv, walks, offset: {"agreement_late": agreements[srv]})
    out = tmp_path / "barriers.json"
    rc = own_barriers.main(["--student", str(tmp_path / "student.pt"), "--init", str(tmp_path / "own1.pt"),
                            "--own-sequences", str(tmp_path), "--probes", str(probes), "--out", str(out),
                            "--offset", "-1.5", "--device", "cpu"])
    result = json.loads(out.read_text(encoding="utf-8"))
    assert read == ["student.pt", "own1.pt"] and rc == 0 and result["pass"]
    assert result["agreement"]["init"]["agreement_late"] == 0.70 and result["holdout_ce"]["init"]["k0"] == 2.80
    agreements["own1.pt"] = 0.90
    assert own_barriers.main(["--student", str(tmp_path / "student.pt"), "--init", str(tmp_path / "own1.pt"),
                              "--own-sequences", str(tmp_path), "--probes", str(probes), "--out", str(out),
                              "--offset", "-1.5", "--device", "cpu"]) == own_barriers.EXIT_BARRIER


@needs_bash
def test_a_preset_ignores_the_round_variables_of_the_calling_environment(monkeypatch):
    """On a box the running script has exported round 2's HF_DIR; reading
    round 1's preset there must still give round 1's."""
    monkeypatch.setenv("HF_DIR", "tier-b/memory_own_round2_20261011")
    monkeypatch.setenv("ROUND", "2")
    assert preset(1)["HF_DIR"] == "tier-b/memory_own_trajectories_20261011"
