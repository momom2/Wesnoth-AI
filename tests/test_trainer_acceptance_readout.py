"""The training acceptance's readout (tools/trainer_acceptance_readout.py)
and the probe's per-game-side sums it reads: a repeat that changes nothing
reads 0, a shift of every decision reads the shift with no error, a shift of
one game reads its share with an error between games, the verdict follows
the bound, every arm is judged against A with the one noise and read
against B, no repeat gives no verdict, other game-sides or counts are
refused; on the test pass the sums give back the probe's own readings."""
from __future__ import annotations

import json

import pytest
from test_sequence_launches import QUIET

from tools import trainer_acceptance_readout as readout

GAMES = ("g0", "g1", "g2", "g3")
RUNS = ("k0", "k64")


def _sides(ce=1.0, belief=0.1, shift=None):
    """Two sides per game, 10 + i decisions and 12 + i positions per side,
    every decision's cross-entropy `ce` (plus shift[game]) and every
    position's belief loss `belief`."""
    shift = shift or {}
    out = []
    for i, g in enumerate(GAMES):
        for side in (1, 2):
            n, m = 10 + i, 12 + i
            c = ce + shift.get(g, 0.0)
            out.append({"file": g, "side": side, **{run: [n, n * c, m, m * belief] for run in RUNS}})
    return out


def _write(path, sides, earlier=None):
    rows = []
    if earlier is not None:
        rows.append({"positions": 10, "final": False, "sides": earlier})
    rows.append({"positions": 20, "final": True, "sides": sides})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _read(tmp_path, a, a2, b, earlier=None, **more):
    """The readout of A, A2 (None: no repeat), B and the arms in `more`."""
    arms = {"B": _write(tmp_path / "b.jsonl", b),
            **{name: _write(tmp_path / f"{name}.jsonl", sides) for name, sides in more.items()}}
    return readout.readout(_write(tmp_path / "a.jsonl", a, earlier),
                           None if a2 is None else _write(tmp_path / "a2.jsonl", a2), arms)


def test_a_repeat_reads_zero_and_a_uniform_shift_reads_the_shift_exactly(tmp_path):
    out = _read(tmp_path, _sides(), _sides(), _sides(ce=1.02), earlier=_sides(ce=5.0))
    ce, b = out["readings"]["ce_all_k0"], out["arms"]["B"]["readings"]["ce_all_k0"]
    assert ce["A"] == pytest.approx(1.0) and b["value"] == pytest.approx(1.02)
    assert ce["A2_minus_A"]["diff"] == 0 and ce["A2_minus_A"]["se"] == 0
    assert b["minus_A"]["diff"] == pytest.approx(0.02) and b["minus_A"]["se"] == pytest.approx(0, abs=1e-12)
    assert out["arms"]["B"]["readings"]["belief_k64"]["minus_A"]["diff"] == pytest.approx(0, abs=1e-12)
    assert b["within"] is False and out["arms"]["B"]["pass"] is False and out["arms"]["B"]["not_worse"] is False
    assert (out["games"], out["game_sides"]) == (4, 8)


def test_a_shift_of_one_game_reads_its_share_with_an_error_between_games(tmp_path):
    out = _read(tmp_path, _sides(), _sides(shift={"g1": 0.01}), _sides(shift={"g3": -0.04}))
    b = out["arms"]["B"]["readings"]["ce_all_k0"]
    new = b["minus_A"]
    n = {g: 2 * (10 + i) for i, g in enumerate(GAMES)}
    assert new["diff"] == pytest.approx(-0.04 * n["g3"] / sum(n.values()))
    assert new["se"] > 0 and new["n_games"] == 4
    rep = out["readings"]["ce_all_k0"]["A2_minus_A"]["diff"]
    assert rep == pytest.approx(0.01 * n["g1"] / sum(n.values()))
    within = abs(new["diff"]) <= abs(rep) + 2 * new["se"]
    assert b["within"] == within
    assert b["not_worse"] is True        # B is better than A


def test_every_arm_is_judged_against_a_with_one_noise_and_read_against_b(tmp_path):
    out = _read(tmp_path, _sides(), _sides(shift={"g1": 0.01}), _sides(ce=1.02), B64=_sides(ce=1.03),
                BHALF=_sides(shift={"g0": 0.001}))
    assert list(out["arms"]) == ["B", "B64", "BHALF"]
    rep = out["readings"]["ce_all_k0"]["A2_minus_A"]["diff"]
    for name in ("B64", "BHALF"):
        r = out["arms"][name]["readings"]["ce_all_k0"]
        assert r["bound"] == pytest.approx(abs(rep) + 2 * r["minus_A"]["se"])
    b64 = out["arms"]["B64"]["readings"]["ce_all_k0"]
    assert b64["minus_A"]["diff"] == pytest.approx(0.03) and b64["minus_B"]["diff"] == pytest.approx(0.01)
    assert "minus_B" not in out["arms"]["B"]["readings"]["ce_all_k0"]
    assert out["arms"]["B64"]["pass"] is False and out["arms"]["BHALF"]["pass"] is True
    text = readout.table(out)
    assert "B64: pass False" in text and "BHALF: pass True" in text


def test_without_a_repeat_the_differences_stand_without_a_verdict(tmp_path):
    out = _read(tmp_path, _sides(), None, _sides(ce=1.02))
    b = out["arms"]["B"]
    assert b["readings"]["ce_all_k0"]["minus_A"]["diff"] == pytest.approx(0.02)
    assert "bound" not in b["readings"]["ce_all_k0"] and b["pass"] is None and b["not_worse"] is None
    assert "A2_minus_A" not in out["readings"]["ce_all_k0"]
    with pytest.raises(SystemExit, match="none named A"):
        readout.readout(tmp_path / "a.jsonl", None, {"A2": tmp_path / "b.jsonl"})


def test_other_game_sides_or_counts_are_refused(tmp_path):
    fewer = [s for s in _sides() if s["file"] != "g3"]
    with pytest.raises(SystemExit, match="other game-sides"):
        _read(tmp_path, _sides(), _sides(), fewer)
    recount = _sides()
    recount[0]["k64"][0] += 1
    with pytest.raises(SystemExit, match="other decisions"):
        _read(tmp_path, _sides(), recount, _sides())


@pytest.mark.slow
def test_the_probe_sides_give_back_the_probe_readings(tmp_path, pass_inputs):
    """A short pass's final probe: its per-game-side sums pool to the
    probe's ce_all and belief at every slot count, and a readout of the pass
    against itself reads 0 everywhere (one holdout game: no standard error)."""
    from tools import sequence_train
    out = tmp_path / "p.pt"
    assert sequence_train.main([*pass_inputs, *QUIET, "--out", str(out), "--packed-trunk"]) == 0
    probe = [json.loads(line) for line in out.with_suffix(".probe.jsonl").read_text(encoding="utf-8").splitlines()][-1]
    sides_path = out.with_suffix(".probe_sides.jsonl")
    sides = [json.loads(line) for line in sides_path.read_text(encoding="utf-8").splitlines()][-1]
    assert sides["final"] and probe["final"] and sides["positions"] == probe["positions"]
    assert len(sides["sides"]) == probe["n_game_sides"] > 0
    for k in ("k0", "k8", "k8_reset"):
        decisions = sum(s[k][0] for s in sides["sides"])
        positions = sum(s[k][2] for s in sides["sides"])
        assert decisions == probe[k]["n_decisions"] and positions == probe[k]["n_positions"]
        assert sum(s[k][1] for s in sides["sides"]) / decisions == pytest.approx(probe[k]["ce_all"], rel=1e-12)
        assert sum(s[k][3] for s in sides["sides"]) / positions == pytest.approx(probe[k]["belief"], rel=1e-12)
    same = readout.readout(sides_path, sides_path, {"B": sides_path})
    assert all(r["minus_A"]["diff"] == 0 for r in same["arms"]["B"]["readings"].values())
    assert set(same["readings"]) >= {"ce_all_k0", "ce_all_k8", "belief_k0", "belief_k8"}
    print(readout.table(same))
