"""The training acceptance's readout (tools/trainer_acceptance_readout.py)
and the probe's per-game-side sums it reads: a repeat that changes nothing
reads 0, a shift of every decision reads the shift with no error, a shift of
one game reads its share with an error between games, the verdict follows
the bound, other game-sides or counts are refused; on the test pass the
sums give back the probe's own readings."""
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


def _read(tmp_path, a, a2, b, earlier=None):
    return readout.readout(_write(tmp_path / "a.jsonl", a, earlier), _write(tmp_path / "a2.jsonl", a2),
                           _write(tmp_path / "b.jsonl", b))


def test_a_repeat_reads_zero_and_a_uniform_shift_reads_the_shift_exactly(tmp_path):
    out = _read(tmp_path, _sides(), _sides(), _sides(ce=1.02), earlier=_sides(ce=5.0))
    ce = out["readings"]["ce_all_k0"]
    assert ce["A"] == pytest.approx(1.0) and ce["B"] == pytest.approx(1.02)
    assert ce["A2_minus_A"]["diff"] == 0 and ce["A2_minus_A"]["se"] == 0
    assert ce["B_minus_A"]["diff"] == pytest.approx(0.02) and ce["B_minus_A"]["se"] == pytest.approx(0, abs=1e-12)
    assert out["readings"]["belief_k64"]["B_minus_A"]["diff"] == pytest.approx(0, abs=1e-12)
    assert ce["within"] is False and out["pass"] is False and out["not_worse"] is False
    assert (out["games"], out["game_sides"]) == (4, 8)


def test_a_shift_of_one_game_reads_its_share_with_an_error_between_games(tmp_path):
    out = _read(tmp_path, _sides(), _sides(shift={"g1": 0.01}), _sides(shift={"g3": -0.04}))
    new = out["readings"]["ce_all_k0"]["B_minus_A"]
    n = {g: 2 * (10 + i) for i, g in enumerate(GAMES)}
    assert new["diff"] == pytest.approx(-0.04 * n["g3"] / sum(n.values()))
    assert new["se"] > 0 and new["n_games"] == 4
    rep = out["readings"]["ce_all_k0"]["A2_minus_A"]["diff"]
    assert rep == pytest.approx(0.01 * n["g1"] / sum(n.values()))
    within = abs(new["diff"]) <= abs(rep) + 2 * new["se"]
    assert out["readings"]["ce_all_k0"]["within"] == within
    assert out["readings"]["ce_all_k0"]["not_worse"] is True        # B is better than A


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
    same = readout.readout(sides_path, sides_path, sides_path)
    assert all(r["B_minus_A"]["diff"] == 0 for r in same["readings"].values())
    assert set(same["readings"]) >= {"ce_all_k0", "ce_all_k8", "belief_k0", "belief_k8"}
    print(readout.table(same))
