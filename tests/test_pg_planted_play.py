"""Stage P's readouts (tools/pg_planted_play.py) on synthetic matches: the
planted step's attacks counted by seat against pi_0's, a game it played
under other weights refusing the readout; the link match's mean z against
2 e times the sharpening slope."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tools.pg_planted_play as pp  # noqa: E402
from wesnoth_ai.server_priors import KIND_ATTACK, KIND_MOVE  # noqa: E402

SHA = "ab" * 32


def _match(tmp_path: Path, n: int, planted_attacks: int, base_attacks: int, monkeypatch, sha=SHA) -> Path:
    """n games, the planted player on side A's label in even games and side
    B's in odd ones; each side's decision log holds its attacks and moves."""
    rng = np.random.default_rng(3)
    games = tmp_path / "games"
    games.mkdir(parents=True)
    logs = {}
    for i in range(n):
        planted_is_a = i % 2 == 0
        side_a = 1 + (i // 2) % 2
        label_a, label_b = ("planted", "own2") if planted_is_a else ("own2", "planted")
        seat = "a" if planted_is_a else "b"
        stem = f"game_{label_a}_{label_b}_{i}"
        result = {"label_a": label_a, "label_b": label_b, "side_a": side_a, "outcome_a": "win" if i % 3 else "loss",
                  f"checkpoint_sha256_{seat}": sha, f"memory_{seat}": 64, f"procedure_{seat}": "raw:t1"}
        (games / f"{stem}.json").write_text(json.dumps(result), encoding="utf-8")
        planted_side = side_a if planted_is_a else 3 - side_a
        decisions = []
        for side in (1, 2):
            k = (planted_attacks if side == planted_side else base_attacks) + int(rng.integers(0, 3))
            decisions += [SimpleNamespace(side=side, kind=KIND_ATTACK, refused=False) for _ in range(k)]
            decisions += [SimpleNamespace(side=side, kind=KIND_MOVE, refused=False) for _ in range(5)]
            decisions.append(SimpleNamespace(side=side, kind=KIND_ATTACK, refused=True))     # refused: not played
        logs[stem] = decisions
    import tools.pg_positions as positions
    import wesnoth_ai.pg_records as records
    monkeypatch.setattr(positions, "read_record", lambda path: Path(path).name.split(".")[0])
    monkeypatch.setattr(records, "decisions_of", lambda stem: logs[stem])
    return games


def test_the_planted_steps_attacks_rise_on_each_seat(tmp_path, monkeypatch):
    games = _match(tmp_path, 40, planted_attacks=8, base_attacks=4, monkeypatch=monkeypatch)
    got = pp.planted(games, "planted", SHA)
    assert got["verdict"] == "pass" and got["weights_check"]["games_refused"] == 0
    for seat in ("A", "B"):
        s = got["seats"][seat]
        assert s["games"] == [20, 20] and s["rise"] == pytest.approx(4.0, abs=1.0) and s["z"] > 2
    flat = pp.planted(_match(tmp_path / "flat", 40, 5, 5, monkeypatch), "planted", SHA)
    assert flat["verdict"] == "fail"


def test_a_game_played_under_other_weights_refuses_the_readout(tmp_path, monkeypatch):
    games = _match(tmp_path, 10, 8, 4, monkeypatch, sha="cd" * 32)
    got = pp.planted(games, "planted", SHA)
    assert got["verdict"] == "fail" and got["weights_check"]["games_refused"] == 10


def test_the_link_reads_the_matchs_mean_z_against_twice_e_times_the_slope(tmp_path):
    games = tmp_path / "link"
    games.mkdir()
    outcomes = ["win"] * 556 + ["loss"] * 444 + ["timeout"] * 0      # mean z 0.112
    for i, o in enumerate(outcomes):
        (games / f"game_c_h_{i}.json").write_text(json.dumps({"outcome_a": o}), encoding="utf-8")
    readout = tmp_path / "readout.json"
    readout.write_text(json.dumps({"sharpening": {"slope": 2.64, "se": 0.08}}), encoding="utf-8")
    got = pp.link(games, readout, 0.02)
    assert got["forecast"] == pytest.approx(0.1056) and got["measured"] == pytest.approx(0.112)
    assert got["temperatures"]["a"] == pytest.approx(1 / 1.02) and got["verdict"] == "holds"
    off = pp.link(games, readout, 0.1)                      # forecast 0.528, read 0.112
    assert off["verdict"].startswith("misses")
