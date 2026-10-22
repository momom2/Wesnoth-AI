"""The planted round's offline readout (tools/pg_plant.py): the plant
reaches both seats through the production join in side A's frame, the
readout's SE is the SE of a sum over independent games, and on a played run
the readout passes, and fails when the join's seat sign is reversed or the
join pairs a game with another's result."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from wesnoth_ai import game_core as gc  # noqa: E402

needs_core = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")
# The planted run (helpers.pg_run.plant_run): 64 training games (eight a
# split) and 4 held out, b and beta at zero. The readout reached 6.4-6.8 SE
# on attacks there (2026-10-22).


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@needs_core
def test_a_planted_copy_reaches_both_seats_in_side_a_frame(tmp_path):
    """z' = z + delta Y in the walk's copy, Y side A's attacks less their
    logged mass, less side B's: the join gives side A z' and side B -z',
    keeps the cap from the result, and copies a game whose walk failed as
    it was."""
    from helpers.pg_games import played_games
    from tools import selfplay_rows
    from tools.match_luck import walk_matches
    from tools.pg_plant import plant_rows
    from tools.pg_positions import RecordIndex, read_record
    from wesnoth_ai.pg_records import decisions_of
    from wesnoth_ai.server_priors import KIND_ATTACK
    games = played_games()
    walk = tmp_path / "walk.jsonl"
    walk_matches([("selfplay", games.games)], walk)
    rows = _read_jsonl(walk)
    rows.append({**rows[0], "game": "game_A_B_s1_999", "error": "no record"})
    index = RecordIndex([games.games])
    delta = 0.5
    planted = plant_rows(rows, index, delta, "attack")
    copy = tmp_path / "planted.jsonl"
    copy.write_text("".join(json.dumps(r) + "\n" for r in planted), encoding="utf-8")
    out = tmp_path / "rows.jsonl"
    assert selfplay_rows.main(["--walk", str(copy), "--no-walk", "--out", str(out), "--seed-base", "0",
                               "--max-errors", "1"]) == 0
    sides = {(r["game"], r["side"]): r for r in _read_jsonl(out)}
    moved = 0
    for row in rows[:-1]:
        side_a = int(row["side_a"])
        y = sum((1 if d.side == side_a else -1) * ((d.kind == KIND_ATTACK) - d.p_att)
                for d in decisions_of(read_record(index.path(row["game"]))))
        a, b = sides[(row["game"], side_a)], sides[(row["game"], 3 - side_a)]
        assert a["z"] == pytest.approx(row["z"] + delta * y) and b["z"] == pytest.approx(-a["z"])
        assert a["capped"] == b["capped"] == (row["z"] == 0)
        moved += abs(y) > 1e-6
    assert moved == len(rows) - 1
    assert planted[-1] == rows[-1]


def test_the_split_se_is_the_se_of_a_sum_over_independent_games():
    """Per-game contributions drawn independently, dealt to eight splits of
    unequal sizes: the SE read from the splits' sums matches the sum's
    spread over repeats, and its square is unbiased."""
    from tools.pg_plant import split_total_se
    rng = np.random.default_rng(3)
    sizes = np.array([60, 90, 100, 110, 80, 120, 70, 170])
    N = int(sizes.sum())
    edges = np.concatenate([[0], np.cumsum(sizes)])
    totals, squares = [], []
    for _ in range(4000):
        c = rng.normal(0.7, 2.0, N)
        parts = [c[edges[i]:edges[i + 1]].sum() / (2 * N) for i in range(len(sizes))]
        totals.append(sum(parts))
        squares.append(split_total_se(parts, sizes) ** 2)
    true_se = 2.0 / (2 * math.sqrt(N))
    assert np.std(totals) == pytest.approx(true_se, rel=0.03)
    assert np.mean(squares) == pytest.approx(true_se ** 2, rel=0.05)
    assert math.isnan(split_total_se([1.0, 0.0], [5, 0]))


def _reversed_seat(original):
    """The join with side B given side A's result: its seat sign lost."""
    def side_rows(walk_row, seed_base):
        rows = original(walk_row, seed_base)
        z_a = next(r["z"] for r in rows if r["side"] == int(walk_row["side_a"]))
        return [{**r, "z": z_a} for r in rows]
    return side_rows


def _shifted_join(original):
    """The join pairing each game with the result of the game before it."""
    previous = {"z": 0.0}

    def side_rows(walk_row, seed_base):
        row = dict(walk_row)
        row["z"], previous["z"] = previous["z"], float(walk_row["z"])
        return original(row, seed_base)
    return side_rows


def test_delta_puts_the_plant_at_its_target_over_the_noisier_seat():
    """The rule on a main pass of known noise: each split's gradient is the
    games' mean plus noise of known per-game spread along the seat's
    readout direction. delta times the seat's projection of u is TARGET_SE
    times the noise of a pass of the planted pass's size, on the seat whose
    ratio is the larger."""
    import torch
    from tools.pg_plant import TARGET_SE, delta_rule
    rng = np.random.default_rng(5)
    games = np.full(8, 500)
    N, planted = int(games.sum()), 1000
    sigma = 3.0                                   # per-game spread of the contributions along w
    w = {"x": torch.tensor([1.0, 0.0]), "y": torch.tensor([0.0])}
    splits = [{"x": torch.tensor([float(rng.normal(0.2, sigma * math.sqrt(n)) / (2 * N)), 0.0]),
               "y": torch.tensor([0.0])} for n in games]
    sq = [{"x": torch.ones(2), "y": torch.ones(1)} for _ in games]
    main_pass = {"names": ["x", "y"], "splits": splits, "sq": sq, "split_games": games.tolist(),
                 "split_windows": [1] * 8, "games": N}
    u = [{"x": torch.tensor([0.004, 1.0]), "y": torch.tensor([2.0])}]
    grad_M = {"A": w, "B": {"x": torch.tensor([0.5, 0.0]), "y": torch.tensor([0.0])}}
    rule = delta_rule(main_pass, grad_M, u, planted)
    true_se = sigma / (2 * math.sqrt(planted))    # the noise of <w, g> at the planted pass's size
    for seat in ("A", "B"):
        assert rule["seats"][seat]["noise_se"] / rule["seats"][seat]["signal_per_delta"] == \
            pytest.approx(true_se / 0.004, rel=0.5)
    ratios = [r["noise_se"] / abs(r["signal_per_delta"]) for r in rule["seats"].values()]
    assert rule["delta"] == pytest.approx(TARGET_SE * max(ratios))


@needs_core
@pytest.mark.slow
def test_the_planted_readout_passes_and_catches_a_reversed_seat_and_a_broken_join(monkeypatch):
    """The plant at a given delta through the walk's copy, the rows, the
    labels and a planted pass; the readout per seat. Correct code passes on
    both seats. With the join's seat sign lost, side B's plant pulls the
    shared weights against side A's, and neither seat moves: both miss
    their forecast from below. With each game joined to the previous game's
    result, the step does not move the forecast's way either. (delta is
    given: at 64 games the plant's own noise caps its resolution near 6.5
    SE, and the rule's delta, 10 SE of the run's label noise alone, reaches
    about 5 of them.)"""
    from helpers.pg_run import PLANT_HELDOUT_FROM, plant_grads, plant_run, planted_readout
    from tools import selfplay_rows
    from tools.pg_plant import AGREE_SE, MIN_RESOLUTION
    import torch
    run = plant_run()
    grads_path = plant_grads(run, run.root / "plant_grads.pt", "attack", delta=3.0)
    grads = torch.load(grads_path, map_location="cpu", weights_only=True)
    assert len(grads["train_games"]) == PLANT_HELDOUT_FROM and len(grads["u"]) == 8

    good = planted_readout(run, grads_path, "good", zero_baseline=True)
    assert good["verdict"] == "pass", good["seats"]
    for seat in ("A", "B"):
        r = good["seats"][seat]
        assert r["resolution"] >= MIN_RESOLUTION and abs(r["miss_se"]) <= AGREE_SE
        assert r["forecast_truncated"] == pytest.approx(r["forecast"], rel=0.05)

    original = selfplay_rows.side_rows
    monkeypatch.setattr(selfplay_rows, "side_rows", _reversed_seat(original))
    reversed_ = planted_readout(run, grads_path, "reversed", zero_baseline=True)
    assert reversed_["verdict"] == "fail"
    for r in reversed_["seats"].values():
        assert r["miss_se"] < -AGREE_SE and abs(r["measured"]) < 0.25 * r["forecast"], reversed_["seats"]

    monkeypatch.setattr(selfplay_rows, "side_rows", _shifted_join(original))
    shifted = planted_readout(run, grads_path, "shifted", zero_baseline=True)
    assert shifted["verdict"] == "fail"
    assert all(r["miss_se"] < -AGREE_SE for r in shifted["seats"].values()), shifted["seats"]
