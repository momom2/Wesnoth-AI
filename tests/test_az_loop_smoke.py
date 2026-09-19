"""The learner loop end to end (slow tier): tools/az_loop.py on a tiny
network, two actors on the mini maps, two iterations, barrier and
stream. The pool smokes cover the actors and the server; this covers
the loop around them, and for the stream the publication after the
step and the columns that describe the regime.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from wesnoth_ai.transformer_policy import TransformerPolicy  # noqa: E402


def _seed_checkpoint(path: Path) -> Path:
    policy = TransformerPolicy(device=torch.device("cpu"), d_model=32,
                               num_layers=1, num_heads=2, d_ff=64)
    policy.save_checkpoint(path)
    return path


def _run(tmp_path: Path, *extra: str) -> list:
    from tools.az_loop import main
    seed = _seed_checkpoint(tmp_path / "seed.pt")
    workdir = tmp_path / "work"
    rc = main(["az_loop.py", "--seed-checkpoint", str(seed),
               "--campaign", str(tmp_path / "campaign.pt"), "--workdir", str(workdir),
               "--iterations", "2", "--games-per-iter", "2", "--actors", "2",
               "--sims", "2", "--max-turns", "2", "--mini-ratio", "1.0",
               "--device", "cpu", "--iteration-timeout", "600",
               "--kl-states", "4", "--pin-every", "100",
               "--log-level", "WARNING", *extra])
    assert rc == 0, f"az_loop exited {rc}"
    with open(workdir / "az_history.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2, rows
    assert (tmp_path / "campaign.pt").exists()
    return rows


@pytest.mark.slow
def test_az_loop_barrier_iterations_complete(tmp_path):
    rows = _run(tmp_path)
    for row in rows:
        assert int(row["n_games"]) == 2
        # Games at the two-turn cap are capped, and the loop discards a
        # capped game's experiences by design; the decisions made are
        # the witness that the actors played and shipped.
        assert int(float(row["decisions"])) > 0
        assert float(row["gen_seconds"]) > 0
        assert row["straddle_mean"] == "", "a barrier iteration has no straddle"


@pytest.mark.slow
def test_az_loop_stream_windows_publish_and_record_the_regime(tmp_path):
    rows = _run(tmp_path, "--stream")
    first, second = rows
    for row in rows:
        assert int(row["n_games"]) == 2
        assert int(float(row["decisions"])) > 0
        assert row["straddle_mean"] != "", "a stream window records its straddle"
        assert row["window_timed_out"] == "0"
        assert row["server_weights_version"] != "", "every window ends with a publication"
    # No publication precedes the first window; the two games in
    # flight at the first publication complete inside the second.
    assert float(first["straddle_mean"]) == 0.0
    assert float(second["straddle_mean"]) > 0.0
    assert int(second["server_weights_version"]) > int(first["server_weights_version"])
    assert row["step_leaves_per_s"] != ""
