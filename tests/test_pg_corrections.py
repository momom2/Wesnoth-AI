"""The labels' per-decision corrections (tools/pg_corrections.py): its sums
are the replay's own per-decision value and luck (tools/pg_positions.py
`luck`); a correction enters by the design's rule (2 SE, 5 %) and I-V, and
gets coefficient 0 otherwise; the command line writes what tools/pg_pass.py
--corrections reads."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from wesnoth_ai import game_core as gc  # noqa: E402

needs_core = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")


@needs_core
def test_the_sums_are_the_replays_own_value_and_luck():
    from helpers.pg_games import played_games
    from tools.match_luck import COLUMNS
    from tools.pg_corrections import side_sums
    from tools.pg_positions import positions_of, read_record
    games = played_games()
    vocab = (dict(games.encoder.unit_type_to_id), dict(games.encoder.faction_to_id))
    beta = np.array([0.3, -0.2, 0.1, 0.5, -0.4, 0.2, 0.7][:len(COLUMNS)])
    seen_luck = False
    for stem, path in games.records.items():
        sums = side_sums(str(path), tuple(beta))
        pos = positions_of(read_record(path), *vocab, game=stem, luck=True)
        for side, rows in pos.sides.items():
            v = np.array([p.v for p in rows])
            y = np.array([float(beta @ np.array(p.luck)) for p in rows])
            want = {"n": len(rows), "v": v.sum(), "vv": (v * v).sum(), "y": y.sum(), "vy": (v * y).sum(),
                    "yy": (y * y).sum()}
            for k, x in want.items():
                assert sums[side][k] == pytest.approx(x, rel=1e-9, abs=1e-9), (stem, side, k)
            seen_luck = seen_luck or bool(np.any(y != 0))
    assert seen_luck


def _rows(games=4000, seed=1, iv_bias=0.0):
    """Synthetic game-sides: R follows each side's mean value reading, the
    luck term y is unrelated; `iv_bias` shifts v's I-V sums."""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(games):
        for side in (1, 2):
            n = 40
            m = rng.normal()
            v = m + 0.3 * rng.normal(size=n)
            y = rng.normal(size=n)
            R = 0.8 * m + 0.6 * rng.normal()
            rows.append({"game": f"g{g}", "fold": g % 2, "R": R, "n": n, "v": v.sum(), "vv": (v * v).sum(),
                         "y": y.sum(), "vy": (v * y).sum(), "yy": (y * y).sum(),
                         "iv_v": rng.normal(size=3) + iv_bias, "iv_y": rng.normal(size=3)})
    return rows


def test_a_correction_enters_by_the_rule_and_its_i_v():
    from tools.pg_corrections import decide
    got = decide(_rows())
    assert got["enters"] == {"v": True, "luck": False}
    assert got["alpha"] > 0.5 and got["gamma"] == 0.0
    assert got["readings"]["v"]["cut"] > 2 * got["readings"]["v"]["cut_se"]
    refused = decide(_rows(iv_bias=1.0))                       # v's I-V means sit far from 0
    assert refused["i_v_pass"]["v"] is False and refused["enters"]["v"] is False and refused["alpha"] == 0.0


@needs_core
def test_the_command_line_writes_what_the_pass_reads(tmp_path):
    from helpers.pg_games import played_games, write_labels
    from tools.match_luck import COLUMNS
    from tools.pg_corrections import main
    games = played_games()
    names = list(games.records)
    labels = write_labels(tmp_path / "labels.jsonl", {g: 0 for g in names},
                          {(g, s): 0.5 if s == 1 else -0.5 for g in names for s in (1, 2)}, heldout=names)
    fits = tmp_path / "fits.json"
    fits.write_text(json.dumps({"fits": {"train": {"beta": [0.1] * len(COLUMNS)}}}), encoding="utf-8")
    out = tmp_path / "corrections.json"
    assert main(["--games", str(games.games), "--labels", str(labels), "--fits", str(fits), "--out", str(out)]) == 0
    c = json.loads(out.read_text(encoding="utf-8"))
    assert {"alpha", "gamma", "beta", "enters", "readings", "i_v"} <= set(c) and len(c["beta"]) == len(COLUMNS)
    assert c["games"] == len(names) and c["fitted_on"] == "the held-out games"
