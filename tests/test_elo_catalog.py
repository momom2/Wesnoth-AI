"""Elo-catalog tests: idempotent edge recording, global refit
chaining, reference anchoring, and the elo_collect auto-update hook
-- production code paths on synthetic edges.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.elo_catalog import (  # noqa: E402
    load_catalog, record_edge, refit, save_catalog, update_from_games,
)


def _fresh(tmp_path) -> Path:
    return tmp_path / "cat.json"


def test_edge_upsert_is_idempotent(tmp_path):
    p = _fresh(tmp_path)
    cat = load_catalog(p)
    record_edge(cat, "dirX:a~b", "A", "B", 30, 0, 10)
    record_edge(cat, "dirX:a~b", "A", "B", 25, 5, 10)   # replaces
    assert len(cat["edges"]) == 1
    assert cat["edges"]["dirX:a~b"]["wins_a"] == 25
    refit(cat)
    save_catalog(cat, p)
    cat2 = load_catalog(p)
    assert cat2["checkpoints"]["A"]["n_games"] == 40


def test_refit_chains_and_anchors(tmp_path):
    """A beats ref 70-30; C beats A 70-30 -> C chains ABOVE A above
    ref, all anchored; D (no path to ref) is flagged unanchored."""
    p = _fresh(tmp_path)
    cat = load_catalog(p)
    cat["reference"] = {"label": "ref", "elo": 0.0}
    record_edge(cat, "s1", "A", "ref", 70, 0, 30)
    record_edge(cat, "s2", "C", "A", 70, 0, 30)
    record_edge(cat, "s3", "D", "E", 10, 0, 10)
    refit(cat)
    ck = cat["checkpoints"]
    assert abs(ck["ref"]["elo"]) < 1e-6
    assert ck["A"]["elo"] > 80          # ~+147 with prior shrinkage
    assert ck["C"]["elo"] > ck["A"]["elo"] + 80
    assert ck["A"]["anchored"] and ck["C"]["anchored"]
    assert not ck["D"]["anchored"] and not ck["E"]["anchored"]


def test_update_from_games_hook(tmp_path):
    """The elo_collect hook: raw game records aggregate to a PURE
    edge, keyed by dir name, and the catalog file lands on disk.
    Non-decisive games are no-result absences (user ruling
    2026-08-17), recorded on the edge but outside the W-D-L."""
    p = _fresh(tmp_path)
    games = (
        [{"label_a": "new", "label_b": "old", "outcome_a": "win"}] * 26
        + [{"label_a": "new", "label_b": "old", "outcome_a": "loss"}] * 10
        + [{"label_a": "new", "label_b": "old", "outcome_a": "draw"}] * 4
    )
    update_from_games(Path("eval_games/run1"), games, path=p)
    cat = load_catalog(p)
    (key, edge), = cat["edges"].items()
    assert key == "run1:new~old"
    assert (edge["wins_a"], edge["draws"], edge["wins_b"]) == (26, 0, 10)
    assert edge["no_result"] == 4
    assert cat["checkpoints"]["new"]["elo"] > \
        cat["checkpoints"]["old"]["elo"]
    # Re-collecting the same dir must not double-count.
    update_from_games(Path("eval_games/run1"), games, path=p)
    cat = load_catalog(p)
    assert len(cat["edges"]) == 1
    assert cat["checkpoints"]["new"]["n_games"] == 36
