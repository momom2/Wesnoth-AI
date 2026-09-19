"""Terrain-view provenance of eval results (2026-09-19): every result
records the EFFECTIVE terrain view per side (terrain_a/terrain_b: the
checkpoint's terrain_multi_hot or the shared server's hello), and the
per-file guard, the batch pre-scan and elo_collect refuse to mix views
within an outdir; an absent field is the one-class view every
checkpoint before that day played in."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _game(**over) -> dict:
    from wesnoth_ai.constants import OBSERVATION_EPOCH
    g = {"label_a": "A", "label_b": "B", "outcome_a": "win", "margin_a": 0.5,
         "procedure_a": "raw", "procedure_b": "raw", "max_turns": 200,
         "combat_stream": "per_game", "observation_epoch": int(OBSERVATION_EPOCH)}
    g.update(over)
    return g


def test_terrain_guard_logic():
    from tools.run_elo_batch import TERRAIN_VIEWS, terrain_refusal, terrain_views_of
    assert terrain_views_of({}) == ("class", "class")                 # absent = one class
    assert terrain_views_of({"terrain_a": "set"}) == ("set", "class")
    assert terrain_refusal("g.json", {}, ("class", "class")) is None
    assert terrain_refusal("g.json", {"terrain_a": "set"}, ("set", "class")) is None
    why = terrain_refusal("g.json", {"terrain_a": "set"}, ("class", "class"))
    assert why and "g.json" in why and "terrain views" in why and "refusing to mix" in why
    assert terrain_refusal("g.json", {}, ("class", "set")) is not None
    assert set(TERRAIN_VIEWS) == {"class", "set"}


def test_collect_refuses_mixed_views_and_reports_one(tmp_path, capsys):
    from tools import elo_collect
    d = tmp_path / "gdir"
    d.mkdir()
    (d / "game_A_B_s1_0.json").write_text(json.dumps(_game()), encoding="utf-8")
    (d / "game_A_B_s1_1.json").write_text(json.dumps(_game(terrain_a="set")), encoding="utf-8")
    with pytest.raises(SystemExit, match="mixed terrain views"):
        elo_collect.main(["x", str(d), "--no-catalog"])
    (d / "game_A_B_s1_1.json").write_text(json.dumps(_game(terrain_a="class", terrain_b="class")),
                                          encoding="utf-8")
    assert elo_collect.main(["x", str(d), "--no-catalog"]) == 0
    assert "terrain view: class/class" in capsys.readouterr().out
    assert elo_collect.ESTIMAND_DEFAULTS["terrain_a"] == "class"


def test_per_file_guard_refuses_other_view_and_skips_same(tmp_path):
    from tools.elo_eval_game import main
    out = tmp_path / "out"
    out.mkdir()
    argv = ["x", "A", "dummy", "B", "dummy", "1", "7", str(out),
            "--mcts-sims", "0", "--device", "cpu"]
    (out / "game_A_B_s1_7.json").write_text(json.dumps(_game(terrain_b="set")), encoding="utf-8")
    with pytest.raises(SystemExit, match="terrain views"):
        main(argv)
    (out / "game_A_B_s1_7.json").write_text(json.dumps(_game()), encoding="utf-8")
    assert main(argv) == 0


def test_batch_pre_scan_refuses_other_view(tmp_path):
    from tools.run_elo_batch import main
    out = tmp_path / "games"
    out.mkdir()
    (out / "game_A_B_s1_10000.json").write_text(json.dumps(_game(terrain_a="set")),
                                                encoding="utf-8")
    with pytest.raises(SystemExit, match="terrain views"):
        main(["x", "--label-a", "A", "--spec-a", "dummy", "--label-b", "B",
              "--spec-b", "dummy", "--outdir", str(out), "--games", "1",
              "--mcts-sims", "0", "--device", "cpu", "--jobs", "1",
              "--time-budget-min", "1", "--min-free-mb", "0"])
    assert sorted(out.glob("game_*.json")) == [out / "game_A_B_s1_10000.json"]


def test_effective_view_from_checkpoint_dummy_random_and_child_peek(tmp_path):
    """A checkpoint plays in its own view, a fresh net in the set view,
    'dummy' in none; the driver reads the same answer, once per spec,
    in a child interpreter, together with the basis."""
    import torch
    from tools import elo_eval_game as g
    from tools.run_elo_batch import (_checkpoint_basis, _checkpoint_flags, _checkpoint_terrain,
                                     _want_terrains)
    from wesnoth_ai.transformer_policy import TransformerPolicy
    arch = dict(device=torch.device("cpu"), d_model=32, num_layers=1, num_heads=2, d_ff=64)
    fresh = str(tmp_path / "fresh.pt")
    old = str(tmp_path / "old.pt")
    TransformerPolicy(**arch).save_checkpoint(fresh)
    TransformerPolicy(terrain_multi_hot=False, relevant_set_hexes=True, **arch).save_checkpoint(old)

    assert g._effective_terrain("dummy", None) == "class"
    assert g._effective_terrain("random", None) == "set"
    assert g._effective_terrain(fresh, None) == "set"
    assert g._effective_terrain(old, None) == "class"

    assert _checkpoint_flags("dummy") == ("full", "class")
    assert _checkpoint_flags("random") == ("full", "set")
    assert _checkpoint_flags(old) == ("relset", "class")
    assert _checkpoint_terrain(fresh) == "set" and _checkpoint_basis(fresh) == "full"

    class _Args:
        spec_a, spec_b = old, "dummy"
    assert _want_terrains(_Args, _checkpoint_terrain) == ("class", "class")
    _Args.spec_b = fresh
    assert _want_terrains(_Args, _checkpoint_terrain) == ("class", "set")
