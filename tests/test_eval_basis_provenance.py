"""Hex-basis provenance of eval results (2026-09-05 review): every
result records the EFFECTIVE basis per side (basis_a/basis_b: the
CLI flag, the checkpoint's relevant_set_hexes or the shared server's
hello), and the per-file guard, the batch pre-scan and elo_collect
refuse to mix bases within an outdir; an absent field is the full
board."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from tests.helpers.eval_records import current_forced_faction  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _game(**over) -> dict:
    """A prior result of the current epoch (a record without the key
    reads as epoch 1 and is refused before any basis check)."""
    from wesnoth_ai.constants import OBSERVATION_EPOCH
    g = {"label_a": "A", "label_b": "B", "outcome_a": "win", "margin_a": 0.5,
         "procedure_a": "raw", "procedure_b": "raw", "max_turns": 200,
         "combat_stream": "per_game", "observation_epoch": int(OBSERVATION_EPOCH),
         "forced_faction": current_forced_faction()}
    g.update(over)
    return g


def test_basis_guard_logic():
    from tools.eval_provenance import BASES, bases_of, basis_refusal
    assert bases_of({}) == ("full", "full")                      # absent = full board
    assert bases_of({"basis_a": "relset"}) == ("relset", "full")
    assert bases_of({"basis_a": None, "basis_b": "relset"}) == ("full", "relset")
    assert basis_refusal("g.json", {}, ("full", "full")) is None
    assert basis_refusal("g.json", {"basis_a": "relset"}, ("relset", "full")) is None
    why = basis_refusal("g.json", {"basis_a": "relset"}, ("full", "full"))
    assert why and "g.json" in why and "hex bases" in why and "refusing to mix" in why
    assert basis_refusal("g.json", {}, ("full", "relset")) is not None
    assert set(BASES) == {"full", "relset"}


def test_collect_refuses_mixed_bases_and_accepts_one(tmp_path, capsys):
    from tools import elo_collect
    d = tmp_path / "gdir"
    d.mkdir()
    (d / "game_A_B_s1_0.json").write_text(json.dumps(_game()), encoding="utf-8")
    (d / "game_A_B_s1_1.json").write_text(json.dumps(_game(basis_a="relset", basis_b="full")),
                                          encoding="utf-8")
    with pytest.raises(SystemExit, match="mixed hex bases"):
        elo_collect.main(["x", str(d), "--no-catalog"])
    # Same basis on every file (one of them pre-field): fitted, and the
    # basis is reported.
    (d / "game_A_B_s1_1.json").write_text(json.dumps(_game(basis_a="full", basis_b="full")),
                                          encoding="utf-8")
    assert elo_collect.main(["x", str(d), "--no-catalog"]) == 0
    assert "hex basis: full/full" in capsys.readouterr().out


def test_per_file_guard_refuses_other_basis_and_skips_same(tmp_path):
    from tools.elo_eval_game import main
    out = tmp_path / "out"
    out.mkdir()
    argv = ["x", "A", "dummy", "B", "dummy", "1", "7", str(out),
            "--mcts-sims", "0", "--device", "cpu"]
    (out / "game_A_B_s1_7.json").write_text(json.dumps(_game(basis_a="relset")),
                                            encoding="utf-8")
    with pytest.raises(SystemExit, match="hex bases"):
        main(argv)
    # A pre-field file played the full board: the dummy game's basis.
    (out / "game_A_B_s1_7.json").write_text(json.dumps(_game()), encoding="utf-8")
    assert main(argv) == 0


def test_batch_pre_scan_refuses_other_basis(tmp_path):
    """A file of another basis aborts the driver before any game is
    scheduled (scan_slots would count it as done otherwise)."""
    from tools.run_elo_batch import main
    out = tmp_path / "games"
    out.mkdir()
    (out / "game_A_B_s1_10000.json").write_text(json.dumps(_game(basis_b="relset")),
                                                encoding="utf-8")
    with pytest.raises(SystemExit, match="hex bases"):
        main(["x", "--label-a", "A", "--spec-a", "dummy", "--label-b", "B",
              "--spec-b", "dummy", "--outdir", str(out), "--games", "1",
              "--mcts-sims", "0", "--device", "cpu", "--jobs", "1",
              "--time-budget-min", "1", "--min-free-mb", "0"])
    assert sorted(out.glob("game_*.json")) == [out / "game_A_B_s1_10000.json"]


def test_a_side_is_labelled_with_the_basis_and_view_it_plays_in(tmp_path):
    """The labels a side's result carries, decided without a checkpoint
    read: the --relevant-set flag forces the subset, a fresh network plays
    the full board in the terrain set, and the flag is part of a worker's
    policy cache key, so a forced side never reuses the plain policy."""
    from types import SimpleNamespace

    import torch
    from tools import elo_eval_game as g
    from tools.run_elo_batch import _want_bases
    from wesnoth_ai.transformer_policy import TransformerPolicy
    assert g._effective_basis("dummy", True, None) == "full"
    assert g._effective_basis("random", False, None) == "full"
    assert g._effective_basis("random", True, None) == "relset"
    assert g._effective_terrain("random", None) == "set"
    assert g._effective_terrain("dummy", None) == "class"
    args = SimpleNamespace(spec_a="a.pt", relevant_set_a=True, spec_b="dummy", relevant_set_b=False)
    assert _want_bases(args, lambda spec: "full") == ("relset", "full")
    full = str(tmp_path / "full.pt")
    TransformerPolicy(device=torch.device("cpu"), d_model=32, num_layers=1, num_heads=2,
                      d_ff=64).save_checkpoint(full)
    g._POLICY_CACHE.clear()
    g._WORKER_MODE = True
    try:
        plain = g._policy_for(full, torch.device("cpu"), "A", False, False, False)
        forced = g._policy_for(full, torch.device("cpu"), "A", False, False, True)
        assert plain is not forced
        assert not plain._inference_encoder.relevant_set_hexes
        assert forced._inference_encoder.relevant_set_hexes
    finally:
        g._WORKER_MODE = False
        g._POLICY_CACHE.clear()
