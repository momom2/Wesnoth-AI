"""Round-24 guards on the eval layer: spec-path validation (C8),
turn-horizon provenance (C9), --catalog-alias validation (C10),
truncated-file replay (C11), and the eval bounce-loop plan drop
(C6)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_missing_spec_path_refused(tmp_path):
    from tools.elo_eval_game import main
    with pytest.raises(SystemExit, match="does not exist"):
        main(["x", "A", str(tmp_path / "nope.pt"),
              "B", "dummy", "1", "7", str(tmp_path / "out")])


def test_batch_missing_spec_path_refused(tmp_path):
    from tools.run_elo_batch import main
    with pytest.raises(SystemExit, match="does not exist"):
        main(["x", "--label-a", "A", "--spec-a",
              str(tmp_path / "nope.pt"), "--label-b", "B",
              "--spec-b", "dummy", "--outdir",
              str(tmp_path / "out"), "--games", "2"])


def test_horizon_mismatch_refused(tmp_path):
    """A result file played under a different --max-turns must not
    be silently reused: the horizon decides decisive-vs-absence,
    the quantity the PURE fit is built on."""
    from tools.elo_eval_game import main
    out = tmp_path / "out"
    out.mkdir()
    prev = {"procedure_a": "tcs:32", "procedure_b": "tcs:32",
            "max_turns": 60}
    (out / "game_A_B_s1_7.json").write_text(json.dumps(prev),
                                            encoding="utf-8")
    with pytest.raises(SystemExit, match="max_turns"):
        main(["x", "A", "dummy", "B", "dummy", "1", "7", str(out),
              "--max-turns", "200", "--device", "cpu"])


def test_collect_refuses_mixed_horizons(tmp_path):
    from tools import elo_collect
    d = tmp_path / "gdir"
    d.mkdir()
    for i, mt in enumerate((60, 200)):
        (d / f"game_A_B_s1_{i}.json").write_text(json.dumps({
            "label_a": "A", "label_b": "B", "outcome_a": "win",
            "margin_a": 0.5, "max_turns": mt}), encoding="utf-8")
    with pytest.raises(SystemExit, match="horizon"):
        elo_collect.main(["x", str(d), "--no-catalog"])


def test_malformed_catalog_alias_refused(tmp_path):
    """A dropped or misdirected alias records the edge under a
    phantom run-local node; both malformations must refuse before
    the catalog is touched."""
    from tools import elo_collect
    d = tmp_path / "gdir"
    d.mkdir()
    for i in range(2):
        (d / f"game_A_B_s1_{i}.json").write_text(json.dumps({
            "label_a": "A", "label_b": "B", "outcome_a": "win",
            "margin_a": 0.5, "procedure_a": "mcts:32",
            "procedure_b": "mcts:32"}), encoding="utf-8")
    cat = tmp_path / "cat.json"
    with pytest.raises(SystemExit, match="catalog-alias"):
        elo_collect.main(["x", str(d), "--catalog-path", str(cat),
                          "--catalog-alias", "CANON_ONLY"])
    with pytest.raises(SystemExit, match="not among"):
        elo_collect.main(["x", str(d), "--catalog-path", str(cat),
                          "--catalog-alias", "typo=CANON"])
    with pytest.raises(SystemExit, match="catalog-alias"):
        elo_collect.main(["x", str(d), "--catalog-path", str(cat),
                          "--catalog-alias", "A="])
    assert not cat.exists()


def test_truncated_result_file_is_replayed(tmp_path):
    """A zero-byte leftover of a killed child re-enters pending
    instead of burning a replacement slot on a phantom absence."""
    from tools.run_elo_batch import result_name, scan_slots, slot_for
    s0 = slot_for(0, 100)
    s1 = slot_for(1, 100)
    (tmp_path / result_name("A", "B", s0[0], s0[1])).write_text("")
    (tmp_path / result_name("A", "B", s1[0], s1[1])).write_text(
        json.dumps({"outcome_a": "win"}), encoding="utf-8")
    n_res, n_nores, pending, extra = scan_slots(
        tmp_path, "A", "B", 2, 100, 1)
    assert (n_res, n_nores, extra) == (1, 0, 0)
    assert [p[0] for p in pending] == [0]


def test_policy_pair_exposes_drop_last_pending():
    """The eval bounce-retry loop drops the bounced decision AND any
    cached plan before re-selecting (mirroring play_one_game); the
    pair must delegate to the policy when it has the hook."""
    from tools.eval_sim import _PolicyPair

    class _Stub:
        def __init__(self):
            self.calls = []

        def drop_last_pending(self, gl):
            self.calls.append(gl)
            return True

    p = _PolicyPair(policy=_Stub(), label="x", side=1)
    assert p.drop_last_pending("g1") is True
    assert p.policy.calls == ["g1"]
    assert _PolicyPair(policy=object(), label="x",
                       side=1).drop_last_pending("g1") is False


def test_collect_stamps_horizon_on_edge(tmp_path):
    """Round-25 C5: the measured turn horizon rides the catalog
    edge's protocol so the cross-dir horizon guard can compare
    it."""
    from tools import elo_collect
    from tools.elo_catalog import load_catalog
    d = tmp_path / "gdir"
    d.mkdir()
    for i in range(2):
        (d / f"game_A_B_s1_{i}.json").write_text(json.dumps({
            "label_a": "A", "label_b": "B", "outcome_a": "win",
            "margin_a": 0.5, "procedure_a": "mcts:32",
            "procedure_b": "mcts:32", "max_turns": 200}),
            encoding="utf-8")
    cat = tmp_path / "cat.json"
    elo_collect.main(["x", str(d), "--catalog-path", str(cat)])
    c = load_catalog(cat)
    (edge,) = c["edges"].values()
    assert edge["protocol"]["max_turns"] == 200


def test_catalog_max_turns_stamps_without_procedure(tmp_path):
    """Round-26 C2: --catalog-max-turns on a legacy dir must stamp
    even when no --catalog-procedure creates a proto -- nesting the
    stamp under `if proto is not None` silently ignored the flag
    exactly where it is needed."""
    from tools import elo_collect
    from tools.elo_catalog import load_catalog
    d = tmp_path / "gdir"
    d.mkdir()
    for i in range(2):
        (d / f"game_A_B_s1_{i}.json").write_text(json.dumps({
            "label_a": "A", "label_b": "B", "outcome_a": "win",
            "margin_a": 0.5}), encoding="utf-8")
    cat = tmp_path / "cat.json"
    elo_collect.main(["x", str(d), "--catalog-path", str(cat),
                      "--catalog-max-turns", "40"])
    c = load_catalog(cat)
    (edge,) = c["edges"].values()
    assert edge["protocol"]["max_turns"] == 40


def test_collect_all_capped_match_reports_no_rating(tmp_path):
    """Round-28 C1: a 40-game match that decided nothing must not
    render as an exact 0.0 +- 0 tie -- unrated labels report null
    in --save-json and 'n/a' in the table."""
    from tools import elo_collect
    d = tmp_path / "gdir"
    d.mkdir()
    for i in range(6):
        (d / f"game_A_B_s1_{i}.json").write_text(json.dumps({
            "label_a": "A", "label_b": "B", "outcome_a": "timeout",
            "margin_a": 0.0, "procedure_a": "mcts:32",
            "procedure_b": "mcts:32"}), encoding="utf-8")
    out = tmp_path / "res.json"
    elo_collect.main(["x", str(d), "--no-catalog",
                      "--save-json", str(out)])
    res = json.loads(out.read_text(encoding="utf-8"))
    pure = res["tables"]["PURE (decisive only, primary)"]
    assert pure["A"]["elo"] is None and pure["B"]["elo"] is None


def test_replacement_slots_preserve_side(tmp_path):
    """Round-30 C5: a replacement keeps the SIDE of the capped slot
    it replaces (the balance invariant), and the scan's chain walk
    reproduces the live scheduling deterministically."""
    from tools.run_elo_batch import (replacement_slot_for,
                                     result_name, scan_slots,
                                     slot_for)
    for i in range(6):
        base_side = slot_for(i, 100)[0]
        for gen in (1, 2, 3):
            assert replacement_slot_for(i, 100, gen)[0] == base_side
    # Base slot 1 (side 2) capped; its gen-1 replacement decisive;
    # base slot 0 decisive: 2 results, 1 no-result, 1 replacement
    # used, nothing pending.
    s1 = replacement_slot_for(1, 100, 0)
    r1 = replacement_slot_for(1, 100, 1)
    s0 = slot_for(0, 100)
    (tmp_path / result_name("A", "B", s1[0], s1[1])).write_text(
        json.dumps({"outcome_a": "timeout"}), encoding="utf-8")
    (tmp_path / result_name("A", "B", r1[0], r1[1])).write_text(
        json.dumps({"outcome_a": "win"}), encoding="utf-8")
    (tmp_path / result_name("A", "B", s0[0], s0[1])).write_text(
        json.dumps({"outcome_a": "loss"}), encoding="utf-8")
    n_res, n_nores, pending, extra = scan_slots(
        tmp_path, "A", "B", 2, 100, 5)
    assert (n_res, n_nores, extra) == (2, 1, 1)
    assert pending == []


def test_scan_budget_is_completion_order_independent(tmp_path):
    """Round-31 C2: classification is order-free -- a decisive
    replacement already on disk is counted even when the guard
    would have been spent elsewhere chain-first, and existing
    replacement files charge the guard before new grants."""
    from tools.run_elo_batch import (replacement_slot_for,
                                     result_name, scan_slots)

    def w(i, gen, oc):
        s_, seed = replacement_slot_for(i, 100, gen)
        (tmp_path / result_name("A", "B", s_, seed)).write_text(
            json.dumps({"outcome_a": oc}), encoding="utf-8")
    # Live chunk (completion order 3,2,0,1): slots 3,2 capped and
    # replaced decisively; slot 0 capped, guard exhausted; slot 1
    # won.
    w(0, 0, "timeout")
    w(1, 0, "win")
    w(2, 0, "timeout")
    w(2, 1, "win")
    w(3, 0, "timeout")
    w(3, 1, "win")
    n_res, n_nores, pending, extra = scan_slots(
        tmp_path, "A", "B", 4, 100, 2)
    assert (n_res, n_nores, extra) == (3, 3, 2)
    assert pending == []


def test_turn_config_mismatch_refused(tmp_path):
    """Round-32 C3: a TCS result file records its turn-search knob
    dict, and replaying the slot under a different frame refuses --
    the 'tcs:sims' procedure tag alone cannot see the frame the leg
    trained with."""
    from tools.elo_eval_game import _ts_config, main
    from tools.turn_search import turn_knobs_dict
    from types import SimpleNamespace
    out = tmp_path / "out"
    out.mkdir()
    prev_tc = turn_knobs_dict(_ts_config(SimpleNamespace()))
    prev_tc["boundary_frame"] = "mover"
    prev = {"procedure_a": "tcs:32", "procedure_b": "tcs:32",
            "max_turns": 200, "turn_config": prev_tc}
    (out / "game_A_B_s1_7.json").write_text(json.dumps(prev),
                                            encoding="utf-8")
    with pytest.raises(SystemExit, match="turn-search config"):
        main(["x", "A", "dummy", "B", "dummy", "1", "7", str(out),
              "--device", "cpu"])
    # Matching frame passes the guard (skips as already played).
    assert main(["x", "A", "dummy", "B", "dummy", "1", "7",
                 str(out), "--turn-boundary-frame", "mover",
                 "--device", "cpu"]) == 0


def test_per_side_sims_procedure_provenance(tmp_path):
    """--mcts-sims-a/-b give each side its own budget (the engine
    test: search-vs-no-search on the same weights). The recorded
    per-side procedure tags must reflect the per-side budgets, and
    the exists-guard must refuse a same-sims replay of a mixed-sims
    slot."""
    from tools.elo_eval_game import main
    out = tmp_path / "out"
    out.mkdir()
    prev = {"procedure_a": "mcts:32", "procedure_b": "raw",
            "max_turns": 200}
    (out / "game_A_B_s1_7.json").write_text(json.dumps(prev),
                                            encoding="utf-8")
    # Matching per-side budgets: guard passes, slot skips.
    assert main(["x", "A", "dummy", "B", "dummy", "1", "7", str(out),
                 "--mcts-sims", "32", "--mcts-sims-b", "0",
                 "--no-turn-search", "--device", "cpu"]) == 0
    # Same global sims without the per-side split: different
    # estimand, refused.
    with pytest.raises(SystemExit, match="refusing to mix"):
        main(["x", "A", "dummy", "B", "dummy", "1", "7", str(out),
              "--mcts-sims", "32", "--no-turn-search",
              "--device", "cpu"])


def test_leaf_batch_mismatch_refused(tmp_path):
    """--mcts-batch-size is estimand provenance: batched
    (virtual-loss) search explores differently than sequential B=1,
    so a slot must never be silently replayed at a different B. A
    pre-flag result file (no mcts_batch key) counts as B=1."""
    from tools.elo_eval_game import main
    out = tmp_path / "out"
    out.mkdir()
    prev = {"procedure_a": "mcts:32", "procedure_b": "mcts:32",
            "max_turns": 200}  # pre-flag file: no mcts_batch key
    (out / "game_A_B_s1_7.json").write_text(json.dumps(prev),
                                            encoding="utf-8")
    # Default B=1 matches the legacy file: guard passes, slot skips.
    assert main(["x", "A", "dummy", "B", "dummy", "1", "7", str(out),
                 "--no-turn-search", "--device", "cpu"]) == 0
    with pytest.raises(SystemExit, match="leaf-batch"):
        main(["x", "A", "dummy", "B", "dummy", "1", "7", str(out),
              "--no-turn-search", "--mcts-batch-size", "16",
              "--device", "cpu"])
    # Precision provenance: on a cpu device the effective value is
    # False, so a slot recorded as bf16 refuses an fp32 replay.
    (out / "game_A_B_s1_8.json").write_text(json.dumps(
        {**prev, "infer_bf16": True}), encoding="utf-8")
    with pytest.raises(SystemExit, match="infer_bf16"):
        main(["x", "A", "dummy", "B", "dummy", "1", "8", str(out),
              "--no-turn-search", "--device", "cpu"])
    # Forcing bf16/compile ON with a cpu device is refused outright:
    # they would silently no-op and mislabel the result.
    with pytest.raises(SystemExit, match="cuda"):
        main(["x", "A", "dummy", "B", "dummy", "1", "9", str(out),
              "--no-turn-search", "--device", "cpu", "--infer-bf16"])


def test_collect_skips_stateless_absences_in_material(tmp_path):
    """Round-32 C5: a timeout_kill artifact carries margin_a null;
    the MATERIAL diagnostic must skip it, not count a phantom
    draw at margin 0.0."""
    from tools.elo_collect import build_pairs
    games = ([{"label_a": "A", "label_b": "B", "outcome_a": "win",
               "margin_a": 0.5}] * 2
             + [{"label_a": "A", "label_b": "B",
                 "outcome_a": "timeout_kill", "margin_a": None}] * 3)
    labels, pure, mat, nores = build_pairs(games, eps=0.02)
    rec = mat[(0, 1)]
    assert (rec.wins_i + rec.wins_j + rec.draws) == 2
    assert nores[(0, 1)] == 3


def test_ts_args_choice_typo_refused(tmp_path):
    """Round-33 C3: a typo in a choice-valued --ts-args knob must
    die in the PARENT, not kill every child while the batch exits
    0."""
    from tools.run_elo_batch import main
    with pytest.raises(SystemExit):
        main(["x", "--label-a", "A", "--spec-a", "dummy",
              "--label-b", "B", "--spec-b", "dummy",
              "--outdir", str(tmp_path / "o"), "--games", "2",
              "--mcts-sims", "8",
              "--ts-args", "--turn-boundary-frame movr"])
