"""tools/turn_gap.py: the turn-level value gap measurement (phase-2
prerequisite, docs/turn_gap_prereg_20260904.md). CPU, a tiny
random-init policy, fresh mini-map positions instead of the manifest."""
from __future__ import annotations

import math
import pickle
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from tools import turn_gap as tg
from tools.scenario_pool import build_scenario_gamestate, random_setup
from tools.wesnoth_sim import WesnothSim
from wesnoth_ai.classes import state_key
from wesnoth_ai.transformer_policy import TransformerPolicy


@pytest.fixture(scope="module")
def policy():
    torch.manual_seed(0)
    return TransformerPolicy(d_model=32, num_layers=1, num_heads=2, d_ff=64,
                             device=torch.device("cpu"))


@pytest.fixture(scope="module")
def positions():
    out = []
    for i, seed in enumerate((3, 5)):
        setup = random_setup(random.Random(seed), mini_maps=True)
        # The constructor wires scenario events and fires init_side(1):
        # the boundary as the mover sees it, as the manifest provides.
        sim = WesnothSim(build_scenario_gamestate(setup), scenario_id=setup.scenario_id)
        out.append(tg.BoundaryPosition(index=i, gs=sim.gs, scenario_id=setup.scenario_id))
    return out


def test_records_carry_the_playout_counts_and_the_cap(policy, positions):
    cfg = tg.GapConfig(k_alternatives=1, playouts=2, temperature=1.0, cap_turns=2, seed=1)
    records = tg.measure_positions(positions, cfg, policy=policy)
    assert len(records) == 2
    for rec, pos in zip(records, positions):
        assert rec["index"] == pos.index
        assert rec["side"] == pos.gs.global_info.current_side
        assert rec["turn_number"] == pos.gs.global_info.turn_number
        assert rec["n_alternatives"] + len(rec["dropped"]) == 1
        candidates = [rec["base"]] + rec["alternatives"]
        for cand in candidates:
            assert len(cand["outcomes"]) == 2
            assert len(cand["capped"]) == len(cand["turns"]) == len(cand["seeds"]) == 2
            assert set(cand["outcomes"]) <= {-1, 0, 1}
            assert cand["mean"] == pytest.approx(sum(cand["outcomes"]) / 2)
            assert cand["n_capped"] == sum(cand["capped"])
            last_turn = rec["turn_number"] + cfg.cap_turns + 1
            for outcome, capped, turn in zip(cand["outcomes"], cand["capped"], cand["turns"]):
                assert turn <= last_turn
                if capped:
                    assert outcome == 0 and turn == last_turn
        expected_gap = ((max(a["mean"] for a in rec["alternatives"]) - rec["base"]["mean"])
                        if rec["alternatives"] else 0.0)
        assert rec["gap"] == pytest.approx(expected_gap)
        assert rec["base_is_best"] == (rec["gap"] <= 0.0)
    seeds = [s for r in records for c in [r["base"]] + r["alternatives"] for s in c["seeds"]]
    assert len(seeds) == len(set(seeds)), "playout salts must be distinct"

    summary = tg.summarize(records, threshold=0.25, null_permutations=20)
    assert summary["n_positions"] == 2
    assert summary["n_gap_ge_threshold"] == sum(r["gap"] >= 0.25 for r in records)
    assert summary["playouts_total"] == 2 * sum(1 + r["n_alternatives"] for r in records)
    assert summary["n_base_best"] == sum(r["base_is_best"] for r in records)


def _candidate(outcomes, n_decisions=5):
    return {"outcomes": list(outcomes), "capped": [o == 0 for o in outcomes],
            "turns": [10] * len(outcomes), "seeds": [], "n_decisions": n_decisions,
            "sample_seed": None, "post_state_key": 0, "terminal_in_turn": False}


def test_a_terminal_candidate_turn_plays_no_playout(policy, positions):
    """When the candidate turn itself ends the game, its result is
    repeated P times (the gap arithmetic reads P entries) but no
    playout runs: the summary counts only playouts played, in the
    total and in the capped fraction."""
    pos = positions[0]
    mover = pos.gs.global_info.current_side
    sim = tg.sim_from_state(pos.gs, pos.scenario_id, 5, "terminal")
    sim.done, sim.winner, sim.ended_by = True, mover, "leader_killed"
    cfg = tg.GapConfig(k_alternatives=1, playouts=3, cap_turns=2, seed=1)
    terminal = {"n_decisions": 1, "sample_seed": None, "post_state_key": 0,
                "terminal_in_turn": True}
    tg._run_playouts(terminal, sim, pos, mover, 5, cfg, 0, policy, "g")
    assert terminal["outcomes"] == [1, 1, 1] and terminal["capped"] == [False] * 3
    assert terminal["playouts_run"] == 0 and len(terminal["seeds"]) == 3
    record = tg.finish_record(0, {}, terminal, [_candidate([0, 0, 0])], [], cfg, 1.0)
    assert record["base"]["mean"] == 1.0 and record["gap"] == -1.0
    s = tg.summarize([record], threshold=0.25, null_permutations=5)
    assert s["playouts_total"] == 3 and s["playouts_capped"] == 3
    assert s["playouts_capped_frac"] == pytest.approx(1.0)
    assert s["n_terminal_in_turn"] == 1
    # A record from before the field: the flag says the same.
    del record["base"]["playouts_run"]
    assert tg.summarize([record], null_permutations=5)["playouts_total"] == 3


def test_summary_arithmetic_on_synthetic_records():
    cfg = tg.GapConfig(k_alternatives=2, playouts=4)
    records = [
        # Base wins everything; one alternative ties, one loses: gap 0.
        tg.finish_record(0, {}, _candidate([1, 1, 1, 1]),
                         [_candidate([1, 1, 1, 1]), _candidate([-1, -1, -1, -1])], [], cfg, 1.0),
        # Base -1, alternative 0: gap +1.
        tg.finish_record(1, {}, _candidate([-1, -1, -1, -1]),
                         [_candidate([1, 1, -1, -1], n_decisions=7)], [], cfg, 1.0),
        # No distinct alternative: gap 0, every playout capped.
        tg.finish_record(2, {}, _candidate([0, 0, 0, 0]), [], [{"identical_to": "base"}],
                         cfg, 1.0),
        # Equal means (gap 0) but the halves disagree: split gap +2.
        tg.finish_record(3, {}, _candidate([1, -1, 1, -1]), [_candidate([-1, 1, -1, 1])],
                         [], cfg, 1.0),
    ]
    assert [r["gap"] for r in records] == [0.0, 1.0, 0.0, 0.0]
    assert [r["base_is_best"] for r in records] == [True, False, True, True]
    assert records[2]["best_alternative_mean"] is None

    s = tg.summarize(records, threshold=0.25, wall_secs=7200.0, dollars_per_hour=0.5,
                     null_permutations=50)
    assert s["n_positions"] == 4
    assert s["n_gap_ge_threshold"] == 1
    assert s["frac_gap_ge_threshold"] == pytest.approx(0.25)
    assert s["frac_gap_ge_threshold_se"] == pytest.approx(math.sqrt(0.25 * 0.75 / 4))
    assert s["mean_gap"] == pytest.approx(0.25)
    assert s["mean_gap_se"] == pytest.approx(0.5 / math.sqrt(4))
    assert s["n_base_best"] == 3 and s["frac_base_best"] == pytest.approx(0.75)
    assert s["n_no_alternative"] == 1
    assert s["alternatives_distinct_mean"] == pytest.approx((2 + 1 + 0 + 1) / 4)
    assert s["playouts_total"] == 12 + 8 + 4 + 8
    assert s["playouts_capped"] == 4
    assert s["playouts_capped_frac"] == pytest.approx(4 / 32)
    assert s["decisions_per_turn_base"] == pytest.approx(5.0)
    assert s["decisions_per_turn_alternatives"] == pytest.approx((5 + 5 + 7 + 5) / 4)
    assert s["dollars"] == pytest.approx(1.0)
    # Split halves: position 0 -> 0, position 1 -> +1, position 2 -> none, 3 -> +2.
    assert s["n_split"] == 3
    assert s["mean_gap_split"] == pytest.approx(1.0)
    assert s["frac_gap_split_ge_threshold"] == pytest.approx(2 / 3)
    assert sum(b["count"] for b in s["histogram"]) == 4
    by_lo = {b["lo"]: b["count"] for b in s["histogram"]}
    assert by_lo[0.0] == 3 and by_lo[1.0] == 1
    assert 0.0 <= s["null_frac_gap_ge_threshold"] <= 1.0
    md = tg.markdown_summary(s)
    assert "1/4 = 0.250" in md and "$1.00" in md


def test_resummarize_reads_a_partial_file(tmp_path):
    cfg = tg.GapConfig(k_alternatives=1, playouts=4)
    records = [tg.finish_record(0, {}, _candidate([-1, -1, -1, -1]),
                                [_candidate([1, 1, 1, 1])], [], cfg, 1.0),
               tg.finish_record(1, {}, _candidate([1, 1, 1, 1]), [], [], cfg, 1.0)]
    path = tmp_path / "run.partial.json"
    tg.write_json(path, {"summary": {"wall_secs": 3600.0}, "positions": records})
    assert not path.with_suffix(".json.tmp").exists()
    summary, report = tg.resummarize(path, 0.25, 0.4)
    assert summary["n_positions"] == 2 and summary["n_gap_ge_threshold"] == 1
    assert summary["dollars"] == pytest.approx(0.4)
    assert "1/2 = 0.500" in report


def test_permutation_null_is_zero_only_for_constant_outcomes():
    cfg = tg.GapConfig(k_alternatives=1, playouts=4)
    constant = tg.finish_record(0, {}, _candidate([1, 1, 1, 1]), [_candidate([1, 1, 1, 1])],
                                [], cfg, 1.0)
    mixed = tg.finish_record(1, {}, _candidate([1, 1, 1, 1]), [_candidate([-1, -1, -1, -1])],
                             [], cfg, 1.0)
    assert tg.null_gap_fraction([constant], 0.25, 100) == 0.0
    assert tg.null_gap_fraction([mixed], 0.25, 200) > 0.0


def test_same_seeds_reproduce_every_playout(policy, positions):
    cfg = tg.GapConfig(k_alternatives=1, playouts=2, temperature=1.0, cap_turns=1, seed=7)

    def trace(rec):
        played = [(c["n_decisions"], c["outcomes"], c["capped"], c["turns"], c["seeds"])
                  for c in [rec["base"]] + rec["alternatives"]]
        dropped = [(d["sample_seed"], d["n_decisions"], d["identical_to"])
                   for d in rec["dropped"]]
        return played, dropped, [a["sample_seed"] for a in rec["alternatives"]]

    first = tg.measure_position(policy, positions[0], cfg)
    second = tg.measure_position(policy, positions[0], cfg)
    assert first["turn_salt"] == second["turn_salt"]
    assert trace(first) == trace(second)
    other = tg.measure_position(policy, positions[0],
                                tg.GapConfig(k_alternatives=1, playouts=2, temperature=1.0,
                                             cap_turns=1, seed=8))
    assert other["base"]["seeds"] != first["base"]["seeds"]


def test_alternatives_identical_to_the_base_are_dropped(policy, positions):
    # Temperature 0 makes every alternative the argmax turn again.
    cfg = tg.GapConfig(k_alternatives=2, playouts=1, temperature=0.0, cap_turns=1, seed=1)
    rec = tg.measure_position(policy, positions[1], cfg)
    assert rec["n_alternatives"] == 0
    assert [d["identical_to"] for d in rec["dropped"]] == ["base", "base"]
    assert rec["gap"] == 0.0 and rec["base_is_best"]
    assert rec["best_alternative_mean"] is None
    assert len(rec["base"]["outcomes"]) == 1


def test_positions_survive_the_process_boundary(positions):
    # --jobs > 1 ships positions to spawned workers by pickling.
    back = pickle.loads(pickle.dumps(positions[0]))
    assert state_key(back.gs) == state_key(positions[0].gs)
    assert back.scenario_id == positions[0].scenario_id
    assert pickle.loads(pickle.dumps(tg.GapConfig())) == tg.GapConfig()


def test_playout_offset_uses_fresh_salts():
    """A confirmation run's playouts must not reuse the salts of the run
    it confirms: with playout_offset=40 the recorded salts are those of
    playouts 40.. (the candidate turns themselves are unchanged)."""
    from tools.turn_gap import GapConfig, playout_salt
    cfg = GapConfig(k_alternatives=0, playouts=2, cap_turns=1, seed=5, playout_offset=40)
    assert [playout_salt(cfg.seed, 7, 0, r) for r in range(cfg.playout_offset,
                                                             cfg.playout_offset + cfg.playouts)] \
        == [playout_salt(5, 7, 0, 40), playout_salt(5, 7, 0, 41)]
    assert playout_salt(5, 7, 0, 40) != playout_salt(5, 7, 0, 0)
    import pytest
    with pytest.raises(ValueError):
        GapConfig(playout_offset=-1)


def test_candidate_turns_record_actions_and_pre_graders(policy, positions):
    """Every candidate turn records its action list (end_turn included)
    so a confirmation can replay it, plus the forward-only pre-graders."""
    from tools.turn_gap import GapConfig, measure_positions
    cfg = GapConfig(k_alternatives=1, playouts=1, cap_turns=1, seed=3)
    records = measure_positions(positions, cfg, policy=policy, jobs=1)
    assert records
    for r in records:
        for cand in [r["base"]] + r["alternatives"]:
            assert len(cand["actions"]) == cand["n_decisions"] + (0 if cand["terminal_in_turn"] else 1)
            assert cand["terminal_in_turn"] or cand["actions"][-1]["type"] == "end_turn"
            assert isinstance(cand["hp_margin_post"], int)
            assert cand["value_post"] is None or -1.0 <= cand["value_post"] <= 1.0
