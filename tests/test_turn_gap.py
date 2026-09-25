"""tools/turn_gap.py: the turn-level value gap measurement (phase-2
prerequisite, docs/turn_gap_prereg_20260904.md). CPU, a tiny
random-init policy, fresh mini-map positions instead of the manifest."""
from __future__ import annotations

import json
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


def test_continue_edit_alternatives_extend_the_base_turn(policy, positions):
    """A continue-edit alternative replays the base turn's actions
    (end_turn excluded) and adds up to k more non-end actions."""
    from tools.turn_gap import GapConfig, measure_positions
    cfg = GapConfig(k_alternatives=0, continue_edits=2, playouts=1, cap_turns=1, seed=3)
    records = measure_positions(positions, cfg, policy=policy, jobs=1)
    assert records
    conts = [a for r in records for a in r["alternatives"] + r["dropped"]
             if a.get("proposer") == "continue"]
    assert conts
    for r in records:
        base_prefix = [a for a in r["base"]["actions"] if a.get("type") != "end_turn"]
        for a in r["alternatives"] + r["dropped"]:
            if a.get("proposer") != "continue":
                continue
            acts = [x for x in a["actions"] if x.get("type") != "end_turn"]
            assert acts[:len(base_prefix)] == base_prefix
            assert 0 <= a["extra_decisions"] <= 2
            assert len(acts) == len(base_prefix) + a["extra_decisions"]


def test_shared_inference_reproduces_the_in_process_measurement(tmp_path, positions):
    """The remote policy base (RemoteEncoder with server-side priors,
    RemoteModel over one CPU server) yields the same candidate turns,
    playout outcomes and pre-grader reads as the same checkpoint
    loaded in-process: on CPU fp32 the seam is exact."""
    from test_eval_inference_server import _tiny_checkpoint
    from tools.eval_inference_server import launch_inference_server
    spec_path = _tiny_checkpoint(tmp_path / "tiny.pt")
    local = tg.load_reference_policy(tg.PolicySpec(spec_path, "cpu", False, False))
    # Both paths must alias a unit type outside the checkpoint's vocab
    # the same way (the server's dict is frozen by construction).
    local._inference_encoder.freeze_vocab()
    server = launch_inference_server(spec_path, tmp_path, "t", device="cpu",
                                     infer_bf16=False, window_ms=1.0, max_batch=2)
    cfg = tg.GapConfig(k_alternatives=1, playouts=2, temperature=1.0, cap_turns=2, seed=1)
    try:
        remote = tg.remote_policy(server.address)
        expected = tg.measure_position(local, positions[0], cfg)
        got = tg.measure_position(remote, positions[0], cfg)
    finally:
        stats = server.shutdown()
    for key in ("base", "alternatives"):
        assert _strip(got[key]) == _strip(expected[key])
    for a, b in zip([expected["base"]] + expected["alternatives"],
                    [got["base"]] + got["alternatives"]):
        if a["value_post"] is not None:
            assert abs(a["value_post"] - b["value_post"]) < 1e-4
        if a["pre_end_turn"] is not None:
            assert abs(a["pre_end_turn"]["value_pre"] - b["pre_end_turn"]["value_pre"]) < 1e-4
    assert stats is not None and stats["requests"] > 0 and stats["connections"] == 1


def _strip(cands):
    """Candidates without the process-local fields (state keys hash
    strings per process) and the value reads (compared with a
    tolerance)."""
    if isinstance(cands, dict):
        cands = [cands]
    out = []
    for c in cands:
        kept = {k: v for k, v in c.items() if k not in ("post_state_key", "value_post")}
        if kept.get("pre_end_turn"):
            kept["pre_end_turn"] = {k: v for k, v in kept["pre_end_turn"].items()
                                    if k != "value_pre"}
        out.append(kept)
    return out


def test_cli_shared_inference_launches_one_server_for_the_workers(tmp_path, positions,
                                                                  monkeypatch):
    """`--shared-inference --jobs 2`: main launches the server, the
    spawned workers reach it, the result file records the shared
    provenance and the server's stats, and the server is shut down."""
    from test_eval_inference_server import _tiny_checkpoint
    spec_path = _tiny_checkpoint(tmp_path / "tiny.pt")
    monkeypatch.setattr(tg, "positions_from_manifest", lambda *a, **k: list(positions))
    out = tmp_path / "gap.json"
    rc = tg.main(["x", "--checkpoint", spec_path, "--device", "cpu", "--jobs", "2",
                  "--shared-inference", "--n-states", "2", "--alternatives", "1",
                  "--playouts", "1", "--cap-turns", "1", "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["provenance"]["shared_inference"] is True
    assert data["provenance"]["policy"]["inference_address"]
    assert data["provenance"]["policy"]["infer_compile"] is False
    stats = data["summary"]["inference_server"]
    assert stats["requests"] > 0 and stats["connections"] == 2
    assert len(data["positions"]) == 2
    assert _server_processes_left() == []


def test_the_decode_reaches_the_base_the_alternatives_and_the_playouts(policy, positions):
    """Under an end_turn offset the base turn is the argmax turn with
    end_turn's logit shifted: at -99 (act while anything else is
    legal) it plays the offset-0 turn's decisions as a prefix and then
    goes on, and every player's label carries the decode."""
    plain = tg.GapConfig(k_alternatives=1, playouts=1, temperature=1.0, cap_turns=1, seed=1)
    acting = tg.GapConfig(k_alternatives=1, playouts=1, temperature=1.0, cap_turns=1, seed=1,
                          end_turn_offset=-99.0)
    a = tg.measure_position(policy, positions[0], plain)
    b = tg.measure_position(policy, positions[0], acting)
    a_moves = [x for x in a["base"]["actions"] if x.get("type") != "end_turn"]
    assert b["base"]["actions"][:len(a_moves)] == a_moves
    assert b["base"]["n_decisions"] >= a["base"]["n_decisions"]
    pairs = tg.reference_pairs(policy, acting)
    assert {p.label for p in pairs.values()} == {"raw:t0+eo-99"}
    assert all(p.policy.end_turn_offset == -99.0 for p in pairs.values())
    assert tg.procedure_tag(acting, 1.0) == "raw:t1+eo-99"
    assert tg.procedure_tag(plain, 0.5) == "raw:t0.5"
    with pytest.raises(ValueError, match="end_turn_rule"):
        tg.GapConfig(end_turn_rule="never")


def test_cli_reference_takes_the_config_checkpoint_and_decode(tmp_path, positions, monkeypatch):
    """`--reference` runs the measurement on configs/reference_player.json's
    checkpoint under its decode, and the result file says so; it
    refuses an explicit checkpoint; a confirmation under another
    decode is refused too."""
    from test_eval_inference_server import _tiny_checkpoint
    spec_path = _tiny_checkpoint(tmp_path / "tiny.pt")
    ref = {"label": "tiny", "checkpoint_hf": "tier-b/tiny.pt", "procedure_tag": "raw:t0+eo-1.5",
           "decode": {"mcts_sims": 0, "raw_temperature": 0.0, "raw_end_turn": "joint",
                      "raw_end_turn_offset": -1.5}}
    monkeypatch.setattr(tg.reference_player, "load", lambda: ref)
    monkeypatch.setattr(tg.reference_player, "ensure_checkpoint", lambda r: Path(spec_path))
    monkeypatch.setattr(tg, "positions_from_manifest", lambda *a, **k: list(positions))
    out = tmp_path / "gap.json"
    rc = tg.main(["x", "--reference", "--device", "cpu", "--n-states", "1",
                  "--alternatives", "1", "--playouts", "1", "--cap-turns", "1",
                  "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["config"]["end_turn_offset"] == -1.5
    assert data["provenance"]["reference_procedure"] == "raw:t0+eo-1.5"
    assert data["provenance"]["alternative_procedure"] == "raw:t1+eo-1.5"
    assert data["provenance"]["reference"]["label"] == "tiny"
    assert data["provenance"]["policy"]["checkpoint"] == str(spec_path)
    with pytest.raises(SystemExit, match="drop --checkpoint"):
        tg.main(["x", "--reference", "--checkpoint", spec_path, "--device", "cpu"])
    with pytest.raises(SystemExit, match="the screen ran the decode"):
        tg.main(["x", "--checkpoint", spec_path, "--device", "cpu", "--confirm-from", str(out),
                 "--playouts", "1", "--cap-turns", "1"])


def _server_processes_left():
    """Names of surviving child processes of this test (the server must
    have exited with the run)."""
    import psutil
    me = psutil.Process()
    return [c.name() for c in me.children(recursive=True)
            if "eval_inference_server" in " ".join(c.cmdline())]


def test_sequential_grading_reproduces_the_flat_outcomes_prefix(policy, positions):
    """With rules that never fire, a sequential run plays the same
    playouts as the flat run (the salts ignore the schedule)."""
    flat = tg.GapConfig(k_alternatives=2, playouts=4, temperature=1.0, cap_turns=2, seed=1)
    seq = tg.GapConfig(k_alternatives=2, playouts=4, temperature=1.0, cap_turns=2, seed=1,
                       rounds=2, threshold=-10.0, stop_margin=10.0)
    a = tg.measure_position(policy, positions[0], flat)
    b = tg.measure_position(policy, positions[0], seq)
    assert b["screen"]["mode"] == "sequential" and b["screen"]["rounds_played"] == 2
    assert b["screen"]["verdict"] == "exhausted"
    for x, y in zip([a["base"]] + a["alternatives"], [b["base"]] + b["alternatives"]):
        assert x["actions"] == y["actions"]
        assert x["outcomes"] == y["outcomes"] and x["seeds"] == y["seeds"]
    assert all(alt["dropped_at"] is None for alt in b["alternatives"])


def test_sequential_rules_drop_and_stop(monkeypatch):
    """Scripted playouts: a clearly better alternative stops the
    position as a hit at the first decidable round, a clearly worse
    one is dropped, a coin-flip one runs to the cap."""
    scripted = {0: [-1, 1] * 20, 1: [1] * 40, 2: [-1] * 40, 3: [1, -1] * 20}

    def fake_play(cand, sim, position, mover, max_turns, cfg, c, policy, label):
        for key in ("outcomes", "capped", "turns", "seeds"):
            cand.setdefault(key, [])
        cand["outcomes"].append(scripted[c][len(cand["outcomes"])])
        cand["capped"].append(False)
    monkeypatch.setattr(tg, "_play_next", fake_play)
    base, alts = {}, [(0, {}, None), (1, {}, None), (2, {}, None)]
    cfg = tg.GapConfig(k_alternatives=3, playouts=40, rounds=4, threshold=0.25)
    screen = tg._grade(base, None, alts, None, 1, 5, cfg, None, "t")
    assert screen["verdict"] == "hit" and screen["rounds_played"] == 2   # the base's own noise
    assert len(alts[2][1]["outcomes"]) == 8
    assert alts[0][1]["dropped_at"] is None            # the winner
    assert alts[1][1]["dropped_at"] == 4               # always worse: gone after round 1
    assert alts[2][1]["dropped_at"] is None            # undecided when the hit stopped it
    # Without the winner the coin flip runs to the cap: exhausted.
    base, alts = {}, [(2, {}, None)]
    screen = tg._grade(base, None, alts, None, 1, 5, cfg, None, "t")
    assert screen["verdict"] == "exhausted" and len(base["outcomes"]) == 40
    assert screen["rounds_played"] == 10


def test_confirmation_replays_the_screened_turns(policy, positions):
    """A confirmation run replays the screen's base and best
    alternative from their recorded actions under the screen's turn
    salt: same turns, same post-turn states; the same seed and offset
    reproduce the screen's playouts, a fresh offset plays new ones."""
    cfg = tg.GapConfig(k_alternatives=2, playouts=2, temperature=1.0, cap_turns=2, seed=1)
    screen = tg.measure_position(policy, positions[0], cfg)
    assert screen["alternatives"], "the fixture position needs a distinct alternative"
    same = tg.measure_position(policy, positions[0], cfg, replay=screen, replay_top=1)
    best = max(screen["alternatives"], key=lambda a: a["mean"])
    assert same["replayed"] is True and same["turn_salt"] == screen["turn_salt"]
    assert same["base"]["actions"] == screen["base"]["actions"]
    assert same["base"]["post_state_key"] == screen["base"]["post_state_key"]
    assert [a["actions"] for a in same["alternatives"]] == [best["actions"]]
    assert same["alternatives"][0]["post_state_key"] == best["post_state_key"]
    best_j = max(range(len(screen["alternatives"])),
                 key=lambda j: screen["alternatives"][j]["mean"])
    assert same["alternatives"][0]["source"] == f"screen_alt{best_j}"
    assert same["n_alternatives_sampled"] == 1
    assert same["base"]["outcomes"] == screen["base"]["outcomes"]
    fresh = tg.measure_position(policy, positions[0],
                                tg.GapConfig(k_alternatives=2, playouts=2, temperature=1.0,
                                             cap_turns=2, seed=1, playout_offset=2),
                                replay=screen, replay_top=1)
    assert fresh["base"]["seeds"] != screen["base"]["seeds"]
    assert fresh["base"]["actions"] == screen["base"]["actions"]


def test_candidates_record_the_position_before_their_end_turn(policy, positions):
    """Every candidate that ends its turn records the commands, recruit
    rejections and digest of the position before its end_turn, which
    rebuild that position from the boundary; a confirmation's replay
    reaches the same digest, and a replay or a rebuild that does not is
    refused."""
    import copy
    from tools.game_record import RecordMismatch
    from tools.turn_value import apply_snapshot
    cfg = tg.GapConfig(k_alternatives=2, continue_edits=1, playouts=1, temperature=1.0,
                       cap_turns=1, seed=1)
    screen = tg.measure_position(policy, positions[0], cfg)
    mover = positions[0].gs.global_info.current_side
    rebuilt = 0
    for cand in [screen["base"]] + screen["alternatives"]:
        snap = cand["pre_end_turn"]
        assert (snap is None) == cand["terminal_in_turn"]
        if snap is None:
            continue
        gs = copy.deepcopy(positions[0].gs)
        apply_snapshot(gs, snap, "candidate")         # raises unless the digest is met
        assert gs.global_info.current_side == mover
        assert -1.0 <= snap["value_pre"] <= 1.0
        rebuilt += 1
    assert rebuilt >= 2
    replayed = tg.measure_position(policy, positions[0], cfg, replay=screen, replay_top=1)
    assert replayed["base"]["pre_end_turn"]["digest"] == screen["base"]["pre_end_turn"]["digest"]
    other = json.loads(json.dumps(screen))
    other["base"]["pre_end_turn"]["rejections"] = [[0, 0, 0]]
    with pytest.raises(RuntimeError, match="pre-end_turn"):
        tg.measure_position(policy, positions[0], cfg, replay=other)
    wrong = dict(screen["base"]["pre_end_turn"], digest="0" * 16)
    with pytest.raises(RecordMismatch):
        apply_snapshot(copy.deepcopy(positions[0].gs), wrong, "candidate")


def test_playouts_record_their_side_readings(policy, positions):
    """With horizon reads and luck on, every playout of every candidate
    carries its readings, the first horizon read being the post-turn
    position the value_post and hp_margin_post pre-graders read."""
    cfg = tg.GapConfig(k_alternatives=1, playouts=2, temperature=1.0, cap_turns=2, seed=1,
                       horizon_reads=3, playout_luck=True)
    rec = tg.measure_position(policy, positions[0], cfg)
    read_count = 0
    for cand in [rec["base"]] + rec["alternatives"]:
        assert len(cand["reads"]) == len(cand["outcomes"])
        for read in cand["reads"]:
            if cand["terminal_in_turn"]:
                assert read is None
                continue
            assert 1 <= len(read["horizon"]) <= 3
            value, margin = read["horizon"][0]
            assert value == pytest.approx(cand["value_post"], abs=1e-6)
            assert margin == cand["hp_margin_post"]
            assert read["luck"]["attacks"] >= read["luck"]["skipped"] >= 0
            read_count += 1
    assert read_count >= 2


def test_resume_measures_only_the_missing_positions(tmp_path, positions, monkeypatch):
    """`--resume` after a cut keeps the positions already in the partial
    file, measures the rest, and refuses a file from another run."""
    from test_eval_inference_server import _tiny_checkpoint
    ckpt = _tiny_checkpoint(tmp_path / "tiny.pt")
    monkeypatch.setattr(tg, "positions_from_manifest", lambda *a, **k: list(positions))
    out = tmp_path / "gap.json"
    args = ["x", "--checkpoint", ckpt, "--device", "cpu", "--n-states", "2",
            "--alternatives", "1", "--playouts", "1", "--cap-turns", "1", "--out", str(out)]
    assert tg.main(args) == 0
    full = json.loads(out.read_text(encoding="utf-8"))
    out.unlink()
    out.with_suffix(".partial.json").write_text(
        json.dumps(dict(full, positions=full["positions"][:1])), encoding="utf-8")
    measured = []
    real = tg.measure_position

    def counting(policy, position, *a, **k):
        measured.append(position.index)
        return real(policy, position, *a, **k)
    monkeypatch.setattr(tg, "measure_position", counting)
    assert tg.main(args + ["--resume"]) == 0
    resumed = json.loads(out.read_text(encoding="utf-8"))
    assert measured == [positions[1].index]
    assert [r["index"] for r in resumed["positions"]] == [p.index for p in positions]
    assert resumed["positions"][0] == full["positions"][0]
    assert tg.main(args + ["--resume"]) == 0 and measured == [positions[1].index]
    with pytest.raises(SystemExit):
        tg.main([a if a != "1" else "2" for a in args] + ["--resume"])


def test_cli_confirm_from_selects_the_large_gap_positions(tmp_path, policy, positions,
                                                          monkeypatch):
    """`--confirm-from FILE` without --positions takes the file's
    positions at gap >= threshold and records the provenance."""
    monkeypatch.setattr(tg, "positions_from_manifest", lambda *a, **k: list(positions))
    monkeypatch.setattr(tg, "load_reference_policy", lambda spec: policy)
    monkeypatch.setattr(tg, "_resolve_inference", lambda args: tg.PolicySpec(
        "x", "cpu", False, False))
    screen = tmp_path / "screen.json"
    rc = tg.main(["x", "--checkpoint", "x", "--n-states", "2", "--alternatives", "2",
                  "--playouts", "2", "--cap-turns", "2", "--out", str(screen)])
    assert rc == 0
    data = json.loads(screen.read_text(encoding="utf-8"))
    data["positions"][0]["gap"] = 0.5      # force one large gap, one small
    data["positions"][1]["gap"] = 0.0
    screen.write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "confirm.json"
    rc = tg.main(["x", "--checkpoint", "x", "--confirm-from", str(screen),
                  "--playouts", "2", "--cap-turns", "2", "--playout-offset", "2",
                  "--out", str(out)])
    assert rc == 0
    conf = json.loads(out.read_text(encoding="utf-8"))
    assert [r["index"] for r in conf["positions"]] == [0]
    assert conf["provenance"]["confirm_from"] == str(screen)
    assert conf["positions"][0]["replayed"] is True
    assert conf["positions"][0]["base"]["actions"] == data["positions"][0]["base"]["actions"]


def test_pooled_z_reads_a_base_blunder():
    """Every alternative beating the base gives a large pooled z; equal
    candidates a small one; a constant position none."""
    def rec(base, alts):
        return {"base": {"outcomes": base},
                "alternatives": [{"outcomes": a} for a in alts]}
    blunder = rec([-1] * 8 + [1] * 2, [[1] * 9 + [-1], [1] * 8 + [-1] * 2])
    equal = rec([1, -1] * 5, [[1, -1] * 5, [-1, 1] * 5])
    assert tg.pooled_z(blunder) > 4.0
    assert abs(tg.pooled_z(equal)) < 1.0
    assert tg.pooled_z(rec([1] * 4, [[1] * 4])) is None
    assert tg.pooled_z(rec([1, -1], [])) is None


def test_sequential_grading_without_alternatives_still_plays_the_base(policy, positions):
    """K = 0 under a sequential schedule: the base plays one round and
    the record carries its mean (the run must not crash on a
    position with no distinct alternative)."""
    cfg = tg.GapConfig(k_alternatives=0, playouts=4, cap_turns=2, seed=1, rounds=2)
    rec = tg.measure_position(policy, positions[0], cfg)
    assert rec["screen"]["verdict"] == "no_alternative"
    assert len(rec["base"]["outcomes"]) == 2 and "mean" in rec["base"]
    assert rec["n_alternatives"] == 0 and rec["n_alternatives_sampled"] == 0


def test_confirmation_offsets_past_the_screen_under_the_same_seed():
    """The CLI default offset of 0 becomes the first index past the
    screen's playouts; an explicit offset inside the screen's range is
    refused; another seed keeps its own offset."""
    screen_cfg = {"seed": 1, "playout_offset": 0, "playouts": 40}
    cfg = tg.GapConfig(seed=1, playouts=160)
    assert tg._fresh_playouts(cfg, screen_cfg, 0).playout_offset == 40
    with pytest.raises(SystemExit):
        tg._fresh_playouts(tg.GapConfig(seed=1, playouts=160, playout_offset=20),
                           screen_cfg, 20)
    assert tg._fresh_playouts(tg.GapConfig(seed=1, playouts=160, playout_offset=40),
                              screen_cfg, 40).playout_offset == 40
    assert tg._fresh_playouts(tg.GapConfig(seed=2, playouts=160), screen_cfg, 0).playout_offset == 0


def test_worker_init_failure_fails_the_task(monkeypatch):
    """A worker that could not load its policy raises on its first
    task instead of the pool respawning it forever."""
    monkeypatch.setattr(tg, "load_reference_policy",
                        lambda spec: (_ for _ in ()).throw(OSError("no checkpoint")))
    monkeypatch.setattr(tg, "_WORKER_INIT_ERROR", None)
    tg._worker_init(tg.PolicySpec("x", "cpu", False, False), "INFO")
    with pytest.raises(RuntimeError, match="no checkpoint"):
        tg._worker_task((None, None, None, 1))


def test_replay_refuses_a_turn_that_does_not_realize(policy, positions):
    """A recorded action the sim refuses, or a base whose replay does
    not reproduce the screen's decisions, aborts the confirmation."""
    cfg = tg.GapConfig(k_alternatives=1, playouts=2, temperature=1.0, cap_turns=2, seed=1)
    screen = tg.measure_position(policy, positions[0], cfg)
    bad = json.loads(json.dumps(screen))
    bad["base"]["actions"] = [{"type": "move", "start_hex": [0, 0], "target_hex": [0, 1]},
                              {"type": "end_turn"}]
    with pytest.raises(RuntimeError):
        tg.measure_position(policy, positions[0], cfg, replay=bad)
    other = json.loads(json.dumps(screen))
    other["base"]["n_decisions"] = screen["base"]["n_decisions"] + 5
    with pytest.raises(RuntimeError, match="n_decisions"):
        tg.measure_position(policy, positions[0], cfg, replay=other)


def test_pregrader_reads_terminal_turns_and_reports_ties():
    """The analysis script: a terminal alternative's value read is its
    outcome; an HP-margin tie is a tie, not a miss; the verdict kills
    only on a strictly lower rank."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "analysis"))
    import turn_gap_pregrader as pg

    def cand(outcomes, value, hp, terminal=False):
        return {"outcomes": outcomes, "capped": [False] * len(outcomes),
                "value_post": value, "hp_margin_post": hp, "terminal_in_turn": terminal}
    records = [{"index": 0, "base": cand([-1, -1, 1, -1], -0.5, 10),
                "alternatives": [cand([1, 1, 1, 1], None, 10, terminal=True)]},
               {"index": 1, "base": cand([1, -1, 1, -1], 0.0, 0),
                "alternatives": [cand([1, 1, -1, 1], 0.4, 5)]}]
    out = pg.analyze(records, 0.25)
    v = out["pregraders"]["value_post"]
    assert v["n_skipped"] == 0 and v["n_confirmed"] == 2
    assert v["n_confirmed_ranked_above"] == 2
    h = out["pregraders"]["hp_margin_post"]
    assert h["n_confirmed_tied"] == 1 and h["n_confirmed_ranked_below"] == 0
    assert pg.verdict(out).startswith("ALIVE") or pg.verdict(out).startswith("inconclusive")
    below = [{"index": 0, "base": cand([-1, -1, 1, -1], 0.5, 10),
              "alternatives": [cand([1, 1, 1, -1], 0.1, 10)]}]
    assert "ranked below" in pg.verdict(pg.analyze(below, 0.25))
