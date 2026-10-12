"""The rollout re-grade (tools/rollout_regrade.py) and its readout
(tools/rollout_readout.py): the world the rollouts play holds only what the
mover sees, the reads land on the player turn starts and are signed to the
mover, a rollout that ended takes the result, the readout's continue split
and reading act as pre-registered, and the readout's estimator reproduces
step 1's recorded figure for the static HP margin."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import critic_bench as cb  # noqa: E402
from tools import rollout_readout as ro  # noqa: E402
from tools import rollout_regrade as rr  # noqa: E402
from wesnoth_ai import game_core as gc  # noqa: E402
from wesnoth_ai.classes import Position  # noqa: E402
from wesnoth_ai.critic_data import player_hp_margin  # noqa: E402

needs_core = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")
FIXTURE = Path(__file__).parent / "fixtures" / "critic_bench"
STEP1 = Path(__file__).parent.parent / "training/metrics/critic_step1_20261009"


def _fixture(index: int) -> dict:
    data = json.loads((FIXTURE / "positions.json").read_text(encoding="utf-8"))
    return next(p for p in data["positions"] if p["index"] == index)


# ---------------------------------------------------------------------
# The world and the reads
# ---------------------------------------------------------------------

@needs_core
def test_the_rollout_world_holds_no_unit_the_mover_cannot_see():
    """Caves of the Basilisk, turn 3 (benchmark position 2): after side 2's
    turn, side 1's leader and two more units are in side 2's fog. The
    rollouts' world has none of them, every other side's unit in it is one
    side 2 sees, and the state the benchmark rebuilt still holds them."""
    record = _fixture(2)
    step1 = rr.load_step1(STEP1 / "rebuild.jsonl")
    boundary = cb.boundary_of(record, FIXTURE)
    row, _pre, post = cb.rebuild_candidate(boundary, record, 0, cb.candidates(record)[0])
    assert rr.step1_problem(row, step1[(2, 0)]) is None, "the state is step 1's"
    mover = int(record["side"])
    true_core = post.core.core
    seen = set(true_core.visible_ids(mover))
    hidden = {d["id"] for d in true_core.units_export() if int(d["side"]) != mover and d["id"] not in seen}
    assert len(hidden) >= 2
    world = rr.mover_world(post, mover, "t")
    in_world = {d["id"]: d for d in world.sim.core.core.units_export()}
    assert not hidden & set(in_world)
    assert all(d["id"] in seen for d in in_world.values() if int(d["side"]) != mover)
    assert hidden <= {d["id"] for d in post.core.core.units_export()}, "the true state is left whole"
    sides = world.sim.core.core.sides_export()
    assert sides[0][2:4] == sides[mover - 1][2:4], "the opponent's gold and income are the mover's"


class _Scripted:
    """A player that plays `plan(sim)` (end_turn when it returns None),
    over a base whose value head reads `value` for the side to move."""

    consults_legality_mask = False

    def __init__(self, plan=lambda sim: None, value: float = 0.25):
        self.plan = plan
        self.memory_slots = 0
        model = lambda encoded, memory=None: SimpleNamespace(value=torch.tensor([value]))  # noqa: E731
        self._base = SimpleNamespace(_inference_encoder=SimpleNamespace(encode=lambda gs: None),
                                     _inference_model=model)
        self.sim = None

    def select_action(self, gs, game_label, sim):
        return self.plan(sim) or {"type": "end_turn"}

    def drop_pending(self, game_label):
        pass

    def drop_last_pending(self, game_label):
        return True


def _pairs(players):
    from tools.eval_players import _PolicyPair
    return {side: _PolicyPair(policy=p, label=f"s{side}", side=side) for side, p in players.items()}


def _post_sim(defender_hp: int = 0):
    """Side 1 (the mover) has just ended its turn: its Spearman stands next
    to side 2's leader (at `defender_hp` when given)."""
    from helpers.parity_games import record, state_of
    from tools.wesnoth_sim import WesnothSim
    leader2 = ("Lieutenant", 2, 9, 3, True) + (({"hp": defender_hp},) if defender_hp else ())
    data = record([("Lieutenant", 1, 2, 3, True), ("Spearman", 1, 8, 3, False), leader2,
                   ("Bowman", 2, 16, 5, False)],
                  special={(2, 3): "Kh", (9, 3): "Kh", (16, 6): "Gg^Vh"}, fog=True)
    sim = WesnothSim(state_of(data), scenario_id="", apply_scenario_events=False, max_turns=20)
    sim._seed_salt = "regrade-test"
    sim.step({"type": "end_turn"})
    assert sim.current_side == 2 and not sim.done
    return sim


@needs_core
def test_the_reads_are_taken_at_the_player_turn_starts_and_signed_to_the_mover():
    """Both sides end every turn: read 1 is the mover's next turn start
    (the value head read as it is), reads 2 and 4 the opponent's (negated),
    and the margins are the mover's over the two player sides there."""
    sim = _post_sim()
    world = rr.mover_world(sim, 1, "t")
    pairs = _pairs({1: _Scripted(), 2: _Scripted()})
    seen = []
    reader = rr.value_reader(pairs, 1)

    def value_of(s):
        seen.append((int(s.turn_number), int(s.current_side), player_hp_margin(s.gs, 1)))
        return reader(s)
    start = int(world.sim.turn_number)
    out = rr.play_rollout(world.sim, pairs, 1, (1, 2, 4), value_of, "g")
    assert [(t, side) for t, side, _ in seen] == [(start + 1, 1), (start + 1, 2), (start + 2, 2)]
    assert [out["reads"][k]["value"] for k in ("1", "2", "4")] == [0.25, -0.25, -0.25]
    assert [out["reads"][k]["margin"] for k in ("1", "2", "4")] == [m for _, _, m in seen]
    assert not any(r["ended"] for r in out["reads"].values())
    assert out["decisions"] == 4 and out["player_turns"] == 4 and out["result"] is None
    assert (int(world.sim.turn_number), int(world.sim.current_side)) == (start + 2, 2), "it stops at read 4"


@needs_core
def test_a_rollout_that_ended_takes_the_result_at_its_later_reads():
    """The mover's Spearman strikes the opponent's 1-HP leader at its first
    turn: where a strike lands, the game is won before read 2, and reads 2
    and 4 hold the result for the mover and the final position's margin."""
    def striker():
        struck = []

        def plan(s):
            if s.current_side == 1 and not struck:
                struck.append(True)
                return {"type": "attack", "start_hex": Position(8, 3), "target_hex": Position(9, 3),
                        "attack_index": 0}
            return None
        return plan
    won = 0
    for r in range(8):
        world = rr.mover_world(_post_sim(defender_hp=1), 1, f"t{r}")
        out = rr.play_rollout(world.sim, _pairs({1: _Scripted(striker()), 2: _Scripted()}), 1, (2, 4),
                              lambda s: 0.5, "g")
        if world.sim.done:
            won += 1
            assert out["result"] == 1.0 and out["ended_by"] == "leader_killed"
            final = player_hp_margin(world.sim.gs, 1)
            assert out["reads"] == {k: {"margin": final, "value": 1.0, "ended": True} for k in ("2", "4")}
        else:
            assert all(not v["ended"] and v["value"] == 0.5 for v in out["reads"].values())
    assert 0 < won < 8, "the dice differ from salt to salt"


def test_the_finished_games_result_is_signed_to_the_mover():
    assert [rr.result_for(SimpleNamespace(winner=w), 2) for w in (2, 1, 0)] == [1.0, -1.0, 0.0]


# ---------------------------------------------------------------------
# The readout
# ---------------------------------------------------------------------

def test_the_continue_split_counts_the_picks_and_drops_their_positions():
    """Three positions of three candidates (slot 2 the continue edit): the
    grader picks the continue edit at the first two and a sampled turn at
    the third, so its gain without the continue picks is the third's."""
    positions = np.repeat([0, 1, 2], 3)
    slots = np.tile([0, 1, 2], 3)
    is_continue = slots == 2
    truth = np.array([0.0, 0.1, 0.5, 0.2, 0.0, 0.6, -0.1, 0.3, 0.0])
    grade = np.array([0, 1, 2, 0, 1, 2, 0, 2, 1], dtype=float)
    out = ro.tempo_split({"g": grade}, truth, positions, slots, is_continue, resamples=50)["g"]
    assert (out["positions"], out["continue_picks"]) == (3, 2)
    assert math.isclose(out["continue_share"], 2 / 3)
    assert out["gain_without_continue"]["positions"] == 1
    assert math.isclose(out["gain_without_continue"]["mean"], 0.4)


def _adjusted(gain, se, without, without_se):
    return {"gains": {"positions": 198, "graders": {"margin_r2": {"mean": gain, "se": se}}},
            "tempo": {"margin_r2": {"positions": 198, "continue_picks": 60,
                                    "gain_without_continue": {"mean": without, "se": without_se,
                                                              "positions": 138}}}}


def test_the_reading_fires_as_pre_registered():
    assert ro.reading(_adjusted(0.060, 0.017, 0.050, 0.020))["reading"] == "Pass"
    assert ro.reading(_adjusted(0.060, 0.017, 0.030, 0.020))["reading"] == "Pass, tempo only"
    assert ro.reading(_adjusted(0.034, 0.017, 0.100, 0.020))["reading"] == "Fail", "two SE exactly is not above"
    assert ro.reading(_adjusted(-0.010, 0.017, 0.050, 0.020))["reading"] == "Fail"
    assert "margin_r2" in ro.reading(_adjusted(0.060, 0.017, 0.050, 0.020))["why"]


def test_fed_the_static_margin_the_readout_reproduces_step_1s_figures():
    """Step 1's static HP margin through the readout's statistics gives the
    selection gain and corrected correlation step 1 recorded for it on the
    luck-adjusted truth right after the turn."""
    recorded = json.loads((STEP1 / "readout.json").read_text(encoding="utf-8"))
    want = recorded["statistics"]["adjusted"]["read0"]
    arrays = cb.load_arrays()
    static = ro.static_grade(arrays, rr.load_step1(STEP1 / "rebuild.jsonl"))
    assert np.isfinite(static).all()
    got = ro.readout_statistics(arrays, {ro.STATIC: static}, np.zeros(len(static), dtype=bool))["adjusted"]
    for key in ("mean", "se"):
        assert math.isclose(got["gains"]["graders"][ro.STATIC][key],
                            want["gains"]["graders"][ro.STATIC][key], rel_tol=1e-12)
    for key in ("corrected", "corrected_se"):
        assert math.isclose(got["correlations"]["graders"][ro.STATIC][key],
                            want["correlations"]["graders"][ro.STATIC][key], rel_tol=1e-12)


def _synthetic_rollouts(path: Path, arrays, static, drop=None) -> None:
    """A record per benchmark candidate, its rollouts reading the static
    margin plus noise; slot 4 the continue edit; `drop` leaves one out."""
    rng = np.random.default_rng(0)
    with path.open("w", encoding="utf-8") as f:
        for k, (i, s) in enumerate(zip(arrays["index"], arrays["slot"])):
            if k == drop:
                continue
            rollouts = [{"r": r, "reads": {read: {"margin": float(static[k] + rng.normal(0, 5)),
                                                  "value": float(np.tanh(static[k] / 100)), "ended": False}
                                           for read in ("2", "4")}} for r in range(rr.ROLLOUTS)]
            f.write(json.dumps({"index": int(i), "slot": int(s), "proposer": "continue" if s == 4 else "sample",
                                "rollouts": rollouts}) + "\n")


def test_the_readout_reads_whole_records_and_refuses_a_hole(tmp_path, capsys):
    arrays = cb.load_arrays()
    static = ro.static_grade(arrays, rr.load_step1(STEP1 / "rebuild.jsonl"))
    _synthetic_rollouts(tmp_path / "whole.jsonl", arrays, static)
    assert ro.main(["--rollouts", str(tmp_path / "whole.jsonl"), "--static", str(STEP1 / "rebuild.jsonl"),
                    "--out", str(tmp_path / "whole")]) == 0
    assert "READING " in capsys.readouterr().out
    payload = json.loads((tmp_path / "whole" / "readout.json").read_text(encoding="utf-8"))
    assert payload["records"]["complete"] == len(arrays["index"])
    assert payload["statistics"]["adjusted"]["tempo"]["margin_r2"]["positions"] == 198
    _synthetic_rollouts(tmp_path / "holed.jsonl", arrays, static, drop=7)
    assert ro.main(["--rollouts", str(tmp_path / "holed.jsonl"), "--static", str(STEP1 / "rebuild.jsonl"),
                    "--out", str(tmp_path / "holed")]) == ro.EXIT_REFUSED
    assert "READING_REFUSED" in capsys.readouterr().out


# ---------------------------------------------------------------------
# The run through the shared server
# ---------------------------------------------------------------------

@needs_core
@pytest.mark.slow
def test_a_run_through_the_shared_server_records_every_candidate_and_resumes(tmp_path):
    """Benchmark position 103 (its base turn ends the game) on one worker
    behind a CPU server, a tiny memory network at 0 slots: every candidate
    recorded with its rollout, the states step 1's; a second run finds
    nothing left to play."""
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(1)
    spec = tmp_path / "memory.pt"
    TransformerPolicy(device=torch.device("cpu"), d_model=32, num_layers=1, num_heads=2, d_ff=64,
                      relevant_set_hexes=True, observation_parity=True, memory_slots=8,
                      relevant_set_version=2).save_checkpoint(spec)
    records = tmp_path / "records.json"
    records.write_text(json.dumps({"positions": [_fixture(103)]}), encoding="utf-8")
    argv = ["--records", str(records), "--dataset", str(FIXTURE), "--checkpoint", str(spec),
            "--step1", str(STEP1 / "rebuild.jsonl"), "--out", str(tmp_path / "out"), "--workers", "1",
            "--device", "cpu", "--rollouts", "1"]
    assert rr.main(argv) == 0
    recs = rr.last_records(rr.read_rollouts(tmp_path / "out" / "rollouts.jsonl"))
    assert sorted(recs) == [(103, 0), (103, 1), (103, 2), (103, 3)]
    assert recs[(103, 0)]["skipped"] == "terminal_in_turn"
    for slot in (1, 2, 3):
        (rollout,) = recs[(103, slot)]["rollouts"]
        assert "error" not in rollout and set(rollout["reads"]) == {"2", "4"}
        value_reads = sum(not r["ended"] for r in rollout["reads"].values())
        assert rollout["decisions"] > 0 and rollout["forwards"] >= rollout["decisions"] + value_reads
    summary = json.loads((tmp_path / "out" / "regrade_summary.json").read_text(encoding="utf-8"))
    assert summary["candidates"]["complete"] == 3 and summary["inference_server"]["requests"] > 0
    assert rr.main(argv) == 0
    assert len(rr.read_rollouts(tmp_path / "out" / "rollouts.jsonl")) == 4, "the resumed run played nothing"
