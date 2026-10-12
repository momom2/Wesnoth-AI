"""The belief worlds of the rollout re-grade (wesnoth_ai/belief_worlds.py,
tools/rollout_worlds.py) and their readout (tools/belief_readout.py): a
placed unit never stands on a hex the mover sees, on one a unit holds or on
one it cannot enter; the belief head's mass decides where RB puts a unit and
a unit it gives no hex takes RU's rule; the estimates read the mover's gold
at its turn start, which every candidate of a position shares; the last
seen world puts a sighted unit at its sighting; nothing the mover sees
differs between any world and the game; and the readout pairs its
differences and reads them as pre-registered."""
from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers.parity_games import FACTION_IDS, record, state_of, vocab_of  # noqa: E402
from tools import belief_readout as br  # noqa: E402
from tools import critic_bench as cb  # noqa: E402
from tools import rollout_readout as ro  # noqa: E402
from tools import rollout_regrade as rr  # noqa: E402
from tools.rollout_worlds import ArmInputs, out_files, world_inputs  # noqa: E402
from wesnoth_ai import belief_worlds as bw  # noqa: E402
from wesnoth_ai import game_core as gc  # noqa: E402
from wesnoth_ai import turn_bench_stats as S  # noqa: E402

needs_core = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")
FIXTURE = Path(__file__).parent / "fixtures" / "critic_bench"
STEP1 = Path(__file__).parent.parent / "training/metrics/critic_step1_20261009"
SALTS = [f"bw:{k}" for k in range(30)]
ESTIMATE = (100, 2)


def _fixture(index: int) -> dict:
    data = json.loads((FIXTURE / "positions.json").read_text(encoding="utf-8"))
    return next(p for p in data["positions"] if p["index"] == index)


def _tiny_policy():
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(1)
    return TransformerPolicy(device=torch.device("cpu"), d_model=32, num_layers=1, num_heads=2, d_ff=64,
                             relevant_set_hexes=True, observation_parity=True, memory_slots=8,
                             relevant_set_version=2)


# ---------------------------------------------------------------------
# A board with few hexes to hide on
# ---------------------------------------------------------------------

OPEN = {(12, 1), (13, 5), (19, 0), (17, 3), (16, 2), (16, 4), (15, 5)}   # grass beyond the mover's sight
WATER = {(14, 0), (14, 6), (18, 6)}                                       # deep water: the merman's only
VILLAGE = (19, 0)                                                         # side 2's, in side 1's fog


def _post_sim():
    """Side 1 has just ended its turn; side 2's four units stand beyond its
    sight, where cave wall leaves seven grass hexes (one a village of side
    2's) and three of deep water (which only the Merman Fighter can
    enter)."""
    from tools.wesnoth_sim import WesnothSim
    special = {(2, 3): "Kh", VILLAGE: "Gg^Vh"}
    for x in range(11, 20):
        for y in range(7):
            if (x, y) in WATER:
                special[(x, y)] = "Wo"
            elif (x, y) not in OPEN:
                special[(x, y)] = "Xu"
    units = [("Lieutenant", 1, 2, 3, True), ("Spearman", 1, 4, 3, False), ("Lieutenant", 2, 17, 3, True),
             ("Spearman", 2, 16, 2, False), ("Bowman", 2, 16, 4, False), ("Merman Fighter", 2, 15, 5, False)]
    data = record(units, special=special, fog=True, villages={2: [VILLAGE]}, gold=(100, 75))
    sim = WesnothSim(state_of(data), scenario_id="", apply_scenario_events=False, max_turns=20)
    sim._seed_salt = "belief-worlds-test"
    sim.step({"type": "end_turn"})
    assert sim.current_side == 2 and not sim.done
    return sim


def _side2(world) -> dict:
    return {d["id"]: d for d in world.sim.core.core.units_export() if d["side"] == 2}


def _mover_view(cs, mover: int, vocab: dict) -> bytes:
    """The mover's encoding of the state, as if it were to move."""
    from wesnoth_ai.critic_data import encode_view, pack_raw
    fork = cs.fork()
    g = dict(fork.core.globals_export())
    g["current_side"] = int(mover)
    fork.core.set_globals(g)
    return pack_raw(encode_view(fork, "obs", vocab, FACTION_IDS))


@needs_core
def test_a_placed_unit_never_stands_where_the_mover_sees_a_unit_holds_or_it_cannot_enter():
    """RU, and RB under a belief spread over the whole map (seen and
    impassable hexes included), over 30 salts: every placed unit stands on a
    hex the mover does not see, no two share one, and its movement cost
    there is below UNREACHABLE_COST; the seven grass hexes, three of which
    the four units truly hold, are the land units' only choices."""
    sim = _post_sim()
    core = sim.core.core
    seen = np.asarray(core.seen_export(1)) != 0
    index = sim.core.geometry().pos_index
    unseen_walls = sum(1 for (x, y), j in index.items() if not seen[j] and (x, y) not in OPEN | WATER)
    assert unseen_walls > 5 * len(OPEN), "most hexes the mover does not see are cave wall"
    everywhere = {hex_: 0.5 for hex_ in index}
    placed = set()
    for arm, belief in (("roster_uniform", None), ("roster_belief", everywhere)):
        for salt in SALTS:
            world, placement = bw.build_arm_world(sim, 1, arm, salt, turn_start=ESTIMATE, belief=belief)
            assert (placement.hidden, placement.placed, placement.left_out) == (4, 4, 0)
            units = _side2(world)
            hexes = [(d["x"], d["y"]) for d in units.values()]
            assert len(set(hexes)) == len(hexes) == 4
            for d in units.values():
                j = index[(d["x"], d["y"])]
                assert not seen[j], (arm, salt, d["name"], d["x"], d["y"])
                assert bw.passable_for(core, d)[j], (arm, salt, d["name"], d["x"], d["y"])
                assert (d["x"], d["y"]) in OPEN or d["name"] == "Merman Fighter"
            placed |= set(hexes)
    assert len(placed) >= 8, "the placements vary from salt to salt"


@needs_core
def test_every_world_but_t_shows_the_villages_and_gold_the_mover_estimates():
    """Side 2's village lies in side 1's fog and side 2 holds other gold
    than the estimate: every world but T shows the village unowned, and
    every estimated world gives side 2 the estimate's gold and income."""
    sim = _post_sim()
    true_core = sim.core.core
    assert [tuple(o) for o in true_core.village_owner_export()] == [(*VILLAGE, 2)]
    assert tuple(true_core.sides_export()[1][2:4]) != ESTIMATE
    for arm in bw.ARMS:
        world, _ = bw.build_arm_world(sim, 1, arm, "t", turn_start=ESTIMATE, belief={VILLAGE: 0.5}, sightings=[])
        core = world.sim.core.core
        if arm == "true":
            assert core.village_owner_export() == true_core.village_owner_export()
            assert core.sides_export() == true_core.sides_export()
            continue
        assert core.village_owner_export() == [] and core.sides_export()[1][4] == 0, arm
        if arm in bw.ESTIMATED_ARMS:
            assert tuple(core.sides_export()[1][2:4]) == ESTIMATE, arm


@needs_core
def test_rb_puts_a_unit_where_the_belief_dominates_and_the_others_take_rus_rule():
    """The belief names one hex, deep water: the merman, the only unit that
    can enter it, lands there at every salt, and the three land units fall
    back to the uniform rule (counted); under RU the merman lands there
    rarely."""
    sim = _post_sim()
    belief = {(14, 0): 0.9}
    at_hex = {"roster_belief": 0, "roster_uniform": 0}
    for arm in at_hex:
        for salt in SALTS:
            world, placement = bw.build_arm_world(sim, 1, arm, salt, turn_start=ESTIMATE, belief=belief)
            merman = next(d for d in _side2(world).values() if d["name"] == "Merman Fighter")
            at_hex[arm] += (merman["x"], merman["y"]) == (14, 0)
            assert placement.fallbacks == (3 if arm == "roster_belief" else 0)
    assert at_hex["roster_belief"] == len(SALTS)
    assert at_hex["roster_uniform"] <= len(SALTS) // 2


@needs_core
def test_the_last_seen_world_puts_a_sighted_unit_at_its_sighting_with_its_hit_points():
    """The Bowman was seen at (12, 1) with 20 HP: it stands there with them.
    The merman's sighting falls on the hex the Bowman took, the Spearman's on
    a hex the mover sees (left out); the leader was never sighted (removed:
    its side stays alive); a row of the mover's own unit is ignored."""
    sim = _post_sim()
    rows = [("u5", "Bowman", 20, 33, 12, 1), ("u6", "Merman Fighter", 25, 38, 12, 1),
            ("u4", "Spearman", 30, 36, 5, 3), ("u1", "Lieutenant", 40, 48, 2, 3)]
    ids = {d["name"]: d["id"] for d in _side2(bw.build_arm_world(sim, 1, "true", "t")[0]).values()}
    assert (ids["Bowman"], ids["Merman Fighter"], ids["Spearman"], ids["Lieutenant"]) == ("u5", "u6", "u4", "u3")
    world, placement = bw.build_arm_world(sim, 1, "last_seen", "t", turn_start=ESTIMATE, sightings=rows)
    units = _side2(world)
    assert set(units) == {"u5"}
    assert (units["u5"]["x"], units["u5"]["y"], units["u5"]["current_hp"]) == (12, 1, 20)
    assert (placement.hidden, placement.placed, placement.left_out, placement.removed) == (4, 1, 2, 1)
    assert world.sim.hidden_leader_sides == {2}
    vocab = vocab_of(sim.core.core.type_names())
    assert _mover_view(world.sim.core, 1, vocab) == _mover_view(sim.core, 1, vocab)
    clamped, placement = bw.build_arm_world(sim, 1, "last_seen", "t", turn_start=ESTIMATE,
                                            sightings=[("u3", "Lieutenant", 999, 48, 19, 0)])
    leader = _side2(clamped)["u3"]
    assert (leader["x"], leader["y"], leader["current_hp"]) == (19, 0, leader["max_hp"])
    assert placement.hp_clamped == 1 and clamped.sim.hidden_leader_sides == frozenset()


# ---------------------------------------------------------------------
# A benchmark position: Caves of the Basilisk, turn 3, side 2 to move
# ---------------------------------------------------------------------

def _candidate(slot: int):
    record_ = _fixture(2)
    boundary = cb.boundary_of(record_, FIXTURE)
    row, pre, post = cb.rebuild_candidate(boundary, record_, slot, cb.candidates(record_)[slot])
    assert rr.step1_problem(row, rr.load_step1(STEP1 / "rebuild.jsonl")[(2, slot)]) is None
    return boundary, pre, post


@needs_core
def test_the_boundary_is_the_movers_turn_start_after_its_income():
    """The position's boundary holds the mover's gold after its init_side
    paid its income: the corpus game's gold right after that command, not
    before it."""
    from tools.replay_dataset import record_core
    record_ = _fixture(2)
    meta, mover = record_["meta"], int(record_["side"])
    with gzip.open(FIXTURE / meta["file"], "rt", encoding="utf-8") as f:
        data = json.load(f)
    cs = record_core(data)
    for cmd in data["commands"]:
        if cmd[0] == "init_side" and int(cs.core.turn_number) >= meta["cut_turn"] and cmd[1] == mover:
            break
        cs.apply_command(list(cmd))
    before = cs.core.sides_export()[mover - 1][2]
    cs.apply_command(list(cmd))
    after = cs.core.sides_export()[mover - 1][2]
    assert after > before
    assert cb.boundary_of(record_, FIXTURE).core.sides_export()[mover - 1][2] == after


@needs_core
def test_two_candidates_that_recruit_differently_share_the_turn_start_estimate():
    """Slot 1 recruits a Troll Whelp and slot 2 a Goblin Spearman: the
    observed world's opponent takes each one's gold after it, the estimated
    worlds' the mover's gold at its turn start, the same for both."""
    golds = {"observed": [], "true_units": [], "none": []}
    for slot in (1, 2):
        boundary, pre, post = _candidate(slot)
        for arm in golds:
            inputs = world_inputs(ArmInputs(arm), boundary, pre, 2)
            world, _ = bw.build_arm_world(post, 2, arm, "t", **inputs)
            golds[arm].append(tuple(world.sim.core.core.sides_export()[0][2:4]))
    start = tuple(boundary.core.sides_export()[1][2:4])
    assert golds["true_units"] == golds["none"] == [start, start]
    assert golds["observed"][0] != golds["observed"][1], "the candidates spent differently"


@needs_core
def test_nothing_the_mover_sees_differs_between_any_world_and_the_game():
    """Three of side 1's units, its leader among them, are in side 2's fog:
    in every arm, at four salts each, side 2's encoding of the world is its
    encoding of the state after the candidate; the counts add up, the
    observed arm is the re-grade's world, and only the true and true-units
    worlds keep the hidden units."""
    boundary, pre, post = _candidate(0)
    mover = 2
    vocab = vocab_of(post.core.core.type_names())
    truth = _mover_view(post.core, mover, vocab)
    policy = _tiny_policy()
    for arm in bw.ARMS:
        inputs = world_inputs(ArmInputs(arm, policy), boundary, pre, mover)
        for salt in SALTS[:4]:
            world, p = bw.build_arm_world(post, mover, arm, salt, **inputs)
            assert _mover_view(world.sim.core, mover, vocab) == truth, (arm, salt)
            assert p.hidden == 3 and p.leader_hidden
            assert p.hidden == p.kept + p.placed + p.left_out + p.removed, (arm, p)
            assert (p.kept == 3) == (arm in ("true", "true_units"))
    observed = bw.build_arm_world(post, mover, "observed", "s")[0].sim.core.core
    regrade = rr.mover_world(post, mover, "s").sim.core.core
    assert observed.units_export() == regrade.units_export() and observed.sides_export() == regrade.sides_export()


@needs_core
def test_the_belief_read_names_each_token_by_its_hex():
    """Per hex token, the probability belief_by_hex gives its hex is the
    sigmoid of the belief head's logit at that token, the token's hex being
    the encoder's own (`hex_positions`)."""
    from wesnoth_ai.memory import MemoryState
    _boundary, pre, _post = _candidate(0)
    policy = _tiny_policy()
    belief = bw.belief_by_hex(policy, pre)
    enc = policy._inference_encoder
    raw = enc.raw_of(gc.view_of(pre.fork()))
    with torch.no_grad():
        logits = policy._inference_model(enc.encode_from_raw(raw), memory=MemoryState(0, None)).belief_logits
    want = {(p.x, p.y): 1.0 / (1.0 + math.exp(-float(v))) for p, v in zip(raw.hex_positions, logits.reshape(-1))}
    assert len(want) == raw.hex_xs.shape[0] > 100
    assert set(belief) == set(want)
    assert max(abs(belief[h] - want[h]) for h in want) < 1e-6
    assert len(set(round(p, 6) for p in want.values())) > 10, "the logits differ from token to token"


# ---------------------------------------------------------------------
# The tool's worlds and the readout
# ---------------------------------------------------------------------

def test_each_world_writes_its_own_records():
    out = Path("o")
    assert out_files(out, "observed") == (out / "rollouts.jsonl", out / "regrade_summary.json")
    assert out_files(out, "last_seen") == (out / "rollouts_last_seen.jsonl", out / "regrade_summary_last_seen.json")


def _write_arm(run_dir: Path, arrays, grade: np.ndarray, world: str, drop=None, foreign: bool = False) -> None:
    """A complete record per benchmark candidate whose rollouts all read
    `grade` at every read; `drop` leaves one row out."""
    path = out_files(run_dir, world)[0]
    with path.open("w", encoding="utf-8") as f:
        for k, (i, s) in enumerate(zip(arrays["index"], arrays["slot"])):
            if k == drop:
                continue
            reads = {read: {"margin": float(grade[k]), "value": 0.0, "ended": False} for read in ("2", "4")}
            rollouts = [{"r": r, "reads": reads,
                         "placement": {"hidden": 2, "placed": 2, "leader_hidden": bool(s == 0)}}
                        for r in range(rr.ROLLOUTS)]
            f.write(json.dumps({"index": int(i), "slot": int(s), "world": "none" if foreign else world,
                                "proposer": "sample", "rollouts": rollouts}) + "\n")


def _run_dir(tmp_path: Path, grades: dict, **kw) -> Path:
    arrays = cb.load_arrays()
    for label, world in br.ARMS.items():
        _write_arm(tmp_path, arrays, grades[label], world, **(kw if label == "RB" else {}))
    return tmp_path


def test_the_readout_pairs_the_arms_and_reads_them(tmp_path, capsys):
    """An arm grading by the truth itself against arms grading by the
    static HP margin: TU - O2 is the difference of the two gains over the
    same positions, two arms that grade alike differ by exactly 0, and with
    LS grading by the truth the reading is Last seen suffices."""
    arrays = cb.load_arrays()
    static = ro.static_grade(arrays, rr.load_step1(STEP1 / "rebuild.jsonl"))
    oracle = S.truth_mean(ro.truths(arrays)["adjusted"])
    grades = {"T": oracle, "TU": oracle, "RB": static, "RU": static, "LS": oracle, "O2": static}
    run_dir = _run_dir(tmp_path, grades)
    assert br.main(["--dir", str(run_dir), "--out", str(tmp_path / "out")]) == 0
    assert "READING Last seen suffices" in capsys.readouterr().out
    payload = json.loads((tmp_path / "out" / "belief_readout.json").read_text(encoding="utf-8"))
    s = payload["statistics"]["adjusted"]["margin_r2"]
    assert s["positions"] == 198
    gain = s["graders"]
    assert math.isclose(s["differences"]["TU-O2"]["mean"], gain["TU"]["mean"] - gain["O2"]["mean"], abs_tol=1e-12)
    assert s["differences"]["TU-O2"]["mean"] > 0.05
    assert s["differences"]["T-TU"] == {"mean": 0.0, "se": 0.0}
    assert payload["agreement_with_TU"]["T"]["share"] == 1.0 > payload["agreement_with_TU"]["O2"]["share"]
    assert payload["placement"]["RB"]["placed"] == 2 * rr.ROLLOUTS * len(arrays["index"])
    assert payload["placement"]["RB"]["leader_hidden_positions"] == 198


def test_the_readout_refuses_a_hole_and_another_worlds_records(tmp_path, capsys):
    arrays = cb.load_arrays()
    static = ro.static_grade(arrays, rr.load_step1(STEP1 / "rebuild.jsonl"))
    grades = {label: static for label in br.ARMS}
    for name, kw in (("hole", {"drop": 7}), ("foreign", {"foreign": True})):
        run_dir = tmp_path / name
        run_dir.mkdir()
        _run_dir(run_dir, grades, **kw)
        assert br.main(["--dir", str(run_dir), "--out", str(run_dir / "out")]) == br.EXIT_REFUSED
        said = capsys.readouterr().out
        assert "READING_REFUSED" in said and "RB:" in said


def _diffs(**z) -> dict:
    """Primary-grader statistics whose paired differences sit at z SE."""
    return {"differences": {f"{a}-{b}": {"mean": 0.01 * z.get(f"{a}_{b}", 0.0), "se": 0.01}
                            for a, b in br.PAIRS}}


def test_the_readings_fire_as_pre_registered():
    assert br.readings(_diffs(TU_O2=1.9, LS_O2=5))["readings"] == ["Unresolved ceiling"]
    assert br.readings(_diffs(TU_O2=3, LS_O2=2.5))["readings"] == ["Last seen suffices"]
    assert br.readings(_diffs(TU_O2=3, RB_RU=2.5, RU_O2=3))["readings"] == ["The belief head knows where"]
    assert br.readings(_diffs(TU_O2=3, RB_RU=1.5, RU_O2=3))["readings"] == ["What exists, not where"]
    assert br.readings(_diffs(TU_O2=3, RB_RU=-2.5, RU_O2=3))["readings"] == ["None of the three"]
    assert br.readings(_diffs(TU_O2=3, RB_RU=2.0, RU_O2=2.0))["readings"] == ["None of the three"], \
        "two SE exactly is not above"
    both = br.readings(_diffs(TU_O2=3, LS_O2=3, RB_RU=3))
    assert both["readings"] == ["Last seen suffices", "The belief head knows where"] and len(both["why"]) == 3


# ---------------------------------------------------------------------
# A run through the shared server
# ---------------------------------------------------------------------

@needs_core
@pytest.mark.slow
def test_an_rb_run_behind_the_server_reads_the_belief_on_its_own_copy(tmp_path):
    """Benchmark position 2 on one worker behind a CPU server, a tiny memory
    network: every candidate's record names the RB world, carries the belief
    read of the worker's own copy and its rollout's placement of the three
    hidden units; the run refuses a records file of another world."""
    spec = tmp_path / "memory.pt"
    _tiny_policy().save_checkpoint(spec)
    records = tmp_path / "records.json"
    records.write_text(json.dumps({"positions": [_fixture(2)]}), encoding="utf-8")
    argv = ["--records", str(records), "--dataset", str(FIXTURE), "--checkpoint", str(spec),
            "--step1", str(STEP1 / "rebuild.jsonl"), "--out", str(tmp_path / "out"), "--workers", "1",
            "--device", "cpu", "--rollouts", "1", "--world", "roster_belief"]
    assert rr.main(argv) == 0
    recs = rr.last_records(rr.read_rollouts(tmp_path / "out" / "rollouts_roster_belief.jsonl"))
    assert sorted(recs) == [(2, s) for s in range(5)]
    for rec in recs.values():
        assert rec["world"] == "roster_belief" and rec["turn_start"] == [22, 2]
        assert rec["belief"]["tokens"] > 100
        (rollout,) = rec["rollouts"]
        assert "error" not in rollout and rollout["placement"]["leader_hidden"]
        p = rollout["placement"]
        assert p["placed"] + p["left_out"] == p["hidden"] >= 2
    summary = json.loads((tmp_path / "out" / "regrade_summary_roster_belief.json").read_text(encoding="utf-8"))
    assert summary["world"] == "roster_belief" and summary["placement"]["placed"] >= 10
    (tmp_path / "out" / "rollouts_none.jsonl").write_bytes(
        (tmp_path / "out" / "rollouts_roster_belief.jsonl").read_bytes())
    with pytest.raises(SystemExit, match="another|world"):
        rr.main(argv[:-1] + ["none"])
