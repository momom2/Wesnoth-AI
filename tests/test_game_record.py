"""Game records (tools/game_record.py): a game the simulator played,
recorded and rebuilt from its record, reaches the same position after
every command; a record that does not reproduce its game is refused;
the record carries the outcome data the simulator and the search
computed; a damaged log loses only its damaged records."""
from __future__ import annotations

import copy
import gzip
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

from tools import game_record  # noqa: E402
from wesnoth_ai.game_core import game_core_class, state_differences  # noqa: E402

# What the simulator keeps on the state for its own bookkeeping and the
# applier does not: the seed counter and the last command's side
# channels.
SIM_ONLY = ("_rng_request_counter", "_last_move_walk", "_last_checkup_strikes",
            "_last_advance_events")

# A one-row corridor of three grass hexes inside an impassable border.
CORRIDOR = "Xv, Xv, Xv, Xv, Xv\nXv, Gg, Gg, Gg, Xv\nXv, Xv, Xv, Xv, Xv"


def _differences(a, b):
    return [d for d in state_differences(a, b, stash=False)
            if not any(k in d for k in SIM_ONLY)]


def _played_game(max_turns=8, use_core=False):
    """A mini game fought to its cap by the deterministic Brawler, with
    a recruit rejection on the way, and the position after each step
    keyed by the index of the step's last command (a step that ends a
    turn records end_turn and the next init_side)."""
    import wesnoth_ai.dummy_policy as dummy_policy
    from sim_test_helpers import Brawler, scenario_setup
    from tools.scenario_pool import build_scenario_gamestate
    from tools.wesnoth_sim import WesnothSim
    setup = scenario_setup(3, mini=True)
    sim = WesnothSim(build_scenario_gamestate(setup), scenario_id=setup.scenario_id,
                     max_turns=max_turns, use_core=use_core)
    sim._seed_salt = "record-test"
    sim.enable_uniform_advancement()
    after = {len(sim.command_history) - 1: copy.deepcopy(sim.gs)}
    cap = dummy_policy._BOOTSTRAP_UNITS
    dummy_policy._BOOTSTRAP_UNITS = 8
    try:
        policy = Brawler()
        while not sim.done:
            if len(sim.command_history) == 12:
                sim.reject_recruit_hex(0, 0)
                after[11] = copy.deepcopy(sim.gs)
            sim.step(policy.select_action(sim.gs, game_label="rec"))
            after[len(sim.command_history) - 1] = copy.deepcopy(sim.gs)
    finally:
        dummy_policy._BOOTSTRAP_UNITS = cap
    return sim, setup, after


def _duel_data(attacker_max_exp=None):
    """A Spearman (side 1) beside a Dwarvish Fighter (side 2), which
    answers a melee attack with its axe or its hammer."""
    spearman = {"uid": 1, "type": "Spearman", "side": 1, "x": 0, "y": 0, "is_leader": True}
    if attacker_max_exp is not None:
        spearman["max_exp"] = attacker_max_exp
    return {
        "game_id": "rec", "map_data": CORRIDOR,
        "starting_units": [
            spearman,
            {"uid": 2, "type": "Dwarvish Fighter", "side": 2, "x": 1, "y": 0, "is_leader": True}],
        "starting_sides": [{"side": k, "gold": 0, "recruit": []} for k in (1, 2)]}


def _stored(rec, tmp_path):
    """`rec` written to a log and read back, as a consumer finds it."""
    path = tmp_path / "records.jsonl.gz"
    game_record.GameRecordLog(path).write(rec)
    [back] = list(game_record.read_records(path))
    return back


def _write_corpus_game(path: Path, data: dict) -> None:
    path.write_bytes(gzip.compress(json.dumps(data).encode("utf-8")))


def _corpus_start(tmp_path, data):
    """A mid-game start cut from `data` as a one-game corpus of empty
    turns, and the simulator continuing it, as self-play builds them.
    The cut lands on an init_side(2)."""
    import random
    from tools.midgame_starts import sample_midgame_start
    from tools.wesnoth_sim import WesnothSim
    data = dict(data, commands=[["init_side", s] if i % 2 == 0 else ["end_turn"]
                                for i, s in enumerate([1, 1, 2, 2] * 3)])
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _write_corpus_game(corpus / "g.json.gz", data)
    (corpus / "value_corpus_index.jsonl").write_text('{"file": "g.json.gz"}\n', encoding="utf-8")
    mg = next(m for m in (sample_midgame_start(random.Random(s), corpus) for s in range(20))
              if m is not None)
    gs, scenario_id, _cut, begin_side, _provenance = mg
    sim = WesnothSim(gs, scenario_id=scenario_id, max_turns=gs.global_info.turn_number + 3,
                     apply_scenario_events=False, begin_side=begin_side, use_core=False)
    return sim, ("__midgame__",) + mg, data


def _advancing_duel(tmp_path):
    """The Spearman, one fight from advancing, attacks on its turn: it
    advances by the simulator's salted uniform channel, and both sides
    then end a turn, so the record holds two turn starts after the
    fight. Commands: init_side 2, end_turn, init_side 1, attack,
    end_turn, init_side 2, end_turn, init_side 1."""
    from wesnoth_ai.classes import Position
    sim, setup, _data = _corpus_start(tmp_path, _duel_data(attacker_max_exp=1))
    sim._seed_salt = "record-test"        # its first uniform draw is index 2 of 3
    sim.enable_uniform_advancement()
    sim.step({"type": "end_turn"})
    sim.step({"type": "attack", "start_hex": Position(0, 0), "target_hex": Position(1, 0),
              "attack_index": 0})
    sim.step({"type": "end_turn"})
    sim.step({"type": "end_turn"})
    return sim, setup


def _duel_record(tmp_path):
    sim, setup = _advancing_duel(tmp_path)
    return sim, _stored(game_record.game_record(sim, setup, game_label="duel"), tmp_path)


# ---------------------------------------------------------------------
# Rebuilding
# ---------------------------------------------------------------------
@pytest.mark.parametrize("use_core", [False, pytest.param(True, marks=pytest.mark.skipif(
    game_core_class() is None, reason="wesnoth_core.GameCore not available"))])
def test_a_recorded_game_rebuilds_position_by_position(tmp_path, use_core):
    """Played on either state of record, the game rebuilds on the Python
    applier, its fingerprints checked, to the simulator's final position."""
    sim, setup, _after = _played_game(use_core=use_core)
    game_record.configure(tmp_path, "t")
    try:
        game_record.record_game(sim, setup, game_label="g", build={})
    finally:
        game_record.configure(None, "")
    [rec] = list(game_record.read_records(tmp_path / "t.jsonl.gz"))
    kinds = [c[0] for c in rec["commands"]]
    assert kinds.count("attack") >= 5 and len(kinds) >= 40, kinds
    assert rec["rejections"] == [[12, 0, 0]]
    assert (rec["winner"], rec["ended_by"]) == (sim.winner, sim.ended_by)
    assert len(rec["turn_digests"]) >= 6
    assert _differences(game_record.rebuild(rec), sim.gs) == []


def test_a_game_with_an_acting_neutral_side_rebuilds_from_its_record(tmp_path):
    """Micro Isar's tentacles take a turn after side 2's, pinned at 0
    movement by the scenario, so each of their ends of turn drops their
    resting status; a record whose stream holds that end_turn without
    the simulator having applied it rebuilds to another position and is
    refused at the next turn start."""
    from tools.scenario_pool import ScenarioSetup, build_scenario_gamestate
    from tools.wesnoth_sim import WesnothSim
    setup = ScenarioSetup(scenario_id="enclave_micro_isar",
                          faction1="Knalgan Alliance", leader1="Dwarvish Steelclad",
                          faction2="Rebels", leader2="Elvish Captain")
    sim = WesnothSim(build_scenario_gamestate(setup), scenario_id=setup.scenario_id,
                     max_turns=10, use_core=False)
    for _ in range(6):
        sim.step({"type": "end_turn"})
    rec = _stored(game_record.game_record(sim, setup, game_label="neutral", build={}), tmp_path)
    assert rec["commands"].count(["init_side", 3]) >= 2
    assert _differences(game_record.rebuild(rec), sim.gs) == []


def test_the_walk_passes_through_every_position_the_game_did():
    sim, setup, after = _played_game(max_turns=4)
    rec = game_record.game_record(sim, setup, game_label="g", build={})
    rec = json.loads(json.dumps(rec))                          # through JSON, as stored
    walked = [copy.deepcopy(gs) for _k, gs, _cmd in game_record.walk(rec)]
    assert len(walked) == len(rec["commands"])
    # the state before command k + 1 is the simulator's after command k
    compared = 0
    for k, gs in sorted(after.items()):
        if k + 1 < len(walked):
            assert _differences(walked[k + 1], gs) == [], k
            compared += 1
    assert compared >= 20


def test_the_fingerprints_hold_in_a_process_with_other_string_hashes(tmp_path):
    """Python salts string hashes per process, and records are rebuilt
    elsewhere: a record written here rebuilds, checked, in a process
    whose salt differs."""
    import os
    import subprocess
    sim, setup, _after = _played_game(max_turns=4)
    path = tmp_path / "g.jsonl.gz"
    game_record.GameRecordLog(path).write(game_record.game_record(sim, setup, game_label="g"))
    ours = os.environ.get("PYTHONHASHSEED", "")
    theirs = str(int(ours) + 1) if ours.isdigit() else "1"
    code = ("import sys; sys.path.insert(0, sys.argv[2]); from tools import game_record as g; "
            "[r] = g.read_records(sys.argv[1]); g.rebuild(r); print('rebuilt')")
    run = subprocess.run([sys.executable, "-c", code, str(path), str(ROOT)], cwd=str(ROOT),
                         env={**os.environ, "PYTHONHASHSEED": theirs},
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0 and "rebuilt" in run.stdout, run.stderr[-3000:]


def test_an_advancement_rebuilds_from_the_recorded_channel_and_salt(tmp_path):
    """The Spearman advances to its third option by the salted uniform
    draw; the rebuild must draw the same, which it can only do with the
    record's channel flag and salt (unsalted, or without the channel,
    the choice is the first option)."""
    sim, rec = _duel_record(tmp_path)
    assert [u.name for u in sim.gs.map.units if u.side == 1] == ["Javelineer"]  # advances_to[2]
    rebuilt = game_record.rebuild(rec)
    assert [u.name for u in rebuilt.map.units if u.side == 1] == ["Javelineer"]
    assert _differences(rebuilt, sim.gs) == []


def test_a_stored_attack_keeps_both_counter_weapon_tables(tmp_path):
    """The Dwarvish Fighter could answer the spear with its axe or its
    hammer: the simulator runs both strike tables to choose, and the
    stored record carries them on the attack."""
    _sim, rec = _duel_record(tmp_path)
    [k] = [i for i, c in enumerate(rec["commands"]) if c[0] == "attack"]
    data = rec["outcomes"][str(k)]["counter_weapon"]
    assert set(data["tables"]) == {"0", "1"} and data["chosen"] == rec["commands"][k][6]
    for table in data["tables"].values():
        assert abs(sum(row[-1] for row in table) - 1.0) < 1e-9


def test_a_record_that_does_not_reproduce_its_game_is_refused(tmp_path):
    """Without its salt the rebuilt Spearman advances to its first
    option: the walk stops at the next turn start with RecordMismatch,
    and goes through when told not to verify."""
    _sim, rec = _duel_record(tmp_path)
    rec["seed_salt"] = ""
    reached = []
    with pytest.raises(game_record.RecordMismatch, match="after command 5 \\(init_side\\)"):
        for k, _gs, _cmd in game_record.walk(rec):
            reached.append(k)
    assert reached == [0, 1, 2, 3, 4, 5] and len(rec["commands"]) == 8
    rebuilt = game_record.rebuild(rec, verify=False)
    assert [u.name for u in rebuilt.map.units if u.side == 1] == ["Swordsman"]


def test_a_midgame_record_is_refused_when_its_corpus_file_changed(tmp_path):
    """A mid-game start carries the digest of its corpus file: the same
    file rebuilds the game, a re-extracted one is refused."""
    sim, setup, data = _corpus_start(tmp_path, _duel_data())
    sim.step({"type": "end_turn"})
    rec = game_record.game_record(sim, setup, game_label="mid")
    assert rec["setup"]["midgame"]["sha256"]
    assert _differences(game_record.rebuild(rec), sim.gs) == []
    data["starting_units"][0]["hp"] = 30
    _write_corpus_game(tmp_path / "corpus" / "g.json.gz", data)
    with pytest.raises(game_record.RecordMismatch, match="corpus file"):
        game_record.rebuild(rec)


# ---------------------------------------------------------------------
# The search's outcome data
# ---------------------------------------------------------------------
def test_a_search_policy_attaches_the_exact_distribution_of_the_attack_it_played():
    """A seeded MCTSPolicy with exact outcome enumeration plays the duel
    through the production game loop; every attack it played carries
    the distribution its search computed for it. Four simulations visit
    each of a duel position's at most four root actions, so every played
    attack was searched."""
    import torch
    from tools.mcts import MCTSConfig
    from tools.mcts_policy import MCTSPolicy
    from tools.replay_dataset import _build_initial_gamestate
    from tools.sim_self_play import _recruit_cost_lookup, play_one_game
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(0)
    base = TransformerPolicy(device=torch.device("cpu"), d_model=32, num_layers=1,
                             num_heads=4, d_ff=64)
    policy = MCTSPolicy(base, MCTSConfig(n_simulations=4, gumbel_root=True, gumbel_m=4,
                                         chance_nodes=True, exact_outcome_enumeration=True,
                                         batch_size=1, add_root_noise=False),
                        rng_seed=5)
    sim = WesnothSim(_build_initial_gamestate(_duel_data()), scenario_id="rec", max_turns=4,
                     use_core=False)
    sim._seed_salt = "search-test"
    play_one_game(sim, policy, lambda delta: 0.0, game_label="s",
                  cost_lookup=_recruit_cost_lookup())
    attacks = [rc for rc in sim.command_history if rc.kind == "attack"]
    assert attacks, [rc.kind for rc in sim.command_history]
    for rc in attacks:
        data = rc.extras["outcomes"]["search"]
        attacker = f"u{1 if rc.side == 1 else 2}"
        assert data["attacker"] == attacker
        assert abs(sum(row[-1] for row in data["outcomes"]) - 1.0) < 1e-9


def test_a_refused_step_keeps_the_previous_attacks_search_distribution():
    from tools.combat_outcomes import enumerate_attack_outcomes
    from tools.replay_dataset import _build_initial_gamestate
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.classes import Position
    sim = WesnothSim(_build_initial_gamestate(_duel_data()), scenario_id="rec", max_turns=4,
                     use_core=False)
    spear, javelin = ({"type": "attack", "start_hex": Position(0, 0),
                       "target_hex": Position(1, 0), "attack_index": w} for w in (0, 1))
    reported = [enumerate_attack_outcomes(sim.gs, spear), enumerate_attack_outcomes(sim.gs, javelin)]

    class _Searched:
        def pop_played_outcomes(self, _label):
            return reported.pop(0)

    policy = _Searched()
    before = len(sim.command_history)
    sim.step(spear)
    game_record.note_search_outcomes(sim, policy, "g", before)
    kept = sim.command_history[-1].extras["outcomes"]["search"]
    before = len(sim.command_history)
    sim.step({"type": "move", "start_hex": Position(0, 0), "target_hex": Position(1, 0)})
    assert sim.last_step_rejected and len(sim.command_history) == before
    game_record.note_search_outcomes(sim, policy, "g", before)
    assert sim.command_history[-1].extras["outcomes"]["search"] == kept
    assert not reported


# ---------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------
def test_reading_skips_a_damaged_record_and_goes_on(tmp_path, caplog):
    """A record cut short by a crash, then more records appended after
    it, the last one cut short too: the reader reads the whole ones and
    names the skipped bytes."""
    import logging
    members = [gzip.compress((json.dumps({"n": n, "pad": "x" * 2000 * n}) + "\n").encode())
               for n in range(1, 5)]
    cut = len(members[0]) + len(members[1]) // 2
    path = tmp_path / "log.jsonl.gz"
    path.write_bytes(members[0] + members[1][:len(members[1]) // 2] + members[2]
                     + members[3][:-3])
    with caplog.at_level(logging.WARNING, logger="game_record"):
        assert [r["n"] for r in game_record.read_records(path)] == [1, 3]
    end = cut + len(members[2])
    skipped = [(len(members[0]), cut), (end, path.stat().st_size)]
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 2 and all(
        m.endswith(f"bytes {a} to {b} hold no whole record (cut short or damaged); skipped")
        for m, (a, b) in zip(messages, skipped)), messages
    assert game_record.complete_members_end(path) == end


def test_imitation_pairs_are_the_players_commands_only(tmp_path):
    """A replay in which a neutral side takes turns: the imitation pairs
    are the two players' commands, none of the neutral side's."""
    from tools.replay_dataset import iter_replay_pairs
    data = dict(_duel_data(), commands=[["init_side", s] if i % 2 == 0 else ["end_turn"]
                                        for i, s in enumerate([1, 1, 2, 2, 3, 3, 1, 1])])
    data["starting_units"].append({"uid": 3, "type": "Tentacle of the Deep", "side": 3,
                                   "x": 2, "y": 0})
    data["starting_sides"].append({"side": 3, "gold": 0, "recruit": []})
    path = tmp_path / "g.json.gz"
    _write_corpus_game(path, data)
    movers = [gs.global_info.current_side for gs, _ai in iter_replay_pairs(path)]
    assert movers == [1, 2, 1]
