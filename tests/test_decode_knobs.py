"""The raw player's decode knobs (tools/raw_player.py, tools/eval_procedure.py;
docs/decode_knobs_prereg_20261012.md): at zero the choice is today's, decision
for decision; each knob moves the choice where it should; the attack
time-of-day slope reads the attacker's modifier at the hex it strikes from, on
hand-built positions; the procedure tags round-trip; and the guards refuse to
pool games played under different knob values."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import raw_player  # noqa: E402
from tools.eval_procedure import (  # noqa: E402
    add_raw_knob_args, decode_procedure, knob_vector, parse_raw_procedure, procedure_of, raw_knob_argv,
    raw_knobs_of,
)
from tools.raw_player import RawPolicyPlayer, pick_index, shift_by_logits, shift_end_turn_prior  # noqa: E402
from wesnoth_ai.constants import OBSERVATION_EPOCH  # noqa: E402
from wesnoth_ai.game_core import game_core_class  # noqa: E402
from tests.helpers.eval_records import current_forced_faction  # noqa: E402

needs_core = pytest.mark.skipif(game_core_class() is None, reason="wesnoth_core phase 32 not installed")
DAY, NIGHT = 1, 4               # tod_start_index of morning (+25) and first watch (-25) at turn 1
MOVE, ATTACK, RECRUIT, END = 1, 0, 2, 3


def _tiny_policy():
    import torch
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(0)
    return TransformerPolicy(d_model=32, num_layers=1, num_heads=2, d_ff=64, device=torch.device("cpu"))


def _states():
    from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate, random_setup
    return [build_scenario_gamestate(random_setup(random.Random(s))) for s in (3, 5, 8)]


def _forbid(monkeypatch):
    """Make every knob function raise: a decode at zero knobs must not enter them."""
    def boom(*a, **k):
        raise AssertionError("a knob function ran with every knob at zero")
    for name in ("shift_by_logits", "knob_logits", "attack_tod_modifiers"):
        monkeypatch.setattr(raw_player, name, boom)


def test_knobs_at_zero_leave_todays_choice_on_both_paths(monkeypatch):
    """Explicit zero knobs: the list path picks the argmax of today's
    shifted priors and the compact path what the list path picks, with
    no knob function entered."""
    import threading
    import torch
    from tools.inference_seam import InferenceServer, RemoteEncoder, RemoteModel
    from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors
    _forbid(monkeypatch)
    policy = _tiny_policy()
    enc, model = policy._inference_encoder, policy._inference_model
    renc = RemoteEncoder(enc.unit_type_to_id, enc.faction_to_id, terrain_multi_hot=enc.terrain_multi_hot,
                         fog_hides_enemy_villages=enc.fog_hides_enemy_villages, server_priors=True)
    rmodel = RemoteModel(InferenceServer(model, enc))
    zeros = dict(attack_offset=0.0, recruit_offset=0.0, attack_tod_slope=0.0)
    for gs in _states():
        with torch.no_grad():
            out = model(enc.encode(gs))
            legal = enumerate_legal_actions_with_priors(enc.encode(gs), out, gs, decision_step=0)
        priors = np.array([la.prior for la in legal])
        is_end = np.array([la.action["type"] == "end_turn" for la in legal])
        today = legal[pick_index(shift_end_turn_prior(priors, is_end, -1.5), 0.0, None)].action
        policy._decision_step = 0
        assert RawPolicyPlayer(policy, 0.0, end_turn_offset=-1.5, **zeros).select_action(gs) == today
        base = SimpleNamespace(_inference_model=rmodel, _inference_encoder=renc,
                               _lock=threading.Lock(), _decision_step=0)
        compact = RawPolicyPlayer(base, 0.0, end_turn_offset=-1.5, **zeros).select_action(gs)
        listed = RawPolicyPlayer(base, 0.0, end_turn_offset=-1.5, compact_selection=False,
                                 **zeros).select_action(gs)
        assert compact == listed


class _Recording(RawPolicyPlayer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.actions = []

    def select_action(self, game_state, **kw):
        action = super().select_action(game_state, **kw)
        self.actions.append(action)
        return action


def _game_actions(**knobs):
    from tools.eval_players import _PolicyPair, _play_one_eval_game
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate, random_setup
    setup = random_setup(random.Random(3))     # Hamlets: side 1 opens with recruits legal
    sim = WesnothSim(build_scenario_gamestate(setup), scenario_id=setup.scenario_id, max_turns=1)
    # An untrained net ends its turn at once at today's -1.5; at -8 it acts.
    a = _Recording(_tiny_policy(), 0.0, end_turn_offset=-8.0, **knobs)
    b = _Recording(_tiny_policy(), 0.0, end_turn_offset=-8.0, **knobs)
    _play_one_eval_game(sim, _PolicyPair(policy=a, label="a", side=1),
                        _PolicyPair(policy=b, label="b", side=2), game_label="g")
    return a.actions + b.actions


def test_knobs_at_zero_play_todays_game_decision_for_decision():
    zero = _game_actions(attack_offset=0.0, recruit_offset=0.0, attack_tod_slope=0.0)
    assert len(zero) > 5
    assert zero == _game_actions()
    # Control: the comparison sees a knob. Side 1 opens on its keep with
    # gold, where +30 on recruits recruits and -30 does not.
    assert _game_actions(recruit_offset=30.0) != _game_actions(recruit_offset=-30.0)


def _choice(priors, kinds, actors, **knobs):
    """The index the player chooses on constructed arrays (no attack's
    time of day read: the slope is 0 here)."""
    player = RawPolicyPlayer(SimpleNamespace(), 0.0, **knobs)
    kinds, priors = np.asarray(kinds), np.asarray(priors, dtype=np.float64)
    logits = player._knob_logits(kinds, [((0, 0), (1, 0))] * int((kinds == ATTACK).sum()), None) \
        if player.knobs_on else None
    return player._choose(priors, np.asarray(actors), kinds == END, logits)


def test_each_offset_moves_the_choice_across_a_small_gap():
    """A move at 0.31, an attack at 0.29 (a gap of 0.067 in logits), a
    recruit at 0.20, end_turn at 0.20."""
    priors, kinds, actors = [0.31, 0.29, 0.20, 0.20], [MOVE, ATTACK, RECRUIT, END], [0, 0, 1, 2]
    assert _choice(priors, kinds, actors) == 0
    assert _choice(priors, kinds, actors, attack_offset=0.05) == 0
    assert _choice(priors, kinds, actors, attack_offset=0.1) == 1
    assert _choice(priors, kinds, actors, recruit_offset=0.5) == 2
    assert _choice([0.29, 0.31, 0.2, 0.2], kinds, actors, attack_offset=-0.1) == 0
    # The offsets are exact logit shifts of the joint distribution.
    logits = np.log(np.asarray(priors))
    off = np.array([0.0, 0.7, -0.3, 0.0])
    want = np.exp(logits + off) / np.exp(logits + off).sum()
    assert np.allclose(shift_by_logits(np.asarray(priors), off), want, atol=1e-15)


# ---------------------------------------------------------------- time of day
def _view(units, *, tod, special=None, time_areas=None):
    """A hand-built position (tests/helpers/parity_games.py), side 1 to move."""
    from helpers.parity_games import core_of, record
    from wesnoth_ai.game_core import view_of
    leaders = [("Lieutenant", 1, 0, 0, True), ("Lieutenant", 2, 19, 6, True)]
    cs = core_of(record(leaders + units, fog=False, tod_start_index=tod, special=special),
                 time_areas=time_areas)
    cs.apply_command(["init_side", 1])
    return view_of(cs)


@needs_core
def test_the_slope_prefers_the_attack_at_the_favourable_time():
    """A lawful Spearman next to a Grunt fights at +25% by day and -25% at
    night, a chaotic Grunt next to a Spearman the reverse; with a slope of
    1 the attack beats a move 0.067 above it by day and loses by night."""
    from tools.abilities import hex_neighbors
    target = (10, 3)
    near = list(hex_neighbors(*target))[0]
    for attacker, defender, sign in (("Spearman", "Orcish Grunt", 1), ("Orcish Grunt", "Spearman", -1)):
        for tod, expected in ((DAY, 0.25 * sign), (NIGHT, -0.25 * sign)):
            gs = _view([(attacker, 1, *near, False), (defender, 2, *target, False)], tod=tod)
            assert raw_player.attack_tod_modifiers(gs, [(near, target)]).tolist() == [expected]
            player = RawPolicyPlayer(SimpleNamespace(), 0.0, attack_tod_slope=1.0)
            kinds = np.array([MOVE, ATTACK])
            logits = player._knob_logits(kinds, [(near, target)], gs)
            chosen = player._choose(np.array([0.31, 0.29]), np.array([0, 0]), kinds == END, logits)
            assert chosen == (1 if expected > 0 else 0), (attacker, tod)


def _corridor():
    """A Grunt at (10, 3) whose only open neighbour is `strike`; `start`
    two hexes from it, next to `strike`; `lamp` next to `strike` and not
    next to `start`. Every other hex of the Grunt's ring is impassable."""
    from tools.abilities import hex_neighbors
    target = (10, 3)
    ring = set(hex_neighbors(*target))
    strike = (9, 3)
    assert strike in ring
    far = [n for n in hex_neighbors(*strike) if n not in ring and n != target]
    start, lamp = next((a, m) for a in far for m in far if m != a and m not in hex_neighbors(*a))
    special = {h: "Xv" for h in ring - {strike}}
    return target, strike, start, lamp, special


@needs_core
def test_the_slope_reads_the_hex_the_attack_strikes_from():
    """The attacker two hexes away moves to the Grunt's one open neighbour
    first (the simulator's move-to-attack hex), so the slope reads the
    time of day there: night in that hex's time area under a day board;
    under a night board, lit back to 0 by a Mage of Light next to it,
    which does not reach the attacker's own hex."""
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.classes import Position
    target, strike, start, lamp, special = _corridor()
    units = [("Spearman", 1, *start, False), ("Orcish Grunt", 2, *target, False)]
    gs = _view(units, tod=DAY, special=special, time_areas={strike: [-25] * 6})
    assert raw_player.attack_tod_modifiers(gs, [(start, target)]).tolist() == [-0.25]
    from wesnoth_ai.game_core import core_for
    assert core_for(gs).core.combat_tod_modifier(*start, *start) == 25        # its own hex: day
    sim = WesnothSim.__new__(WesnothSim)
    sim.gs = gs
    attacker = next(u for u in gs.map.units if (u.position.x, u.position.y) == start)
    assert sim._find_attack_hex(attacker, Position(*target)) == Position(*strike)

    lit = _view(units + [("Mage of Light", 1, *lamp, False)], tod=NIGHT, special=special)
    assert raw_player.attack_tod_modifiers(lit, [(start, target)]).tolist() == [0.0]
    assert core_for(lit).core.combat_tod_modifier(*start, *start) == -25      # unlit where it stands
    # An attack on the leader across the board has no hex to strike from:
    # read as 0 and counted. No unit on the hex, a hex off the map: no reading.
    unread = {}
    assert raw_player.attack_tod_modifiers(lit, [(start, (19, 6))], unread).tolist() == [0.0]
    assert unread == {"no hex to strike from": 1}
    assert core_for(lit).core.combat_tod_modifier(5, 5, *strike) is None
    assert core_for(lit).core.combat_tod_modifier(*start, 99, 99) is None


# ---------------------------------------------------------------- tags and flags
def test_the_procedure_tag_names_every_knob_and_round_trips():
    knobs = {"attack_offset": 0.4, "recruit_offset": -0.2, "attack_tod_slope": 1.3}
    tag = procedure_of(0, False, False, 0.0, raw_end_turn_offset=-2.1, raw_knobs=knobs)
    assert tag == "raw:t0+eo-2.1+ao0.4+ro-0.2+at1.3"
    parsed = parse_raw_procedure(tag)
    assert parsed == {"raw_temperature": 0.0, "raw_end_turn": "joint", "raw_end_turn_offset": -2.1, **knobs}
    assert procedure_of(0, False, False, parsed["raw_temperature"],
                        raw_end_turn_offset=parsed["raw_end_turn_offset"],
                        raw_knobs={k: parsed[k] for k in knobs}) == tag
    # Zero knobs leave today's tag; the actor rule rides along.
    assert procedure_of(0, False, False, 0.0, raw_end_turn_offset=-1.5,
                        raw_knobs=dict.fromkeys(knobs, 0.0)) == "raw:t0+eo-1.5"
    assert parse_raw_procedure("raw:t0+endm+ro0.5")["raw_end_turn"] == "actor"
    for bad in ("raw:t0+zz1", "mcts:32", "raw"):
        with pytest.raises(ValueError):
            parse_raw_procedure(bad)
    # Every vector of the search: tag -> values -> the same tag.
    from tools import reference_player
    from tools.knob_search import search_vectors
    ref = reference_player.load()
    for vec in search_vectors(knob_vector(ref["decode"])):
        decode = reference_player.with_decode(ref, vec)["decode"]
        back = parse_raw_procedure(decode_procedure(decode))
        assert [back["raw_end_turn_offset"], back["attack_offset"], back["recruit_offset"],
                back["attack_tod_slope"]] == list(vec.values())


def test_the_knob_flags_round_trip_through_a_parser():
    import argparse
    ap = argparse.ArgumentParser()
    add_raw_knob_args(ap)
    knobs = {"attack_offset": -0.75, "recruit_offset": 0.0, "attack_tod_slope": 3.25}
    args = ap.parse_args(raw_knob_argv(knobs, "b"))
    assert raw_knobs_of(args, "b") == knobs and not any(raw_knobs_of(args, "a").values())
    assert raw_knob_argv(knobs, "b") == ["--raw-attack-offset-b", "-0.75", "--raw-attack-tod-slope-b", "3.25"]


def test_the_reference_config_carries_its_knob_vector(tmp_path):
    from tools import reference_player
    ref = reference_player.load()
    assert ref["procedure_tag"] == decode_procedure(ref["decode"])
    assert not [f for f in reference_player.batch_flags("b", ref) if "attack" in f or "recruit" in f]
    vec = {"raw_end_turn_offset": -2.1, "raw_attack_offset": 0.4}        # absent keys are 0
    moved = reference_player.with_decode(ref, vec)
    flags = reference_player.batch_flags("a", moved)
    assert flags[flags.index("--raw-attack-offset-a") + 1] == "0.4"
    assert "--raw-recruit-offset-a" not in flags and moved["procedure_tag"] == "raw:t0+eo-2.1+ao0.4"
    stale = dict(moved, procedure_tag=ref["procedure_tag"])
    path = tmp_path / "ref.json"
    path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(SystemExit, match="procedure_tag"):
        reference_player.load(path)
    with pytest.raises(ValueError, match="no decode reads"):
        knob_vector({"raw_attack_ofset": 0.4})


# ---------------------------------------------------------------- guards
def _batch(tmp_path, *extra):
    from tools.run_elo_batch import main
    return main(["x", "--label-a", "A", "--spec-a", "random", "--label-b", "B", "--spec-b", "dummy",
                 "--outdir", str(tmp_path / "games"), "--games", "2", "--mcts-sims", "0",
                 "--device", "cpu", "--jobs", "1", "--time-budget-min", "0", "--min-free-mb", "0",
                 *extra])


def test_the_drivers_refuse_knobs_a_side_would_not_read(tmp_path):
    for extra in (["--mcts-sims", "8", "--raw-attack-offset-a", "0.4"],
                  ["--raw-recruit-offset-a", "0.4"],                         # the legacy sampler
                  ["--raw-temperature-b", "0", "--raw-attack-tod-slope-b", "1"]):   # 'dummy'
        with pytest.raises(SystemExit) as refused:
            _batch(tmp_path, *extra)
        assert refused.value.code == 2, extra
    _batch(tmp_path, "--raw-temperature-a", "0", "--raw-attack-tod-slope-a", "1")


def _prev(procedure_a: str) -> dict:
    """A result file that passes every guard but the procedure's."""
    return {"label_a": "A", "label_b": "B", "procedure_a": procedure_a, "procedure_b": "raw",
            "max_turns": 200, "combat_stream": "per_game", "observation_epoch": OBSERVATION_EPOCH,
            "forced_faction": current_forced_faction(), "terrain_a": "set", "terrain_b": "class",
            "outcome_a": "win", "seed": 7}


def test_an_outdir_refuses_games_of_another_knob_value(tmp_path):
    from tools.elo_eval_game import main as play
    games = tmp_path / "games"
    games.mkdir()
    slot = games / "game_A_B_s1_7.json"
    slot.write_text(json.dumps(_prev("raw:t0+ao0.4")), encoding="utf-8")
    with pytest.raises(SystemExit, match="holds procedure"):
        _batch(tmp_path, "--raw-temperature-a", "0", "--raw-attack-offset-a", "0.5")
    _batch(tmp_path, "--raw-temperature-a", "0", "--raw-attack-offset-a", "0.4")
    game = ["x", "A", "random", "B", "dummy", "1", "7", str(games), "--mcts-sims", "0",
            "--raw-temperature-a", "0", "--device", "cpu"]
    with pytest.raises(SystemExit, match="refusing to mix estimands"):
        play(game + ["--raw-attack-offset-a", "0.5"])
    assert play(game + ["--raw-attack-offset-a", "0.4"]) == 0          # the same knobs: kept


def test_the_fit_refuses_a_dir_that_mixes_knob_values(tmp_path):
    from tools import elo_collect
    d = tmp_path / "games"
    d.mkdir()
    for seed, tag in ((1, "raw:t0+eo-1.5"), (2, "raw:t0+eo-1.5+at1.3")):
        (d / f"game_A_B_s1_{seed}.json").write_text(json.dumps(
            {"label_a": "A", "label_b": "B", "outcome_a": "win", "margin_a": 0.5, "max_turns": 200,
             "procedure_a": tag, "procedure_b": "raw:t0+eo-1.5"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="mixed procedures"):
        elo_collect.main(["x", str(d), "--no-catalog"])
