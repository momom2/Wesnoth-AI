"""The decision log (tools/decision_log.py, RawPolicyPlayer(log_decisions=True)):
the entry's law columns on synthetic laws, the refusal and merge rules, and
short real-core games played with the log on and off."""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.decision_log import (  # noqa: E402
    ENTRY_KEYS, action_key, compact_columns, decision_entry, finish_log, law_columns, merge_logs,
)
from wesnoth_ai.server_priors import KIND_ATTACK, KIND_END_TURN, KIND_MOVE, KIND_RECRUIT  # noqa: E402


def test_law_columns_read_the_normalised_law():
    actor = np.array([0, 0, 0, 1, 2, 3])
    kind = np.array([KIND_ATTACK, KIND_MOVE, KIND_MOVE, KIND_MOVE, KIND_RECRUIT, KIND_END_TURN])
    prior = np.array([0.1, 0.2, 0.1, 0.2, 0.2, 0.2]) * 0.98          # mass 0.98
    q = prior / prior.sum()
    for pick in range(len(prior)):
        c = law_columns(actor, kind, prior, pick)
        assert c["n"] == 6 and c["mass"] == pytest.approx(0.98)
        assert c["logp"] == pytest.approx(math.log(q[pick]))
        assert c["cdf"][0] == pytest.approx(q[:pick].sum()) and c["cdf"][1] == pytest.approx(q[:pick + 1].sum())
        assert c["H"] == pytest.approx(float(-(q * np.log(q)).sum()))
        assert (c["p_end"], c["p_att"], c["p_rec"]) == pytest.approx((q[5], q[0], q[4]))
    assert law_columns(actor, kind, prior, 1)["p_att_actor"] == pytest.approx(0.1 / 0.4)
    assert math.isnan(law_columns(actor, kind, prior, 3)["p_att_actor"])     # no attack for actor 1
    assert math.isnan(law_columns(actor, kind, prior, 5)["p_att_actor"])     # end_turn


def _entry(side, at, refused=False):
    e = decision_entry(side=side, at=at, actor=np.array([0, 1]), kind=np.array([KIND_MOVE, KIND_END_TURN]),
                       target=np.array([4, -1]), weapon=np.array([-1, -1]), prior=np.array([0.5, 0.5]),
                       pick=0, value=0.25)
    e["refused"] = refused
    return e


def test_a_decision_that_emitted_nothing_is_refused():
    """Reported by the harness or not, a decision whose next decision
    starts at the same command is refused, and so is a last decision at
    the end of the commands."""
    log = finish_log([_entry(1, 0), _entry(1, 1, refused=True), _entry(1, 1), _entry(1, 3), _entry(1, 3),
                      _entry(1, 5)], n_commands=5)
    assert [e["refused"] for e in log] == [False, True, False, True, False, True]
    assert [e["cmd"] for e in log] == [0, -1, 1, -1, 3, -1]


def test_merge_puts_the_sides_in_play_order_and_keeps_the_contract_columns():
    a = finish_log([_entry(1, 0), _entry(1, 1, refused=True), _entry(1, 1), _entry(1, 2), _entry(1, 7)], 9)
    b = finish_log([_entry(2, 3), _entry(2, 4), _entry(2, 8)], 9)
    merged = merge_logs([b, a])
    assert [(e["side"], e["cmd"]) for e in merged] == [(1, 0), (1, -1), (1, 1), (1, 2), (2, 3), (2, 4),
                                                      (1, 7), (2, 8)]
    assert all(tuple(e) == ENTRY_KEYS for e in merged)
    with pytest.raises(ValueError, match="two sides"):
        merge_logs([finish_log([_entry(1, 2)], 9), finish_log([_entry(2, 2)], 9)])


# ---------------------------------------------------------------------
# Real games on the core
# ---------------------------------------------------------------------
def _tiny_policy():
    import torch
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(0)
    return TransformerPolicy(d_model=32, num_layers=1, num_heads=2, d_ff=64, device=torch.device("cpu"))


def play(policy, seed: int, log: bool, max_turns: int = 3):
    """One temperature-1 game of `policy` against itself, sides seeded as
    elo_eval_game seeds them: (the simulator, its setup, the merged log or
    None)."""
    from tools.eval_players import _PolicyPair, _play_one_eval_game
    from tools.raw_player import RawPolicyPlayer
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate, random_setup
    setup = random_setup(random.Random(seed))
    sim = WesnothSim(build_scenario_gamestate(setup), scenario_id=setup.scenario_id, max_turns=max_turns)
    sim._seed_salt = f"elo:{seed}"
    pa = RawPolicyPlayer(policy, 1.0, seed=2 * seed, log_decisions=log)
    pb = RawPolicyPlayer(policy, 1.0, seed=2 * seed + 1, log_decisions=log)
    _play_one_eval_game(sim, _PolicyPair(policy=pa, label="a", side=1),
                        _PolicyPair(policy=pb, label="b", side=2), game_label="g")
    n = len(sim.command_history)
    merged = merge_logs([pa.pop_decision_log("g", n), pb.pop_decision_log("g", n)]) if log else None
    return sim, setup, merged


def _fingerprint(sim):
    from wesnoth_ai.classes import state_digest
    return ([list(rc.cmd) for rc in sim.command_history], list(sim.recruit_rejections),
            state_digest(sim.gs, version=sim.digest_version))


def _law_at(policy, gs):
    import torch
    from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors
    with torch.no_grad():
        enc = policy._inference_encoder.encode(gs)
        out = policy._inference_model(enc)
        return compact_columns(enumerate_legal_actions_with_priors(enc, out, gs)), float(out.value.reshape(-1)[0])


def test_logging_changes_no_pick_and_the_log_matches_the_law_it_was_played_from():
    """With the log on and off the games are the same command for command,
    with the same final digest; each logged decision's logp is log(prior /
    mass) of the played action in the law recomputed on the record's
    position before its command, its cdf brackets that action, and its
    first command is the action's."""
    from tools.game_record import game_record, walk
    policy = _tiny_policy()
    for seed in (11,):
        off, _, _ = play(policy, seed, log=False)
        on, setup, log = play(policy, seed, log=True)
        assert _fingerprint(on) == _fingerprint(off)
        assert log and all(tuple(e) == ENTRY_KEYS for e in log)
        live = [e for e in log if not e["refused"]]
        assert [e["cmd"] for e in live] == sorted({e["cmd"] for e in live})
        by_cmd = {e["cmd"]: e for e in live}
        rec = game_record(on, setup, game_label="g")
        checked = 0
        for k, gs, cmd in walk(rec):
            e = by_cmd.get(k)
            if e is None:
                continue
            law, value = _law_at(policy, gs)
            keys = [action_key(*row) for row in zip(law["actor"], law["kind"], law["target"], law["weapon"])]
            i = keys.index(e["action"])
            q = law["prior"] / law["prior"].sum()
            assert e["side"] == gs.global_info.current_side
            assert e["logp"] == pytest.approx(math.log(q[i]), abs=1e-9)
            assert e["cdf"][0] == pytest.approx(q[:i].sum(), abs=1e-9)
            assert e["cdf"][1] - e["cdf"][0] == pytest.approx(q[i], abs=1e-9)
            assert e["v"] == pytest.approx(value, abs=1e-6)
            kind = e["action"][1]
            want = {KIND_ATTACK: ("move", "attack"), KIND_MOVE: ("move",), KIND_RECRUIT: ("recruit",),
                    KIND_END_TURN: ("end_turn",)}[kind]
            assert cmd[0] in want, (e, cmd)
            checked += 1
        assert checked == len(live)


def test_a_refused_recruit_is_logged_refused_and_changes_nothing_else(monkeypatch):
    """The harness refuses the game's first recruit (as it refuses one onto
    a hex a hidden unit holds with no castle hex vacant): the log marks
    that decision refused with no command, the side decides again from the
    same command, and the game is the same with the log off."""
    from tools import eval_players
    real = eval_players._would_recruit_bounce

    def bounce_first_recruit():
        fired = []

        def bounce(action, gs):
            if action.get("type") == "recruit" and not fired:
                fired.append(1)
                return True
            return real(action, gs)
        return bounce

    policy = _tiny_policy()
    monkeypatch.setattr(eval_players, "_would_recruit_bounce", bounce_first_recruit())
    off, _, _ = play(policy, 21, log=False)
    monkeypatch.setattr(eval_players, "_would_recruit_bounce", bounce_first_recruit())
    on, _, log = play(policy, 21, log=True)
    assert _fingerprint(on) == _fingerprint(off)
    refused = [i for i, e in enumerate(log) if e["refused"]]
    assert len(refused) == 1, "the game played no recruit to refuse"
    i = refused[0]
    assert log[i]["cmd"] == -1 and log[i]["action"][1] == KIND_RECRUIT
    assert log[i + 1]["side"] == log[i]["side"]
    live = [e["cmd"] for e in log if not e["refused"]]
    assert live == sorted(set(live)) and len(live) == len(log) - 1
    assert on.recruit_rejections and on.recruit_rejections[0][0] == log[i + 1]["cmd"]


def test_compact_and_list_paths_log_the_same_entry():
    """Behind a shared inference server the player logs from the compact
    arrays; the entry is the list path's on the same state."""
    import threading
    from types import SimpleNamespace
    from tools.inference_seam import InferenceServer, RemoteEncoder, RemoteModel
    from tools.raw_player import RawPolicyPlayer
    from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate, random_setup
    policy = _tiny_policy()
    enc, model = policy._inference_encoder, policy._inference_model
    renc = RemoteEncoder(enc.unit_type_to_id, enc.faction_to_id, terrain_multi_hot=enc.terrain_multi_hot,
                         fog_hides_enemy_villages=enc.fog_hides_enemy_villages, server_priors=True)
    rmodel = RemoteModel(InferenceServer(model, enc))
    sim = SimpleNamespace(command_history=[None] * 7)
    for s in (3, 5):
        gs = build_scenario_gamestate(random_setup(random.Random(s)))
        entries = []
        for compact in (True, False):
            base = SimpleNamespace(_inference_model=rmodel, _inference_encoder=renc,
                                   _lock=threading.Lock(), _decision_step=0)
            player = RawPolicyPlayer(base, 1.0, seed=s, compact_selection=compact, log_decisions=True)
            player.select_action(gs, game_label="t", sim=sim)
            entries.append(player.pop_decision_log("t", 8)[0])
        a, b = entries
        assert a["action"] == b["action"] and a["cmd"] == b["cmd"] == 7 and a["n"] == b["n"]
        for key in ("mass", "logp", "H", "p_end", "p_att", "p_rec", "v"):
            assert a[key] == pytest.approx(b[key], abs=1e-9), key
        assert a["cdf"] == pytest.approx(b["cdf"], abs=1e-9)
