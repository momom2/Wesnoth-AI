"""A unit that levels up on its own side's turn with several types to become
is the side's to choose (docs/blindspot_audit_20261125.md, gap 2;
docs/wesnoth_rules.md, "Who chooses an advancement, and what an unasked unit
becomes"; the core's `ask_advance`).

The simulator stops after the fight with the unit waiting. While it waits,
the recruit rows hold the types it may become, on its own hex, and nothing
else is legal; the policy gives its choice as a recruit of one of them,
which the simulator applies as an `["advance", choice]` command. A unit
that levels up on the enemy's turn takes the first type listed, as Wesnoth
1.18 gives a player.

The board is The Freelands with every unit replaced (the surgery of
tests/test_sim_advance.py). A Skeleton becomes a Revenant or a Deathblade.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from wesnoth_ai.classes import Position  # noqa: E402
from wesnoth_ai.game_core import game_core_class  # noqa: E402

pytestmark = pytest.mark.skipif(game_core_class() is None, reason="wesnoth_core.GameCore not available")

OPTIONS = ["Revenant", "Deathblade"]
AT, FOE = Position(10, 10), Position(11, 10)


def _sim(*, attacker_exp: int, defender: str = "Spearman", defender_hp=None, defender_exp: int = 0,
         ask: bool = True):
    """Side 1's Skeleton on (10, 10), `attacker_exp` experience of its 27,
    beside a unit of side 2 on (11, 10); a leader each, far away; side 1 to
    move."""
    from sim_test_helpers import commit_view, fresh_scenario_sim
    from test_sim_advance import _make
    sim = fresh_scenario_sim(seed=7, max_turns=10, scenario_id="multiplayer_The_Freelands")
    xpmod = int(getattr(sim.gs.global_info, "_experience_modifier", 100) or 100)
    sim.gs.map.units.clear()
    foe = _make(defender, 2, FOE.x, FOE.y, 2, current_hp=defender_hp, exp_modifier=xpmod)
    foe.current_exp = defender_exp if defender_exp >= 0 else foe.max_exp + defender_exp
    for u in (_make("Skeleton", 1, AT.x, AT.y, 1, current_exp=attacker_exp, exp_modifier=xpmod), foe,
              _make("Skeleton", 1, 20, 20, 100, is_leader=True, exp_modifier=xpmod),
              _make("Skeleton", 2, 21, 20, 101, is_leader=True, exp_modifier=xpmod)):
        sim.gs.map.units.add(u)
    sim.gs.global_info.current_side = 1
    if not ask:
        sim.gs.global_info.__dict__.pop("_ask_advance", None)
    commit_view(sim)
    sim._begin_side_turn(1)
    return sim


def _attack(sim):
    sim.step({"type": "attack", "start_hex": AT, "target_hex": FOE, "attack_index": 0})
    return sim.command_history[-1]


def _unit_at(sim, pos: Position):
    return next(u for u in sim.gs.map.units if (u.position.x, u.position.y) == (pos.x, pos.y))


def _encoded(sim):
    from tools.inference_seam import RemoteEncoder
    encoder = RemoteEncoder({}, {}, relevant_set=True, fog_hides_enemy_villages=True, server_priors=True,
                            terrain_multi_hot=True, observation_parity=True, relevant_set_version=2)
    return encoder.encode(sim.gs)


def test_the_side_is_asked_and_its_choice_is_what_the_unit_becomes():
    """The Skeleton kills a Spearman at 1 hit point and passes its 27
    experience. The attack leaves it a Skeleton that waits; the options
    stand in the recruit rows on its hex and are the only legal orders;
    choosing the second makes it a Deathblade."""
    sim = _sim(attacker_exp=22, defender_hp=1)
    rc = _attack(sim)
    assert rc.kind == "attack" and "advance_choices" not in rc.extras
    unit = _unit_at(sim, AT)
    assert unit.name == "Skeleton" and unit.current_exp >= unit.max_exp
    waits = sim.core.core.advance_options(1)
    assert waits is not None and (waits[1], waits[2]) == (AT.x, AT.y) and list(waits[3]) == OPTIONS
    assert sim.core.core.advance_options(2) is None

    encoded = _encoded(sim)
    assert list(encoded.recruit_types) == OPTIONS
    m = encoded._masks
    U, R, H = m.n_units, m.n_recruits, m.n_hexes
    assert R == 2 and list(m.actor_mask[:U]) == [0] * U, "no unit acts"
    assert list(m.actor_mask[U:U + R]) == [1, 1] and m.actor_mask[U + R] == 0, "the options, and no end of turn"
    here = [k for k, p in enumerate(encoded.hex_positions) if (p.x, p.y) == (AT.x, AT.y)]
    for r in range(R):
        row = np.unpackbits(m.union_valid[U + r])[:H]
        assert np.flatnonzero(row).tolist() == here, "an option's one target is the unit's hex"
    assert not np.unpackbits(m.move_valid, axis=1)[:, :H].any() and not np.unpackbits(m.attack_valid, axis=1)[:, :H].any()

    sim.step({"type": "recruit", "unit_type": "Deathblade", "target_hex": AT})
    rc = sim.command_history[-1]
    assert not sim.last_step_rejected and rc.kind == "advance" and list(rc.cmd) == ["advance", 1]
    assert rc.extras["advance_choices"] == [(1, 1)]
    unit = _unit_at(sim, AT)
    assert unit.name == "Deathblade" and unit.current_exp < unit.max_exp
    assert sim.core.core.advance_options(1) is None
    after = _encoded(sim)
    assert after._masks.actor_mask[-1] == 1, "the turn goes on"
    assert list(after.recruit_types) != OPTIONS


def test_the_first_option_is_as_good_a_choice():
    sim = _sim(attacker_exp=22, defender_hp=1)
    _attack(sim)
    sim.step({"type": "recruit", "unit_type": "Revenant", "target_hex": AT})
    assert _unit_at(sim, AT).name == "Revenant"
    assert sim.command_history[-1].extras["advance_choices"] == [(1, 0)]


def test_any_other_order_is_refused_while_the_unit_waits():
    """An order that is not one of the options changes nothing and is
    decided again. Eight refusals in a row end the side's turn, as for any
    order the simulator cannot apply, and the unit then takes the first
    type: the game cannot go on with a unit that waits."""
    sim = _sim(attacker_exp=22, defender_hp=1)
    _attack(sim)
    n = len(sim.command_history)
    for action in ({"type": "end_turn"},
                   {"type": "move", "start_hex": Position(20, 20), "target_hex": Position(20, 19)},
                   {"type": "recruit", "unit_type": "Spearman", "target_hex": AT}):
        sim.step(action)
        assert sim.last_step_rejected and len(sim.command_history) == n
        assert _unit_at(sim, AT).name == "Skeleton"
    for _ in range(8):
        if not sim.last_step_rejected and len(sim.command_history) > n:
            break
        sim.step({"type": "end_turn"})
    assert [rc.kind for rc in sim.command_history[n:]][0] == "end_turn"
    assert next(u for u in sim.gs.map.units if u.id == sim.core.core.unit_id_at(AT.x, AT.y, 1)).name == "Revenant"
    assert sim.core.core.advance_options(1) is None


def _tiny_policy():
    import torch
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(1)
    return TransformerPolicy(d_model=32, num_layers=1, num_heads=2, d_ff=64, device=torch.device("cpu"))


def test_a_player_that_builds_its_masks_in_python_is_offered_the_options_alone():
    """The masks built in Python (`action_sampler._build_legality_masks`,
    read by a player that is not served its masks by the core) follow the
    core's rule while a unit waits: the options on the unit's hex, no unit
    and no end of turn. Before 2026-11-27 they offered the turn's ordinary
    orders, which the simulator refuses, and an in-process game stopped on
    "a mask/simulator disagreement"."""
    import torch
    from wesnoth_ai.action_sampler import (_build_legality_masks, _enumerate_legal_actions_reference,
                                           enumerate_legal_actions_with_priors)
    sim = _sim(attacker_exp=22, defender_hp=1)
    _attack(sim)
    policy = _tiny_policy()
    with torch.no_grad():
        encoded = policy._inference_encoder.encode(sim.gs)
        output = policy._inference_model(encoded)
        masks = _build_legality_masks(encoded, sim.gs)
        legal = enumerate_legal_actions_with_priors(encoded, output, sim.gs)
        reference = _enumerate_legal_actions_reference(encoded, output, sim.gs)
    assert list(encoded.recruit_types) == OPTIONS
    U, R = encoded.unit_tokens.size(1), encoded.recruit_tokens.size(1)
    valid = masks.actor_valid[0].tolist()
    assert valid[:U] == [0.0] * U and valid[U:U + R] == [1.0] * R and valid[U + R] == 0.0
    here = [k for k, p in enumerate(encoded.hex_positions) if (p.x, p.y) == (AT.x, AT.y)]
    for r in range(R):
        assert masks.target_valid[U + r].nonzero().flatten().tolist() == here
    assert not masks.target_valid_move.any() and not masks.target_valid_attack.any()

    assert [la.action for la in legal] == [la.action for la in reference]
    assert sorted(la.action["unit_type"] for la in legal) == sorted(OPTIONS)
    assert all(la.action["type"] == "recruit" and la.action["target_hex"] == AT for la in legal)
    assert abs(sum(la.prior for la in legal) - 1.0) < 1e-5

    sim.step(legal[1].action)
    assert not sim.last_step_rejected and sim.command_history[-1].kind == "advance"
    assert _unit_at(sim, AT).name == legal[1].action["unit_type"]
    with torch.no_grad():
        after = policy._inference_encoder.encode(sim.gs)
        assert _build_legality_masks(after, sim.gs).actor_valid[0, -1] == 1.0, "the turn goes on"


def test_an_in_process_player_chooses_and_is_not_refused():
    """The raw player deciding in process, the one a game without the
    inference server plays with: its order while the unit waits is one of
    the options, at any seed."""
    from tools.raw_player import RawPolicyPlayer
    policy = _tiny_policy()
    for seed in range(6):
        sim = _sim(attacker_exp=22, defender_hp=1)
        _attack(sim)
        player = RawPolicyPlayer(policy, 1.0, seed=seed)
        action = player.select_action(sim.gs, game_label=f"wait{seed}", sim=sim)
        assert action["type"] == "recruit" and action["unit_type"] in OPTIONS
        sim.step(action)
        assert not sim.last_step_rejected and sim.last_step_refusal is None
        assert _unit_at(sim, AT).name == action["unit_type"]


def test_a_unit_that_levels_up_on_the_enemys_turn_takes_the_first_type():
    """Side 2's Skeleton, one experience short, is attacked by side 1's
    and gains its level's worth: nobody is asked."""
    sim = _sim(attacker_exp=0, defender="Skeleton", defender_exp=-1)
    rc = _attack(sim)
    assert rc.extras["advance_choices"] == [(2, 0)]
    assert _unit_at(sim, FOE).name == "Revenant" and _unit_at(sim, AT).name == "Skeleton"
    assert sim.core.core.advance_options(1) is None and sim.core.core.advance_options(2) is None


def test_the_defenders_advancement_follows_the_attackers_choice():
    """Both level up in one fight. The engine advances the attacker, then
    the defender (attack.cpp:1556-1567): the defender waits with it."""
    sim = _sim(attacker_exp=26, defender="Skeleton", defender_exp=-1)
    rc = _attack(sim)
    assert "advance_choices" not in rc.extras
    assert _unit_at(sim, FOE).name == "Skeleton", "not before the attacker"
    sim.step({"type": "recruit", "unit_type": "Deathblade", "target_hex": AT})
    assert sim.command_history[-1].extras["advance_choices"] == [(1, 1), (2, 0)]
    assert (_unit_at(sim, AT).name, _unit_at(sim, FOE).name) == ("Deathblade", "Revenant")


def test_without_the_rule_the_attacker_takes_the_first_type_at_once():
    """The core's default, which reconstruction and the records of earlier
    formats play under."""
    sim = _sim(attacker_exp=22, defender_hp=1, ask=False)
    rc = _attack(sim)
    assert rc.extras["advance_choices"] == [(1, 0)]
    assert _unit_at(sim, AT).name == "Revenant"
    assert sim.core.core.advance_options(1) is None


def test_the_wait_survives_the_round_trip_through_a_state_and_a_fork():
    from wesnoth_ai.game_core import CoreState
    sim = _sim(attacker_exp=22, defender_hp=1)
    _attack(sim)
    again = CoreState.from_state(sim.core.to_state())
    assert again.core.advance_options(1) == sim.core.core.advance_options(1)
    again.apply_command(["advance", 1])
    assert again.core.advance_options(1) is None
    fork = sim.fork()
    fork.step({"type": "recruit", "unit_type": "Revenant", "target_hex": AT})
    assert _unit_at(fork, AT).name == "Revenant" and _unit_at(sim, AT).name == "Skeleton"


def test_the_record_rebuilds_and_the_replay_holds_the_choice():
    """The record carries the rule and the `advance` command, and rebuilds
    the game; rebuilt without the rule it is another game. The exported
    replay writes the choice as the [choose] after the attack's random
    seed, with the chosen value."""
    import json
    from tools import game_record
    from tools.sim_to_replay import _build_replay_wml
    from wesnoth_ai.rules.scenario_pool import ScenarioSetup
    sim = _sim(attacker_exp=22, defender_hp=1)
    _attack(sim)
    sim.step({"type": "recruit", "unit_type": "Deathblade", "target_hex": AT})
    kinds = [rc.kind for rc in sim.command_history]
    assert kinds[-2:] == ["attack", "advance"]
    wml = _build_replay_wml(sim.command_history[-2:])
    assert wml.count("[choose]") == 1 and 'value="1"' in wml
    assert wml.index("[attack]") < wml.index("[random_seed]") < wml.index("[choose]")
    setup = ScenarioSetup("multiplayer_The_Freelands", "Undead", "Revenant", "Undead", "Revenant")
    rec = json.loads(json.dumps(game_record.game_record(sim, setup, game_label="g")))
    assert rec["format"] == game_record.FORMAT == 8 and rec["ask_advance"] is True
    assert ["advance", 1] in rec["commands"]


def test_the_choice_is_a_legal_decision_of_the_recruit_kind():
    """What the gradient pass and the decision log need: at the position
    where the unit waits, the action (option's row, recruit, the unit's
    hex) is legal under the packed masks, and ending the turn is not."""
    from tools.pg_positions import KIND_COMMANDS, action_is_legal
    from wesnoth_ai.server_priors import KIND_END_TURN, KIND_RECRUIT, core_packed_masks
    sim = _sim(attacker_exp=22, defender_hp=1)
    _attack(sim)
    types = sorted({u.name for u in sim.gs.map.units} | set(OPTIONS))
    raw = sim.core.encode_raw(type_to_id={t: i for i, t in enumerate(types)}, faction_to_id={}, masks=True,
                              relevant_set=True, fog_hides_enemy_villages=True, terrain_multi_hot=True,
                              observation_parity=True, relevant_set_version=2, unit_row_version=2)
    masks = core_packed_masks(raw)
    U, R = masks.n_units, masks.n_recruits
    here = next(k for k, (x, y) in enumerate(zip(raw.hex_xs, raw.hex_ys)) if (int(x), int(y)) == (AT.x, AT.y))
    assert list(raw.recruit_types) == OPTIONS
    assert (int(raw.recruit_xs[0]), int(raw.recruit_ys[0])) == (AT.x, AT.y), "the options stand on the unit's hex"
    for r in range(R):
        assert action_is_legal(masks, (U + r, KIND_RECRUIT, here, -1))
    assert not action_is_legal(masks, (U + R, KIND_END_TURN, -1, -1))
    assert "advance" in KIND_COMMANDS[KIND_RECRUIT]


def test_whole_games_with_choices_play_through_the_server_and_the_pass(tmp_path):
    """At 10% experience units level up within a few fights. A tiny network
    plays such games through the inference server, sampling: its
    advancement choices are recruit-kind decisions among the options, the
    records rebuild, and the gradient pass reproduces every probability the
    players logged, the choices' included (its exact check)."""
    import json
    from unittest import mock
    import tools.elo_eval_game as eg
    import tools.pg_game_pass as gp
    import wesnoth_ai.rules.scenario_pool as pool
    from helpers.pg_games import save_network, selfplay_games, tiny_network
    from tools.game_record import read_records, rebuild
    from tools.pg_positions import KIND_RECRUIT

    real = pool.build_scenario_gamestate

    def quick(setup, **kw):
        return real(setup, **{"experience_modifier": 10, **kw})

    vocab, encoder, model = tiny_network()
    checkpoint = save_network(tmp_path / "pi.pt", vocab, encoder, model)
    games = tmp_path / "games"
    with mock.patch.object(pool, "build_scenario_gamestate", quick), \
            mock.patch.object(eg, "build_scenario_gamestate", quick):
        records = selfplay_games(checkpoint, games, [(1, 13), (2, 7), (1, 47), (2, 52)], max_turns=14)
        choices = types = 0
        for path in records.values():
            [rec] = list(read_records(path))
            assert rec["ask_advance"] is True
            rebuild(rec)                                   # every turn's fingerprint
            commands = rec["commands"]
            for d in rec["decisions"]:
                if d.get("refused") or commands[d["cmd"]][0] != "advance":
                    continue
                choices += 1
                types += int(commands[d["cmd"]][1] > 0)
                assert commands[d["cmd"] - 1][0] == "attack", "a choice follows the attack that earned it"
                assert int(d["action"][1]) == KIND_RECRUIT, "the choice is logged as a recruit-kind decision"
        assert choices >= 3, f"only {choices} advancement choices in four games: the fixture tests nothing"
        assert types >= 1, "no choice took a type other than the first"
        grad = tmp_path / "grad.pt"
        assert gp.main(["--checkpoint", str(checkpoint), "--games", str(games), "--out", str(grad),
                        "--device", "cpu", "--workers", "0"]) == 0
    replay = json.loads(grad.with_suffix(".json").read_text(encoding="utf-8"))["replay"]
    assert replay["half_mean_square"] < 1e-9, replay


def test_the_match_loops_bounce_check_lets_the_choice_through():
    """A match loop asks, before each recruit, whether the engine would
    refuse it for an occupied hex with no vacant castle hex left
    (`_would_recruit_bounce`), and decides again if so. The choice is given
    as a recruit on the waiting unit's own hex: read as a recruit it would
    bounce every time and the loop would never end. The premise is shown on
    the same board without the wait."""
    from tools.selfplay_game import _would_recruit_bounce
    order = {"type": "recruit", "unit_type": "Deathblade", "target_hex": AT}
    plain = _sim(attacker_exp=22, defender_hp=1, ask=False)
    _attack(plain)
    assert _would_recruit_bounce(order, plain.gs), "the premise: a recruit on that hex would bounce"
    sim = _sim(attacker_exp=22, defender_hp=1)
    _attack(sim)
    assert not _would_recruit_bounce(order, sim.gs)
    sim.step(order)
    assert _unit_at(sim, AT).name == "Deathblade"
