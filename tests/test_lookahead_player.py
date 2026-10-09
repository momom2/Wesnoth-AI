"""The look-ahead player (tools/lookahead_player.py): the raw player's prior
tilted by a one-step look-ahead with exact chance, on a world that holds
only what the deciding side knows (wesnoth_ai/lookahead_world.py)."""
from __future__ import annotations

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
from tools.lookahead_player import LookaheadPlayer, candidate_indices, tilted_choice  # noqa: E402
from tools.raw_player import RawPolicyPlayer  # noqa: E402
from wesnoth_ai import game_core as gc  # noqa: E402
from wesnoth_ai.classes import Position, deep_state_fingerprint  # noqa: E402
from wesnoth_ai.lookahead_config import LookaheadConfig, config_record, procedure_tag  # noqa: E402
from wesnoth_ai.lookahead_evaluators import Evaluator, MaterialEvaluator  # noqa: E402
from wesnoth_ai.lookahead_world import build_world, expand  # noqa: E402

pytestmark = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")

SLOTS = 8
OFFSET = -1.5
NULL_OFFSET = -6.0             # a random network ends its turns; the null game needs actions
TYPES = ["Lieutenant", "Spearman", "Bowman"]
HIDDEN = ("u3", "u5")          # the enemy leader and bowman, in side 1's fog at the start


def _sim(max_turns: int = 3, defender_hp: int = 0):
    """Side 1's Spearman stands next to side 2's; side 2's leader and a
    Bowman stand in side 1's fog, with a village of side 2's."""
    from tools.wesnoth_sim import WesnothSim
    spear2 = ("Spearman", 2, 9, 3, False) + (({"hp": defender_hp},) if defender_hp else ())
    data = record([("Lieutenant", 1, 2, 3, True), ("Spearman", 1, 8, 3, False),
                   ("Lieutenant", 2, 18, 3, True), spear2, ("Bowman", 2, 15, 6, False)],
                  special={(2, 3): "Kh", (2, 2): "Ch", (3, 3): "Ch", (1, 3): "Ch", (16, 6): "Gg^Vh",
                           (17, 3): "Kh", (17, 2): "Ch"},
                  recruits={1: ["Spearman", "Bowman"], 2: ["Spearman"]}, fog=True,
                  villages={2: [(16, 6)]})
    sim = WesnothSim(state_of(data), scenario_id="", apply_scenario_events=False,
                     max_turns=max_turns)
    sim._seed_salt = "lookahead-test"
    return sim


def _policy():
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(0)
    return TransformerPolicy(d_model=32, num_layers=1, num_heads=2, d_ff=64, device=torch.device("cpu"),
                             relevant_set_hexes=True, observation_parity=True, memory_slots=SLOTS,
                             relevant_set_version=2)


def _raw(policy, offset: float = OFFSET):
    return RawPolicyPlayer(policy, 0.0, memory_slots=SLOTS, end_turn_offset=offset)


def _cfg(**kw):
    base = {"k": 10_000, "c": 1.0, "sigma": 0.1}
    base.update(kw)
    return LookaheadConfig(**base)


class _Stub(Evaluator):
    """Values a state by `value_of(core, side)` and keeps the ids of every
    state's side-2 units (a unit created in the observed world takes the
    next id after the units on its board, which may repeat a removed
    unit's)."""

    def __init__(self, value_of=lambda core, side: 0.0):
        super().__init__()
        self.value_of = value_of
        self.seen = []

    def values(self, states, side):
        self.states += len(states)
        self.seen += [{d["id"] for d in s.core.core.units_export() if d["side"] == 2} for s in states]
        return np.array([self.value_of(s.core.core, side) for s in states], dtype=np.float64)


def _live(sim) -> tuple:
    """What a look-ahead could corrupt in the live game: the core's whole
    state as a view exports it, its state key, sighting records and
    cleared hexes, and the simulator's dice counter."""
    core = sim.core.core
    return (deep_state_fingerprint(sim.core.to_state()), sim.core.state_key(),
            [core.sightings_export(s) for s in (1, 2)], core.fog_cleared_export(), sim._rng_requests)


def _decide(player, sim, label="g"):
    from wesnoth_ai.game_core import snapshot_view
    return player.select_action(snapshot_view(sim.gs), game_label=label, sim=sim)


def test_with_c_zero_the_player_plays_the_raw_players_game():
    """The null control: every candidate expanded and valued (k covers all
    legal actions), the clip at 0, and the same game decision for decision,
    the live simulator untouched by the look-ahead."""
    policy = _policy()
    raw_sim, la_sim = _sim(), _sim()
    raw = _raw(policy, NULL_OFFSET)
    la = LookaheadPlayer(_raw(policy, NULL_OFFSET), _cfg(c=0.0), MaterialEvaluator(250.0))
    n = 0
    while not raw_sim.done and n < 60:
        want = _decide(raw, raw_sim, "r")
        before = _live(la_sim)
        got = _decide(la, la_sim, "l")
        assert _live(la_sim) == before, "the look-ahead changed the live game"
        assert got == want, f"decision {n}"
        raw_sim.step(want)
        la_sim.step(got)
        n += 1
    tel = la.telemetry("l")
    assert n >= 10 and tel["operated"] >= 5
    assert tel["candidates_by_kind"]["attack"] > 0 and tel["candidates_by_kind"]["end_turn"] > 0
    assert tel["failed"] == {} and tel["flips"] == 0


def test_attack_outcomes_are_the_exact_fight_distribution():
    from tools.combat_outcomes import enumerate_attack_outcomes
    sim = _sim(defender_hp=9)
    world = build_world(sim, 1, "observed", "t")
    attack = {"type": "attack", "start_hex": Position(8, 3), "target_hex": Position(9, 3), "attack_index": 0}
    outs = expand(world, attack)
    assert math.isclose(sum(o.prob for o in outs), 1.0, abs_tol=1e-12)
    dist = enumerate_attack_outcomes(sim.gs, attack)
    got, want = {}, {}
    for o in outs:
        hp = {d["id"]: d["current_hp"] for d in o.core.core.units_export()}
        key = (hp.get("u2", 0), hp.get("u4", 0))
        got[key] = got.get(key, 0.0) + o.prob
    for k, p in dist.probs.items():
        want[(k[0], k[1])] = want.get((k[0], k[1]), 0.0) + p
    assert len(outs) == len(dist.probs) and set(got) == set(want)
    assert all(math.isclose(got[k], want[k], abs_tol=1e-12) for k in want)
    kill = sum(o.prob for o in outs if "u4" not in o.core.core.unit_ids())
    assert 0.0 < kill < 1.0
    assert math.isclose(kill, sum(p for k, p in dist.probs.items() if k[1] == 0), abs_tol=1e-12)
    # From two hexes away the attacker walks next to the target first, as step() does.
    moved = expand(world, {"type": "attack", "start_hex": Position(2, 3), "target_hex": Position(9, 3),
                           "attack_index": 0})
    assert math.isclose(sum(o.prob for o in moved), 1.0, abs_tol=1e-12)
    for o in moved:
        leader = next(d for d in o.core.core.units_export() if d["id"] == "u1")
        assert abs(leader["x"] - 9) <= 1 and abs(leader["y"] - 3) <= 1


def test_the_observed_world_holds_only_what_the_side_sees():
    """Hidden enemies are absent from every evaluated state of the observed
    world and present in god view; the side's own observation of the world
    is its observation of the game."""
    from wesnoth_ai.critic_data import encode_view, pack_raw
    sim = _sim()
    assert set(HIDDEN).isdisjoint(sim.core.core.visible_ids(1))
    vocab = vocab_of(TYPES)
    world = build_world(sim, 1, "observed", "t")
    assert pack_raw(encode_view(world.sim.core, "obs", vocab, FACTION_IDS)) \
        == pack_raw(encode_view(sim.core, "obs", vocab, FACTION_IDS))
    assert world.sim.core.core.village_owner_export() == [], "side 2's fogged village shows no flag"
    for determinization, present in (("observed", False), ("godview", True)):
        stub = _Stub()
        la = LookaheadPlayer(_raw(_policy()), _cfg(determinization=determinization), stub)
        _decide(la, sim)
        assert len(stub.seen) > 10
        assert all(set(HIDDEN) <= ids if present else ids <= {"u4"} for ids in stub.seen)
        assert la.telemetry("g")["failed"] == {}


def test_a_hidden_leader_keeps_its_side_alive():
    sim = _sim()
    world = build_world(sim, 1, "observed", "t")
    assert world.sim.hidden_leader_sides == {2}
    (after,) = expand(world, {"type": "end_turn"})
    assert after.terminal is None and after.side_to_move == 2


def test_a_preferred_candidate_wins_within_twice_c_and_never_beyond():
    """A stub that values one non-argmax candidate's state at +1 and every
    other at -1, with sigma small enough that the clip binds: the
    candidate is played exactly when its log-prior gap to the argmax is
    under 2c."""
    policy = _policy()
    sim = _sim()
    decoded = _raw(policy).decode(sim.gs, game_label="twin")
    base = int(np.argmax(decoded.priors))
    i = next(int(j) for j in np.argsort(-decoded.priors)
             if int(j) != base and decoded.action(int(j))["type"] == "move")
    # The world the player builds at its first decision of game "g".
    (preferred,) = expand(build_world(sim, 1, "observed", "lookahead:g:0"), decoded.action(i))
    key = preferred.core.state_key()
    gap = float(np.log(decoded.priors[base]) - np.log(decoded.priors[i]))
    assert gap > 0

    def prefers(core, side):
        return 1.0 if int(core.state_key()) == key else -1.0

    for c, flips in ((gap / 2 + 0.01, True), (max(gap / 2 - 0.01, 0.0), False)):
        la = LookaheadPlayer(_raw(policy), _cfg(k=len(decoded), c=c, sigma=1e-6), _Stub(prefers))
        played = _decide(la, _sim())
        assert (played == decoded.action(i)) is flips, (c, gap)
        assert (played == decoded.action(base)) is not flips


def test_the_clipped_target():
    priors = np.array([0.5, 0.3, 0.2])
    q = np.array([0.0, 1.0, np.nan])
    # V = (0.5 * 0 + 0.3 * 1) / 0.8; the NaN candidate keeps its prior score.
    pick, v = tilted_choice(priors, q, sigma=1.0, c=10.0)
    assert math.isclose(v, 0.375) and pick == 1
    assert tilted_choice(priors, q, sigma=1.0, c=0.0)[0] == 0
    gap = math.log(0.5 / 0.3)
    assert tilted_choice(priors, q, sigma=1e-9, c=gap / 2 + 1e-6)[0] == 1
    assert tilted_choice(priors, q, sigma=1e-9, c=gap / 2 - 1e-6)[0] == 0
    assert tilted_choice(np.array([0.4, 0.4]), np.array([0.0, 0.0]), 1.0, 1.0)[0] == 0, "ties go first"


def test_attacks_only_keeps_the_argmax_and_the_attacks():
    from tools.raw_player import KIND_OF_TYPE, Decoded
    kinds = np.array([KIND_OF_TYPE[t] for t in ("move", "attack", "recruit", "attack", "end_turn")])
    decoded = Decoded(np.array([0.3, 0.1, 0.25, 0.05, 0.3]), kinds == 3, np.zeros(5, dtype=np.int64),
                      kinds, lambda i: {})
    assert candidate_indices(decoded, 0, 3, "all") == [0, 2, 4]
    assert candidate_indices(decoded, 0, 4, "attacks") == [0, 1]
    assert candidate_indices(decoded, 0, 5, "attacks") == [0, 1, 3]


def _critic_checkpoint(path: Path, view: str = "obs") -> Path:
    from wesnoth_ai.critic import build_critic
    torch.manual_seed(1)
    arch = {"d_model": 32, "num_layers": 1, "num_heads": 2, "d_ff": 64}
    encoder, model = build_critic(arch, vocab_of(TYPES), FACTION_IDS, torch.device("cpu"))
    torch.save({"arch": arch, "model_state": model.state_dict(), "encoder_state": encoder.state_dict(),
                "unit_type_to_id": dict(encoder.unit_type_to_id), "faction_to_id": dict(encoder.faction_to_id),
                "critic": {"view": view}}, path)
    return path


def test_the_end_turn_state_is_valued_for_the_deciding_side(tmp_path):
    """After end_turn the opponent moves: the critic's value of that state
    is the opponent's, and the deciding side reads it negated."""
    from wesnoth_ai.critic import critic_values
    from wesnoth_ai.critic_data import encode_view
    from wesnoth_ai.lookahead_evaluators import CriticEvaluator
    ckpt = _critic_checkpoint(tmp_path / "critic.pt")
    with pytest.raises(ValueError, match="view"):
        CriticEvaluator(str(ckpt), "true")
    critic = CriticEvaluator(str(ckpt), "obs")
    world = build_world(_sim(), 1, "observed", "t")
    (after_end,) = expand(world, {"type": "end_turn"})
    (after_move,) = expand(world, {"type": "move", "start_hex": Position(8, 3), "target_hex": Position(8, 4)})
    assert (after_end.side_to_move, after_move.side_to_move) == (2, 1)
    mover_value = [critic_values(critic.encoder, critic.model,
                                 [encode_view(o.core, "obs", critic.type_to_id, critic.faction_to_id)],
                                 critic.device)[0] for o in (after_end, after_move)]
    assert abs(mover_value[0]) > 1e-6
    got = critic.values([after_end, after_move], 1)
    assert np.allclose(got, [-mover_value[0], mover_value[1]], atol=1e-6)
    assert np.allclose(critic.values([after_end], 2), [mover_value[0]], atol=1e-6)
    material = MaterialEvaluator(100.0)
    assert np.allclose(material.values([after_end], 1), -material.values([after_end], 2))


def test_the_memory_advances_once_per_decision_on_the_played_state():
    """The look-ahead forwards the policy once per decision, on the state
    the game is in, never on a candidate's: its memory equals a raw
    player's fed the same states, even when it plays other actions."""
    from tools.elo_eval_game import _CountingModel
    policy = _policy()
    counter = _CountingModel(policy._inference_model)
    policy._inference_model = counter
    sim = _sim()
    la = LookaheadPlayer(_raw(policy), _cfg(c=5.0, sigma=1e-3), MaterialEvaluator(50.0))
    ref = _raw(_policy())
    decisions = 0
    while not sim.done and decisions < 25:
        from wesnoth_ai.game_core import snapshot_view
        state = snapshot_view(sim.gs)
        side = int(sim.current_side)
        ref.advance_memory(state, game_label="g")
        sim.step(la.select_action(state, game_label="g", sim=sim))
        decisions += 1
        assert torch.equal(la._raw.memory_of("g", side), ref.memory_of("g", side))
    assert counter.n_forwards == decisions
    assert la.telemetry("g")["flips"] > 0, "the operator played other actions than the prior's argmax"


def test_procedure_tags_and_the_guards(tmp_path):
    from tools.elo_eval_game import main
    from tools.eval_procedure import godview_refusal, lookahead_refusal, procedure_of
    from tools.eval_provenance import lookahead_record_of
    from wesnoth_ai.constants import OBSERVATION_EPOCH
    from tests.helpers.eval_records import current_forced_faction
    cfg = _cfg(k=8, c=1.0, sigma=0.1)
    assert procedure_tag(cfg, OFFSET) == "la:material:k8c1s0.1+eo-1.5"
    assert procedure_of(0, False, False, 0.0, raw_end_turn_offset=OFFSET, lookahead=cfg) \
        == "la:material:k8c1s0.1+eo-1.5"
    assert procedure_tag(_cfg(k=8, kinds="attacks", determinization="godview")) == "la:material:k8c1s0.1+atk+godview"
    assert lookahead_refusal("a", "x.pt", 0, 0.5, "joint") is not None
    assert lookahead_refusal("a", "x.pt", 0, 0.0, "joint") is None
    fair, god = "la:material:k8c1s0.1", "la:material:k8c1s0.1+godview"
    assert godview_refusal({(god, "raw:t0")}) is None, "a god-view match is a match of its own"
    assert godview_refusal({(god, "raw:t0"), (fair, "raw:t0")}) is not None

    configs = {}
    for name, c in (("one", 1.0), ("two", 2.0)):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"k": 8, "c": c, "sigma": 0.1}), encoding="utf-8")
        configs[name] = path
    _, rec = lookahead_record_of(configs["one"])
    assert rec == config_record(_cfg(k=8, c=1.0, sigma=0.1))
    out = tmp_path / "out"
    out.mkdir()
    (out / "game_A_B_s1_7.json").write_text(json.dumps({
        "procedure_a": "la:material:k8c1s0.1", "procedure_b": "raw", "max_turns": 200,
        "combat_stream": "per_game", "observation_epoch": OBSERVATION_EPOCH,
        "forced_faction": current_forced_faction(), "terrain_a": "set",
        "checkpoint_sha256_a": None, "checkpoint_sha256_b": None,
        "lookahead_a": rec, "lookahead_b": None}), encoding="utf-8")
    argv = ["x", "A", "random", "B", "dummy", "1", "7", str(out), "--mcts-sims", "0",
            "--raw-temperature-a", "0", "--device", "cpu"]
    assert main(argv + ["--lookahead-a", str(configs["one"])]) == 0
    with pytest.raises(SystemExit, match="refusing to mix"):
        main(argv + ["--lookahead-a", str(configs["two"])])
    with pytest.raises(SystemExit, match="refusing to mix"):
        main(argv)
    with pytest.raises(SystemExit, match="argmax"):
        main(argv[:-4] + ["--raw-temperature-a", "0.5", "--device", "cpu",
                          "--lookahead-a", str(configs["one"])])


def test_the_collector_keeps_look_ahead_configurations_apart(tmp_path):
    from tools.elo_collect import dir_estimands
    one = {"lookahead_a": {"k": 8}, "lookahead_b": None}
    assert dir_estimands([one, dict(one)])["lookahead_a"] == {"k": 8}
    with pytest.raises(SystemExit, match="lookahead_a"):
        dir_estimands([one, {"lookahead_a": {"k": 4}, "lookahead_b": None}])


def test_a_rollout_evaluator_is_an_interface_only():
    from wesnoth_ai.lookahead_evaluators import build_evaluator
    with pytest.raises(NotImplementedError):
        build_evaluator({"name": "rollout"})
