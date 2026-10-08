"""Step 1's free readouts (tools/prior_gaps.py): the reference's priors at a
recorded decision, read with its memory as in play, and what is read off
them."""
from __future__ import annotations

import math
import sys
import threading
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import prior_gaps as pg  # noqa: E402
from tools.raw_player import RawPolicyPlayer  # noqa: E402
from wesnoth_ai import game_core as gc  # noqa: E402
from wesnoth_ai.classes import Position  # noqa: E402

needs_core = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")


def _base():
    from helpers.parity_games import vocab_of
    from wesnoth_ai.encoder import GameStateEncoder
    from wesnoth_ai.model import WesnothModel
    torch.manual_seed(0)
    encoder = GameStateEncoder(d_model=16, unit_type_to_id=vocab_of(["Lieutenant", "Spearman"]),
                               relevant_set_hexes=True, fog_hides_enemy_villages=True, terrain_multi_hot=True,
                               observation_parity=True, relevant_set_version=2).eval()
    encoder.freeze_vocab()
    model = WesnothModel(d_model=16, num_layers=1, num_heads=2, d_ff=32, observation_parity=True,
                         memory_slots=4).eval()
    return NS(_inference_model=model, _inference_encoder=encoder, _lock=threading.Lock(), _decision_step=0)


def _position():
    from helpers.parity_games import core_of, record
    cs = core_of(record([("Lieutenant", 1, 1, 3, True), ("Spearman", 1, 5, 3, False),
                         ("Lieutenant", 2, 18, 3, True), ("Spearman", 2, 6, 3, False)], fog=True))
    cs.apply_command(["init_side", 1])
    gs = cs.to_state()
    gc.bind_view(gs, cs.fork())
    return gs


@needs_core
def test_the_priors_are_the_decodes_and_the_memory_goes_on_as_in_play():
    base = _base()
    gs = _position()
    shifted = RawPolicyPlayer(base, 0.0, end_turn_offset=-1.5, memory_slots=4)
    plain = RawPolicyPlayer(base, 0.0, end_turn_offset=0.0, memory_slots=4)
    legal, priors = shifted.legal_priors(gs, game_label="g")
    _, raw = plain.legal_priors(gs, game_label="g")
    assert len(legal) > 2 and priors.sum() == pytest.approx(1.0) and raw.sum() == pytest.approx(1.0)
    end = [i for i, la in enumerate(legal) if la.action["type"] == "end_turn"]
    assert len(end) == 1
    p, q = raw[end[0]], priors[end[0]]
    assert math.log(q / (1 - q)) == pytest.approx(math.log(p / (1 - p)) - 1.5, abs=1e-9)
    first = shifted.memory_of("g", 1)
    assert first is not None
    _, again = shifted.legal_priors(gs, game_label="g")
    assert not torch.equal(shifted.memory_of("g", 1), first), "the second decision read and wrote the first's"
    assert not np.allclose(again, priors), "the same position reads differently with a memory carried"
    shifted.drop_pending("g")
    assert shifted.memory_of("g", 1) is None


@needs_core
def test_a_recorded_game_is_read_at_each_decision_of_the_players_side():
    from helpers.critic_games import played_record
    rec = played_record(max_turns=3)
    player = RawPolicyPlayer(_base(), 0.0, end_turn_offset=-1.5, memory_slots=4)
    readings, tally = pg.read_game(rec, player, 1, "g")
    side, own = 0, 0
    for cmd in rec["commands"]:
        side = int(cmd[1]) if cmd[0] == "init_side" else side
        own += int(side == 1 and cmd[0] in pg.DECISION_KINDS)
    assert tally["decisions"] == len(readings) == own - tally["premoves"] > 0
    assert all(r["legal"] >= 1 for r in readings)
    assert player.memory_of("g", 1) is None, "the game's memory goes with it"


def _legal(*kinds):
    return [NS(action={"type": k}) for k in kinds]


def test_a_decisions_reading_holds_the_gaps_and_the_attacks_in_its_top_eight():
    legal = _legal("attack", "move", "attack", "end_turn", *["move"] * 7, "attack")
    priors = np.array([0.30, 0.20, 0.15, 0.10] + [0.03] * 7 + [0.04])
    r = pg.decision_reading(legal, priors)
    assert r["legal"] == 12 and r["top_type"] == "attack" and len(r["gaps"]) == 7
    assert r["gaps"][0] == pytest.approx(math.log(0.30 / 0.20))
    assert r["attacks_top"] == 3, "the last attack ranks fifth"
    assert pg.decision_reading([], np.zeros(0))["legal"] == 0


def test_the_clip_shares_read_the_gap_to_the_second_action():
    readings = [{"legal": 3, "gaps": [g, g + 1], "attacks_top": a, "top_type": "move"}
                for g, a in ((0.5, 2), (1.5, 0), (3.0, 1), (5.0, 2))]
    readings.append({"legal": 1, "gaps": [], "attacks_top": 0, "top_type": "end_turn"})
    s = pg.summarize(readings)
    assert s["decisions"] == 5 and s["decisions_multi"] == 4 and s["single_legal"] == 1
    assert s["flip_possible"] == {"c=0.5": 0.25, "c=1.0": 0.5, "c=2.0": 0.75}
    assert s["two_or_more_attacks_in_top8"] == 0.5
    assert s["gaps"]["top1_minus_2"]["n"] == 4


def test_an_attack_from_a_distance_is_one_decision_and_matches_its_two_commands():
    move = ["move", [3, 4, 5], [2, 2, 3], 1]
    attack = ["attack", 5, 3, 6, 3, 1, 0, "seed"]
    assert pg.is_premove(move, attack)
    assert not pg.is_premove(move, ["attack", 4, 2, 6, 3, 1, 0, "s"])
    chosen = {"type": "attack", "start_hex": Position(3, 2), "target_hex": Position(6, 3), "attack_index": 1}
    assert pg.matches(chosen, move, attack)
    assert not pg.matches(dict(chosen, attack_index=0), move, attack)
    assert pg.matches({"type": "move", "start_hex": Position(3, 2), "target_hex": Position(5, 3)}, move, attack)
    assert pg.matches({"type": "recruit", "unit_type": "Spearman", "target_hex": Position(2, 3)},
                      ["recruit", "Spearman", 2, 3, "s"], None)
    assert pg.matches({"type": "end_turn"}, ["end_turn"], None)
    assert not pg.matches(chosen, ["end_turn"], None)


def test_the_player_is_found_by_its_label():
    result = {"label_a": "cand64", "label_b": "parity2", "side_a": 2, "checkpoint_sha256_a": "x",
              "checkpoint_sha256_b": "y"}
    assert pg.player_side(result, "cand64") == (2, "x")
    assert pg.player_side(result, "parity2") == (1, "y")
    with pytest.raises(ValueError):
        pg.player_side(result, "obs8")
