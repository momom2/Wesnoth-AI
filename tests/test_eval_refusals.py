"""A match game does not play on after the simulator refuses an action the
legality mask offered (2026-09-29 audit): the argmax player would repeat it,
losing turns or hanging until the per-game timeout, and the verdict would
count the damage silently. Scripted players pick without the mask and are
refused by design; their games go on under the simulator's loop guard.
Every result records the draw (factions, leaders) and the turns the guard
ended, and the reference player is pinned by its SHA-256."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class _Doomed:
    """A policy that always orders its leader onto its own hex."""
    consults_legality_mask = True

    def select_action(self, gs, *, game_label="default", sim=None):
        from wesnoth_ai.classes import Position
        side = gs.global_info.current_side
        leader = next(u for u in gs.map.units if u.side == side and u.is_leader)
        return {"type": "move", "start_hex": leader.position,
                "target_hex": Position(x=leader.position.x, y=leader.position.y)}

    def drop_pending(self, game_label):
        pass


def _sim():
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate, random_setup
    setup = random_setup(random.Random(5), forced_faction=None, mini_maps=False, category="fogless")
    return WesnothSim(build_scenario_gamestate(setup), scenario_id=setup.scenario_id, max_turns=2)


def test_a_refused_masked_action_fails_the_game():
    from tools.eval_players import _PolicyPair, _play_one_eval_game
    with pytest.raises(RuntimeError, match="mask/simulator disagreement"):
        _play_one_eval_game(_sim(), _PolicyPair(_Doomed(), "A", 1), _PolicyPair(_Doomed(), "B", 2),
                            game_label="g")


def test_a_scripted_players_refusals_end_its_turns_instead():
    from tools.eval_players import _PolicyPair, _play_one_eval_game
    scripted = _Doomed()
    scripted.consults_legality_mask = False
    sim = _sim()
    _play_one_eval_game(sim, _PolicyPair(scripted, "A", 1), _PolicyPair(scripted, "B", 2), game_label="g")
    assert sim.forced_end_turns.get(1, 0) >= 1 and sim.forced_end_turns.get(2, 0) >= 1


def test_a_result_records_the_draw_and_the_guard(tmp_path):
    from tools.elo_eval_game import main
    out = tmp_path / "games"
    assert main(["x", "A", "dummy", "B", "dummy", "1", "7", str(out),
                 "--mcts-sims", "0", "--max-turns", "2", "--device", "cpu"]) == 0
    result = json.loads((out / "game_A_B_s1_7.json").read_text(encoding="utf-8"))
    from wesnoth_ai.constants import DEFAULT_FACTIONS
    assert result["faction_a"] in DEFAULT_FACTIONS and result["faction_b"] in DEFAULT_FACTIONS
    assert result["leader_a"] and result["leader_b"]
    assert result["forced_end_turns_a"] >= 0 and result["max_actions_per_side"] == 2000
    import wesnoth_ai
    assert result["code_version"] == wesnoth_ai.__version__ and isinstance(result["rust_core"], bool)


def test_the_reference_is_pinned_by_its_hash(tmp_path):
    from tools import reference_player
    ref = dict(reference_player.load())
    bogus = tmp_path / "obs8.pt"
    bogus.write_bytes(b"not the adopted checkpoint")
    with pytest.raises(SystemExit, match="not the adopted reference"):
        reference_player.verify_checkpoint(bogus, ref)
    from types import SimpleNamespace

    from tools.run_elo_batch import _refuse_a_changed_reference
    args = SimpleNamespace(label_a="cand", spec_a="cand.pt", label_b=ref["label"], spec_b=str(bogus))
    with pytest.raises(SystemExit, match="names the reference player"):
        _refuse_a_changed_reference(args, {"cand.pt": "x", str(bogus): "y"})
    _refuse_a_changed_reference(args, {"cand.pt": "x", str(bogus): ref["checkpoint_sha256"]})
