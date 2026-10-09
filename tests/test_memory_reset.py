"""The raw player's per-turn memory reset (procedure tag '+mr',
docs/memory_in_play_parity3_prereg_20261009.md): each of a side's turns
starts from the learned initial memory and the memory is carried within
the turn only. Off, the player carries the memory from each decision of
the side to its next, as before the option existed."""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.eval_procedure import memory_reset_refusal, procedure_of  # noqa: E402
from tools.raw_player import RawPolicyPlayer  # noqa: E402
from wesnoth_ai import game_core as gc  # noqa: E402

needs_core = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")
D = 32


class _SpyModel:
    """Answers every decision with end_turn and a memory that names the call."""

    def __init__(self):
        self.seen = []

    def __call__(self, encoded, memory=None):
        self.seen.append(memory)
        state = torch.full((memory.k, 4), float(len(self.seen)))
        return SimpleNamespace(legal_compact=SimpleNamespace(prior=[], kind=[], actor=[]), memory=state)


def _player(reset: bool):
    spy = _SpyModel()
    base = SimpleNamespace(_inference_model=spy, _inference_encoder=SimpleNamespace(encode=lambda gs: gs),
                           _lock=threading.Lock(), _decision_step=0)
    return RawPolicyPlayer(base, 0.0, memory_slots=3, memory_reset_each_turn=reset), spy


def _state(side, turn):
    return SimpleNamespace(global_info=SimpleNamespace(current_side=side, turn_number=turn))


def _read(spy):
    return [None if m.state is None else float(m.state[0, 0]) for m in spy.seen]


SEQUENCE = ((1, 1), (1, 1), (2, 1), (1, 2), (1, 2), (2, 2), (2, 2), (1, 3))


@pytest.mark.parametrize("reset, expected", [
    (False, [None, 1.0, None, 2.0, 4.0, 3.0, 6.0, 5.0]),
    (True, [None, 1.0, None, None, 4.0, None, 6.0, None]),
], ids=["carried", "reset each turn"])
def test_each_side_turn_starts_from_the_initial_memory_under_the_reset(reset, expected):
    player, spy = _player(reset)
    for side, turn in SEQUENCE:
        player.select_action(_state(side, turn), game_label="g")
    assert _read(spy) == expected


def test_a_refused_first_decision_of_a_turn_is_decided_again_from_the_initial_memory():
    player, spy = _player(True)
    player.select_action(_state(1, 1), game_label="g")
    player.select_action(_state(1, 2), game_label="g")
    player.drop_last_pending("g")
    player.select_action(_state(1, 2), game_label="g")
    player.select_action(_state(1, 2), game_label="g")
    assert _read(spy) == [None, None, None, 3.0]


def test_the_reset_is_refused_without_a_memory_and_named_in_the_procedure():
    base = SimpleNamespace(_inference_model=SimpleNamespace(memory_slots=8), _inference_encoder=None,
                           _lock=threading.Lock(), _decision_step=0)
    with pytest.raises(ValueError, match="memory slots"):
        RawPolicyPlayer(base, 0.0, memory_slots=0, memory_reset_each_turn=True)
    player = RawPolicyPlayer(base, 0.0, memory_slots=8, memory_reset_each_turn=True)
    with pytest.raises(ValueError, match="per-turn"):
        player.set_memory("g", 1, torch.zeros(8, D))
    assert memory_reset_refusal("a", 0, 0.0, 64, True) is None
    assert memory_reset_refusal("a", 0, 0.0, 0, False) is None
    for sims, temp, memory in ((32, None, 64), (0, None, 64), (0, 0.0, 0), (0, 0.0, None)):
        assert memory_reset_refusal("a", sims, temp, memory, True) is not None
    assert procedure_of(0, False, False, 0.0, raw_end_turn_offset=-1.5) == "raw:t0+eo-1.5"
    assert procedure_of(0, False, False, 0.0, raw_end_turn_offset=-1.5, memory_reset=True) == "raw:t0+eo-1.5+mr"


def _memory_checkpoint(tmp_path, slots=8):
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(1)
    spec = tmp_path / "memory.pt"
    TransformerPolicy(device=torch.device("cpu"), d_model=D, num_layers=1, num_heads=2, d_ff=64,
                      relevant_set_hexes=True, observation_parity=True, memory_slots=slots,
                      relevant_set_version=2).save_checkpoint(spec)
    return spec


def test_a_match_refuses_the_reset_for_a_side_without_memory_slots(tmp_path):
    from tools.elo_eval_game import main
    spec = _memory_checkpoint(tmp_path)
    with pytest.raises(SystemExit, match="memory slots"):
        main(["x", "A", str(spec), "B", str(spec), "1", "7", str(tmp_path / "games"), "--mcts-sims", "0",
              "--raw-temperature-a", "0", "--raw-temperature-b", "0", "--memory-a", "0",
              "--memory-reset-a", "--max-turns", "2", "--device", "cpu"])


@needs_core
@pytest.mark.slow
def test_a_real_game_feeds_the_reset_side_the_initial_memory_at_each_of_its_turns(tmp_path, monkeypatch):
    """One game of the checkpoint at 8 slots with the reset (A) against
    itself at 8 slots without it (B), deep into each turn (offset -6).
    Every forward's memory is checked against the rule of its player: B
    reads what its side's previous decision wrote, A the same within a
    turn and the learned initial memory at each turn's first decision. The
    result and the game record name the reset, and the outdir refuses the
    plain decode."""
    from tools import raw_player
    from tools.elo_eval_game import main
    from tools.game_record import read_records
    log = []
    real_forward = raw_player.RawPolicyPlayer._forward

    def forward(self, base, encoded, game_label, game_state):
        real_model, seen = base._inference_model, {}

        def model(enc, memory=None):
            seen["memory"] = memory
            return real_model(enc, memory=memory)
        base._inference_model = model
        try:
            output = real_forward(self, base, encoded, game_label, game_state)
        finally:
            base._inference_model = real_model
        gi = game_state.global_info
        log.append(("forward", self.memory_reset_each_turn, int(gi.current_side), int(gi.turn_number),
                    seen["memory"], output.memory, encoded, base))
        return output

    def drop_last(self, game_label):
        log.append(("refused", self.memory_reset_each_turn))
        return real_drop(self, game_label)

    real_drop = raw_player.RawPolicyPlayer.drop_last_pending
    monkeypatch.setattr(raw_player.RawPolicyPlayer, "_forward", forward)
    monkeypatch.setattr(raw_player.RawPolicyPlayer, "drop_last_pending", drop_last)
    spec = _memory_checkpoint(tmp_path)
    out = tmp_path / "games"
    common = ["x", "A", str(spec), "B", str(spec), "1", "7", str(out), "--mcts-sims", "0",
              "--raw-temperature-a", "0", "--raw-temperature-b", "0", "--memory-a", "8", "--memory-b", "8",
              "--raw-end-turn-offset-a", "-6", "--raw-end-turn-offset-b", "-6", "--max-turns", "3",
              "--device", "cpu"]
    assert main(common + ["--memory-reset-a"]) == 0
    across = {True: 0, False: 0}             # decisions whose side's last write was in an earlier turn
    for reset in (True, False):
        carried = carried_turn = None        # the side's last write and its turn
        before = (None, None)                # the same before the last decision, for a refusal
        for entry in [e for e in log if e[1] == reset]:
            if entry[0] == "refused":
                carried, carried_turn = before
                continue
            _, _, side, turn, memory, written, encoded, base = entry
            assert memory.k == 8
            new_turn = carried_turn is not None and carried_turn != turn
            across[reset] += new_turn
            if reset and new_turn:
                assert memory.state is None, f"turn {turn}: the reset side's first decision reads the initial memory"
                with torch.no_grad():
                    explicit = base._inference_model(encoded, memory=raw_player.MemoryState(
                        8, base._inference_model._inner.initial_memory(8)))
                assert torch.allclose(explicit.memory, written, atol=1e-6)
            elif carried is None:
                assert memory.state is None
            else:
                assert torch.equal(memory.state, carried), f"turn {turn}: the memory carried from the last decision"
            before, carried, carried_turn = (carried, carried_turn), written, turn
    assert across[True] >= 2 and across[False] >= 2, across
    result = json.loads((out / "game_A_B_s1_7.json").read_text(encoding="utf-8"))
    assert (result["memory_reset_a"], result["memory_reset_b"]) == (True, False)
    assert (result["procedure_a"], result["procedure_b"]) == ("raw:t0+eo-6+mr", "raw:t0+eo-6")
    game = next(read_records(out / "game_A_B_s1_7.game.jsonl.gz"))
    assert game["players"]["a"]["procedure"] == "raw:t0+eo-6+mr"
    with pytest.raises(SystemExit, match="procedure"):
        main(common)
