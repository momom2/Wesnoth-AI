"""The sequence trainer (tools/sequence_train.py): the streams walk every
decision once in order and resume where they stood; a step's labels train
the policy on the winners' decisions only and never at a position that
names no action; the belief loss averages over the tokens with no visible
unit; and a pass cut and resumed ends with the weights of the uncut pass."""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.replay_dataset import ActionIndices, timeout_label  # noqa: E402
from wesnoth_ai.imitation_loss import build_imitation_targets  # noqa: E402
from wesnoth_ai.sequence_loss import (GameFacts, SequenceLabelError, belief_loss,  # noqa: E402
                                      step_labels)
from wesnoth_ai.sequence_streams import (K_CHOICES, GameSide, Step, StreamSchedule,  # noqa: E402
                                         epoch_order, memory_size)


def _schedule(n_sides=9, n_streams=3, seed=5):
    lengths = {GameSide(f"g{i}", s): (i * 7 + s * 3) % 11 for i in range(n_sides) for s in (1, 2)}
    return StreamSchedule(epoch_order(list(lengths), seed), lengths, n_streams, seed), lengths


def _walk(schedule, T):
    out = []
    while not schedule.exhausted():
        for steps in schedule.window(T):
            out.extend(steps)
    return out


def test_the_streams_walk_every_decision_once_in_order_and_resume_where_they_stood():
    schedule, lengths = _schedule()
    steps = _walk(schedule, T=4)
    seen = Counter((s.game_side, s.offset) for s in steps)
    assert set(seen.values()) == {1}
    assert sorted(seen) == sorted((g, o) for g, n in lengths.items() for o in range(n))
    by_side = {}
    for s in steps:
        by_side.setdefault(s.game_side, []).append(s)
    for side_steps in by_side.values():
        assert [s.offset for s in side_steps] == list(range(len(side_steps)))
        assert [s.starts for s in side_steps] == [True] + [False] * (len(side_steps) - 1)
        assert len({s.slot for s in side_steps}) == 1, "a game-side stays on its slot"
    cut, _ = _schedule()
    head = [s for steps in cut.window(4) + cut.window(4) for s in steps]
    resumed, _ = _schedule()
    resumed.load_state_dict(json.loads(json.dumps(cut.state_dict())))
    assert head + _walk(resumed, T=4) == steps


def test_half_the_game_sides_train_the_whole_memory():
    ks = Counter(memory_size(20260929, GameSide(f"g{i}", 1 + i % 2)) for i in range(8000))
    assert set(ks) == set(K_CHOICES)
    assert abs(ks[64] / 8000 - 0.5) < 0.03 and abs(ks[0] / 8000 - 0.125) < 0.02


def _position(label, hidden=(), no_visible=(True, True, False, True)):
    return SimpleNamespace(label=label, hidden_tokens=np.array(hidden, dtype=np.int64),
                           no_visible_unit=np.array(no_visible, dtype=bool))


def test_a_steps_labels_train_the_policy_on_the_winners_decisions_only():
    games = {"g": GameFacts(winner=1, n_commands=8, policy_weight=0.5)}
    move = ActionIndices(action_type="move", actor_idx=1, target_idx=2, type_idx=1)
    steps = [Step(0, GameSide("g", 1), 0, 8, True), Step(1, GameSide("g", 2), 0, 8, True),
             Step(2, GameSide("g", 1), 1, 8, False)]
    positions = [_position(move, hidden=[3]), _position(move), _position(timeout_label())]
    labels = step_labels(positions, steps, [(2, 1, 4)] * 3, games, seed=1, value_states_per_game=16,
                         value_weight=1.0)
    assert [pw for _, _, pw in labels.zw] == [0.5, 0.0, 0.0], "winner, loser, a turn that ran out"
    assert [z for z, _, _ in labels.zw] == [1, -1, 1]
    assert [vw for _, vw, _ in labels.zw] == [1.0, 1.0, 1.0], "8 commands: every state is a value state"
    assert labels.belief_target[0].tolist() == [0, 0, 0, 1] and labels.belief_mask[0].tolist() == [1, 1, 0, 1]
    wrong = _position(ActionIndices(action_type="move", actor_idx=9, target_idx=0, type_idx=1))
    with pytest.raises(SequenceLabelError):
        step_labels([wrong], steps[:1], [(2, 1, 4)], games, seed=1, value_states_per_game=16, value_weight=1.0)


def test_a_turn_that_ran_out_trains_the_value_and_no_policy():
    targets = build_imitation_targets([timeout_label()], [(1, 1.0, 1.0)], [(2, 1, 4)], n_types=2,
                                      n_weapons=3, n_atoms=51, type_loss_weights={}, device=torch.device("cpu"))
    assert targets.ok["actor"][0] and targets.ok["value"][0]
    assert targets.policy_w[0] == 0.0


def test_the_belief_loss_averages_over_the_tokens_with_no_visible_unit():
    logits = torch.tensor([[0.0, 2.0, -1.0, 5.0]])
    target = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    expected = (np.log(2.0) + np.log1p(np.exp(-2.0))) / 2
    assert belief_loss(logits, target, mask).item() == pytest.approx(expected, rel=1e-6)
    assert belief_loss(logits, target, torch.zeros_like(mask)).item() == 0.0


# ---- a whole pass, cut and resumed --------------------------------------

def _dataset(tmp_path):
    """Three small games written from scratch, extracted into a corpus
    directory with its manifest; the third is held out."""
    from helpers.synthetic_replay import move, turn, two_sides, write_replay
    from tools.replay_extract import extract_replay
    rides = [[(3, 5), (3, 6)], [(3, 5), (4, 5)], [(3, 5), (3, 4)]]
    dataset = tmp_path / "corpus"
    dataset.mkdir()
    rows = []
    for i, ride in enumerate(rides):
        commands = [*turn(1, move(1, ride)), *turn(2, move(2, [(11, 4), (11, 5)])),
                    *turn(1), *turn(2)]
        rec = extract_replay(write_replay(tmp_path / f"g{i}.bz2",
                                          two_sides(extra1=[("Cavalryman", 3, 5, False)]), commands))
        name = f"g{i}.json.gz"
        with gzip.open(dataset / name, "wt", encoding="utf-8") as f:
            json.dump(rec, f)
        rows.append({"file": name, "winner_side": 1 + i % 2, "winner_actions": 3,
                     "holdout": i == 2, "corpus_version": 4})
    (dataset / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return dataset


@pytest.mark.slow
def test_a_resumed_pass_ends_with_the_weights_of_the_uncut_pass(tmp_path):
    from wesnoth_ai import game_core as gc
    if gc.game_core_class() is None:
        pytest.skip("wesnoth_core.GameCore not available")
    from tools import preencode_sequences, sequence_train
    dataset = _dataset(tmp_path)
    sequences = tmp_path / "sequences"
    assert preencode_sequences.main(["--dataset", str(dataset), "--out", str(sequences),
                                     "--workers", "1", "--log-level", "WARNING"]) == 0
    common = ["--sequences", str(sequences), "--dataset", str(dataset), "--device", "cpu",
              "--streams", "2", "--window", "3", "--warmup-steps", "2", "--probe-every", "1000000",
              "--barrier-positions", "1000000", "--checkpoint-every", "1000000", "--log-level", "WARNING",
              "--probe-ks", "0,8", "--signal-every", "3",
              "--d-model", "32", "--num-layers", "1", "--num-heads", "2", "--d-ff", "64"]
    uncut, cut = tmp_path / "uncut.pt", tmp_path / "cut.pt"
    assert sequence_train.main([*common, "--out", str(uncut)]) == 0
    assert sequence_train.main([*common, "--out", str(cut), "--max-positions", "4"]) == 0
    mid = torch.load(cut, map_location="cpu", weights_only=True)["sequence_resume"]["state"]["positions"]
    assert 4 <= mid < 12
    assert sequence_train.main([*common, "--out", str(cut), "--resume"]) == 0
    a = torch.load(uncut, map_location="cpu", weights_only=True)
    b = torch.load(cut, map_location="cpu", weights_only=True)
    assert a["sequence_resume"]["state"]["positions"] == b["sequence_resume"]["state"]["positions"] == 12
    for key in ("model_state", "encoder_state"):
        for name, tensor in a[key].items():
            assert torch.equal(tensor, b[key][name]), name
    probe = [json.loads(line) for line in uncut.with_suffix(".probe.jsonl").read_text().splitlines()]
    assert probe[-1]["k0"]["n_positions"] == probe[-1]["k8"]["n_positions"] > 0
    # The telemetry rows leave the training untouched (the weights above
    # match with rows written in both runs) and split the gradient by term.
    signal = [json.loads(line) for line in uncut.with_suffix(".signal.jsonl").read_text().splitlines()]
    assert signal and set(signal[-1]["gradient"]) == {"encoder", "trunk", "heads", "memory", "all"}
    shares = [signal[-1]["gradient"]["all"][t]["share"] for t in
              ("actor", "type", "target", "weapon", "value", "belief")]
    assert sum(x for x in shares if x is not None) == pytest.approx(1.0, abs=1e-6)


def test_the_holdout_ce_of_a_checkpoint_without_a_memory_reads_every_holdout_decision(tmp_path):
    from wesnoth_ai import game_core as gc
    if gc.game_core_class() is None:
        pytest.skip("wesnoth_core.GameCore not available")
    from tools.holdout_ce import holdout_ce
    from wesnoth_ai.transformer_policy import TransformerPolicy
    dataset = _dataset(tmp_path)
    spec = tmp_path / "net.pt"
    TransformerPolicy(device=torch.device("cpu"), d_model=32, num_layers=1, num_heads=2, d_ff=64,
                      relevant_set_hexes=True).save_checkpoint(spec)
    result = holdout_ce(spec, dataset, torch.device("cpu"))
    assert result["n_games"] == 1 and result["skipped_games"] == 0
    assert result["n_decisions"] == 6, "both sides' move and two end_turns of the held-out game"
    assert result["ce_all"] > 0 and result["ce_winners"] > 0
