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


@pytest.fixture(scope="module")
def pass_inputs(tmp_path_factory):
    """The three games pre-encoded by the script run by path, as the box
    runs it (its worker processes pickle the records, which this process
    must be able to read), and the trainer's arguments for a tiny network."""
    from wesnoth_ai import game_core as gc
    if gc.game_core_class() is None:
        pytest.skip("wesnoth_core.GameCore not available")
    import subprocess
    tmp_path = tmp_path_factory.mktemp("pass")
    dataset = _dataset(tmp_path)
    sequences = tmp_path / "sequences"
    root = Path(__file__).resolve().parent.parent
    subprocess.run([sys.executable, str(root / "tools" / "preencode_sequences.py"), "--dataset", str(dataset),
                    "--out", str(sequences), "--workers", "1", "--log-level", "WARNING"],
                   cwd=root, check=True, timeout=600)
    common = ["--sequences", str(sequences), "--dataset", str(dataset), "--device", "cpu",
              "--streams", "2", "--window", "3", "--warmup-steps", "2", "--checkpoint-every", "1000000",
              "--log-level", "WARNING", "--probe-ks", "0,8", "--signal-every", "3",
              "--d-model", "32", "--num-layers", "1", "--num-heads", "2", "--d-ff", "64"]
    return common


@pytest.mark.slow
def test_a_resumed_pass_ends_with_the_weights_of_the_uncut_pass(tmp_path, pass_inputs):
    from tools import sequence_train
    common = [*pass_inputs, "--probe-every", "1000000", "--barrier-positions", "1000000"]
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
    assert probe[-1]["final"] and probe[-1]["positions"] == probe[-1]["total_positions"] == 12
    assert probe[-1]["k0"]["n_positions"] == probe[-1]["k8"]["n_positions"] > 0
    assert probe[-1]["belief_carried"]["n_games"] == probe[-1]["belief_paired"]["n_games"] == 1
    assert "1-5" in probe[-1]["k8"]["value_auc_by_turn"]
    # The pass's checkpoint is a match player: the policy loader reads its
    # structure and loads every weight.
    from tools.eval_players import _load_policy, peek_checkpoint_arch
    arch = peek_checkpoint_arch(uncut)
    assert (arch["observation_parity"], arch["memory_slots"], arch["relevant_set_version"]) == (True, 64, 2)
    assert arch["relevant_set_hexes"] and arch["terrain_multi_hot"] and arch["fog_hides_enemy_villages"]
    policy = _load_policy(uncut, torch.device("cpu"), label="arm")
    assert torch.equal(policy._model.slot_memory.initial, a["model_state"]["slot_memory.initial"])
    # The telemetry rows leave the training untouched (the weights above
    # match with rows written in both runs) and split the gradient by term.
    signal = [json.loads(line) for line in uncut.with_suffix(".signal.jsonl").read_text().splitlines()]
    assert signal and set(signal[-1]["gradient"]) == {"encoder", "trunk", "heads", "memory", "all"}
    shares = [signal[-1]["gradient"]["all"][t]["share"] for t in
              ("actor", "type", "target", "weapon", "value", "belief")]
    assert sum(x for x in shares if x is not None) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.slow
def test_a_failed_memory_barrier_stops_every_resume(tmp_path, pass_inputs):
    """One holdout game gives the paired difference no standard error, so
    the barrier fails at its probe; a resume stops at once, untrained."""
    from tools import sequence_train
    common = [*pass_inputs, "--probe-every", "3", "--barrier-positions", "3"]
    out = tmp_path / "arm.pt"
    assert sequence_train.main([*common, "--out", str(out)]) == sequence_train.EXIT_MEMORY_BARRIER
    before = torch.load(out, map_location="cpu", weights_only=True)["sequence_resume"]["state"]
    assert before["barrier_failed"] and before["positions"] < 12
    assert sequence_train.main([*common, "--out", str(out), "--resume"]) == sequence_train.EXIT_MEMORY_BARRIER
    after = torch.load(out, map_location="cpu", weights_only=True)["sequence_resume"]["state"]
    assert after["positions"] == before["positions"]


def test_the_same_turn_auc_compares_the_winners_first_decision_of_a_turn_with_the_losers():
    from tools.sequence_probe import same_turn_auc
    sides = {GameSide("g", 1): {"turn": [1, 1, 2, 7], "value": [0.5, -1.0, 0.2, 0.9]},
             GameSide("g", 2): {"turn": [1, 2, 2, 7], "value": [0.1, 0.3, 5.0, 0.9]}}
    out = same_turn_auc(sides, {"g": 1})
    assert out["1-5"]["auc"] == pytest.approx(0.5), "turn 1: 0.5 > 0.1; turn 2: 0.2 < 0.3"
    assert out["6-10"]["auc"] == pytest.approx(0.5), "turn 7: a tie"
    assert out["11-15"]["n_games"] == 0


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


def _trainer(pass_inputs, tmp_path, *extra):
    from tools import sequence_train
    tmp_path.mkdir(parents=True, exist_ok=True)
    args = sequence_train.parse_args([*pass_inputs, "--probe-every", "1000000", "--barrier-positions",
                                      "1000000", "--signal-every", "1000000", "--out", str(tmp_path / "arm.pt"),
                                      *extra])
    torch.manual_seed(args.seed)
    return sequence_train.Trainer(args, torch.device("cpu"))


@pytest.mark.slow
def test_a_memory_carries_its_gradient_within_a_window_and_is_detached_between_windows(tmp_path, pass_inputs):
    """Windows of two over game-sides of three decisions: a game-side's first
    decision reads the learned initial memory, the next decision in the same
    window reads the previous write with its gradient, and a window's first
    decision of a continuing game-side reads a detached state."""
    trainer = _trainer(pass_inputs, tmp_path, "--window", "2")
    prepare, train_window = trainer._prepare, trainer.train_window
    reads, where = [], {"t": 0}

    def spy_prepare(steps, memories):
        out = prepare(steps, memories)
        for s, m in zip(steps, out[1]):
            initial = s.k > 0 and torch.equal(m, trainer.model.initial_memory(s.k))
            reads.append((where["t"], s.starts, s.k, bool(m.requires_grad), initial))
        where["t"] += 1
        return out

    def spy_window():
        where["t"] = 0
        return train_window()

    trainer._prepare, trainer.train_window = spy_prepare, spy_window
    assert trainer.run() == 0
    cases = set()
    for t, starts, k, grad, initial in reads:
        if starts:
            assert grad and (k == 0 or initial), "a game-side starts from the learned initial memory"
            cases.add(("start", t > 0))
        elif t > 0:
            assert grad and not initial, "inside a window the previous write carries its gradient"
            cases.add(("carried", True))
        else:
            assert not grad, "a window begins from a detached state"
            cases.add(("carried", False))
    assert cases == {("start", False), ("start", True), ("carried", True), ("carried", False)}


@pytest.mark.slow
def test_a_non_finite_step_is_skipped_and_a_run_of_them_stops_the_pass(tmp_path, pass_inputs, monkeypatch):
    from tools import sequence_train
    real = sequence_train.belief_loss
    trainer = _trainer(pass_inputs, tmp_path / "one")
    monkeypatch.setattr(sequence_train, "belief_loss", lambda *a, **kw: real(*a, **kw) * (
        float("nan") if trainer.state["steps"] == 0 else 1.0))
    assert trainer.run() == 0
    assert trainer.state["nonfinite_steps"] == 1
    assert all(torch.isfinite(p).all() for p in trainer.params), "the skipped update left no NaN behind"

    stuck = _trainer(pass_inputs, tmp_path / "all", "--window", "1")
    before = [p.detach().clone() for p in stuck.params]
    monkeypatch.setattr(sequence_train, "belief_loss", lambda *a, **kw: real(*a, **kw) * float("nan"))
    assert stuck.run() == sequence_train.EXIT_NONFINITE
    assert stuck.state["nonfinite_steps"] == sequence_train.NONFINITE_LIMIT
    assert all(torch.equal(a, b) for a, b in zip(before, stuck.params))


@pytest.mark.slow
def test_the_probe_reads_each_game_side_alike_whatever_its_batch(tmp_path, pass_inputs):
    """The probe hands a finished game-side's row to the next one; each
    game-side's readings do not depend on which others share its batch."""
    from tools.sequence_probe import run_sides
    trainer = _trainer(pass_inputs, tmp_path)
    sides = [GameSide(f, s) for f in sorted(trainer.games) for s in (1, 2)]
    kw = dict(device=torch.device("cpu"), type_loss_weights=trainer.type_loss_weights)
    trainer.model.eval()
    trainer.encoder.eval()
    with torch.no_grad():
        one = run_sides(trainer.model, trainer.encoder, sides, trainer.loader.sides, 8, batch=1, **kw)
        many = run_sides(trainer.model, trainer.encoder, sides, trainer.loader.sides, 8, batch=3, **kw)
    assert set(one) == set(many) == set(sides)
    for g in sides:
        assert len(one[g]["belief"]) == len(trainer.loader.sides(g)) > 0
        for key in ("ce", "value", "belief"):
            a, b = one[g][key], many[g][key]
            assert [x is None for x in a] == [x is None for x in b], (g, key)
            assert np.allclose([x for x in a if x is not None], [x for x in b if x is not None],
                               atol=1e-5), (g, key)


@pytest.mark.slow
def test_the_holdout_ce_reads_only_the_games_the_pre_encoding_kept(tmp_path, pass_inputs):
    from tools.holdout_ce import holdout_ce, main
    from wesnoth_ai.transformer_policy import TransformerPolicy
    args = dict(zip(pass_inputs[::2], pass_inputs[1::2]))
    sequences, dataset = Path(args["--sequences"]), Path(args["--dataset"])
    spec = tmp_path / "net.pt"
    TransformerPolicy(device=torch.device("cpu"), d_model=32, num_layers=1, num_heads=2, d_ff=64,
                      relevant_set_hexes=True).save_checkpoint(spec)
    assert holdout_ce(spec, dataset, torch.device("cpu"), sequences=sequences)["n_games"] == 1
    trimmed = tmp_path / "trimmed"
    trimmed.mkdir()
    man = json.loads((sequences / "sequence_manifest.json").read_text(encoding="utf-8"))
    man["games"] = {f: c for f, c in man["games"].items() if f != "g2.json.gz"}
    (trimmed / "sequence_manifest.json").write_text(json.dumps(man), encoding="utf-8")
    out = tmp_path / "ce.json"
    assert main([str(spec), "--dataset", str(dataset), "--sequences", str(trimmed), "--out", str(out),
                 "--device", "cpu"]) == 1
    assert not out.exists(), "a run with no decision writes nothing"


@pytest.mark.parametrize("carried, paired, passes", [
    ({"diff": -0.00112, "se": 0.000136}, {"diff": -7.9e-05, "se": 4.7e-05}, True),
    ({"diff": -1e-04, "se": 1e-04}, {"diff": -1e-03, "se": 1e-04}, False),
    ({"diff": -0.00112, "se": 0.000136, "n_nonfinite": 1}, {"diff": -1e-03, "se": 1e-04}, False),
    ({"diff": -0.00112, "se": None}, {"diff": -1e-03, "se": 1e-04}, False),
], ids=["the 2026-10-01 probe", "carried no better than wiped", "non-finite", "no standard error"])
def test_the_memory_barrier_asks_whether_the_memory_remembers(carried, paired, passes):
    """The barrier (amended 2026-10-01): the belief loss with the memory
    carried beats the memory wiped at every decision by two standard
    errors, paired over holdout games. Against 0 slots is a reading, not
    the barrier: the 2026-10-01 probe read carried at -8 standard errors
    and 0 slots at -1.7."""
    from tools.sequence_probe import memory_barrier_passes
    results = {"belief_carried": {"k": 64, "n_nonfinite": 0, **carried},
               "belief_paired": {"k": 64, "n_nonfinite": 0, **paired}}
    assert memory_barrier_passes(results) is passes


def test_the_learning_rate_holds_then_falls_linearly_to_zero():
    """The cooldown of a warmup-stable-decay schedule: the multiplier is 1
    until `decay_from` of the pass, then falls linearly to 0 at its end."""
    from tools.sequence_train import lr_factor
    assert [lr_factor(p, 1000, 0.5) for p in (0, 499, 500, 750, 1000)] == [1.0, 1.0, 1.0, 0.5, 0.0]
    assert lr_factor(900, 1000, None) == 1.0
    assert lr_factor(1000, 1000, 0.0) == 0.0 and lr_factor(250, 1000, 0.0) == 0.75


@pytest.mark.slow
def test_a_second_pass_starts_from_the_first(tmp_path, pass_inputs):
    """--init-from starts a new pass from a finished pass's weights and
    optimizer state, on a schedule of its own seed; with --decay-from the
    checkpoint where the cooldown starts is kept as <out>.stable.pt."""
    from tools import sequence_train
    common = [*pass_inputs, "--probe-every", "1000000", "--barrier-positions", "1000000"]
    first, second = tmp_path / "first.pt", tmp_path / "second.pt"
    assert sequence_train.main([*common, "--out", str(first)]) == 0
    a = torch.load(first, map_location="cpu", weights_only=True)
    pass2 = [*common, "--out", str(second), "--seed", "7", "--warmup-steps", "0", "--decay-from", "0.5"]
    assert sequence_train.main([*pass2, "--init-from", str(first), "--max-positions", "0"]) == 0
    b = torch.load(second, map_location="cpu", weights_only=True)
    for key in ("model_state", "encoder_state"):
        for name, tensor in a[key].items():
            assert torch.equal(tensor, b[key][name]), name
    assert torch.equal(a["optimizer_state"]["state"][0]["exp_avg"], b["optimizer_state"]["state"][0]["exp_avg"])
    assert b["sequence_resume"]["state"]["positions"] == 0
    assert b["training_meta"]["init_from"]["positions"] == a["sequence_resume"]["state"]["positions"] == 12
    assert sequence_train.main([*pass2, "--resume"]) == 0
    stable = torch.load(tmp_path / "second.stable.pt", map_location="cpu", weights_only=True)
    end = torch.load(second, map_location="cpu", weights_only=True)
    assert 6 <= stable["sequence_resume"]["state"]["positions"] < 12
    assert end["sequence_resume"]["state"]["positions"] == 12
    assert end["training_meta"]["init_from"]["path"] == str(first)


@pytest.mark.slow
def test_a_second_pass_refuses_what_it_cannot_continue(tmp_path, pass_inputs):
    """Another architecture, a failed memory barrier, or --resume beside
    --init-from: no pass starts."""
    from tools import sequence_train
    first, failed = tmp_path / "first.pt", tmp_path / "failed.pt"
    quiet = [*pass_inputs, "--probe-every", "1000000", "--barrier-positions", "1000000"]
    assert sequence_train.main([*quiet, "--out", str(first)]) == 0
    with pytest.raises(SystemExit, match="arch"):
        sequence_train.main([*quiet, "--out", str(tmp_path / "x.pt"), "--init-from", str(first), "--d-ff", "128"])
    probed = [*pass_inputs, "--probe-every", "3", "--barrier-positions", "3"]
    assert sequence_train.main([*probed, "--out", str(failed)]) == sequence_train.EXIT_MEMORY_BARRIER
    with pytest.raises(SystemExit, match="memory barrier"):
        sequence_train.main([*quiet, "--out", str(tmp_path / "y.pt"), "--init-from", str(failed)])
    with pytest.raises(SystemExit):
        sequence_train.main([*quiet, "--out", str(first), "--init-from", str(first), "--resume"])
