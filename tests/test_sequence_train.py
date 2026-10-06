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
from wesnoth_ai.sequence_streams import (GameSide, Step, StreamSchedule,  # noqa: E402
                                         epoch_order)


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
def test_every_step_logs_where_its_gradient_goes(tmp_path, pass_inputs):
    """<out>.steps.jsonl: a row per step whose per-group gradient norms make
    up the step's whole gradient norm, with the memory's write among them."""
    trainer = _trainer(pass_inputs, tmp_path)
    assert trainer.run() == 0
    rows = [json.loads(line) for line in (tmp_path / "arm.steps.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows and [r["step"] for r in rows] == list(range(1, len(rows) + 1))
    for r in rows:
        assert sum(v * v for v in r["grad"].values()) == pytest.approx(r["grad_norm"] ** 2, rel=1e-4)
    assert any(r["grad"]["memory_write"] > 0 for r in rows)


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


def _one_position_steps(pass_inputs):
    """The tiny pass in steps of one position (one stream, a window of 1):
    twelve steps, so a cooldown over its second half is visible."""
    args = list(pass_inputs)
    for flag, value in (("--streams", "1"), ("--window", "1")):
        args[args.index(flag) + 1] = value
    return [*args, "--probe-every", "1000000", "--barrier-positions", "1000000"]


@pytest.mark.slow
def test_a_second_pass_starts_from_the_first(tmp_path, pass_inputs):
    """--init-from starts a new pass from a finished pass's weights and
    optimizer state, on a schedule of its own seed; with --decay-from the
    learning rate ends below its peak, and the checkpoint where the cooldown
    starts is kept as <out>.stable.pt, written once: a resume past it keeps
    it as it was."""
    import hashlib
    from tools import sequence_train
    common = _one_position_steps(pass_inputs)
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
    assert sequence_train.main([*pass2, "--resume", "--max-positions", "9"]) == 0
    stable = tmp_path / "second.stable.pt"
    before = hashlib.sha256(stable.read_bytes()).hexdigest()
    assert torch.load(stable, map_location="cpu", weights_only=True)["sequence_resume"]["state"]["positions"] == 6
    assert sequence_train.main([*pass2, "--resume"]) == 0
    assert hashlib.sha256(stable.read_bytes()).hexdigest() == before, "written once, at the cooldown's start"
    end = torch.load(second, map_location="cpu", weights_only=True)
    assert end["sequence_resume"]["state"]["positions"] == 12
    assert end["training_meta"]["init_from"]["path"] == str(first)
    assert 0.0 < end["optimizer_state"]["param_groups"][0]["lr"] < 2.8e-4, "the rate cooled down"


@pytest.mark.slow
def test_a_crash_while_keeping_the_stable_checkpoint_is_repaired_on_resume(tmp_path, pass_inputs, monkeypatch):
    """The stable file is written before the pass records it as kept: a pass
    killed while writing it rewrites it when resumed."""
    from tools import sequence_train
    common = _one_position_steps(pass_inputs)
    first, second = tmp_path / "first.pt", tmp_path / "second.pt"
    assert sequence_train.main([*common, "--out", str(first)]) == 0
    pass2 = [*common, "--out", str(second), "--seed", "7", "--warmup-steps", "0", "--decay-from", "0.5",
             "--checkpoint-every", "1"]
    real = sequence_train.save_checkpoint

    def killed_on_the_stable_file(path, *rest):
        if str(path).endswith(".stable.pt"):
            raise OSError("the disk went away")
        return real(path, *rest)

    monkeypatch.setattr(sequence_train, "save_checkpoint", killed_on_the_stable_file)
    with pytest.raises(OSError):
        sequence_train.main([*pass2, "--init-from", str(first)])
    monkeypatch.setattr(sequence_train, "save_checkpoint", real)
    assert not (tmp_path / "second.stable.pt").exists()
    assert sequence_train.main([*pass2, "--resume"]) == 0
    stable = torch.load(tmp_path / "second.stable.pt", map_location="cpu", weights_only=True)
    assert stable["sequence_resume"]["state"]["positions"] == 6


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


@pytest.mark.slow
def test_a_pass_keeps_its_rate_areas_and_a_new_pass_starts_from_them(tmp_path, pass_inputs):
    """The trainer's areas are those of the rates it applied (the replay of
    its settings, for a pass whose rate follows the step count); every probe
    row carries them; a new pass starts from the areas of the checkpoint it
    starts from and refuses a second source."""
    from tools import lr_law, sequence_train
    common = [*_one_position_steps(pass_inputs), "--probe-every", "4"]
    first, second = tmp_path / "first.pt", tmp_path / "second.pt"
    assert sequence_train.main([*common, "--out", str(first)]) == 0
    state = torch.load(first, map_location="cpu", weights_only=True)["sequence_resume"]["state"]
    want = lr_law.Areas()
    for s, rate in enumerate(lr_law.pass_rates(state["steps"], state["positions"], 2.8e-4, 2, None)):
        want.step(rate, warming=(s + 1) < 2)
    assert state["areas"] == pytest.approx(want.to_dict())
    rows = lr_law.read_probes(first.with_suffix(".probe.jsonl"))
    assert [r["positions"] for r in rows] == [4, 8, 12] and rows[-1]["areas"]["s1"] == pytest.approx(want.s1)
    assert sequence_train.main([*common, "--out", str(second), "--init-from", str(first), "--warmup-steps", "0",
                                "--max-positions", "0"]) == 0
    carried = torch.load(second, map_location="cpu", weights_only=True)["sequence_resume"]["state"]["areas"]
    assert carried == pytest.approx(state["areas"])
    with pytest.raises(SystemExit, match="carries"):
        sequence_train.main([*common, "--out", str(tmp_path / "third.pt"), "--init-from", str(first),
                             "--initial-areas", json.dumps(want.to_dict())])
    older = torch.load(first, map_location="cpu", weights_only=True)
    del older["sequence_resume"]["state"]["areas"]
    torch.save(older, tmp_path / "older.pt")
    given = {"s1": 1.5, "s2": 0.25, "m": 0.01, "prev": 2.8e-4}
    assert sequence_train.main([*common, "--out", str(tmp_path / "fourth.pt"), "--init-from", str(tmp_path / "older.pt"),
                                "--initial-areas", json.dumps(given), "--max-positions", "0"]) == 0
    assert torch.load(tmp_path / "fourth.pt", map_location="cpu",
                      weights_only=True)["sequence_resume"]["state"]["areas"] == pytest.approx(given)
    points = tmp_path / "points.jsonl"
    points.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="areas"):
        sequence_train.main([*common, "--out", str(tmp_path / "fifth.pt"), "--init-from", str(tmp_path / "older.pt"),
                             "--anneal-rule", "0.03", "--law-points", str(points), "--law-key", "k8"])


@pytest.mark.slow
def test_a_pass_lowering_from_its_start_keeps_the_areas_its_rates_make(tmp_path, pass_inputs):
    """A pass that warms up for two steps while its rate falls from the first
    position: its areas are the replay of its settings, the step where the
    warm-up ends included, and it keeps no copy of its start."""
    from tools import lr_law, sequence_train
    out = tmp_path / "low.pt"
    assert sequence_train.main([*_one_position_steps(pass_inputs), "--out", str(out), "--decay-from", "0"]) == 0
    state = torch.load(out, map_location="cpu", weights_only=True)["sequence_resume"]["state"]
    want = lr_law.Areas()
    for s, rate in enumerate(lr_law.pass_rates(state["steps"], state["positions"], 2.8e-4, 2, 0.0, 1)):
        want.step(rate, warming=(s + 1) < 2)
    assert want.s2 > 0 and state["areas"] == pytest.approx(want.to_dict())
    assert not (tmp_path / "low.stable.pt").exists()


@pytest.mark.slow
def test_a_pass_of_fewer_positions_lowers_its_rate_over_them_and_ends_there(tmp_path, pass_inputs):
    from tools import lr_law, sequence_train
    common = _one_position_steps(pass_inputs)
    first, low = tmp_path / "first.pt", tmp_path / "low.pt"
    assert sequence_train.main([*common, "--out", str(first)]) == 0
    lowering = [*common, "--init-from", str(first), "--warmup-steps", "0", "--decay-from", "0"]
    assert sequence_train.main([*lowering, "--out", str(low), "--pass-positions", "6"]) == 0
    ck = torch.load(low, map_location="cpu", weights_only=True)
    assert ck["sequence_resume"]["state"]["positions"] == 6
    final = lr_law.read_probes(low.with_suffix(".probe.jsonl"))[-1]
    assert final["final"] and final["positions"] == final["total_positions"] == 6
    assert ck["optimizer_state"]["param_groups"][0]["lr"] == pytest.approx(2.8e-4 / 6), \
        "the last step runs at a sixth of the peak: the line reaches 0 at the pass's end"
    with pytest.raises(SystemExit, match="pass_positions"):
        sequence_train.main([*common, "--out", str(low), "--resume", "--warmup-steps", "0", "--decay-from", "0",
                             "--pass-positions", "8"])


@pytest.mark.slow
def test_the_anneal_rule_stops_the_pass_where_it_decides_and_a_resume_stops_again(tmp_path, pass_inputs, monkeypatch):
    """The rule runs after every probe on the earlier passes' points and
    this pass's, with the trainer's S1, its peak rate and an epoch of its
    steps (two streams of three positions a step: 12 positions, 2 steps): a
    decision to lower stops the pass there (exit 6), kept in the checkpoint
    so a resume trains nothing more; a decision to hold lets the pass go on."""
    from tools import lr_law, sequence_train
    common = [*pass_inputs, "--probe-every", "1000000", "--barrier-positions", "1000000"]
    first = tmp_path / "first.pt"
    assert sequence_train.main([*common, "--out", str(first)]) == 0
    points = tmp_path / "law_points.jsonl"
    lr_law.write_points(points, [lr_law.Point(s1, s2, 2.0 + 1.5 * s1 ** -0.25 - 1.2 * s2)
                                 for s1, s2 in ((0.5, 0), (1, 0), (1.5, 0), (2, 0), (2.5, 0.05), (3, 0.15))])
    rule = [*common, "--init-from", str(first), "--warmup-steps", "0", "--probe-every", "4",
            "--law-points", str(points), "--law-key", "k8"]
    lowered = tmp_path / "lowered.pt"
    calls, decide = [], sequence_train.law_decide

    def spy(earlier, this_pass, s1, peak, epoch_steps, threshold):
        calls.append((len(earlier), len(this_pass), s1, peak, epoch_steps, threshold))
        return decide(earlier, this_pass, s1, peak, epoch_steps, threshold)

    monkeypatch.setattr(sequence_train, "law_decide", spy)
    epoch = lr_law.read_probes(first.with_suffix(".probe.jsonl"))[-1]["total_positions"]
    short = ["--pass-positions", str(epoch - 2)]          # the rule still reads an epoch, not the pass
    assert sequence_train.main([*rule, *short, "--out", str(lowered), "--anneal-rule", "1e9"]) \
        == sequence_train.EXIT_LOWER
    state = torch.load(lowered, map_location="cpu", weights_only=True)["sequence_resume"]["state"]
    probe = lr_law.read_probes(lowered.with_suffix(".probe.jsonl"))[-1]
    decisions = [json.loads(line) for line in lowered.with_suffix(".anneal.jsonl").read_text().splitlines()]
    assert state["anneal"] == "lower" and 4 <= state["positions"] == probe["positions"] < probe["total_positions"]
    assert [(d["action"], d["positions"]) for d in decisions] == [("lower", state["positions"])]
    assert calls == [(6, 1, pytest.approx(probe["areas"]["s1"]), 2.8e-4, epoch / 6.0, 1e9)], \
        "the rule reads the earlier probes, this pass's, the trainer's S1, its peak and an epoch of its steps"
    resumed = [*common, "--out", str(lowered), "--resume", "--warmup-steps", "0", "--probe-every", "4",
               "--law-points", str(points), "--law-key", "k8", "--anneal-rule", "1e9", *short]
    assert sequence_train.main(resumed) == sequence_train.EXIT_LOWER
    assert torch.load(lowered, map_location="cpu",
                      weights_only=True)["sequence_resume"]["state"]["positions"] == state["positions"]
    held = tmp_path / "held.pt"
    assert sequence_train.main([*rule, "--out", str(held), "--anneal-rule", "-1", "--probe-every", "8"]) == 0
    assert [json.loads(line)["action"] for line in held.with_suffix(".anneal.jsonl").read_text().splitlines()] \
        == ["hold"]
    with pytest.raises(SystemExit, match="law-key"):
        sequence_train.main([*rule, "--out", str(tmp_path / "x.pt"), "--anneal-rule", "0.03", "--law-key", "k64"])




@pytest.mark.slow
def test_a_pass_killed_while_deciding_decides_at_the_same_probe_on_resume(tmp_path, pass_inputs, monkeypatch):
    """The rule's decision is saved with its probe, and no save falls between
    them: a pass killed while deciding repeats that probe, at that position,
    on resume, and decides there rather than one probe later."""
    from tools import lr_law, sequence_train
    common = _one_position_steps(pass_inputs)
    first = tmp_path / "first.pt"
    assert sequence_train.main([*common, "--out", str(first)]) == 0
    points = tmp_path / "law_points.jsonl"
    lr_law.write_points(points, [lr_law.Point(s1, s2, 2.0 + 1.5 * s1 ** -0.25 - 1.2 * s2)
                                 for s1, s2 in ((0.5, 0), (1, 0), (1.5, 0), (2, 0), (2.5, 0.05), (3, 0.15))])
    rule = ["--warmup-steps", "0", "--probe-every", "4", "--checkpoint-every", "1", "--law-points", str(points),
            "--law-key", "k8", "--anneal-rule", "1e9"]
    out = tmp_path / "cut.pt"

    def killed(*_):
        raise OSError("killed while deciding")

    monkeypatch.setattr(sequence_train, "law_decide", killed)
    with pytest.raises(OSError, match="deciding"):
        sequence_train.main([*common, *rule, "--out", str(out), "--init-from", str(first)])
    monkeypatch.undo()
    assert sequence_train.main([*common, *rule, "--out", str(out), "--resume"]) == sequence_train.EXIT_LOWER
    assert torch.load(out, map_location="cpu", weights_only=True)["sequence_resume"]["state"]["positions"] == 4
    assert [r["positions"] for r in lr_law.read_probes(out.with_suffix(".probe.jsonl"))] == [4]
