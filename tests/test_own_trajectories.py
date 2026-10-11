"""The memory player's own trajectories in the sequence trainer
(docs/memory_own_trajectories_prereg_20261011.md): which game-sides are own,
which recorded commands were decisions, the teacher's 0-slot raw
distribution as each one's target, the soft-target loss, the trainer's mixed
batches, and the memory a trained game-side carries, which must be the one
its player carried in the match."""
from __future__ import annotations

import gzip
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.own_trajectories import select_own_sides  # noqa: E402
from tools.raw_player import RawPolicyPlayer  # noqa: E402
from tools.replay_dataset import ActionIndices  # noqa: E402
from wesnoth_ai import game_core as gc  # noqa: E402
from wesnoth_ai import own_loss  # noqa: E402
from wesnoth_ai.memory import MemoryState  # noqa: E402
from wesnoth_ai.sequence_records import OwnTarget  # noqa: E402
from wesnoth_ai.sequence_streams import GameSide, Step, StreamSchedule, epoch_order, merge_windows  # noqa: E402
from wesnoth_ai.server_priors import (KIND_ATTACK, KIND_END_TURN, KIND_MOVE, KIND_RECRUIT,  # noqa: E402
                                      CompactActions, PackedMasks)

needs_core = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")
TINY = ["--d-model", "32", "--num-layers", "1", "--num-heads", "2", "--d-ff", "64"]
SLOTS = 64


# ---- which game-sides are own -----------------------------------------------

def _result(d: Path, stem: str, side_a: int, a: dict, b: dict, record: bool = True) -> None:
    res = {"side_a": side_a, "outcome_a": "win", "label_a": "parity3", "label_b": "parity3"}
    for x, fields in (("a", a), ("b", b)):
        res.update({f"{k}_{x}": v for k, v in fields.items()})
    (d / f"{stem}.json").write_text(json.dumps(res), encoding="utf-8")
    if record:
        (d / f"{stem}.game.jsonl.gz").write_bytes(b"")


def test_own_sides_are_the_teachers_at_its_slots_without_a_reset(tmp_path):
    """By the checkpoint's SHA-256, the memory size and the reset flag, never
    by label: both sides of a self-pin, neither a reset side nor a 0-slot,
    16-slot or other checkpoint's side, and a side without its record is
    counted as dropped."""
    me, other = "a" * 64, "b" * 64
    own = {"checkpoint_sha256": me, "memory": SLOTS, "forwards": 40, "raw_end_turn_offset": -1.5}
    _result(tmp_path, "game_selfpin_s2_1", 2, own, {**own, "memory_reset": False, "forwards": 41})
    _result(tmp_path, "game_reset_s1_2", 1, {**own, "memory_reset": True}, {**own, "memory": 0})
    _result(tmp_path, "game_other_s1_3", 1, {**own, "memory": 16}, {**own, "checkpoint_sha256": other})
    _result(tmp_path, "game_norecord_s1_4", 1, own, {**own, "memory": 0}, record=False)
    games, counts = select_own_sides([tmp_path], me, SLOTS)
    assert list(games) == [f"{tmp_path.name}__game_selfpin_s2_1"]
    sides = games[f"{tmp_path.name}__game_selfpin_s2_1"]
    assert [(s.side, s.forwards, s.offset, s.winner) for s in sides] == [(2, 40, -1.5, 2), (1, 41, -1.5, 2)]
    assert counts["own"] == 2 and counts["results"] == 4 and counts["game_sides"] == 8
    assert counts["not_own: memory reset each turn"] == 1 and counts["not_own: another checkpoint"] == 1
    assert counts["not_own: another memory size"] == 3 and counts["dropped: no game record"] == 1


# ---- the soft-target loss -----------------------------------------------------

def _all_legal(U: int, R: int, H: int, W: int) -> PackedMasks:
    A = U + R + 1
    bits = np.packbits(np.ones((A, H), dtype=bool), axis=1)
    type_valid = np.zeros((A, 2), dtype=np.uint8)
    type_valid[:U] = 1
    n_attacks = np.zeros(A, dtype=np.int8)
    n_attacks[:U] = W
    return PackedMasks(actor_mask=np.ones(A, dtype=np.uint8), type_valid=type_valid, attack_valid=bits,
                       move_valid=bits, union_valid=bits, n_attacks=n_attacks, end_turn_bias=0.0,
                       n_units=U, n_recruits=R, n_hexes=H)


def _one_hot(actor: int, kind: int, target: int, weapon: int):
    return OwnTarget(actor=np.array([actor], np.int32), kind=np.array([kind], np.int8),
                     target=np.array([target], np.int32), weapon=np.array([weapon], np.int8),
                     prior=np.array([0.3]), end_prior=0.1)


@pytest.mark.parametrize("label, key", [
    (ActionIndices("attack", actor_idx=1, target_idx=3, weapon_idx=1, type_idx=0), (1, KIND_ATTACK, 3, 1)),
    (ActionIndices("move", actor_idx=0, target_idx=4, type_idx=1), (0, KIND_MOVE, 4, -1)),
    (ActionIndices("recruit", actor_idx=2, target_idx=2), (2, KIND_RECRUIT, 2, -1)),
    (ActionIndices("end_turn", actor_idx=3), (3, KIND_END_TURN, -1, -1)),
], ids=["attack", "move", "recruit", "end_turn"])
def test_a_one_hot_target_gives_the_hard_labels_loss(label, key, monkeypatch):
    """Where every choice is legal, the soft-target loss of a one-hot target
    is the recipe's hard-label policy loss without label smoothing."""
    from wesnoth_ai import imitation_loss
    U, R, H, W, T = 2, 1, 6, 2, 2
    torch.manual_seed(3)
    A = U + R + 1
    padded = SimpleNamespace(actor_logits=torch.randn(1, A), type_logits=torch.randn(1, A, T),
                             target_logits=torch.randn(1, A, H), weapon_logits=torch.randn(1, A, W),
                             value_logits=torch.randn(1, 5))
    batch = own_loss.stage_own_batch([0], [_all_legal(U, R, H, W)], [_one_hot(*key)], A, H, T, W,
                                     torch.device("cpu"))
    soft = own_loss.own_policy_loss(padded, batch)
    monkeypatch.setattr(imitation_loss, "LABEL_SMOOTHING", 0.0)
    targets = imitation_loss.build_imitation_targets([label], [(None, 0.0, 1.0)], [(U, R, H)], n_types=T,
                                                     n_weapons=W, n_atoms=5, type_loss_weights={},
                                                     device=torch.device("cpu"))
    parts = imitation_loss.imitation_loss_parts(padded, targets, eps=0.0)
    hard = parts.actor_raw + parts.type + parts.target + parts.weapon
    assert soft.item() == pytest.approx(hard.item(), rel=1e-6)


def _tiny_modules(seed: int = 0):
    from tools.preencode_sequences import fresh_vocab
    from tools.sequence_setup import build_modules
    torch.manual_seed(seed)
    vocab = fresh_vocab()
    encoder, model = build_modules(*vocab, torch.device("cpu"), {"d_model": 32, "num_layers": 1, "num_heads": 2,
                                                                 "d_ff": 64})
    return vocab, encoder.eval(), model.eval()


def _position_state():
    from helpers.parity_games import core_of, record
    cs = core_of(record([("Lieutenant", 1, 1, 3, True), ("Spearman", 1, 6, 1, False),
                         ("Spearman", 1, 2, 4, False), ("Lieutenant", 2, 18, 3, True),
                         ("Spearman", 2, 7, 2, False)], recruits={1: ["Spearman", "Bowman"]}, fog=True))
    cs.apply_command(["init_side", 1])
    gs = cs.to_state()
    gc.bind_view(gs, cs.fork())
    return gs


def _candidate(gs, vocab):
    """The position's candidate as tools/own_trajectories.py builds it."""
    from collections import Counter
    from tools.inference_seam import build_light_encoded
    from tools.preencode_sequences import encode_position
    from tools.replay_dataset import _action_indices
    from wesnoth_ai.sequence_records import OwnCandidate
    from wesnoth_ai.server_priors import pack_masks
    pos = encode_position(gs, _action_indices(gs, ["end_turn"]), "test", *vocab, Counter())
    pos.masks = pack_masks(build_light_encoded(pos.raw, torch.device("cpu")), gs)
    return OwnCandidate(position=pos)


@needs_core
def test_the_soft_target_loss_is_the_cross_entropy_to_the_target_under_the_decodes_priors():
    """The student's log prior of each target action, under the masks the
    player decided with, is the log of the prior the raw player decodes
    from (`server_priors`, an implementation of its own), and the loss is
    the target's cross-entropy to it."""
    from wesnoth_ai.server_priors import batched_priors
    vocab, encoder, model = _tiny_modules()
    pos = _candidate(_position_state(), vocab).position
    with torch.no_grad():
        padded = model.forward_embedded(encoder.embed_staged(encoder.stage_raws([pos.raw])),
                                        memory=[model.initial_memory(8)])
    compact = batched_priors(padded, [pos.masks])[0]
    assert len(compact.prior) > own_loss.TOP_ACTIONS, "the position has more legal actions than a target keeps"
    every = OwnTarget(actor=compact.actor, kind=compact.kind, target=compact.target, weapon=compact.weapon,
                      prior=np.asarray(compact.prior, dtype=np.float64), end_prior=0.0)
    U, R, H = padded.sizes[0]
    batch = own_loss.stage_own_batch([0], [pos.masks], [every], U + R + 1, H, 2, padded.weapon_logits.shape[2],
                                     torch.device("cpu"))
    lp = own_loss.legal_log_probs(padded, batch)[0].numpy()
    assert np.allclose(np.exp(lp), compact.prior, rtol=1e-4, atol=1e-7)
    target = own_loss.top_target(compact.actor, compact.kind, compact.target, compact.weapon, compact.prior)
    batch = own_loss.stage_own_batch([0], [pos.masks], [target], U + R + 1, H, 2, padded.weapon_logits.shape[2],
                                     torch.device("cpu"))
    keys = own_loss.keys_of([every])[0]
    log_prior = {k: np.log(p) for k, p in zip(keys, compact.prior)}
    want = -sum(float(t) * log_prior[k] for k, t in zip(own_loss.keys_of([target])[0], target.prob))
    assert own_loss.own_policy_loss(padded, batch).item() == pytest.approx(want, rel=1e-4)


# ---- the targets -------------------------------------------------------------

def _server(model, encoder):
    from tools.inference_seam import InferenceServer
    return InferenceServer(model, encoder, device=torch.device("cpu"))


@needs_core
def test_the_stored_target_is_the_frozen_networks_raw_distribution_at_0_slots(tmp_path):
    """The target the labelling stores is the raw player's list of legal
    priors at 0 slots (`enumerate_legal_actions_with_priors`, no end_turn
    offset), its 16 most probable renormalised; the 64-slot reading of the
    same network differs from it."""
    from tools.own_targets import label_games
    from tools.preencode_sequences import read_record, record_path
    from wesnoth_ai.action_sampler import enumerate_legal_actions_with_priors
    from wesnoth_ai.sequence_records import OwnCandidateGame, OwnCandidateSide
    vocab, encoder, model = _tiny_modules()
    gs = _position_state()
    game = OwnCandidateGame(file="g", winner=1, n_commands=2)
    game.sides[1] = OwnCandidateSide(side=1, candidates=[_candidate(gs, vocab)], forwards=1, rejections=0,
                                     offset=-1.5)
    with torch.no_grad():
        counts = label_games(_server(model, encoder), SLOTS, [game], tmp_path, reference_offset=-1.5)
    assert counts["positions"] == 1 and counts["game_sides_kept"] == 1
    target = read_record(record_path(tmp_path, "g")).sides[1][0].own_target
    assert target.prob.sum() == pytest.approx(1.0, abs=1e-6) and len(target.prob) == own_loss.TOP_ACTIONS

    def legal_priors(slots):
        with torch.no_grad():
            encoded = encoder.encode(gs)
            out = model(encoded, memory=MemoryState(slots, None))
            legal = enumerate_legal_actions_with_priors(encoded, out, gs)
        kinds = {"attack": KIND_ATTACK, "move": KIND_MOVE, "recruit": KIND_RECRUIT, "end_turn": KIND_END_TURN}
        return {(la.actor_idx, kinds[la.action["type"]], -1 if la.target_idx is None else la.target_idx,
                 -1 if la.weapon_idx is None else la.weapon_idx): la.prior for la in legal}

    zero = legal_priors(0)
    best = sorted(zero.items(), key=lambda kv: -kv[1])[:own_loss.TOP_ACTIONS]
    mass = sum(p for _, p in best)
    assert own_loss.keys_of([target])[0] == [k for k, _ in best]
    assert np.allclose(target.prob, [p / mass for _, p in best], atol=1e-5)
    assert target.end_prior == pytest.approx(zero[next(k for k in zero if k[1] == KIND_END_TURN)], rel=1e-5)
    full = legal_priors(SLOTS)
    assert max(abs(full[k] - zero[k]) for k in zero) > 1e-4, "the memory changes the network's reading"


def _walk_out(priors, memory):
    """An inference output with the given (actor, kind, target, weapon) -> prior."""
    keys = list(priors)
    compact = CompactActions(actor=np.array([k[0] for k in keys], np.int32),
                             kind=np.array([k[1] for k in keys], np.int8),
                             target=np.array([k[2] for k in keys], np.int32),
                             weapon=np.array([k[3] for k in keys], np.int8),
                             prior=np.array([priors[k] for k in keys]))
    return SimpleNamespace(legal_compact=compact, memory=memory)


MOVE = ActionIndices("move", actor_idx=0, target_idx=5, type_idx=1)
ATTACK = ActionIndices("attack", actor_idx=0, target_idx=9, weapon_idx=0, type_idx=0)
AFTER = ActionIndices("attack", actor_idx=0, target_idx=9, weapon_idx=0, type_idx=0)
END = ActionIndices("end_turn", actor_idx=2)


def _label_side(out_dir: Path, p_attack: float, p_move: float, forwards: int):
    """One game-side of three candidates (an approach move with its attack's
    label, the attack, an end_turn) and one refused recruit, labelled under
    a teacher with the given priors: its decisions' labels and the counts."""
    from tools.own_targets import OwnLabeller
    from wesnoth_ai.sequence_records import OwnCandidate, OwnCandidateGame, OwnCandidateSide, SequencePosition
    raw = SimpleNamespace(unit_positions=[0, 1], recruit_types=[])

    def position(label):
        return SequencePosition(raw=raw, label=label, hidden_tokens=np.zeros(0, np.int64),
                                no_visible_unit=np.zeros(0, bool), turn=7)

    cands = [OwnCandidate(position(MOVE), attack_label=ATTACK), OwnCandidate(position(AFTER)),
             OwnCandidate(position(END))]
    game = OwnCandidateGame(file="g", winner=0, n_commands=6)
    game.sides[1] = OwnCandidateSide(side=1, candidates=cands, forwards=forwards, rejections=1, offset=-1.5)
    labeller = OwnLabeller(out_dir, reference_offset=-1.5)
    w = next(labeller.walks([game]))
    out = _walk_out({(0, KIND_ATTACK, 9, 0): p_attack, (0, KIND_MOVE, 5, -1): p_move,
                     (2, KIND_END_TURN, -1, -1): 0.1}, None)
    labels = []
    while not w.done:
        merged = labeller(w, out, out) == 2
        labels.append(w.payload.candidates[w.t].attack_label if merged else w.positions[w.t].label)
        w.t += 2 if merged else 1
    labeller(w, None, None)
    return labels, labeller.totals


def test_an_approach_and_its_attack_are_one_decision_when_the_player_preferred_the_attack(tmp_path):
    """A move followed by an attack from its end is one attack decision when
    the match's forward count leaves room for a merge and the teacher at the
    player's slots puts the attack above the move (the attack's position is
    then no decision), and two decisions otherwise; a game-side whose
    decisions and refused recruits miss the match's count of its forwards is
    dropped and counted."""
    labels, totals = _label_side(tmp_path, 0.5, 0.4, forwards=3)
    assert labels == [ATTACK, END] and totals["approach_merged"] == 1 and totals["positions"] == 2
    labels, totals = _label_side(tmp_path, 0.4, 0.5, forwards=2)
    assert labels == [MOVE, AFTER, END] and totals["approach_merged"] == 0
    assert totals["dropped: forward count mismatch"] == 1 and totals["positions"] == 0
    assert totals["positions_dropped"] == 3


def test_a_game_side_whose_count_leaves_no_merge_merges_nothing(tmp_path):
    """Candidates plus refused recruits equal to the match's forwards leave
    no room for a merge: a near tie the teacher would merge stays two
    decisions, and the game-side is kept."""
    labels, totals = _label_side(tmp_path, 0.5001, 0.5, forwards=4)
    assert labels == [MOVE, AFTER, END] and totals["approach_merged"] == 0
    assert totals["positions"] == 3 and totals["game_sides_kept"] == 1


@pytest.mark.parametrize("student, passes", [
    ({"k0": 2.80, "k64": 2.78}, True),
    ({"k0": 2.86, "k64": 2.70}, False),
    ({"k0": 2.80, "k64": 2.85}, False),
], ids=["the memory gives up its gain", "the base policy moved", "the memory broke the policy"])
def test_the_holdout_barriers_read_success_as_success(student, passes):
    """Read on this run's records (parity3 at 2.79 at 0 slots, 2.69 at 64): the
    student's loss at 0 slots within 0.05 of parity3's, its loss at 64 slots
    no higher than parity3's at 0 slots plus 0.05. A student that stops
    leaning on its memory passes."""
    from tools.own_barriers import holdout_ce_barrier
    start = {"positions": 0, "final": False, "k0": {"ce_all": 2.79}, "k64": {"ce_all": 2.69}}
    final = {"positions": 2000384, "final": True, **{k: {"ce_all": v} for k, v in student.items()}}
    assert holdout_ce_barrier([start, final], 0.05)["pass"] is passes


@pytest.mark.parametrize("late, passes", [(951, True), (949, False)])
def test_the_replay_barrier_wants_95_percent_of_late_decisions_reproduced(late, passes):
    from tools.own_barriers import replay_barrier
    assert replay_barrier({"repro_late": late, "decisions_late": 1000, "repro": 10, "decisions": 2000})["pass"] \
        is passes


def test_a_resume_refuses_streams_of_other_lengths():
    """The schedule's state names its game-sides' lengths: a resume over
    records of other lengths (a rebuilt corpus or own records that came out
    otherwise) is refused, naming the streams."""
    lengths = {GameSide(f"g{i}", s): 3 + i + s for i in range(6) for s in (1, 2)}
    saved = StreamSchedule(epoch_order(list(lengths), 5), lengths, 2, 5, name="own").state_dict()
    StreamSchedule(epoch_order(list(lengths), 5), lengths, 2, 5, name="own").load_state_dict(saved)
    other = dict(lengths)
    other[GameSide("g3", 1)] += 1
    with pytest.raises(ValueError, match="own.*lengths"):
        StreamSchedule(epoch_order(list(other), 5), other, 2, 5, name="own").load_state_dict(saved)


# ---- the trainer -------------------------------------------------------------

def test_mixed_windows_hold_equal_numbers_of_own_and_corpus_positions():
    """Half the slots walk own game-sides, at the player's memory size, the
    other half the corpus's at the nested-dropout sizes: every time step of
    every window holds as many of each, until a kind runs out."""
    human = {GameSide(f"h{i}", s): 3 + (i * 5 + s) % 7 for i in range(12) for s in (1, 2)}
    own = {GameSide(f"o{i}", s): 4 + (i * 3 + s) % 9 for i in range(10) for s in (1, 2)}
    a = StreamSchedule(epoch_order(list(human), 5), human, 3, 5)
    b = StreamSchedule(epoch_order(list(own), 5), own, 3, 5, k=SLOTS, slot_offset=3)
    full = 0
    while not (a.exhausted() or b.exhausted()):
        for steps in merge_windows(a.window(4), b.window(4)):
            mine = [s for s in steps if s.game_side.file.startswith("o")]
            theirs = [s for s in steps if s.game_side.file.startswith("h")]
            assert {s.slot for s in mine} <= {3, 4, 5} and {s.slot for s in theirs} <= {0, 1, 2}
            assert all(s.k == SLOTS for s in mine)
            if not (a.exhausted() or b.exhausted()):
                assert len(mine) == len(theirs) == 3
                full += 1
    assert full > 10


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """Three small games pre-encoded in this process (the third held out)."""
    if gc.game_core_class() is None:
        pytest.skip("wesnoth_core.GameCore not available")
    import wesnoth_core
    from helpers.synthetic_replay import move, turn, two_sides, write_replay
    from tools.preencode_sequences import (ENCODING, MANIFEST_NAME, encode_game_sequence, encoding_fingerprint,
                                           fresh_vocab, record_path, write_record)
    from tools.replay_extract import extract_replay
    from wesnoth_ai.constants import OBSERVATION_EPOCH
    tmp = tmp_path_factory.mktemp("own_corpus")
    dataset, sequences = tmp / "corpus", tmp / "sequences"
    dataset.mkdir()
    sequences.mkdir()
    vocab = fresh_vocab()
    rows, games = [], {}
    for i, ride in enumerate([[(3, 5), (3, 6)], [(3, 5), (4, 5)], [(3, 5), (3, 4)]]):
        commands = [*turn(1, move(1, ride)), *turn(2, move(2, [(11, 4), (11, 5)])), *turn(1), *turn(2)]
        rec = extract_replay(write_replay(tmp / f"g{i}.bz2", two_sides(extra1=[("Cavalryman", 3, 5, False)]),
                                          commands))
        name = f"g{i}.json.gz"
        with gzip.open(dataset / name, "wt", encoding="utf-8") as f:
            json.dump(rec, f)
        rows.append({"file": name, "winner_side": 1 + i % 2, "winner_actions": 3, "holdout": i == 2})
        seq = encode_game_sequence(rec, name, 1 + i % 2, *vocab)
        write_record(record_path(sequences, name), seq)
        games[name] = seq.counts
    (dataset / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    phase = int(wesnoth_core.__phase__)
    (sequences / MANIFEST_NAME).write_text(json.dumps({
        "fingerprint": encoding_fingerprint(*vocab, 5, phase), "encoding": ENCODING,
        "observation_epoch": int(OBSERVATION_EPOCH), "core_phase": phase, "unit_type_to_id": vocab[0],
        "faction_to_id": vocab[1], "games": games}), encoding="utf-8")
    return SimpleNamespace(dataset=dataset, sequences=sequences, vocab=vocab, manifest=json.loads(
        (sequences / MANIFEST_NAME).read_text(encoding="utf-8")))


def _args(corpus, out: Path, *extra):
    from tools import sequence_train
    return sequence_train.parse_args([
        "--sequences", str(corpus.sequences), "--dataset", str(corpus.dataset), "--device", "cpu",
        "--streams", "2", "--window", "2", "--warmup-steps", "2", "--checkpoint-every", "1000000",
        "--probe-every", "1000000", "--barrier-positions", "1000000", "--signal-every", "1000000",
        "--log-level", "WARNING", "--probe-ks", "0,8", *TINY, "--out", str(out), *extra])


def test_without_own_streams_a_step_is_the_recipes_step(corpus, tmp_path):
    """Without own streams the trainer's window is the recipe's, bit for bit:
    the same steps through the recipe's own parts (`_prepare`,
    `_step_forward`), summed and stepped as before own streams existed."""
    from torch.utils.checkpoint import checkpoint
    from tools import sequence_train
    trainers = []
    for name in ("a", "b"):
        args = _args(corpus, tmp_path / f"{name}.pt")
        torch.manual_seed(args.seed)
        trainers.append(sequence_train.Trainer(args, torch.device("cpu")))
    a, b = trainers
    assert a.own is None and a.signal_terms == sequence_train.SIGNAL_TERMS
    torch.manual_seed(11)
    a.train_window()
    torch.manual_seed(11)
    total = torch.zeros(())
    for steps in b.schedule.window(b.args.window):
        staged, mems, targets, b_target, b_mask = b._prepare(steps, b.memories)
        pv, bsum, mem, _, _ = checkpoint(b._step_forward, staged, mems, targets, b_target, b_mask,
                                         use_reentrant=False)
        total = total + pv + b.args.belief_weight * bsum
        for i, s in enumerate(steps):
            b.memories[s.slot] = mem[i, :s.k]
    loss = total / float(b.args.streams * b.args.window)
    for group in b.opt.param_groups:
        group["lr"] = b._lr_now()
    loss.backward()
    b.clip_groups.clip(b.args.grad_clip)
    b.opt.step()
    assert len(a.params) == len(b.params)
    for pa, pb in zip(a.params, b.params):
        assert torch.equal(pa, pb)
    assert sorted(a.memories) == sorted(b.memories)
    for s in a.memories:
        assert torch.equal(a.memories[s], b.memories[s].detach())


# ---- the memory along an own game-side is the player's -----------------------

class _LoggedPlayer(RawPolicyPlayer):
    """The raw player at the reference decode, logging the memory each of
    its decisions wrote; a refused decision's entry goes, as its memory does."""

    def __init__(self, base):
        super().__init__(base, 0.0, memory_slots=SLOTS, end_turn_offset=-1.5)
        self.written, self.forwards = [], 0

    def _forward(self, base, encoded, game_label, game_state):
        out = super()._forward(base, encoded, game_label, game_state)
        self.written.append(out.memory.detach().clone())
        self.forwards += 1
        return out

    def drop_last_pending(self, game_label):
        self.written.pop()
        return super().drop_last_pending(game_label)


def _play(model, encoder, out_dir: Path, sha: str):
    """A mini game between two 64-slot players of the network, through the
    match loop, recorded as a match records it."""
    from sim_test_helpers import scenario_setup
    from tools.eval_players import _play_one_eval_game, _PolicyPair
    from tools.game_record import GameRecordLog, game_record
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate
    setup = scenario_setup(3, mini=True)
    sim = WesnothSim(build_scenario_gamestate(setup), scenario_id=setup.scenario_id, max_turns=3)
    sim._seed_salt = "own-test"
    players = {}
    for side in (1, 2):
        base = SimpleNamespace(_inference_model=model, _inference_encoder=encoder, _lock=threading.Lock(),
                               _decision_step=0)
        players[side] = _LoggedPlayer(base)
    stem = "game_p_p_s1_7"
    _play_one_eval_game(sim, _PolicyPair(players[1], "p", 1), _PolicyPair(players[2], "p", 2), game_label=stem)
    GameRecordLog(out_dir / f"{stem}.game.jsonl.gz").write(game_record(
        sim, setup, game_label=stem, players={"a": {"label": "p", "side": 1, "memory": SLOTS},
                                              "b": {"label": "p", "side": 2, "memory": SLOTS}}))
    outcome = {1: "win", 2: "loss"}.get(sim.winner, "timeout")
    res = {"side_a": 1, "outcome_a": outcome}
    for x, side in (("a", 1), ("b", 2)):
        res.update({f"checkpoint_sha256_{x}": sha, f"memory_{x}": SLOTS, f"forwards_{x}": players[side].forwards,
                    f"raw_end_turn_offset_{x}": -1.5})
    (out_dir / f"{stem}.json").write_text(json.dumps(res), encoding="utf-8")
    return players


@needs_core
def test_the_memory_along_an_own_game_side_is_the_players(corpus, tmp_path):
    """A game the network played at 64 slots, through selection, the
    candidates and the labelling, trains its decisions one for one: each
    game-side holds the player's decisions, and the trainer's memory carried
    along it is the memory the player carried at the same decisions."""
    from tools import sequence_train
    from tools.own_targets import label_games, own_manifest
    from tools.own_trajectories import candidates_of, read_game_record
    from tools.preencode_sequences import MANIFEST_NAME
    vocab, encoder, model = _tiny_modules(seed=4)
    games_dir, own_dir = tmp_path / "games", tmp_path / "own"
    games_dir.mkdir()
    own_dir.mkdir()
    players = _play(model, encoder, games_dir, "c" * 64)
    games, selection = select_own_sides([games_dir], "c" * 64, SLOTS)
    [(name, sides)] = games.items()
    game = candidates_of(read_game_record(Path(sides[0].record)), sides, *vocab)
    with torch.no_grad():
        counts = label_games(_server(model, encoder), SLOTS, [game], own_dir, reference_offset=-1.5)
    assert counts["game_sides_kept"] == 2, counts
    cand = {**{k: corpus.manifest[k] for k in ("fingerprint", "encoding", "observation_epoch", "core_phase",
                                                "unit_type_to_id", "faction_to_id")},
            "teacher": "tiny", "teacher_sha256": "c" * 64, "slots": SLOTS, "selection": dict(selection),
            "errors": {}, "games": {name: {"holdout": False}}}
    (own_dir / MANIFEST_NAME).write_text(json.dumps(own_manifest(cand, {name: counts}, 16, -1.5)),
                                         encoding="utf-8")
    out = tmp_path / "arm.pt"
    args = _args(corpus, out, "--own-sequences", str(own_dir), "--pass-positions", "8")
    args.signal_every = 1
    trainer = sequence_train.Trainer(args, torch.device("cpu"))
    trainer.model.load_state_dict(model.state_dict())
    trainer.encoder.load_state_dict(encoder.state_dict())
    trainer.model.eval()
    trainer.encoder.eval()
    assert trainer.own is not None and trainer.signal_terms[-1] == "own"
    for side, player in players.items():
        g = GameSide(name, side)
        n = len(trainer.loader.sides(g))
        assert n == len(player.written) > 0
        memories = {}
        with torch.no_grad():
            for i in range(n):
                step = Step(slot=1, game_side=g, offset=i, k=SLOTS, starts=i == 0)
                staged, mems, targets, b_target, b_mask = trainer._prepare([step], memories)
                own = trainer._own_batch([step], staged)
                total, _, mem, _, _, ce = trainer._step_forward_own(staged, mems, targets, b_target, b_mask, own)
                memories[1] = mem[0, :SLOTS]
                assert torch.isfinite(total) and float(ce[0]) >= 0.0
                assert torch.allclose(memories[1], player.written[i], atol=1e-4), (side, i)
    # A training window mixes the two kinds one for one, and its telemetry
    # splits the own loss out as a term of its own.
    trainer.model.train()
    trainer.encoder.train()
    sums = trainer.train_window()
    assert sums["own_n"] == sums["positions"] // 2 == 2 and sums["own"] > 0
    signal = json.loads(out.with_suffix(".signal.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    shares = [signal["gradient"]["all"][t]["share"] for t in trainer.signal_terms]
    assert sum(x for x in shares if x is not None) == pytest.approx(1.0, abs=1e-6)
    assert signal["gradient"]["all"]["own"]["norm"] > 0
    assert "own" in json.loads(out.with_suffix(".steps.jsonl").read_text(encoding="utf-8").splitlines()[-1])
