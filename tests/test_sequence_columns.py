"""Columnar sequence records (wesnoth_ai/sequence_columns.py,
tools/columnize_sequences.py) and the sequence trainer's `--columns`
(docs/trainer_speed_20261013.md, round 2): every position read back from
columns equals the pickled one; the inputs staged from columns are the
pickles' byte for byte; a pass trains the same weights from either."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.bench_sequence_step import set_dropout

QUIET = ["--probe-every", "1000000", "--barrier-positions", "1000000", "--signal-every", "1000000"]


def _option(args, flag):
    return Path(args[args.index(flag) + 1])


def _with(args, flag, value):
    out = list(args)
    out[out.index(flag) + 1] = str(value)
    return out


def _trainer(args, tmp_path, *extra):
    from tools import sequence_train
    tmp_path.mkdir(parents=True, exist_ok=True)
    return sequence_train.build_trainer(sequence_train.parse_args([*args, *QUIET, "--out", str(tmp_path / "arm.pt"),
                                                                   *extra]))


# ---- the records ---------------------------------------------------------------------

@pytest.mark.slow
def test_every_position_read_from_columns_is_the_pickled_one(columnar):
    """Both test passes' records, own masks and targets included: every
    position read back from the columns equals the pickled one in every
    field the columns keep, dtypes and bytes; the converter's manifest
    counts them."""
    from tools.columnize_sequences import load_columns_manifest
    from tools.preencode_sequences import load_manifest, read_record, record_path
    from wesnoth_ai.sequence_columns import columns_path, position_difference, read_sides
    seen = {"corpus": 0, "own": 0}
    for kind, flag in (("corpus", "--sequences"), ("own", "--own-sequences")):
        directory = _option(columnar[kind], flag)
        cols = load_columns_manifest(directory)
        assert cols["fingerprint"] == load_manifest(directory)["fingerprint"] and not cols["errors"]
        for game in load_manifest(directory)["games"]:
            pickled = read_record(record_path(directory, game))
            sides = read_sides(columns_path(directory, game))
            assert sorted(sides) == sorted(pickled.sides)
            for side, positions in pickled.sides.items():
                assert len(sides[side]) == len(positions)
                for i, p in enumerate(positions):
                    assert position_difference(sides[side].position(i), p) is None, (game, side, i)
                    seen[kind] += 1
        assert cols["positions"] == seen[kind]
    assert seen["corpus"] > 10 and seen["own"] > 10


@pytest.mark.slow
def test_a_difference_in_any_kept_field_is_named(columnar):
    """`position_difference` is the converter's acceptance: a position
    differing in one value of an array, a label index, a dtype or an own
    mask is named, not passed."""
    import dataclasses
    from tools.preencode_sequences import load_manifest, read_record, record_path
    from wesnoth_ai.sequence_columns import position_difference
    directory = _option(columnar["own"], "--own-sequences")
    game = next(iter(load_manifest(directory)["games"]))
    p = next(q for q in read_record(record_path(directory, game)).sides[1] if q.raw.unit_feats.size)
    feats = p.raw.unit_feats.copy()
    feats[0, 0] += 1.0
    assert position_difference(dataclasses.replace(p, raw=dataclasses.replace(p.raw, unit_feats=feats)), p) \
        == "raw.unit_feats"
    assert position_difference(dataclasses.replace(
        p, raw=dataclasses.replace(p.raw, hex_xs=p.raw.hex_xs.astype(np.int32))), p) == "raw.hex_xs"
    label = dataclasses.replace(p.label, actor_idx=p.label.actor_idx + 1)
    assert position_difference(dataclasses.replace(p, label=label), p) == "label.actor_idx"
    bits = p.masks.move_valid.copy()
    bits[0, 0] ^= 1
    assert position_difference(dataclasses.replace(p, masks=dataclasses.replace(p.masks, move_valid=bits)), p) \
        == "masks.move_valid"


@pytest.mark.slow
def test_the_trainer_refuses_columns_of_other_records(columnar, tmp_path):
    """`--columns` over a directory whose columns were written from other
    records (another manifest fingerprint), or that lacks a game's, stops
    with the converter's command in the message."""
    corpus = tmp_path / "corpus"
    shutil.copytree(_option(columnar["corpus"], "--sequences"), corpus)
    args = _with(columnar["corpus"], "--sequences", corpus)
    path = corpus / "columns_manifest.json"
    cols = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**cols, "fingerprint": "other"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="columnize_sequences"):
        _trainer(args, tmp_path / "a", "--columns")
    games = dict(cols["games"])
    games.pop(next(iter(games)))
    path.write_text(json.dumps({**cols, "games": games}), encoding="utf-8")
    with pytest.raises(SystemExit, match="no columns for 1 games"):
        _trainer(args, tmp_path / "b", "--columns")


# ---- the staged inputs -----------------------------------------------------------------

def _views_bytes(views):
    return {k: (str(v.dtype), tuple(v.shape), v.contiguous().view(torch.uint8).numpy().tobytes() if v.numel() else b"")
            for k, v in views.items()}


def _inputs_bytes(x):
    """Every staged byte of a time step's inputs, with its layout."""
    out = {"staged": _views_bytes(x.staged.views), "totals": x.staged.totals, "sizes": x.staged.sizes,
           "sightings": x.staged.sighting_counts, "their": x.staged.their_field,
           "targets": x.targets.host.numpy().tobytes(), "target_fields": x.targets.layout.fields,
           "ok": {k: v.tobytes() for k, v in x.targets.ok.items()}, "policy_w": x.targets.policy_w.tobytes(),
           "belief": _views_bytes({"target": x.belief_target, "mask": x.belief_mask})}
    if x.own is not None:
        out["own"] = _views_bytes({"rows": x.own.rows, "actor": x.own.actor, "kind": x.own.kind,
                                   "target": x.own.target, "weapon": x.own.weapon, "prob": x.own.prob,
                                   **x.own.views})
    return out


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["corpus", "own"])
def test_the_columns_stage_the_pickles_bytes(columnar, tmp_path, monkeypatch, kind):
    """Every time step of the pass, game-sides at memory sizes of their own
    (rows of differing widths in a step): the staged batch, the imitation
    and belief targets and the own rows' masks and targets staged from
    columns are the pickles' byte for byte, field for field, and so are the
    rows' sizes."""
    from wesnoth_ai import sequence_streams
    monkeypatch.setattr(sequence_streams, "memory_size", lambda seed, g: 8 * (int(g.file[1]) + 1) - 5 * (g.side - 1))
    trainers = [_trainer(columnar[kind], tmp_path / name, "--window", "2", *extra)
                for name, extra in (("records", ()), ("columns", ("--columns",)))]
    steps_seen = own_steps = 0
    while True:
        windows = [t._window() for t in trainers]
        assert windows[0] == windows[1]
        if not windows[0]:
            break
        for steps in windows[0]:
            a, b = (t.inputs.stage(steps) for t in trainers)
            assert _inputs_bytes(a) == _inputs_bytes(b)
            assert trainers[0].inputs.sizes(steps) == trainers[1].inputs.sizes(steps)
            steps_seen += 1
            own_steps += a.own is not None
    assert steps_seen >= 3 and (own_steps > 0) == (kind == "own")


# ---- the pass ----------------------------------------------------------------------------

def _pass(args, tmp_path, *extra):
    """The whole pass in windows of two, every dropout at 0.3: per window
    its returned values, gradients, parameters after the step and carried
    memories."""
    trainer = _trainer(args, tmp_path, "--window", "2", *extra)
    set_dropout(trainer, 0.3)
    seen, zero_grad = [], trainer.opt.zero_grad

    def hooked(*a, **kw):
        seen.append([None if p.grad is None else p.grad.clone() for p in trainer.params])
        return zero_grad(*a, **kw)

    trainer.opt.zero_grad = hooked
    out = []
    while True:
        result = trainer.train_window()
        if not result:
            return out
        out.append((result, seen[-1], [p.detach().clone() for p in trainer.params],
                    {s: m.clone() for s, m in trainer.memories.items()}))


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["corpus", "own"])
def test_a_pass_on_columns_trains_what_a_pass_on_pickles_trains(columnar, tmp_path, kind):
    """The whole pass from columns and from the pickles, on the packed
    trunk with every activation kept: the same losses and logged sums, the
    same gradients, weights and carried memories after every window, to
    the bit."""
    extra = ("--packed-trunk", "--activation-budget-gb", "1000")
    a = _pass(columnar[kind], tmp_path / "records", *extra)
    b = _pass(columnar[kind], tmp_path / "columns", *extra, "--columns")
    assert len(a) == len(b) >= 3
    for (ra, ga, pa, ma), (rb, gb, pb, mb) in zip(a, b):
        assert ra == rb
        assert all((x is None and y is None) or torch.equal(x, y) for x, y in zip(ga, gb))
        assert all(torch.equal(x, y) for x, y in zip(pa, pb))
        assert ma.keys() == mb.keys() and all(torch.equal(ma[s], mb[s]) for s in ma)


# ---- hand-built game-sides: every field's cases ------------------------------------------

def _hand_masks(g, U, R, H, biased):
    from wesnoth_ai.server_priors import PackedMasks
    A, HB = U + R + 1, (H + 7) // 8
    return PackedMasks(
        actor_mask=(g.random(A) > 0.3).astype(np.uint8), type_valid=(g.random((A, 2)) > 0.4).astype(np.uint8),
        attack_valid=g.integers(0, 256, (A, HB)).astype(np.uint8),
        move_valid=g.integers(0, 256, (A, HB)).astype(np.uint8),
        union_valid=g.integers(0, 256, (A, HB)).astype(np.uint8), n_attacks=g.integers(0, 4, A).astype(np.int8),
        end_turn_bias=float(g.normal()), n_units=U, n_recruits=R, n_hexes=H,
        type_bias=g.normal(size=(A, 2)).astype(np.float32) if biased else None,
        attack_bias=g.normal(size=(A, H)).astype(np.float32) if biased else None)


def _hand_target(g, U, R, H):
    from wesnoth_ai.sequence_records import OwnTarget
    n = int(g.integers(1, 17))
    kind = g.integers(0, 4, n).astype(np.int8)
    prior = g.random(n) + 0.01
    return OwnTarget(actor=g.integers(0, U + R + 1, n).astype(np.int32), kind=kind,
                     target=np.where(kind == 3, -1, g.integers(0, H, n)).astype(np.int32),
                     weapon=np.where(kind == 0, g.integers(0, 3, n), -1).astype(np.int8),
                     prior=prior / (prior.sum() * 1.25), end_prior=float(g.random() * 0.2))


def _hand_label(g, U, R, H):
    from tools.replay_dataset import ActionIndices
    kind = ["attack", "move", "recruit", "end_turn", "timeout"][int(g.integers(0, 5))]
    actor = U + R if kind in ("end_turn", "timeout") else int(g.integers(0, U + R + 1))
    return ActionIndices(action_type=kind, actor_idx=actor,
                         target_idx=None if kind in ("end_turn", "timeout") or g.random() < 0.1 else int(g.integers(0, H)),
                         weapon_idx=int(g.integers(0, 4)) if kind == "attack" else None,
                         type_idx={"attack": 0, "move": 1}.get(kind), target_off_subset=bool(g.random() < 0.1))


def _hand_side(seed, n, own):
    from helpers.model_streams import hand_raw
    from wesnoth_ai.sequence_records import SequencePosition
    g = np.random.default_rng(seed)
    out = []
    for i in range(n):
        U, R, H, S = (int(x) for x in (g.integers(1, 9), g.integers(0, 5), g.integers(3, 30), g.integers(0, 4)))
        raw = hand_raw(seed * 1000 + i, U, R, H, parity=True, S=S)
        raw.their_faction_id, raw.hex_subset = 0, True
        hidden = np.sort(g.choice(H, size=int(g.integers(0, min(H, 4) + 1)), replace=False)).astype(np.int64)
        out.append(SequencePosition(raw=raw, label=_hand_label(g, U, R, H), hidden_tokens=hidden,
                                    no_visible_unit=g.random(H) > 0.4, turn=int(g.integers(1, 30)),
                                    masks=_hand_masks(g, U, R, H, biased=bool(i % 3 == 1)) if own else None,
                                    own_target=_hand_target(g, U, R, H) if own else None))
    return out


def test_hand_built_positions_stage_the_same_bytes_from_columns(tmp_path):
    """Game-sides built by hand with every case the records hold: hidden
    enemy tokens, sightings, timeouts and labels without a target, weapon or
    type, own masks with and without their biases, targets of 1 to 16
    actions. Written as columns and read back, every position equals the
    original; staged from columns and from the positions, batches of rows
    drawn across game-sides and offsets are the same bytes."""
    import types

    from tools.sequence_inputs import ColumnInputs, RecordInputs
    from tools.sequence_own import OwnStreams
    from tools.sequence_setup import build_modules
    from wesnoth_ai.sequence_columns import position_difference, read_sides, write_columns
    from wesnoth_ai.sequence_loss import GameFacts
    from wesnoth_ai.sequence_records import GameSequence
    from wesnoth_ai.sequence_streams import GameSide, Step
    games = {}
    for k, own in enumerate((False, False, True, True)):
        f = f"{'own' if own else 'g'}{k}"
        games[f] = GameSequence(file=f, winner=1 + k % 2, n_commands=40 + k,
                                sides={1: _hand_side(10 * k + 1, 6 + k, own), 2: _hand_side(10 * k + 2, 5, own)})
        write_columns(tmp_path / f, games[f])
    columns = {(f, s): side for f in games for s, side in read_sides(tmp_path / f).items()}
    for (f, s), side in columns.items():
        for i, p in enumerate(games[f].sides[s]):
            assert position_difference(side.position(i), p) is None, (f, s, i)
    encoder, model = build_modules({f"type{i}": i for i in range(40)}, {"": 0, "a": 1}, torch.device("cpu"),
                                   {"d_model": 16, "num_layers": 1, "num_heads": 2, "d_ff": 32})
    own_files = frozenset(f for f in games if f.startswith("own"))
    own = types.SimpleNamespace(files=own_files)
    own.batch = types.MethodType(OwnStreams.batch, own)
    facts = {f: GameFacts(winner=seq.winner, n_commands=seq.n_commands, policy_weight=0.0 if f in own_files else 1.5)
             for f, seq in games.items()}
    args = types.SimpleNamespace(seed=7, value_states_per_game=16)
    common = (encoder, model, own, facts, args, 1.0, {"attack": 1.3, "move": 0.9, "recruit": 1.0}, torch.device("cpu"))
    records = RecordInputs(types.SimpleNamespace(sides=lambda g: games[g.file].sides[g.side]), *common)
    cols = ColumnInputs(types.SimpleNamespace(side=lambda g: columns[(g.file, g.side)]), *common)
    rng = np.random.default_rng(5)
    seen = {"own": 0, "biased": 0, "hidden": 0}
    for trial in range(12):
        keys = list(columns)
        picks = [keys[int(j)] for j in rng.choice(len(keys), size=int(rng.integers(1, 7)))]
        steps = [Step(b, GameSide(f, s), int(rng.integers(0, len(columns[(f, s)]))), 8, False)
                 for b, (f, s) in enumerate(picks)]
        a, b = records.stage(steps), cols.stage(steps)
        assert _inputs_bytes(a) == _inputs_bytes(b), trial
        assert records.sizes(steps) == cols.sizes(steps)
        seen["own"] += a.own is not None
        seen["biased"] += a.own is not None and "attack_bias" in a.own.views
        seen["hidden"] += bool(a.belief_target.sum() > 0)
    assert all(v > 0 for v in seen.values()), seen
