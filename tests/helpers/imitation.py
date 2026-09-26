"""Imitation pairs for the trainer's tests: states from dummy-vs-dummy
sim games, one label per head shape (tests/test_imitation_flat_batch.py,
tests/test_signal_telemetry.py)."""
import random

from tools.replay_dataset import ActionIndices

ARCH = dict(d_model=64, num_layers=2, num_heads=2, d_ff=128)
TYPE_W = {"move": 0.189, "attack": 0.628, "recruit": 1.748, "end_turn": 1.435}


def states(n):
    from tools.bench_states import harvest_states
    from tools.scenario_pool import build_scenario_gamestate, random_setup
    out = [build_scenario_gamestate(random_setup(random.Random(7)))]
    out += harvest_states(n - 1, seed=11)
    return out[:n]


def labels(raws, rng):
    """One label per sample over every head shape: unit moves and
    attacks (type, target, weapon), a recruit, end_turn, an actor
    outside the sample (contributes nothing), a value-only pair and a
    pair without an outcome."""
    kinds = ["move", "attack", "recruit", "end_turn", "skip", "value_only", "no_value", "attack"]
    ais, zw = [], []
    for b, raw in enumerate(raws):
        U, R, H = len(raw.unit_ids), len(raw.recruit_types), int(raw.hex_xs.shape[0])
        A = U + R + 1
        kind = kinds[b % len(kinds)]
        z, vw, pw = (1 if b % 2 else -1), 0.7, 1.0
        if kind == "move" and U:
            ai = ActionIndices("move", rng.randrange(U), target_idx=rng.randrange(H), type_idx=1)
        elif kind == "attack" and U:
            ai = ActionIndices("attack", rng.randrange(U), target_idx=rng.randrange(H),
                               weapon_idx=rng.randrange(2), type_idx=0)
        elif kind == "recruit" and R:
            ai = ActionIndices("recruit", U + rng.randrange(R), target_idx=rng.randrange(H))
        elif kind == "skip":
            ai = ActionIndices("move", A + 3, target_idx=0, type_idx=1)
        elif kind == "value_only":
            ai, pw = ActionIndices("end_turn", A - 1), 0.0
        elif kind == "no_value":
            ai, z, vw = ActionIndices("end_turn", A - 1), None, 0.0
        else:
            ai = ActionIndices("end_turn", A - 1)
        ais.append(ai)
        zw.append((z, vw, pw))
    return ais, zw


def per_sample_reference(model, enc, raws, ais, zw, dev):
    """The per-sample loss path (`encode_from_raw` + `forward_batch` +
    `_loss_parts_for_output`): each pair's LossParts and the batched
    flow's total over them."""
    from tools.supervised_train import _loss_parts_for_output
    encoded = [enc.encode_from_raw(r, device=dev) for r in raws]
    outs = model.forward_batch(encoded)
    parts = [_loss_parts_for_output(o, ai, dev, type_loss_weights=TYPE_W,
                                    value_z=z, value_weight=vw, policy_weight=pw)
             for o, ai, (z, vw, pw) in zip(outs, ais, zw)]
    total = (sum(pw * (p.actor + p.type + p.target + p.weapon) for p, (_, _, pw) in zip(parts, zw))
             + sum(vw * p.value for p, (_, vw, _) in zip(parts, zw)))
    return parts, total
