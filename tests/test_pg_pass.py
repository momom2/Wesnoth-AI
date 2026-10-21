"""The gradient pass (tools/pg_pass.py, tools/pg_schedule.py): the probe
gradients are the logs' sums (P1), the splits add to the total, the
accumulated windows are one big batch's gradient, and a shadow row reads
the memory from before it and writes nothing."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from wesnoth_ai import game_core as gc  # noqa: E402
from wesnoth_ai.pg_records import decisions_of, read_labels  # noqa: E402
from wesnoth_ai.sequence_streams import GameSide, Step  # noqa: E402
from wesnoth_ai.server_priors import KIND_ATTACK, KIND_END_TURN, KIND_RECRUIT  # noqa: E402

needs_core = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")
CPU = torch.device("cpu")
SPLITS = {"g0": 0, "g1": 3, "g2": 5}
R = {("g0", 1): 0.7, ("g0", 2): -0.4, ("g1", 1): -0.9, ("g1", 2): 0.3, ("g2", 1): 0.5, ("g2", 2): -1.1}


def _labels(tmp_path):
    from helpers.pg_games import played_games, write_labels
    games = played_games()
    return games, read_labels(write_labels(tmp_path / "labels.jsonl", SPLITS, R))


def _run(tmp_path, **config):
    from tools.pg_pass import GradientPass, PassConfig
    games, labels = _labels(tmp_path)
    return games, GradientPass(games.checkpoint, games.records, labels, CPU, PassConfig(**config)).run()


def _log_sums(games, R_of):
    """The logs' sum over every decision of R (1[c] - p(c)), by class: for
    attacks p(attack | the sampled actor), no term where its actor has
    no attack."""
    from tools.pg_positions import read_record
    sums = {"end_turn": 0.0, "attack": 0.0, "recruit": 0.0}
    for game, path in games.records.items():
        for d in decisions_of(read_record(path)):
            r = R_of[(game, d.side)]
            sums["end_turn"] += r * ((d.kind == KIND_END_TURN) - d.p_end)
            sums["recruit"] += r * ((d.kind == KIND_RECRUIT) - d.p_rec)
            if not math.isnan(d.p_att_actor):
                sums["attack"] += r * ((d.kind == KIND_ATTACK) - d.p_att_actor)
    return {c: v / (2 * len(games.records)) for c, v in sums.items()}


@needs_core
def test_the_probe_gradients_are_the_logs_sums_and_fail_on_a_shifted_label(tmp_path):
    """P1 end to end: the trainer's gradients at the three probe scalars
    equal the logs' (1/2N) sum R (1[c] - p(c)); with R moved by one
    game-side they do not. The trainer's log pi of every logged action is
    the logged one (the replay identity, fp32)."""
    from tools.pg_positions import read_record
    games, result = _run(tmp_path, streams=2, window=4, fp32=True)
    want = _log_sums(games, R)
    sides = sorted(R)
    shifted = _log_sums(games, {sides[i]: R[sides[(i + 1) % len(sides)]] for i in range(len(sides))})
    tol = 2e-5
    for c in want:
        assert result["probes"][c] == pytest.approx(want[c], abs=tol), c
    assert max(abs(result["probes"][c] - shifted[c]) for c in want) > 50 * tol
    assert abs(want["attack"]) > 50 * tol and abs(want["recruit"]) > 50 * tol
    assert result["replay"]["max_abs"] < 1e-4
    assert result["shadow"] == 2
    assert result["decisions"] == sum(len(decisions_of(read_record(p))) for p in games.records.values())
    print(f"pass: {result['seconds'] / result['decisions'] * 1e3:.2f} ms per position on the CPU "
          f"({result['decisions']} decisions, {result['windows']} windows)")


def _reference_gradient(games):
    """sum over game-sides of R sum_t grad log pi, each game-side walked
    alone with its memory's gradient never cut, by code of its own: the
    network's forward per decision, a shadow row reading the memory before
    it and writing nothing. Divided by 2N, as the pass divides."""
    from tools.pg_pass import load_network, stage_step
    from tools.pg_positions import positions_of, read_record
    from wesnoth_ai.param_groups import named_model_parameters
    from wesnoth_ai.pg_loss import action_log_probs
    encoder, model = load_network(games.checkpoint, CPU)
    total = torch.zeros(())
    for game, path in games.records.items():
        pos = positions_of(read_record(path), *games.vocab, game=game)
        for side, rows in pos.sides.items():
            memory = model.initial_memory(64)
            for p in rows:
                staged = encoder.stage_raws([p.raw], device=CPU)
                out = model.forward_embedded(encoder.embed_staged(staged), memory=[memory])
                total = total + R[(game, side)] * action_log_probs(out, stage_step([p], staged, model, CPU))[0]
                if not p.shadow:
                    memory = out.memory_padded[0, :64]
    named = named_model_parameters(model, encoder)
    grads = torch.autograd.grad(total, [p for _, p in named], allow_unused=True)
    return {n: (torch.zeros_like(p) if g is None else g) / (2 * len(games.records)) for (n, p), g in zip(named, grads)}


def _flat(d, names):
    return torch.cat([d[n].flatten() for n in names])


@needs_core
def test_the_splits_add_to_the_total_and_the_windows_to_one_big_batch(tmp_path):
    """With windows longer than every game-side, the pass's total is the
    gradient of the whole sum taken at once; the eight split sums add to
    it, each split holds its games' part only, and a split of one window
    holds that window's square as its second moment. With short windows
    the memory's gradient is cut and the total moves. The head slopes are
    the total's derivative along each head's own weights."""
    from tools.pg_pass import load_network
    from wesnoth_ai.pg_loss import HEAD_PARAMETERS, HEADS
    games, result = _run(tmp_path, streams=2, window=64, fp32=True)
    names = result["names"]
    total = _flat(result["total"], names)
    ref = _flat(_reference_gradient(games), names)
    assert ref.norm() > 0
    assert torch.allclose(total, ref, rtol=1e-4, atol=1e-6 * float(ref.abs().max()))
    splits = [_flat(s, names) for s in result["splits"]]
    assert torch.allclose(sum(splits), total, atol=1e-7)
    assert [i for i, s in enumerate(splits) if s.norm() > 0] == sorted(SPLITS.values())
    assert result["split_games"] == [1, 0, 0, 1, 0, 1, 0, 0]
    one_window = [i for i, k in enumerate(result["split_windows"]) if k == 1]
    assert one_window
    for i in one_window:
        g_w = splits[i] * 2 * result["games"]
        assert torch.allclose(_flat(result["sq"][i], names), g_w * g_w, rtol=1e-5, atol=1e-12)
    params = {n: p.detach() for n, p in load_network(games.checkpoint, CPU)[1].named_parameters()}
    for head in HEADS:
        along = sum(float((result["total"][n] * params[n[len("model."):]]).sum()) for n in HEAD_PARAMETERS[head])
        assert result["head_slopes"][head] == pytest.approx(along, rel=1e-3, abs=1e-7), head
    _, cut = _run(tmp_path, streams=2, window=2, fp32=True)
    assert not torch.allclose(_flat(cut["total"], names), total, rtol=1e-3, atol=0.0)


@needs_core
def test_a_shadow_row_reads_the_memory_from_before_it_and_writes_nothing():
    """Along one slot: a decision, a refused one, a decision. The third
    reads what the first wrote, through the shadow; the shadow's own write
    reaches nothing. A game-side that starts with a shadow reads the
    initial memory twice."""
    from helpers.pg_games import tiny_network
    from tools.pg_schedule import PGMemoryRouter
    _, _, model = tiny_network()
    router = PGMemoryRouter(model, CPU, {})
    g = GameSide("g", 1)
    steps = [[Step(slot=0, game_side=g, offset=t, k=64, starts=t == 0)] for t in range(3)]
    written = [torch.randn(1, 64, model.d_model, requires_grad=True) for _ in range(3)]
    reads = []
    for t, shadow in enumerate((False, True, False)):
        reads.append(router.batch(steps[t]).padded)
        router.wrote(steps[t], written[t], [shadow])
    assert torch.equal(reads[1], written[0].detach()) and torch.equal(reads[2], written[0].detach())
    reads[2].sum().backward()
    assert written[0].grad is not None and written[0].grad.abs().sum() > 0
    assert written[1].grad is None
    assert torch.equal(router.memories()[0], written[2][0])
    router = PGMemoryRouter(model, CPU, {})
    first = router.batch(steps[0]).padded
    router.wrote(steps[0], written[0], [True])
    second = router.batch(steps[1]).padded
    assert torch.equal(first, second) and torch.equal(first[0], model.initial_memory(64).detach())
    assert np.isfinite(float(second.sum()))
