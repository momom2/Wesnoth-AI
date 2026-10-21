"""The network evaluator (tools/pg_eval.py): the trainer's recomputation of
every held-out decision's log probability is the logged one (the replay
identity), the readings move along the bound direction, and a written
candidate is the network the readings read."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from wesnoth_ai import game_core as gc  # noqa: E402
from wesnoth_ai.pg_records import decisions_of, read_labels  # noqa: E402

needs_core = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")
CPU = torch.device("cpu")
R = {("g0", 1): 0.7, ("g0", 2): -0.4, ("g1", 1): -0.9, ("g1", 2): 0.3, ("g2", 1): 0.5, ("g2", 2): -1.1}
HALVES = {"g0": "A", "g1": "B", "g2": "A"}


def _direction(model, encoder, seed: int = 7):
    """A small random direction over every trained parameter."""
    from wesnoth_ai.param_groups import named_model_parameters
    g = torch.Generator().manual_seed(seed)
    return {n: torch.randn(p.shape, generator=g) * 0.05 * (float(p.detach().abs().mean()) + 1e-3)
            for n, p in named_model_parameters(model, encoder)}


def _evaluator(tmp_path, checkpoint=None, direction=None):
    from helpers.pg_games import played_games, write_labels
    from tools.pg_eval import EvalConfig, EvalContext
    games = played_games()
    labels = read_labels(write_labels(tmp_path / "labels.jsonl", {g: 0 for g in games.records}, R,
                                      heldout=list(games.records), halves=HALVES))
    if direction is None:
        direction = _direction(games.model, games.encoder)
    context = EvalContext(checkpoint or games.checkpoint, games.records, labels, CPU,
                          EvalConfig(streams=2, window=4, kl_fraction=1.0))
    return games, context.bind(direction)


def _logged(games, ev):
    """The logs' values of every held-out decision, in the evaluator's order."""
    from tools.pg_positions import read_record
    rows = []
    for gs in ev.game_sides():
        rows += [d for d in decisions_of(read_record(games.records[gs.game])) if d.side == gs.side]
    return rows


@needs_core
def test_the_recomputed_log_probabilities_are_the_logged_ones(tmp_path):
    """The replay identity in fp32: at theta_k the evaluator's log pi of
    each held-out decision, shadow rows included, memory carried along each
    game-side, is the generator's logged value; the game-sides partition the
    decisions and carry their labels."""
    games, ev = _evaluator(tmp_path)
    sides = ev.game_sides()
    assert [gs.start for gs in sides[1:]] == [gs.stop for gs in sides[:-1]] and sides[0].start == 0
    assert {(gs.game, gs.side): (gs.R, gs.half) for gs in sides} == {k: (r, HALVES[k[0]]) for k, r in R.items()}
    logged = _logged(games, ev)
    base = ev.base_logp()
    assert len(base) == len(logged) == sides[-1].stop
    assert any(d.refused for d in logged)
    assert np.max(np.abs(base - np.array([d.logp for d in logged]))) < 1e-4
    seats = ev.class_mass_by_seat(0.0)
    want = {"end_turn": "p_end", "attack": "p_att", "recruit": "p_rec"}
    for c, key in want.items():
        for seat in (1, 2):
            per_side = [sum(getattr(d, key) for d in logged[gs.start:gs.stop]) for gs in sides if gs.side == seat]
            assert np.allclose(seats[seat][c], per_side, atol=1e-4), (c, seat)
    ev.context.close()


@needs_core
def test_the_readings_move_along_the_bound_direction(tmp_path):
    """Off theta_k the log probabilities move, the KL from pi_k grows as the
    square of the step near 0 and is 0 at it; a written candidate, read as
    the network with no step, gives the step's readings."""
    from tools.pg_pass import load_network
    games, ev = _evaluator(tmp_path)
    ev.evaluate([0.0, 0.5, 1.0])
    base, half, one = ev.logp(0.0), ev.logp(0.5), ev.logp(1.0)
    assert np.max(np.abs(one - base)) > 1e-3
    assert ev.mean_kl(0.0) == 0.0
    k_half, k_one = ev.mean_kl(0.5), ev.mean_kl(1.0)
    assert k_half > 0 and k_one / k_half == pytest.approx(4.0, rel=0.15)
    assert ev.mean_kl(-0.5) / k_half == pytest.approx(1.0, rel=0.15)
    path = tmp_path / "candidate.pt"
    sha = ev.write_candidate(1.0, path)
    assert len(sha) == 64
    encoder, model = load_network(path, CPU)
    _, again = _evaluator(tmp_path, checkpoint=path, direction={})
    assert np.allclose(again.base_logp(), one, atol=1e-5)
    assert not np.allclose(half, one, atol=1e-5)
    ev.context.close()
    again.context.close()
    assert model.memory_slots == 64 and encoder.observation_parity


@needs_core
def test_the_stage_c_factory_binds_evaluators_to_the_pass_network(tmp_path):
    """`network_factory` gives the driver theta_k under the pass's names and
    the checkpoint's hash, and binds protocol evaluators; a direction of the
    wrong shape is refused."""
    from helpers.pg_games import played_games, write_labels
    from tools.eval_provenance import file_sha256
    from tools.pg_eval import network_factory
    from wesnoth_ai.param_groups import named_model_parameters
    from wesnoth_ai.pg_protocol import Evaluator
    games = played_games()
    labels = write_labels(tmp_path / "labels.jsonl", {g: 0 for g in games.records}, R,
                          heldout=list(games.records), halves=HALVES)
    factory, params, sha = network_factory(str(games.checkpoint), str(games.games), str(labels), device="cpu",
                                           streams="2", kl_fraction="1.0")
    assert sha == file_sha256(games.checkpoint)
    named = dict(named_model_parameters(games.model, games.encoder))
    assert set(params) == set(named) and all(torch.equal(params[n], named[n].detach()) for n in named)
    ev = factory({})
    assert isinstance(ev, Evaluator) and len(ev.game_sides()) == 6
    name = next(iter(named))
    with pytest.raises(ValueError):
        factory({name: torch.zeros(3, 1, 7)})
    ev.context.close()


def test_a_held_out_game_whose_seats_sit_in_two_halves_is_refused(tmp_path):
    """Halves are drawn per game: both seats of a held-out game share one."""
    import dataclasses
    from helpers.pg_games import write_labels
    from wesnoth_ai.pg_records import RecordFormatError, labelled_games
    rows = read_labels(write_labels(tmp_path / "labels.jsonl", {"g0": 0}, R, heldout=["g0"], halves={"g0": "A"}))
    assert set(labelled_games(rows, heldout=True)) == {"g0"}
    split = [r if r.side == 1 else dataclasses.replace(r, half="B") for r in rows]
    with pytest.raises(RecordFormatError):
        labelled_games(split, heldout=True)
