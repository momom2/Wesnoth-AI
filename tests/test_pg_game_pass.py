"""The whole-game gradient pass (tools/pg_game_pass.py): its sum is the
uncut gradient of sum (z - v_t) log pi with the value head's cross-entropy
on the head alone, however the game-sides are grouped; another network's
game-sides are left out; a log it does not reproduce is refused."""
from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from wesnoth_ai import game_core as gc  # noqa: E402
from wesnoth_ai.model_output import VALUE_N_ATOMS  # noqa: E402
from wesnoth_ai.sequence_streams import GameSide  # noqa: E402

needs_core = pytest.mark.skipif(gc.game_core_class() is None, reason="wesnoth_core.GameCore not available")
CPU = torch.device("cpu")
# Side A's result by the suite's names (helpers.pg_games.SUITE): a win, a
# loss and a game without a winner.
OUTCOMES = {"g0": "win", "g1": "loss", "g2": "cap"}


def _games_with_results(tmp_path, outcomes=OUTCOMES):
    """The suite's games copied with side A's results rewritten, the
    records' winners with them: (the suite, {game: record path}, {(game,
    side): z})."""
    from helpers.pg_games import played_games
    from tools.game_record import GameRecordLog
    from tools.pg_positions import read_record, result_path
    games = played_games()
    dst = tmp_path / "games"
    dst.mkdir(parents=True)
    for f in games.games.iterdir():
        shutil.copy(f, dst / f.name)
    records = {stem: dst / path.name for stem, path in games.records.items()}
    z = {}
    for name, outcome in outcomes.items():
        stem = games.names[name]
        path = records[stem]
        res = json.loads(result_path(path).read_text(encoding="utf-8"))
        res["outcome_a"] = outcome
        result_path(path).write_text(json.dumps(res), encoding="utf-8")
        side_a = int(res["side_a"])
        z_a = {"win": 1.0, "loss": -1.0}.get(outcome, 0.0)
        if z_a:
            rec = read_record(path)
            rec["winner"] = side_a if z_a > 0 else 3 - side_a
            path.unlink()
            GameRecordLog(path).write(rec)
        z[(stem, side_a)], z[(stem, 3 - side_a)] = z_a, -z_a
    return games, records, z


def _run(games, records, **config):
    from tools.pg_game_pass import GamePass, GamePassConfig
    return GamePass(games.checkpoint, records, CPU, GamePassConfig(**config)).run()


def _reference(games, records, z, leave_out=()):
    """Each game-side walked alone, its memory's gradient never cut, by code
    of its own (helpers.pg_games.side_forwards): the gradient of
    sum (z - v_t) log pi on every parameter, and on the value head's
    parameters that of minus the cross-entropy of its law against z; divided
    by the game-sides."""
    from helpers.pg_games import side_forwards
    from tools.pg_game_pass import value_cross_entropy
    from tools.pg_pass import load_network
    from tools.pg_positions import positions_of, read_record
    from wesnoth_ai.param_groups import group_of, named_model_parameters
    from wesnoth_ai.pg_loss import action_log_probs
    encoder, model = load_network(games.checkpoint, CPU)
    policy, value, sides = torch.zeros(()), torch.zeros(()), 0
    for game, path in records.items():
        pos = positions_of(read_record(path), *games.vocab, game=game)
        for side, rows in pos.sides.items():
            if (game, side) in leave_out:
                continue
            sides += 1
            for p, (out, batch) in zip(rows, side_forwards(encoder, model, rows)):
                policy = policy + (z[(game, side)] - p.v) * action_log_probs(out, batch)[0]
                value = value - value_cross_entropy(out.value_logits.float(), torch.tensor([z[(game, side)]]))[0]
    named = named_model_parameters(model, encoder)
    params = [p for _, p in named]
    g_policy = torch.autograd.grad(policy, params, allow_unused=True, retain_graph=True)
    g_value = torch.autograd.grad(value, params, allow_unused=True)
    out = {}
    for (n, p), a, b in zip(named, g_policy, g_value):
        g = torch.zeros_like(p) if a is None else a
        if group_of(n) == "value_head" and b is not None:
            g = g + b
        out[n] = g / sides
    return out


def _flat(d, names):
    return torch.cat([d[n].flatten() for n in names])


@needs_core
def test_the_pass_is_the_uncut_gradient_however_the_game_sides_are_grouped(tmp_path):
    """One group holding every game-side gives the gradient of the whole sum
    taken at once, each side weighted by its result less the logged value
    estimate. Small groups, each step recomputed in the backward, give the
    same sum: nothing is cut at a group's edge, since a group holds whole
    game-sides."""
    from tools.pg_positions import read_record
    from wesnoth_ai.pg_records import decisions_of
    games, records, z = _games_with_results(tmp_path)
    one = _run(games, records, streams=8, group_decisions=10 ** 6)
    names = one["names"]
    ref = _flat(_reference(games, records, z), names)
    total = _flat(one["total"], names)
    assert ref.norm() > 0
    assert torch.allclose(total, ref, rtol=1e-4, atol=1e-6 * float(ref.abs().max()))
    assert len(one["groups"]) == 1 and one["game_sides"] == 6 and one["skipped_sides"] == 0
    assert one["decisions"] == sum(len(decisions_of(read_record(p))) for p in records.values())
    assert one["shadow"] == 2 and one["finite"]
    assert one["replay"]["max_abs"] < 1e-4 and abs(one["replay"]["kl"]) < 1e-6
    assert one["norm"] == pytest.approx(float(total.norm()), rel=1e-5)
    small = _run(games, records, streams=2, group_decisions=40, recompute=True)
    assert len(small["groups"]) > 2
    assert torch.allclose(_flat(small["total"], names), total, rtol=1e-4, atol=1e-6 * float(ref.abs().max()))
    assert sum(g["decisions"] for g in small["groups"]) == one["decisions"]


@needs_core
def test_the_value_loss_trains_the_value_head_and_nothing_else(tmp_path):
    """With and without the value loss every parameter outside the value
    head has the same gradient to the bit; the value head has one only with
    it."""
    from wesnoth_ai.param_groups import group_of
    games, records, _ = _games_with_results(tmp_path)
    with_value = _run(games, records, streams=8)
    without = _run(games, records, streams=8, value_loss=False)
    head = [n for n in with_value["names"] if group_of(n) == "value_head"]
    assert head
    for n in with_value["names"]:
        if n in head:
            assert float(without["total"][n].abs().max()) == 0.0, n
        else:
            assert torch.equal(with_value["total"][n], without["total"][n]), n
    assert sum(float(with_value["total"][n].abs().sum()) for n in head) > 0
    assert with_value["value"]["cross_entropy"] > 0 and without["value"]["cross_entropy"] == 0.0


@needs_core
def test_another_networks_game_side_is_left_out_and_a_misplayed_own_one_is_refused(tmp_path):
    """A game-side another checkpoint played (a past self) is not trained
    and is counted; the sum is the other game-sides'. A game-side of the
    pass's own checkpoint played at another memory size stops the pass, and
    so does a result the record's winner contradicts."""
    from tools.game_record import GameRecordLog
    from tools.pg_game_pass import GamePass
    from tools.pg_positions import ForeignGameError, read_record, result_path
    games, records, z = _games_with_results(tmp_path)
    stem = games.names["g1"]
    path = records[stem]
    res = json.loads(result_path(path).read_text(encoding="utf-8"))
    side_b = 3 - int(res["side_a"])
    res["checkpoint_sha256_b"] = "0" * 64
    result_path(path).write_text(json.dumps(res), encoding="utf-8")
    result = _run(games, records, streams=8)
    assert result["skipped_sides"] == 1 and result["game_sides"] == 5
    names = result["names"]
    ref = _flat(_reference(games, records, z, leave_out={(stem, side_b)}), names)
    assert torch.allclose(_flat(result["total"], names), ref, rtol=1e-4, atol=1e-6 * float(ref.abs().max()))

    rec = read_record(path)
    rec["players"]["a"]["memory"] = 0
    path.unlink()
    GameRecordLog(path).write(rec)
    with pytest.raises(ForeignGameError, match=f"{stem} side {3 - side_b}"):
        GamePass(games.checkpoint, records, CPU)

    games, records, _ = _games_with_results(tmp_path / "again")
    won = records[games.names["g0"]]
    rec = read_record(won)
    rec["winner"] = 3 - int(rec["winner"])
    won.unlink()
    GameRecordLog(won).write(rec)
    with pytest.raises(ForeignGameError, match="contradicts"):
        GamePass(games.checkpoint, records, CPU)


@needs_core
def test_a_log_the_pass_does_not_reproduce_is_refused_and_its_sums_kept_aside(tmp_path):
    """Logged log probabilities 0.05 above the network's on one game: the
    replay's KL estimate is 0.05 times that game's share of the decisions,
    above the limit. The command line keeps the sums aside, exits 3 and
    writes no grad.pt; on the games as played it writes grad.pt and its
    scalars."""
    from tools.game_record import GameRecordLog
    from tools.pg_game_pass import EXIT_REFUSED, GamePassRefused, main
    from tools.pg_positions import read_record
    from wesnoth_ai.pg_records import decisions_of
    games, records, _ = _games_with_results(tmp_path)
    out = tmp_path / "grad.pt"
    argv = ["--checkpoint", str(games.checkpoint), "--games", str(tmp_path / "games"), "--out", str(out),
            "--device", "cpu", "--streams", "4"]
    assert main(argv) == 0
    kept = torch.load(out, weights_only=True)
    summary = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert kept["game_sides"] == summary["game_sides"] == 6 and "total" not in summary
    assert math.isfinite(summary["norm"]) and summary["norm"] > 0
    out.unlink()

    shifted = records[games.names["g2"]]
    rec = read_record(shifted)
    for d in rec["decisions"]:
        d["logp"] += 0.05
    shifted.unlink()
    GameRecordLog(shifted).write(rec)
    total = sum(len(decisions_of(read_record(p))) for p in records.values())
    with pytest.raises(GamePassRefused) as caught:
        _run(games, records, streams=4)
    assert caught.value.result["replay"]["kl"] == pytest.approx(0.05 * len(rec["decisions"]) / total, rel=1e-3)
    assert main(argv) == EXIT_REFUSED and not out.exists()
    aside = torch.load(tmp_path / "grad.refused.pt", weights_only=True)
    assert aside["replay"]["kl"] > aside["replay"]["limit"]


def test_groups_are_dealt_longest_first_within_both_caps():
    """Game-sides go to groups longest first; a group closes at its
    game-sides or when the next would pass its decisions; a game-side longer
    than the cap is a group of its own; one without a decision is left
    out."""
    from tools.pg_game_pass import deal_groups
    lengths = {GameSide("a", 1): 50, GameSide("a", 2): 5, GameSide("b", 1): 30, GameSide("b", 2): 30,
               GameSide("c", 1): 0, GameSide("c", 2): 12}
    groups = deal_groups(lengths, streams=2, group_decisions=60)
    assert groups == [[GameSide("a", 1)], [GameSide("b", 1), GameSide("b", 2)], [GameSide("c", 2), GameSide("a", 2)]]
    assert deal_groups(lengths, streams=8, group_decisions=10 ** 6) == [
        [GameSide("a", 1), GameSide("b", 1), GameSide("b", 2), GameSide("c", 2), GameSide("a", 2)]]
    assert deal_groups(lengths, streams=8, group_decisions=20)[0] == [GameSide("a", 1)]
    with pytest.raises(ValueError):
        deal_groups(lengths, streams=0, group_decisions=60)


def test_the_value_cross_entropy_projects_the_result_on_the_atoms():
    """A result on an atom puts all its mass there (+1 the last, -1 the
    first, 0 the middle of an odd count); one between two atoms shares it in
    proportion. Lowering the cross-entropy moves the law's expectation
    toward the result."""
    from tools.pg_game_pass import value_cross_entropy
    K = VALUE_N_ATOMS
    torch.manual_seed(0)
    logits = torch.randn(4, K)
    logp = torch.log_softmax(logits, dim=-1)
    z = torch.tensor([1.0, -1.0, 0.0, 1.0 - 3.0 / (K - 1)])
    ce = value_cross_entropy(logits, z)
    assert ce[0] == pytest.approx(-float(logp[0, K - 1]))
    assert ce[1] == pytest.approx(-float(logp[1, 0]))
    assert K % 2 == 1 and ce[2] == pytest.approx(-float(logp[2, (K - 1) // 2]))
    assert ce[3] == pytest.approx(-0.5 * float(logp[3, K - 3] + logp[3, K - 2]), rel=1e-5)
    logits = torch.zeros(1, K, requires_grad=True)
    value_cross_entropy(logits, torch.tensor([1.0])).sum().backward()
    atoms = torch.linspace(-1.0, 1.0, K)
    assert float((-logits.grad[0] * atoms).sum()) > 0
