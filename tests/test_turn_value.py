"""The turn-ranking value function (docs/turn_value_prereg_20260925.md):
positions drawn from recorded games, each candidate's pre-end_turn
state rebuilt and read by the frozen trunk exactly as the player read it
while playing, the arms' objective, and the pre-registered rule."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

from tools import turn_value, turn_value_data, turn_value_fit  # noqa: E402
from tools.game_record import GameRecordLog, game_record  # noqa: E402


@pytest.fixture(scope="module")
def recorded():
    from test_game_record import _played_game
    sim, setup, _ = _played_game(max_turns=5)
    return json.loads(json.dumps(game_record(sim, setup, game_label="g", build={})))


@pytest.fixture(scope="module")
def games_dir(tmp_path_factory, recorded):
    """Three recorded mini games, one per file, as a match writes them."""
    d = tmp_path_factory.mktemp("games")
    for name in ("game_a.game.jsonl.gz", "game_b.game.jsonl.gz", "game_c.game.jsonl.gz"):
        GameRecordLog(d / name).write(recorded)
    return d


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory, recorded):
    """A random tiny net whose vocab holds every unit type and faction
    of the recorded game, recruit lists included (a type outside the
    vocab reads differently live, where the vocab grows, and offline,
    where it maps to the overflow row)."""
    from tools.game_record import walk
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(1)
    pol = TransformerPolicy(device=torch.device("cpu"), d_model=32, num_layers=1,
                            num_heads=2, d_ff=64)
    for _k, gs, _cmd in walk(recorded):
        pol._inference_encoder.register_names(gs)
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    pol.save_checkpoint(path)
    return str(path)


def test_turn_starts_are_picked_by_turn_side_and_cap():
    commands = [["init_side", 1], ["move"], ["end_turn"], ["init_side", 2], ["end_turn"],
                ["init_side", 3], ["end_turn"],
                ["init_side", 1], ["end_turn"], ["init_side", 2], ["end_turn"],
                ["init_side", 1], ["move"], ["end_turn"], ["init_side", 2]]
    sel = turn_value_data.Selection(per_game=10, min_turn=2, seed=1)
    picks = turn_value_data.pick_turn_starts(commands, "g", sel)
    # turn 2 for both player sides and turn 3 for side 1; the neutral
    # side and the record's last command are not turn starts to measure
    assert picks == [(7, 2, 1), (9, 2, 2), (11, 3, 1)]
    capped = turn_value_data.pick_turn_starts(commands, "g", turn_value_data.Selection(2, 2, 1))
    assert len(capped) == 2 and set(capped) <= set(picks)
    assert capped == turn_value_data.pick_turn_starts(commands, "g", turn_value_data.Selection(2, 2, 1))


def test_splits_are_by_game_and_do_not_depend_on_the_seed():
    names = [f"game_{i}.game.jsonl.gz" for i in range(50)]
    splits = turn_value_data.assign_splits(names, proxy=10, stop=5)
    assert sorted(splits.values()).count("proxy") == 10
    assert sorted(splits.values()).count("stop") == 5
    assert turn_value_data.assign_splits(list(reversed(names)), 10, 5) == splits


def _generate(games_dir, checkpoint, out, per_game="2"):
    return turn_value_data.main([
        "turn_value_data", "--games-dir", str(games_dir), "--out", str(out),
        "--checkpoint", checkpoint, "--per-game", per_game, "--alternatives", "1",
        "--continue-edits", "1", "--cap-turns", "1", "--proxy-games", "1",
        "--stop-games", "1", "--device", "cpu", "--jobs", "1"])


@pytest.fixture(scope="module")
def generated(tmp_path_factory, games_dir, checkpoint):
    out = tmp_path_factory.mktemp("data") / "data.jsonl.gz"
    assert _generate(games_dir, checkpoint, out) == 0
    return out


def test_a_candidate_is_read_offline_as_the_player_read_it(generated, games_dir, checkpoint):
    """Generated positions rebuild from their game records, each
    candidate's pre-end_turn state from its snapshot, and the frozen
    trunk's value head reads that state as the player's own value head
    read it while playing."""
    out = generated
    records = turn_value.load_positions(out)
    assert len(records) == 6
    assert sorted(r["meta"]["split"] for r in records) == ["fit"] * 2 + ["proxy"] * 2 + ["stop"] * 2
    model, encoder = turn_value.load_reference_model(Path(checkpoint), torch.device("cpu"))
    cache = turn_value.build_cache(records, model, encoder, torch.device("cpu"),
                                   games_dir=games_dir, dataset=Path("unused"),
                                   jobs=1, batch=3)
    live = {}
    for r in records:
        for slot, cand in enumerate(turn_value.candidates(r)):
            if cand["pre_end_turn"]:
                live[(r["index"], slot)] = (cand["pre_end_turn"]["value_pre"],
                                            float(np.mean(cand["outcomes"])))
    keys = list(zip(cache["index"].tolist(), cache["slot"].tolist()))
    assert sorted(keys) == sorted(live) and len(keys) >= 8
    for row, key in enumerate(keys):
        assert cache["value_reference"][row].item() == pytest.approx(live[key][0], abs=1e-4)
        assert cache["y"][row].item() == live[key][1]
    assert cache["feats"].shape == (len(keys), model.d_model)
    assert cache["turn_luck"].shape == (len(keys), 2) and torch.isfinite(cache["turn_luck"]).all()

    # The same run continues where it stopped; another run is refused.
    size = out.stat().st_size
    assert _generate(games_dir, checkpoint, out) == 0
    assert out.stat().st_size == size
    with pytest.raises(SystemExit):
        _generate(games_dir, checkpoint, out, per_game="3")


def test_fit_and_evaluate_write_a_verdict_for_each_arm(tmp_path, generated, games_dir,
                                                      checkpoint):
    """The box's last stages end to end: features for the training log
    and for a validation file, both arms fitted over every label and
    rank weight, every judged grader given a verdict by the rule, and
    every grader read on the proxy games."""
    train = tmp_path / "train.pt"
    base = ["turn_value", "features", "--checkpoint", checkpoint, "--games-dir", str(games_dir)]
    assert turn_value.main(base + ["--positions", str(generated), "--out", str(train)]) == 0
    validation = tmp_path / "validation.json"
    validation.write_text(json.dumps({"positions": turn_value.load_positions(generated)}),
                          encoding="utf-8")
    val_cache = tmp_path / "validation.pt"
    assert turn_value.main(base + ["--positions", str(validation), "--out", str(val_cache)]) == 0
    heads = tmp_path / "heads"
    assert turn_value.main(["turn_value", "fit", "--train", str(train),
                            "--out-dir", str(heads)]) == 0
    verdict = tmp_path / "verdict.json"
    assert turn_value.main(["turn_value", "evaluate", "--heads", str(heads),
                            "--validation", f"{validation}={val_cache}", "--train", str(train),
                            "--out", str(verdict)]) == 0
    out = json.loads(verdict.read_text(encoding="utf-8"))
    assert {"linear", "head", "rollout", "value_reference", "value_pre"} <= set(out["validation"])
    for name in turn_value_fit.JUDGED:
        assert out["validation"][name]["verdict"].split()[0] in (
            "PASS", "FAIL", "INCONCLUSIVE", "UNDECIDED")
    assert {"linear", "head", "value_reference"} <= set(out["proxy"])
    assert out["counts"]["train"]["positions"] == {"fit": 2, "stop": 2, "proxy": 2}
    cache = torch.load(train, weights_only=False)
    proxy_rows = [s == "proxy" for s in cache["split"]]
    assert all(torch.isfinite(cache["outcomes"][i, :2]).all() for i, p in enumerate(proxy_rows) if p)


def test_the_linear_arm_solves_its_objective():
    """At the closed-form solution the objective's gradient vanishes, at
    every rank weight; the ranking term ignores what is common to a
    position."""
    gen = torch.Generator().manual_seed(0)
    feats = torch.randn(60, 5, generator=gen, dtype=torch.float64)
    groups = turn_value_fit.position_ids(torch.arange(60) // 3)
    y = (feats[:, 0] + 0.3 * torch.randn(60, generator=gen, dtype=torch.float64)).clamp(-1, 1)
    w = torch.ones(60, dtype=torch.float64)
    problem = turn_value_fit.LinearProblem(feats, w, groups)
    for rank_weight in (0.0, 1.0, 10.0):
        arm = problem.solve(y, rank_weight, ridge=0.01)
        theta = torch.cat([arm.coef, torch.tensor([arm.intercept], dtype=torch.float64)])
        theta.requires_grad_(True)
        v = ((feats - arm.mean) / arm.std) @ theta[:-1] + theta[-1]
        objective = (turn_value_fit.turn_loss(v, y, w, groups, rank_weight)
                     + 0.01 * (theta[:-1] ** 2).sum())
        objective.backward()
        assert theta.grad.abs().max().item() < 1e-8, rank_weight
    v = v.detach()
    shift = torch.randn(20, generator=gen, dtype=torch.float64)[groups]
    within = turn_value_fit.within_error
    assert float(within(v + shift, y, w, groups)) == pytest.approx(float(within(v, y, w, groups)))


def test_the_corrected_correlation_recovers_the_true_one():
    """Graders whose within-position correlation with the true candidate
    values is known, read against 20 win/loss playouts per candidate:
    the observed correlation is attenuated by the playout noise, the
    corrected one lands on the truth."""
    rng = np.random.default_rng(1)
    n_positions, n_cands, n_playouts = 400, 5, 20
    values = np.clip(rng.normal(0, 0.3, (n_positions, 1))
                     + rng.normal(0, 0.2, (n_positions, n_cands)), -0.95, 0.95)
    wins = rng.random((n_positions, n_cands, n_playouts)) < (1 + values[..., None]) / 2
    truth = np.where(wins, 1.0, -1.0).reshape(-1, n_playouts)
    positions = np.repeat(np.arange(n_positions), n_cands)
    games = positions // 2
    within = (values - values.mean(axis=1, keepdims=True)).ravel()
    for target in (0.5, 0.8):
        noise_sd = within.std() * math.sqrt((1 / target ** 2 - 1) / (1 - 1 / n_cands))
        grade = values.ravel() + rng.normal(0, noise_sd, values.size)
        g_within = grade - np.repeat(grade.reshape(n_positions, n_cands).mean(axis=1), n_cands)
        true_r = float(np.corrcoef(g_within, within)[0, 1])
        stats = turn_value_fit.corrected_correlation(grade, truth, positions, games)
        assert abs(stats["corrected"] - true_r) < 3 * stats["corrected_se"], (target, stats)
        assert stats["observed"] < true_r - 0.1
        assert 0.02 < stats["corrected_se"] < 0.1


def test_luck_adjusted_labels_keep_their_mean_and_shed_noise():
    """When the outcome follows the luck of the playout's fights and of
    the candidate turn's own, subtracting the fitted luck terms leaves
    the labels' mean and lowers their variance."""
    rng = np.random.default_rng(2)
    n = 4000
    luck = rng.normal(0, 1, (n, 1, 2))
    turn = rng.normal(0, 1, (n, 2))
    p_win = 1 / (1 + np.exp(-(0.2 + 1.5 * luck[:, 0, 0] - 0.5 * luck[:, 0, 1]
                              + 0.8 * turn[:, 0])))
    outcomes = np.where(rng.random(n) < p_win, 1.0, -1.0)[:, None]
    part = {"outcomes": torch.tensor(outcomes), "luck": torch.tensor(luck),
            "turn_luck": torch.tensor(turn), "horizon_value": torch.zeros(n, 1, 0)}
    beta, r2, used = turn_value_fit.luck_coefficients(part)
    assert used == n and r2 > 0.1
    assert beta[0] > 0 > beta[1] and beta[2] > 0 and abs(beta[3]) < beta[2] / 3
    raw = turn_value_fit.labels(part, turn_value_fit.LabelSpec(0.0, False), beta)
    adjusted = turn_value_fit.labels(part, turn_value_fit.LabelSpec(0.0, True), beta)
    assert abs(float(adjusted.mean() - raw.mean())) < 3 * float(raw.std()) / math.sqrt(n)
    assert float(adjusted.var()) < 0.9 * float(raw.var())
