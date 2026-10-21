"""The coin labels (tools/coin_labels.py) and their checks
(tools/coin_plant.py) on a synthetic game whose decision effects are known:
the labels are unbiased for the runner-up's minus the argmax's result under
the coin policy whatever beta, a variate read after the coin breaks them,
the plant reads 1 per mover side and its controls read -1 and 0, the dice
control fails a flipped result the plant cannot see, the result's
orientation follows side A, and the attack coin reads the hooked player."""
from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import coin_labels, coin_plant  # noqa: E402
from tools.coin_labels import build_table, decision_labels, readout  # noqa: E402
from tools.coin_rows import extract_game  # noqa: E402
from tools.match_luck import COLUMNS  # noqa: E402
from helpers.coin_games import P, W, attack_game, coin_rows_of, entry, synthetic_game  # noqa: E402


def synthetic(n: int, seed: int = 0, leak: float = 0.0):
    rng = np.random.default_rng(seed)
    games = [synthetic_game(rng, k, leak) for k in range(n)]
    table = build_table(coin_rows_of(games), [g[2] for g in games])
    truths = np.concatenate([g[3] for g in games])
    return games, table, truths


@pytest.fixture(scope="module")
def data():
    return synthetic(12_000)


def _fitted_beta(table) -> np.ndarray:
    from tools.match_luck_readout import fit_beta
    keep = table.decisive
    return fit_beta(table.z[keep], table.S[keep])


def test_labels_are_unbiased_for_the_decision_effect_whatever_beta(data):
    """Per pair type, the mean label lies within 4 standard errors of the
    mean known effect under the coin policy (a runner-up attack's coin also
    brings a decision whose expected effect it carries), for beta 0, the
    least-squares beta, and five times it; the fitted beta reads tighter."""
    _games, table, truths = data
    keep = table.decisive[table.g]
    pair = table.k0 * 4 + table.k1
    ses = {}
    for name, beta in (("zero", np.zeros(len(COLUMNS))), ("fit", _fitted_beta(table)),
                       ("five", 5 * _fitted_beta(table))):
        T = decision_labels(table, beta)
        for c in np.unique(pair[keep]):
            sel = keep & (pair == c)
            got = coin_labels.clustered_mean(T[sel], table.g[sel])
            assert abs(got["mean"] - truths[sel].mean()) < 4 * got["se"], (name, coin_labels.pair_name(c // 4, c % 4), got)
            ses[(name, int(c))] = got["se"]
    assert all(ses[("fit", c)] < ses[("zero", c)] for c in np.unique(pair[keep]))
    # The margin slope of the attack type reads in its bins.
    r = readout(table, _fitted_beta(table), W)
    bins = r["labels"]["by_bin"]
    assert len(bins) == 7 and sum(b["n"] for b in bins.values()) == int(keep.sum())


def test_the_labels_read_the_effect_where_a_wrong_formula_would_not(data):
    """The test can fail: dropping the mover's sign moves the mean label of
    the move>move type (effect +0.08) by many SE."""
    _games, table, truths = data
    keep = table.decisive[table.g] & (table.k0 == 1) & (table.k1 == 1)
    beta = _fitted_beta(table)
    R = coin_labels.residuals(table, beta)[table.g]
    omega = (table.d - table.p) / (table.p * (1 - table.p))
    right = coin_labels.clustered_mean((table.s * omega * R)[keep], table.g[keep])
    wrong = coin_labels.clustered_mean((omega * R)[keep], table.g[keep])
    assert abs(wrong["mean"] - truths[keep].mean()) > 5 * wrong["se"]
    assert abs(right["mean"] - truths[keep].mean()) < 4 * right["se"]


def test_a_variate_read_after_the_coin_biases_the_labels():
    """I2's negative control: a luck column that counts the coins (signed
    by mover) has mean zero given nothing, but not given the coin; with a
    beta on it the labels move off the known effects."""
    _games, table, truths = synthetic(12_000, seed=5, leak=12.0)
    keep = table.decisive[table.g]
    beta = np.zeros(len(COLUMNS))
    beta[COLUMNS.index("hp")] = 0.004
    T = decision_labels(table, beta)
    got = coin_labels.clustered_mean(T[keep], table.g[keep])
    assert abs(got["mean"] - truths[keep].mean()) > 6 * got["se"]


def test_capped_games_are_dropped_counted_and_their_coins_checked(data):
    _games, table, _truths = data
    r = readout(table, _fitted_beta(table), W)
    assert r["games"]["capped"] == int((table.z == 0).sum()) > 0
    assert r["band"]["decisions"] == int(table.decisive[table.g].sum())
    assert r["i_coin"]["capped"]["n"] == int((~table.decisive[table.g]).sum())
    assert r["checks"]["i_coin_overall"]["pass"] and r["checks"]["i_coin_capped"]["pass"]
    assert r["checks"]["i_coin_gap"]["pass"] and r["checks"]["i_coin_index"]["pass"]
    psg = r["band"]["per_side_game"]
    assert psg["side_games"] == 2 * r["games"]["decisive"] and 8 < psg["mean"] < 10


def test_i_coin_fails_coins_that_do_not_land_at_their_propensity(data):
    _games, table, _truths = data
    import dataclasses
    biased = dataclasses.replace(table, p=np.full_like(table.p, 0.24))
    r = readout(biased, _fitted_beta(table), W)
    assert not r["checks"]["i_coin_overall"]["pass"]
    tied = dataclasses.replace(table, d=(table.gap < 0.1).astype(float))
    assert not readout(tied, _fitted_beta(table), W)["checks"]["i_coin_gap"]["pass"]


def test_the_result_is_side_a_s_and_the_mover_s_sign_follows_side_a():
    """A game with side A on side 2: a win for side A is z = +1 and a
    decision by side 2 has s = +1, one by side 1 s = -1."""
    rec = {"winner": 2, "players": {"a": {"side": 2}}, "commands": [],
           "coins": {"decisions": [entry(2, 0, 0.1, 1, ("move", "move")), entry(1, 0, 0.2, 0, ("move", "attack"))]}}
    row = extract_game("m", "g", {"outcome_a": "win", "side_a": 2}, rec)
    manifest = {"match": "m", "manifest": {"results": 1}}
    table = build_table([row, manifest], [{"match": "m", "game": "g", "z": 1, "S": dict.fromkeys(COLUMNS, 0.0)}])
    assert table.z.tolist() == [1.0] and table.s.tolist() == [1.0, -1.0]
    T = decision_labels(table, np.zeros(len(COLUMNS)))
    assert T.tolist() == pytest.approx([1 / P, -(-P) / (P * (1 - P))])
    with pytest.raises(ValueError, match="contradicts"):
        extract_game("m", "g", {"outcome_a": "loss", "side_a": 2}, rec)
    with pytest.raises(ValueError, match="join"):
        build_table([row, manifest], [{"match": "m", "game": "g", "z": -1, "S": dict.fromkeys(COLUMNS, 0.0)}])


def _plant(games, table, beta):
    rows = [coin_plant.planted_row("m", rec["game_label"], rec, 1.0, "move>move", "test") for _res, rec, _l, _t in games]
    return readout(table, beta, W, rows)


def test_the_coin_plant_reads_one_per_mover_side(data):
    games, table, _truths = data
    r = _plant(games, table, _fitted_beta(table))
    for side in ("a", "b"):
        assert r["checks"][f"plant_{side}"]["pass"], r["plant"]
        assert r["plant"][side]["se"] < 0.03 and r["plant"][side]["planted"] > 10_000


def test_a_wrong_mover_sign_reads_minus_one(data, monkeypatch):
    games, table, _truths = data
    monkeypatch.setattr(coin_labels, "mover_sign", lambda side, side_a: -np.where(side == side_a, 1, -1))
    flipped = build_table(coin_rows_of(games), [g[2] for g in games])
    r = _plant(games, flipped, _fitted_beta(flipped))
    for side in ("a", "b"):
        p = r["plant"][side]
        assert abs(p["slope"] + 1) < 4 * p["se"] and not r["checks"][f"plant_{side}"]["pass"]


def test_a_coin_read_one_decision_off_reads_zero(data, monkeypatch):
    games, table, _truths = data
    monkeypatch.setattr(coin_labels, "coin_outcomes", lambda band: np.roll(np.asarray(band["d"], float), 1))
    shifted = build_table(coin_rows_of(games), [g[2] for g in games])
    r = _plant(games, shifted, _fitted_beta(shifted))
    for side in ("a", "b"):
        p = r["plant"][side]
        assert abs(p["slope"]) < 4 * p["se"] and not r["checks"][f"plant_{side}"]["pass"]


def test_the_dice_control_fails_a_flipped_result_the_plant_cannot_see(data):
    """z read with the wrong orientation, in the coin rows and the walk's
    rows alike (so the join holds): the plant still reads 1, the dice
    control reads about -0.0042 and fails."""
    games, table, _truths = data
    beta = _fitted_beta(table)
    good = _plant(games, table, beta)
    assert good["checks"]["dice_control"]["pass"], good["checks"]["dice_control"]
    flip = {"win": "loss", "loss": "win"}
    flipped, luck = [], []
    for res, rec, lr, _t in games:
        res = {**res, "outcome_a": flip.get(res["outcome_a"], res["outcome_a"])}
        rec = {**rec, "winner": 0 if rec["winner"] == 0 else 3 - rec["winner"]}
        flipped.append((res, rec))
        luck.append({**lr, "z": -lr["z"]})
    bad_table = build_table(coin_rows_of(flipped), luck)
    plant_rows = [coin_plant.planted_row("m", rec["game_label"], {**rec, "winner": 0 if rec["winner"] == 0 else 3 - rec["winner"]},
                                         1.0, "move>move", "test") for _res, rec, _l, _t in games]
    bad = readout(bad_table, beta, W, plant_rows)
    assert bad["checks"]["plant_a"]["pass"] and bad["checks"]["plant_b"]["pass"]
    dc = bad["checks"]["dice_control"]
    assert not dc["pass"] and dc["slope"] < 0


def test_the_dice_window_is_the_design_s():
    lo, hi = coin_plant.dice_window(2_000)
    assert (round(lo, 5), round(hi, 5)) == (0.00245, 0.00587)
    lo, hi = coin_plant.dice_window(100_000)
    assert (round(lo, 5), round(hi, 5)) == (0.00268, 0.00565)
    rng = np.random.default_rng(1)
    s = rng.normal(0, 75, 100_000)
    z = np.where(rng.random(len(s)) < (1 + np.clip(0.0042 * s, -1, 1)) / 2, 1.0, -1.0)
    assert coin_plant.dice_control(z, s)["pass"] and not coin_plant.dice_control(-z, s)["pass"]


def _attack_games(n, effect, seed=2):
    rng = np.random.default_rng(seed)
    games = [attack_game(rng, k, effect) for k in range(n)]
    return build_table(coin_rows_of(games), [lr for _res, _rec, lr in games])


def test_the_attack_coin_reads_twice_the_hooked_player_s_effect():
    """Delta -0.2247 on the hooked player's result: the label reads 2 Delta
    and the hooked player's Elo about -78, inside the design's windows; a
    hook that lowers the other player reads +0.449 and fails both."""
    table = _attack_games(20_000, 0.2247)
    r = readout(table, np.zeros(len(COLUMNS)), W)
    a = r["attack"]
    assert abs(a["label"]["mean"] + 0.449) < 4 * a["label"]["se"]
    assert abs(a["hooked_elo"]["elo"] - coin_labels.elo_of_mean(-0.2247, 0.0)[0]) < 4 * a["hooked_elo"]["se"]
    assert r["checks"]["attack_label"]["pass"] and r["checks"]["attack_elo"]["pass"]
    wrong = readout(_attack_games(20_000, -0.2247), np.zeros(len(COLUMNS)), W)
    assert not wrong["checks"]["attack_label"]["pass"] and not wrong["checks"]["attack_elo"]["pass"]
    assert "labels" not in r and r["checks"]["attack_coin"]["pass"]


def _write_games(directory: Path, games) -> None:
    directory.mkdir(parents=True)
    for res, rec, _l, _t in games:
        (directory / f"{rec['game_label']}.json").write_text(json.dumps(res), encoding="utf-8")
        (directory / f"{rec['game_label']}.game.jsonl.gz").write_bytes(gzip.compress((json.dumps(rec) + "\n").encode()))


def test_the_command_line_extracts_plants_and_reads_a_games_directory(tmp_path, capsys):
    """extract and plant on a games directory, each rerun adding nothing,
    then the readout from the files, with a beta file."""
    games, _table, _truths = synthetic(300, seed=9)
    _write_games(tmp_path / "games", games)
    coins, plant, rows = tmp_path / "coins.jsonl", tmp_path / "plant.jsonl", tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps(g[2]) + "\n" for g in games), encoding="utf-8")
    beta = tmp_path / "beta.json"
    beta.write_text(json.dumps({"columns": list(COLUMNS), "beta": [0.0] * len(COLUMNS)}), encoding="utf-8")
    for _ in range(2):
        assert coin_labels.main(["extract", "--match", f"m={tmp_path / 'games'}", "--out", str(coins)]) == 0
        assert coin_plant.main(["plant", "--match", f"m={tmp_path / 'games'}", "--out", str(plant)]) == 0
    coin_lines = [json.loads(line) for line in coins.read_text().splitlines()]
    assert len([r for r in coin_lines if "game" in r]) == len(plant.read_text().splitlines()) == 300
    assert [r["manifest"] for r in coin_lines if "manifest" in r] == [{"results": 300}] * 2
    assert {json.loads(line)["pair"] for line in plant.read_text().splitlines()} == {"end_turn>move"}
    out = tmp_path / "readout.json"
    assert coin_labels.main(["readout", "--coins", str(coins), "--rows", str(rows), "--beta", str(beta),
                             "--band", str(W), "--plant", str(plant), "--save-json", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["games"]["read"] == 300 and set(r["plant"]) == {"a", "b"} and r["checks"]["matches_whole"]["pass"]
    assert "verdict:" in capsys.readouterr().out
    assert math.isfinite(r["labels"]["pooled"]["mean"])
