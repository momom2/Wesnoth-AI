"""The coin readout's hardening after the 2026-10-20 review: every match is
read whole or left out and failed (its result files, coin rows and walk
rows counted against each other), the walk's fights must lie inside their
supports, I-log reads a recruit's hex (accepting the engine's relocation
only) and an approach's attack origin, each match's slots are counted
against its plan, the margin bins hold their edges, earlier luck that sets
band membership and effect size leaves the labels unbiased, and a pulled
records directory reads offline (tools/coin_labels.py, tools/coin_rows.py)."""
from __future__ import annotations

import gzip
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import coin_labels  # noqa: E402
from tools.coin_labels import build_table, decision_labels, margin_bins, readout  # noqa: E402
from tools.coin_rows import extract_rows, parse_plan, played_status  # noqa: E402
from tools.match_luck import COLUMNS  # noqa: E402
from helpers.coin_games import W, coin_rows_of, effect, entry, synthetic_game  # noqa: E402

BETA0 = np.zeros(len(COLUMNS))


def _games(n, match, seed=0, base=0):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        res, rec, lr, truths = synthetic_game(rng, k, match=match, game=f"game_{match}_{k}")
        out.append(({**res, "seed": base + k}, rec, lr, truths))
    return out


def _two_matches():
    return _games(400, "m1", seed=1), _games(400, "m2", seed=2)


@pytest.mark.parametrize("break_m2, why", [
    (lambda rows, luck: (rows, luck[:-1]), "1 unwalked"),
    (lambda rows, luck: ([r for r in rows if "manifest" not in r], luck), "no manifest"),
    (lambda rows, luck: (rows[:-1] + [{"match": "m2", "manifest": {"results": 401}}], luck),
     "401 result files, 400 coin rows"),
    (lambda rows, luck: (rows[:3] + [{"match": "m2", "game": "x", "seed": 3, "error": "no record"}] + rows[3:], luck),
     "1 extract errors"),
    (lambda rows, luck: (rows, luck[:-1] + [{**luck[-1], "error": "RecordMismatch"}]), "1 walk errors"),
    (lambda rows, luck: (rows, luck + [{**luck[-1], "game": "stray"}]), "1 walk rows unjoined"),
])
def test_a_match_not_read_whole_is_left_out_and_fails_the_verdict(break_m2, why):
    m1, m2 = _two_matches()
    rows2, luck2 = break_m2(coin_rows_of(m2, "m2"), [g[2] for g in m2])
    table = build_table(coin_rows_of(m1, "m1") + rows2, [g[2] for g in m1] + luck2)
    assert table.excluded == ["m2"] and any(why in d for d in table.matches["m2"]["drops"]), table.matches["m2"]
    assert len(table.keys) == 400 and {k[0] for k in table.keys} == {"m1"}
    r = readout(table, BETA0, W)
    assert "matches_whole" in r["verdict"] and "LEFT OUT" in coin_labels.report_text(r)
    whole = readout(build_table(coin_rows_of(m1, "m1") + coin_rows_of(m2, "m2"), [g[2] for g in m1 + m2]), BETA0, W)
    assert whole["checks"]["matches_whole"]["pass"] and whole["games"]["read"] == 800


def test_the_extraction_writes_a_row_for_every_result_file(tmp_path):
    games = _games(5, "m", seed=3)
    d = tmp_path / "games"
    d.mkdir()
    for k, (res, rec, _lr, _t) in enumerate(games):
        (d / f"{rec['game_label']}.json").write_text(json.dumps(res), encoding="utf-8")
        if k == 1:
            continue                                        # a result file without its record
        data = b"not gzip" if k == 2 else gzip.compress((json.dumps(rec) + "\n").encode())
        (d / f"{rec['game_label']}.game.jsonl.gz").write_bytes(data)
    rows = list(extract_rows("m", d))
    assert rows[-1] == {"match": "m", "manifest": {"results": 5}}
    errors = [r for r in rows[:-1] if "error" in r]
    assert len(rows) == 6 and len(errors) == 2 and {e["error"] for e in errors} >= {"no record"}
    table = build_table(rows, [g[2] for g in games])
    assert table.excluded == ["m"] and "2 extract errors" in table.matches["m"]["drops"]


def test_a_fight_outside_its_support_fails_the_readout_and_the_others_are_reported():
    games = _games(300, "m", seed=4)
    luck = [g[2] for g in games]
    luck[7] = {**luck[7], "counts": {**luck[7]["counts"], "outside_support": 1, "not_enumerable": 2}}
    r = readout(build_table(coin_rows_of(games), luck), BETA0, W)
    assert not r["checks"]["luck_support"]["pass"] and r["fights"]["not_enumerable"] == 2
    assert "1 outside their support, 2 not enumerable" in coin_labels.report_text(r)
    luck[7] = {k: v for k, v in luck[7].items() if k != "counts"}
    unread = readout(build_table(coin_rows_of(games), luck), BETA0, W)
    assert not unread["checks"]["luck_support"]["pass"] and unread["fights"]["rows_without_counts"] == 1
    clean = readout(build_table(coin_rows_of(games), [g[2] for g in games]), BETA0, W)
    assert clean["checks"]["luck_support"]["pass"]


RECRUIT = {"kind": "recruit", "unit": "Spearman", "to": [3, 4]}


@pytest.mark.parametrize("cmd, rejected, status", [
    (["recruit", "Spearman", 3, 4, "s"], set(), "matched"),
    (["recruit", "Spearman", 5, 4, "s"], {(2, 3, 4)}, "relocated"),
    (["recruit", "Spearman", 5, 4, "s"], set(), "mismatched"),
    (["recruit", "Spearman", 5, 4, "s"], {(1, 3, 4)}, "mismatched"),
    (["recruit", "Bowman", 3, 4, "s"], set(), "mismatched"),
])
def test_a_recruit_is_read_at_its_hex_and_the_engine_s_relocation_alone_is_accepted(cmd, rejected, status):
    e = {"cmd": 2, "d": 0, "a0": RECRUIT, "a1": {"kind": "end_turn"}}
    assert played_status(e, [["init_side", 1], ["move", [0], [0], 1], cmd], rejected) == status


ATTACK = {"kind": "attack", "from": [1, 1], "to": [4, 4], "weapon": 0}


@pytest.mark.parametrize("commands, status", [
    ([["move", [1, 2, 3], [1, 2, 3], 1], ["attack", 3, 3, 4, 4, 0, 0, "s"]], "matched"),
    ([["move", [1, 2, 3], [1, 2, 3], 1], ["attack", 2, 2, 4, 4, 0, 0, "s"]], "mismatched"),
    ([["move", [1, 2, 3], [1, 2, 3], 1], ["attack", 3, 3, 4, 4, 1, 0, "s"]], "mismatched"),
    ([["move", [1, 2, 3], [1, 2, 3], 1], ["end_turn"]], "unresolved"),
    ([["attack", 1, 1, 4, 4, 0, 0, "s"]], "matched"),
    ([["attack", 2, 1, 4, 4, 0, 0, "s"]], "mismatched"),
])
def test_an_approach_s_attack_starts_where_its_move_ended(commands, status):
    assert played_status({"cmd": 0, "d": 1, "a0": {"kind": "end_turn"}, "a1": ATTACK}, commands) == status


def test_the_readout_counts_each_match_s_slots_against_its_plan(tmp_path):
    games = [g for g in _games(60, "b03", seed=5, base=603000) if g[0]["seed"] - 603000 not in (41, 47, 52)][:50]
    plan_file = tmp_path / "matches.txt"
    plan_file.write_text("Tue Oct 21 2026 b03: games 60 from seed 603000, rc=124 cut, 1 s, results and decisive 50 50\n"
                         "2026-10-21T01:00:00Z a: games 6400 from seed 500000, rc=0 ok, 9 s\n", encoding="utf-8")
    plan = parse_plan(plan_file)
    assert plan == {"b03": (60, 603000), "a": (6400, 500000)}
    table = build_table(coin_rows_of(games, "b03"), [{**g[2], "match": "b03"} for g in games], plan)
    slots = table.matches["b03"]["slots"]
    assert slots["with_result"] == 50 and slots["missing"] == 10 and slots["cut"]
    assert slots["complete_prefix"] == 41 and slots["first_missing"][:3] == [41, 47, 52]
    text = coin_labels.report_text(readout(table, BETA0, W))
    assert "50 of 60 slots have a result" in text and "a prefix of its slots: 0-40 complete, 10 missing" in text
    whole = _games(60, "b03", seed=5, base=603000)
    full = build_table(coin_rows_of(whole, "b03"), [{**g[2], "match": "b03"} for g in whole], plan)
    assert not full.matches["b03"]["slots"]["cut"]


def test_a_gap_on_a_bin_edge_falls_in_the_bin_it_opens():
    gaps = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.3499, 0.35, 0.149999])
    assert margin_bins(gaps, 0.35).tolist() == [0, 1, 2, 3, 4, 5, 6, 6, 6, 2]
    assert int(0.15 / 0.05) == 2, "the floating division the bins no longer use"
    rec = {"winner": 1, "players": {"a": {"side": 1}}, "commands": [],
           "coins": {"decisions": [entry(1, 0, 0.15, 1, ("move", "move"))]}}
    from tools.coin_rows import extract_game
    rows = [extract_game("m", "g", {"outcome_a": "win", "side_a": 1}, rec), {"match": "m", "manifest": {"results": 1}}]
    luck = [{"match": "m", "game": "g", "z": 1, "S": dict.fromkeys(COLUMNS, 0.0), "counts": {}}]
    assert list(readout(build_table(rows, luck), BETA0, W)["labels"]["by_bin"]) == ["[0.15,0.20)"]


def _pre_coin_game(rng, k):
    """A game whose first fight, before any decision, sets how many band
    decisions there are (8 after bad luck for side A, 16 after good) and
    shifts every decision's effect by its sign; that luck is in S and in z."""
    side_a = 1 if k % 2 == 0 else 2
    early = 50.0 * rng.uniform(-1, 1)
    n = 16 if early > 0 else 8
    shift = 0.03 * np.sign(early)
    kinds = rng.integers(3, size=n)
    gaps, coins = rng.uniform(0, W, n), rng.random(n) < 0.25
    pairs = [("move", "move"), ("move", "attack"), ("end_turn", "move")]
    entries, truths, mean = [], [], 0.03 * (1 if side_a == 1 else -1) + 0.0042 * early
    for j in range(n):
        side, pair = 1 + (j // 4) % 2, pairs[int(kinds[j])]
        delta = effect(pair, float(gaps[j])) + shift
        entries.append(entry(side, j, float(gaps[j]), int(coins[j]), pair))
        truths.append(delta)
        mean += (1 if side == side_a else -1) * delta * int(coins[j])
    late = 50.0 * rng.uniform(-1, 1)
    mean += 0.0042 * late
    z = 1 if rng.random() < (1 + np.clip(mean, -1, 1)) / 2 else -1
    game = f"g{k}"
    S = {**dict.fromkeys(COLUMNS, 0.0), "hp": early + late, "material": 0.5 * early}
    res = {"outcome_a": "win" if z == 1 else "loss", "seed": k, "side_a": side_a}
    rec = {"game_label": game, "winner": side_a if z == 1 else 3 - side_a, "players": {"a": {"side": side_a}},
           "commands": [], "coins": {"decisions": entries}}
    return res, rec, {"match": "m", "game": game, "z": z, "S": S, "counts": {}}, truths


@pytest.mark.slow
def test_earlier_luck_that_sets_band_membership_and_effect_size_leaves_the_labels_unbiased():
    """The reviewer's construction, 120,000 games through the production
    extraction, table and labels: per pair type and pooled, the mean label
    lies within 4 SE of the mean known effect, for beta 0 and the fitted
    beta, though the decisions after good luck are more and larger."""
    rng = np.random.default_rng(11)
    games = [_pre_coin_game(rng, k) for k in range(120_000)]
    table = build_table(coin_rows_of(games), [g[2] for g in games])
    truths = np.concatenate([g[3] for g in games])
    from tools.match_luck_readout import fit_beta
    pair = table.k0.astype(int) * 4 + table.k1.astype(int)
    for beta in (BETA0, fit_beta(table.z, table.S)):
        T = decision_labels(table, beta)
        for sel in [np.ones(len(T), bool)] + [pair == c for c in np.unique(pair)]:
            got = coin_labels.clustered_mean(T[sel], table.g[sel])
            assert abs(got["mean"] - truths[sel].mean()) < 4 * got["se"], (got, truths[sel].mean())
    # Without the earlier luck the mean effect would be 0.0225 (the three
    # types' at the mean gap); with it, +0.03 x (16 - 8) / 24 more.
    assert truths.mean() > 0.0275, "the luck before the coins must move the decisions' mix and effects"


def _tarball(path: Path, games, match):
    with tarfile.open(path, "w:gz") as tf:
        for res, rec, _lr, _t in games:
            for name, data in ((f"{rec['game_label']}.json", json.dumps(res).encode()),
                               (f"{rec['game_label']}.game.jsonl.gz", gzip.compress((json.dumps(rec) + "\n").encode()))):
                info = tarfile.TarInfo(f"games_{match}/{name}")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))


def test_a_pulled_records_directory_reads_offline(tmp_path, capsys):
    """games_NAME.tar.gz, rows_NAME.jsonl and matches.txt as the box leaves
    them: the coin rows and the plant are made afresh from the records, the
    slots read against the plan, --matches picks the matches."""
    pulled = tmp_path / "pulled"
    pulled.mkdir()
    for match, base in (("b00", 600000), ("b01", 600300)):
        games = _games(300, match, seed=base, base=base)
        _tarball(pulled / f"games_{match}.tar.gz", games, match)
        (pulled / f"rows_{match}.jsonl").write_text("".join(json.dumps(g[2]) + "\n" for g in games), encoding="utf-8")
    (pulled / "matches.txt").write_text("x b00: games 300 from seed 600000, rc=0\nx b01: games 400 from seed 600300, rc=3\n",
                                        encoding="utf-8")
    beta = tmp_path / "beta.json"
    beta.write_text(json.dumps({"columns": list(COLUMNS), "beta": [0.0] * len(COLUMNS)}), encoding="utf-8")
    out = tmp_path / "readout.json"
    assert coin_labels.main(["readout", "--from-records", str(pulled), "--beta", str(beta), "--band", str(W),
                             "--save-json", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["games"]["read"] == 600 and r["checks"]["matches_whole"]["pass"] and set(r["plant"]) == {"a", "b"}
    assert not r["matches"]["b00"]["slots"]["cut"] and r["matches"]["b01"]["slots"]["missing"] == 100
    assert "a prefix of its slots" in capsys.readouterr().out
    assert coin_labels.main(["readout", "--from-records", str(pulled), "--matches", "b01", "--beta", str(beta),
                             "--band", str(W), "--save-json", str(out)]) == 0
    assert list(json.loads(out.read_text())["matches"]) == ["b01"]
    with pytest.raises(SystemExit):
        coin_labels.main(["readout", "--from-records", str(pulled), "--coins", str(out), "--beta", str(beta), "--band", "0.35"])
