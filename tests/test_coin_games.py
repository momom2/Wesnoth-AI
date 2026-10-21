"""The coins in real games, through the per-game worker
(tools/elo_eval_game.py main) and the match driver: off, a game and its
record are the decode's before the coins, bit for bit; on, every band
decision is in the record, the command played at its index is the coin's
choice (I-log through tools/coin_labels.py's extraction), the tags and the
result's counts say so, and the attack coin's side is the one its seed
names."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wesnoth_ai.game_core import game_core_class  # noqa: E402

needs_core = pytest.mark.skipif(game_core_class() is None, reason="wesnoth_core wheel absent or stale")


@pytest.fixture(scope="module")
def spec(tmp_path_factory) -> str:
    import torch
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(4)
    path = tmp_path_factory.mktemp("net") / "net.pt"
    TransformerPolicy(device=torch.device("cpu"), d_model=32, num_layers=1, num_heads=2,
                      d_ff=64).save_checkpoint(path)
    return str(path)


def _play(spec: str, out: Path, *flags, seed: int = 7, side_a: int = 1):
    from tools.elo_eval_game import main
    from tools.game_record import read_records
    assert main(["x", "A", spec, "B", spec, str(side_a), str(seed), str(out), "--mcts-sims", "0",
                 "--raw-temperature-a", "0", "--raw-temperature-b", "0", "--raw-end-turn-offset-a", "-6",
                 "--raw-end-turn-offset-b", "-6", "--max-turns", "3", "--device", "cpu", *flags]) == 0
    stem = f"game_A_B_s{side_a}_{seed}"
    result = json.loads((out / f"{stem}.json").read_text(encoding="utf-8"))
    record, = list(read_records(out / f"{stem}.game.jsonl.gz"))
    return result, record


def _old_choose(self, priors, actors, is_end, is_attack, coin_key=None, is_recruit=None):
    """The decode before the coins and the recruit rule: the end_turn
    shift, then the end_turn rule."""
    from tools.raw_player import actor_rule_index, pick_index, shift_end_turn_prior
    priors = shift_end_turn_prior(priors, is_end, self.end_turn_offset)
    if self.end_turn_rule == "actor":
        return actor_rule_index(priors, actors, is_end, self.temperature, self._rng), None
    return pick_index(priors, self.temperature, self._rng), None


@needs_core
def test_without_coins_a_game_and_its_record_are_the_old_decode_s(spec, tmp_path, monkeypatch):
    from tools.raw_player import RawPolicyPlayer
    monkeypatch.setattr(RawPolicyPlayer, "_rule_choice", lambda *a, **k: pytest.fail("the rule asked with it off"))
    result, record = _play(spec, tmp_path / "now")
    assert "coins" not in record and "recruit_rule" not in record
    assert not any(k.startswith(("coin_", "band_", "attack_coin", "recruit_rule", "rule_")) for k in result)
    assert result["procedure_a"] == result["procedure_b"] == "raw:t0+eo-6"
    monkeypatch.setattr(RawPolicyPlayer, "_choose", _old_choose)
    monkeypatch.setattr(RawPolicyPlayer, "_log_band", lambda *a, **k: pytest.fail("a band logged with coins off"))
    _old_result, old_record = _play(spec, tmp_path / "old")
    assert len(record["commands"]) > 20
    assert json.dumps(record, sort_keys=True) == json.dumps(old_record, sort_keys=True)


@needs_core
def test_with_coins_every_band_decision_is_recorded_and_played_as_its_coin_said(spec, tmp_path):
    """Band 50 nats (nearly every decision with two actions or more), P
    1/2, on both sides: the record holds each side's band decisions in
    order, each with G below the band, P, D, a0 and a1; the command at
    each one's index is a1 where D is 1 and a0 where it is 0; the game
    differs from the one without coins."""
    from tools.coin_rows import extract_game
    _off, plain = _play(spec, tmp_path / "off")
    result, record = _play(spec, tmp_path / "on", "--coin-band-a", "50", "--coin-band-b", "50",
                           "--coin-p-a", "0.5", "--coin-p-b", "0.5")
    assert result["procedure_a"] == result["procedure_b"] == "raw:t0+eo-6+coin50p0.5"
    coins = record["coins"]
    assert coins["bands"] == {"a": {"band": 50.0, "p": 0.5}, "b": {"band": 50.0, "p": 0.5}}
    entries = coins["decisions"]
    for side, player in ((1, "a"), (2, "b")):
        mine = [e for e in entries if e["side"] == side]
        assert len(mine) == result[f"band_decisions_{player}"] > 5
        assert sum(e["d"] for e in mine) == result[f"coin_ones_{player}"]
        assert all(i < j for i, j in zip([e["i"] for e in mine], [e["i"] for e in mine][1:]))
    assert {e["d"] for e in entries} == {0, 1}
    assert all(0 <= e["gap"] < 50 and e["p"] == 0.5 and e["a0"] != e["a1"] for e in entries)
    assert [e["cmd"] for e in entries] == sorted(e["cmd"] for e in entries)
    row = extract_game("m", record["game_label"], result, record)
    assert row["i_log"]["mismatched"] == 0 and row["i_log"]["matched"] >= 0.9 * len(entries)
    assert record["commands"] != plain["commands"]


@needs_core
def test_the_attack_coin_puts_the_offset_on_the_side_its_seed_names(spec, tmp_path):
    from tools.coin_decode import attack_coin
    from tools.elo_eval_game import _raw_coins
    seeds = {attack_coin(s): s for s in range(100, 120)}
    assert set(seeds) == {0, 1}
    for d, seed in seeds.items():
        result, record = _play(spec, tmp_path / f"seed{seed}", "--attack-coin-offset", "1", seed=seed)
        seat = "a" if d else "b"
        assert record["coins"]["attack"] == {"offset": 1.0, "p": 0.5, "d": d, "seat": seat}
        assert record["coins"]["decisions"] == [] and result["attack_coin_seat"] == seat
        assert result["procedure_a"] == result["procedure_b"] == "raw:t0+eo-6+aoc1"
    from argparse import Namespace
    args = Namespace(attack_coin_offset=1.0, coin_band_a=0.0, coin_band_b=0.35, coin_p_b=0.25, seed=9)
    assert _raw_coins(args, "a", "a") == {"attack_offset": 1.0}
    assert _raw_coins(args, "b", "a") == {"coin_band": 0.35, "coin_p": 0.25, "coin_seed": 9}


@needs_core
def test_a_coin_game_never_joins_a_reference_outdir(spec, tmp_path):
    _play(spec, tmp_path / "games")
    with pytest.raises(SystemExit, match="refusing to mix"):
        _play(spec, tmp_path / "games", "--coin-band-a", "0.35")


def test_a_refused_band_decision_is_marked_in_the_log():
    from types import SimpleNamespace
    from tools.raw_player import RawPolicyPlayer
    player = RawPolicyPlayer(SimpleNamespace(), 0.0, coin_band=0.35, coin_seed=1)
    state = SimpleNamespace(global_info=SimpleNamespace(current_side=2))
    key = player._coin_key("g", state)
    player._log_band("g", key, (0, 1, 0.1, 1), None, {"type": "end_turn"},
                     {"type": "move", "start_hex": SimpleNamespace(x=1, y=2), "target_hex": SimpleNamespace(x=2, y=2)})
    player.drop_last_pending("g")
    later = player._coin_key("g", state)
    player.drop_last_pending("g")              # a decision that was not a band one
    log = player.pop_coin_log("g")
    assert key == (2, 0) and later == (2, 1) and len(log) == 1 and log[0]["refused"] is True
    assert log[0]["a1"] == {"kind": "move", "from": [1, 2], "to": [2, 2]} and player.pop_coin_log("g") == []


@needs_core
@pytest.mark.slow
def test_the_driver_plays_coin_games_and_tags_them(spec, tmp_path):
    """run_elo_batch hands the coins to each game (a child process per
    game): the results carry the coin tags, the records the coins."""
    from tools.game_record import read_records
    from tools.run_elo_batch import main
    out = tmp_path / "batch"
    rc = main(["x", "--label-a", "A", "--spec-a", spec, "--label-b", "B", "--spec-b", spec, "--outdir", str(out),
               "--games", "2", "--max-extra-games", "0", "--mcts-sims", "0", "--raw-temperature-a", "0",
               "--raw-temperature-b", "0", "--coin-band-a", "50", "--coin-band-b", "50", "--attack-coin-offset", "1",
               "--max-turns", "2", "--device", "cpu", "--jobs", "1", "--min-free-mb", "0", "--time-budget-min", "20"])
    assert rc in (0, 4)
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("game_*.json"))]
    assert len(results) == 2
    for res in results:
        assert res["procedure_a"] == res["procedure_b"] == "raw:t0+aoc1+coin50p0.25"
        stem = f"game_A_B_s{res['side_a']}_{res['seed']}"
        record, = list(read_records(out / f"{stem}.game.jsonl.gz"))
        assert record["coins"]["attack"]["seat"] == res["attack_coin_seat"]
        assert np.sum([e["side"] == res["side_a"] for e in record["coins"]["decisions"]]) == res["band_decisions_a"]
