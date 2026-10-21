"""The recruit rule of match L (docs/recruit_rule_l_prereg_20261021.md): its
frozen file and value (tools/recruit_rule.py), R0's rows and capped value
(tools/recruit_rule_fit.py), the decode hook (tools/raw_player.py through
tools/elo_eval_game.py) and L's readout (tools/recruit_rule_readout.py).

The decode reads the rule's inputs off the live game exactly as R0 read
them off the records: batch 1's recorded coin games replayed on the core
give, at every recruit band decision, the decode's inputs and value equal to
R0's. A file whose content does not hash to its hash is refused; the seat
coin lands on each player at 1/2; the capped value is exact under an
interaction a first-order sum misses; in real games the rule plays only its
seat, at most its cap, and the readout rereads every decision; the readout
reads a planted hooked effect and fails a rule played past its cap or not as
logged."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from helpers.rule_games import rule_file, spread_rule, write_games, write_rule  # noqa: E402
from tools.recruit_rule import RecruitRule, earlier_recruits, rule_seat_coin  # noqa: E402
from tools.recruit_rule_fit import ELO, Rows, recruit_rows, rule_readings  # noqa: E402
from wesnoth_ai.game_core import game_core_class  # noqa: E402

needs_core = pytest.mark.skipif(game_core_class() is None, reason="wesnoth_core wheel absent or stale")
FIXTURE = ROOT / "tests" / "fixtures" / "recruit_rule_batch1.game.jsonl.gz"


def frozen_rule_path() -> Path:
    paths = sorted(ROOT.glob("configs/recruit_rule_bd_*[0-9a-f].json"))
    assert len(paths) == 1, paths
    return paths[0]


def test_a_rule_file_whose_content_differs_from_its_hash_is_refused():
    data = rule_file({"Loyalists|Mage": 0.1}, {}, 0.0)
    assert RecruitRule.from_data(data).short_hash == data["hash"][:12]
    tampered = json.loads(json.dumps(data))
    tampered["rule"]["units"]["Loyalists|Mage"] = 0.2
    with pytest.raises(ValueError, match="hashes to"):
        RecruitRule.from_data(tampered)
    RecruitRule.load(frozen_rule_path())


def test_the_value_reads_unit_matchup_and_count_and_counts_the_movers_recruits_only():
    rule = RecruitRule.from_data(rule_file({"L|A": 0.3, "L|B": -0.1}, {"L|A@N": 0.05, "L|B@N": 0.2}, 0.01))
    assert rule.value("L", "N", "B", "A", 4, 1) == pytest.approx((0.3 + 0.1) + (0.05 - 0.2) + 0.03)
    assert rule.value("L", "D", "B", "C", 0, 0) == pytest.approx(0.1)        # keys the fit never saw are 0
    commands = [["init_side", 1], ["recruit", "A", 1, 1, "s"], ["recruit", "B", 1, 2, "s"], ["recruit", "A", 2, 1, "s"],
                ["end_turn"], ["init_side", 2], ["recruit", "B", 9, 9, "s"], ["move", [1, 2], [1, 1], 1]]
    assert earlier_recruits(commands, 1, "A", "B") == (2, 1)
    assert earlier_recruits(commands, 2, "A", "B") == (0, 1)


def test_the_seat_coin_puts_the_rule_on_each_player_at_one_half_apart_from_the_attack_coin():
    from tools.coin_decode import attack_coin
    from tools.recruit_rule_readout import seat_of_seed
    seeds = range(700_000, 720_000)
    d = np.array([rule_seat_coin(s) for s in seeds])
    se = 0.5 / np.sqrt(len(d))
    assert abs(d.mean() - 0.5) < 3 * se
    assert abs(np.mean(d == np.array([attack_coin(s) for s in seeds])) - 0.5) < 3 * se
    assert [seat_of_seed(s) for s in seeds[:500]] == ["a" if x else "b" for x in d[:500]]


@needs_core
def test_the_decode_reads_r0s_inputs_and_value_at_every_recruit_band_decision_of_batch_1_games():
    """Each recorded coin game replayed on the core; at every recruit band
    decision R0 reads off the record, the player's rule step (`_rule_choice`)
    is handed the live state and the simulator's commands so far, and its
    logged inputs, command index and f equal R0's (f by R0's design matrix
    times the frozen rule's weights)."""
    from tools.game_record import read_records, walk_core
    from tools.raw_player import RawPolicyPlayer
    from tools.recruit_rule_readout import recomputed_values
    from wesnoth_ai.game_core import bind_view
    path = frozen_rule_path()
    rule = RecruitRule.load(path)
    rule_data = json.loads(path.read_text(encoding="utf-8"))["rule"]
    seen, factions = 0, set()
    for rec in read_records(FIXTURE):
        rows = recruit_rows(rec)
        at = {}
        for r in rows:
            at.setdefault(r["cmd"], []).append(r)
        player = RawPolicyPlayer(SimpleNamespace(), 0.0, recruit_rule=rule)
        for k, cs, _cmd in walk_core(rec, verify=False):
            for r in at.get(k, ()):
                snap = cs.fork()
                gs = snap.to_state()
                bind_view(gs, snap)
                sim = SimpleNamespace(command_history=[SimpleNamespace(cmd=c) for c in rec["commands"][:k]])
                a0 = {"type": "recruit", "unit_type": r["u0"], "target_hex": SimpleNamespace(x=1, y=1)}
                a1 = {"type": "recruit", "unit_type": r["u1"], "target_hex": SimpleNamespace(x=1, y=2)}
                player._rule_choice("g", (r["side"], r["i"]), (0, 1, r["gap"], 0), gs, sim, a0, a1)
        log = player.pop_rule_log("g")
        assert len(log) == len(rows) > 0
        for e, r in zip(log, rows):
            assert {k: e[k] for k in ("side", "fac", "enemy", "u0", "u1", "n0", "n1", "cmd")} == \
                   {k: r[k] for k in ("side", "fac", "enemy", "u0", "u1", "n0", "n1", "cmd")}
        assert [e["f"] for e in log] == recomputed_values(rule_data, rows).tolist()
        seen += len(rows)
        factions |= {r["fac"] for r in rows}
    assert seen >= 40 and len(factions) == 6


def _rows_of(T, d, gid, rank) -> Rows:
    n = len(T)
    return Rows(match=np.zeros(n, int), gid=gid, side=np.ones(n, int), cmd=rank.copy(), d=d,
                fac=np.full(n, "X"), enemy=np.full(n, "Y"), u0=np.full(n, "X|a"), u1=np.full(n, "X|b"),
                dd=np.zeros(n), st=np.ones(n, int), T=T, K=rank.copy(), dev=np.zeros(n, int),
                side_games=int(gid.max()) + 1, games_by_match={})


def test_the_capped_value_is_exact_under_an_interaction_a_first_order_sum_misses():
    """Three rule rows a side-game, coins at P 1/4; the second row's effect
    grows by g when the first was played: the capped rule (first two rows
    played) is worth e0 + e1 + g over playing neither, the capped reading
    finds it, and its unweighted first-order sum reads e0 + e1 + g/2."""
    rng = np.random.default_rng(3)
    n, p = 40_000, 0.25
    e0, e1, e2, g = 0.2, 0.1, 0.1, 0.3
    d = (rng.random((n, 3)) < p).astype(int)
    mean = e0 * d[:, 0] + (e1 + g * d[:, 0]) * d[:, 1] + e2 * d[:, 2]
    z = np.where(rng.random(n) < (1 + mean) / 2, 1.0, -1.0)
    T = ((d - p) / (p * (1 - p)) * z[:, None]).ravel()
    gid, rank = np.repeat(np.arange(n), 3), np.tile(np.arange(3), n)
    capped = rule_readings(_rows_of(T, d.ravel(), gid, rank), np.ones(3 * n, bool))["capped"]
    truth = (e0 + e1 + g) * ELO
    assert abs(capped["elo"] - truth) < 3 * capped["se"]
    assert abs(capped["first_order"] - (e0 + e1 + g / 2) * ELO) < 3 * capped["se"] < 0.5 * g * ELO


def test_r0_reads_a_records_recruit_rows_with_their_counts_factions_and_prior_coins():
    rec = {"game_label": "g", "setup": {"faction1": "Loyalists", "faction2": "Undead"},
           "commands": [["init_side", 1], ["recruit", "Mage", 1, 1, "s"], ["recruit", "Spearman", 1, 2, "s"],
                        ["end_turn"], ["init_side", 2], ["recruit", "Ghoul", 9, 9, "s"], ["end_turn"],
                        ["init_side", 1], ["recruit", "Spearman", 1, 1, "s"], ["end_turn"]]}

    def band(side, i, cmd, d, u0, u1, kind="recruit", refused=False):
        e = {"side": side, "i": i, "cmd": cmd, "gap": 0.1, "p": 0.25, "d": d,
             "a0": {"kind": kind, "unit": u0}, "a1": {"kind": "recruit", "unit": u1}}
        return {**e, "refused": True} if refused else e
    rec["coins"] = {"decisions": [band(1, 0, 1, 1, "Mage", "Spearman"), band(1, 1, 2, 0, "Spearman", "Spearman"),
                                  band(2, 0, 5, 0, "Ghoul", "Skeleton", refused=True),
                                  band(2, 1, 5, 1, "Skeleton", "Ghoul", kind="move"),
                                  band(1, 3, 8, 0, "Spearman", "Mage")]}
    rows = recruit_rows(rec)
    got = [(r["side"], r["fac"], r["enemy"], r["u0"], r["u1"], r["n0"], r["n1"], r["st"], r["K"], r["dev"])
           for r in rows]
    assert got == [(1, "Loyalists", "Undead", "Mage", "Spearman", 0, 0, 1, 0, 0),
                   (2, "Undead", "Loyalists", "Ghoul", "Skeleton", 0, 0, 1, 2, 1),
                   (1, "Loyalists", "Undead", "Spearman", "Mage", 1, 1, 2, 3, 2)]
    rec["commands"][5] = ["recruit", "Ghoul", 9, 9, "s"]
    rec["coins"]["decisions"][0]["cmd"] = 5
    with pytest.raises(ValueError, match="a decision of side 1 at a command of side 2"):
        recruit_rows(rec)


@pytest.fixture(scope="module")
def spec(tmp_path_factory) -> str:
    import torch
    from wesnoth_ai.transformer_policy import TransformerPolicy
    torch.manual_seed(4)
    path = tmp_path_factory.mktemp("net") / "net.pt"
    TransformerPolicy(device=torch.device("cpu"), d_model=32, num_layers=1, num_heads=2,
                      d_ff=64).save_checkpoint(path)
    return str(path)


@needs_core
def test_the_rule_plays_its_seat_at_most_its_cap_and_the_readout_rereads_each_decision(spec, tmp_path):
    """Band 50 nats and cap 1: the seat coin of each seed names the rule's
    player in the result and the record, both tags carry the rule's hash,
    and the readout finds every decision on the hooked side, its inputs and
    f equal to R0's, the switches its values and cap give, each played as
    logged. (Which decisions arise depends on the platform's numerics, so
    the counts are bounded, not fixed.)"""
    from tools.elo_eval_game import main
    from tools.recruit_rule_readout import extract_rows, readout
    from wesnoth_ai.rules.scenario_pool import load_factions
    recruits = {k: list(v.recruit) for k, v in load_factions().items()}
    data = spread_rule(recruits, band=50.0, cap=1)
    path = write_rule(tmp_path / "rule", data)
    games = tmp_path / "games"
    assert {"a" if rule_seat_coin(s) else "b" for s in (7, 8)} == {"a", "b"}
    for seed in (7, 8):
        assert main(["x", "A", spec, "B", spec, "1", str(seed), str(games), "--mcts-sims", "0",
                     "--raw-temperature-a", "0", "--raw-temperature-b", "0", "--raw-end-turn-offset-a", "-6",
                     "--raw-end-turn-offset-b", "-6", "--max-turns", "5", "--device", "cpu",
                     "--recruit-rule", str(path)]) == 0
        result = json.loads((games / f"game_A_B_s1_{seed}.json").read_text(encoding="utf-8"))
        assert result["procedure_a"] == result["procedure_b"] == f"raw:t0+eo-6+rr{data['hash'][:12]}"
        assert result["recruit_rule_seat"] == ("a" if rule_seat_coin(seed) else "b")
    r = readout(extract_rows("m", games, data), data, {"seal": {"elo": 20.0, "se": 3.0}})
    c = r["checks"]
    assert c["matches_whole"]["pass"] and c["seat_coin_logged"]["pass"] and c["i_cap"]["pass"], c
    assert c["i_cap"]["switches"] >= 1 and c["i_cap"]["max_switches_per_side_game"] == 1


def test_the_rule_switches_at_most_its_cap_each_side_game_and_logs_every_decision():
    """The player's rule step on stub states: a1 where f > 0 until the
    side's cap, a0 after it and wherever f <= 0, each side's cap its own;
    two recruits of one type are not a rule decision."""
    from tools.raw_player import RawPolicyPlayer
    rule = RecruitRule.from_data(rule_file({"L|A": 0.1, "N|A": 0.1}, {}, 0.0, cap=2))
    player = RawPolicyPlayer(SimpleNamespace(), 0.0, recruit_rule=rule)
    sides = [SimpleNamespace(faction="L"), SimpleNamespace(faction="N")]
    sim = SimpleNamespace(command_history=[SimpleNamespace(cmd=["init_side", 1])])

    def act(unit, y):
        return {"type": "recruit", "unit_type": unit, "target_hex": SimpleNamespace(x=1, y=y)}

    def choose(side, i, u0, u1):
        state = SimpleNamespace(global_info=SimpleNamespace(current_side=side), sides=sides)
        return player._rule_choice("g", (side, i), (0, 1, 0.1, 0), state, sim, act(u0, 1), act(u1, 2))["unit_type"]
    played = [choose(1, 0, "B", "A"), choose(1, 1, "A", "B"), choose(1, 2, "B", "A"), choose(1, 3, "B", "B"),
              choose(1, 4, "B", "A"), choose(2, 0, "B", "A")]
    assert played == ["A", "A", "A", "B", "B", "A"]
    log = player.pop_rule_log("g")
    assert [(e["side"], e["i"], e["switched"], e["switches_before"]) for e in log] == \
           [(1, 0, True, 0), (1, 1, False, 1), (1, 2, True, 1), (1, 4, False, 2), (2, 0, True, 0)]
    assert [round(e["f"], 9) for e in log] == [0.1, -0.1, 0.1, 0.1, 0.1] and log[0]["enemy"] == "N"


def test_the_readout_reads_the_hooked_seat_and_fails_a_rule_past_its_cap_or_played_otherwise(tmp_path):
    from tools.recruit_rule_readout import extract_rows, outcome, readout
    from helpers.rule_games import RECRUITS
    data = spread_rule(RECRUITS)
    effect = 0.1
    write_games(tmp_path / "ok", np.random.default_rng(1), data, 800, 720_000, effect)
    r = readout(extract_rows("ok", tmp_path / "ok", data), data, {"seal": {"elo": 20.0, "se": 3.0}})
    want = 400 * np.log10((1 + effect) / (1 - effect))
    h = r["hooked"]
    assert r["verdict"] == "PASS", r["checks"]
    assert abs(h["logit"]["elo"] - want) < 3 * h["logit"]["se"]
    assert abs(h["elo_collect"]["elo"] - h["logit"]["elo"]) < 1.0
    assert r["checks"]["i_cap"]["switches"] > 0 and r["games"]["decisive"] == 800
    for variant, field in (("overcap", "switch_differs"), ("misplayed", "played_not_as_logged")):
        write_games(tmp_path / variant, np.random.default_rng(2), data, 100, 730_000, effect, variant=variant)
        bad = readout(extract_rows(variant, tmp_path / variant, data), data, {"seal": {"elo": 20.0, "se": 3.0}})
        assert bad["checks"]["i_cap"][field] > 0 and "i_cap" in bad["verdict"], bad["checks"]["i_cap"]
    assert [outcome(x, 24.6, 7.0, 3.57) for x in (24.0, 32.0, 10.0, 3.0)] == ["inside", "above", "below", "null"]
