"""The luck terms of tools/match_luck.py and the readout of
tools/match_luck_readout.py: an attack's luck has mean zero over the
fight's outcomes and over the simulator's own dice, a leader's death is a
lost game only when it was its side's last, a recorded game's attacks land
in their enumerated supports, a walk resumes past a row cut short, the
fetch keeps only the pinned tarballs, the cross-fitted correction never
fits on the game it corrects, and the readout recovers a known luck share
and pools only the matches played at argmax."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

from tools import match_luck  # noqa: E402
from tools import match_luck_readout as readout_mod  # noqa: E402
from tools.match_luck import COLUMNS  # noqa: E402

# A one-row corridor of three grass hexes inside an impassable border.
CORRIDOR = "Xv, Xv, Xv, Xv, Xv\nXv, Gg, Gg, Gg, Xv\nXv, Xv, Xv, Xv, Xv"


def _duel(defender_hp: int, attacker_one_xp_short: bool = False,
          attacker: str = "Spearman", defender: str = "Dwarvish Fighter", extra_units=()):
    """(the duel's CoreState, the attack command with seed "0"): two
    leaders side by side, side 1's attacking side 2's, the defender at
    `defender_hp`, so the fight can end the game; `extra_units` stand on
    the corridor's third hex."""
    from tools.replay_dataset import _build_initial_gamestate
    from wesnoth_ai.game_core import CoreState
    data = {"game_id": "luck", "map_data": CORRIDOR,
            "starting_units": [
                {"uid": 1, "type": attacker, "side": 1, "x": 0, "y": 0, "is_leader": True},
                {"uid": 2, "type": defender, "side": 2, "x": 1, "y": 0, "is_leader": True},
                *extra_units],
            "starting_sides": [{"side": k, "gold": 0, "recruit": []} for k in (1, 2)]}
    try:
        cs = CoreState.from_state(_build_initial_gamestate(data))
    except RuntimeError as e:                        # no current wheel: the suite's banner says so
        pytest.skip(str(e))
    core = cs.core
    att = next(u for u in core.units_export() if u["side"] == 1)
    dfd = next(u for u in core.units_export() if u["side"] == 2)
    core.update_unit(dfd["id"], {"current_hp": defender_hp})
    if attacker_one_xp_short:
        core.update_unit(att["id"], {"current_exp": att["max_exp"] - 1})
    counter = core.counter_weapon_choice(0, 0, 1, 0, 0)[0]
    return cs, ["attack", 0, 0, 1, 0, 0, counter, "0"]


def _fight(cs, cmd, side_a=1):
    core = cs.core
    probs, att_id, dfd_id = core.attack_outcomes(*cmd[1:6], True)
    leaders = match_luck.leader_counts(core.units_export())
    fighters = (match_luck.fighter_of(core.unit_export(att_id), side_a, leaders),
                match_luck.fighter_of(core.unit_export(dfd_id), side_a, leaders))
    rest = match_luck.rest_material(core.units_export(), side_a, {att_id, dfd_id})
    return match_luck.coarse_distribution(probs, fighters), fighters, rest


@pytest.mark.parametrize("side_a", [1, 2])
def test_the_luck_of_a_fight_has_mean_zero_over_its_outcomes(side_a):
    cs, cmd = _duel(defender_hp=12)
    dist, fighters, rest = _fight(cs, cmd, side_a)
    assert sum(p for states, p in dist.items() if states[1][0] == match_luck.DEAD) > 0.1, \
        "the defending leader must be able to die, so the terminal columns are exercised"
    for column in COLUMNS:
        mean = sum(p * match_luck.luck_terms(dist, states, fighters, rest)[column]
                   for states, p in dist.items())
        assert abs(mean) < 1e-12, (column, mean)
    killed = next(s for s in dist if s[1][0] == match_luck.DEAD)
    assert match_luck.column_values(fighters, killed, rest)["terminal"] == (1.0 if side_a == 1 else -1.0)


def test_a_leader_killed_ends_the_game_only_when_it_was_its_sides_last():
    """A side keeps playing while it has a leader (the simulator's
    _check_game_over): a second leader beside the defender makes its
    death a loss of material, not a lost game."""
    second = {"uid": 3, "type": "Dwarvish Fighter", "side": 2, "x": 2, "y": 0, "is_leader": True}
    cs, cmd = _duel(defender_hp=12, extra_units=[second])
    dist, fighters, rest = _fight(cs, cmd)
    killed = next(s for s in dist if s[1][0] == match_luck.DEAD)
    values = match_luck.column_values(fighters, killed, rest)
    assert values["terminal"] == 0.0
    assert values["material_nonterminal"] == pytest.approx(rest + values["material"])
    assert rest < 0, "the second leader is side B's material"


@pytest.mark.parametrize("case", ["plain", "one_xp_short", "plague"])
def test_the_realized_luck_averages_to_zero_over_the_simulators_dice(case):
    """The attack played on forks of one position with many seeds: the
    simulator's own strikes are the distribution the expectation must be
    taken over, with, one experience point short, its advancement to the
    first type (which the matches use), and with a plague attack, the
    corpse that rises under the victim's id. A wrong distribution (uniform
    weights over the outcomes, say) moves the mean by many standard
    errors."""
    if case == "plague":
        cs, cmd = _duel(defender_hp=6, attacker="Walking Corpse", defender="Spearman")
    else:
        cs, cmd = _duel(defender_hp=12, attacker_one_xp_short=case == "one_xp_short")
    n = 1500
    terms = []
    advanced = 0
    for i in range(n):
        a = match_luck.attack_luck(cs, cmd[:7] + [f"{(i * 2654435761) % 2**32:08x}"], 1)
        assert a.reason == "enumerated", a.reason
        terms.append(a.terms)
        advanced += a.advanced
    if case == "one_xp_short":
        assert advanced > n // 2, "the attacker survives and advances in most fights"
    else:
        assert advanced == 0
    if case == "plague":
        kills = sum(1 for t in terms if t["terminal"] > 0)
        assert 0.1 * n < kills < 0.9 * n, kills
    for column in COLUMNS:
        xs = np.array([t[column] for t in terms])
        se = xs.std(ddof=1) / math.sqrt(n)
        assert se > 0, column
        assert abs(xs.mean()) < 4.5 * se, (column, xs.mean(), se)


def test_a_recorded_game_is_walked_into_a_row(tmp_path):
    """A mini game played and recorded as a match records it, walked by the
    production path: every attack enumerated and realized inside its
    support, the luck columns filled."""
    import wesnoth_ai.dummy_policy as dummy_policy
    from sim_test_helpers import Brawler, scenario_setup
    from tools import game_record
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate
    setup = scenario_setup(3, mini=True)
    try:
        sim = WesnothSim(build_scenario_gamestate(setup), scenario_id=setup.scenario_id, max_turns=6)
    except RuntimeError as e:
        pytest.skip(str(e))
    sim._seed_salt = "luck-test"
    cap = dummy_policy._BOOTSTRAP_UNITS
    dummy_policy._BOOTSTRAP_UNITS = 8
    try:
        policy = Brawler()
        while not sim.done:
            sim.step(policy.select_action(sim.gs, game_label="luck"))
    finally:
        dummy_policy._BOOTSTRAP_UNITS = cap
    game = "game_luck_test_1"
    rec = game_record.game_record(sim, setup, game_label=game,
                                  players={"a": {"side": 2}, "b": {"side": 1}})
    games = tmp_path / "games"
    game_record.GameRecordLog(games / f"{game}.game.jsonl.gz").write(rec)
    outcome = {2: "win", 1: "loss"}.get(int(sim.winner), "capped")
    (games / f"{game}.json").write_text(json.dumps({"outcome_a": outcome, "side_a": 2, "turns": 6}),
                                        encoding="utf-8")
    out = tmp_path / "rows.jsonl"
    assert match_luck.walk_matches([("m", games)], out) == 1
    assert match_luck.walk_matches([("m", games)], out) == 0, "a rerun skips the games already written"
    [row] = readout_mod.load_rows(out)
    assert "error" not in row, row.get("error")
    counts = row["counts"]
    assert counts["attacks"] > 0
    assert counts.get("outside_support", 0) == 0
    assert counts["attacks"] == sum(counts.get(k, 0) for k in (
        "enumerated", "outside_support", "not_enumerable", "counter_weapon_differs", "no_units"))
    assert counts["enumerated"] > 0
    assert row["z"] == {"win": 1, "loss": -1}.get(outcome, 0)
    assert all(math.isfinite(row["S"][c]) for c in COLUMNS)


def test_cross_fitting_never_fits_on_the_game_it_corrects():
    rng = np.random.default_rng(4)
    n = 200
    X = rng.normal(size=(n, 2))
    z = np.sign(X @ np.array([0.7, -0.3]) + rng.normal(size=n))
    folds = readout_mod.two_folds(n, seed=0)
    assert abs(int(folds.sum()) - n // 2) <= 1
    G, betas = readout_mod.cross_fit(z, X, folds)
    i = int(np.flatnonzero(folds == 0)[0])
    z2, X2 = z.copy(), X.copy()
    z2[i], X2[i] = -z[i], X[i] * 50.0
    G2, betas2 = readout_mod.cross_fit(z2, X2, folds)
    assert np.allclose(betas2[0], betas[0]), "the beta correcting game i moved with game i"
    assert np.isclose(G2[i], z2[i] - X2[i] @ betas[0])
    same_fold = (folds == 0) & (np.arange(n) != i)
    assert np.allclose(G2[same_fold], G[same_fold])
    assert not np.allclose(betas2[1], betas[1]), "game i does enter the other fold's fit"


def _synthetic_rows(n, b, sigma, seed):
    """Games whose result is the sign of b x luck + skill noise, the luck
    in the material column and noise elsewhere."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        luck = float(rng.normal())
        z = 1 if b * luck + sigma * rng.normal() > 0 else -1
        S = {c: float(rng.normal()) for c in COLUMNS}
        S["material"] = luck
        rows.append({"match": f"m{k % 2}", "game": f"g{k:05d}", "z": z, "decisive": True, "turns": 10 + k % 30,
                     "scenario_id": f"map{k % 3}", "S": S, "SS": {c: S[c] ** 2 for c in COLUMNS},
                     "counts": {"attacks": 1, "enumerated": 1},
                     "raw_temperature_a": 0.0, "raw_temperature_b": 0.0})
    return rows


def test_the_readout_recovers_a_known_luck_share():
    """For z = sign(b L + sigma e) with L and e standard normal, the best
    linear correction removes (2 / pi) b^2 / (b^2 + sigma^2) of Var(z)."""
    b, sigma = 1.0, 1.0
    truth = 1.0 - (2.0 / math.pi) * b * b / (b * b + sigma * sigma)
    result = readout_mod.readout(_synthetic_rows(8000, b, sigma, seed=1), seed=3, n_boot=200)
    material = result["pooled"]["material"]
    assert abs(material["ratio"] - truth) < 0.03, (material["ratio"], truth)
    lo, hi = material["ratio_ci95"]
    assert lo < truth < hi
    assert abs(material["equivalent_games"] - 800 * material["ratio"]) < 1e-9
    assert abs(material["bias_t"]) < 4
    assert abs(result["pooled"]["joint"]["ratio"] - truth) < 0.03
    assert result["pooled"]["hp"]["ratio"] > 0.98, "a column of noise removes nothing"
    for m in result["matches"].values():
        assert abs(m["corrected"]["material"]["ratio"] - truth) < 0.05


def test_the_pooled_fit_reads_only_matches_played_at_argmax():
    rows = _synthetic_rows(600, 1.0, 1.0, seed=2)
    for r in rows:
        if r["match"] == "m1":
            r["raw_temperature_a"] = 0.5
    result = readout_mod.readout(rows, seed=0, n_boot=50)
    assert result["pooled_matches"] == ["m0"] and result["not_pooled"] == ["m1"]
    assert result["pooled_facts"]["walked"] == 300
    assert set(result["pooled_by_match"]) == {"m0"}
    assert result["matches"]["m1"]["corrected"], "a match left out of the pool is still read alone"


def test_a_last_row_cut_short_is_dropped_and_its_game_walked_again(tmp_path):
    """A walk cut mid-write (or a copy of rows.jsonl taken while a row was
    written) leaves a partial last line: the next walk drops it, keeps
    every whole row and appends after a newline."""
    out = tmp_path / "rows.jsonl"
    whole = (json.dumps({"match": "m", "game": "game_1"}) + "\n").encode()
    out.write_bytes(whole + b'{"match": "m", "game": "ga')
    games = tmp_path / "games"
    games.mkdir()
    assert match_luck.walk_matches([("m", games)], out) == 0
    assert out.read_bytes() == whole
    assert match_luck.done_keys(out) == {("m", "game_1")}


def test_fetch_lists_the_pinned_tarballs_and_leaves_out_the_others(tmp_path):
    host = tmp_path / "host"
    host.mkdir()
    (host / "games_good.tar.gz").write_bytes(b"the games played")
    (host / "games_changed.tar.gz").write_bytes(b"other games")
    pin = match_luck.sha256_of(host / "games_good.tar.gz")
    specs = [match_luck.parse_spec(f"{name}=tier-b/run/games_{name}.tar.gz={pin}")
             for name in ("good", "changed", "absent")]

    def download(path, data):
        local = host / path.rsplit("/", 1)[-1]
        if not local.exists():
            raise FileNotFoundError(path)
        return local

    manifest = tmp_path / "data.txt"
    assert match_luck.fetch_matches(specs, tmp_path / "data", manifest, download=download) == 1
    [line] = manifest.read_text(encoding="utf-8").splitlines()
    assert line.split("\t") == ["good", str(host / "games_good.tar.gz"), "16", "tier-b/run/games_good.tar.gz"]
    for bad in ("good=tier-b/run/games_good.tar.gz", f"good=tier-b/run/games_good.json={pin}",
                f"good=tier-b/run/games_good.tar.gz={pin[:-1]}X"):
        with pytest.raises(argparse.ArgumentTypeError):
            match_luck.parse_spec(bad)
