"""The near-tie coins and the attack coin at the raw player's decode
(tools/coin_decode.py, tools/raw_player.py): the coins' frequencies and
independence, the band decision's choice, the off path's choice, the
procedure tags and the drivers' refusals."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from tools.coin_decode import attack_coin, coin_uniform, near_tie, near_tie_coin  # noqa: E402
from tools.raw_player import RawPolicyPlayer, pick_index, shift_attack_prior, shift_end_turn_prior  # noqa: E402

N = 100_000


def _within(ones: int, n: int, p: float, k: float = 4.0) -> bool:
    return abs(ones - n * p) <= k * np.sqrt(n * p * (1 - p))


def _corr(a, b) -> float:
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


@pytest.mark.parametrize("p", [0.25, 0.1])
def test_a_coin_lands_one_at_its_propensity(p):
    d = [near_tie_coin(seed, 1 + k % 2, k // 2, p) for k, seed in enumerate(range(N))]
    assert _within(sum(d), N, p)
    assert _within(sum(attack_coin(s) for s in range(N)), N, 0.5)


def test_coins_are_independent_of_each_other_of_the_index_and_of_the_dice():
    """Each pair of streams a game draws from correlates at most as noise
    does (|r| below 4 / sqrt(n)): the two sides' coins at one index, a side's
    coins at consecutive indices, the coin and the index, the coin and the
    simulator's dice seed of the same game and request (the first 8 hex
    digits of sha256 of "<salt>:<request>", WesnothSim._next_seed, under an
    eval game's salt "elo:<seed>"), the near-tie and attack coins."""
    seeds = np.arange(20_000)
    line = 4 / np.sqrt(len(seeds) * 10)
    u1, u2, u_next, u_dice, idx = [], [], [], [], []
    for s in seeds:
        for i in range(10):
            u1.append(coin_uniform("near_tie_coin", s, 1, i))
            u2.append(coin_uniform("near_tie_coin", s, 2, i))
            u_next.append(coin_uniform("near_tie_coin", s, 1, i + 1))
            dice_seed = hashlib.sha256(f"elo:{s}:{i + 1}".encode()).hexdigest()[:8]
            u_dice.append(int(dice_seed, 16) / 2 ** 32)
            idx.append(i)
    for other in (u2, u_next, u_dice, idx):
        assert abs(_corr(u1, other)) < line
    att = [coin_uniform("attack_coin", s) for s in seeds]
    first = [coin_uniform("near_tie_coin", s, 1, 0) for s in seeds]
    assert abs(_corr(att, first)) < 4 / np.sqrt(len(seeds))


def test_a_coin_is_a_pure_function_of_its_key():
    assert [near_tie_coin(7, 2, 31, 0.25) for _ in range(3)] == [near_tie_coin(7, 2, 31, 0.25)] * 3
    assert len({coin_uniform("near_tie_coin", 7, 2, i) for i in range(1000)}) == 1000


def test_near_tie_takes_the_first_maximum_then_the_first_of_the_others():
    a0, a1, gap = near_tie(np.array([0.1, 0.4, 0.2, 0.4, 0.3]))
    assert (a0, a1, gap) == (1, 3, 0.0)
    a0, a1, gap = near_tie(np.array([0.1, 0.5, 0.25, 0.15]))
    assert (a0, a1) == (1, 2) and gap == pytest.approx(np.log(2.0))
    assert near_tie(np.array([1.0])) is None
    assert near_tie(np.array([1.0, 0.0]))[2] == np.inf


def _player(**kw):
    return RawPolicyPlayer(SimpleNamespace(), 0.0, end_turn_offset=kw.pop("end_turn_offset", -1.5), **kw)


def _old_choose(priors, is_end, offset):
    """The decode before the coins: the end_turn shift, then the argmax."""
    return pick_index(shift_end_turn_prior(priors, is_end, offset), 0.0, np.random.default_rng(0))


def test_without_coins_the_choice_is_the_old_decode_on_every_prior():
    rng = np.random.default_rng(3)
    player = _player()
    for _ in range(2000):
        n = int(rng.integers(1, 40))
        priors = rng.dirichlet(np.full(n, 0.3))
        is_end = np.zeros(n, bool)
        is_end[int(rng.integers(n))] = True
        is_attack = rng.random(n) < 0.3
        pick, band = player._choose(priors, np.arange(n), is_end, is_attack)
        assert band is None and pick == _old_choose(priors, is_end, -1.5)
    assert shift_attack_prior(priors, is_attack, 0.0) is priors


def test_a_band_decision_plays_the_runner_up_exactly_when_its_coin_lands_one():
    """Where the shifted priors' two best differ by less than w, the coin
    (keyed by seed, side, index) picks a1 or a0; at a gap of w or more the
    argmax plays whatever the coin would have said."""
    player = _player(coin_band=0.35, coin_p=0.25, coin_seed=11)
    priors = np.array([0.05, 0.30, 0.25, 0.40])       # end_turn at 3, shifted below the others
    is_end = np.array([False, False, False, True])
    no_attack = np.zeros(4, bool)
    shifted = shift_end_turn_prior(priors, is_end, -1.5)
    a0, a1, gap = near_tie(shifted)
    assert gap < 0.35 and (a0, a1) == (1, 2)
    seen = set()
    for index in range(200):
        pick, band = player._choose(priors, np.arange(4), is_end, no_attack, (2, index))
        d = near_tie_coin(11, 2, index, 0.25)
        assert band == (a0, a1, pytest.approx(gap), d) and pick == (a1 if d else a0)
        seen.add(d)
    assert seen == {0, 1}
    wide = np.array([0.05, 0.60, 0.10, 0.25])
    for index in range(50):
        pick, band = player._choose(wide, np.arange(4), is_end, no_attack, (1, index))
        assert band is None and pick == 1


def test_the_attack_offset_raises_every_attack_by_the_same_factor():
    priors = np.array([0.4, 0.3, 0.2, 0.1])
    is_attack = np.array([False, True, True, False])
    out = shift_attack_prior(priors, is_attack, 1.0)
    assert out.sum() == pytest.approx(1.0)
    assert out[1] / out[0] == pytest.approx(np.e * 0.3 / 0.4) and out[3] / out[0] == pytest.approx(0.25)
    player = _player(end_turn_offset=0.0, attack_offset=1.0)
    assert player._choose(priors, np.arange(4), np.zeros(4, bool), is_attack)[0] == 1
    assert _player(end_turn_offset=0.0)._choose(priors, np.arange(4), np.zeros(4, bool), is_attack)[0] == 0


@pytest.mark.parametrize("kw, why", [
    ({"coin_band": 0.35, "coin_seed": 1, "end_turn_rule": "actor"}, "joint argmax"),
    ({"coin_band": 0.35, "coin_seed": 1, "coin_p": 1.0}, "strictly between"),
    ({"coin_band": 0.35}, "coin_seed"),
    ({"coin_band": -0.1}, ">= 0"),
])
def test_the_player_refuses_coins_it_cannot_play(kw, why):
    with pytest.raises(ValueError, match=why):
        RawPolicyPlayer(SimpleNamespace(), 0.0, **kw)
    with pytest.raises(ValueError, match="joint argmax"):
        RawPolicyPlayer(SimpleNamespace(), 0.5, coin_band=0.35, coin_seed=1)


def test_a_coin_player_has_a_tag_of_its_own():
    from tools.eval_procedure import procedure_of
    ref = procedure_of(0, False, False, 0.0, raw_end_turn_offset=-1.5)
    assert ref == "raw:t0+eo-1.5"
    assert procedure_of(0, False, False, 0.0, raw_end_turn_offset=-1.5, coin_band=0.35,
                        coin_p=0.25) == "raw:t0+eo-1.5+coin0.35p0.25"
    assert procedure_of(0, False, False, 0.0, raw_end_turn_offset=-1.5,
                        attack_coin_offset=1.0) == "raw:t0+eo-1.5+aoc1"
    assert procedure_of(0, False, False, 0.0, raw_end_turn_offset=-1.5, coin_p=0.25) == ref


@pytest.mark.parametrize("side_kw, why", [
    ({"sims": 32}, "joint argmax"),
    ({"raw_temperature": 0.5}, "joint argmax"),
    ({"raw_end_turn": "actor"}, "joint argmax"),
    ({"spec": "dummy"}, "joint argmax"),
    ({"coin_p": 0.0}, "strictly between"),
])
def test_coin_flags_that_would_not_apply_are_refused(side_kw, why):
    from tools.eval_procedure import coin_refusal
    kw = {"spec": "x.pt", "sims": 0, "raw_temperature": 0.0, "raw_end_turn": "joint", "coin_band": 0.35,
          "coin_p": 0.25, "attack_coin_offset": 0.0}
    assert coin_refusal("a", **kw) is None
    assert why in coin_refusal("a", **{**kw, **side_kw})
    attack = {**kw, "coin_band": 0.0, "attack_coin_offset": 1.0}
    assert coin_refusal("b", **attack) is None
    assert "attack offset" in coin_refusal("b", **{**attack, "raw_temperature": None})


def test_the_driver_refuses_coins_on_a_side_that_cannot_play_them(tmp_path):
    from tools.run_elo_batch import main
    argv = ["x", "--label-a", "A", "--spec-a", "random", "--label-b", "B", "--spec-b", "dummy",
            "--outdir", str(tmp_path / "games"), "--games", "2", "--device", "cpu", "--jobs", "1",
            "--time-budget-min", "0", "--min-free-mb", "0", "--raw-temperature-a", "0"]
    with pytest.raises(SystemExit):
        main(argv + ["--coin-band-b", "0.35"])
    with pytest.raises(SystemExit):
        main(argv + ["--attack-coin-offset", "1"])


def test_the_driver_hands_the_coins_to_each_game():
    from argparse import Namespace
    from tools.eval_procedure import coin_argv
    args = Namespace(attack_coin_offset=1.0, coin_band_a=0.35, coin_p_a=0.25, coin_band_b=0.0, coin_p_b=0.25)
    assert coin_argv(args) == ["--attack-coin-offset", "1.0", "--coin-band-a", "0.35", "--coin-p-a", "0.25"]
    assert coin_argv(Namespace(attack_coin_offset=0.0, coin_band_a=0.0, coin_p_a=0.25,
                               coin_band_b=0.0, coin_p_b=0.25)) == []
    assert coin_argv(Namespace(attack_coin_offset=0.0, coin_band_a=0.0, coin_p_a=0.25, coin_band_b=0.0,
                               coin_p_b=0.25, recruit_rule="/r/rule.json")) == ["--recruit-rule", "/r/rule.json"]
