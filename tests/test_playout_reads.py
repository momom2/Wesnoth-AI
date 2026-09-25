"""Side readings of a playout (tools/playout_reads.py): the luck of an
attack has mean zero given the position before it, whatever the side,
the kill odds or the advancement rule; the horizon reads the positions
the simulator was in at each player side's turn start; and the replay
the readings walk reaches the positions the simulator played."""
from __future__ import annotations

import copy
import hashlib
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

from tools import playout_reads as pr  # noqa: E402
from tools.replay_dataset import _apply_command, _build_initial_gamestate  # noqa: E402
from wesnoth_ai.classes import Position, state_digest  # noqa: E402

# A one-row corridor of three grass hexes inside an impassable border.
CORRIDOR = "Xv, Xv, Xv, Xv, Xv\nXv, Gg, Gg, Gg, Xv\nXv, Xv, Xv, Xv, Xv"
SPEAR = {"type": "attack", "start_hex": Position(0, 0), "target_hex": Position(1, 0),
         "attack_index": 0}
TRIALS = 800


# ---------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------
def _duel_sim(spearman=None, fighter=None):
    """A Spearman (side 1, to move) beside a Dwarvish Fighter (side 2),
    which answers the spear with its axe or its hammer; `spearman` and
    `fighter` override unit fields (hp, max_exp)."""
    from tools.wesnoth_sim import WesnothSim
    units = [dict({"uid": 1, "type": "Spearman", "side": 1, "x": 0, "y": 0, "is_leader": True},
                  **(spearman or {})),
             dict({"uid": 2, "type": "Dwarvish Fighter", "side": 2, "x": 1, "y": 0,
                   "is_leader": True}, **(fighter or {}))]
    data = {"game_id": "reads", "map_data": CORRIDOR, "starting_units": units,
            "starting_sides": [{"side": k, "gold": 0, "recruit": []} for k in (1, 2)]}
    return WesnothSim(_build_initial_gamestate(data), scenario_id="reads", max_turns=4,
                      use_core=False)


def _duel_fight(uniform=False, **units):
    """(position before the spear attack, the attack command the
    simulator records for it)."""
    sim = _duel_sim(**units)
    if uniform:
        sim._seed_salt = "reads-duel"
        sim.enable_uniform_advancement()
    before = copy.deepcopy(sim.gs)
    sim.step(SPEAR)
    [rc] = sim.command_history[-1:]
    assert rc.kind == "attack"
    return before, list(rc.cmd)


@pytest.fixture(scope="module")
def played():
    """A Brawler game on a mini map whose tentacles (side 3) take their
    turns, from the position after side 1's first end_turn (the mover's
    post-turn position, as a playout starts from) to its end: the start
    position, the commands and rejections from there, the simulator's
    digest after each step and after a rejection (keyed by moment and
    command index), its player turn starts, and the positions before
    the attacks it played."""
    import wesnoth_ai.dummy_policy as dummy_policy
    from sim_test_helpers import Brawler, require_scenario_data
    from tools.scenario_pool import ScenarioSetup, build_scenario_gamestate
    from tools.wesnoth_sim import WesnothSim
    require_scenario_data()
    setup = ScenarioSetup(scenario_id="2p_mini_edited", faction1="Knalgan Alliance",
                          leader1="Dwarvish Steelclad", faction2="Rebels",
                          leader2="Elvish Captain")
    sim = WesnothSim(build_scenario_gamestate(setup), scenario_id=setup.scenario_id,
                     max_turns=8, use_core=False)
    sim._seed_salt = "reads-test"
    sim.enable_uniform_advancement()
    mover = 1
    cap = dummy_policy._BOOTSTRAP_UNITS
    dummy_policy._BOOTSTRAP_UNITS = 8
    out = {"mover": mover, "digests": {}, "turn_starts": [], "attacks": []}
    try:
        policy = Brawler()
        while not sim.done:
            n = len(sim.command_history)
            if "start" in out and "rejected_at" not in out and n >= out["start"] + 12:
                sim.reject_recruit_hex(0, 0)
                out["rejected_at"] = n - out["start"]
                out["digests"][("before", n - out["start"])] = state_digest(sim.gs)
            before = copy.deepcopy(sim.gs)
            sim.step(policy.select_action(sim.gs, game_label="reads"))
            added = sim.command_history[n:]
            if not added:                                # a refused step: nothing applied
                continue
            if "start" not in out:
                if added[-1].cmd == ["init_side", 2]:
                    out["start"] = len(sim.command_history)
                    out["post_gs"] = copy.deepcopy(sim.gs)
                continue
            last = len(sim.command_history) - 1 - out["start"]
            out["digests"][("after", last)] = state_digest(sim.gs)
            if added[-1].kind == "init_side" and added[-1].cmd[1] in (1, 2):
                gi = sim.gs.global_info
                out["turn_starts"].append((gi.turn_number, gi.current_side,
                                           _hp_by_side(sim.gs)))
            if added[0].kind == "attack":
                out["attacks"].append((before, list(added[0].cmd)))
    finally:
        dummy_policy._BOOTSTRAP_UNITS = cap
    start = out["start"]
    out["commands"] = [list(rc.cmd) for rc in sim.command_history[start:]]
    out["rejections"] = [(k - start, x, y) for k, x, y in sim.recruit_rejections if k >= start]
    out["salt"] = sim._seed_salt
    out["final_digest"] = state_digest(sim.gs)
    return out


# ---------------------------------------------------------------------
# Luck: mean zero given the position before the attack
# ---------------------------------------------------------------------
def _hp_by_side(gs):
    totals = {}
    for u in gs.map.units:
        totals[u.side] = totals.get(u.side, 0) + u.current_hp
    return totals


def _margin(hp_by_side, mover):
    """The mover's hit points minus everyone else's."""
    return sum(hp if side == mover else -hp for side, hp in hp_by_side.items())


def _board_margin(gs, mover):
    return _margin(_hp_by_side(gs), mover)


def _board_kills(before, after, mover):
    """Opponent units lost minus mover units lost, over the board."""
    def lost(side_is_mover):
        count = [sum(1 for u in gs.map.units if (u.side == mover) == side_is_mover)
                 for gs in (before, after)]
        return count[0] - count[1]
    return lost(False) - lost(True)


def _seeded(cmd, i):
    seeded = list(cmd)
    seeded[7] = hashlib.sha256(f"reads:{cmd[:7]}:{i}".encode()).hexdigest()[:8]
    return seeded


def _mean_and_error(xs):
    mean = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))
    return mean, sd / math.sqrt(len(xs))


def _check_zero_mean(before, cmd, mover, *, kill_possible):
    """Applies the attack under TRIALS seeds (and advancement salts)
    and checks, for the HP margin and the kill margin over the board,
    that realized minus the forecast has mean zero within 3 standard
    errors, and that the module reads each trial's realized change as
    the board does. Returns the forecast and the attacker's type after
    each trial (None when it died)."""
    forecast = pr.forecast_attack(before, cmd, mover)
    assert forecast is not None
    luck_hp, luck_kills, attacker_types = [], [], []
    for i in range(TRIALS):
        after = copy.deepcopy(before)
        after.global_info._advance_salt = f"reads-advance:{i}"
        _apply_command(after, _seeded(cmd, i))
        board = (_board_margin(after, mover) - _board_margin(before, mover),
                 _board_kills(before, after, mover))
        assert pr.realized_change(after, forecast) == board, i
        luck_hp.append(board[0] - forecast.hp_change)
        luck_kills.append(board[1] - forecast.kill_change)
        attacker_types.append(next((u.name for u in after.map.units
                                    if (u.id, u.side) == forecast.attacker), None))
    mean, error = _mean_and_error(luck_hp)
    assert error > 0 and abs(mean) <= 3 * error, (forecast, mean, error)
    mean, error = _mean_and_error(luck_kills)
    if kill_possible:
        assert error > 0 and abs(mean) <= 3 * error, (forecast, mean, error)
    else:
        assert forecast.kill_change == 0 and max(map(abs, luck_kills)) == 0
    return forecast, attacker_types


def test_luck_has_mean_zero_whatever_the_side_the_kill_odds_or_the_advancement():
    """Four fights, each from one position under many seeds: full hit
    points (no kill possible; the Fighter picks between two weapons),
    low hit points on both sides read from the defender's side, and a
    Spearman one fight from advancing: to its first option with the
    uniform advancement channel off, uniformly among three options of
    different hit points with it on."""
    before, cmd = _duel_fight()
    _check_zero_mean(before, cmd, mover=1, kill_possible=False)
    before, cmd = _duel_fight(spearman={"hp": 10}, fighter={"hp": 10})
    forecast, types = _check_zero_mean(before, cmd, mover=2, kill_possible=True)
    assert forecast.signs == (-1, 1) and None in types
    before, cmd = _duel_fight(spearman={"max_exp": 1})
    _forecast, types = _check_zero_mean(before, cmd, mover=1, kill_possible=False)
    assert set(types) == {"Swordsman"}
    before, cmd = _duel_fight(uniform=True, spearman={"max_exp": 1})
    _forecast, types = _check_zero_mean(before, cmd, mover=1, kill_possible=False)
    assert set(types) == {"Swordsman", "Pikeman", "Javelineer"}


def test_luck_has_mean_zero_on_an_attack_of_a_real_game(played):
    """An attack of the played game where a kill is possible, read from
    the attacker's side."""
    forecasts = [(before, cmd, pr.forecast_attack(before, cmd, 1))
                 for before, cmd in played["attacks"]]
    real = [(before, cmd) for before, cmd, forecast in forecasts
            if forecast is not None and 0.05 < abs(forecast.kill_change) < 0.95]
    assert real, "the game played no attack where a kill was possible"
    before, cmd = real[0]
    _check_zero_mean(before, cmd, mover=1, kill_possible=True)


def test_the_luck_of_a_playout_is_realized_minus_expected_and_skips_what_it_cannot_read():
    before, cmd = _duel_fight(spearman={"hp": 10}, fighter={"hp": 10})
    forecast = pr.forecast_attack(before, cmd, 1)
    for i in range(4):
        seeded = _seeded(cmd, i)
        after = copy.deepcopy(before)
        _apply_command(after, seeded)
        hp_change, kill_change = pr.realized_change(after, forecast)
        reads = pr.playout_reads(before, [seeded], [], mover=1, value_of=None,
                                 horizon_reads=0, luck=True)
        assert reads == {"horizon": [], "luck": {
            "hp": hp_change - forecast.hp_change, "kills": kill_change - forecast.kill_change,
            "attacks": 1, "skipped": 0}}
    no_seed = cmd[:7] + [""]                             # the applier skips it
    other_weapon = cmd[:6] + [1 - cmd[6]] + cmd[7:]
    reads = pr.playout_reads(before, [no_seed, other_weapon], [], mover=1, value_of=None,
                             horizon_reads=0, luck=True)
    assert reads["luck"] == {"hp": 0.0, "kills": 0.0, "attacks": 2, "skipped": 2}


def test_a_plague_corpse_that_takes_the_dead_units_id_is_not_read_as_its_survival():
    """The applier numbers a plague corpse after the highest id left on
    the board, which is the dead unit's own when it had the highest:
    the corpse is found by id and side, never taken for the victim."""
    from tools.wesnoth_sim import WesnothSim
    units = [{"uid": 1, "type": "Walking Corpse", "side": 1, "x": 0, "y": 0, "is_leader": True},
             {"uid": 2, "type": "Spearman", "side": 2, "x": 1, "y": 0, "is_leader": True,
              "hp": 1}]
    data = {"game_id": "reads", "map_data": CORRIDOR, "starting_units": units,
            "starting_sides": [{"side": k, "gold": 0, "recruit": []} for k in (1, 2)]}
    sim = WesnothSim(_build_initial_gamestate(data), scenario_id="reads", max_turns=4,
                     use_core=False)
    before = copy.deepcopy(sim.gs)
    sim.step(SPEAR)
    cmd = list(sim.command_history[-1].cmd)
    forecast = pr.forecast_attack(before, cmd, 1)
    for i in range(40):
        after = copy.deepcopy(before)
        _apply_command(after, _seeded(cmd, i))
        if {(u.id, u.side) for u in after.map.units} == {("u1", 1), ("u2", 1)}:
            break
    else:
        pytest.fail("no seed of 40 killed the Spearman with the plague")
    hp = [{u.id: u.current_hp for u in gs.map.units if u.side == 1}["u1"] for gs in (before, after)]
    assert pr.realized_change(after, forecast) == (hp[1] - hp[0] + 1, 1)


# ---------------------------------------------------------------------
# Horizon reads
# ---------------------------------------------------------------------
def _turn_and_side(gs):
    """A stub value that names the position it was read on."""
    return gs.global_info.turn_number * 10 + gs.global_info.current_side


def test_horizon_reads_the_start_and_each_player_turn_start_after_it(played):
    mover = played["mover"]
    commands = played["commands"]
    assert ["init_side", 3] in commands                  # the neutral side took turns
    post = played["post_gs"].global_info
    starts = ([(post.turn_number, post.current_side, _hp_by_side(played["post_gs"]))]
              + played["turn_starts"])
    expected = [[float(turn * 10 + side), _margin(hp, mover)] for turn, side, hp in starts]
    assert len(expected) >= 5
    args = (played["post_gs"], commands, played["rejections"])
    full = pr.playout_reads(*args, mover=mover, value_of=_turn_and_side, horizon_reads=100,
                            luck=True, advance_salt=played["salt"])
    assert full["horizon"] == expected                   # shorter than asked: the game ended
    assert full["luck"]["attacks"] == sum(c[0] == "attack" for c in commands) >= 5
    assert full["luck"]["skipped"] == 0
    first = pr.playout_reads(*args, mover=mover, value_of=_turn_and_side, horizon_reads=3,
                             luck=False, advance_salt=played["salt"])
    assert first == {"horizon": expected[:3]}
    other_side = pr.playout_reads(*args, mover=2, value_of=None, horizon_reads=2, luck=False,
                                  advance_salt=played["salt"])
    assert other_side == {"horizon": [[None, _margin(hp, 2)] for _, _, hp in starts[:2]]}


# ---------------------------------------------------------------------
# The replay
# ---------------------------------------------------------------------
def test_the_replay_passes_through_every_position_the_simulator_played(played):
    """Forecasting each attack on the way, as the luck read does, leaves
    the replay on the simulator's positions."""
    gs = pr.start_state(played["post_gs"], played["salt"])
    compared = 0
    for moment, k, state, cmd in pr.replay(gs, played["commands"], played["rejections"]):
        if moment == "before" and cmd[0] == "attack":
            pr.forecast_attack(state, cmd, played["mover"])
        want = played["digests"].get((moment, k))
        if want is not None:
            assert state_digest(state) == want, (moment, k)
            compared += 1
    assert compared == len(played["digests"]) >= 20
    assert ("before", played["rejected_at"]) in played["digests"]    # a rejection replayed
    assert state_digest(gs) == played["final_digest"]


def test_an_advancement_replays_only_with_the_playouts_salt():
    """On the uniform channel the playout draws its advancements from
    its own salt, not from the one its start position carries."""
    from tools.wesnoth_sim import WesnothSim
    duel = _duel_sim(spearman={"max_exp": 1})
    duel._seed_salt = "turn"
    duel.enable_uniform_advancement()
    post_gs = copy.deepcopy(duel.gs)
    playout = WesnothSim(copy.deepcopy(post_gs), "reads", max_turns=4,
                         apply_scenario_events=False, begin_turn=False, use_core=False)
    playout._seed_salt = "record-test"                   # draws Javelineer, "turn" Pikeman
    playout.step(SPEAR)
    playout.step({"type": "end_turn"})
    commands = [rc.cmd for rc in playout.command_history]

    def replayed(salt):
        gs = pr.start_state(post_gs, salt)
        for _ in pr.replay(gs, commands, []):
            pass
        return gs

    assert [u.name for u in playout.gs.map.units if u.side == 1] == ["Javelineer"]
    assert state_digest(replayed("record-test")) == state_digest(playout.gs)
    assert [u.name for u in replayed("turn").map.units if u.side == 1] == ["Pikeman"]
    with pytest.raises(ValueError, match="advance_salt"):
        pr.playout_reads(post_gs, commands, [], mover=1, value_of=None, horizon_reads=1,
                         luck=False)
