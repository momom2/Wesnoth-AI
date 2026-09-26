"""Scenario starts plus midgame states from dummy-played games, some
with legal attacks: positions that exercise every branch of the
legal-action enumeration."""
import random


def _has_attack(gs):
    """Any own unit adjacent to a visible enemy it may still attack."""
    from tools.abilities import hex_neighbors
    side = gs.global_info.current_side
    enemies = {(u.position.x, u.position.y) for u in gs.map.units if u.side != side}
    for u in gs.map.units:
        if u.side == side and not u.has_attacked:
            if any(n in enemies for n in hex_neighbors(u.position.x, u.position.y)):
                return True
    return False


def _states(n_attack=4, n_plain=6):
    """Scenario starts (recruit branch) plus dummy-played midgame
    states, some of which have legal attacks."""
    import copy
    from tools.elo_ladder import _ScriptedAdapter
    from tools.eval_players import _PolicyPair, _play_one_eval_game
    from wesnoth_ai.rules.scenario_pool import build_scenario_gamestate, random_setup
    from tools.wesnoth_sim import WesnothSim
    from wesnoth_ai.dummy_policy import DummyPolicy
    starts = [build_scenario_gamestate(random_setup(random.Random(s))) for s in (1, 2)]
    attack, plain = [], []

    class _Rec:
        def __init__(self, inner):
            self._inner = inner
            self._n = 0

        def select_action(self, gs, **kw):
            self._n += 1
            if gs.global_info.turn_number >= 3 and self._n % 4 == 0:
                if _has_attack(gs) and len(attack) < n_attack:
                    attack.append(copy.deepcopy(gs))
                elif len(plain) < n_plain:
                    plain.append(copy.deepcopy(gs))
            return self._inner.select_action(gs, **kw)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    g = 0
    while (len(attack) < n_attack or len(plain) < n_plain) and g < 30:
        setup = random_setup(random.Random(100 + g))
        g += 1
        sim = WesnothSim(build_scenario_gamestate(setup), scenario_id=setup.scenario_id,
                         max_turns=25)
        _play_one_eval_game(
            sim,
            _PolicyPair(policy=_Rec(_ScriptedAdapter(DummyPolicy())), label="a", side=1),
            _PolicyPair(policy=_Rec(_ScriptedAdapter(DummyPolicy())), label="b", side=2),
            game_label=f"enum{g}")
    assert len(attack) >= 1, "no attack-bearing state harvested"
    return starts + plain + attack
