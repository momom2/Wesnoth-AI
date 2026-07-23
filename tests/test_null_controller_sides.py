#!/usr/bin/env python3
"""controller=null sides never act (2026-07-19 Silverhead desync).

The engine's turn loop skips empty teams entirely
(playsingle_controller.cpp:198-210 skip_empty_sides): a
controller=null side gets no [init_side] and no actions, ever.
Our armed-neutral machinery gave Silverhead Crossing's null side 3
("Nani the Shapeshifter", an armed Tentacle of the Deep) a full RCA
turn each round; the exported [init_side] side_number=3 commands
drifted playback's turn counter, and with it the ToD phase --
surfacing as damage-verification overrides (19-vs-12 on a Wose
strike = afternoon-vs-night lawful bonus) and cascading [choose]
errors. Minis' tentacle sides declare controller=ai and keep their
turns.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


def _silverhead_sim(max_turns=6):
    from sim_test_helpers import fresh_scenario_sim
    return fresh_scenario_sim(
        seed=0, max_turns=max_turns,
        scenario_id="multiplayer_Silverhead_Crossing")


def test_null_side_recorded_on_gamestate():
    sim = _silverhead_sim()
    nulls = getattr(sim.gs.global_info, "_null_controller_sides",
                    frozenset())
    assert 3 in nulls
    # The serpent itself still exists on the map (attackable,
    # blocks its hex) -- only its TURN is gone.
    assert any(u.side == 3 for u in sim.gs.map.units)


def test_null_side_gets_no_turn_and_no_init_side():
    sim = _silverhead_sim()
    while not sim.done:
        sim.step({"type": "end_turn"})
    sides_with_commands = {rc.side for rc in sim.command_history}
    init_sides = [rc for rc in sim.command_history
                  if rc.kind == "init_side"]
    assert 3 not in sides_with_commands
    # Strict 1-2 alternation: playback's turn counter stays aligned.
    seq = [rc.cmd[1] for rc in init_sides]
    assert seq == [1 if i % 2 == 0 else 2 for i in range(len(seq))], seq


def test_ai_controlled_neutral_still_acts_on_minis():
    """Regression guard: the gate must key on CONTROLLER, not on
    'side 3 exists' -- minis' tentacle sides (controller=ai) keep
    their neutral turn."""
    from sim_test_helpers import fresh_scenario_sim
    sim = fresh_scenario_sim(seed=1, max_turns=4, mini=True,
                             scenario_id="enclave_micro_isar")
    nulls = getattr(sim.gs.global_info, "_null_controller_sides",
                    frozenset())
    assert 3 not in nulls
    has_armed_neutral = any(u.side == 3 for u in sim.gs.map.units)
    while not sim.done:
        sim.step({"type": "end_turn"})
    if has_armed_neutral:
        assert any(rc.side == 3 for rc in sim.command_history), (
            "ai-controlled neutral side lost its turn")
