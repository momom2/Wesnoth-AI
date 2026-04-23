#!/usr/bin/env python3
"""Integration test: simulate the full Python-Wesnoth turn loop.

No live Wesnoth here — we feed JSON payloads shaped like what the Lua
state_collector emits, drive the full Python pipeline (parse → select
action → write action.lua), and verify each step.
"""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).parent))

from state_converter import StateConverter
from classes import Position


def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def _mk_unit(**overrides):
    """Helper: unit dict with sensible defaults matching state_collector output."""
    base = {
        "id": "u",
        "name": "Dwarvish Fighter",
        "side": 1,
        "is_leader": False,
        "x": 1, "y": 1,
        "max_hp": 40, "current_hp": 40,
        "max_moves": 5, "current_moves": 5,
        "max_exp": 32, "current_exp": 0,
        "cost": 14,
        "alignment": "neutral",
        "levelup_names": [],
        "has_attacked": False,
        "attacks": [],
        "resistances": [0, 0, 0, 0, 0, 0],
        "defenses": {},       # empty → code fills with 100
        "movement_costs": {}, # empty → code fills with 99
        "abilities": [],
        "traits": [],
        "statuses": [],
    }
    base.update(overrides)
    return base


def _mk_side(**overrides):
    base = {
        "gold": 100,
        "village_gold": 2,
        "village_support": 1,
        "base_income": 0,
        "recruits": [],
        "num_villages": 0,
    }
    base.update(overrides)
    return base


def test_full_pipeline():
    """Parse a state, pick an action, write it to a Lua file."""
    print_header("Integration Test: Full Communication Pipeline")

    payload = json.dumps({
        "game_id": "integration_test",
        "current_side": 1,
        "turn_number": 3,
        "time_of_day": "afternoon",
        "game_over": False,
        "map": {
            "width": 44,
            "height": 40,
            "hexes": [
                {"x": 20, "y": 20, "full_code": "Kh", "terrain_types": ["Kh"], "modifiers": ["keep"]},
                {"x": 21, "y": 20, "full_code": "Ch", "terrain_types": ["Ch"], "modifiers": ["castle"]},
                {"x": 25, "y": 20, "full_code": "Gg", "terrain_types": ["Gg"], "modifiers": []},
            ],
            "units": [
                _mk_unit(id="side1_leader", name="Dwarvish Fighter", side=1, x=20, y=20,
                         is_leader=True, current_hp=35, current_exp=10),
                _mk_unit(id="side1_grunt", name="Dwarvish Guardsman", side=1, x=21, y=20,
                         max_hp=30, current_hp=30, max_moves=4, current_moves=4,
                         max_exp=38, current_exp=5, cost=17),
                _mk_unit(id="side2_leader", name="Drake Clasher", side=2, x=25, y=20,
                         is_leader=True, max_hp=43, current_hp=43, current_moves=0,
                         max_exp=34, cost=20, alignment="lawful", has_attacked=True),
            ],
            "fog": [],
            "mask": [],
        },
        "sides": [
            _mk_side(gold=75, num_villages=2),
            _mk_side(gold=80, num_villages=1),
        ],
    })

    print("Step 1: Parsing JSON state...")
    converter = StateConverter()
    try:
        game_state = converter.convert_payload_to_game_state(payload)
        print(f"  ✓ Parsed. Turn={game_state.global_info.turn_number} "
              f"side={game_state.global_info.current_side} "
              f"map={game_state.map.size_x}x{game_state.map.size_y} "
              f"units={len(game_state.map.units)} hexes={len(game_state.map.hexes)} "
              f"sides={len(game_state.sides)}")
    except Exception as e:
        print(f"  ✗ Failed to parse: {e}")
        import traceback; traceback.print_exc()
        return False

    print("\nStep 2: Selecting an action...")
    my_units = [u for u in game_state.map.units
                if u.side == game_state.global_info.current_side]
    try:
        leader = next(u for u in my_units if u.is_leader)
        action = {
            'type': 'move',
            'start_hex': leader.position,
            'target_hex': Position(x=leader.position.x + 1, y=leader.position.y),
        }
        print(f"  ✓ Move {leader.name} "
              f"from ({leader.position.x},{leader.position.y}) "
              f"to ({action['target_hex'].x},{action['target_hex'].y})")
    except Exception as e:
        print(f"  ✗ {e}")
        return False

    print("\nStep 3: Converting action for Wesnoth (0-idx → 1-idx)...")
    try:
        wesnoth_action = converter.convert_action_to_json(action)
        assert wesnoth_action['start_x'] == leader.position.x + 1
        assert wesnoth_action['start_y'] == leader.position.y + 1
        print(f"  ✓ Wesnoth action: {wesnoth_action}")
    except Exception as e:
        print(f"  ✗ {e}")
        import traceback; traceback.print_exc()
        return False

    print("\nStep 4: Writing action.lua...")
    try:
        from wesnoth_interface import WesnothGame
        from constants import ACTION_FILE_NAME
        with TemporaryDirectory() as tmpdir:
            game = WesnothGame("integration_test", Path("dummy.cfg"))
            game.game_dir = Path(tmpdir)
            game.action_path = game.game_dir / ACTION_FILE_NAME

            assert game.send_action(wesnoth_action), "send_action returned False"
            lua_content = game.action_path.read_text()
            assert lua_content.startswith("return "), "bad Lua chunk"
            assert wesnoth_action['type'] in lua_content
            assert 'seq =' in lua_content, "action.lua missing seq field"
            print(f"  ✓ {lua_content.strip()}")
    except Exception as e:
        print(f"  ✗ {e}")
        import traceback; traceback.print_exc()
        return False

    print("\n" + "=" * 70)
    print("  ✅ INTEGRATION TEST PASSED")
    print("=" * 70)
    return True


def test_multiple_turns():
    """Unit-type IDs and gold/turn transitions behave consistently."""
    print_header("Multi-Turn Simulation")

    converter = StateConverter()

    turn1 = json.dumps({
        "game_id": "multi_turn",
        "current_side": 1,
        "turn_number": 1,
        "time_of_day": "morning",
        "game_over": False,
        "map": {
            "width": 10, "height": 10,
            "hexes": [], "units": [
                _mk_unit(name="Dwarvish Fighter", side=1, x=5, y=5, is_leader=True),
            ],
            "fog": [], "mask": [],
        },
        "sides": [_mk_side(gold=100)],
    })

    turn2 = json.dumps({
        "game_id": "multi_turn",
        "current_side": 1,
        "turn_number": 2,
        "time_of_day": "morning",
        "game_over": False,
        "map": {
            "width": 10, "height": 10,
            "hexes": [], "units": [
                _mk_unit(name="Dwarvish Fighter", side=1, x=6, y=5, is_leader=True,
                         current_hp=38, current_moves=4, current_exp=5),
                _mk_unit(name="Dwarvish Guardsman", side=1, x=5, y=5,
                         max_hp=30, current_hp=30, max_moves=4, current_moves=4,
                         max_exp=38),
            ],
            "fog": [], "mask": [],
        },
        "sides": [_mk_side(gold=83)],
    })

    try:
        state1 = converter.convert_payload_to_game_state(turn1)
        fighter_id_1 = next(iter(state1.map.units)).name_id
        print(f"  Turn 1: {len(state1.map.units)} units, fighter ID = {fighter_id_1}")

        state2 = converter.convert_payload_to_game_state(turn2)
        fighter = next(u for u in state2.map.units if u.name == "Dwarvish Fighter")
        print(f"  Turn 2: {len(state2.map.units)} units, fighter ID = {fighter.name_id}")

        assert fighter_id_1 == fighter.name_id, "unit-type ID changed across turns"
        assert state2.global_info.turn_number == 2
        assert state2.sides[0].current_gold < state1.sides[0].current_gold
        print("  ✓ IDs stable, state advances correctly")
        return True
    except Exception as e:
        print(f"  ✗ {e}")
        import traceback; traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 70)
    print("  Integration Test Suite")
    print("=" * 70)

    tests = [test_full_pipeline, test_multiple_turns]
    results = []
    for t in tests:
        try:
            results.append(t())
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            import traceback; traceback.print_exc()
            results.append(False)

    passed, total = sum(results), len(results)
    print("\n" + "=" * 70)
    print(f"  {passed}/{total} passed")
    print("=" * 70)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
