#!/usr/bin/env python3
"""
test_integration.py
Integration test simulating the full Python-Wesnoth communication loop

This simulates what happens during actual gameplay:
1. Wesnoth outputs WML state
2. Python parses it to GameState
3. Python generates an action
4. Action is written to Lua file
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from state_converter import StateConverter
from classes import Position

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def test_full_pipeline():
    """Simulate a complete game turn."""
    print_header("Integration Test: Full Communication Pipeline")
    
    # Simulate WML output from Wesnoth
    wml_from_wesnoth = """
game_id="integration_test"
current_side=1
turn_number=3
time_of_day="afternoon"
game_over=false
[map]
    width=44
    height=40
    [hex]
        x=20
        y=20
        full_code="Kh"
    [/hex]
    [hex]
        x=21
        y=20
        full_code="Ch"
    [/hex]
    [hex]
        x=25
        y=20
        full_code="Gg"
    [/hex]
    [unit]
        id="side1_leader"
        name="Dwarvish Fighter"
        side=1
        x=20
        y=20
        max_hp=40
        current_hp=35
        max_moves=5
        current_moves=5
        max_exp=32
        current_exp=10
        cost=14
        alignment="neutral"
        is_leader=true
        has_attacked=false
    [/unit]
    [unit]
        id="side1_grunt"
        name="Dwarvish Guardsman"
        side=1
        x=21
        y=20
        max_hp=30
        current_hp=30
        max_moves=4
        current_moves=4
        max_exp=38
        current_exp=5
        cost=17
        alignment="neutral"
        is_leader=false
        has_attacked=false
    [/unit]
    [unit]
        id="side2_leader"
        name="Drake Clasher"
        side=2
        x=25
        y=20
        max_hp=43
        current_hp=43
        max_moves=5
        current_moves=0
        max_exp=34
        current_exp=0
        cost=20
        alignment="lawful"
        is_leader=true
        has_attacked=true
    [/unit]
[/map]
[side]
    gold=75
    village_gold=2
    base_income=0
    num_villages=2
[/side]
[side]
    gold=80
    village_gold=2
    base_income=0
    num_villages=1
[/side]
"""
    
    print("Step 1: Parsing WML state from Wesnoth...")
    converter = StateConverter()
    
    try:
        game_state = converter.convert_wml_to_game_state(wml_from_wesnoth)
        print(f"  ✓ Parsed successfully")
        print(f"    - Game ID: {game_state.game_id}")
        print(f"    - Turn: {game_state.global_info.turn_number}")
        print(f"    - Current side: {game_state.global_info.current_side}")
        print(f"    - Map size: {game_state.map.size_x}x{game_state.map.size_y}")
        print(f"    - Units: {len(game_state.map.units)}")
        print(f"    - Hexes: {len(game_state.map.hexes)}")
        print(f"    - Sides: {len(game_state.sides)}")
        
    except Exception as e:
        print(f"  ✗ Failed to parse: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nStep 2: Extracting game information...")
    try:
        # Get units for current side
        current_side = game_state.global_info.current_side
        my_units = [u for u in game_state.map.units if u.side == current_side]
        enemy_units = [u for u in game_state.map.units if u.side != current_side]
        
        print(f"  ✓ Current side: {current_side}")
        print(f"    - My units: {len(my_units)}")
        for unit in my_units:
            print(f"      • {unit.name} at ({unit.position.x}, {unit.position.y}) - {unit.current_hp}/{unit.max_hp} HP")
        print(f"    - Enemy units: {len(enemy_units)}")
        for unit in enemy_units:
            print(f"      • {unit.name} at ({unit.position.x}, {unit.position.y}) - {unit.current_hp}/{unit.max_hp} HP")
        print(f"    - Gold: {game_state.sides[current_side-1].current_gold}")
        
    except Exception as e:
        print(f"  ✗ Failed to extract info: {e}")
        return False
    
    print("\nStep 3: Generating AI action...")
    try:
        # Simple AI logic: Move leader towards enemy
        leader = next(u for u in my_units if u.is_leader)
        enemy_leader = next(u for u in enemy_units if u.is_leader)
        
        # Create a move action (in Python 0-indexed coords)
        action = {
            'type': 'move',
            'start_hex': leader.position,
            'target_hex': Position(x=leader.position.x + 1, y=leader.position.y)
        }
        
        print(f"  ✓ Generated action: {action['type']}")
        print(f"    - Move {leader.name}")
        print(f"    - From: ({leader.position.x}, {leader.position.y})")
        print(f"    - To: ({action['target_hex'].x}, {action['target_hex'].y})")
        
    except Exception as e:
        print(f"  ✗ Failed to generate action: {e}")
        return False
    
    print("\nStep 4: Converting action to Wesnoth format...")
    try:
        # Convert to Wesnoth format (1-indexed coords)
        action_for_wesnoth = converter.convert_action_to_json(action)
        
        print(f"  ✓ Converted action:")
        print(f"    - Type: {action_for_wesnoth['type']}")
        print(f"    - Start: ({action_for_wesnoth['start_x']}, {action_for_wesnoth['start_y']}) [1-indexed]")
        print(f"    - Target: ({action_for_wesnoth['target_x']}, {action_for_wesnoth['target_y']}) [1-indexed]")
        
        # Verify coordinate conversion
        assert action_for_wesnoth['start_x'] == leader.position.x + 1, "Start X should be 1-indexed"
        assert action_for_wesnoth['start_y'] == leader.position.y + 1, "Start Y should be 1-indexed"
        print(f"  ✓ Coordinate conversion correct (0-indexed → 1-indexed)")
        
    except Exception as e:
        print(f"  ✗ Failed to convert action: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nStep 5: Simulating Lua file generation...")
    try:
        from wesnoth_interface import WesnothGame
        from tempfile import TemporaryDirectory
        
        with TemporaryDirectory() as tmpdir:
            game = WesnothGame("integration_test", Path("dummy.cfg"))
            game.game_dir = Path(tmpdir)
            game.action_file = game.game_dir / "action_input.lua"
            
            # Write action
            success = game.send_action(action_for_wesnoth)
            assert success, "Failed to write action file"
            
            # Read back
            lua_content = game.action_file.read_text()
            
            print(f"  ✓ Generated Lua file:")
            print(f"    {lua_content[:150]}...")
            
            # Verify content
            assert 'return' in lua_content, "Should start with 'return'"
            assert action_for_wesnoth['type'] in lua_content, "Should contain action type"
            print(f"  ✓ Lua file format valid")
        
    except Exception as e:
        print(f"  ✗ Failed to generate Lua file: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("  ✅ INTEGRATION TEST PASSED")
    print("  All pipeline steps completed successfully!")
    print("=" * 70)
    return True

def test_multiple_turns():
    """Simulate multiple turns to test state continuity."""
    print_header("Multi-Turn Simulation")
    
    converter = StateConverter()
    
    # Track unit type ID consistency across turns
    turn1_wml = """
game_id="multi_turn"
current_side=1
turn_number=1
[map]
    width=10
    height=10
    [unit]
        name="Dwarvish Fighter"
        side=1
        x=5
        y=5
        max_hp=40
        current_hp=40
        max_moves=5
        current_moves=5
        max_exp=32
        current_exp=0
        alignment="neutral"
        is_leader=true
        has_attacked=false
    [/unit]
[/map]
[side]
    gold=100
[/side]
"""
    
    turn2_wml = """
game_id="multi_turn"
current_side=1
turn_number=2
[map]
    width=10
    height=10
    [unit]
        name="Dwarvish Fighter"
        side=1
        x=6
        y=5
        max_hp=40
        current_hp=38
        max_moves=5
        current_moves=4
        max_exp=32
        current_exp=5
        alignment="neutral"
        is_leader=true
        has_attacked=false
    [/unit]
    [unit]
        name="Dwarvish Guardsman"
        side=1
        x=5
        y=5
        max_hp=30
        current_hp=30
        max_moves=4
        current_moves=4
        max_exp=38
        current_exp=0
        alignment="neutral"
        is_leader=false
        has_attacked=false
    [/unit]
[/map]
[side]
    gold=83
[/side]
"""
    
    try:
        print("Turn 1:")
        state1 = converter.convert_wml_to_game_state(turn1_wml)
        units1 = list(state1.map.units)
        print(f"  Units: {len(units1)}")
        print(f"  Unit types: {[u.name for u in units1]}")
        fighter_id_1 = units1[0].name_id
        print(f"  'Dwarvish Fighter' ID: {fighter_id_1}")
        
        print("\nTurn 2:")
        state2 = converter.convert_wml_to_game_state(turn2_wml)
        units2 = list(state2.map.units)
        print(f"  Units: {len(units2)}")
        print(f"  Unit types: {[u.name for u in units2]}")
        
        # Find the fighter again
        fighter = next(u for u in units2 if u.name == "Dwarvish Fighter")
        fighter_id_2 = fighter.name_id
        print(f"  'Dwarvish Fighter' ID: {fighter_id_2}")
        
        # Verify consistency
        assert fighter_id_1 == fighter_id_2, "Unit type ID should be consistent across turns"
        print(f"\n  ✓ Unit type IDs consistent across turns")
        
        # Verify state changes
        assert state2.global_info.turn_number == 2, "Turn number should advance"
        assert state2.sides[0].current_gold < state1.sides[0].current_gold, "Gold should decrease"
        print(f"  ✓ Game state updates correctly")
        
        return True
        
    except Exception as e:
        print(f"\n  ✗ Multi-turn test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run integration tests."""
    print("\n" + "=" * 70)
    print("  Integration Test Suite")
    print("=" * 70)
    
    tests = [
        test_full_pipeline,
        test_multiple_turns,
    ]
    
    results = []
    for test_func in tests:
        try:
            passed = test_func()
            results.append(passed)
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    passed_count = sum(results)
    total_count = len(results)
    
    print("\n" + "=" * 70)
    print(f"  Integration Tests: {passed_count}/{total_count} passed")
    print("=" * 70)
    
    if passed_count == total_count:
        print("\n🎉 All integration tests passed!")
        print("\nYou're ready to test with real Wesnoth!")
        return 0
    else:
        print(f"\n❌ {total_count - passed_count} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
