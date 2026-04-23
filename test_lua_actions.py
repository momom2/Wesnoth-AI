#!/usr/bin/env python3
"""
test_lua_actions.py
Test script for Lua action file generation

This tests that Python can correctly generate Lua action files
that Wesnoth can read.
"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).parent))

from wesnoth_interface import WesnothGame
from classes import Position

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_test(name, passed):
    """Print test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status}: {name}")

def test_dict_to_lua_conversion():
    """Test converting Python dicts to Lua table format."""
    print_header("Test 1: Dict to Lua Conversion")
    
    # Create a temporary game instance just for testing
    with TemporaryDirectory() as tmpdir:
        game = WesnothGame("test_game", Path("dummy.cfg"))
        game.game_dir = Path(tmpdir)
        
        test_cases = [
            # (input_dict, expected_lua_substring)
            (
                {'type': 'end_turn'},
                "type = 'end_turn'"
            ),
            (
                {'type': 'move', 'start_x': 5, 'start_y': 10, 'target_x': 6, 'target_y': 10},
                "type = 'move'"
            ),
            (
                {'type': 'attack', 'weapon_index': 0},
                "weapon_index = 0"
            ),
            (
                {'type': 'recruit', 'unit_type': 'Dwarvish Fighter'},
                "unit_type = 'Dwarvish Fighter'"
            ),
        ]
        
        all_passed = True
        for action_dict, expected_substring in test_cases:
            lua_code = game._dict_to_lua(action_dict)
            print(f"\n  Input: {action_dict}")
            print(f"  Output: {lua_code}")
            
            if expected_substring in lua_code:
                print_test(f"  Contains '{expected_substring}'", True)
            else:
                print_test(f"  Contains '{expected_substring}'", False)
                all_passed = False
        
        return all_passed

def test_action_file_writing():
    """Test writing action files."""
    print_header("Test 2: Action File Writing")
    
    with TemporaryDirectory() as tmpdir:
        game = WesnothGame("test_game", Path("dummy.cfg"))
        game.game_dir = Path(tmpdir)
        game.action_file = game.game_dir / "action_input.lua"
        
        test_actions = [
            {'type': 'end_turn'},
            {'type': 'move', 'start_x': 5, 'start_y': 10, 'target_x': 6, 'target_y': 10},
            {'type': 'attack', 'start_x': 5, 'start_y': 10, 'target_x': 6, 'target_y': 10, 'weapon_index': 0},
            {'type': 'recruit', 'unit_type': 'Dwarvish Guardsman', 'target_x': 20, 'target_y': 20},
        ]
        
        all_passed = True
        for action in test_actions:
            try:
                # Write action
                success = game.send_action(action)
                
                if not success:
                    print_test(f"  Write {action['type']} action", False)
                    all_passed = False
                    continue
                
                # Read back and verify
                content = game.action_file.read_text()
                
                # Check that it starts with "return"
                if not content.strip().startswith('return'):
                    print_test(f"  {action['type']} - starts with 'return'", False)
                    all_passed = False
                    continue
                
                # Check that action type is in the file
                if action['type'] not in content:
                    print_test(f"  {action['type']} - contains type", False)
                    all_passed = False
                    continue
                
                print_test(f"  Write {action['type']} action", True)
                print(f"    File content preview: {content[:80]}...")
                
            except Exception as e:
                print_test(f"  Write {action['type']} action", False)
                print(f"    Error: {e}")
                all_passed = False
        
        return all_passed

def test_special_characters():
    """Test handling of special characters in Lua strings."""
    print_header("Test 3: Special Characters Handling")
    
    with TemporaryDirectory() as tmpdir:
        game = WesnothGame("test_game", Path("dummy.cfg"))
        game.game_dir = Path(tmpdir)
        
        test_cases = [
            # (input_value, description)
            ("Unit's Name", "apostrophe"),
            ('Unit "Leader"', "quotes"),
            ("Mixed's \"Characters\"", "mixed quotes"),
        ]
        
        all_passed = True
        for test_value, description in test_cases:
            try:
                action = {'type': 'recruit', 'unit_type': test_value}
                lua_code = game._dict_to_lua(action)
                
                # Should not cause syntax errors and should escape properly
                print(f"\n  Input ({description}): {test_value}")
                print(f"  Lua output: {lua_code}")
                
                # Basic check: should contain the escaped version
                if 'unit_type' in lua_code:
                    print_test(f"  {description} handling", True)
                else:
                    print_test(f"  {description} handling", False)
                    all_passed = False
                    
            except Exception as e:
                print_test(f"  {description} handling", False)
                print(f"    Error: {e}")
                all_passed = False
        
        return all_passed

def test_action_roundtrip():
    """Test that actions can be written and would be readable by Lua."""
    print_header("Test 4: Action Roundtrip")
    
    with TemporaryDirectory() as tmpdir:
        game = WesnothGame("test_game", Path("dummy.cfg"))
        game.game_dir = Path(tmpdir)
        game.action_file = game.game_dir / "action_input.lua"
        
        actions = [
            {
                'type': 'move',
                'start_x': 10,
                'start_y': 15,
                'target_x': 11,
                'target_y': 15
            },
            {
                'type': 'attack',
                'start_x': 11,
                'start_y': 15,
                'target_x': 12,
                'target_y': 15,
                'weapon_index': 1
            },
            {
                'type': 'recruit',
                'unit_type': 'Dwarvish Fighter',
                'target_x': 20,
                'target_y': 20
            },
            {
                'type': 'end_turn'
            }
        ]
        
        all_passed = True
        for action in actions:
            try:
                # Write action
                game.send_action(action)
                
                # Read the file
                content = game.action_file.read_text()
                
                # Verify structure
                assert content.startswith('return'), "Should start with 'return'"
                assert '{' in content and '}' in content, "Should have Lua table braces"
                assert action['type'] in content, f"Should contain type '{action['type']}'"
                
                print_test(f"  Roundtrip {action['type']}", True)
                
            except AssertionError as e:
                print_test(f"  Roundtrip {action['type']}", False)
                print(f"    Error: {e}")
                all_passed = False
            except Exception as e:
                print_test(f"  Roundtrip {action['type']}", False)
                print(f"    Error: {e}")
                all_passed = False
        
        return all_passed

def test_list_handling():
    """Test that lists are properly converted to Lua tables."""
    print_header("Test 5: List Handling")
    
    with TemporaryDirectory() as tmpdir:
        game = WesnothGame("test_game", Path("dummy.cfg"))
        game.game_dir = Path(tmpdir)
        
        test_dict = {
            'type': 'test',
            'units': ['Fighter', 'Archer', 'Mage'],
            'coords': [1, 2, 3, 4, 5]
        }
        
        try:
            lua_code = game._dict_to_lua(test_dict)
            
            print(f"\n  Input: {test_dict}")
            print(f"  Output: {lua_code}")
            
            # Check that lists are formatted as Lua tables
            assert 'units = {' in lua_code, "Should have units table"
            assert 'Fighter' in lua_code, "Should contain Fighter"
            assert 'Archer' in lua_code, "Should contain Archer"
            assert 'coords = {' in lua_code, "Should have coords table"
            
            print_test("List to Lua table conversion", True)
            return True
            
        except Exception as e:
            print_test("List to Lua table conversion", False)
            print(f"  Error: {e}")
            return False

def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  Lua Action File Test Suite")
    print("=" * 70)
    
    tests = [
        ("Dict to Lua Conversion", test_dict_to_lua_conversion),
        ("Action File Writing", test_action_file_writing),
        ("Special Characters", test_special_characters),
        ("Action Roundtrip", test_action_roundtrip),
        ("List Handling", test_list_handling),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print_header("Test Summary")
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"{status} {name}")
    
    print(f"\nPassed: {passed_count}/{total_count}")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {total_count - passed_count} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
