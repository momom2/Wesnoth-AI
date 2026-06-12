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
    status = "âœ“ PASS" if passed else "âœ— FAIL"
    print(f"{status}: {name}")

def test_action_file_writing():
    """Test writing action files."""
    print_header("Test 2: Action File Writing")
    
    with TemporaryDirectory() as tmpdir:
        game = WesnothGame("test_game", Path("dummy.cfg"))
        game.game_dir = Path(tmpdir)
        # Repoint the action file at the test's temp dir.
        from constants import ACTION_FILE_NAME
        game.action_path = game.game_dir / ACTION_FILE_NAME
        
        test_actions = [
            {'type': 'end_turn'},
            {'type': 'move', 'start_x': 5, 'start_y': 10, 'target_x': 6, 'target_y': 10},
            {'type': 'attack', 'start_x': 5, 'start_y': 10, 'target_x': 6, 'target_y': 10, 'weapon_index': 0},
            {'type': 'recruit', 'unit_type': 'Dwarvish Guardsman', 'target_x': 20, 'target_y': 20},
        ]
        
        # Plain asserts (a returned bool passes silently under pytest).
        for action in test_actions:
            assert game.send_action(action), \
                f"send_action failed for {action['type']}"
            content = game.action_path.read_text()
            assert content.strip().startswith('return'), \
                f"{action['type']}: file must start with 'return'"
            assert action['type'] in content, \
                f"{action['type']}: action type missing from file"
            print_test(f"  Write {action['type']} action", True)

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
        
        # Plain asserts (a returned bool passes silently under pytest).
        for test_value, description in test_cases:
            action = {'type': 'recruit', 'unit_type': test_value}
            lua_code = game._dict_to_lua(action)
            assert 'unit_type' in lua_code, \
                f"{description}: unit_type key missing from Lua output"
            print_test(f"  {description} handling", True)

def test_action_roundtrip():
    """Test that actions can be written and would be readable by Lua."""
    print_header("Test 4: Action Roundtrip")
    
    with TemporaryDirectory() as tmpdir:
        game = WesnothGame("test_game", Path("dummy.cfg"))
        game.game_dir = Path(tmpdir)
        # Repoint the action file at the test's temp dir.
        from constants import ACTION_FILE_NAME
        game.action_path = game.game_dir / ACTION_FILE_NAME
        
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
        
        # Plain asserts: the old version swallowed failures into a
        # returned bool, which pytest treats as a PASS (it only
        # warns on non-None returns) -- a False result was silent.
        for action in actions:
            game.send_action(action)
            content = game.action_path.read_text()
            assert content.startswith('return'), "Should start with 'return'"
            assert '{' in content and '}' in content, "Should have Lua table braces"
            assert action['type'] in content, f"Should contain type '{action['type']}'"
            print_test(f"  Roundtrip {action['type']}", True)

def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  Lua Action File Test Suite")
    print("=" * 70)
    
    tests = [
        ("Action File Writing", test_action_file_writing),
        ("Special Characters", test_special_characters),
        ("Action Roundtrip", test_action_roundtrip),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\nâœ— Test '{name}' crashed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print_header("Test Summary")
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "âœ“" if passed else "âœ—"
        print(f"{status} {name}")
    
    print(f"\nPassed: {passed_count}/{total_count}")
    
    if passed_count == total_count:
        print("\nðŸŽ‰ All tests passed!")
        return 0
    else:
        print(f"\nâŒ {total_count - passed_count} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
