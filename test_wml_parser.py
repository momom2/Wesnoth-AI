#!/usr/bin/env python3
"""
test_wml_parser.py
Test script for WML parsing functionality

This script tests the WML parser with various test cases to ensure
it can correctly parse Wesnoth's WML format output.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

from state_converter import StateConverter
from classes import Position, Terrain, TerrainModifiers

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_test(name, passed):
    """Print test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status}: {name}")

def test_basic_wml_parsing():
    """Test basic WML structure parsing."""
    print_header("Test 1: Basic WML Parsing")
    
    wml_input = """
game_id="game_0"
current_side=1
turn_number=5
time_of_day="morning"
game_over=false
    """
    
    converter = StateConverter()
    try:
        result = converter.parse_wml(wml_input)
        
        # Check parsed values
        assert result.get('game_id') == 'game_0', "game_id should be 'game_0'"
        assert result.get('current_side') == 1, "current_side should be 1"
        assert result.get('turn_number') == 5, "turn_number should be 5"
        assert result.get('time_of_day') == 'morning', "time_of_day should be 'morning'"
        assert result.get('game_over') == False, "game_over should be False"
        
        print_test("Basic attribute parsing", True)
        print(f"  Parsed: {result}")
        return True
        
    except Exception as e:
        print_test("Basic attribute parsing", False)
        print(f"  Error: {e}")
        return False

def test_nested_tags():
    """Test parsing nested WML tags."""
    print_header("Test 2: Nested Tag Parsing")
    
    wml_input = """
game_id="game_0"
[map]
    width=50
    height=50
    [hex]
        x=10
        y=15
        full_code="Gg"
    [/hex]
    [hex]
        x=11
        y=15
        full_code="Hh"
    [/hex]
[/map]
    """
    
    converter = StateConverter()
    try:
        result = converter.parse_wml(wml_input)
        
        # Check nested structure
        assert 'map' in result, "Should have 'map' key"
        assert isinstance(result['map'], dict), "map should be a dict"
        assert result['map'].get('width') == 50, "width should be 50"
        assert result['map'].get('height') == 50, "height should be 50"
        
        # Check hex tags (should be in a list since there are multiple)
        hexes = result['map'].get('hex')
        assert hexes is not None, "Should have hex entries"
        assert isinstance(hexes, list), "Multiple hex tags should be in a list"
        assert len(hexes) == 2, "Should have 2 hexes"
        assert hexes[0].get('x') == 10, "First hex x should be 10"
        assert hexes[1].get('x') == 11, "Second hex x should be 11"
        
        print_test("Nested tag parsing", True)
        print(f"  Map width: {result['map']['width']}")
        print(f"  Hexes found: {len(hexes)}")
        return True
        
    except Exception as e:
        print_test("Nested tag parsing", False)
        print(f"  Error: {e}")
        return False

def test_multiple_same_tags():
    """Test parsing multiple tags with the same name."""
    print_header("Test 3: Multiple Same-Named Tags")
    
    wml_input = """
[side]
    side=1
    gold=100
[/side]
[side]
    side=2
    gold=150
[/side]
    """
    
    converter = StateConverter()
    try:
        result = converter.parse_wml(wml_input)
        
        # Multiple same-named tags should be in a list
        sides = result.get('side')
        assert sides is not None, "Should have side entries"
        assert isinstance(sides, list), "Multiple sides should be in a list"
        assert len(sides) == 2, "Should have 2 sides"
        assert sides[0].get('side') == 1, "First side should be side 1"
        assert sides[0].get('gold') == 100, "First side should have 100 gold"
        assert sides[1].get('side') == 2, "Second side should be side 2"
        assert sides[1].get('gold') == 150, "Second side should have 150 gold"
        
        print_test("Multiple same-named tags", True)
        print(f"  Sides found: {len(sides)}")
        print(f"  Side 1 gold: {sides[0]['gold']}")
        print(f"  Side 2 gold: {sides[1]['gold']}")
        return True
        
    except Exception as e:
        print_test("Multiple same-named tags", False)
        print(f"  Error: {e}")
        return False

def test_type_conversion():
    """Test automatic type conversion."""
    print_header("Test 4: Type Conversion")
    
    wml_input = """
string_value="hello"
int_value=42
float_value=3.14
bool_true=true
bool_false=false
quoted_number="100"
    """
    
    converter = StateConverter()
    try:
        result = converter.parse_wml(wml_input)
        
        # Check types
        assert isinstance(result.get('string_value'), str), "Should be string"
        assert result.get('string_value') == "hello", "String value should be 'hello'"
        
        assert isinstance(result.get('int_value'), int), "Should be int"
        assert result.get('int_value') == 42, "Int value should be 42"
        
        assert isinstance(result.get('float_value'), float), "Should be float"
        assert abs(result.get('float_value') - 3.14) < 0.001, "Float value should be 3.14"
        
        assert isinstance(result.get('bool_true'), bool), "Should be bool"
        assert result.get('bool_true') == True, "Bool value should be True"
        
        assert isinstance(result.get('bool_false'), bool), "Should be bool"
        assert result.get('bool_false') == False, "Bool value should be False"
        
        assert isinstance(result.get('quoted_number'), str), "Quoted number should remain string"
        assert result.get('quoted_number') == "100", "Quoted number should be '100'"
        
        print_test("Type conversion", True)
        print(f"  string_value: {result['string_value']} (type: {type(result['string_value']).__name__})")
        print(f"  int_value: {result['int_value']} (type: {type(result['int_value']).__name__})")
        print(f"  float_value: {result['float_value']} (type: {type(result['float_value']).__name__})")
        print(f"  bool_true: {result['bool_true']} (type: {type(result['bool_true']).__name__})")
        return True
        
    except Exception as e:
        print_test("Type conversion", False)
        print(f"  Error: {e}")
        return False

def test_realistic_game_state():
    """Test parsing a realistic game state structure."""
    print_header("Test 5: Realistic Game State")
    
    wml_input = """
game_id="game_0"
current_side=1
turn_number=1
time_of_day="dawn"
game_over=false
[map]
    width=44
    height=40
    [hex]
        x=1
        y=1
        full_code="Gg"
    [/hex]
    [hex]
        x=2
        y=1
        full_code="Gg^Vh"
    [/hex]
    [unit]
        id="leader_1"
        name="Dwarvish Fighter"
        side=1
        x=20
        y=20
        max_hp=40
        current_hp=40
        max_moves=5
        current_moves=5
        max_exp=32
        current_exp=0
        cost=14
        alignment="neutral"
        is_leader=true
        has_attacked=false
    [/unit]
    [unit]
        id="leader_2"
        name="Drake Clasher"
        side=2
        x=25
        y=20
        max_hp=43
        current_hp=43
        max_moves=5
        current_moves=5
        max_exp=34
        current_exp=0
        cost=20
        alignment="lawful"
        is_leader=true
        has_attacked=false
    [/unit]
[/map]
[side]
    gold=100
    village_gold=2
    base_income=0
    num_villages=0
[/side]
[side]
    gold=100
    village_gold=2
    base_income=0
    num_villages=0
[/side]
    """
    
    converter = StateConverter()
    try:
        result = converter.parse_wml(wml_input)
        
        # Validate structure
        assert result.get('game_id') == 'game_0', "game_id mismatch"
        assert result.get('current_side') == 1, "current_side mismatch"
        assert result.get('turn_number') == 1, "turn_number mismatch"
        
        # Check map
        assert 'map' in result, "Missing map"
        map_data = result['map']
        assert map_data.get('width') == 44, "Map width mismatch"
        assert map_data.get('height') == 40, "Map height mismatch"
        
        # Check hexes
        hexes = map_data.get('hex')
        assert isinstance(hexes, list), "Hexes should be a list"
        assert len(hexes) == 2, "Should have 2 hexes"
        
        # Check units
        units = map_data.get('unit')
        assert isinstance(units, list), "Units should be a list"
        assert len(units) == 2, "Should have 2 units"
        assert units[0].get('name') == "Dwarvish Fighter", "First unit name mismatch"
        assert units[1].get('name') == "Drake Clasher", "Second unit name mismatch"
        
        # Check sides
        sides = result.get('side')
        assert isinstance(sides, list), "Sides should be a list"
        assert len(sides) == 2, "Should have 2 sides"
        assert sides[0].get('gold') == 100, "Side 1 gold mismatch"
        
        print_test("Realistic game state parsing", True)
        print(f"  Game: {result['game_id']}, Turn: {result['turn_number']}")
        print(f"  Map: {map_data['width']}x{map_data['height']}")
        print(f"  Units: {len(units)}")
        print(f"  Sides: {len(sides)}")
        return True
        
    except Exception as e:
        print_test("Realistic game state parsing", False)
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_conversion_pipeline():
    """Test the full WML to GameState conversion."""
    print_header("Test 6: Full Conversion Pipeline")
    
    # Minimal but complete game state
    wml_input = """
game_id="test_game"
current_side=1
turn_number=1
time_of_day="morning"
game_over=false
[map]
    width=10
    height=10
    [hex]
        x=1
        y=1
        full_code="Gg"
    [/hex]
    [unit]
        id="test_unit"
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
        cost=17
        alignment="neutral"
        is_leader=false
        has_attacked=false
    [/unit]
[/map]
[side]
    gold=100
    village_gold=2
    base_income=0
    num_villages=0
[/side]
    """
    
    converter = StateConverter()
    try:
        # Debug: Parse and show structure
        parsed = converter.parse_wml(wml_input)
        print(f"\n  DEBUG - Parsed WML structure:")
        print(f"    Keys: {list(parsed.keys())}")
        if 'map' in parsed:
            print(f"    Map keys: {list(parsed['map'].keys())}")
            if 'unit' in parsed['map']:
                print(f"    'unit' type: {type(parsed['map']['unit'])}")
                print(f"    'unit' value: {parsed['map']['unit']}")
        
        # Test full conversion
        game_state = converter.convert_wml_to_game_state(wml_input)
        
        # Validate GameState object
        assert game_state.game_id == "test_game", "game_id mismatch"
        assert game_state.global_info.current_side == 1, "current_side mismatch"
        assert game_state.global_info.turn_number == 1, "turn_number mismatch"
        assert game_state.map.size_x == 10, "map width mismatch"
        assert game_state.map.size_y == 10, "map height mismatch"
        
        print(f"\n  DEBUG - GameState after conversion:")
        print(f"    Units count: {len(game_state.map.units)}")
        print(f"    Hexes count: {len(game_state.map.hexes)}")
        print(f"    Sides count: {len(game_state.sides)}")
        
        assert len(game_state.map.units) > 0, "Should have units"
        assert len(game_state.sides) > 0, "Should have sides"
        
        # Check coordinate conversion (Wesnoth 1-indexed -> Python 0-indexed)
        unit = list(game_state.map.units)[0]
        assert unit.position.x == 4, f"Unit x should be 4 (5-1), got {unit.position.x}"
        assert unit.position.y == 4, f"Unit y should be 4 (5-1), got {unit.position.y}"
        
        print_test("Full conversion pipeline", True)
        print(f"  Game ID: {game_state.game_id}")
        print(f"  Map size: {game_state.map.size_x}x{game_state.map.size_y}")
        print(f"  Units: {len(game_state.map.units)}")
        print(f"  First unit: {unit.name} at ({unit.position.x}, {unit.position.y})")
        print(f"  Sides: {len(game_state.sides)}")
        print(f"  Side 1 gold: {game_state.sides[0].current_gold}")
        return True
        
    except Exception as e:
        print_test("Full conversion pipeline", False)
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_coordinate_conversion():
    """Test Wesnoth 1-indexed to Python 0-indexed coordinate conversion."""
    print_header("Test 7: Coordinate Conversion")
    
    converter = StateConverter()
    
    test_cases = [
        ((1, 1), (0, 0)),
        ((5, 10), (4, 9)),
        ((20, 30), (19, 29)),
        ((1, 50), (0, 49)),
    ]
    
    all_passed = True
    for wesnoth_coords, expected_python in test_cases:
        python_coords = converter.wesnoth_to_python_coords(*wesnoth_coords)
        if python_coords != expected_python:
            print_test(f"  Wesnoth {wesnoth_coords} -> Python {expected_python}", False)
            print(f"    Got: {python_coords}")
            all_passed = False
        else:
            print_test(f"  Wesnoth {wesnoth_coords} -> Python {expected_python}", True)
    
    # Test reverse conversion
    for wesnoth_coords, python_coords in test_cases:
        back_to_wesnoth = converter.python_to_wesnoth_coords(*python_coords)
        if back_to_wesnoth != wesnoth_coords:
            print_test(f"  Python {python_coords} -> Wesnoth {wesnoth_coords}", False)
            print(f"    Got: {back_to_wesnoth}")
            all_passed = False
        else:
            print_test(f"  Python {python_coords} -> Wesnoth {wesnoth_coords}", True)
    
    return all_passed

def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  WML Parser Test Suite")
    print("=" * 70)
    
    tests = [
        ("Basic WML Parsing", test_basic_wml_parsing),
        ("Nested Tag Parsing", test_nested_tags),
        ("Multiple Same-Named Tags", test_multiple_same_tags),
        ("Type Conversion", test_type_conversion),
        ("Realistic Game State", test_realistic_game_state),
        ("Full Conversion Pipeline", test_full_conversion_pipeline),
        ("Coordinate Conversion", test_coordinate_conversion),
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
