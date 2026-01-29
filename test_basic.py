#!/usr/bin/env python3
# test_basic.py
# Basic tests that don't require Wesnoth installation

import asyncio
import json
from pathlib import Path
import tempfile
import shutil

from classes import GameConfig
from game_manager import TrainingManager
from local_game_launcher import WesnothConfig


def test_game_config_creation():
    """Test that GameConfig can be created properly."""
    print("Testing GameConfig creation...", end=" ")

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "game_0"
        base_path.mkdir(parents=True)

        config = GameConfig(
            game_id="game_0",
            map_name="test_map",
            faction1="Rebels",
            faction2="Northerners",
            state_file=base_path / "state.json",
            action_file=base_path / "action.json",
            signal_file=base_path / "signal"
        )

        assert config.game_id == "game_0"
        assert config.state_file.parent == base_path
        print("OK")


def test_wesnoth_config():
    """Test WesnothConfig creation."""
    print("Testing WesnothConfig...", end=" ")

    config = WesnothConfig(
        wesnoth_exe=Path("wesnoth"),
        userdata_dir=Path.home() / ".local/share/wesnoth/1.16",
        addon_dir=Path.home() / ".local/share/wesnoth/1.16/data/add-ons"
    )

    assert config.wesnoth_exe == Path("wesnoth")
    print("OK")


def test_state_file_format():
    """Test that we can create and parse expected state file format."""
    print("Testing state file format...", end=" ")

    # Create a mock state file
    state = {
        "game_id": "test_game",
        "turn": 1,
        "side": 1,
        "gold": 100,
        "game_over": False,
        "winner": None,
        "map": {
            "width": 30,
            "height": 20,
            "mask": [],
            "fog": [],
            "hexes": [
                {
                    "x": 10,
                    "y": 10,
                    "terrain_types": ["FLAT"],
                    "modifiers": []
                }
            ],
            "units": [
                {
                    "name": "Elvish Archer",
                    "side": 1,
                    "is_leader": False,
                    "x": 10,
                    "y": 10,
                    "max_hp": 29,
                    "max_moves": 6,
                    "max_exp": 36,
                    "cost": 14,
                    "alignment": "NEUTRAL",
                    "levelup_names": ["Elvish Ranger", "Elvish Marksman"],
                    "current_hp": 29,
                    "current_moves": 6,
                    "current_exp": 0,
                    "has_attacked": False,
                    "attacks": [
                        {
                            "name": "bow",
                            "type": "PIERCE",
                            "damage": 5,
                            "strikes": 4,
                            "is_ranged": True,
                            "specials": []
                        }
                    ],
                    "resistances": {
                        "blade": 0,
                        "pierce": 0,
                        "impact": 0,
                        "fire": 0,
                        "cold": 0,
                        "arcane": 0
                    },
                    "defenses": {},
                    "movement_costs": {},
                    "abilities": [],
                    "traits": ["DEXTROUS"],
                    "statuses": {
                        "poisoned": False,
                        "slowed": False,
                        "petrified": False
                    }
                }
            ]
        },
        "recruits": [
            {
                "name": "Elvish Fighter",
                "hp": 33,
                "moves": 5,
                "exp": 32,
                "cost": 14,
                "alignment": "NEUTRAL",
                "levelup_names": ["Elvish Captain", "Elvish Hero"],
                "attacks": [],
                "resistances": {},
                "defenses": {},
                "movement_costs": {},
                "abilities": [],
                "traits": []
            }
        ]
    }

    # Test JSON serialization
    with tempfile.TemporaryDirectory() as tmpdir:
        state_file = Path(tmpdir) / "state.json"

        # Write
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

        # Read back
        with open(state_file, 'r') as f:
            loaded_state = json.load(f)

        assert loaded_state["game_id"] == "test_game"
        assert loaded_state["map"]["width"] == 30
        assert len(loaded_state["map"]["units"]) == 1
        assert loaded_state["map"]["units"][0]["name"] == "Elvish Archer"

    print("OK")


def test_action_file_format():
    """Test action file format."""
    print("Testing action file format...", end=" ")

    actions = [
        {
            "type": "move",
            "start_x": 10,
            "start_y": 10,
            "target_x": 11,
            "target_y": 10
        },
        {
            "type": "attack",
            "start_x": 11,
            "start_y": 10,
            "target_x": 12,
            "target_y": 10,
            "attack_index": 0
        },
        {
            "type": "recruit",
            "unit_type": "Elvish Fighter",
            "target_x": 9,
            "target_y": 9
        },
        {
            "type": "end_turn"
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        action_file = Path(tmpdir) / "action.json"

        for action in actions:
            # Write
            with open(action_file, 'w') as f:
                json.dump(action, f)

            # Read back
            with open(action_file, 'r') as f:
                loaded_action = json.load(f)

            assert loaded_action["type"] == action["type"]

    print("OK")


def test_directory_structure():
    """Test that game directory structure is created correctly."""
    print("Testing directory structure creation...", end=" ")

    with tempfile.TemporaryDirectory() as tmpdir:
        games_dir = Path(tmpdir) / "games"

        # Simulate creating multiple game directories
        for i in range(3):
            game_dir = games_dir / f"game_{i}"
            game_dir.mkdir(parents=True, exist_ok=True)

            state_file = game_dir / "state.json"
            action_file = game_dir / "action.json"
            signal_file = game_dir / "signal"

            assert game_dir.exists()

            # Create files
            state_file.touch()
            action_file.touch()
            signal_file.touch()

            assert state_file.exists()
            assert action_file.exists()
            assert signal_file.exists()

    print("OK")


def test_signal_mechanism():
    """Test signal file mechanism."""
    print("Testing signal file mechanism...", end=" ")

    with tempfile.TemporaryDirectory() as tmpdir:
        signal_file = Path(tmpdir) / "signal"

        # Signal doesn't exist initially
        assert not signal_file.exists()

        # Create signal
        signal_file.touch()
        assert signal_file.exists()

        # Remove signal
        signal_file.unlink()
        assert not signal_file.exists()

    print("OK")


def main():
    """Run all basic tests."""
    print("=" * 60)
    print("Running Basic Tests (no Wesnoth required)")
    print("=" * 60)

    tests = [
        test_game_config_creation,
        test_wesnoth_config,
        test_state_file_format,
        test_action_file_format,
        test_directory_structure,
        test_signal_mechanism,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL - {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
