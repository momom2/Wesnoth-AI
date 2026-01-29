#!/usr/bin/env python3
# test_imports.py
# Quick script to test that all imports work correctly

import sys

def test_imports():
    """Test that all modules can be imported."""
    errors = []

    try:
        print("Testing assumptions.py...", end=" ")
        import assumptions
        print("OK")
    except Exception as e:
        errors.append(f"assumptions.py: {e}")
        print("FAIL")

    try:
        print("Testing classes.py...", end=" ")
        import classes
        print("OK")
    except Exception as e:
        errors.append(f"classes.py: {e}")
        print("FAIL")

    try:
        print("Testing encodings.py...", end=" ")
        import encodings
        print("OK")
    except Exception as e:
        errors.append(f"encodings.py: {e}")
        print("FAIL")

    try:
        print("Testing transformer.py...", end=" ")
        import transformer
        print("OK")
    except Exception as e:
        errors.append(f"transformer.py: {e}")
        print("FAIL")

    try:
        print("Testing local_game_launcher.py...", end=" ")
        import local_game_launcher
        print("OK")
    except Exception as e:
        errors.append(f"local_game_launcher.py: {e}")
        print("FAIL")

    try:
        print("Testing game_manager.py...", end=" ")
        import game_manager
        print("OK")
    except Exception as e:
        errors.append(f"game_manager.py: {e}")
        print("FAIL")

    if errors:
        print("\n=== ERRORS ===")
        for error in errors:
            print(f"  {error}")
        return 1
    else:
        print("\n=== ALL IMPORTS SUCCESSFUL ===")
        return 0


if __name__ == "__main__":
    sys.exit(test_imports())
