# main.py
# Entry point for Wesnoth AI training

import asyncio
import sys
import argparse
import subprocess

from game_manager import GameManager
from constants import NUM_PARALLEL_GAMES

def check_wesnoth_version():
    """Check Wesnoth version compatibility."""
    from constants import WESNOTH_PATH
    
    try:
        result = subprocess.run(
            [str(WESNOTH_PATH), "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        
        if result.returncode == 0:
            version_line = result.stdout.strip().split('\n')[0]
            # TODO: Does not work. version_line is empty. Why?
            print(f"✓ Wesnoth version: {version_line}")
            
            # Check if version is 1.14 or later (required for JSON functions)
            if "1.14" in version_line or "1.15" in version_line or "1.16" in version_line or "1.17" in version_line or "1.18" in version_line:
                return True
            else:
                print(f"WARNING: Wesnoth version may be incompatible. Recommend 1.14+. Wesnoth version: {version_line}")
                print("Some Lua functions (wesnoth.format_json, wesnoth.parse_json) may not be available.")
                return True  # Continue anyway
        else:
            print("WARNING: Could not determine Wesnoth version")
            return True
    except Exception as e:
        print(f"WARNING: Could not check Wesnoth version: {e}")
        return True

def check_setup():
    """Check that all required directories and files exist."""
    from constants import (
        WESNOTH_PATH, BASE_PATH, ADDONS_PATH, SCENARIOS_PATH,
        LUA_PATH, LOGS_PATH, CHECKPOINTS_PATH, REPLAYS_PATH
    )
    
    print("Checking setup...")
    
    # Check Wesnoth executable
    if not WESNOTH_PATH.exists():
        print(f"ERROR: Wesnoth not found at {WESNOTH_PATH}")
        print("Please update WESNOTH_PATH in constants.py")
        return False
    print(f"✓ Wesnoth found at {WESNOTH_PATH}")
    
    # Check Wesnoth version
    if not check_wesnoth_version():
        return False
    
    # Create directories if needed
    for dir_path in [LOGS_PATH, CHECKPOINTS_PATH, REPLAYS_PATH]:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ {dir_path}")
    
    # Check Lua files
    lua_files = [
        LUA_PATH / "state_collector.lua",
        LUA_PATH / "action_executor.lua",
        LUA_PATH / "ca_state_sender.lua",
        LUA_PATH / "ca_action_executor.lua"
    ]
    
    for lua_file in lua_files:
        if not lua_file.exists():
            print(f"ERROR: Lua file not found: {lua_file}")
            print("Please place Lua files in the add-ons directory")
            return False
        print(f"✓ {lua_file.name}")
    
    # Check scenario
    scenario_file = SCENARIOS_PATH / "training_scenario.cfg"
    if not scenario_file.exists():
        print(f"ERROR: Scenario not found: {scenario_file}")
        return False
    print(f"✓ training_scenario.cfg")
    
    # Check add-ons directory
    if not ADDONS_PATH.exists():
        print(f"ERROR: Add-ons directory not found: {ADDONS_PATH}")
        return False
    # Check AI config
    ai_config = ADDONS_PATH / "ai_config.cfg"
    if not ai_config.exists():
        print(f"ERROR: AI config not found: {ai_config}")
        return False
    print(f"✓ ai_config.cfg")
    # Check _main.cfg
    main_cfg = ADDONS_PATH / "_main.cfg"
    if not main_cfg.exists():
        print(f"ERROR: _main.cfg not found: {main_cfg}")
        return False
    print(f"✓ _main.cfg")
    
    print("\n✓ All checks passed!")
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Wesnoth AI Training")
    parser.add_argument(
        '--games',
        type=int,
        default=NUM_PARALLEL_GAMES,
        help=f'Number of parallel games (default: {NUM_PARALLEL_GAMES})'
    )
    parser.add_argument(
        '--check-setup',
        action='store_true',
        help='Check setup and exit'
    )

    args = parser.parse_args()

    # Check setup
    if not check_setup():
        sys.exit(1)
    
    if args.check_setup:
        print("\nSetup check complete. Run without --check-setup to start training.")
        sys.exit(0)
    
    # Start training
    print(f"\nStarting training with {args.games} parallel games...")
    print("Press Ctrl+C to stop training and save checkpoint.\n")
    
    manager = GameManager(num_games=args.games)
    
    try:
        asyncio.run(manager.run_training())
    except KeyboardInterrupt:
        print("\n\nTraining stopped by user.")
        print("Checkpoint saved.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
