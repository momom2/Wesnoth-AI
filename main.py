"""Entry point for Wesnoth AI training.

Responsibilities:
- Verify local setup (Wesnoth executable, add-on source files, userdata dir).
- Install the project's add-on into Wesnoth's userdata directory as a
  directory junction, so edits to `add-ons/wesnoth_ai/` show up live in
  Wesnoth without re-copying.
- Launch training.
"""

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path

from constants import (
    ADDON_INSTALL_PATH,
    ADDONS_PATH,
    CHECKPOINTS_PATH,
    GAMES_PATH,
    LOGS_PATH,
    LUA_PATH,
    NUM_PARALLEL_GAMES,
    REPLAYS_PATH,
    SCENARIOS_PATH,
    WESNOTH_LOGS_PATH,
    WESNOTH_PATH,
    WESNOTH_USERDATA_PATH,
)


def _is_junction_or_symlink(path: Path) -> bool:
    """True if `path` is a symlink or a Windows directory junction."""
    if path.is_symlink():
        return True
    # Windows junctions are not symlinks; detect via reparse point attribute.
    if os.name == "nt" and path.exists():
        try:
            import stat
            attrs = path.lstat().st_file_attributes  # type: ignore[attr-defined]
            return bool(attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT)
        except (AttributeError, OSError):
            return False
    return False


def install_addon() -> bool:
    """Ensure the add-on is linked into Wesnoth's userdata add-ons directory.

    Creates a directory junction (Windows) or symlink (POSIX) from
    ADDON_INSTALL_PATH → ADDONS_PATH so Wesnoth can discover the scenario
    while we keep the source of truth in the project tree.

    Returns True on success or if already correctly installed.
    """
    if not ADDONS_PATH.exists():
        print(f"ERROR: Source add-on missing at {ADDONS_PATH}")
        return False

    if not WESNOTH_USERDATA_PATH.exists():
        print(f"ERROR: Wesnoth userdata dir not found at {WESNOTH_USERDATA_PATH}")
        print("Launch Wesnoth once to create it, then re-run.")
        return False

    ADDON_INSTALL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Case 1: nothing at the install path — create the link.
    if not ADDON_INSTALL_PATH.exists() and not ADDON_INSTALL_PATH.is_symlink():
        return _create_link(ADDONS_PATH, ADDON_INSTALL_PATH)

    # Case 2: existing link — check it points at our source. Use samefile()
    # because Windows junctions resolve with a `\\?\` extended-path prefix
    # that breaks naive string comparison.
    if _is_junction_or_symlink(ADDON_INSTALL_PATH):
        try:
            if ADDON_INSTALL_PATH.samefile(ADDONS_PATH):
                print(f"✓ Add-on already linked: {ADDON_INSTALL_PATH} → {ADDONS_PATH}")
                return True
        except OSError:
            pass

        try:
            target = os.readlink(ADDON_INSTALL_PATH)
        except OSError:
            target = "<unreadable link target>"
        print(
            f"ERROR: {ADDON_INSTALL_PATH} is a link to {target}, "
            f"not to the project add-on {ADDONS_PATH}."
        )
        print("Remove it manually and re-run, or point it at the project.")
        return False

    # Case 3: a real directory lives there — refuse to clobber it.
    print(
        f"ERROR: {ADDON_INSTALL_PATH} exists and is NOT a link. Refusing to "
        f"replace it to avoid destroying user data."
    )
    print("Inspect it, move it aside, and re-run.")
    return False


def _create_link(source: Path, link: Path) -> bool:
    """Create a directory junction (Windows) or symlink (POSIX)."""
    if os.name == "nt":
        # mklink /J creates a directory junction — no admin required.
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(source)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"ERROR: mklink failed: {result.stderr.strip() or result.stdout.strip()}")
            return False
        print(f"✓ Created junction {link} → {source}")
        return True

    try:
        os.symlink(source, link, target_is_directory=True)
    except OSError as e:
        print(f"ERROR: symlink failed: {e}")
        return False
    print(f"✓ Created symlink {link} → {source}")
    return True


def check_setup() -> bool:
    """Verify everything needed before we launch Wesnoth."""
    print("Checking setup...")

    if not WESNOTH_PATH.exists():
        print(f"ERROR: Wesnoth not found at {WESNOTH_PATH}")
        print("Update WESNOTH_PATH in constants.py to match your install.")
        return False
    print(f"✓ Wesnoth executable: {WESNOTH_PATH}")

    # Wesnoth on Windows is a GUI-subsystem binary, so we can't read its
    # --version output via a pipe. Trust the path and move on.

    for dir_path in [LOGS_PATH, CHECKPOINTS_PATH, REPLAYS_PATH, GAMES_PATH]:
        dir_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Project dirs ready")

    required_lua = [
        LUA_PATH / "state_collector.lua",
        LUA_PATH / "action_executor.lua",
        LUA_PATH / "ca_turn_loop.lua",
        LUA_PATH / "json_encoder.lua",
    ]
    required_other = [
        SCENARIOS_PATH / "training_scenario.cfg",
        ADDONS_PATH / "ai_config.cfg",
        ADDONS_PATH / "_main.cfg",
    ]
    for f in required_lua + required_other:
        if not f.exists():
            print(f"ERROR: missing {f}")
            return False
    print(f"✓ Add-on source files present")

    if not install_addon():
        return False

    if not WESNOTH_LOGS_PATH.exists():
        print(
            f"NOTE: {WESNOTH_LOGS_PATH} does not exist yet. Wesnoth will "
            f"create it on first run."
        )

    print("\n✓ Setup OK.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Wesnoth AI Training")
    parser.add_argument(
        "--games",
        type=int,
        default=NUM_PARALLEL_GAMES,
        help=f"Number of parallel games (default: {NUM_PARALLEL_GAMES})",
    )
    parser.add_argument(
        "--check-setup",
        action="store_true",
        help="Verify setup (including add-on install) and exit.",
    )
    args = parser.parse_args()

    if not check_setup():
        return 1

    if args.check_setup:
        print("\nSetup check complete. Run without --check-setup to start training.")
        return 0

    # Import late so a misconfigured environment doesn't crash the
    # setup-check path with unrelated import errors.
    from game_manager import GameManager

    print(f"\nStarting training with {args.games} parallel games...")
    print("Press Ctrl+C to stop training and save checkpoint.\n")

    manager = GameManager(num_games=args.games)
    try:
        asyncio.run(manager.run_training())
    except KeyboardInterrupt:
        print("\n\nTraining stopped by user. Checkpoint saved.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
