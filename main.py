"""Entry point for Wesnoth AI training.

Responsibilities:
- Verify local setup (Wesnoth executable, add-on source files, userdata dir).
- Install the project's add-on into Wesnoth's userdata directory as a
  directory junction, so edits to `add-ons/wesnoth_ai/` show up live in
  Wesnoth without re-copying.
- Launch training.
"""

import argparse
import os
import sys
from pathlib import Path

# Windows `cmd` defaults stdout to cp1252, which can't encode the ✓
# characters our setup banners use. Reconfigure to UTF-8 so the script
# runs in a fresh terminal without `chcp 65001` or PYTHONIOENCODING.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (OSError, AttributeError):
        pass

from constants import (
    ADDON_INSTALL_PATH,
    ADDONS_PATH,
    CHECKPOINTS_PATH,
    GAMES_PATH,
    LOGS_PATH,
    LUA_PATH,
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


def _prune_stale_game_dirs(games_path) -> int:
    """Remove per-game state-channel dirs (`g<rand>/`) left behind by
    previous Wesnoth processes.

    Each parallel training game lives under `games_path/g<10-digit>/`.
    The Wesnoth-side Lua writes JSON state there via `std_print` and
    has no way to clean up (sandbox has no `os.remove`). Without
    auto-cleanup these dirs accumulate to the thousands -- gitignored
    so they don't pollute commits, but they bloat inode count and
    slow `git status` / shell tab-completion in the parent dir.

    Returns the count of removed dirs. Failures (open file handles,
    permission issues) are tolerated -- we log and skip rather than
    propagate.
    """
    import shutil
    if not games_path.exists():
        return 0
    n = 0
    for child in games_path.iterdir():
        # Only target the `g<digits>` pattern -- don't blow away any
        # other content a user may have placed in the dir manually.
        if not child.is_dir():
            continue
        name = child.name
        if not (name.startswith("g") and name[1:].isdigit()):
            continue
        try:
            shutil.rmtree(child, ignore_errors=False)
            n += 1
        except OSError as e:
            # Ignore -- usually a Wesnoth process still holds a
            # handle, or filesystem race. We'll try again next run.
            print(f"  (skipped {child.name}: {e})")
    return n


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

    # Auto-clean stale per-game state-channel dirs. Each parallel
    # game writes state JSON under `GAMES_PATH/g<rand>/`; turn_stage.lua
    # can't delete them (Lua sandbox has no `os.remove`), so they
    # accumulate. After ~thousands of training games this inflates
    # inode count without serving any purpose -- the dir contents
    # are only meaningful for the duration of the game that wrote
    # them. Sweep here at startup so the directory stays tidy.
    n_cleaned = _prune_stale_game_dirs(GAMES_PATH)
    if n_cleaned:
        print(f"✓ Cleaned {n_cleaned} stale g<NN> game-state dir(s)")

    print(f"✓ Project dirs ready")

    # turn_stage.lua is the active AI stage (custom Lua engine that
    # replaces the default Wesnoth RCA, bypassing its blacklist-on-
    # failure rule); it loads state_collector / action_executor /
    # json_encoder via wesnoth.require. ai_config.cfg holds the [ai]
    # block both sides include from training_scenario.cfg.
    # The MP variant + headless_*.lua are dormant artifacts of a prior
    # headless experiment — present in-tree but not on the load path
    # and not required here.
    required_lua = [
        LUA_PATH / "state_collector.lua",
        LUA_PATH / "action_executor.lua",
        LUA_PATH / "turn_stage.lua",
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
    """Setup + maintenance CLI for the live-Wesnoth eval pipeline.

    The training path migrated to the in-process simulator on
    2026-04-29 (`tools/sim_self_play.py`); the live-Wesnoth IPC
    training path + `--display` mode were retired 2026-05-11.
    What's left in main.py:

      - `--check-setup`: verify Wesnoth + add-on install link.
        Still useful because `tools/eval_vs_builtin.py` drives
        real Wesnoth subprocesses for evaluation against the
        built-in RCA AI.
      - `--clean-games`: sweep stale per-game state-channel dirs
        the live-Wesnoth Lua side can't clean (sandbox excludes
        `os.remove`).

    To run a self-play training cycle, use
    `tools/sim_self_play.py` (or `cluster/run.sh start selfplay`
    via the GUI).
    To watch a trained model play, use `tools/sim_demo_game.py`
    (exports a Wesnoth-loadable .bz2 the GUI can replay).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Wesnoth AI setup / maintenance CLI. Training itself "
            "runs via tools/sim_self_play.py; demos via "
            "tools/sim_demo_game.py."
        )
    )
    parser.add_argument(
        "--check-setup",
        action="store_true",
        help="Verify Wesnoth + add-on install link, then exit. "
             "Required before any live-Wesnoth eval run.",
    )
    parser.add_argument(
        "--clean-games",
        action="store_true",
        help=(
            "Sweep per-game state-channel dirs "
            "(add-ons/wesnoth_ai/games/g*/) left behind by previous "
            "Wesnoth processes, then exit. Equivalent to the "
            "auto-clean pass inside --check-setup but doesn't run "
            "the rest of the setup verification."
        ),
    )
    args = parser.parse_args()

    if args.clean_games:
        n = _prune_stale_game_dirs(GAMES_PATH)
        print(f"Removed {n} stale g<NN> game-state dir(s) from {GAMES_PATH}")
        return 0

    if not check_setup():
        return 1

    if args.check_setup:
        print("\nSetup check complete.")
        return 0

    # No more training path here; redirect the user.
    print(
        "\nNothing to do. To run a training cycle:\n"
        "  python tools/sim_self_play.py [--mcts] [...]\n"
        "  (or `cluster/run.sh start selfplay`)\n"
        "\nTo watch a trained model play one game:\n"
        "  python tools/sim_demo_game.py [--scenario multiplayer_*]\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
