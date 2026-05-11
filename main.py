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
from typing import Optional

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


def _resolve_resume_path(spec: str) -> Optional[Path]:
    """Resolve --resume argument to an existing checkpoint file, or None
    on error (error printed to stdout)."""
    if spec == "latest":
        candidates = sorted(
            CHECKPOINTS_PATH.glob("checkpoint_*.pt"),
            # Sort numerically: checkpoint_900 > checkpoint_100 > checkpoint_90.
            key=lambda p: int(p.stem.split("_", 1)[1]) if p.stem.split("_", 1)[1].isdigit() else -1,
        )
        if not candidates:
            print(f"ERROR: --resume latest: no checkpoints in {CHECKPOINTS_PATH}")
            return None
        return candidates[-1]
    path = Path(spec)
    if not path.exists():
        print(f"ERROR: --resume path does not exist: {path}")
        return None
    return path


def main() -> int:
    # Import `policy` eagerly so --help can show the registered choices.
    import policy

    parser = argparse.ArgumentParser(description="Wesnoth AI Training")
    parser.add_argument(
        "--games",
        type=int,
        default=NUM_PARALLEL_GAMES,
        help=f"Number of parallel games (default: {NUM_PARALLEL_GAMES})",
    )
    parser.add_argument(
        "--policy",
        choices=policy.available(),
        default="dummy",
        help="Which policy drives the AI sides (default: dummy).",
    )
    parser.add_argument(
        "--check-setup",
        action="store_true",
        help="Verify setup (including add-on install) and exit.",
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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="PATH_OR_LATEST",
        help=(
            "Resume a trainable policy from a checkpoint. Pass 'latest' to "
            "auto-pick the newest checkpoint_*.pt in training/checkpoints/, "
            "or an explicit path. Silently ignored for non-trainable policies."
        ),
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help=(
            "Watch ONE game with animations on (2x turbo) instead of "
            "training. Forces --games 1, switches Wesnoth to the "
            "ai_display scenario, and disables train_step / "
            "checkpointing / observe so the loaded policy is treated "
            "as read-only. Pair with --policy transformer --resume "
            "<ckpt> to demo a trained model."
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
        print("\nSetup check complete. Run without --check-setup to start training.")
        return 0

    # Import late so a misconfigured environment doesn't crash the
    # setup-check path with unrelated import errors.
    from game_manager import GameManager

    # --display overrides --games and --policy to a sensible default
    # for "watch ONE game with the trained model". User can still pass
    # --games / --policy explicitly to combine flags, but the common
    # case (`python main.py --display --resume <ckpt>`) just works.
    if args.display:
        if args.games != NUM_PARALLEL_GAMES and args.games != 1:
            print(f"NOTE: --display forces 1 game (you passed --games {args.games}).")
        args.games = 1
        scenario_id = "ai_display"
        eval_mode = True
        mode_msg = "DISPLAY mode (no training, 2x turbo, animations on)"
    else:
        scenario_id = "ai_training"
        eval_mode = False
        mode_msg = "TRAINING mode"

    print(f"\nStarting: {args.games} parallel games, policy={args.policy}")
    print(f"  {mode_msg} (scenario={scenario_id})")
    print("Press Ctrl+C to stop.\n")

    policy_obj = policy.get_policy(args.policy)
    if args.resume is not None:
        loader = getattr(policy_obj, "load_checkpoint", None)
        if loader is None:
            print(f"NOTE: policy '{args.policy}' is not checkpoint-loadable; "
                  f"ignoring --resume.")
        else:
            ckpt_path = _resolve_resume_path(args.resume)
            if ckpt_path is None:
                return 1
            loader(ckpt_path)
            print(f"✓ Resumed from {ckpt_path.name}")

    manager = GameManager(
        num_games=args.games,
        policy=policy_obj,
        scenario_id=scenario_id,
        eval_mode=eval_mode,
    )
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
