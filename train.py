#!/usr/bin/env python3
# train.py
# Main entry point for training the Wesnoth AI

import asyncio
import logging
import argparse
from pathlib import Path

from game_manager import TrainingManager
from local_game_launcher import WesnothConfig


def setup_logging(log_level: str = "INFO"):
    """Configure logging for the training system."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train a transformer-based AI for Battle for Wesnoth'
    )

    parser.add_argument(
        '--num-games',
        type=int,
        default=4,
        help='Number of parallel games to run (default: 4)'
    )

    parser.add_argument(
        '--wesnoth-exe',
        type=Path,
        default=Path('wesnoth'),
        help='Path to Wesnoth executable (default: wesnoth in PATH)'
    )

    parser.add_argument(
        '--userdata-dir',
        type=Path,
        default=Path.home() / '.local/share/wesnoth/1.18',
        help='Path to Wesnoth userdata directory'
    )

    parser.add_argument(
        '--addon-dir',
        type=Path,
        default=None,
        help='Path to Wesnoth addons directory (default: userdata-dir/data/add-ons)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    parser.add_argument(
        '--resume',
        type=Path,
        default=None,
        help='Path to checkpoint to resume training from'
    )

    return parser.parse_args()


async def main():
    """Main training function."""
    args = parse_args()
    setup_logging(args.log_level)

    logger = logging.getLogger("main")
    logger.info("Starting Wesnoth AI training system")

    # Configure Wesnoth paths
    addon_dir = args.addon_dir
    if addon_dir is None:
        addon_dir = args.userdata_dir / "data/add-ons"

    wesnoth_config = WesnothConfig(
        wesnoth_exe=args.wesnoth_exe,
        userdata_dir=args.userdata_dir,
        addon_dir=addon_dir
    )

    # Validate Wesnoth installation
    if not wesnoth_config.wesnoth_exe.exists() and wesnoth_config.wesnoth_exe.name == "wesnoth":
        logger.warning(
            "Wesnoth executable not found in PATH. "
            "Please specify --wesnoth-exe or ensure 'wesnoth' is in your PATH."
        )

    logger.info(f"Wesnoth executable: {wesnoth_config.wesnoth_exe}")
    logger.info(f"Userdata directory: {wesnoth_config.userdata_dir}")
    logger.info(f"Addon directory: {wesnoth_config.addon_dir}")

    # Create training manager
    manager = TrainingManager(
        num_parallel_games=args.num_games,
        wesnoth_config=wesnoth_config
    )

    # Load checkpoint if resuming
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        # TODO: Implement checkpoint loading
        # manager.load_checkpoint(args.resume)

    # Start training
    logger.info(f"Starting training with {args.num_games} parallel games")
    try:
        await manager.run_training_loop()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        await manager.cleanup()
        logger.info("Training complete")


if __name__ == "__main__":
    asyncio.run(main())
