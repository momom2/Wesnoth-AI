#!/usr/bin/env python3
# test_single_training_run.py
# Quick test of the training system with simulator

import asyncio
import logging
from pathlib import Path

from game_manager import TrainingManager
from local_game_launcher import WesnothConfig


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_short_training():
    """Test training with simulator for a few iterations."""
    logger = logging.getLogger("test")

    logger.info("Starting short training test with simulator")

    # Create dummy config (won't be used with simulator)
    wesnoth_config = WesnothConfig(
        wesnoth_exe=Path("wesnoth"),
        userdata_dir=Path.home() / ".local/share/wesnoth/1.18",
        addon_dir=Path.home() / ".local/share/wesnoth/1.18/data/add-ons"
    )

    # Create manager with just 1 game
    manager = TrainingManager(num_parallel_games=1, wesnoth_config=wesnoth_config)

    logger.info("Training manager created, starting game...")

    # Run for a limited time
    try:
        # Run for 30 seconds
        await asyncio.wait_for(manager.run_training_loop(), timeout=30.0)
    except asyncio.TimeoutError:
        logger.info("Test timeout reached (this is expected)")
    except KeyboardInterrupt:
        logger.info("Test interrupted")
    finally:
        await manager.cleanup()
        logger.info("Test complete")

    # Check that we processed some states
    logger.info(f"Training stats: {manager.training_stats}")

    return 0


if __name__ == "__main__":
    exit(asyncio.run(test_short_training()))
