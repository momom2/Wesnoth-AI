#!/usr/bin/env python
"""
Quick start script for Wesnoth AI Server
Simplifies starting the AI server with common options
"""

import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Start the Wesnoth AI Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_ai.py                    # Start with default settings
  python start_ai.py --port 15002       # Start on custom port
  python start_ai.py --load-checkpoint  # Load latest checkpoint
  python start_ai.py --training         # Enable training mode
        """
    )

    parser.add_argument(
        '--port',
        type=int,
        default=15001,
        help='Port to listen on (default: 15001)'
    )

    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Host to bind to (default: localhost)'
    )

    parser.add_argument(
        '--load-checkpoint',
        action='store_true',
        help='Load latest checkpoint if available'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to specific checkpoint file to load'
    )

    parser.add_argument(
        '--training',
        action='store_true',
        help='Enable training mode (saves experiences to replay buffer)'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Temperature for action selection (higher = more random, default: 1.0)'
    )

    parser.add_argument(
        '--exploration',
        type=float,
        default=0.1,
        help='Exploration factor (0-1, default: 0.1)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    # Import here to avoid slow startup for --help
    import logging
    from ai_server import AIServer

    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        print("Debug logging enabled")

    # Print startup info
    print("=" * 60)
    print("Wesnoth Transformer AI Server")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Temperature: {args.temperature}")
    print(f"Exploration: {args.exploration}")
    print(f"Training mode: {'Enabled' if args.training else 'Disabled'}")

    if args.load_checkpoint or args.checkpoint:
        print("Checkpoint loading: Enabled")
        if args.checkpoint:
            print(f"  Specific checkpoint: {args.checkpoint}")
    else:
        print("Checkpoint loading: Disabled (starting with fresh model)")

    print("=" * 60)
    print("\nStarting server...")
    print("Press Ctrl+C to stop\n")

    # Create and start server
    try:
        server = AIServer(host=args.host, port=args.port)

        # Update action selector settings
        server.action_selector.temperature = args.temperature
        server.action_selector.exploration_factor = args.exploration

        # Load specific checkpoint if requested
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
            if checkpoint_path.exists():
                print(f"Loading checkpoint: {checkpoint_path}")
                import torch
                checkpoint = torch.load(checkpoint_path)
                server.ai_model.load_state_dict(checkpoint['model_state'])
                print("Checkpoint loaded successfully")
            else:
                print(f"Warning: Checkpoint not found: {checkpoint_path}")
                print("Starting with fresh model")

        server.start()

    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
