# Wesnoth AI Training System

A transformer-based reinforcement learning system for training AI agents to play Battle for Wesnoth.

## Overview

This system trains an AI to play Wesnoth through self-play using a transformer architecture with experience replay. The implementation uses local game execution with file-based inter-process communication between Python (training) and Lua (game interface).

## Architecture

**Training Pipeline:**
- Python manages multiple parallel Wesnoth game instances
- Transformer model processes game states and generates actions
- Experience replay buffer stores and samples training data
- Periodic checkpoints save model state

**Game Integration:**
- Wesnoth runs with custom Lua AI addon
- Game state serialized to JSON and passed to Python
- Python AI decisions written to JSON and executed in-game
- File-based signaling coordinates communication

## Requirements

- Python 3.8+
- PyTorch (latest stable)
- NumPy
- Battle for Wesnoth 1.18+

## Installation

```bash
# Install Python dependencies
pip install torch numpy

# Install Wesnoth (if not already installed)
# Linux: sudo apt-get install wesnoth
# macOS: brew install wesnoth
# Windows: Download from https://wesnoth.org
```

## Usage

### Basic Training

```bash
python train.py
```

Starts training with 4 parallel games using default settings.

### Advanced Configuration

```bash
python train.py \
  --num-games 8 \
  --wesnoth-exe /path/to/wesnoth \
  --userdata-dir ~/.local/share/wesnoth/1.18 \
  --log-level INFO
```

**Options:**
- `--num-games N` - Number of parallel games (default: 4)
- `--wesnoth-exe PATH` - Wesnoth executable path
- `--userdata-dir PATH` - Wesnoth userdata directory
- `--addon-dir PATH` - Wesnoth addons directory (optional)
- `--log-level LEVEL` - Logging verbosity (DEBUG|INFO|WARNING|ERROR)
- `--resume PATH` - Resume from checkpoint

## Testing

### Without Wesnoth

```bash
python test_imports.py      # Verify module imports
python test_basic.py         # Test data structures and file I/O
python test_mock_game.py     # Test communication protocol
```

### With Wesnoth

```bash
python train.py --num-games 1 --log-level DEBUG
```

Monitor `training.log` for errors and `games/game_0/state.json` for output.

## Project Structure

```
├── train.py                    # Main entry point
├── game_manager.py             # Training orchestration
├── local_game_launcher.py      # Process management
├── transformer.py              # Model architecture
├── classes.py                  # Data structures
├── assumptions.py              # Hyperparameters
├── encodings.py                # Input/output encoding
├── wesnoth_plugin/             # Wesnoth integration
│   ├── ai_plugin.lua
│   ├── _main.cfg
│   └── training_scenario.cfg
└── test_*.py                   # Test suite
```

## Configuration

### Hyperparameters

Edit `assumptions.py`:

```python
MAX_ACTIONS_ALLOWED = 2000        # Max actions per game
REPLAY_BUFFER_SIZE = 100000       # Experience buffer size
REPLAY_BATCH_SIZE = 512           # Training batch size
CHECKPOINT_FREQUENCY = 1000       # Save every N games
MEMORY_STATE_SIZE = 256           # Memory vector size
```

### Wesnoth Paths

**Linux:** `~/.local/share/wesnoth/1.18`
**Windows:** `%USERPROFILE%\Documents\My Games\Wesnoth1.18`
**macOS:** `~/Library/Application Support/Wesnoth_1.18`

## Implementation Status

**Completed:**
- Local game execution framework
- File-based IPC protocol
- Lua AI plugin structure
- Training manager architecture
- Experience replay system
- Checkpoint/replay saving

**In Progress:**
- Transformer model implementation
- Training loss functions
- State/action validation with Wesnoth

**Planned:**
- Self-supervised consistency loss
- Value prefix prediction
- Map/faction selection logic
- Performance optimization

## References

- [Wesnoth Lua AI](https://wiki.wesnoth.org/LuaAI)
- [Custom AI Development](https://wiki.wesnoth.org/Creating_Custom_AIs)
- [WML Reference](https://wiki.wesnoth.org/ReferenceWML)

## License

Educational and research use only.
