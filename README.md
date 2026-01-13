# Wesnoth Transformer AI

A reinforcement learning AI for [Battle for Wesnoth](https://wesnoth.org/) using a transformer-based architecture.

## Quick Start

### Prerequisites

```bash
# Python dependencies
pip install torch numpy

# Lua dependencies (for Wesnoth integration)
luarocks install luasocket
luarocks install dkjson
```

### Running the AI

#### Option 1: Quick Start Script (Recommended)

```bash
# Basic start
python start_ai.py

# With options
python start_ai.py --temperature 1.5 --exploration 0.2 --load-checkpoint

# See all options
python start_ai.py --help
```

#### Option 2: Direct Start

```bash
python ai_server.py
```

### Set up Wesnoth

See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for details:
- Copy `lua_ai_bridge.lua` to your Wesnoth add-ons directory
- Load the test scenario in Wesnoth
- The AI will automatically connect and start making decisions!

## Project Structure

```
├── ai_server.py           # Main AI server (listens on port 15001)
├── start_ai.py            # Quick start script with options
├── server.py              # WML protocol server (legacy, port 15000)
├── transformer.py         # Neural network model
├── action_selector.py     # Converts model outputs to actions
├── game_manager.py        # Training manager with loss implementation
├── classes.py             # Data structures
├── assumptions.py         # Configuration parameters
├── lua_ai_bridge.lua      # Wesnoth Lua integration
├── test_scenario.cfg      # Test scenario for Wesnoth
├── test_all.py            # Comprehensive unit tests
├── README.md              # This file
├── INTEGRATION_GUIDE.md   # Detailed setup and usage guide
├── PROJECT_STATUS.md      # Complete project status report
├── CHANGELOG.md           # Version history
└── improvements.md        # Todo list and ideas
```

## Current Status

### ✅ Working
- Basic transformer architecture
- AI server with JSON protocol
- Lua bridge for Wesnoth
- Action selection from transformer outputs (recruit/move/attack)
- Test scenario configuration
- Checkpoint save/load functionality
- Training loop with value and consistency losses
- Quick start script with configuration options
- Comprehensive unit tests (16/19 passing)

### ⏳ Planned
- Self-play training
- EfficientZero improvements
- Advanced state representation (fog of war, defense caps, etc.)
- Training metrics and monitoring

## Architecture

```
┌─────────────┐
│   Wesnoth   │ ← Game runs here
│   (Lua AI)  │
└──────┬──────┘
       │ JSON/TCP
       │ Port 15001
       ▼
┌─────────────┐
│ ai_server   │ ← Python server
│   .py       │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ transformer │ ← Neural network
│   .py       │
└─────────────┘
```

The AI:
1. Wesnoth Lua script collects game state
2. Sends state to Python server as JSON
3. Transformer model outputs action probabilities
4. ActionSelector samples an action
5. Action sent back to Wesnoth
6. Wesnoth executes the action

## Key Files

### AI Server (`ai_server.py`)
Main server that hosts the transformer model and responds to game state queries.

### Action Selector (`action_selector.py`)
Converts transformer outputs (logits) into concrete Wesnoth actions:
- Recruit decisions
- Unit movement
- Attack target selection

### Transformer Model (`transformer.py`)
Neural network that outputs:
- **start_logits**: Which unit to move
- **target_logits**: Where to move/attack
- **attack_logits**: Attack vs move decision
- **recruit_logits**: Which unit to recruit
- **value**: State value estimate

### Lua Bridge (`lua_ai_bridge.lua`)
Wesnoth Lua candidate action that:
- Collects game state (units, map, fog, etc.)
- Sends to AI server
- Executes returned actions

## Configuration

See `assumptions.py` for key parameters:

```python
# Model architecture
MAX_UNIT_TYPE = 1000
MEMORY_STATE_SIZE = 256

# Training
MAX_ACTIONS_ALLOWED = 2000
REPLAY_BUFFER_SIZE = 100000
REPLAY_BATCH_SIZE = 512

# Server
HOST = 'localhost'
PORT = 15000  # WML server
# AI server uses 15001
```

## Common Issues

**"Failed to connect to AI server"**
- Make sure `ai_server.py` is running
- Check that port 15001 is available

**"Module 'socket' not found" in Wesnoth**
- Install LuaSocket: `luarocks install luasocket`

**"Module 'json' not found" in Wesnoth**
- Install dkjson: `luarocks install dkjson`

**Import errors in Python**
- Install dependencies: `pip install torch numpy`

## Documentation

- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - Detailed setup and integration guide
- **[improvements.md](improvements.md)** - Todo list and improvement ideas
- **[dependencies.md](dependencies.md)** - Required libraries

## References

- [Wesnoth LuaAI Documentation](https://wiki.wesnoth.org/LuaAI)
- [Creating Custom AIs](https://wiki.wesnoth.org/Creating_Custom_AIs)
- [MultiplayerServerWML Protocol](https://wiki.wesnoth.org/MultiplayerServerWML)
- [EfficientZero Paper](https://arxiv.org/abs/2111.00210)

## Development

To continue development:

1. **Complete training loop** (`game_manager.py`):
   - Implement loss calculation
   - Add policy and value updates
   - Implement consistency loss

2. **Add checkpoint management**:
   - Save/load model weights
   - Resume training from checkpoint

3. **Improve action selection**:
   - Better invalid action handling
   - Proper hex distance calculation
   - Movement point validation

4. **Add monitoring**:
   - Win rate tracking
   - Training metrics
   - Action distributions

## License

[Add license information]

## Contributing

[Add contribution guidelines]
