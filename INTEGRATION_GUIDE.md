# Wesnoth Transformer AI Integration Guide

## Overview

This project implements a reinforcement learning AI for Battle for Wesnoth using a transformer-based architecture. The AI integrates with Wesnoth through a Lua bridge that communicates with a Python server.

## Architecture

```
┌─────────────────┐
│  Wesnoth Game   │
│   (Lua AI)      │
└────────┬────────┘
         │ JSON over TCP
         │ (Port 15001)
         ▼
┌─────────────────┐
│  ai_server.py   │
│  (AI Server)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   transformer   │
│  (AI Model)     │
└─────────────────┘
```

### Components

1. **lua_ai_bridge.lua**: Lua script that runs inside Wesnoth and collects game state
2. **ai_server.py**: Python server that receives game states and returns actions
3. **transformer.py**: Neural network model for action selection
4. **game_manager.py**: Training manager for self-play and learning
5. **server.py**: WML protocol server (for testing/alternative integration)

## Setup Instructions

### Prerequisites

- Battle for Wesnoth (1.14+)
- Python 3.8+
- PyTorch
- NumPy

### Installation Steps

1. **Install Python dependencies**:
   ```bash
   pip install torch numpy
   ```

2. **Set up Wesnoth add-on**:
   ```bash
   # Create add-on directory
   mkdir -p ~/.local/share/wesnoth/1.XX/data/add-ons/TransformerAI

   # Copy Lua bridge
   cp lua_ai_bridge.lua ~/.local/share/wesnoth/1.XX/data/add-ons/TransformerAI/
   ```

3. **Install LuaSocket** (required for TCP communication in Lua):
   - On Linux: `luarocks install luasocket`
   - On Windows: Download from LuaSocket website or use LuaRocks for Windows
   - On macOS: `luarocks install luasocket`

4. **Install JSON library for Lua**:
   - `luarocks install dkjson` or `luarocks install luajson`

5. **Copy test scenario**:
   ```bash
   cp test_scenario.cfg ~/.local/share/wesnoth/1.XX/data/scenarios/
   ```

## Running the AI

### Method 1: AI Server Mode (Recommended)

This mode allows the transformer AI to play Wesnoth games in real-time.

1. **Start the AI server**:
   ```bash
   python ai_server.py
   ```

2. **Launch Wesnoth and load the test scenario**:
   - Start Wesnoth
   - Go to Multiplayer → Custom Game
   - Load "Transformer AI Test" scenario
   - Start the game

3. **The AI will**:
   - Connect to the server on port 15001
   - Send game state each turn
   - Execute actions returned by the transformer model

### Method 2: WML Server Mode (Legacy/Testing)

This mode was the original approach using the Wesnoth multiplayer protocol.

1. **Start the WML server**:
   ```bash
   python server.py
   ```

2. **Connect Wesnoth client**:
   - This requires more extensive integration with Wesnoth's multiplayer system
   - Currently semi-functional

## Configuration

### assumptions.py

Key configuration parameters:

```python
# Model parameters
MAX_UNIT_TYPE = 1000
MAX_ATTACKS = 4
MEMORY_STATE_SIZE = 256

# Training parameters
MAX_ACTIONS_ALLOWED = 2000
ACTION_PENALTY = 0.01
TIMEOUT_PENALTY = 1.0
REPLAY_BUFFER_SIZE = 100000
REPLAY_BATCH_SIZE = 512

# Server parameters
HOST = 'localhost'
PORT = 15000  # WML server
# AI server uses PORT + 1 (15001)
```

### lua_ai_bridge.lua

Configuration at the top of the file:

```lua
local HOST = "localhost"
local PORT = 15001
local TIMEOUT = 5
```

## Current Status

### ✅ Completed
- Basic transformer architecture (transformer.py)
- Class definitions for game state (classes.py)
- WML protocol server (server.py) - semi-functional
- JSON-based AI server (ai_server.py)
- Lua bridge for Wesnoth integration
- Test scenario configuration

### 🚧 In Progress
- Action selection logic in ai_server.py
- Training loop in game_manager.py
- Experience replay and learning

### ⏳ Todo
- Complete action selection from transformer outputs
- Implement proper state conversion in game_manager.py
- Add self-play training loop
- Implement EfficientZero improvements (value prefix, consistency loss)
- Add proper fog of war handling
- Implement advancement selection
- Add detailed logging and monitoring
- Create training metrics dashboard

## Architecture Details

### State Representation

The AI receives game state as:

```json
{
  "game_id": "game_12345",
  "turn": 5,
  "side": 1,
  "gold": 150,
  "map": {
    "width": 30,
    "height": 30,
    "hexes": [...],
    "units": [...]
  },
  "recruits": [...],
  "game_over": false
}
```

### Action Format

The AI returns actions as:

```json
{
  "type": "action",
  "action_type": "move",  // or "attack", "recruit", "end_turn"
  "unit_id": "unit_123",
  "target_x": 15,
  "target_y": 20,
  "weapon_index": 0
}
```

### Transformer Model

The model outputs:
- **start_logits**: Probability distribution over which unit to act with
- **target_logits**: Probability distribution over target hexes
- **attack_logits**: Probability of attacking vs moving
- **recruit_logits**: Probability distribution over unit types to recruit
- **value**: Estimated value of current state

## Known Issues

### server.py Issues
- [request_choice] handling was failing (fixed)
- Limited WML protocol implementation
- Only handles basic game setup, not full game state

### Integration Challenges
- LuaSocket may not be available in all Wesnoth builds
- JSON library needs to be installed separately for Lua
- Wesnoth's Lua AI API has some limitations

### Model Issues
- Action selection logic not fully implemented
- Training loop incomplete
- No checkpoint loading/saving in ai_server.py

## Debugging

### Enable verbose logging:

In server.py or ai_server.py:
```python
logging.getLogger().setLevel(logging.DEBUG)
```

In lua_ai_bridge.lua:
```lua
wesnoth.log("debug", "Message here")
```

### Check logs:
- `wesnoth_AI_test_server.log` - WML server log
- `ai_server.log` - AI server log
- `~/.local/share/wesnoth/1.XX/stderr.txt` - Wesnoth Lua errors

### Common Issues

**"Failed to connect to AI server"**
- Ensure ai_server.py is running
- Check firewall settings
- Verify port 15001 is not in use

**"Module 'socket' not found"**
- Install LuaSocket: `luarocks install luasocket`
- Check Wesnoth's Lua path includes LuaRocks modules

**"Module 'json' not found"**
- Install dkjson: `luarocks install dkjson`
- Or install luajson: `luarocks install luajson`

## Next Steps

1. **Complete action selection**:
   - Implement `_select_action()` in ai_server.py
   - Add proper sampling from logits
   - Handle invalid actions gracefully

2. **Implement training loop**:
   - Complete `maybe_update_ai()` in game_manager.py
   - Add policy and value loss calculation
   - Implement consistency loss (EfficientZero)

3. **Add self-play**:
   - Support multiple concurrent games
   - Implement opponent AI (random or self-play)
   - Add game outcome tracking

4. **Improve state representation**:
   - Add fog of war
   - Include defense caps
   - Track gold explicitly
   - Add terrain effects

5. **Add monitoring**:
   - Win rate tracking
   - Average game length
   - Loss curves
   - Action distribution

## References

- [Wesnoth Wiki - LuaAI](https://wiki.wesnoth.org/LuaAI)
- [Wesnoth Wiki - Creating Custom AIs](https://wiki.wesnoth.org/Creating_Custom_AIs)
- [Wesnoth Wiki - MultiplayerServerWML](https://wiki.wesnoth.org/MultiplayerServerWML)
- [EfficientZero Paper](https://arxiv.org/abs/2111.00210)
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)

## Contributing

When making changes:

1. Update relevant documentation
2. Test with ai_server.py before committing
3. Check logs for errors
4. Document any new assumptions in assumptions.py
5. Add TODOs with specific action items

## License

[Add license information]
