# Migration Notes: Server-Based to Local Execution

## Summary

Migrated from network-based multiplayer server architecture to local game process management with file-based IPC.

## Changes

### Removed
- `server.py` - Network server implementation
- Server constants from `assumptions.py` (HOST, PORT, TIMEOUT, LOG_FILE, HANDSHAKE_SIZE, CHUNK_SIZE)

### Added
- `local_game_launcher.py` - Local Wesnoth process management
- `wesnoth_plugin/` - Lua AI addon
  - `ai_plugin.lua` - Game state serialization and action execution
  - `_main.cfg` - Addon configuration
  - `training_scenario.cfg` - Scenario template
- `train.py` - Main entry point
- Test suite (`test_*.py`)
- Documentation (`README.md`, `TESTING.md`, `QUICK_START.md`)
- `.gitignore`

### Modified
- `game_manager.py` - Integrated with LocalGameManager
- `classes.py` - Added GameConfig dataclass
- `assumptions.py` - Replaced server constants with local execution constants
- `improvements.md` - Updated status

## Architecture Comparison

### Previous (Server-Based)
```
Wesnoth Client ←TCP→ Python Server ←TCP→ Wesnoth Client
                         ↓
                     AI Model
```

### Current (Local Execution)
```
Python Training Manager
    ↓
LocalGameManager → Wesnoth Processes
    ↓
AI Model ←JSON Files→ Lua AI Plugin
```

## File-Based IPC Protocol

Each game instance uses:
- `state.json` - Wesnoth → Python (game state)
- `action.json` - Python → Wesnoth (AI decision)
- `signal` - Notification flag (empty file)

**Flow:**
1. Lua AI writes state → creates signal
2. Python detects signal → reads state → deletes signal
3. Python writes action
4. Lua AI reads action → deletes action → executes

## Benefits

- Eliminates network complexity
- Direct process control
- Easier debugging (inspect JSON files)
- No network latency
- Simpler deployment

## Implementation Status

**Completed:**
- Process management framework
- IPC protocol implementation
- Lua plugin structure
- Training manager integration
- Test suite

**Pending:**
- Lua plugin validation with Wesnoth
- State format verification
- Action execution testing
- Performance optimization

## Testing

```bash
# No Wesnoth required
python test_imports.py
python test_basic.py
python test_mock_game.py

# With Wesnoth
python train.py --num-games 1 --log-level DEBUG
```

## Rollback

```bash
git checkout HEAD~1 server.py assumptions.py game_manager.py improvements.md
rm -rf wesnoth_plugin/ local_game_launcher.py train.py test_*.py *.md .gitignore
```
