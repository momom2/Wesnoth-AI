# Testing Guide

## Test Levels

### Level 1: Module Import Validation

Verifies all Python modules can be imported without errors.

```bash
python test_imports.py
```

**Expected:** All imports successful.

### Level 2: Basic Functionality

Tests data structures, file I/O, and communication protocol components.

```bash
python test_basic.py
```

**Tests:**
- GameConfig and WesnothConfig creation
- JSON state/action file formats
- Directory structure management
- Signal file mechanism

**Expected:** 6/6 tests pass.

### Level 3: Mock Game Communication

Validates file-based IPC with simulated games (no Wesnoth required).

```bash
python test_mock_game.py
```

**Tests:**
- Single game communication loop
- Parallel game coordination
- State serialization/deserialization
- Action transmission

**Expected:** Both single and parallel game tests complete successfully.

### Level 4: Wesnoth Integration

Requires Wesnoth installation.

#### 4.1 Verify Installation

```bash
# Check Wesnoth
which wesnoth          # Linux/macOS
where wesnoth          # Windows
wesnoth --version

# Locate userdata directory
ls ~/.local/share/wesnoth/1.16                           # Linux
dir "%USERPROFILE%\Documents\My Games\Wesnoth1.16"       # Windows
ls ~/Library/Application\ Support/Wesnoth_1.16           # macOS
```

#### 4.2 Install Lua Plugin

**Automatic:** Plugin installs on first run.

**Manual:**
```bash
# Linux/macOS
cp -r wesnoth_plugin ~/.local/share/wesnoth/1.16/data/add-ons/External_AI

# Windows (PowerShell)
Copy-Item -Recurse wesnoth_plugin "$env:USERPROFILE\Documents\My Games\Wesnoth1.16\data\add-ons\External_AI"
```

#### 4.3 Test Plugin Loading

```bash
wesnoth --debug --log-info=scripting/lua
```

Check console for Lua errors related to `External_AI`.

### Level 5: Single Game Test

Tests actual Wesnoth process execution and communication.

```bash
python train.py --num-games 1 --log-level DEBUG
```

**Monitor:**
1. Wesnoth process starts (check process list)
2. `games/game_0/` directory created
3. `state.json` file generated
4. State contains expected fields
5. Actions transmitted and executed
6. Process terminates cleanly

**Inspect state file:**
```bash
cat games/game_0/state.json | python -m json.tool
```

### Level 6: Full Training Run

```bash
python train.py --num-games 4 --log-level INFO
```

**Monitor:**
- `training.log` for errors
- Game state updates
- CPU/memory usage
- Training statistics

## Troubleshooting

### Wesnoth Not Found

```bash
# Specify explicit path
python train.py --wesnoth-exe /usr/games/wesnoth
```

### Plugin Load Failure

Check installation:
```bash
ls ~/.local/share/wesnoth/1.16/data/add-ons/External_AI/
```

Files should include: `_main.cfg`, `ai_plugin.lua`, `training_scenario.cfg`

### State File Not Generated

1. Verify plugin installed correctly
2. Check Wesnoth console for errors
3. Increase timeout in `assumptions.py`:
   ```python
   WESNOTH_STATE_TIMEOUT = 120.0
   ```
4. Run with debug logging

### State Format Mismatch

Compare actual vs. expected:
```bash
cat games/game_0/state.json | python -m json.tool > actual_state.json
```

Verify against `game_manager.py:156-243` for expected format.

### Process Crashes

1. Check system resources (RAM, CPU)
2. Reduce parallel games: `--num-games 1`
3. Check Wesnoth logs in userdata directory
4. Run with `--log-level DEBUG`

## Validation Checklist

- [ ] All Python modules import
- [ ] Basic functionality tests pass
- [ ] Mock communication tests pass
- [ ] Wesnoth installed and accessible
- [ ] Lua plugin loads without errors
- [ ] State files generated correctly
- [ ] State format matches expectations
- [ ] Actions executed in Wesnoth
- [ ] Games complete successfully
- [ ] Parallel games function correctly
- [ ] Training loop runs without crashes

## Performance Benchmarks

Monitor these metrics during testing:

- **Game startup time:** < 10 seconds
- **State generation frequency:** 1-5 seconds per turn
- **Memory per game:** ~200-500 MB
- **CPU usage:** Proportional to `--num-games`

## Next Steps

1. Validate all test levels pass
2. Verify state serialization format
3. Test action execution for all action types
4. Profile file I/O performance
5. Implement comprehensive error handling
6. Complete transformer model training logic
