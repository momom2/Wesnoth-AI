# Troubleshooting Guide

## Issue: Timeout Waiting for Game State

**Symptoms:**
```
ERROR - Timeout waiting for game state
```

**Root Cause:** Wesnoth process is running but the Lua plugin is not generating state files.

### Diagnostic Steps

#### 1. Stop the Training Process

```bash
# Press Ctrl+C to stop
# Then kill any remaining Wesnoth processes
pkill wesnoth  # Linux/macOS
taskkill /F /IM wesnoth.exe  # Windows
```

#### 2. Check Game Directory

```bash
ls -la games/game_0/
```

**Expected files:**
- `scenario.cfg` - Generated scenario file
- `state.json` - Game state (MISSING if plugin not working)
- `signal` - Signal file (MISSING if plugin not working)

**If state.json doesn't exist:** Plugin isn't running.

#### 3. Test Wesnoth Manually

The current approach requires extensive Wesnoth integration. Since the Lua plugin is complex and untested, we should **pause Wesnoth integration** and focus on the Python implementation first.

## Recommended Approach

### Option 1: Continue with Mock Games (Recommended)

The mock game tests work perfectly. Continue development using the mock game infrastructure:

```bash
# These all pass
python test_imports.py
python test_basic.py
python test_mock_game.py
```

**Benefits:**
- Fast iteration
- No Wesnoth dependency
- Test all Python logic
- Validate training pipeline

**Next Steps:**
1. Complete transformer model implementation
2. Implement training loss functions
3. Test experience replay with mock data
4. Add checkpoint save/load
5. Validate training loop logic

### Option 2: Simplify Wesnoth Integration

The current Lua plugin is complex and makes many assumptions. A simpler approach:

**Create a Python-based game simulator:**
- Implement basic Wesnoth game rules in Python
- No Lua, no IPC complexity
- Fast, debuggable, testable
- Get training working first
- Add real Wesnoth later

### Option 3: Debug Lua Plugin (Advanced)

If you need real Wesnoth integration now:

#### A. Verify Lua Syntax

```bash
# Check Lua syntax
lua -c wesnoth_plugin/ai_plugin.lua
```

#### B. Test Plugin Separately

Create a minimal Wesnoth scenario to test the plugin in isolation.

#### C. Add Debug Logging

Modify `ai_plugin.lua` to write debug logs:

```lua
-- Add at the start of functions
local function debug_log(msg)
    local f = io.open("lua_debug.log", "a")
    f:write(os.date("%Y-%m-%d %H:%M:%S") .. " - " .. msg .. "\n")
    f:close()
end

-- Use throughout
debug_log("Plugin loaded")
debug_log("evaluation() called")
```

#### D. Test JSON Encoding

Wesnoth's `wesnoth.json_encode` might not be available or work as expected. Test with:

```lua
-- In Wesnoth debug console
:lua print(wesnoth.json_encode({test = "value"}))
```

## Current State Assessment

### What Works ✓
- Python architecture is sound
- File-based IPC protocol is well-designed
- Mock games demonstrate the concept
- All Python tests pass

### What Doesn't Work ✗
- Lua plugin untested and likely has issues
- Wesnoth integration is speculative
- No validation of Lua API usage
- Complex IPC without debugging tools

## Recommendation

**Focus on Python-side implementation first:**

1. Use mock games for development
2. Complete the transformer model
3. Implement and test training logic
4. Get a working training pipeline with mock data
5. **Then** tackle Wesnoth integration with proper testing

**Wesnoth integration is a separate project** that requires:
- Understanding Wesnoth's Lua API
- Testing in actual Wesnoth environment
- Iterative debugging
- Likely multiple rewrites

Don't let Wesnoth integration block progress on the AI model.

## Quick Fix: Disable Wesnoth, Use Mocks

To continue with the current codebase without Wesnoth:

### 1. Create Mock Game Mode

Add to `game_manager.py`:

```python
# At the top
USE_MOCK_GAMES = True  # Set to False when Wesnoth ready

# In maintain_game_count
if USE_MOCK_GAMES:
    # Use mock game instead
    from test_mock_game import MockWesnothGame
    wesnoth_game = MockWesnothGame(config)
    asyncio.create_task(wesnoth_game.run())
else:
    # Real Wesnoth
    wesnoth_game = await self.local_game_manager.create_game(config)
```

### 2. Run Training with Mocks

Now you can test the full training pipeline without Wesnoth:

```bash
python train.py --num-games 4
```

This will exercise:
- Game management
- State processing
- AI decision making
- Experience replay
- Checkpoint saving

All without needing Wesnoth to work.

## When to Revisit Wesnoth Integration

Revisit Wesnoth integration when:
1. Training pipeline works with mock games
2. Model is implemented and training
3. You're ready to validate on real game data
4. You have time to debug Lua integration properly

At that point, create a separate branch and methodically test:
1. Lua syntax
2. Plugin loads in Wesnoth
3. Basic state serialization
4. Action execution
5. Full game loop
