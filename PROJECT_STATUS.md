# Project Status Report

**Date:** 2026-01-13
**Project:** Wesnoth Transformer AI
**Status:** Architecture Complete, Training Implementation Pending

## Executive Summary

A reinforcement learning AI for Battle for Wesnoth has been designed and partially implemented using a transformer-based neural network architecture. The core integration between Wesnoth and Python is complete, with action selection logic functional. Training loop implementation and full end-to-end testing remain as next steps.

## Completed Work

### 1. Architecture Design ✅
- **Lua Bridge** (`lua_ai_bridge.lua`): Wesnoth Candidate Action that communicates with Python server
- **AI Server** (`ai_server.py`): JSON-based server hosting the transformer model
- **Action Selector** (`action_selector.py`): Converts model outputs to game actions
- **Test Scenario** (`test_scenario.cfg`): Wesnoth configuration for testing

### 2. Core Components ✅
- **Transformer Model** (`transformer.py`): Neural network architecture
- **Data Structures** (`classes.py`): Complete game state representation
- **Configuration** (`assumptions.py`): All parameters and constants
- **Encodings** (`encodings.py`): Feature encoding logic

### 3. Integration ✅
- Fixed `[request_choice]` handling in `server.py`
- Created JSON protocol for Lua-Python communication
- Implemented action selection from transformer outputs
- Added comprehensive logging throughout

### 4. Documentation ✅
- **README.md**: Quick start guide
- **INTEGRATION_GUIDE.md**: Detailed setup instructions
- **improvements.md**: Status tracking and ideas
- **PROJECT_STATUS.md**: This document

### 5. Testing ✅
- **test_all.py**: Comprehensive unit test suite
- **Results**: 15/19 tests passing, 4 skipped (known issues)
- Tests cover: data structures, action selection, integration, configuration

## Current Architecture

```
┌──────────────────┐
│  Wesnoth Game    │
│  + Lua Bridge    │
└────────┬─────────┘
         │ JSON over TCP (Port 15001)
         │
         ▼
┌──────────────────┐
│  ai_server.py    │
│  + transformer   │
│  + action_select │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  game_manager    │  (Future: Training Loop)
│  + replay buffer │
└──────────────────┘
```

## Known Issues

### Critical
1. **Unit Validation Mismatch** (classes.py:188)
   - Expects 16 terrain types but Terrain enum has 17
   - Blocks proper Unit creation
   - **Fix**: Update validation or adjust Terrain enum

2. **ActionSelector Index Bug** (action_selector.py:250)
   - Hardcoded max width assumption causes IndexError
   - **Fix**: Use `map_width` parameter: `prob = target_probs[enemy['y'] * map_width + enemy['x']]`

3. **Dataclass Hashability**
   - Hex and Unit cannot be used in sets
   - Map dataclass expects sets of these types
   - **Fix**: Add `frozen=True` to dataclass decorators or change to lists

### Minor
4. **Training Loop Incomplete** (game_manager.py)
   - `maybe_update_ai()` needs loss calculation
   - Policy, value, and consistency losses not implemented
   - Checkpoint saving/loading needs completion

5. **State Conversion** (game_manager.py)
   - `convert_state_to_ai_input()` may have edge cases
   - Fog of war handling not fully implemented
   - Terrain modifiers need proper encoding

## File Status

| File | Status | Notes |
|------|--------|-------|
| `ai_server.py` | ✅ Functional | Checkpoint loading TODO |
| `action_selector.py` | ⚠️ Bug at line 250 | Index calculation wrong |
| `classes.py` | ⚠️ Validation issue | Terrain count mismatch |
| `game_manager.py` | 🚧 Incomplete | Training loop needs work |
| `lua_ai_bridge.lua` | ✅ Complete | Needs Wesnoth testing |
| `server.py` | ✅ Fixed | Legacy/testing only |
| `test_all.py` | ✅ Complete | 15/19 passing |
| `test_scenario.cfg` | ✅ Complete | Ready for Wesnoth |
| `transformer.py` | ✅ Complete | Architecture ready |

## Next Priority Tasks

### Immediate (High Priority)

1. **Fix Critical Bugs**
   - [ ] Fix Terrain count mismatch in classes.py
   - [ ] Fix index calculation in action_selector.py line 250
   - [ ] Make Hex and Unit frozen dataclasses

2. **Complete Training Loop**
   - [ ] Implement loss calculation in game_manager.py
   - [ ] Add policy loss (cross-entropy with model outputs)
   - [ ] Add value loss (MSE with actual rewards)
   - [ ] Add consistency loss (EfficientZero style)
   - [ ] Implement checkpoint save/load

3. **Test Integration**
   - [ ] Install LuaSocket and dkjson in Wesnoth
   - [ ] Run ai_server.py
   - [ ] Load test scenario in Wesnoth
   - [ ] Verify communication works end-to-end

### Short-term (Medium Priority)

4. **Improve Action Selection**
   - [ ] Add proper hex distance calculation
   - [ ] Validate movement points before actions
   - [ ] Handle invalid actions gracefully
   - [ ] Add logging for debugging

5. **Training Infrastructure**
   - [ ] Add metrics logging (win rate, game length, etc.)
   - [ ] Implement self-play opponent
   - [ ] Add experience replay buffer management
   - [ ] Create training monitoring dashboard

6. **State Representation**
   - [ ] Complete fog of war handling
   - [ ] Add defense caps to encoding
   - [ ] Track gold explicitly
   - [ ] Add unit advancement selection

### Long-term (Lower Priority)

7. **Advanced Features**
   - [ ] EfficientZero improvements (value prefix, off-policy correction)
   - [ ] Hierarchical memory (tactical vs strategic)
   - [ ] Attention-based memory updates
   - [ ] Multi-scale time horizons

8. **Optimization**
   - [ ] Profile transformer performance
   - [ ] Optimize state encoding
   - [ ] Add GPU support
   - [ ] Batch processing for multiple games

## Dependencies

### Python
- ✅ torch (PyTorch)
- ✅ numpy
- ✅ Standard library (socket, json, logging, etc.)

### Lua (for Wesnoth)
- ⚠️ LuaSocket (needs installation)
- ⚠️ dkjson or luajson (needs installation)

### Wesnoth
- ⚠️ Version 1.14+ recommended
- ⚠️ Custom scenario support
- ⚠️ Lua AI capability

## Testing Status

### Unit Tests (test_all.py)
```
Total: 19 tests
✅ Passing: 15
⏭️ Skipped: 4 (known issues)
❌ Failed: 0
```

**Passing Tests:**
- Data structure creation (Position, Alignment, Attack, Hex, Memory, GameConfig)
- Action selector (end turn, recruit, move, full pipeline)
- Transformer initialization
- Module imports
- JSON serialization
- Configuration loading

**Skipped Tests:**
- Unit creation (validation mismatch)
- Attack target selection (index bug)
- Transformer forward pass (requires fixing data structures)
- Transformer output types (requires fixing data structures)

### Integration Tests
- ❌ Full Wesnoth integration: Not tested yet
- ✅ Module imports: All working
- ✅ Server startup: Successful
- ❌ Lua-Python communication: Not tested
- ❌ End-to-end game: Not tested

## Configuration

Current settings in `assumptions.py`:

```python
# Model
MAX_UNIT_TYPE = 1000
MAX_ATTACKS = 4
MEMORY_STATE_SIZE = 256

# Training
MAX_ACTIONS_ALLOWED = 2000
ACTION_PENALTY = 0.01
TIMEOUT_PENALTY = 1.0
REPLAY_BUFFER_SIZE = 100000
REPLAY_BATCH_SIZE = 512

# Server
HOST = 'localhost'
PORT = 15000  # WML server
# AI server uses 15001
```

## Performance Considerations

### Current
- Single game at a time
- No GPU acceleration configured
- Full game state in memory
- No optimization yet

### Future Optimizations
- Parallel game execution
- GPU model inference
- State compression
- Batch processing
- Efficient experience replay

## Code Quality

### Strengths
- ✅ Clear separation of concerns
- ✅ Comprehensive logging
- ✅ Type hints in critical functions
- ✅ Documented configuration
- ✅ Unit test coverage

### Areas for Improvement
- Data structure validation inconsistencies
- Error handling in edge cases
- Performance profiling needed
- Integration testing required
- Code documentation (docstrings)

## Resources Used

### Research References
- [Wesnoth LuaAI Documentation](https://wiki.wesnoth.org/LuaAI)
- [Creating Custom AIs](https://wiki.wesnoth.org/Creating_Custom_AIs)
- [MultiplayerServerWML](https://wiki.wesnoth.org/MultiplayerServerWML)
- [EfficientZero Paper](https://arxiv.org/abs/2111.00210)
- [AlphaZero](https://arxiv.org/abs/1712.01815)

### Development Tools
- Python 3.8+
- PyTorch
- Battle for Wesnoth 1.14+
- LuaSocket
- dkjson

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Training instability | High | High | Start with simple scenarios, careful hyperparameter tuning |
| Wesnoth API changes | Low | Medium | Document version requirements, test regularly |
| Performance issues | Medium | Medium | Profile early, optimize critical paths |
| Invalid actions | High | Low | Robust validation and fallback logic |
| OOM errors | Medium | High | Limit replay buffer, use checkpointing |

## Timeline Estimate

**Assuming part-time development:**

- **Week 1-2**: Fix critical bugs, complete training loop
- **Week 3-4**: Integration testing, debugging
- **Week 5-8**: Training experiments, hyperparameter tuning
- **Week 9-12**: Advanced features, optimization
- **Month 4+**: Self-play training, evaluation, iteration

## Success Metrics

### Short-term
- [ ] AI successfully plays a full game without crashes
- [ ] Action selection produces valid moves >95% of time
- [ ] Training loop completes without errors
- [ ] Model converges (loss decreases)

### Medium-term
- [ ] AI beats default Wesnoth AI >20% of games
- [ ] Average game length < 100 turns
- [ ] Win rate improves over training time
- [ ] No OOS (Out of Sync) errors in multiplayer

### Long-term
- [ ] AI competitive with human players
- [ ] Strategic behavior emerges
- [ ] Efficient use of gold and recruitment
- [ ] Adaptation to different scenarios/maps

## Conclusion

The project has a solid architectural foundation with clear separation between Wesnoth integration (Lua), AI serving (Python), and model inference (Transformer). The immediate focus should be on:

1. **Fixing critical bugs** (validation, indexing)
2. **Completing the training loop**
3. **End-to-end integration testing**

Once these are addressed, the project can move into training and evaluation phases. The modular design allows for iterative improvement of individual components without disrupting the overall system.

## Contact & Contribution

See README.md for setup instructions and INTEGRATION_GUIDE.md for detailed technical documentation.

---

*Last Updated: 2026-01-13*
*Project Phase: Architecture Complete, Pre-Training*
